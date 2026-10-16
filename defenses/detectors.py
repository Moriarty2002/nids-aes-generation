from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.registry import register_detector
from models.wrapper import ModelWrapper, PyTorchModelWrapper
from .base import BaseDetector


@register_detector("mahalanobis")
class MahalanobisDetector(BaseDetector):
    """Class-conditional Gaussian detector over penultimate NIDS features.

    Fits empirical class means and a tied pooled within-class covariance, then
    returns each sample's minimum squared Mahalanobis distance to any class mean.
    Larger values therefore indicate more anomalous traffic and can be passed
    directly to AUROC with adversarial samples as the positive class.

    Reference paper: Lee et al., *A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks*, NeurIPS 2018.
    Original research code: https://github.com/pokaxpoka/deep_Mahalanobis_detector

    This is a typed reimplementation suitable for variable-shaped NIDS tensors;
    it does not copy code from the original repository. A small diagonal ridge
    regularizer stabilizes the empirical covariance when dimensions are collinear.
    """

    REGISTRY_NAME = "mahalanobis"

    def __init__(self, covariance_ridge: float = 1e-5, score_batch_size: int = 1024) -> None:
        if covariance_ridge < 0:
            raise ValueError("covariance_ridge must be >= 0")
        if score_batch_size < 1:
            raise ValueError("score_batch_size must be >= 1")
        self.covariance_ridge = float(covariance_ridge)
        self.score_batch_size = int(score_batch_size)
        self.class_labels_: np.ndarray | None = None
        self.class_means_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.feature_dim_: int | None = None
        self.n_fit_samples_: int = 0

    @staticmethod
    def _features_to_numpy(features: torch.Tensor) -> np.ndarray:
        if not isinstance(features, torch.Tensor):
            raise TypeError("ModelWrapper.extract_features must return a Tensor")
        values = features.detach().to(dtype=torch.float64).cpu().numpy()
        if values.ndim != 2:
            raise ValueError(f"Expected 2-D flattened penultimate features, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Penultimate features contain NaN or infinite values")
        return values

    def fit(self, dataloader: DataLoader, wrapper: ModelWrapper) -> "MahalanobisDetector":
        """Estimate class means and one tied empirical covariance on training data using the wrapper-specific feature extractor.

        Call this only after the target model has finished its selected training
        procedure (including PGD or TRADES when enabled), so statistics correspond
        to the actual defended representation used at inference.
        """
        feature_batches: list[np.ndarray] = []
        label_batches: list[np.ndarray] = []
        # PyTorch wrappers require temporary eval mode to freeze BatchNorm and
        # dropout. sklearn estimators have no train/eval state, but implement the
        # same polymorphic extract_features contract and need no special path.
        pytorch_wrapper = isinstance(wrapper, PyTorchModelWrapper)
        was_training = wrapper.model.training if pytorch_wrapper else False
        if pytorch_wrapper:
            wrapper.model.eval()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        raise ValueError("Detector fit expects batches shaped as (x, y)")
                    x_batch, y_batch = batch[0], batch[1]
                    feature_batches.append(self._features_to_numpy(wrapper.extract_features(x_batch)))
                    if isinstance(y_batch, torch.Tensor):
                        labels = y_batch.detach().cpu().numpy()
                    else:
                        labels = np.asarray(y_batch)
                    label_batches.append(labels.reshape(-1).astype(np.int64, copy=False))
        finally:
            if pytorch_wrapper:
                wrapper.model.train(was_training)

        if not feature_batches:
            raise ValueError("Cannot fit MahalanobisDetector: training DataLoader is empty")
        features = np.concatenate(feature_batches, axis=0)
        labels = np.concatenate(label_batches, axis=0)
        if len(features) != len(labels):
            raise ValueError("Feature and label counts differ during detector fitting")
        class_labels = np.unique(labels)
        if class_labels.size < 1:
            raise ValueError("Cannot fit MahalanobisDetector without observed classes")

        means = np.vstack([features[labels == label].mean(axis=0) for label in class_labels])
        class_index = {int(label): index for index, label in enumerate(class_labels)}
        residuals = features - means[np.fromiter((class_index[int(label)] for label in labels), dtype=np.int64)]
        # MLE tied covariance: pool residuals from all classes and divide by N.
        covariance = (residuals.T @ residuals) / max(len(residuals), 1)
        covariance = (covariance + covariance.T) * 0.5
        if self.covariance_ridge:
            covariance += self.covariance_ridge * np.eye(covariance.shape[0], dtype=np.float64)
        precision = np.linalg.pinv(covariance, hermitian=True)

        self.class_labels_ = class_labels.astype(np.int64, copy=False)
        self.class_means_ = means.astype(np.float64, copy=False)
        self.covariance_ = covariance.astype(np.float64, copy=False)
        self.precision_ = precision.astype(np.float64, copy=False)
        self.feature_dim_ = int(features.shape[1])
        self.n_fit_samples_ = int(len(features))
        return self

    def score(self, x: Any, wrapper: ModelWrapper) -> np.ndarray:
        """Return nearest-class squared Mahalanobis distance (higher = anomaly)."""
        if self.class_means_ is None or self.precision_ is None or self.class_labels_ is None:
            raise RuntimeError("MahalanobisDetector must be fitted or loaded before score()")
        if isinstance(x, torch.Tensor):
            n_samples = int(x.shape[0])
        else:
            x = np.asarray(x)
            if x.ndim < 2:
                raise ValueError(f"Expected batch-shaped input, got {x.shape}")
            n_samples = int(x.shape[0])
        feature_chunks: list[np.ndarray] = []
        for start in range(0, n_samples, self.score_batch_size):
            x_chunk = x[start : start + self.score_batch_size]
            with torch.no_grad():
                feature_chunks.append(self._features_to_numpy(wrapper.extract_features(x_chunk)))
        if not feature_chunks:
            return np.empty((0,), dtype=np.float64)
        features = np.concatenate(feature_chunks, axis=0)
        if features.shape[1] != self.feature_dim_:
            raise ValueError(
                f"Feature dimension mismatch: detector was fit on {self.feature_dim_}, got {features.shape[1]}"
            )
        distances: list[np.ndarray] = []
        assert self.class_means_ is not None and self.precision_ is not None
        for means_chunk_start in range(0, len(features), self.score_batch_size):
            z = features[means_chunk_start : means_chunk_start + self.score_batch_size]
            # Broadcast to (batch, classes, dimensions), then compute d' Sigma^-1 d.
            diff = z[:, None, :] - self.class_means_[None, :, :]
            projected = diff @ self.precision_
            mahalanobis_sq = np.sum(projected * diff, axis=-1)
            distances.append(np.sqrt(np.clip(mahalanobis_sq, 0.0, None)).min(axis=1))
        return np.concatenate(distances).astype(np.float64, copy=False)

    def state_dict(self) -> dict[str, Any]:
        if self.class_labels_ is None or self.class_means_ is None or self.covariance_ is None or self.precision_ is None:
            raise RuntimeError("Cannot serialize an unfitted MahalanobisDetector")
        return {
            "format_version": 1,
            "name": self.REGISTRY_NAME,
            "covariance_ridge": self.covariance_ridge,
            "score_batch_size": self.score_batch_size,
            "class_labels": self.class_labels_.copy(),
            "class_means": self.class_means_.copy(),
            "covariance": self.covariance_.copy(),
            "precision": self.precision_.copy(),
            "feature_dim": int(self.feature_dim_ or 0),
            "n_fit_samples": int(self.n_fit_samples_),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        labels = np.asarray(state["class_labels"], dtype=np.int64)
        means = np.asarray(state["class_means"], dtype=np.float64)
        covariance = np.asarray(state["covariance"], dtype=np.float64)
        precision = np.asarray(state["precision"], dtype=np.float64)
        if means.ndim != 2 or len(labels) != len(means):
            raise ValueError("Invalid Mahalanobis detector class means/labels shapes")
        feature_dim = int(state.get("feature_dim", means.shape[1]))
        expected = (feature_dim, feature_dim)
        if means.shape[1] != feature_dim or covariance.shape != expected or precision.shape != expected:
            raise ValueError("Invalid Mahalanobis detector covariance/precision dimensions")
        if not (np.isfinite(means).all() and np.isfinite(covariance).all() and np.isfinite(precision).all()):
            raise ValueError("Mahalanobis detector state contains non-finite values")
        self.covariance_ridge = float(state.get("covariance_ridge", self.covariance_ridge))
        self.score_batch_size = int(state.get("score_batch_size", self.score_batch_size))
        self.class_labels_ = labels
        self.class_means_ = means
        self.covariance_ = covariance
        self.precision_ = precision
        self.feature_dim_ = feature_dim
        self.n_fit_samples_ = int(state.get("n_fit_samples", 0))
