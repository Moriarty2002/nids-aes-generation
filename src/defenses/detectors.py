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

    @classmethod
    def _fit_batch(
        cls,
        batch: Any,
        wrapper: ModelWrapper,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract one bounded feature/label batch without retaining prior batches.

        The detector works on penultimate representations rather than packet
        tensors, so input layout and device placement remain the wrapper's
        responsibility. Statistics are accumulated in CPU float64 for accuracy.
        """
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Detector fit expects batches shaped as (x, y)")
        x_batch, y_batch = batch[0], batch[1]
        features = cls._features_to_numpy(wrapper.extract_features(x_batch))
        if isinstance(y_batch, torch.Tensor):
            labels = y_batch.detach().cpu().numpy()
        else:
            labels = np.asarray(y_batch)
        labels = labels.reshape(-1).astype(np.int64, copy=False)
        if features.shape[0] != labels.size:
            raise ValueError("Feature and label counts differ during detector fitting")
        if features.shape[1] == 0:
            raise ValueError("Penultimate features must have at least one column")
        return features, labels

    def fit(self, dataloader: DataLoader, wrapper: ModelWrapper) -> "MahalanobisDetector":
        """Estimate class means and pooled MLE covariance in exactly two passes.

        Pass 1 accumulates per-class feature sums and counts, giving
        mu_c = (sum_{i:y_i=c} f_i) / n_c. Pass 2 sums the within-class
        residual outer products, Sigma = sum_i (f_i-mu_{y_i})
        (f_i-mu_{y_i})^T / N. A diagonal ridge stabilizes the pseudoinverse.

        The dataloader must be *reiterable* and must expose the same examples
        and feature representations on both passes. Random ordering is fine;
        replacement sampling, stochastic augmentation, or a one-shot stream
        may make the two-pass estimator inconsistent. Sample and class counts
        are checked across passes, but cannot detect every dataset mutation.

        CPU working memory is O(F^2 + C*F + B*F) rather than O(N*F), for
        feature width F, observed classes C, and dataloader batch size B.
        Fit only after NIDS training is complete, so statistics describe the
        actual defended representation used for inference.
        """
        class_sums: dict[int, np.ndarray] = {}
        class_counts: dict[int, int] = {}
        feature_dim: int | None = None
        n_samples = 0

        # Hold eval mode for BOTH passes: dropout/BatchNorm must not change
        # the representation in between computing class means and covariance.
        pytorch_wrapper = isinstance(wrapper, PyTorchModelWrapper)
        was_training = wrapper.model.training if pytorch_wrapper else False
        if pytorch_wrapper:
            wrapper.model.eval()
        try:
            with torch.no_grad():
                # accumulate only O(C*F) sufficient statistics.
                for batch in dataloader:
                    features, labels = self._fit_batch(batch, wrapper)
                    if feature_dim is None:
                        feature_dim = int(features.shape[1])
                    elif features.shape[1] != feature_dim:
                        raise ValueError("Penultimate feature dimension changed between batches")
                    if not labels.size:
                        continue
                    for label in np.unique(labels):
                        key = int(label)
                        selected = labels == label
                        if key not in class_sums:
                            class_sums[key] = np.zeros(feature_dim, dtype=np.float64)
                            class_counts[key] = 0
                        np.add(
                            class_sums[key],
                            features[selected].sum(axis=0, dtype=np.float64),
                            out=class_sums[key],
                        )
                        class_counts[key] += int(np.count_nonzero(selected))
                    n_samples += int(labels.size)

                if n_samples == 0 or feature_dim is None:
                    raise ValueError("Cannot fit MahalanobisDetector: training DataLoader is empty")

                class_labels = np.asarray(sorted(class_sums), dtype=np.int64)
                means = np.stack(
                    [class_sums[int(label)] / class_counts[int(label)] for label in class_labels],
                    axis=0,
                )
                if not np.isfinite(means).all():
                    raise ValueError("Class means contain NaN or infinite values")

                # covariance from centered feature batches, with at
                # most one class subset's residuals resident at a time.
                covariance = np.zeros((feature_dim, feature_dim), dtype=np.float64)
                second_counts = np.zeros(class_labels.size, dtype=np.int64)
                second_samples = 0
                for batch in dataloader:
                    features, labels = self._fit_batch(batch, wrapper)
                    if features.shape[1] != feature_dim:
                        raise ValueError("Penultimate feature dimension changed between passes")
                    if not labels.size:
                        continue
                    class_indices = np.searchsorted(class_labels, labels)
                    if (
                        np.any(class_indices >= class_labels.size)
                        or np.any(class_labels[class_indices] != labels)
                    ):
                        raise ValueError("DataLoader produced unseen classes on the second pass")
                    second_counts += np.bincount(
                        class_indices, minlength=class_labels.size
                    )
                    second_samples += int(labels.size)
                    for index in np.unique(class_indices):
                        # Boolean indexing allocates only a class subset of
                        # THIS batch. Never mutate the wrapper's feature tensor.
                        residuals = features[class_indices == index]
                        np.subtract(residuals, means[index], out=residuals)
                        np.add(covariance, residuals.T @ residuals, out=covariance)

                expected_counts = np.asarray(
                    [class_counts[int(label)] for label in class_labels], dtype=np.int64
                )
                if second_samples != n_samples or not np.array_equal(second_counts, expected_counts):
                    raise ValueError(
                        "DataLoader must yield the same sample and class counts on both fit passes; "
                        "disable replacement sampling and changing data sources"
                    )
        finally:
            if pytorch_wrapper:
                wrapper.model.train(was_training)

        # MLE covariance uses N, NOT (N-C), to match the pre-refactor model.
        np.divide(covariance, n_samples, out=covariance)
        # Compensate for summation asymmetry before Hermitian pseudoinversion.
        covariance = (covariance + covariance.T) * 0.5
        if self.covariance_ridge:
            covariance.flat[::feature_dim + 1] += self.covariance_ridge
        if not np.isfinite(covariance).all():
            raise ValueError("Pooled covariance contains NaN or infinite values")
        precision = np.linalg.pinv(covariance, hermitian=True)
        if not np.isfinite(precision).all():
            raise ValueError("Pooled precision contains NaN or infinite values")

        # Publish the fitted state only after BOTH complete passes succeeded.
        self.class_labels_ = class_labels
        self.class_means_ = means
        self.covariance_ = covariance
        self.precision_ = precision
        self.feature_dim_ = feature_dim
        self.n_fit_samples_ = n_samples
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
