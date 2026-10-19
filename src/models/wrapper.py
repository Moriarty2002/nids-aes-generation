from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import logging
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.exceptions import NotFittedError
import torch
from torch import nn

from core.exceptions import IncompatibleThreatModelError


@dataclass(frozen=True)
class ModelCapabilities:
    differentiable: bool
    gradient_input: bool
    probabilistic_output: bool
    model_family: str


class ModelWrapper(ABC):
    """Unified target/surrogate contract across differentiable and tree-based models.

    ``extract_features`` is deliberately polymorphic: neural wrappers return a
    penultimate hidden representation, linear estimators return decision margins,
    and tree ensembles return bounded-dimensional aggregated tree margins.
    Inference detectors can therefore operate without knowing model internals.
    """

    capabilities: ModelCapabilities

    def extract_features(self, x: Any) -> torch.Tensor:
        """Return a batch-first CPU tensor representing model features.

        The default raises rather than fabricating a feature representation for a
        plugin that does not implement one. Existing model wrappers remain
        instantiable because this method is not abstract.
        """
        raise IncompatibleThreatModelError(
            f"{type(self).__name__} does not implement extract_features(x)"
        )

    @abstractmethod
    def predict(self, x: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, x: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_gradients(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device) -> "ModelWrapper":
        raise NotImplementedError


class PyTorchModelWrapper(ModelWrapper):
    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.capabilities = ModelCapabilities(True, True, True, "pytorch")

    def to(self, device: torch.device) -> "PyTorchModelWrapper":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def _tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def logits(self, x: Any) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self.model(self._tensor(x))
        finally:
            self.model.train(was_training)

    def extract_features(self, x: Any) -> torch.Tensor:
        """Extract penultimate-layer activations as a ``(batch, features)`` tensor.

        The default path registers pre-forward hooks on all ``nn.Linear`` modules and
        uses the input to the last linear module executed by ``forward``. This directly
        captures the vector immediately before the classification head and works with
        the bundled MLP, CNN, LSTM and Transformer architectures without coupling the
        wrapper to their internal attribute names.

        Custom architectures without a linear head can expose one of ``penultimate_features``,
        ``extract_features`` or ``forward_features`` returning a batch-first tensor. All
        inference hooks are removed even on exceptions, and the model's prior train/eval
        mode is restored.
        """
        tensor = self._tensor(x)
        was_training = self.model.training
        self.model.eval()
        handles: list[Any] = []
        captured: list[torch.Tensor] = []

        def capture_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                captured.append(inputs[0])

        try:
            linear_modules = [module for module in self.model.modules() if isinstance(module, nn.Linear)]
            if linear_modules:
                handles = [module.register_forward_pre_hook(capture_input) for module in linear_modules]
                with torch.no_grad():
                    _ = self.model(tensor)
                if not captured:
                    raise RuntimeError("No activations were captured from the model's linear layers")
                features = captured[-1]
            else:
                features = None
                for method_name in ("penultimate_features", "extract_features", "forward_features"):
                    method = getattr(self.model, method_name, None)
                    if callable(method):
                        with torch.no_grad():
                            candidate = method(tensor)
                        if isinstance(candidate, (tuple, list)):
                            candidate = next(
                                (item for item in reversed(candidate) if isinstance(item, torch.Tensor)), None
                            )
                        if not isinstance(candidate, torch.Tensor):
                            raise TypeError(
                                f"{method_name}() must return a Tensor or a tuple/list containing one"
                            )
                        features = candidate
                        break
                if features is None:
                    raise IncompatibleThreatModelError(
                        f"{type(self.model).__name__} has no nn.Linear classification head and exposes no "
                        "penultimate_features/extract_features/forward_features method"
                    )
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

        if features.ndim < 2:
            raise ValueError(
                f"Penultimate features must be batch-first with at least 2 dimensions, got {tuple(features.shape)}"
            )
        if features.shape[0] != tensor.shape[0]:
            raise ValueError(
                f"Feature extractor changed batch size: input={tensor.shape[0]}, features={features.shape[0]}"
            )
        return features.detach().reshape(features.shape[0], -1)

    def forward_differentiable(self, x: Any) -> torch.Tensor:
        """Return logits with autograd enabled for white-box attacks and losses."""
        self.model.eval()
        return self.model(self._tensor(x))

    def predict(self, x: Any) -> np.ndarray:
        out = self.logits(x)
        if out.ndim == 1 or out.shape[-1] == 1:
            return (torch.sigmoid(out) >= 0.5).long().reshape(-1).cpu().numpy()
        return torch.argmax(out, dim=-1).detach().cpu().numpy()

    def predict_proba(self, x: Any) -> np.ndarray:
        out = self.logits(x)
        if out.ndim == 1 or out.shape[-1] == 1:
            positive = torch.sigmoid(out.reshape(-1))
            probs = torch.stack((1.0 - positive, positive), dim=-1)
        else:
            probs = torch.softmax(out, dim=-1)
        return probs.detach().cpu().numpy()

    def get_gradients(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            x_adv = self._tensor(x).detach().clone().requires_grad_(True)
            logits = self.model(x_adv)
            if y is None:
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    y = (torch.sigmoid(logits) >= 0.5).long().reshape(-1)
                else:
                    y = logits.detach().argmax(dim=-1)
            y = y.to(self.device, dtype=torch.long)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.reshape(-1), y.float().reshape(-1), reduction="sum"
                )
            else:
                loss = nn.functional.cross_entropy(logits, y, reduction="sum")
            gradient = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            return gradient.detach()
        finally:
            self.model.train(was_training)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.model(self._tensor(x))
        labels = y.to(self.device, dtype=torch.long)
        if logits.ndim == 1 or logits.shape[-1] == 1:
            return nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float().reshape(-1))
        return nn.functional.cross_entropy(logits, labels)


class SklearnModelWrapper(ModelWrapper):
    """Adapter for RandomForest/XGBoost/sklearn estimators exposing predict and optionally predict_proba."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.capabilities = ModelCapabilities(False, False, hasattr(estimator, "predict_proba"), "sklearn")

    def to(self, device: torch.device) -> "SklearnModelWrapper":
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict(self._numpy(x)))

    def predict_proba(self, x: Any) -> np.ndarray:
        if not hasattr(self.estimator, "predict_proba"):
            raise IncompatibleThreatModelError(
                f"{type(self.estimator).__name__} does not expose predict_proba"
            )
        return np.asarray(self.estimator.predict_proba(self._numpy(x)))

    @staticmethod
    def _numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _hidden_activation(name: str, values: np.ndarray) -> np.ndarray:
        """Apply the named sklearn MLP hidden-layer activation."""
        normalized = str(name).lower()
        if normalized == "identity":
            return values
        if normalized == "logistic":
            clipped = np.clip(values, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        if normalized == "tanh":
            return np.tanh(values)
        if normalized == "relu":
            return np.maximum(values, 0.0)
        raise IncompatibleThreatModelError(
            f"Unsupported sklearn MLP activation {name!r} for feature extraction"
        )

    def extract_features(self, x: Any) -> torch.Tensor:
        """Extract CPU features from an sklearn estimator.

        Supported cases:
          * ``MLPClassifier``/``MLPRegressor``: manually evaluate hidden layers
            using ``coefs_``/``intercepts_`` and the estimator's configured
            hidden activation, stopping immediately before the output layer.
          * Linear estimators such as ``LogisticRegression``: return the raw
            ``decision_function`` margins.
          * Tree ensembles such as ``RandomForestClassifier``: concatenate
            each tree's clipped log-probability margins. If this would exceed
            256 dimensions, average margins across trees instead to avoid
            covariance blow-up in feature-space detectors.

        The return value is always a detached CPU ``float32`` tensor with shape
        ``(batch, feature_dimension)``. These are detector features, not
        differentiable attack inputs; sklearn remains a non-gradient model.
        """
        features = np.asarray(self._numpy(x), dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(
                f"Sklearn feature extraction requires a 2-D (N, D) matrix, got {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("Sklearn feature extraction input contains NaN or infinity")
        try:
            # sklearn estimators expose n_features_in_ after fit.
            expected = getattr(self.estimator, "n_features_in_", None)
            if expected is not None and features.shape[1] != int(expected):
                raise ValueError(f"Expected {int(expected)} input features, got {features.shape[1]}")

            # MLPClassifier/MLPRegressor: each coefs_[i] maps layer i to i+1;
            # the final matrix is the classification head, so don't apply it.
            coefs = getattr(self.estimator, "coefs_", None)
            intercepts = getattr(self.estimator, "intercepts_", None)
            if coefs is not None and intercepts is not None:
                if len(coefs) < 2 or len(coefs) != len(intercepts):
                    raise IncompatibleThreatModelError(
                        "A sklearn MLP must have at least one hidden layer to expose penultimate features"
                    )
                value = features
                activation = str(getattr(self.estimator, "activation", "relu"))
                for weights, bias in zip(coefs[:-1], intercepts[:-1]):
                    value = self._hidden_activation(activation, value @ np.asarray(weights) + np.asarray(bias))
                output = value

            # Random forests and similar tree ensembles. Each tree's probability
            # columns are aligned to the parent estimator's class schema.
            elif getattr(self.estimator, "estimators_", None) is not None and callable(
                getattr(self.estimator, "predict_proba", None)
            ):
                trees = list(self.estimator.estimators_)
                if trees and isinstance(trees[0], (list, tuple, np.ndarray)):
                    raise IncompatibleThreatModelError(
                        "Multi-output tree ensembles are not supported by Mahalanobis feature extraction"
                    )
                if not trees:
                    raise IncompatibleThreatModelError("Fitted tree ensemble has no estimators_")
                global_classes = np.asarray(getattr(self.estimator, "classes_", []))
                if global_classes.ndim != 1 or global_classes.size < 2:
                    raise IncompatibleThreatModelError("Tree ensemble has no supported single-output class schema")
                tree_margins: list[np.ndarray] = []
                for tree in trees:
                    probs = np.asarray(tree.predict_proba(features), dtype=np.float64)
                    if probs.ndim != 2:
                        raise IncompatibleThreatModelError(
                            "Each tree must return a 2-D probability matrix for single-output classification"
                        )
                    if probs.shape[1] == len(global_classes):
                        aligned = probs
                    else:
                        aligned = np.zeros((len(features), len(global_classes)), dtype=np.float64)
                        tree_classes = np.asarray(getattr(tree, "classes_", []))
                        if tree_classes.ndim != 1 or len(tree_classes) != probs.shape[1]:
                            raise IncompatibleThreatModelError(
                                "Unable to align tree probability columns to ensemble classes"
                            )
                        for column, class_value in enumerate(tree_classes):
                            match = np.flatnonzero(global_classes == class_value)
                            if match.size:
                                aligned[:, int(match[0])] = probs[:, column]
                            elif np.issubdtype(tree_classes.dtype, np.number):
                                encoded = int(class_value)
                                if float(class_value) == encoded and 0 <= encoded < len(global_classes):
                                    aligned[:, encoded] = probs[:, column]
                    tree_margins.append(np.log(np.clip(aligned, 1e-7, 1.0 - 1e-7)))
                
                # If the concatenated tree margins would exceed 256 dimensions, average across trees instead to avoid covariance blow-up in feature-space detectors.
                # this is a security-aware design choice: the Mahalanobis detector is intended to operate in a bounded feature space, and high-dimensional concatenation can lead to overfitting and reduced robustness.
                # Remember that averaging the trees destroys the inter-tree variance signal, but it preserves the overall class margin information, which is often sufficient for detection purposes.
                stacked_dimension = len(trees) * len(global_classes)
                if stacked_dimension > 256:
                    print(f"Warning: Concatenated tree margins would exceed 256 dimensions ({stacked_dimension}), averaging across trees instead.")
                    # Mean over the tree axis: result shape (N, C), not (N, T*C).
                    output = np.mean(np.stack(tree_margins, axis=0), axis=0)
                else:
                    output = np.concatenate(tree_margins, axis=1)

            # Linear classifiers, including LogisticRegression, return one raw
            # margin per class (or one signed margin in binary classification).
            elif callable(getattr(self.estimator, "decision_function", None)):
                output = np.asarray(self.estimator.decision_function(features), dtype=np.float64)
                if output.ndim == 1:
                    output = output.reshape(-1, 1)
                elif output.ndim != 2:
                    raise IncompatibleThreatModelError(
                        f"decision_function returned unsupported shape {output.shape}"
                    )
            else:
                raise IncompatibleThreatModelError(
                    f"Feature extraction is unsupported for fitted estimator {type(self.estimator).__name__}; "
                    "supported types are sklearn MLPs, linear estimators with decision_function, "
                    "and single-output tree ensembles exposing estimators_ and predict_proba."
                )
        except NotFittedError as exc:
            raise IncompatibleThreatModelError(
                f"Estimator {type(self.estimator).__name__} must be fitted before feature extraction"
            ) from exc

        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 1:
            output = output.reshape(-1, 1)
        if output.ndim != 2 or output.shape[0] != len(features):
            raise ValueError(f"Feature extractor returned invalid shape {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("Extracted sklearn features contain NaN or infinity")
        return torch.as_tensor(np.ascontiguousarray(output), dtype=torch.float32, device="cpu")

    def get_gradients(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raise IncompatibleThreatModelError(
            "White-box gradient access is unavailable for non-differentiable sklearn/tree models. "
            "Use a score/query-based attack or a differentiable surrogate model."
        )


def ensure_gradient_compatible(model: ModelWrapper) -> None:
    if not model.capabilities.differentiable or not model.capabilities.gradient_input:
        raise IncompatibleThreatModelError(
            f"Model family {model.capabilities.model_family!r} is not compatible with gradient-based attacks"
        )
