from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
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
    """Unified target/surrogate contract for all attack and evaluation code."""

    capabilities: ModelCapabilities

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
        """Place a model on its worker device and default it to inference mode."""
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.capabilities = ModelCapabilities(True, True, True, "pytorch")

    def to(self, device: torch.device) -> "PyTorchModelWrapper":
        """Move the wrapped model while preserving the wrapper identity."""
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def _tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def logits(self, x: Any) -> torch.Tensor:
        """Run inference without changing the model's caller-visible train mode."""
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
        latest_features: torch.Tensor | None = None

        def capture_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            nonlocal latest_features
            if inputs and isinstance(inputs[0], torch.Tensor):
                latest_features = inputs[0]

        try:
            linear_modules = [module for module in self.model.modules() if isinstance(module, nn.Linear)]
            if linear_modules:
                handles = [module.register_forward_pre_hook(capture_input) for module in linear_modules]
                with torch.no_grad():
                    _ = self.model(tensor)
                if latest_features is None:
                    raise RuntimeError("No activations were captured from the model's linear layers")
                features = latest_features
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
        """Return logits with autograd enabled for attacks and training losses."""
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
            assert y is not None
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
