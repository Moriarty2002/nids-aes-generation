from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import inspect
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score, log_loss
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

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

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: Mapping[str, Any],
        *,
        logger: logging.Logger | None = None,
        output_dir: Any = None,
        seed: int | None = None,
        perturbation_mask_fn: Any = None,
    ) -> Mapping[str, Any]:
        """Fit the wrapped model and return standardized training metrics."""
        raise NotImplementedError

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



def forward_model(model: nn.Module, x: torch.Tensor, valid_mask: Any = None) -> torch.Tensor:
    """Forward with optional real-packet occupancy (never infer from sentinel values)."""
    if "valid_mask" in inspect.signature(model.forward).parameters:
        if valid_mask is None:
            raise ValueError("Sequence-model training/inference requires a valid_mask")
        return model(x, valid_mask=torch.as_tensor(valid_mask, dtype=torch.bool, device=x.device))
    return model(x)

class PyTorchModelWrapper(ModelWrapper):
    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._active_valid_mask: torch.Tensor | np.ndarray | None = None
        self._supports_valid_mask = "valid_mask" in inspect.signature(self.model.forward).parameters
        self.capabilities = ModelCapabilities(True, True, True, "pytorch")

    def to(self, device: torch.device) -> "PyTorchModelWrapper":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: Mapping[str, Any],
        *,
        logger: logging.Logger | None = None,
        output_dir: Any = None,
        seed: int | None = None,
        perturbation_mask_fn: Any = None,
        progress: Any = None,
    ) -> Mapping[str, Any]:
        from models.trainer import train_classifier

        configured_alpha = config.get("pgd_alpha")
        epsilon = float(config.get("pgd_epsilon", 0.04))
        summary = train_classifier(
            self,
            train_loader,
            val_loader,
            epochs=int(config.get("epochs", 20)),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            adversarial_training=bool(config.get("adversarial_training", False)),
            defense_mode=str(config.get("defense_mode", "none")),
            pgd_epsilon=epsilon,
            pgd_alpha=epsilon / 4.0 if configured_alpha is None else float(configured_alpha),
            pgd_steps=int(config.get("pgd_steps", 10)),
            trades_beta=float(config.get("trades_beta", 6.0)),
            early_stopping_patience=(
                None if config.get("early_stopping_patience") is None
                else int(config.get("early_stopping_patience", 0))
            ),
            early_stopping_min_delta=float(config.get("early_stopping_min_delta", 0.0)),
            early_stopping_monitor=str(config.get("early_stopping_monitor", "val_loss")),
            output_dir=output_dir,
            seed=int(config.get("seed", 0) if seed is None else seed),
            logger=logger,
            perturbation_mask_fn=perturbation_mask_fn,
            progress=progress,
        )
        return {
            "train_loss": float(summary.train_loss),
            "validation_loss": float(summary.validation_loss or 0.0),
            "defense_mode": summary.defense_mode,
        }

    @contextmanager
    def attack_mask_context(self, valid_mask: Any):
        """Bind an authorized surrogate's per-batch mask during attack generation.

        This never binds the target or exposes its parameters to attacks.
        """
        previous = self._active_valid_mask
        self._active_valid_mask = valid_mask
        try:
            yield
        finally:
            self._active_valid_mask = previous

    def _forward(self, x: Any, valid_mask: Any = None) -> torch.Tensor:
        tensor = self._tensor(x)
        if not self._supports_valid_mask:
            return self.model(tensor)
        mask = self._active_valid_mask if valid_mask is None else valid_mask
        if mask is None:
            raise ValueError("Sequence models require a real-packet valid_mask")
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if tuple(mask.shape) != (len(tensor), tensor.shape[1]):
            raise ValueError("Sequence-model valid_mask does not match input topology")
        return self.model(tensor, valid_mask=mask)

    def _tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def logits(self, x: Any, valid_mask: Any = None) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self._forward(x, valid_mask=valid_mask)
        finally:
            self.model.train(was_training)

    def extract_features(self, x: Any, valid_mask: Any = None) -> torch.Tensor:
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
                    _ = self._forward(tensor, valid_mask=valid_mask)
                if not captured:
                    raise RuntimeError("No activations were captured from the model's linear layers")
                features = captured[-1]
            else:
                features = None
                for method_name in ("penultimate_features", "extract_features", "forward_features"):
                    method = getattr(self.model, method_name, None)
                    if callable(method):
                        with torch.no_grad():
                            candidate = method(tensor)  # Custom unheaded plugin API
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

    def extract_intermediate_features(self, x: Any, valid_mask: Any = None, *, differentiable: bool = False) -> dict[str, torch.Tensor]:
        """Return executed hidden endpoints for Lee-style *defense* scoring.

        CNN uses post-masked-normalization features at each temporal resolution;
        Transformer uses executed encoder block outputs, not attention internals
        bypassed by functional calls; BiLSTM uses actual directional hidden states.
        All temporal pooling excludes padded tokens, including zero-length flows.
        This is never an attack-side evaluation-target oracle.
        """
        from models.architectures.cnn import CNN1DNIDS
        from models.architectures.transformer import TransformerNIDS
        from models.architectures.lstm import BiLSTMNIDS
        from torch.nn import functional as F

        tensor = self._tensor(x)
        model = self.model
        valid = None if valid_mask is None else torch.as_tensor(
            valid_mask, dtype=torch.bool, device=self.device)
        if tensor.ndim == 3 and (valid is None or tuple(valid.shape) != tuple(tensor.shape[:2])):
            raise ValueError('Sequence hidden extraction requires exact (N,T) valid_mask')
        if isinstance(model, CNN1DNIDS):
            # These are real executed masked-normalization outputs; each block
            # subsequently pools the time dimension by a factor of two.
            selected = [(f'blocks.{i}.2', block[2]) for i, block in enumerate(model.blocks)]
            mode = 'cnn'
        elif isinstance(model, TransformerNIDS):
            selected = [('projection', model.projection)] + [
                (f'transformer.layers.{i}', layer)
                for i, layer in enumerate(model.transformer.layers)]
            mode = 'transformer'
        elif isinstance(model, BiLSTMNIDS):
            if model.lstm.num_layers < 2:
                raise IncompatibleThreatModelError(
                    'Lee multilayer BiLSTM requires at least two executed recurrent layers')
            selected = [('lstm', model.lstm)]
            mode = 'lstm'
        else:
            # Explicitly exclude logits, and use only actual executed hidden
            # linear operations on compatible non-sequence custom/MLP models.
            if tensor.ndim != 2:
                raise IncompatibleThreatModelError('Unknown sequence architecture has no verified Lee endpoints')
            linear = [(n,m) for n,m in model.named_modules() if n and isinstance(m, nn.Linear)]
            selected = linear[:-1]
            mode = 'linear'
        if len(selected) < (1 if mode == 'lstm' else 2):
            raise IncompatibleThreatModelError('Multilayer detector requires two actual hidden stages')

        captured: dict[str, Any] = {}
        handles: list[Any] = []
        def capture(name: str):
            def hook(_module: nn.Module, _args: tuple[Any,...], output: Any) -> None:
                captured[name] = output
            return hook
        was_training = model.training
        model.eval()
        try:
            handles = [module.register_forward_hook(capture(name)) for name,module in selected]
            # Input-gradient preprocessing for detector scoring needs a graph;
            # routine Gaussian fit/score extraction never retains one.
            with (torch.enable_grad() if differentiable else torch.no_grad()):
                self._forward(tensor, valid_mask=valid)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        if list(captured) != [name for name,_ in selected]:
            raise ValueError('Expected hidden endpoint was not executed in the declared order')

        result: dict[str, torch.Tensor] = {}
        if mode == 'lstm':
            # nn.LSTM returns the actual final hidden state of each trained
            # bidirectional recurrent layer. Packed input omits padding.
            output = captured['lstm']
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError('Unexpected recurrent hidden-state output')
            hidden = output[1][0]
            active = (valid.sum(1) > 0).to(hidden.dtype).unsqueeze(1)
            for i in range(model.lstm.num_layers):
                state = torch.cat([hidden[2*i], hidden[2*i+1]], dim=-1)
                value = state * active
                result[f'lstm.layer_{i}'] = value if differentiable else value.detach()
        else:
            for index, (name,_) in enumerate(selected):
                output = captured[name]
                if not isinstance(output, torch.Tensor) or output.shape[0] != len(tensor):
                    raise ValueError(f'Hidden endpoint {name} is not a batch-first tensor')
                if output.ndim == 2:
                    pooled = output
                elif mode == 'cnn' and output.ndim == 3:
                    mask = valid
                    for _ in range(index):
                        mask = F.max_pool1d(mask[:,None].float(),2).squeeze(1) > 0
                    if output.shape[-1] != mask.shape[1]:
                        raise ValueError(f'CNN {name} temporal occupancy changed unexpectedly')
                    pooled = (output * mask[:,None].to(output.dtype)).sum(-1) / mask.sum(-1).clamp_min(1).unsqueeze(1)
                elif mode == 'transformer' and output.ndim == 3:
                    if output.shape[1] != valid.shape[1]:
                        raise ValueError(f'Transformer {name} token count changed')
                    pooled = (output * valid[:,:,None].to(output.dtype)).sum(1) / valid.sum(1).clamp_min(1).unsqueeze(1)
                else:
                    raise IncompatibleThreatModelError(f'Unsupported actual hidden topology at {name}')
                result[name] = pooled if differentiable else pooled.detach()
        if len(result) < 2 or any(v.ndim != 2 or v.shape[-1] > 1024 or not torch.isfinite(v).all() for v in result.values()):
            raise IncompatibleThreatModelError('Invalid or oversized fitted hidden feature endpoint')
        return result

    def forward_differentiable(self, x: Any, valid_mask: Any = None) -> torch.Tensor:
        """Return logits with autograd enabled for white-box attacks and losses."""
        self.model.eval()
        return self._forward(x, valid_mask=valid_mask)

    def predict(self, x: Any, valid_mask: Any = None) -> np.ndarray:
        out = self.logits(x, valid_mask=valid_mask)
        if out.ndim == 1 or out.shape[-1] == 1:
            return (torch.sigmoid(out) >= 0.5).long().reshape(-1).cpu().numpy()
        return torch.argmax(out, dim=-1).detach().cpu().numpy()

    def predict_proba(self, x: Any, valid_mask: Any = None) -> np.ndarray:
        out = self.logits(x, valid_mask=valid_mask)
        if out.ndim == 1 or out.shape[-1] == 1:
            positive = torch.sigmoid(out.reshape(-1))
            probs = torch.stack((1.0 - positive, positive), dim=-1)
        else:
            probs = torch.softmax(out, dim=-1)
        return probs.detach().cpu().numpy()

    def get_gradients(self, x: torch.Tensor, y: torch.Tensor | None = None, valid_mask: Any = None) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            x_adv = self._tensor(x).detach().clone().requires_grad_(True)
            logits = self._forward(x_adv, valid_mask=valid_mask)
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

    def loss(self, x: torch.Tensor, y: torch.Tensor, valid_mask: Any = None) -> torch.Tensor:
        logits = self._forward(x, valid_mask=valid_mask)
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

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: Mapping[str, Any],
        *,
        logger: logging.Logger | None = None,
        output_dir: Any = None,
        seed: int | None = None,
        perturbation_mask_fn: Any = None,
    ) -> Mapping[str, Any]:
        del seed, perturbation_mask_fn
        from workflow.io import _write_sklearn_training_curve

        def collect(loader: DataLoader[Any]) -> tuple[np.ndarray, np.ndarray]:
            # sklearn fits the entire ordered matrix, not shuffled minibatches.
            # A direct CPU TensorDataset view avoids retaining O(N) temporary
            # batch arrays and concatenating an additional full corpus copy.
            if not isinstance(loader.sampler, SequentialSampler):
                raise ValueError('Sklearn training requires a sequential, non-shuffled DataLoader')
            dataset = loader.dataset
            if isinstance(dataset, TensorDataset):
                if len(dataset.tensors) < 2 or not len(dataset):
                    raise ValueError('Empty or malformed sklearn training dataset')
                x = self._numpy(dataset.tensors[0])
                y = self._numpy(dataset.tensors[1]).reshape(-1).astype(np.int64, copy=False)
                return x, y
            # Generic non-TensorDataset adapters: preallocate once, stream
            # examples in order, never retain a list of batch-sized arrays.
            x_buffer: np.ndarray | None = None
            y_buffer = np.empty(len(dataset), dtype=np.int64)
            offset = 0
            for batch in loader:
                if len(batch) < 2:
                    raise ValueError('Training batches require features and labels')
                chunk = self._numpy(batch[0])
                if x_buffer is None:
                    x_buffer = np.empty((len(dataset), *chunk.shape[1:]), dtype=chunk.dtype)
                stop = offset + len(chunk)
                x_buffer[offset:stop] = chunk
                y_buffer[offset:stop] = self._numpy(batch[1]).reshape(-1)
                offset = stop
            if x_buffer is None or offset != len(dataset):
                raise ValueError('Invalid sklearn DataLoader sample count')
            return x_buffer, y_buffer

        train_x, train_y = collect(train_loader)
        val_x, val_y = collect(val_loader)
        if train_x.ndim != 2 or val_x.ndim != 2:
            raise ValueError("scikit-learn estimators require 2-D flow vectors")
        if bool(config.get("adversarial_training", False)):
            raise ValueError(
                "model.training.adversarial_training is a PyTorch option. "
                "Use empirical_at with a differentiable surrogate for sklearn."
            )
        defense_mode = str(config.get("defense_mode", "none") or "none").lower()
        if defense_mode not in {"none", "", "null", "empirical_at"}:
            raise ValueError(f"Unsupported sklearn defense_mode={defense_mode!r}")
        expected_classes = config.get("expected_classes")
        if expected_classes is not None and not np.array_equal(
            np.unique(train_y), np.arange(int(expected_classes), dtype=np.int64)
        ):
            raise ValueError(
                "The training split does not contain every encoded class. "
                "Increase the dataset size or revise the split."
            )

        if logger:
            logger.info(
                "training.start framework=sklearn samples=%d validation=%d defense=%s",
                len(train_y), len(val_y), defense_mode,
            )
        self.estimator.fit(train_x, train_y)
        train_pred = np.asarray(self.estimator.predict(train_x))
        val_pred = np.asarray(self.estimator.predict(val_x))
        train_accuracy = float(accuracy_score(train_y, train_pred))
        validation_accuracy = float(accuracy_score(val_y, val_pred))
        train_f1 = float(f1_score(train_y, train_pred, average="macro", zero_division=0))
        validation_f1 = float(f1_score(val_y, val_pred, average="macro", zero_division=0))
        if callable(getattr(self.estimator, "predict_proba", None)):
            train_prob = np.asarray(self.estimator.predict_proba(train_x))
            val_prob = np.asarray(self.estimator.predict_proba(val_x))
            classes = np.asarray(self.estimator.classes_, dtype=np.int64)
            train_loss = float(log_loss(train_y, train_prob, labels=classes))
            validation_loss = float(log_loss(val_y, val_prob, labels=classes))
        else:
            train_loss = float("nan")
            validation_loss = float("nan")
        _write_sklearn_training_curve(
            output_dir,
            train_loss=train_loss,
            validation_loss=validation_loss,
            train_accuracy=train_accuracy,
            validation_accuracy=validation_accuracy,
            train_f1=train_f1,
            validation_f1=validation_f1,
            defense_mode=defense_mode,
        )
        return {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "defense_mode": defense_mode,
        }

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
