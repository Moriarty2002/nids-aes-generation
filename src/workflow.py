from __future__ import annotations

import csv
import gc
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from attacks.base import BaseAttack, FeatureSpaceAttack, ProblemSpaceAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from core.exceptions import ConfigurationError
from core.registry import ATTACKS, DATASETS, DETECTORS, NIDS, PREPROCESSORS
from evaluator.engine import EvaluationEngine, FeatureNormalization
from evaluator.logger import ExperimentReporter
from evaluator.metrics import classification_metrics
from models.trainer import TrainingBatch, train_classifier
from models.wrapper import ModelWrapper, PyTorchModelWrapper, SklearnModelWrapper
from orchestrator.device import configure_process_device
from orchestrator.runner import seed_everything


def _cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve Hydra interpolation before worker code mutates configuration."""
    if OmegaConf.is_config(config):
        resolved = OmegaConf.to_container(config, resolve=True)
        if not isinstance(resolved, dict):
            raise ConfigurationError("Workflow configuration must resolve to a mapping")
        return cast(dict[str, Any], resolved)
    return dict(config)


_FLATTENING_MODEL_NAMES = frozenset({
    "mlp", "ffn", "dnn", "linear", "logistic_regression",
    "random_forest", "rf", "randomforest", "xgboost", "xgb", "svm", "jmila_surrogate",
})
_SEQUENCE_MODEL_NAMES = frozenset({
    "cnn", "lstm", "bilstm", "rnn", "gru", "transformer",
})


def _model_requires_flatten(model_cfg: Mapping[str, Any]) -> bool:
    """Infer whether a registered model expects 2-D flow vectors.

    A model config override wins, then the registered class's ``requires_flatten``
    capability, then the built-in architecture convention. This makes custom
    registry models extensible without dataset-specific YAML switches.
    """
    explicit = model_cfg.get("requires_flatten")
    if explicit is not None:
        return bool(explicit)

    name = str(model_cfg.get("name", "")).strip().lower().replace("-", "_")
    registered_cls = next(
        (item.component for item in NIDS.metadata() if item.name == name),
        None,
    )
    capability = getattr(registered_cls, "requires_flatten", None)
    if capability is not None:
        return bool(capability)
    if name in _FLATTENING_MODEL_NAMES:
        return True
    if name in _SEQUENCE_MODEL_NAMES:
        return False
    raise ConfigurationError(
        f"Cannot infer tensor shaping for model {name!r}. "
        "Set model.requires_flatten explicitly or define requires_flatten on the registered model class."
    )


def _inject_dynamic_dataset_shape(cfg: dict[str, Any]) -> None:
    """Derive the dataset flatten mode from the selected target architecture.

    The override is applied only when the dataset constructor supports a
    ``flatten`` parameter, preserving all existing flat/tabular adapters.
    The target topology determines the dataset's stored representation.
    Surrogates may use a different topology: EAT maps the same ordered
    packet-feature coordinates bijectively between (N,T*F) and (N,T,F),
    without reordering features or changing their physical interpretation.
    """
    dataset_cfg_raw = cfg.get("dataset")
    model_cfg = cfg.get("model")
    if not isinstance(dataset_cfg_raw, Mapping) or not isinstance(model_cfg, Mapping):
        raise ConfigurationError("Both dataset and model configuration blocks are required")

    dataset_cfg = dict(dataset_cfg_raw)
    params = dict(dataset_cfg.get("params", {}))
    dataset_cls = DATASETS.get(str(dataset_cfg["name"]))
    constructor = inspect.signature(dataset_cls)
    supports_flatten = "flatten" in constructor.parameters
    target_flatten = _model_requires_flatten(model_cfg)
    if supports_flatten:
        params["flatten"] = target_flatten

    dataset_cfg["params"] = params
    cfg["dataset"] = dataset_cfg


def _instantiate_dataset(cfg: Mapping[str, Any]):
    """Create the registered dataset plugin from its Hydra configuration."""
    dataset_cfg = dict(cfg["dataset"])
    cls = DATASETS.get(str(dataset_cfg["name"]))
    return cls(**dict(dataset_cfg.get("params", {})))


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    temp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _model_input_dim(
    model_name: str,
    input_shape: tuple[int, ...],
    *,
    requires_flatten: bool | None = None,
) -> tuple[int, dict[str, Any]]:
    """Translate a materialized sample shape into an architecture constructor input size."""
    if len(input_shape) == 2:
        return int(input_shape[-1]), {}
    if len(input_shape) != 3:
        raise ConfigurationError(
            f"Unsupported NIDS input shape {input_shape}; expected 2-D or 3-D tensors"
        )

    packets, features = int(input_shape[1]), int(input_shape[2])
    params: dict[str, Any] = {}
    normalized_name = model_name.strip().lower().replace("-", "_")
    flatten = (
        normalized_name in _FLATTENING_MODEL_NAMES
        if requires_flatten is None
        else bool(requires_flatten)
    )
    if flatten:
        return packets * features, params
    if normalized_name in {"cnn", "transformer"}:
        params["sequence_features"] = features
    return features, params


def _build_model(
    model_cfg: Mapping[str, Any],
    input_shape: tuple[int, ...],
    *,
    device: torch.device,
    num_classes: int,
    seed: int | None = None,
) -> ModelWrapper:
    """Build a registered PyTorch or sklearn NIDS using the shared wrapper contract.

    PyTorch architectures are wrapped in ``PyTorchModelWrapper``. Estimators
    exposing callable ``fit`` and ``predict`` methods are wrapped in
    ``SklearnModelWrapper``. The optional seed is injected as ``random_state``
    when the registered constructor supports that parameter and it was not
    explicitly configured.
    """
    model_name = str(model_cfg["name"])
    arch = NIDS.get(model_name)
    params = dict(model_cfg.get("params", {}))
    requires_flatten = _model_requires_flatten(model_cfg)
    input_dim, shape_params = _model_input_dim(
        model_name,
        input_shape,
        requires_flatten=requires_flatten,
    )
    for key, value in shape_params.items():
        params.setdefault(key, value)

    try:
        signature = inspect.signature(arch)
        accepts_random_state = "random_state" in signature.parameters
    except (TypeError, ValueError):
        accepts_random_state = False
    if accepts_random_state and params.get("random_state") is None and seed is not None:
        params["random_state"] = int(seed)

    model = arch(input_dim=input_dim, num_classes=int(num_classes), **params)
    if isinstance(model, nn.Module):
        return PyTorchModelWrapper(model, device)
    if callable(getattr(model, "fit", None)) and callable(getattr(model, "predict", None)):
        return SklearnModelWrapper(model)
    raise ConfigurationError(
        f"Registered model {model_name!r} produced {type(model).__name__}; expected torch.nn.Module "
        "or an estimator exposing callable fit(...) and predict(...)."
    )


def _load_checkpoint(
    wrapper: ModelWrapper,
    checkpoint: str | Path,
    *,
    checkpoint_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore torch state or a fitted sklearn estimator checkpoint.

    Passing metadata that was already loaded avoids deserializing a potentially
    large Random Forest twice during evaluation.
    """
    if checkpoint_data is None:
        map_location: Any = wrapper.device if isinstance(wrapper, PyTorchModelWrapper) else "cpu"
        data = torch.load(str(checkpoint), map_location=map_location, weights_only=False)
    else:
        data = dict(checkpoint_data)
    if isinstance(wrapper, PyTorchModelWrapper):
        model_state = data.get("model_state")
        if model_state is None:
            raise ConfigurationError(f"Checkpoint {checkpoint} has no PyTorch model_state")
        wrapper.model.load_state_dict(model_state, strict=True)
        wrapper.model.eval()
    elif isinstance(wrapper, SklearnModelWrapper):
        estimator = data.get("estimator")
        if estimator is None:
            raise ConfigurationError(
                f"Checkpoint {checkpoint} has no serialized sklearn estimator. "
                "Re-train this model with the updated workflow."
            )
        if not callable(getattr(estimator, "fit", None)) or not callable(getattr(estimator, "predict", None)):
            raise ConfigurationError(f"Checkpoint {checkpoint} contains an invalid sklearn estimator")
        wrapper.estimator = estimator
        wrapper.capabilities = type(wrapper.capabilities)(
            differentiable=False,
            gradient_input=False,
            probabilistic_output=callable(getattr(estimator, "predict_proba", None)),
            model_family="sklearn",
        )
    else:
        raise ConfigurationError(f"Unsupported model wrapper type {type(wrapper).__name__}")
    return data


def _checkpoint_path(model_cfg: Mapping[str, Any], seed: int, *, role: str) -> Path:
    template = str(model_cfg.get("checkpoint", ""))
    if not template:
        raise ConfigurationError(f"{role}.checkpoint is required for evaluate_attack")
    return Path(template.format(seed=seed))


def _normalization_from_checkpoint(
    checkpoint_data: Mapping[str, Any],
    dataset_feature_names: tuple[str, ...],
    *,
    role: str,
) -> FeatureNormalization:
    feature_names = tuple(str(x) for x in checkpoint_data["feature_names"])
    if feature_names != dataset_feature_names:
        raise ConfigurationError(
            f"Dataset feature schema differs from {role} checkpoint schema: "
            f"{dataset_feature_names} != {feature_names}"
        )
    mean = np.asarray(checkpoint_data["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint_data["normalization_std"], dtype=np.float32)
    return FeatureNormalization(mean=mean, std=std, feature_names=feature_names)


def _load_checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return dict(torch.load(checkpoint, map_location="cpu", weights_only=False))


def _split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y), dtype=np.int64)
    try:
        return tuple(
            np.asarray(part, dtype=np.int64)
            for part in train_test_split(indices, test_size=0.2, random_state=seed, stratify=y)
        )  # type: ignore[return-value]
    except ValueError:
        # Tiny/rare-class datasets may be impossible to stratify. Preserve one
        # training example for every class before splitting the remaining data.
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        forced_train = np.asarray(
            [np.flatnonzero(y == class_value)[0] for class_value in classes],
            dtype=np.int64,
        )
        forced_set = set(forced_train.tolist())
        remaining = np.asarray(
            [index for index in indices if int(index) not in forced_set],
            dtype=np.int64,
        )
        rng.shuffle(remaining)
        train_size = max(1, min(len(indices) - 1, int(round(0.8 * len(indices)))))
        additional_count = max(0, train_size - len(forced_train))
        train_idx = np.concatenate((forced_train, remaining[:additional_count]))
        train_set = set(train_idx.tolist())
        val_idx = np.asarray(
            [index for index in indices if int(index) not in train_set],
            dtype=np.int64,
        )
        return train_idx, val_idx


def _write_sklearn_training_curve(
    output_dir: Path,
    *,
    train_loss: float,
    validation_loss: float,
    train_accuracy: float,
    validation_accuracy: float,
    train_f1: float,
    validation_f1: float,
    defense_mode: str,
) -> None:
    """Write a one-row training curve for estimators trained in a single fit call.

    The column names match ``models.trainer`` so existing cross-seed plotting
    does not fail when a Random Forest seed has no neural-network epoch loop.
    The file is atomically replaced to avoid partial reads by monitoring jobs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "training_curve.csv"
    temporary = output_dir / f"training_curve.{os.getpid()}.tmp"
    fields = (
        "epoch", "trn_loss", "trn_acc", "trn_f1", "val_loss", "val_acc",
        "val_f1", "learning_rate", "defense_mode",
    )
    row = {
        "epoch": 1,
        "trn_loss": train_loss,
        "trn_acc": train_accuracy,
        "trn_f1": train_f1,
        "val_loss": validation_loss,
        "val_acc": validation_accuracy,
        "val_f1": validation_f1,
        "learning_rate": 0.0,
        "defense_mode": defense_mode,
    }
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a tensor/array attack output into a detached CPU NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _make_endpoint_perturbation_mask_fn(
    feature_names: Sequence[str],
    normalization: Any,
    *,
    immutable_features: Sequence[str] | None = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Define a non-MitM attacker's editable feature set for robust training.

    DIR=1 means the packet belongs to the victim, so its entire feature row
    cannot be changed by an endpoint attacker. All DIR columns are immutable;
    any additional immutable protocol fields (FLG by default) are frozen on
    attacker-owned packets as well. The feature-space attack owns threat-model
    interpretation; the trainer receives only this generic mask callback.

    Padding is *not* inferred from feature values here: shuffled batch masks
    travel as an optional third training tensor and are intersected by the
    trainer before PGD-AT/TRADES optimization.
    """
    names = tuple(str(name).upper() for name in feature_names)
    if not names:
        raise ConfigurationError("Asymmetric robust training requires feature_names")
    immutable = tuple(dict.fromkeys(
        str(name).upper() for name in (
            immutable_features if immutable_features is not None
            else (name for name in ("DIR", "FLG") if name in names)
        )
    ))
    unknown = set(immutable).difference(names)
    if unknown:
        raise ConfigurationError(
            f"Unknown robust-training immutable features: {sorted(unknown)}"
        )
    if not hasattr(normalization, "mean") or not hasattr(normalization, "std"):
        raise ConfigurationError("Robust-training directional mask requires normalization metadata")
    mean = normalization.mean
    std = normalization.std
    immutable_indices = tuple(names.index(name) for name in immutable)

    def endpoint_mask(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a device-local binary edit mask for a normalized training batch."""
        del y  # Packet ownership is determined by DIR, not a class label.
        allowed = FeatureSpaceAttack.victim_packet_perturbation_mask(
            x,
            names,
            normalization_mean=mean,
            normalization_std=std,
        )
        if immutable_indices:
            view = allowed.reshape(x.shape[0], -1, len(names))
            for index in immutable_indices:
                view[..., index] = 0.0
        return allowed

    return endpoint_mask


def _validated_packet_mask(
    value: np.ndarray | torch.Tensor,
    *,
    sample_count: int,
    input_shape: Sequence[int],
    feature_count: int | None = None,
) -> torch.Tensor:
    """Validate a real-packet bitmap before binding it to shuffled examples.

    The only authoritative source of padding is the dataset's explicit bitmap;
    inspecting zero-valued normalized inputs cannot distinguish empty padding
    from legitimate zero-valued protocol features.
    """
    packet_mask = torch.as_tensor(value, dtype=torch.bool)
    if len(input_shape) not in (2, 3):
        raise ConfigurationError("Packet masking requires flattened or sequence inputs")
    if packet_mask.ndim != 2 or int(packet_mask.shape[0]) != sample_count:
        raise ConfigurationError("dataset.valid_mask must have shape (N,T)")
    if input_shape[0] != sample_count:
        raise ConfigurationError("Input/packet-mask sample counts differ")
    if len(input_shape) == 3:
        if int(packet_mask.shape[1]) != int(input_shape[1]):
            raise ConfigurationError("dataset.valid_mask T disagrees with sequence shape")
    elif (int(packet_mask.shape[1]) < 1 or
          int(input_shape[1]) % int(packet_mask.shape[1]) != 0):
        raise ConfigurationError("Flattened feature width must be divisible by dataset.valid_mask T")
    if feature_count is not None:
        if feature_count < 1:
            raise ConfigurationError("feature_count must be positive")
        if len(input_shape) == 3 and input_shape[2] != feature_count:
            raise ConfigurationError("Sequence width differs from feature_names")
        if len(input_shape) == 2 and input_shape[1] != packet_mask.shape[1] * feature_count:
            raise ConfigurationError("Flattened width differs from T times feature_names")
    return packet_mask


def _run_empirical_at(
    *,
    eat_cfg: Mapping[str, Any],
    x_tensor: torch.Tensor,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    valid_mask_full: np.ndarray | None,
    feature_names: tuple[str, ...],
    normalization: Any,
    num_classes: int,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
    logger: logging.Logger | None,
    x_physical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Train a differentiable surrogate and generate gatekeeper-filtered EAT samples.

    Sklearn targets remain flat (N,T*F); sequence surrogates see a bijective
    (N,T,F) view of precisely the same ordered coordinates. Attack and clean
    float64 physical buffers are reshaped together, so the gatekeeper checks
    physical packet ownership before accepted examples return to flat target
    layout. Normalization statistics are unchanged (F or T*F entries), and
    attack batches remain bounded at most 256 samples for memory safety.
    """
    surrogate_cfg_raw = eat_cfg.get("surrogate", eat_cfg.get("surrogate_model", {}))
    if not isinstance(surrogate_cfg_raw, Mapping):
        raise ConfigurationError("empirical_at.surrogate must be a mapping with a registered model name")
    surrogate_cfg = dict(surrogate_cfg_raw)
    surrogate_name = str(surrogate_cfg.get("name", "jmila_surrogate"))
    surrogate_model_cfg: dict[str, Any] = {
        "name": surrogate_name,
        "params": dict(surrogate_cfg.get("params", {})),
        # Infer from registered architecture unless explicitly overridden.
        # A CNN/LSTM/Transformer cannot be silently treated as a flat MLP.
        "requires_flatten": _model_requires_flatten(surrogate_cfg),
    }
    if x_tensor.ndim != 2:
        raise ConfigurationError(
            "Sklearn EAT targets require flattened (N,T*F) inputs"
        )
    feature_count = len(feature_names)
    if feature_count == 0 or x_tensor.shape[1] % feature_count != 0:
        raise ConfigurationError(
            "Cannot reshape EAT samples: target width must be divisible by feature_names"
        )
    packets_per_flow = int(x_tensor.shape[1] // feature_count)
    if packets_per_flow < 1:
        raise ConfigurationError("EAT samples must contain at least one packet")

    # Only tensor topology changes. This is a row-major, bijective reshape:
    # flat[t*F + j] == sequence[t,j]. Neither packet order nor the physical
    # feature schema changes, and reshape does not copy contiguous inputs.
    surrogate_input = (
        x_tensor if surrogate_model_cfg["requires_flatten"]
        else x_tensor.reshape(len(x_tensor), packets_per_flow, feature_count)
    )

    surrogate_wrapper = _build_model(
        surrogate_model_cfg,
        tuple(surrogate_input.shape),
        device=device,
        num_classes=num_classes,
        seed=seed,
    )
    if not isinstance(surrogate_wrapper, PyTorchModelWrapper) or not surrogate_wrapper.capabilities.differentiable:
        raise ConfigurationError(
            "empirical_at.surrogate must be a differentiable PyTorch model, for example jmila_surrogate"
        )

    surrogate_training = dict(surrogate_cfg.get("training", {}))
    surrogate_output = checkpoint_dir / "empirical_at_surrogate"
    surrogate_output.mkdir(parents=True, exist_ok=True)
    s_batch_size = int(surrogate_training.get("batch_size", 128))
    if s_batch_size < 1:
        raise ConfigurationError("empirical_at.surrogate.training.batch_size must be >= 1")
    pin = device.type == "cuda"
    generator = torch.Generator().manual_seed(int(seed))
    surrogate_train_x = surrogate_input[train_idx]
    surrogate_val_x = surrogate_input[val_idx]
    surrogate_train_y = torch.as_tensor(y_np[train_idx], dtype=torch.long)
    surrogate_train_ds = TensorDataset(surrogate_train_x, surrogate_train_y)
    if valid_mask_full is not None:
        surrogate_packet_mask = _validated_packet_mask(
            valid_mask_full, sample_count=len(x_tensor), input_shape=tuple(x_tensor.shape),
            feature_count=len(feature_names),
        )
        # Keep validity aligned with examples when the loader shuffles.
        surrogate_train_ds = TensorDataset(
            surrogate_train_x, surrogate_train_y, surrogate_packet_mask[train_idx]
        )
    surrogate_val_ds = TensorDataset(surrogate_val_x, torch.as_tensor(y_np[val_idx], dtype=torch.long))
    surrogate_train_loader = DataLoader(
        surrogate_train_ds, batch_size=s_batch_size, shuffle=True, generator=generator,
        num_workers=0, pin_memory=pin,
    )
    surrogate_val_loader = DataLoader(
        surrogate_val_ds, batch_size=s_batch_size, shuffle=False, num_workers=0, pin_memory=pin,
    )

    epsilon = float(surrogate_training.get("pgd_epsilon", 0.04))
    configured_alpha = surrogate_training.get("pgd_alpha")
    alpha = epsilon / 4.0 if configured_alpha is None else float(configured_alpha)
    early_stopping_patience_raw = surrogate_training.get("early_stopping_patience", 0)
    early_stopping_patience = None if early_stopping_patience_raw is None else int(early_stopping_patience_raw)
    early_stopping_min_delta = float(surrogate_training.get("early_stopping_min_delta", 0.0))
    early_stopping_monitor = str(surrogate_training.get("early_stopping_monitor", "val_loss"))

    surrogate_mask_fn = _make_endpoint_perturbation_mask_fn(
        feature_names,
        normalization,
        immutable_features=surrogate_training.get("immutable_features"),
    )
    surrogate_summary = train_classifier(
        surrogate_wrapper,
        surrogate_train_loader,
        surrogate_val_loader,
        epochs=int(surrogate_training.get("epochs", 5)),
        lr=float(surrogate_training.get("lr", 1e-3)),
        weight_decay=float(surrogate_training.get("weight_decay", 1e-5)),
        adversarial_training=bool(
            surrogate_training.get("adversarial_training", False)
        ),
        defense_mode=str(surrogate_training.get("defense_mode", "none")),
        pgd_epsilon=epsilon,
        pgd_alpha=alpha,
        pgd_steps=int(surrogate_training.get("pgd_steps", 10)),
        trades_beta=float(surrogate_training.get("trades_beta", 6.0)),
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_monitor=early_stopping_monitor,
        output_dir=surrogate_output,
        seed=seed,
        logger=logger,
        perturbation_mask_fn=surrogate_mask_fn,
    )
    surrogate_checkpoint = surrogate_output / "nids.pt"
    torch.save(
        {
            "model_state": surrogate_wrapper.model.state_dict(),
            "model_family": "pytorch",
            "model_name": surrogate_name,
            "model_params": surrogate_model_cfg["params"],
            "num_classes": int(num_classes),
            "classes": list(range(num_classes)),
            "feature_names": list(feature_names),
            "input_shape": list(surrogate_input.shape[1:]),
            "normalization_mean": normalization.mean,
            "normalization_std": normalization.std,
            "normalization_version": getattr(normalization, "version", "unknown"),
            "seed": int(seed),
            "training_purpose": "empirical_at_surrogate",
        },
        surrogate_checkpoint,
    )
    surrogate_wrapper.model.eval()

    augmentation_cfg = dict(eat_cfg.get("augmentation", {}))
    ratio = float(augmentation_cfg.get("ratio", 0.5))
    if not 0.0 <= ratio <= 1.0:
        raise ConfigurationError("empirical_at.augmentation.ratio must be in [0, 1]")
    filter_realizable = bool(augmentation_cfg.get("filter_realizable", True))
    if not filter_realizable:
        raise ConfigurationError(
            "Empirical AT requires augmentation.filter_realizable=true so unphysical samples are never fitted."
        )
    requested_batch = int(augmentation_cfg.get("batch_size", 256))
    if requested_batch < 1:
        raise ConfigurationError("empirical_at.augmentation.batch_size must be >= 1")
    attack_batch_size = min(requested_batch, 256)
    if requested_batch > 256 and logger:
        logger.warning("Capping EAT attack batches from %d to 256 for memory safety", requested_batch)

    benign_label = int(augmentation_cfg.get("benign_label", eat_cfg.get("benign_label", 0)))
    if not 0 <= benign_label < num_classes:
        raise ConfigurationError(f"EAT benign_label={benign_label} is outside [0, {num_classes - 1}]")
    train_labels = y_np[train_idx]
    malicious_idx = train_idx[train_labels != benign_label]
    requested_count = int(np.ceil(len(malicious_idx) * ratio)) if ratio > 0 else 0
    requested_count = min(requested_count, len(malicious_idx))
    selection_rng = np.random.default_rng(int(seed))
    selected_idx = (
        np.sort(selection_rng.choice(malicious_idx, size=requested_count, replace=False)).astype(np.int64)
        if requested_count else np.empty((0,), dtype=np.int64)
    )

    attack_cfg = eat_cfg.get("attack", {"name": "pgd", "params": {}})
    if not isinstance(attack_cfg, Mapping):
        raise ConfigurationError("empirical_at.attack must be a mapping with a registered attack name")
    attack_cls, attack_params = _instantiate_attack(
        {}, feature_names, target_label=benign_label, attack_cfg_override=attack_cfg,
    )
    if not issubclass(attack_cls, FeatureSpaceAttack):
        raise ConfigurationError("Empirical AT requires a feature-space attack such as PGD")
    if getattr(attack_cls, "requires_gradients", False) and not surrogate_wrapper.capabilities.differentiable:
        raise ConfigurationError(f"EAT attack {attack_cls.__name__} requires a differentiable surrogate")
    attack = attack_cls(**attack_params).bind_model(surrogate_wrapper)
    if isinstance(attack, FeatureSpaceAttack):
        discrete_fields = {"PL", "TTL", "WIN"}
        immutable_features = {
            str(name).upper() for name in getattr(attack, "immutable_features", ())
        }
        if not bool(getattr(attack, "discrete_projection", False)) and not discrete_fields.issubset(
            immutable_features
        ):
            mutable_discrete = sorted(discrete_fields - immutable_features)
            raise ConfigurationError(
                "Empirical AT cannot attack discrete protocol features without lattice projection; "
                f"enable discrete_projection or mark these features immutable: {mutable_discrete}"
            )

    gatekeeper = UniversalConstraintGatekeeper(
        strict=False,
        normalization_mean=np.asarray(normalization.mean, dtype=np.float32),
        normalization_std=np.asarray(normalization.std, dtype=np.float32),
    )
    valid_adversarial: list[np.ndarray] = []
    valid_adversarial_labels: list[np.ndarray] = []
    generated = 0
    accepted_count = 0
    attack_success_count = 0
    violation_counts: dict[str, int] = {}
    if logger:
        logger.info(
            "eat.start seed=%d surrogate=%s attack=%s selected=%d batch_size=%d "
            "target_label=%d",
            seed, surrogate_name, attack_cls.__name__, len(selected_idx),
            attack_batch_size, benign_label,
        )

    for b_start in range(0, len(selected_idx), attack_batch_size):
        batch_indices = selected_idx[b_start : b_start + attack_batch_size]
        x_clean = x_tensor[batch_indices].detach().cpu().numpy().astype(np.float32, copy=False)
        x_clean_phys = x_physical[batch_indices]
        # Keep authoritative clean coordinates in the SAME topology as the
        # surrogate's attack input. This is essential for physical victim-row
        # restoration and independent gatekeeper directional comparisons.
        if tuple(x_clean_phys.shape) != tuple(x_clean.shape):
            raise ConfigurationError("EAT clean physical and normalized shapes differ")
        x_attack = (
            x_clean if surrogate_model_cfg["requires_flatten"]
            else x_clean.reshape(len(x_clean), packets_per_flow, feature_count)
        )
        x_attack_phys = (
            x_clean_phys if surrogate_model_cfg["requires_flatten"]
            else x_clean_phys.reshape(len(x_clean), packets_per_flow, feature_count)
        )
        y_clean = y_np[batch_indices].astype(np.int64, copy=False)
        batch_kwargs: dict[str, Any] = {
            "seed": int(seed + b_start),
            "target_label": benign_label,
            "normalization_mean": normalization.mean,
            "normalization_std": normalization.std,
            "_clean_authoritative_physical": x_attack_phys,
        }
        batch_valid_mask = None
        if valid_mask_full is not None:
            batch_valid_mask = np.asarray(valid_mask_full, dtype=bool)[batch_indices]
            batch_kwargs["valid_mask"] = batch_valid_mask
        result = attack.generate(x_attack, y_clean, **batch_kwargs)
        if result.adversarial is None or result.raw_adversarial is not None:
            raise ConfigurationError("EAT feature-space attack must return result.adversarial")
        x_adv = _as_numpy(result.adversarial).astype(np.float32, copy=False)
        if x_adv.shape != x_attack.shape:
            raise ConfigurationError(
                f"EAT attack {attack_cls.__name__} changed surrogate shape "
                f"{x_attack.shape} -> {x_adv.shape}"
            )
        result_names = tuple(result.feature_names or feature_names)
        if result_names != feature_names:
            raise ConfigurationError("EAT attack output feature schema differs from the target dataset")

        gate = gatekeeper.validate(
            x_adv,
            feature_names,
            valid_mask=batch_valid_mask,
            authoritative_physical=result.authoritative_physical,   # From AttackResult
            original_physical=x_attack_phys,                       # Pristine baseline in surrogate layout
        )
        accepted = np.asarray(gate.accepted, dtype=bool).copy()
        if result.intrinsic_feasible_mask is not None:
            intrinsic = np.asarray(result.intrinsic_feasible_mask, dtype=bool).reshape(-1)
            if len(intrinsic) != len(accepted):
                raise ConfigurationError("Attack intrinsic_feasible_mask length does not match generated batch")
            accepted &= intrinsic

        generated += len(x_adv)
        accepted_count += int(accepted.sum())
        if result.success_mask is not None:
            attack_success_count += int(np.asarray(_as_numpy(result.success_mask), dtype=bool).sum())
        for violation in gate.violations:
            violation_counts[violation.rule] = violation_counts.get(violation.rule, 0) + int(
                len(np.unique(np.asarray(violation.indices, dtype=np.int64)))
            )
        if accepted.any():
            # The sklearn target ONLY receives (N,T*F). Reshape after oracle
            # validation, preserving the exact float32 coordinate encoding.
            # Boolean indexing already allocates its necessary output buffer.
            x_adv_target = (
                x_adv.reshape(x_clean.shape) if x_adv.ndim == 3 else x_adv
            )
            if x_adv_target.shape != x_clean.shape:
                raise ConfigurationError("EAT output does not match flat sklearn target shape")
            valid_adversarial.append(x_adv_target[accepted])
            # Keep the original malicious label when robustifying the target NIDS.
            valid_adversarial_labels.append(y_clean[accepted].copy())
        if logger:
            logger.info(
                "eat.batch seed=%d batch=%d:%d generated=%d accepted=%d success=%d",
                seed, b_start, b_start + len(batch_indices), len(x_adv),
                int(accepted.sum()),
                int(np.asarray(_as_numpy(result.success_mask), dtype=bool).sum())
                if result.success_mask is not None else 0,
            )
        del result, x_adv, x_clean

    if valid_adversarial:
        adversarial_x = np.vstack(valid_adversarial).astype(np.float32, copy=False)
        adversarial_y = np.concatenate(valid_adversarial_labels).astype(np.int64, copy=False)
    else:
        adversarial_x = np.empty((0, *tuple(x_tensor.shape[1:])), dtype=np.float32)
        adversarial_y = np.empty((0,), dtype=np.int64)

    stats = {
        "eat_surrogate_trained": 1.0,
        "eat_samples_selected": float(len(selected_idx)),
        "eat_samples_generated": float(generated),
        "eat_samples_realizable": float(accepted_count),
        "eat_realizability_rate": float(accepted_count / generated) if generated else 1.0,
        "eat_surrogate_attack_success_count": float(attack_success_count),
        "eat_surrogate_fit_loss": float(surrogate_summary.train_loss),
        "eat_surrogate_validation_loss": float(surrogate_summary.validation_loss or 0.0),
    }
    for rule, count in sorted(violation_counts.items()):
        stats[f"eat_violation_{rule}"] = float(count)
    if logger:
        logger.info(
            "[Seed %d] EAT generated %d adversarial samples; %d passed realizability checks (RR=%.2f%%)",
            seed, generated, accepted_count, 100.0 * stats["eat_realizability_rate"],
        )
    return adversarial_x, adversarial_y, stats


def train_nids_seed(*, config: Mapping[str, Any], seed: int, gpu: str, logger: logging.Logger | None = None) -> Mapping[str, float]:
    """Train and checkpoint one PyTorch or scikit-learn target NIDS seed.

    The split and normalization contracts remain identical across model families.
    PyTorch models use ``train_classifier`` (including optional PGD-AT/TRADES),
    while sklearn estimators use ``fit`` directly and never enter a torch optimizer.
    """
    cfg = _cfg(config)
    device_info = configure_process_device(gpu)
    seed_everything(seed)
    _inject_dynamic_dataset_shape(cfg)
    if logger:
        logger.info(
            "train_nids.start seed=%d device=%s model=%s dataset=%s",
            seed, device_info.torch_device, cfg["model"]["name"], cfg["dataset"]["name"],
        )

    dataset = _instantiate_dataset(cfg)
    x_raw, y_np = dataset.raw_arrays()
    y_np = np.asarray(y_np, dtype=np.int64)
    if logger:
        logger.info(
            "train_nids.data_loaded seed=%d samples=%d features=%d shape=%s classes=%d",
            seed, len(y_np), len(dataset.feature_names), tuple(x_raw.shape),
            len(getattr(dataset, "classes", np.unique(y_np))),
        )
    if len(y_np) < 10:
        raise ConfigurationError("Training requires at least 10 samples")

    train_idx, val_idx = _split_indices(y_np, seed)
    if logger:
        logger.info(
            "train_nids.split seed=%d train=%d validation=%d",
            seed, len(train_idx), len(val_idx),
        )
    if hasattr(dataset, "valid_mask"):
        normalization = dataset.fit_normalization(
            x_raw[train_idx],
            valid_mask=np.asarray(dataset.valid_mask)[train_idx],
        )
    else:
        normalization = dataset.fit_normalization(x_raw[train_idx])
    x_norm = normalization.normalize(x_raw).astype(np.float32)
    x_tensor = torch.as_tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_np, dtype=torch.long)
    del x_norm

    model_cfg = dict(cfg["model"])
    classes = getattr(dataset, "classes", tuple(np.unique(y_np)))
    num_classes = int(len(classes))
    wrapper = _build_model(
        model_cfg,
        (len(x_tensor), *tuple(x_tensor.shape[1:])),
        device=device_info.torch_device,
        num_classes=num_classes,
        seed=seed,
    )
    if logger:
        logger.info(
            "train_nids.model_ready seed=%d model=%s family=%s input_shape=%s parameters=%d",
            seed, model_cfg["name"], type(wrapper).__name__, tuple(x_tensor.shape[1:]),
            sum(parameter.numel() for parameter in wrapper.model.parameters())
            if isinstance(wrapper, PyTorchModelWrapper) else 0,
        )

    tcfg = dict(model_cfg.get("training", {}))
    batch_size = int(tcfg.get("batch_size", 256))
    if batch_size < 1:
        raise ConfigurationError("model.training.batch_size must be >= 1")
    pin = device_info.torch_device.type == "cuda"
    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    val_ds = TensorDataset(x_tensor[val_idx], y_tensor[val_idx])

    configured_mode = str(tcfg.get("defense_mode", "none") or "none").strip().lower()
    robust_training = (
        isinstance(wrapper, PyTorchModelWrapper)
        and (configured_mode in {"pgd", "trades"} or bool(tcfg.get("adversarial_training", False)))
    )
    train_defense_ds = train_ds
    if robust_training and hasattr(dataset, "valid_mask") and dataset.valid_mask is not None:
        packet_mask = _validated_packet_mask(
            dataset.valid_mask, sample_count=len(x_tensor), input_shape=tuple(x_tensor.shape),
            feature_count=len(dataset.feature_names),
        )
        train_defense_ds = TensorDataset(
            train_ds.tensors[0], train_ds.tensors[1], packet_mask[train_idx]
        )

    train_loader = cast(
        DataLoader[TrainingBatch],
        DataLoader(train_defense_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin),
    )
    val_loader = cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin),
    )

    checkpoint_dir = Path(cfg.get("output_dir", "../runs")) / "checkpoints" / f"{model_cfg['name']}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    defense_mode = str(tcfg.get("defense_mode", "none") or "none").lower()
    requested_strategy = str(tcfg.get("strategy", "auto") or "auto").strip().lower()
    if requested_strategy in {"torch", "pytorch", "dl"}:
        requested_strategy = "pytorch"
    elif requested_strategy in {"sklearn", "scikit_learn", "scikit-learn", "estimator"}:
        requested_strategy = "sklearn"
    elif requested_strategy not in {"auto", ""}:
        raise ConfigurationError(
            f"Unsupported model.training.strategy={requested_strategy!r}; use 'pytorch', 'sklearn', or 'auto'."
        )
    actual_strategy = "sklearn" if isinstance(wrapper, SklearnModelWrapper) else "pytorch"
    if requested_strategy not in {"auto", ""} and requested_strategy != actual_strategy:
        raise ConfigurationError(
            f"model.training.strategy={requested_strategy!r} conflicts with registered model "
            f"{model_cfg['name']!r}, which uses {actual_strategy!r} training."
        )

    detector_cfg = dict(cfg.get("detector", {}) or {})
    detector_enabled = bool(detector_cfg.get("enabled", False))
    augmented_train_x: np.ndarray | None = None
    augmented_train_y: np.ndarray | None = None
    # Detector fitting consumes (x,y) pairs. The third packet-validity tensor
    # is exclusively for robust inner optimization, not detector feature fits.
    detector_train_loader = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        if train_defense_ds is not train_ds else train_loader
    )
    eat_stats: dict[str, float] = {}

    if logger:
        logger.info(
            "train_nids.strategy seed=%d requested=%s actual=%s defense=%s detector=%s",
            seed, requested_strategy or "auto", actual_strategy, defense_mode,
            detector_cfg.get("name", "none") if detector_enabled else "none",
        )

    if isinstance(wrapper, SklearnModelWrapper):
        if bool(tcfg.get("adversarial_training", False)):
            raise ConfigurationError(
                "model.training.adversarial_training is a PyTorch option. For sklearn estimators, "
                "use defense_mode=empirical_at with a differentiable surrogate."
            )
        if defense_mode not in {"none", "", "null", "empirical_at"}:
            raise ConfigurationError(
                f"Unsupported sklearn defense_mode={defense_mode!r}; use 'none' or 'empirical_at'."
            )

        clean_train_x = x_tensor[train_idx].cpu().numpy()
        val_x = x_tensor[val_idx].cpu().numpy()
        clean_train_y = y_np[train_idx]
        val_y = y_np[val_idx]
        train_x = clean_train_x
        train_y = clean_train_y
        if train_x.ndim != 2 or val_x.ndim != 2:
            raise ConfigurationError(
                "scikit-learn estimators require 2-D flow vectors; check model.requires_flatten and the dataset"
            )
        expected_classes = np.arange(num_classes, dtype=np.int64)
        if not np.array_equal(np.unique(train_y), expected_classes):
            raise ConfigurationError(
                "The training split does not contain every encoded class. Increase the dataset size or revise "
                "the split so the estimator learns the complete label schema."
            )

        augmented_train_x = clean_train_x
        augmented_train_y = clean_train_y
        if defense_mode == "empirical_at":
            eat_cfg = dict(tcfg.get("empirical_at", {}) or {})
            if not eat_cfg:
                raise ConfigurationError(
                    "defense_mode=empirical_at requires model.training.empirical_at settings; "
                    "compose configs/model/training/empirical_at.yaml."
                )
            valid_mask_full = getattr(dataset, "valid_mask", None)
            adversarial_x, adversarial_y, eat_stats = _run_empirical_at(
                eat_cfg=eat_cfg,
                x_tensor=x_tensor,
                y_np=y_np,
                train_idx=train_idx,
                val_idx=val_idx,
                valid_mask_full=valid_mask_full,
                feature_names=tuple(dataset.feature_names),
                normalization=normalization,
                num_classes=num_classes,
                device=device_info.torch_device,
                seed=seed,
                checkpoint_dir=checkpoint_dir,
                logger=logger,
                x_physical=x_raw
            )
            del x_raw
            augmented_train_x = np.vstack((clean_train_x, adversarial_x))
            augmented_train_y = np.concatenate((clean_train_y, adversarial_y), axis=0)
            eat_stats["eat_augmented_samples"] = float(len(adversarial_y))
            eat_stats["eat_training_samples_total"] = float(len(augmented_train_y))
            # Mahalanobis reference moments must describe the CLEAN training
            # distribution, even though the defended sklearn classifier was fit
            # on (clean + adversarial) traffic. Including adversarial examples
            # would shift class-conditional means and pooled covariance, changing
            # the anomaly detector's reference hypothesis. 
            detector_dataset = train_ds
            detector_train_loader = cast(
                DataLoader[tuple[torch.Tensor, torch.Tensor]],
                DataLoader(
                    detector_dataset,
                    batch_size=min(batch_size, 256),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin,
                ),
            )
        else:
            del x_raw
        
        estimator = wrapper.estimator
        if logger:
            logger.info(
                "training.start framework=sklearn seed=%d model=%s samples=%d validation=%d defense=%s",
                seed, model_cfg["name"], len(augmented_train_y), len(val_y), defense_mode,
            )
        estimator.fit(augmented_train_x, augmented_train_y)
        # Report training metrics on the actual augmented fit set, not only on
        # the original clean subset. Validation remains clean and held out.
        inference_batch_size = 256

        def batched_estimator_predict(features: np.ndarray) -> np.ndarray:
            predictions = [
                np.asarray(estimator.predict(features[start:start + inference_batch_size]))
                for start in range(0, len(features), inference_batch_size)
            ]
            return (
                np.concatenate(predictions).astype(np.int64, copy=False)
                if predictions
                else np.empty(0, dtype=np.int64)
            )

        def batched_estimator_predict_proba(features: np.ndarray) -> np.ndarray:
            probabilities = [
                np.asarray(estimator.predict_proba(features[start:start + inference_batch_size]))
                for start in range(0, len(features), inference_batch_size)
            ]
            return (
                np.concatenate(probabilities, axis=0).astype(np.float64, copy=False)
                if probabilities
                else np.empty((0, len(getattr(estimator, "classes_", ()))), dtype=np.float64)
            )

        train_pred = batched_estimator_predict(augmented_train_x)
        val_pred = batched_estimator_predict(val_x)
        train_acc = float(accuracy_score(augmented_train_y, train_pred))
        val_acc = float(accuracy_score(val_y, val_pred))
        train_f1 = float(f1_score(augmented_train_y, train_pred, average="macro", zero_division=0))
        val_f1 = float(f1_score(val_y, val_pred, average="macro", zero_division=0))

        if callable(getattr(estimator, "predict_proba", None)):
            train_prob = batched_estimator_predict_proba(augmented_train_x)
            val_prob = batched_estimator_predict_proba(val_x)
            estimator_classes = np.asarray(estimator.classes_, dtype=np.int64)
            train_loss = float(log_loss(augmented_train_y, train_prob, labels=estimator_classes))
            validation_loss = float(log_loss(val_y, val_prob, labels=estimator_classes))
        else:
            train_loss = float("nan")
            validation_loss = float("nan")

        defense_mode = "empirical_at" if eat_stats else "none"
        _write_sklearn_training_curve(
            checkpoint_dir,
            train_loss=train_loss,
            validation_loss=validation_loss,
            train_accuracy=train_acc,
            validation_accuracy=val_acc,
            train_f1=train_f1,
            validation_f1=val_f1,
            defense_mode=defense_mode,
        )
        if logger:
            logger.info(
                "[Seed %d] Fitted sklearn %s | train=%d | validation=%d | "
                "train log-loss=%.6f | validation log-loss=%.6f",
                seed, model_cfg["name"], len(augmented_train_y), len(val_y), train_loss, validation_loss,
            )

    else:
        if defense_mode == "empirical_at":
            raise ConfigurationError(
                "defense_mode=empirical_at is specifically for sklearn targets; PyTorch targets should use "
                "defense_mode=pgd or defense_mode=trades."
            )
        pgd_epsilon = float(tcfg.get("pgd_epsilon", 0.04))
        configured_alpha = tcfg.get("pgd_alpha")
        pgd_alpha = pgd_epsilon / 4.0 if configured_alpha is None else float(configured_alpha)

        early_stopping_patience_raw = tcfg.get("early_stopping_patience", 0)
        early_stopping_patience = None if early_stopping_patience_raw is None else int(early_stopping_patience_raw)
        early_stopping_min_delta = float(tcfg.get("early_stopping_min_delta", 0.0))
        early_stopping_monitor = str(tcfg.get("early_stopping_monitor", "val_loss"))

        endpoint_mask_fn = (
            _make_endpoint_perturbation_mask_fn(
                tuple(dataset.feature_names),
                normalization,
                immutable_features=tcfg.get("immutable_features"),
            )
            if robust_training else None
        )
        summary = train_classifier(
            wrapper,
            train_loader,
            val_loader,
            epochs=int(tcfg.get("epochs", 20)),
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            adversarial_training=bool(tcfg.get("adversarial_training", False)),
            defense_mode=str(tcfg["defense_mode"]) if tcfg.get("defense_mode") is not None else None,
            pgd_epsilon=pgd_epsilon,
            pgd_alpha=pgd_alpha,
            pgd_steps=int(tcfg.get("pgd_steps", 10)),
            trades_beta=float(tcfg.get("trades_beta", 6.0)),
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_monitor=early_stopping_monitor,
            output_dir=checkpoint_dir,
            seed=seed,
            logger=logger,
            perturbation_mask_fn=endpoint_mask_fn,
        )

        train_loss = float(summary.train_loss)
        validation_loss = float(summary.validation_loss or 0.0)
        defense_mode = str(summary.defense_mode)

    # Fit inference detectors after the target has finished fitting. The wrapper's
    # polymorphic extract_features contract supports PyTorch and sklearn targets.
    detector = None
    detector_name: str | None = None
    detector_params: dict[str, Any] = {}
    if detector_enabled:
        if not callable(getattr(wrapper, "extract_features", None)):
            raise ConfigurationError(
                f"Model wrapper {type(wrapper).__name__} does not implement extract_features required by the detector"
            )
        detector_name = str(detector_cfg.get("name", "mahalanobis"))
        detector_params = dict(detector_cfg.get("params", {}))
        detector_cls = DETECTORS.get(detector_name)
        detector = detector_cls(**detector_params)
        detector.fit(detector_train_loader, wrapper)
        detector_path = checkpoint_dir / "detector.pt"
        torch.save(
            {
                "name": detector_name,
                "params": detector_params,
                "state_dict": detector.state_dict(),
                "seed": int(seed),
                "fit_split": "train",
                "classifier_defense_mode": defense_mode,
            },
            detector_path,
        )
        if logger:
            logger.info(
                "Fitted %s detector on %d training samples after %s training; saved to %s",
                detector_name, len(detector_train_loader.dataset), defense_mode, detector_path,
            )

    checkpoint = checkpoint_dir / "nids.pt"
    checkpoint_payload: dict[str, Any] = {
        "model_state": wrapper.model.state_dict() if isinstance(wrapper, PyTorchModelWrapper) else None,
        "estimator": wrapper.estimator if isinstance(wrapper, SklearnModelWrapper) else None,
        "model_family": "pytorch" if isinstance(wrapper, PyTorchModelWrapper) else "sklearn",
        "model_name": model_cfg["name"],
        "model_params": model_cfg.get("params", {}),
        "num_classes": num_classes,
        "classes": list(classes),
        "class_mapping": dict(getattr(dataset, "class_mapping", {})),
        "feature_names": list(dataset.feature_names),
        "input_shape": list(x_tensor.shape[1:]),
        "normalization_mean": normalization.mean,
        "normalization_std": normalization.std,
        "normalization_version": normalization.version,
        "defense_mode": defense_mode,
        "detector": None if detector is None else {
            "name": detector_name,
            "params": detector_params,
            "checkpoint": "detector.pt",
            "fit_split": "train",
        },
        "seed": int(seed),
    }
    torch.save(checkpoint_payload, checkpoint)
    if logger:
        logger.info(
            "train_nids.checkpoint_saved seed=%d path=%s family=%s defense=%s train_loss=%.5f val_loss=%.5f",
            seed, checkpoint, checkpoint_payload["model_family"], defense_mode,
            train_loss, validation_loss,
        )
    result = {
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "defense_mode_pgd": float(defense_mode == "pgd"),
        "defense_mode_trades": float(defense_mode == "trades"),
        "detector_fitted": float(detector is not None),
        "model_is_sklearn": float(isinstance(wrapper, SklearnModelWrapper)),
    }
    if detector is not None:
        result["detector_fit_samples"] = float(len(detector_train_loader.dataset))
    result.update(eat_stats)
    result["defense_mode_empirical_at"] = float(defense_mode == "empirical_at")
    return result


def _instantiate_attack(
    cfg: Mapping[str, Any],
    feature_names: tuple[str, ...],
    *,
    target_label: int | None = None,
    attack_cfg_override: Mapping[str, Any] | None = None,
) -> tuple[type[BaseAttack], dict[str, Any]]:
    """Instantiate an attack from the global evaluation config or a local block."""
    attack_cfg = dict(attack_cfg_override if attack_cfg_override is not None else cfg["attack"])
    attack_name = str(attack_cfg["name"])
    attack_cls = ATTACKS.get(attack_name)
    params = {key: value for key, value in attack_cfg.items() if key not in {"name", "params", "variant", "training"}}
    params.update(dict(attack_cfg.get("params", {})))
    if issubclass(attack_cls, FeatureSpaceAttack):
        params.setdefault("feature_names", feature_names)

    signature = inspect.signature(attack_cls)
    declared_parameters = signature.parameters
    if target_label is not None:
        if "target_label" in declared_parameters:
            params["target_label"] = int(target_label)
        if "targeted" in declared_parameters:
            params["targeted"] = True

    # ``**kwargs`` is not evidence that an attack supports a semantic option.
    # In particular, TRADES accepts kwargs only to forward feature-space
    # configuration to its base class; forwarding target_label/targeted would
    # make FeatureSpaceAttack.__init__ reject the argument.
    for semantic_option in ("target_label", "targeted"):
        if semantic_option in params and semantic_option not in declared_parameters:
            raise ConfigurationError(
                f"Attack {attack_cls.__name__} does not declare {semantic_option!r}; "
                "remove it from the attack configuration."
            )
    return attack_cls, params



def evaluate_attack_seed(*, config: Mapping[str, Any], seed: int, gpu: str, logger: logging.Logger | None = None) -> Mapping[str, float]:
    """Evaluate one attack seed against a frozen target-model checkpoint.

    The optional surrogate is used only to generate adversarial inputs; target
    inference, constraint checks, and reporting remain owned by the target
    evaluation path.
    """
    cfg = _cfg(config)
    device_info = configure_process_device(gpu)
    seed_everything(seed)
    _inject_dynamic_dataset_shape(cfg)
    dataset = _instantiate_dataset(cfg)
    x_raw, y_np = dataset.raw_arrays()
    y_np = np.asarray(y_np, dtype=np.int64)
    evaluation_cfg = dict(cfg.get("evaluation", {}))
    max_samples = evaluation_cfg.get("max_samples")
    sample_indices: np.ndarray | None = None
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ConfigurationError("evaluation.max_samples must be >= 1 or null")
        if len(y_np) > max_samples:
            sample_indices = np.sort(
                np.random.default_rng(seed).choice(len(y_np), size=max_samples, replace=False)
            )
            x_raw = x_raw[sample_indices]
            y_np = y_np[sample_indices]
    feature_names = tuple(dataset.feature_names)
    if logger:
        logger.info(
            "evaluate.start seed=%d device=%s dataset=%s samples=%d features=%d attack=%s "
            "target=%s max_samples=%s",
            seed, device_info.torch_device, cfg["dataset"]["name"], len(y_np),
            len(feature_names), cfg["attack"]["name"], cfg["model"]["name"], max_samples or "all",
        )
    
    # keep x_raw for floating drift checks, but the target model will receive a normalized copy
    clean_physical = x_raw
    # -------------------- target model --------------------
    target_cfg = dict(cfg["model"])
    target_ckpt = _checkpoint_path(target_cfg, seed, role="model")
    target_meta = _load_checkpoint_metadata(target_ckpt)
    target_norm = _normalization_from_checkpoint(target_meta, feature_names, role="target model")
    x_target = target_norm.normalize(x_raw).astype(np.float32, copy=False)
    # The target-normalized copy is now the evaluation input.
    # Drop the dataset plugin's canonical/raw cache before loading
    # the surrogate or starting the attack. Sequence datasets implement this
    # release hook; other dataset plugins may simply be garbage-collected here.
    release_raw_features = getattr(dataset, "release_raw_features", None)
    if callable(release_raw_features):
        release_raw_features()
    gc.collect()
    target_num_classes = int(target_meta["num_classes"])
    target_wrapper = _build_model(
        target_cfg,
        tuple(x_target.shape),
        device=device_info.torch_device,
        num_classes=target_num_classes,
        seed=seed,
    )
    _load_checkpoint(target_wrapper, target_ckpt, checkpoint_data=target_meta)
    if logger:
        logger.info(
            "evaluate.target_loaded seed=%d model=%s family=%s checkpoint=%s input_shape=%s",
            seed, target_cfg["name"], type(target_wrapper).__name__, target_ckpt, tuple(x_target.shape[1:]),
        )

    # Optional inference-time detector fitted after the target model's training-time
    # defense. Detector state lives adjacent to the classifier checkpoint.
    detector_cfg = dict(cfg.get("detector", {}) or {})
    detector = None
    if bool(detector_cfg.get("enabled", False)):
        if not callable(getattr(target_wrapper, "extract_features", None)):
            raise ConfigurationError(
                f"Target wrapper {type(target_wrapper).__name__} does not support detector feature extraction"
            )
        detector_name = str(detector_cfg.get("name", "mahalanobis"))
        detector_params = dict(detector_cfg.get("params", {}))
        detector_cls = DETECTORS.get(detector_name)
        detector = detector_cls(**detector_params)
        configured_detector_path = detector_cfg.get("checkpoint")
        detector_path = (
            Path(str(configured_detector_path).format(seed=seed))
            if configured_detector_path
            else target_ckpt.parent / "detector.pt"
        )
        if not detector_path.exists():
            raise ConfigurationError(
                f"Detector {detector_name!r} is enabled but its fitted state is missing: {detector_path}. "
                "Run train_nids with detector.enabled=true after training the classifier."
            )
        detector_payload = torch.load(detector_path, map_location="cpu", weights_only=False)
        saved_name = str(detector_payload.get("name", detector_name))
        if saved_name != detector_name:
            raise ConfigurationError(
                f"Configured detector {detector_name!r} does not match checkpoint detector {saved_name!r}"
            )
        detector.load_state_dict(detector_payload["state_dict"])
        if logger:
            logger.info("Loaded %s detector state from %s", detector_name, detector_path)

    # -------------------- optional surrogate --------------------
    surrogate_wrapper: ModelWrapper | None = None
    attack_input: Any = x_target
    attack_norm = target_norm
    surrogate_meta: dict[str, Any] | None = None
    attack_input_is_target_normalized = False
    surrogate_cfg_raw = cfg.get("surrogate_model")
    surrogate_cfg = dict(surrogate_cfg_raw) if surrogate_cfg_raw else None
    surrogate_enabled = bool(surrogate_cfg.get("enabled", True)) if surrogate_cfg is not None else False
    if surrogate_cfg is not None and surrogate_enabled:
        surrogate_ckpt = _checkpoint_path(surrogate_cfg, seed, role="surrogate_model")
        surrogate_meta = _load_checkpoint_metadata(surrogate_ckpt)
        surrogate_norm = _normalization_from_checkpoint(
            surrogate_meta,
            feature_names,
            role="surrogate model",
        )
        # The model input shape is identical across target/surrogate schemas.
        # Convert target-normalized feature batches to surrogate normalization
        # inside EvaluationEngine rather than allocating x_surrogate for all N.
        surrogate_wrapper = _build_model(
            surrogate_cfg,
            tuple(x_target.shape),
            device=device_info.torch_device,
            num_classes=int(surrogate_meta["num_classes"]),
            seed=seed,
        )
        _load_checkpoint(surrogate_wrapper, surrogate_ckpt, checkpoint_data=surrogate_meta)
        attack_input = x_target
        attack_norm = surrogate_norm
        attack_input_is_target_normalized = True
        if logger:
            logger.info(
                "evaluate.surrogate_loaded seed=%d model=%s checkpoint=%s normalization=on_demand",
                seed, surrogate_cfg["name"], surrogate_ckpt,
            )

    benign_label = int(evaluation_cfg.get("benign_label", 0))

    # The attack receives only the surrogate (if present), never the target model.
    # Every evaluation attack is explicitly directed toward the benign class.
    attack_cls, attack_params = _instantiate_attack(
        cfg,
        feature_names,
        target_label=benign_label,
    )
    attack: BaseAttack = attack_cls(**attack_params)
    attack_model = surrogate_wrapper or target_wrapper
    if getattr(attack_cls, "requires_gradients", False) and not attack_model.capabilities.differentiable:
        raise ConfigurationError(
            f"Attack {attack_cls.__name__} requires input gradients, but the selected attack model "
            f"is {attack_model.capabilities.model_family!r}. Configure a differentiable PyTorch "
            "surrogate_model for attacks against a Random Forest target."
        )
    if surrogate_wrapper is not None and getattr(attack_cls, "requires_gradients", False) and not isinstance(surrogate_wrapper, PyTorchModelWrapper):
        raise ConfigurationError(
            "Gradient-based attacks require a PyTorch surrogate_model; the configured surrogate is not differentiable."
        )
    if logger:
        logger.info(
            "evaluate.attack_ready seed=%d attack=%s params=%s model=%s problem_space=%s",
            seed, attack_cls.__name__, {key: value for key, value in attack_params.items() if key != "feature_names"},
            type(attack_model).__name__, issubclass(attack_cls, ProblemSpaceAttack),
        )

    preprocessor = None
    if issubclass(attack_cls, ProblemSpaceAttack):
        raw_traffic = getattr(dataset, "raw_traffic", None)
        if raw_traffic is None:
            raise ConfigurationError(
                "Problem-space evaluation requires dataset.raw_traffic (PCAP/packet objects) "
                "to be provided by the dataset plugin"
            )
        preprocessing_cfg = dict(cfg.get("preprocessing", {}))
        processor_name = str(preprocessing_cfg.get("name", ""))
        if not processor_name:
            raise ConfigurationError("preprocessing.name is required for problem-space evaluation")
        preprocessor = PREPROCESSORS.get(processor_name)(**dict(preprocessing_cfg.get("params", {})))
        if hasattr(preprocessor, "load_state_dict"):
            preprocessor.load_state_dict(
                {
                    "normalize": True,
                    "mean": target_norm.mean.tolist(),
                    "std": target_norm.std.tolist(),
                }
            )
    else:
        raw_traffic = None

    strict_constraints = bool(evaluation_cfg.get("strict_constraints", False))
    strict_multiclass = bool(evaluation_cfg.get("strict_multiclass", True))
    class_names = tuple(
        str(name) for name in getattr(
            dataset, "classes", tuple(f"Class_{i}" for i in range(target_num_classes))
        )
    )
    gatekeeper = UniversalConstraintGatekeeper(
        strict=strict_constraints,
        normalization_mean=target_norm.mean,
        normalization_std=target_norm.std,
    )
    engine = EvaluationEngine(
        target_wrapper,
        attack,
        gatekeeper,
        attack_model=attack_model,
        preprocessor=preprocessor,
        attack_normalization=attack_norm,
        target_normalization=target_norm,
        class_labels=list(range(target_num_classes)),
        class_names=class_names,
        benign_label=benign_label,
        target_label=attack_params.get("target_label"),
        c_fp=float(evaluation_cfg.get("c_fp", 1.0)),
        c_fn=float(evaluation_cfg.get("c_fn", 1.0)),
        strict_constraints=strict_constraints,
        strict_multiclass=strict_multiclass,
        detector=detector,
        batch_size=int(evaluation_cfg.get("batch_size", 256)),
        max_samples=max_samples,
        logger=logger,
    )

    valid_mask = getattr(dataset, "valid_mask", None)
    if sample_indices is not None:
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask)[sample_indices]
        if raw_traffic is not None:
            if isinstance(raw_traffic, np.ndarray):
                raw_traffic = raw_traffic[sample_indices]
            else:
                raw_traffic = [raw_traffic[index] for index in sample_indices]
    
    # Keep features and labels as NumPy arrays on the CPU.
    # EvaluationEngine will slice them, and the individual Attack strategies will move the slices to the GPU dynamically per-batch.
    y_attack = y_np
    attack_features = (
        attack_input
        if not issubclass(attack_cls, ProblemSpaceAttack) 
        else None
    )

    output = engine.run(
        x_target,
        y_attack,
        feature_names=feature_names,
        attack_input=attack_features,
        attack_input_is_target_normalized=(
            attack_input_is_target_normalized and not issubclass(attack_cls, ProblemSpaceAttack)
        ),
        clean_physical=clean_physical,
        raw_traffic=raw_traffic,
        valid_mask=valid_mask,
        attack_kwargs={"seed": int(seed)},
    )

    clean_metrics = classification_metrics(
        y_np,
        output.baseline_predictions,
        labels=np.arange(target_num_classes, dtype=np.int64),
    ).as_dict()
    metrics = dict(output.metrics)
    for key, value in clean_metrics.items():
        metrics[f"baseline_{key}"] = float(value)
    metrics["attack_generated_sample_count"] = float(len(y_np))
    metrics["surrogate_used"] = float(surrogate_wrapper is not None)
    if output.detector_clean_scores is not None and output.detector_adversarial_scores is not None:
        detector_scores_dir = Path(cfg.get("output_dir", "../runs")) / "per_seed" / "detector_scores"
        detector_scores_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            detector_scores_dir / f"detector_scores_seed{seed}.npz",
            clean_scores=output.detector_clean_scores,
            adversarial_scores=output.detector_adversarial_scores,
            accepted_mask=output.accepted_mask,
            labels=y_np,
        )
    if surrogate_meta is not None:
        metrics["surrogate_num_classes"] = float(surrogate_meta["num_classes"])

    if ExperimentReporter is not None:
        reporter = ExperimentReporter(
            run_dir=Path(cfg.get("output_dir", "../runs")),
            class_names=class_names,
        )
        y_true_scored = y_np[output.scored_mask]
        y_pred_scored = output.adversarial_predictions[output.scored_mask]
        reporter.save_predictions(
            seed, y_true_scored, output.baseline_predictions[output.scored_mask], y_pred_scored
        )
        reporter.save_seed_report(seed, metrics, y_true_scored, y_pred_scored)
        reporter.generate_confusion_matrix(seed, y_true_scored, y_pred_scored)

    if logger:
        logger.info(
            "evaluate.report_saved seed=%d asr=%.2f%% realizability=%.2f%% "
            "scored=%d rejected=%d",
            seed, 100.0 * metrics.get("asr", 0.0),
            100.0 * metrics.get("realizability_rate", 0.0),
            int(metrics.get("scored_sample_count", 0.0)),
            int(metrics.get("rejected_sample_count", 0.0)),
        )

    result_dir = Path(cfg.get("output_dir", "../runs")) / "per_seed"
    _save_json(
        result_dir / f"evaluate_seed_{seed}.json",
        {
            "seed": int(seed),
            "target_model": target_cfg.get("name"),
            "surrogate_model": None if surrogate_cfg is None else surrogate_cfg.get("name"),
            "metrics": metrics,
        },
    )
    return metrics



def preprocess_seed(*, config: Mapping[str, Any], seed: int, gpu: str, logger: logging.Logger | None = None) -> Mapping[str, float]:
    cfg = _cfg(config)
    configure_process_device(gpu)
    processor_cfg = dict(cfg["preprocessing"])
    processor_cls = PREPROCESSORS.get(str(processor_cfg["name"]))
    processor = processor_cls(**dict(processor_cfg.get("params", {})))
    if logger:
        logger.info(
            "preprocess.start seed=%d processor=%s input=%s output=%s params=%s",
            seed, processor_cfg["name"], processor_cfg["input"], processor_cfg["output"],
            processor_cfg.get("params", {}),
        )
    result = processor.transform(processor_cfg["input"], fit=True)
    output = Path(str(processor_cfg["output"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=result.features,
        feature_names=np.asarray(result.feature_names),
        metadata=json.dumps(result.metadata),
    )
    _save_json(output.with_suffix(output.suffix + ".json"), processor.state_dict() if hasattr(processor, "state_dict") else {})
    if logger:
        logger.info(
            "preprocess.done seed=%d rows=%d features=%d output=%s",
            seed, len(result.features), result.features.shape[-1] if result.features.ndim >= 2 else 0, output,
        )
    return {
        "rows": float(len(result.features)),
        "features": float(result.features.shape[-1] if result.features.ndim >= 2 else 0),
    }



def train_attack_seed(*, config: Mapping[str, Any], seed: int, gpu: str, logger: logging.Logger | None = None) -> Mapping[str, float]:
    """Generic attack-training entrypoint for generator/optimizer plugins."""
    cfg = _cfg(config)
    attack_name = str(cfg["attack"]["name"])
    attack_cls = ATTACKS.get(attack_name)
    train_method = getattr(attack_cls, "train", None)
    if logger:
        logger.info("train_attack.start seed=%d attack=%s", seed, attack_name)
    if train_method is None:
        raise ConfigurationError(
            f"Attack {attack_name!r} is generation-only. Add a concrete train(...) implementation "
            "to the plugin before using action=train_attack."
        )
    result = train_method(config=cfg, seed=seed, gpu=gpu)
    if logger:
        logger.info("train_attack.done seed=%d metrics=%s", seed, sorted(result))
    return {str(key): float(value) for key, value in dict(result).items()}
