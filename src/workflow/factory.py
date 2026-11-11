from __future__ import annotations

import inspect
from typing import Any, Mapping

import torch
from torch import nn
from omegaconf import OmegaConf

from attacks.base import BaseAttack, FeatureSpaceAttack
from core.exceptions import ConfigurationError
from core.registry import ATTACKS, DATASETS, NIDS
from models.wrapper import ModelWrapper, PyTorchModelWrapper, SklearnModelWrapper

_FLATTENING_MODEL_NAMES = frozenset({
    "mlp", "ffn", "dnn", "linear", "logistic_regression",
    "random_forest", "rf", "randomforest", "xgboost", "xgb", "svm", "jmila_surrogate",
})
_SEQUENCE_MODEL_NAMES = frozenset({
    "cnn", "lstm", "bilstm", "rnn", "gru", "transformer",
})

def _cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve Hydra interpolation before worker code mutates configuration."""
    if OmegaConf.is_config(config):
        resolved = OmegaConf.to_container(config, resolve=True)
        if not isinstance(resolved, dict):
            raise ConfigurationError("Workflow configuration must resolve to a mapping")
        return dict(resolved)
    return dict(config)

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
        # An optional appended occupancy channel is structural, not an
        # attacker-controlled protocol measurement. Freeze custom names just
        # as the standard FEAT_PAD_MASK is frozen by the base finalizer.
        dataset_params = dict(cfg.get("dataset", {}).get("params", {}))
        if bool(dataset_params.get("include_padding_indicator", False)):
            indicator = str(dataset_params.get("padding_indicator_name", "FEAT_PAD_MASK")).upper()
            if indicator not in {str(name).upper() for name in feature_names}:
                raise ConfigurationError("Configured padding indicator is missing from dataset feature_names")
            params["immutable_features"] = tuple(dict.fromkeys(
                [str(name).upper() for name in params.get("immutable_features", ())] + [indicator]
            ))

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
