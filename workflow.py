from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from attacks.base import BaseAttack, FeatureSpaceAttack, ProblemSpaceAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from core.exceptions import ConfigurationError
from core.registry import ATTACKS, DATASETS, DETECTORS, NIDS, PREPROCESSORS
from evaluator.engine import EvaluationEngine, FeatureNormalization
from evaluator.logger import ExperimentReporter
from evaluator.metrics import classification_metrics
from models.trainer import train_classifier
from models.wrapper import PyTorchModelWrapper
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
    "random_forest", "rf", "xgboost", "xgb", "svm",
})
_SEQUENCE_MODEL_NAMES = frozenset({
    "cnn", "lstm", "bilstm", "rnn", "gru", "transformer",
})


def _model_requires_flatten(model_cfg: Mapping[str, Any]) -> bool:
    """Return whether the registered model expects dataset-level flattening.

    Known architectures are convention-driven. Custom registry plugins can set
    ``model.requires_flatten`` explicitly, avoiding any dataset YAML toggle.
    """
    explicit = model_cfg.get("requires_flatten")
    if explicit is not None:
        return bool(explicit)
    name = str(model_cfg.get("name", "")).strip().lower().replace("-", "_")
    if name in _FLATTENING_MODEL_NAMES:
        return True
    if name in _SEQUENCE_MODEL_NAMES:
        return False
    raise ConfigurationError(
        f"Cannot infer tensor shaping for model {name!r}. "
        "Set model.requires_flatten explicitly for a custom NIDS plugin."
    )


def _inject_dynamic_dataset_shape(cfg: dict[str, Any]) -> None:
    """Derive the dataset flatten mode from the selected target architecture.

    The override is applied only when the dataset constructor supports a
    ``flatten`` parameter, preserving all existing flat/tabular adapters.
    Target and surrogate shapes must agree when a sequence dataset is used;
    otherwise the same generated tensor cannot faithfully represent both
    attack and target model input contracts.
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

    surrogate_cfg = cfg.get("surrogate_model")
    if isinstance(surrogate_cfg, Mapping) and bool(surrogate_cfg.get("enabled", False)):
        surrogate_flatten = _model_requires_flatten(surrogate_cfg)
        if supports_flatten and surrogate_flatten != target_flatten:
            raise ConfigurationError(
                "Target and surrogate models require incompatible dataset shapes: "
                f"target flatten={target_flatten}, surrogate flatten={surrogate_flatten}. "
                "Use compatible sequence/flat model families or provide an explicit adapter."
            )

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


def _model_input_dim(model_name: str, input_shape: tuple[int, ...]) -> tuple[int, dict[str, Any]]:
    if len(input_shape) == 2:
        return int(input_shape[-1]), {}
    if len(input_shape) != 3:
        raise ConfigurationError(f"Unsupported NIDS input shape {input_shape}; expected 2-D or 3-D tensors")

    packets, features = int(input_shape[1]), int(input_shape[2])
    params: dict[str, Any] = {}
    if model_name == "mlp":
        # MLP consumes a sequence as a flattened vector.
        return packets * features, params
    if model_name in {"cnn", "transformer"}:
        params["sequence_features"] = features
    # LSTM/CNN/Transformer consume the packet feature dimension directly.
    return features, params


def _build_model(
    model_cfg: Mapping[str, Any],
    input_shape: tuple[int, ...],
    *,
    device: torch.device,
    num_classes: int,
) -> PyTorchModelWrapper:
    model_name = str(model_cfg["name"])
    arch = NIDS.get(model_name)
    params = dict(model_cfg.get("params", {}))
    input_dim, shape_params = _model_input_dim(model_name, input_shape)
    for key, value in shape_params.items():
        params.setdefault(key, value)
    model = arch(input_dim=input_dim, num_classes=int(num_classes), **params)
    return PyTorchModelWrapper(model, device)


def _load_checkpoint(wrapper: PyTorchModelWrapper, checkpoint: str | Path) -> dict[str, Any]:
    """Load model weights on the worker's device and switch to inference mode."""
    data = torch.load(str(checkpoint), map_location=wrapper.device, weights_only=False)
    wrapper.model.load_state_dict(data["model_state"], strict=True)
    wrapper.model.eval()
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
        # Tiny/rare-class datasets may be impossible to stratify. Keep the split deterministic.
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        split = max(1, min(len(indices) - 1, int(round(0.8 * len(indices)))))
        return indices[:split], indices[split:]


def train_nids_seed(*, config: Mapping[str, Any], seed: int, gpu: str, logger: logging.Logger | None = None) -> Mapping[str, float]:
    """Train one reproducible classifier seed and persist its artifacts.

    Each process owns one seed, model, and output checkpoint. Keeping this
    boundary seed-local makes multiprocessing failures easier to isolate and
    prevents workers from sharing mutable training state.
    """
    cfg = _cfg(config)
    device_info = configure_process_device(gpu)
    seed_everything(seed)
    _inject_dynamic_dataset_shape(cfg)

    dataset = _instantiate_dataset(cfg)
    x_raw, y_np = dataset.raw_arrays()
    if len(y_np) < 10:
        raise ConfigurationError("Training requires at least 10 samples")

    train_idx, val_idx = _split_indices(y_np, seed)
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
    # The dataset has already been materialized into tensors. Release the NumPy
    # feature buffers before model construction to reduce the worker peak.
    del x_raw, x_norm

    model_cfg = dict(cfg["model"])
    num_classes = int(len(getattr(dataset, "classes", tuple(np.unique(y_np)))))
    wrapper = _build_model(
        model_cfg,
        (len(x_tensor), *tuple(x_tensor.shape[1:])),
        device=device_info.torch_device,
        num_classes=num_classes,
    )

    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    val_ds = TensorDataset(x_tensor[val_idx], y_tensor[val_idx])
    tcfg = dict(model_cfg.get("training", {}))
    batch_size = int(tcfg.get("batch_size", 256))
    pin = device_info.torch_device.type == "cuda"
    train_loader = cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin),
    )
    val_loader = cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin),
    )

    checkpoint_dir = Path(cfg.get("output_dir", "runs")) / "checkpoints" / f"{model_cfg['name']}_seed{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pgd_epsilon = float(tcfg.get("pgd_epsilon", 0.04))
    configured_alpha = tcfg.get("pgd_alpha")
    pgd_alpha = pgd_epsilon / 4.0 if configured_alpha is None else float(configured_alpha)
    summary = train_classifier(
        wrapper,
        train_loader,
        val_loader,
        epochs=int(tcfg.get("epochs", 20)),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        adversarial_training=bool(tcfg.get("adversarial_training", False)),
        defense_mode=(str(tcfg["defense_mode"]) if tcfg.get("defense_mode") is not None else None),
        pgd_epsilon=pgd_epsilon,
        pgd_alpha=pgd_alpha,
        pgd_steps=int(tcfg.get("pgd_steps", 10)),
        trades_beta=float(tcfg.get("trades_beta", 6.0)),
        output_dir=checkpoint_dir,
        seed=seed,
    )

    # Inference-time detector fitting happens only after the classifier has finished
    # training (including PGD/TRADES). Fit strictly on this seed's training split.
    detector_cfg = dict(cfg.get("detector", {}) or {})
    detector = None
    detector_name: str | None = None
    detector_params: dict[str, Any] = {}
    detector_state: dict[str, Any] | None = None
    if bool(detector_cfg.get("enabled", False)):
        detector_name = str(detector_cfg.get("name", "mahalanobis"))
        detector_params = dict(detector_cfg.get("params", {}))
        detector_cls = DETECTORS.get(detector_name)
        detector = detector_cls(**detector_params)
        detector.fit(train_loader, wrapper)
        detector_state = detector.state_dict()
        detector_path = checkpoint_dir / "detector.pt"
        torch.save(
            {
                "name": detector_name,
                "params": detector_params,
                "state_dict": detector_state,
                "seed": int(seed),
                "fit_split": "train",
                "classifier_defense_mode": summary.defense_mode,
            },
            detector_path,
        )
        if logger:
            logger.info(
                "Fitted %s detector on %d training samples after %s training; saved to %s",
                detector_name,
                len(train_ds),
                summary.defense_mode,
                detector_path,
            )

    checkpoint = checkpoint_dir / "nids.pt"
    checkpoint_payload = {
        "model_state": wrapper.model.state_dict(),
        "model_name": model_cfg["name"],
        "model_params": model_cfg.get("params", {}),
        "num_classes": num_classes,
        "classes": list(getattr(dataset, "classes", tuple(str(v) for v in sorted(np.unique(y_np).tolist())))),
        "class_mapping": dict(getattr(dataset, "class_mapping", {})),
        "feature_names": list(dataset.feature_names),
        "input_shape": list(x_tensor.shape[1:]),
        "normalization_mean": normalization.mean,
        "normalization_std": normalization.std,
        "normalization_version": normalization.version,
        "defense_mode": summary.defense_mode,
        "detector": None if detector is None else {
            "name": detector_name,
            "params": detector_params,
            "checkpoint": "detector.pt",
            "fit_split": "train",
        },
        "seed": int(seed),
    }
    torch.save(checkpoint_payload, checkpoint)
    result = {
        "train_loss": float(summary.train_loss),
        "validation_loss": float(summary.validation_loss or 0.0),
        "defense_mode_pgd": float(summary.defense_mode == "pgd"),
        "defense_mode_trades": float(summary.defense_mode == "trades"),
        "detector_fitted": float(detector is not None),
    }
    if detector is not None:
        result["detector_fit_samples"] = float(len(train_ds))
    return result


def _instantiate_attack(
    cfg: Mapping[str, Any],
    feature_names: tuple[str, ...],
    *,
    target_label: int | None = None,
) -> tuple[type[BaseAttack], dict[str, Any]]:
    attack_cfg = dict(cfg["attack"])
    attack_name = str(attack_cfg["name"])
    attack_cls = ATTACKS.get(attack_name)
    params = {key: value for key, value in attack_cfg.items() if key not in {"name", "params", "variant", "training"}}
    params.update(dict(attack_cfg.get("params", {})))
    if issubclass(attack_cls, FeatureSpaceAttack):
        params.setdefault("feature_names", feature_names)

    if target_label is not None:
        signature = inspect.signature(attack_cls)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if "target_label" in signature.parameters or accepts_kwargs:
            params["target_label"] = int(target_label)
        if "targeted" in signature.parameters:
            params["targeted"] = True
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
    feature_names = tuple(dataset.feature_names)

    # -------------------- target model --------------------
    target_cfg = dict(cfg["model"])
    target_ckpt = _checkpoint_path(target_cfg, seed, role="model")
    target_meta = _load_checkpoint_metadata(target_ckpt)
    target_norm = _normalization_from_checkpoint(target_meta, feature_names, role="target model")
    x_target = target_norm.normalize(x_raw).astype(np.float32)
    target_num_classes = int(target_meta["num_classes"])
    target_wrapper = _build_model(
        target_cfg,
        tuple(x_target.shape),
        device=device_info.torch_device,
        num_classes=target_num_classes,
    )
    _load_checkpoint(target_wrapper, target_ckpt)

    # Optional inference-time detector fitted after the target model's training-time
    # defense. Detector state lives adjacent to the classifier checkpoint.
    detector_cfg = dict(cfg.get("detector", {}) or {})
    detector = None
    if bool(detector_cfg.get("enabled", False)):
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
    surrogate_wrapper: PyTorchModelWrapper | None = None
    attack_input: Any = x_target
    attack_norm = target_norm
    surrogate_meta: dict[str, Any] | None = None
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
        x_surrogate = surrogate_norm.normalize(x_raw).astype(np.float32)
        surrogate_wrapper = _build_model(
            surrogate_cfg,
            tuple(x_surrogate.shape),
            device=device_info.torch_device,
            num_classes=int(surrogate_meta["num_classes"]),
        )
        _load_checkpoint(surrogate_wrapper, surrogate_ckpt)
        attack_input = x_surrogate
        attack_norm = surrogate_norm

    evaluation_cfg = dict(cfg.get("evaluation", {}))
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
        benign_label=benign_label,
        target_label=attack_params.get("target_label"),
        c_fp=float(evaluation_cfg.get("c_fp", 1.0)),
        c_fn=float(evaluation_cfg.get("c_fn", 1.0)),
        strict_constraints=strict_constraints,
        strict_multiclass=strict_multiclass,
        detector=detector,
        batch_size=int(evaluation_cfg.get("batch_size", 256)),
    )

    valid_mask = getattr(dataset, "valid_mask", None)
    if isinstance(attack_model, PyTorchModelWrapper):
        y_attack: Any = torch.as_tensor(y_np, dtype=torch.long, device=attack_model.device)
        attack_features: Any = (
            torch.as_tensor(attack_input, dtype=torch.float32, device=attack_model.device)
            if not issubclass(attack_cls, ProblemSpaceAttack)
            else None
        )
    else:
        y_attack = y_np
        attack_features = np.asarray(attack_input, dtype=np.float32) if not issubclass(attack_cls, ProblemSpaceAttack) else None

    output = engine.run(
        x_target,
        y_attack,
        feature_names=feature_names,
        attack_input=attack_features,
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
        detector_scores_dir = Path(cfg.get("output_dir", "runs")) / "per_seed" / "detector_scores"
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

    reporter = ExperimentReporter(
        run_dir=Path(cfg.get("output_dir", "runs")),
        class_names=[
            str(name) for name in getattr(
                dataset, "classes", tuple(f"Class_{i}" for i in range(target_num_classes))
            )
        ],
    )
    y_true_scored = y_np[output.scored_mask]
    y_pred_scored = output.adversarial_predictions[output.scored_mask]
    reporter.save_predictions(
        seed, y_true_scored, output.baseline_predictions[output.scored_mask], y_pred_scored
    )
    reporter.save_seed_report(seed, metrics, y_true_scored, y_pred_scored)
    reporter.generate_confusion_matrix(seed, y_true_scored, y_pred_scored)

    if logger:
        logger.info(f"Evaluation completed for seed {seed}. ASR: {metrics.get('asr', 0.0):.2%}, RR: {metrics.get('realizability_rate', 0.0):.2%}")

    result_dir = Path(cfg.get("output_dir", "runs")) / "per_seed"
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
    if train_method is None:
        raise ConfigurationError(
            f"Attack {attack_name!r} is generation-only. Add a concrete train(...) implementation "
            "to the plugin before using action=train_attack."
        )
    result = train_method(config=cfg, seed=seed, gpu=gpu)
    return {str(key): float(value) for key, value in dict(result).items()}
