from __future__ import annotations

import csv
import gc
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.exceptions import ConfigurationError
from core.dual_representation import encode_normalized32
from core.provenance import create_split_manifest, validate_research_split_manifest, manifest_hash, save_split_manifest, split_indices, source_revision_sha256, frozen_model_state_sha256
from core.registry import ATTACKS, DETECTORS, PREPROCESSORS
from defenses.empirical_at import _make_endpoint_perturbation_mask_fn, _run_empirical_at, _validated_packet_mask
from models.wrapper import PyTorchModelWrapper, SklearnModelWrapper
from orchestrator.device import configure_process_device
from orchestrator.runner import seed_everything
from orchestrator.progress import ProgressCallback
from .factory import _build_model, _cfg, _inject_dynamic_dataset_shape, _instantiate_dataset
from .io import _save_json, _write_sklearn_training_curve
from .seeds import role_seed
from .experiment import ExperimentStore



def train_nids_seed(*, config: Mapping[str, Any], seed: int, gpu: str,
                    logger: logging.Logger | None = None,
                    progress: ProgressCallback | None = None) -> Mapping[str, float]:
    """Train one seed, cleaning uncommitted checkpoint directories on failure."""
    staged_dirs: list[Path] = []
    try:
        return _train_nids_seed(config=config, seed=seed, gpu=gpu,
                                logger=logger, staged_dirs=staged_dirs, progress=progress)
    finally:
        for staged in staged_dirs:
            if staged.exists():
                shutil.rmtree(staged)


def _train_nids_seed(*, config: Mapping[str, Any], seed: int, gpu: str,
                     logger: logging.Logger | None, staged_dirs: list[Path],
                     progress: ProgressCallback | None = None) -> Mapping[str, float]:
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

    if progress is not None:
        progress('load_dataset')  # Opaque I/O: no invented percentage.
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

    training_options = dict(cfg['model'].get('training', {}))
    mask_required = (
        str(training_options.get('defense_mode', 'none') or 'none').strip().lower()
        in {'pgd', 'trades', 'empirical_at'}
        or bool(training_options.get('adversarial_training', False))
    )
    # Admit the original topology before normalization or model construction.
    # Sequence-capable wrappers and flat defenders must share this boundary.
    checked_packet_mask = (
        _validated_packet_mask(getattr(dataset, 'valid_mask', None),
            sample_count=len(y_np), input_shape=tuple(x_raw.shape),
            feature_count=len(dataset.feature_names))
        if mask_required else None
    )

    if progress is not None:
        progress('validate_and_split')
    # One immutable partition shared with evaluation and every attack trainer.
    # No downstream component is allowed to invent its own training membership.
    store = ExperimentStore(cfg.get('experiment_dir'), dataset=cfg['dataset']['name'], create=True)
    model_cfg_for_path = cfg['model']
    final_checkpoint = store.model_path(
        model_cfg_for_path['name'],
        model_cfg_for_path.get('training', {}).get('defense_mode', 'none'), seed,
        role=model_cfg_for_path.get('artifact_role'),
    )
    store.validate_template(model_cfg_for_path.get('checkpoint'), final_checkpoint, seed, role='model')
    # Never write a partial fitted model into a path visible to evaluation.
    checkpoint_dir = store.new_staging_directory(final_checkpoint.parent)
    staged_dirs.append(checkpoint_dir)
    split_seed = role_seed(cfg, "split", seed)
    budget_profile = str(cfg.get('budget', {}).get('name', 'smoke'))
    if budget_profile not in {'research', 'smoke'}:
        raise ConfigurationError('Training budget must be research or smoke')
    split_options = dict(cfg.get('data_split') or {})
    if split_options.get('policy', budget_profile) != budget_profile:
        raise ConfigurationError('data_split.policy must agree with budget.name')
    split_options['policy'] = budget_profile
    manifest = create_split_manifest(dataset, y_np, split_seed, split_options)
    if budget_profile == 'research':
        validate_research_split_manifest(manifest, y_np, dataset)
    store.validate_partition(manifest)
    train_idx = split_indices(manifest, dataset, "train")
    val_idx = split_indices(manifest, dataset, "validation")
    split_file = checkpoint_dir / "split_manifest.json"
    save_split_manifest(split_file, manifest)
    split_digest = manifest_hash(manifest)
    if logger:
        logger.info(
            "train_nids.split seed=%d train=%d validation=%d",
            seed, len(train_idx), len(val_idx),
        )
    if progress is not None:
        progress('normalize_training_data')
    if hasattr(dataset, "valid_mask"):
        normalization = dataset.fit_normalization(
            x_raw[train_idx],
            valid_mask=np.asarray(dataset.valid_mask if checked_packet_mask is None
                                  else checked_packet_mask.cpu().numpy())[train_idx],
        )
    else:
        normalization = dataset.fit_normalization(x_raw[train_idx])
    # Canonical float64 arithmetic with the exact float32 statistics that
    # will be serialized/loaded for target inference. Never round-trip from
    # normalized float32 to fabricate physical reference data.
    x_norm = encode_normalized32(
        np.asarray(x_raw), tuple(dataset.feature_names),
        np.asarray(normalization.mean, dtype=np.float32),
        np.asarray(normalization.std, dtype=np.float32),
    )
    x_tensor = torch.as_tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_np, dtype=torch.long)
    del x_norm

    if progress is not None:
        progress('setup_model_and_batches')
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
    packet_mask_for_model = (
        (checked_packet_mask if checked_packet_mask is not None
         else torch.as_tensor(dataset.valid_mask, dtype=torch.bool))
        if isinstance(wrapper, PyTorchModelWrapper) and getattr(wrapper, "_supports_valid_mask", False)
        else None
    )
    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx], packet_mask_for_model[train_idx]) if packet_mask_for_model is not None else TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    val_ds = TensorDataset(x_tensor[val_idx], y_tensor[val_idx], packet_mask_for_model[val_idx]) if packet_mask_for_model is not None else TensorDataset(x_tensor[val_idx], y_tensor[val_idx])

    configured_mode = str(tcfg.get("defense_mode", "none") or "none").strip().lower()
    robust_training = (
        wrapper.capabilities.model_family == "pytorch"
        and (configured_mode in {"pgd", "trades"} or bool(tcfg.get("adversarial_training", False)))
    )
    train_defense_ds = train_ds
    if robust_training and packet_mask_for_model is None:
        assert checked_packet_mask is not None
        train_defense_ds = TensorDataset(
            train_ds.tensors[0], train_ds.tensors[1], checked_packet_mask[train_idx]
        )

    train_loader = cast(
        DataLoader[Any],
        DataLoader(train_defense_ds, batch_size=batch_size,
                   shuffle=wrapper.capabilities.model_family == "pytorch",
                   generator=torch.Generator().manual_seed(int(seed)) if wrapper.capabilities.model_family == "pytorch" else None,
                   num_workers=0, pin_memory=pin),
    )
    val_loader = cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin),
    )

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
    actual_strategy = wrapper.capabilities.model_family
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
    detector_train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin
    )
    if logger:
        logger.info(
            "train_nids.strategy seed=%d requested=%s actual=%s defense=%s detector=%s",
            seed, requested_strategy or "auto", actual_strategy, defense_mode,
            detector_cfg.get("name", "none") if detector_enabled else "none",
        )

    eat_stats: dict[str, float] = {}
    if defense_mode == "empirical_at" and wrapper.capabilities.model_family == "sklearn":
        eat_cfg = dict(tcfg.get("empirical_at", {}) or {})
        if not eat_cfg:
            raise ConfigurationError(
                "defense_mode=empirical_at requires model.training.empirical_at settings; "
                "compose configs/model/training/empirical_at.yaml."
            )
        adversarial_x, adversarial_y, eat_stats = _run_empirical_at(
            eat_cfg=eat_cfg, x_tensor=x_tensor, y_np=y_np, train_idx=train_idx, val_idx=val_idx,
            valid_mask_full=checked_packet_mask.cpu().numpy(),
            feature_names=tuple(dataset.feature_names), normalization=normalization,
            num_classes=num_classes, device=device_info.torch_device, seed=seed,
            checkpoint_dir=checkpoint_dir, logger=logger, x_physical=x_raw,
            sample_ids=dataset.sample_ids, split_manifest=manifest,
            class_mapping=dataset.class_mapping,
            seed_roles={'split': int(split_seed), 'target': int(seed),
                        'surrogate': role_seed(cfg, 'surrogate', seed),
                        'attack': role_seed(cfg, 'attack', seed)},
            padding_values={**getattr(dataset, "pad_values", {}), **(
                {str(getattr(dataset, "padding_indicator_name")): 1.0}
                if getattr(dataset, "include_padding_indicator", False) else {}
            )} if hasattr(dataset, "pad_values") else None,
            padding_indicator_name=(str(dataset.padding_indicator_name)
                                    if getattr(dataset, "include_padding_indicator", False) else None),
        )
        augmented_train_x = np.vstack((x_tensor[train_idx].cpu().numpy(), adversarial_x))
        augmented_train_y = np.concatenate((y_np[train_idx], adversarial_y), axis=0)
        eat_stats["eat_augmented_samples"] = float(len(adversarial_y))
        eat_stats["eat_training_samples_total"] = float(len(augmented_train_y))
        train_loader = DataLoader(
            TensorDataset(torch.as_tensor(augmented_train_x, dtype=torch.float32),
                          torch.as_tensor(augmented_train_y, dtype=torch.long)),
            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )
        detector_train_loader = DataLoader(
            train_ds, batch_size=min(batch_size, 256), shuffle=False, num_workers=0, pin_memory=pin,
        )
    elif defense_mode == "empirical_at":
        raise ConfigurationError(
            "defense_mode=empirical_at is specifically for sklearn targets; PyTorch targets should use "
            "defense_mode=pgd or defense_mode=trades."
        )

    fit_config = dict(tcfg)
    fit_config["defense_mode"] = defense_mode
    fit_config["seed"] = seed
    fit_config["expected_classes"] = num_classes
    endpoint_mask_fn = (
        _make_endpoint_perturbation_mask_fn(
            tuple(dataset.feature_names), normalization,
            immutable_features=tcfg.get("immutable_features"),
        )
        if robust_training else None
    )
    # Only the PyTorch concrete wrapper supports minibatch callbacks. sklearn
    # estimators expose an opaque fit call; represent it as an active stage.
    fit_kwargs = {'logger': logger, 'output_dir': checkpoint_dir,
                  'seed': seed, 'perturbation_mask_fn': endpoint_mask_fn}
    if isinstance(wrapper, PyTorchModelWrapper):
        fit_kwargs['progress'] = progress
    elif progress is not None:
        progress('sklearn_fit')
    fit_result = wrapper.fit(train_loader, val_loader, fit_config, **fit_kwargs)
    train_loss = float(fit_result["train_loss"])
    validation_loss = float(fit_result["validation_loss"])
    defense_mode = str(fit_result["defense_mode"])

    # Fingerprint the finished classifier BEFORE detector fitting and before nids.pt
    # serialization. This avoids a circular checkpoint hash.
    final_model_digest = frozen_model_state_sha256(wrapper)

    # Fit inference detectors after the target has finished fitting. The wrapper's
    # polymorphic extract_features contract supports PyTorch and sklearn targets.
    if progress is not None:
        progress('fit_detector' if detector_enabled else 'finalize_model')
    detector = None
    detector_name: str | None = None
    detector_params: dict[str, Any] = {}
    # Lee calibration is a separately registered action after the independently
    # pretrained surrogate exists. Never publish a fit-only unusable detector.
    if detector_enabled and detector_cfg.get('name') != 'mahalanobis_multilayer_val_calibrated':
        if not callable(getattr(wrapper, "extract_features", None)):
            raise ConfigurationError(
                f"Model wrapper {type(wrapper).__name__} does not implement extract_features required by the detector"
            )
        detector_name = str(detector_cfg.get("name", "mahalanobis"))
        detector_params = dict(detector_cfg.get("params", {}))
        detector_cls = DETECTORS.get(detector_name)
        detector = detector_cls(**detector_params)
        detector.fit(detector_train_loader, wrapper)
        if detector_cfg.get('checkpoint'):
            raise ConfigurationError('Explicit external detector checkpoints are not supported in experiment_dir')
        detector_path = checkpoint_dir / "detector.pt"
        detector_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": detector_name,
                "params": detector_params,
                "state_dict": detector.state_dict(),
                "seed": int(seed),
                "fit_split": "train",
                "classifier_defense_mode": defense_mode,
                "split_manifest_sha256": split_digest,
                "target_model_state_sha256": final_model_digest,
            },
            detector_path,
        )
        if logger:
            logger.info(
                "Fitted %s detector on %d training samples after %s training; saved to %s",
                detector_name, len(detector_train_loader.dataset), defense_mode, detector_path,
            )

    if progress is not None:
        progress('publish_checkpoint')
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
        "model_state_sha256": final_model_digest,
        "split_manifest_sha256": split_digest,
        "dataset_identity": manifest["identity"],
        "dataset_transform": manifest["identity"].get("transformation"),
        "split_manifest": "split_manifest.json",
        "benchmark_protocol_version": 2,
        "training_budget_profile": budget_profile,
        "experiment_id": store.read()['experiment_id'],
        "source_revision_sha256": source_revision_sha256(),
        "seed_roles": {"split": int(split_seed), "model": int(seed)},
    }
    torch.save(checkpoint_payload, checkpoint)
    # Compact sidecar lets offline generators validate checkpoint-to-split
    # membership without deserializing a potentially multi-GB estimator.
    _save_json(checkpoint_dir / "training_provenance.json", {
        "seed": int(seed),
        "benchmark_protocol_version": 2,
        "training_budget_profile": budget_profile,
        "experiment_id": store.read()['experiment_id'],
        "source_revision_sha256": source_revision_sha256(),
        "split_manifest_sha256": split_digest,
        "model_state_sha256": final_model_digest,
        "class_mapping": dict(getattr(dataset, "class_mapping", {})),
        "feature_names": list(dataset.feature_names),
        "input_shape": list(x_tensor.shape[1:]),
        "dataset_identity": manifest["identity"],
        "sequence_policy": manifest["identity"].get("transformation", {}).get("sequence_policy") if manifest["identity"].get("transformation") else None,
        "checkpoint_path": store.relative(final_checkpoint),
        "classes": list(classes),
        "seed_roles": {"split": int(split_seed), "model": int(seed), "surrogate": None, "attack": None, "detector": int(seed) if detector_enabled else None},
    })
    store.publish_model(
        checkpoint_dir, final_checkpoint, manifest,
        state_sha256=final_model_digest, seed=seed, split_hash=split_digest,
    )
    if progress is not None:
        progress('checkpoint_published', status='completed')
    if logger:
        logger.info(
            "train_nids.checkpoint_saved seed=%d path=%s family=%s defense=%s train_loss=%.5f val_loss=%.5f",
            seed, final_checkpoint, checkpoint_payload["model_family"], defense_mode,
            train_loss, validation_loss,
        )
    result = {
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "defense_mode_pgd": float(defense_mode == "pgd"),
        "defense_mode_trades": float(defense_mode == "trades"),
        "detector_fitted": float(detector is not None),
        "model_is_sklearn": float(isinstance(wrapper, SklearnModelWrapper)),
        "dataset_total_flow_count": float(len(dataset)),
        "dataset_truncated_flow_count": float(getattr(dataset, "truncated_flow_count", 0)),
        "dataset_discarded_packet_positions": float(getattr(dataset, "discarded_packet_positions", 0)),
    }
    if detector is not None:
        result["detector_fit_samples"] = float(len(detector_train_loader.dataset))
    result.update(eat_stats)
    result["defense_mode_empirical_at"] = float(defense_mode == "empirical_at")
    return result



def preprocess_seed(*, config: Mapping[str, Any], seed: int, gpu: str,
                    logger: logging.Logger | None = None,
                    progress: ProgressCallback | None = None) -> Mapping[str, float]:
    cfg = _cfg(config)
    configure_process_device(gpu)
    if progress is not None:
        progress('preprocess_active')  # Deferred plugin internals are not metered.
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


def train_attack_seed(*, config: Mapping[str, Any], seed: int, gpu: str,
                      logger: logging.Logger | None = None,
                      progress: ProgressCallback | None = None) -> Mapping[str, float]:
    """Generic attack-training entrypoint for generator/optimizer plugins."""
    if progress is not None:
        progress('generator_preflight')
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
    # Source classifier seed and generator-noise/checkpoint seed are distinct.
    # Resolve only the authorized source checkpoint before invoking the existing
    # plugin trainer. Every generator still writes its own attack-seed artifact;
    # provenance reads the source's *actual* seed and immutable split IDs.
    store = ExperimentStore(cfg.get('experiment_dir'), dataset=cfg['dataset']['name'])
    source_role = ('surrogate_model' if isinstance(cfg.get('surrogate_model'), Mapping)
                   and cfg['surrogate_model'].get('enabled', True) else 'model')
    if source_role != 'surrogate_model':
        raise ConfigurationError('Offline generator must use an independently authorized surrogate; target fallback prohibited')
    source_cfg = cfg.get(source_role)
    if not isinstance(source_cfg, Mapping) or not source_cfg.get('checkpoint'):
        raise ConfigurationError('Offline generator requires surrogate_model.checkpoint')
    source_seed = role_seed(cfg, 'surrogate', seed)
    source_path = store.model_path(source_cfg['name'], 'none', source_seed, role='surrogates')
    store.validate_template(source_cfg['checkpoint'], source_path, source_seed, role=source_role)
    store.require('model', source_path, seed=source_seed)
    cfg[source_role] = {**dict(source_cfg), 'checkpoint': str(source_path)}
    # The generator's stable path is shared by training and evaluation.
    attack_cfg = cfg['attack']
    variant = str(attack_cfg.get('variant', 'default'))
    generator_path = store.generator_path(attack_name, variant, seed)
    configured = attack_cfg.get('checkpoint', attack_cfg.get('params', {}).get('checkpoint'))
    store.validate_template(configured, generator_path, seed, role='attack')
    with store.claim_generator(generator_path):
        if progress is not None:
            progress('generator_training')  # Plugins have no common metered loop.
        result = train_method(config=cfg, seed=seed, gpu=gpu)
        if progress is not None:
            progress('publish_generator')
        store.publish_generator(generator_path, seed=seed, configuration=attack_cfg)
    if progress is not None:
        progress('generator_published', status='completed')
    if logger:
        logger.info("train_attack.done seed=%d metrics=%s", seed, sorted(result))
    return {str(key): float(value) for key, value in dict(result).items()}
