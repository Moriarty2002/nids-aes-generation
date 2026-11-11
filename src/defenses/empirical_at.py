from __future__ import annotations

from contextlib import nullcontext

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from attacks.base import FeatureSpaceAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from core.exceptions import ConfigurationError
from models.trainer import train_classifier
from models.wrapper import PyTorchModelWrapper
from workflow.factory import _build_model, _instantiate_attack, _model_requires_flatten

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
        [str(name).upper() for name in (
            immutable_features if immutable_features is not None
            else (name for name in ("DIR", "FLG") if name in names)
        )] + (["FEAT_PAD_MASK"] if "FEAT_PAD_MASK" in names else [])
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
    padding_values: Mapping[str, float] | None = None,
    padding_indicator_name: str | None = None,
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
    surrogate_val_y = torch.as_tensor(y_np[val_idx], dtype=torch.long)
    surrogate_val_ds = (
        TensorDataset(surrogate_val_x, surrogate_val_y, surrogate_packet_mask[val_idx])
        if valid_mask_full is not None and surrogate_wrapper._supports_valid_mask
        else TensorDataset(surrogate_val_x, surrogate_val_y)
    )
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
        {}, feature_names,
        target_label=None if str(attack_cfg.get("name", "")).lower() == "trades" else benign_label,
        attack_cfg_override=attack_cfg,
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
        padding_values=padding_values,
        padding_indicator_name=padding_indicator_name,
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
            
        # for feature space attacks that works on raw physical features, we need to pass the physical features to the attack
        physical_feature_attack = bool(getattr(attack, "operates_on_physical_features", False))
        attack_generation_input = (
            np.array(
                x_attack_phys,
                dtype=np.float64,
                copy=True,
                order="C",
            )
            if physical_feature_attack
            else x_attack
        )    
        
        with surrogate_wrapper.attack_mask_context(batch_valid_mask):
            result = attack.generate(attack_generation_input, y_clean, **batch_kwargs)
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
            attack_budget=attack.final_budget_contract(),
            budget_normalization_mean=np.asarray(normalization.mean, dtype=np.float32),
            budget_normalization_std=np.asarray(normalization.std, dtype=np.float32),
        )
        accepted = np.asarray(gate.accepted, dtype=bool).copy()
        if result.intrinsic_feasible_mask is not None:
            intrinsic = np.asarray(result.intrinsic_feasible_mask)
            if intrinsic.dtype != np.bool_ or intrinsic.shape != accepted.shape:
                raise ConfigurationError("EAT intrinsic_feasible_mask must have boolean (B,) shape")
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
            logger.debug(
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
