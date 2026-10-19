"""Shared, differentiable helpers for L-infinity feature-space attacks.

These helpers intentionally do not import or invoke the universal constraint
Gatekeeper. Protocol/physical realizability is evaluated after generation by
the EvaluationEngine.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor


def immutable_feature_mask(
    x: Tensor,
    feature_names: Sequence[str],
    immutable_features: Sequence[str],
) -> Tensor:
    """Create a batch-shaped mask that zeros configured immutable features.

    For flattened packet sequences, a feature's mask is repeated for each
    packet.  Unknown feature names fail fast to avoid silently allowing edits
    because of a configuration typo.
    """
    mask = torch.ones_like(x)
    if not immutable_features:
        return mask
    if not feature_names:
        raise ValueError("immutable_features requires a non-empty feature_names schema")

    names = tuple(str(name) for name in feature_names)
    unknown = sorted(set(immutable_features).difference(names))
    if unknown:
        raise ValueError(f"Unknown immutable feature name(s): {unknown}; available: {names}")

    feature_count = len(names)
    editable = torch.ones(feature_count, device=x.device, dtype=x.dtype)
    for name in immutable_features:
        editable[names.index(str(name))] = 0.0

    if x.ndim == 3:
        if x.shape[-1] != feature_count:
            raise ValueError(
                f"Sequence tensor has {x.shape[-1]} features, but feature_names has {feature_count}"
            )
        return editable.view(1, 1, feature_count).expand_as(x)

    if x.ndim == 2:
        if x.shape[-1] % feature_count != 0:
            raise ValueError(
                f"Flattened width {x.shape[-1]} is not divisible by {feature_count} features"
            )
        packet_count = x.shape[-1] // feature_count
        return editable.repeat(packet_count).view(1, -1).expand_as(x)

    raise ValueError(f"Expected a 2-D or 3-D batch tensor, got shape {tuple(x.shape)}")


def _bound_tensor(
    value: float | Sequence[float] | np.ndarray | Tensor | None,
    x: Tensor,
    feature_names: Sequence[str],
    *,
    default: float,
) -> float | Tensor:
    """Convert optional scalar/per-feature bounds to a broadcastable shape."""
    if value is None:
        return default

    bound = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    if bound.ndim == 0:
        return bound

    feature_count = len(feature_names)
    if x.ndim == 3:
        packets, features = int(x.shape[1]), int(x.shape[2])
        if bound.ndim == 1 and bound.numel() == features:
            return bound.view(1, 1, features)
        if bound.numel() == packets * features:
            return bound.reshape(1, packets, features)
    else:
        width = int(x.shape[-1])
        if bound.ndim == 1 and bound.numel() == width:
            return bound.view(1, width)
        if feature_count and bound.ndim == 1 and bound.numel() == feature_count and width % feature_count == 0:
            return bound.repeat(width // feature_count).view(1, width)

    try:
        torch.broadcast_shapes(tuple(bound.shape), tuple(x.shape))
    except RuntimeError as exc:
        raise ValueError(
            f"Bound shape {tuple(bound.shape)} cannot broadcast to input shape {tuple(x.shape)}"
        ) from exc
    return bound


def project_linf_delta(
    x0: Tensor,
    delta: Tensor,
    mask: Tensor,
    *,
    epsilon: float,
    clip_min: float | Sequence[float] | np.ndarray | Tensor | None,
    clip_max: float | Sequence[float] | np.ndarray | Tensor | None,
    feature_names: Sequence[str],
) -> Tensor:
    """Project a delta to the L-infinity budget, box bounds, and edit mask.

    As is standard for bounded evasion attacks, this assumes clean features are
    already in the configured box domain. It also re-applies ``mask`` after box
    clipping so clipping can never accidentally modify a victim-owned feature.
    """
    delta = delta.clamp(min=-float(epsilon), max=float(epsilon)) * mask
    lower = _bound_tensor(clip_min, x0, feature_names, default=-float("inf"))
    upper = _bound_tensor(clip_max, x0, feature_names, default=float("inf"))
    candidate = torch.clamp(x0 + delta, min=lower, max=upper)
    projected = (candidate - x0) * mask
    # Explicitly retain both properties after projection, including at each BIM step.
    return projected.clamp(min=-float(epsilon), max=float(epsilon)) * mask


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Mean cross entropy, with support for both C-class and one-logit binary NIDS."""
    if logits.ndim == 1 or (logits.ndim >= 2 and logits.shape[-1] == 1):
        binary_labels = labels.float().reshape(-1)
        if torch.any((binary_labels < 0) | (binary_labels > 1)):
            raise ValueError("One-logit binary models require labels in {0, 1}")
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1), binary_labels, reduction="mean"
        )
    if logits.ndim != 2:
        raise ValueError(
            f"Expected NIDS logits with shape (batch, classes) or one logit per row, got {tuple(logits.shape)}"
        )
    return torch.nn.functional.cross_entropy(logits, labels.long().reshape(-1), reduction="mean")


def validate_target_label(logits: Tensor, target_label: int) -> None:
    """Fail early if the requested benign/target class cannot be represented."""
    if logits.ndim == 1 or (logits.ndim >= 2 and logits.shape[-1] == 1):
        if target_label not in (0, 1):
            raise ValueError(f"One-logit binary NIDS only supports target_label 0 or 1, got {target_label}")
    elif logits.ndim == 2 and not 0 <= target_label < int(logits.shape[-1]):
        raise ValueError(
            f"target_label={target_label} is outside the model class range [0, {logits.shape[-1] - 1}]"
        )


def predicted_labels(logits: Tensor) -> Tensor:
    """Convert a standard NIDS logits batch to integer class predictions."""
    if logits.ndim == 1 or (logits.ndim >= 2 and logits.shape[-1] == 1):
        return (torch.sigmoid(logits.reshape(-1)) >= 0.5).long()
    if logits.ndim != 2:
        raise ValueError(f"Unsupported logits shape {tuple(logits.shape)}")
    return logits.argmax(dim=-1)


def resolve_target(
    configured_target: int | None,
    configured_targeted: bool,
    kwargs: dict[str, object],
) -> tuple[bool, int | None]:
    """Resolve a per-call target override while preserving constructor config."""
    override = kwargs.get("target_label")
    effective_target = int(override) if override is not None else configured_target
    targeted = bool(configured_targeted or effective_target is not None)
    if targeted and effective_target is None:
        raise ValueError("target_label is required for targeted NIDS evasion")
    return targeted, effective_target
