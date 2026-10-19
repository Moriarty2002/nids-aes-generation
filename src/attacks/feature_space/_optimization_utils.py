"""Shared helpers for magnitude-dependent white-box feature-space attacks.

The helpers keep feature-space optimization differentiable. They do not invoke
or approximate the UniversalConstraintGatekeeper; protocol realizability is
checked by EvaluationEngine only after an attack has generated its candidates.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from ._linf_utils import validate_target_label

Bound = float | Sequence[float] | np.ndarray | Tensor | None


def class_scores(logits: Tensor) -> Tensor:
    """Return ``(batch, classes)`` scores, expanding a one-logit binary head.

    A single binary logit ``z`` represents scores ``[0, z]``. This preserves the
    usual 0.5 threshold and allows targeted margin attacks to use one shared
    multiclass formulation for binary anomaly detection and multiclass attack
    classification.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.ndim != 2:
        raise ValueError(
            "White-box tabular attacks require logits shaped (batch, classes) "
            f"or one logit per sample; received {tuple(logits.shape)}"
        )
    if logits.shape[-1] == 1:
        return torch.cat((torch.zeros_like(logits), logits), dim=-1)
    if logits.shape[-1] < 2:
        raise ValueError("Model must expose at least two classes")
    return logits


def resolve_bounds(
    x0: Tensor,
    feature_names: Sequence[str],
    clip_min: Bound,
    clip_max: Bound,
) -> tuple[float | Tensor, float | Tensor]:
    """Create broadcastable feature bounds for flat vectors or packet sequences."""
    lower = _bound_tensor(clip_min, x0, feature_names, default=-float("inf"))
    upper = _bound_tensor(clip_max, x0, feature_names, default=float("inf"))
    if isinstance(lower, Tensor) or isinstance(upper, Tensor):
        lower_tensor = torch.as_tensor(lower, device=x0.device, dtype=x0.dtype)
        upper_tensor = torch.as_tensor(upper, device=x0.device, dtype=x0.dtype)
        try:
            if torch.isnan(lower_tensor).any() or torch.isnan(upper_tensor).any():
                raise ValueError("clip_min and clip_max must not contain NaN")
            if torch.any(lower_tensor > upper_tensor):
                raise ValueError("clip_min must be <= clip_max for every feature")
        except RuntimeError as exc:
            raise ValueError("clip_min and clip_max are not mutually broadcastable") from exc
        return lower_tensor, upper_tensor
    if lower > upper:
        raise ValueError("clip_min must be <= clip_max")
    return lower, upper


def _bound_tensor(
    value: Bound,
    x: Tensor,
    feature_names: Sequence[str],
    *,
    default: float,
) -> float | Tensor:
    """Convert scalar, per-feature or per-position bounds to broadcastable values."""
    if value is None:
        return default
    bound = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    if bound.ndim == 0:
        return bound

    feature_count = len(feature_names)
    if x.ndim == 3:
        _, packets, features = x.shape
        if bound.ndim == 1 and bound.numel() == features:
            return bound.view(1, 1, features)
        if bound.numel() == packets * features:
            return bound.reshape(1, packets, features)
    elif x.ndim == 2:
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


def project_box_delta(
    x0: Tensor,
    delta: Tensor,
    mask: Tensor,
    lower: float | Tensor,
    upper: float | Tensor,
) -> Tensor:
    """Project a candidate perturbation into the feature box and edit mask.

    Masking is applied both before and after clipping. Therefore, a frozen DIR
    field or any feature belonging to a victim-sent packet cannot be changed by
    a bounds operation, including when a clean input is outside configured
    bounds already.
    """
    masked_delta = delta * mask
    candidate = torch.clamp(x0 + masked_delta, min=lower, max=upper)
    return (candidate - x0) * mask


def restore_module_modes(model: torch.nn.Module) -> list[tuple[torch.nn.Module, bool]]:
    """Capture per-module modes and switch the model to deterministic eval mode."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    return modes


def restore_saved_modes(modes: list[tuple[torch.nn.Module, bool]]) -> None:
    """Restore exact modes instead of blindly calling ``model.train()``."""
    for module, was_training in modes:
        module.training = was_training


def input_predictions(logits: Tensor) -> Tensor:
    """Decode logits using the wrapper's exact predict() threshold convention.

    In a one-logit binary head, the framework predicts class 1 only when
    sigmoid(logit) > 0.5; a tie belongs to class 0. Multiclass predictions use
    argmax as usual.
    """
    if logits.ndim == 1 or (logits.ndim >= 2 and logits.shape[-1] == 1):
        return (torch.sigmoid(logits.reshape(-1)) > 0.5).long()
    if logits.ndim != 2:
        raise ValueError(f"Unsupported logits shape {tuple(logits.shape)}")
    return logits.argmax(dim=-1)


__all__ = [
    "Bound",
    "class_scores",
    "input_predictions",
    "project_box_delta",
    "resolve_bounds",
    "restore_module_modes",
    "restore_saved_modes",
    "validate_target_label",
]
