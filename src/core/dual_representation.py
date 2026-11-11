"""Canonical float32 model encoding and float64 physical decoding.

All participating pipeline stages use these exact routines. Neither operation
claims that decoding a previously quantized float32 value recovers its original
pre-normalization physical source.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import torch

Array = np.ndarray | torch.Tensor
Stats = Sequence[float] | Array


def layout(x: Array, feature_names: Sequence[str]) -> tuple[int, int, int]:
    f = len(feature_names)
    if not f or len(set(str(n).upper() for n in feature_names)) != f:
        raise ValueError("feature_names must be nonempty and unique")
    if x.ndim == 3:
        if x.shape[2] != f:
            raise ValueError("Physical/normalized sequence feature width mismatch")
        return int(x.shape[0]), int(x.shape[1]), f
    if x.ndim == 2 and x.shape[1] % f == 0:
        return int(x.shape[0]), int(x.shape[1] // f), f
    raise ValueError("Expected (B,F), (B,T*F), or (B,T,F)")


def stats_views(x: Array, names: Sequence[str], mean: Stats | None, std: Stats | None) -> tuple[Array, Array]:
    _, t, f = layout(x, names)
    if (mean is None) != (std is None):
        raise ValueError("Normalization mean and std must be supplied together")
    if isinstance(x, torch.Tensor):
        if mean is None:
            return (torch.zeros((1, 1, f), dtype=torch.float64, device=x.device),
                    torch.ones((1, 1, f), dtype=torch.float64, device=x.device))
        mu = torch.as_tensor(mean, dtype=torch.float64, device=x.device).reshape(-1)
        sigma = torch.as_tensor(std, dtype=torch.float64, device=x.device).reshape(-1)
        if mu.numel() != sigma.numel() or mu.numel() not in (f, t*f):
            raise ValueError("Statistics must have F or T*F elements")
        if not bool(torch.isfinite(mu).all()) or not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0).any()):
            raise ValueError("Normalization statistics must be finite with positive std")
        return mu.reshape(1, -1, f), sigma.reshape(1, -1, f)
    if mean is None:
        return np.zeros((1, 1, f), dtype=np.float64), np.ones((1, 1, f), dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    sigma = np.asarray(std, dtype=np.float64).reshape(-1)
    if mu.size != sigma.size or mu.size not in (f, t*f):
        raise ValueError("Statistics must have F or T*F elements")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Normalization statistics must be finite with positive std")
    return mu.reshape(1, -1, f), sigma.reshape(1, -1, f)


def decode_physical64(normalized: Array, names: Sequence[str], mean: Stats | None, std: Stats | None) -> Array:
    """Allocate one float64 batch, then denormalize in-place."""
    b, t, f = layout(normalized, names)
    mu, sigma = stats_views(normalized, names, mean, std)
    if isinstance(normalized, torch.Tensor):
        with torch.no_grad():
            out = normalized.detach().to(torch.float64).contiguous()
            view = out.view(b, t, f)
            view.mul_(sigma).add_(mu)
            return out
    out = np.array(normalized, dtype=np.float64, copy=True, order="C")
    view = out.reshape(b, t, f)
    np.multiply(view, sigma, out=view)
    np.add(view, mu, out=view)
    return out


def encode_normalized32(physical: Array, names: Sequence[str], mean: Stats | None, std: Stats | None) -> Array:
    """Canonical physical->float32 encoding. One output and one B*T scratch.

    Float64 subtract/divide precede a single float32 cast. No reconstruction
    or clipping is performed inside this function.
    """
    if physical.dtype != (torch.float64 if isinstance(physical, torch.Tensor) else np.float64):
        raise TypeError("Authoritative physical input must be float64")
    b, t, f = layout(physical, names)
    mu, sigma = stats_views(physical, names, mean, std)
    if isinstance(physical, torch.Tensor):
        with torch.no_grad():
            src = physical.reshape(b, t, f)
            out = torch.empty(physical.shape, dtype=torch.float32, device=physical.device)
            target = out.view(b, t, f)
            scratch = torch.empty((b, t), dtype=torch.float64, device=physical.device)
            for j in range(f):
                scratch.copy_(src[..., j])
                scratch.sub_(mu[..., j]).div_(sigma[..., j])
                target[..., j].copy_(scratch)
            return out
    src = physical.reshape(b, t, f)
    out = np.empty(physical.shape, dtype=np.float32)
    target = out.reshape(b, t, f)
    scratch = np.empty((b, t), dtype=np.float64)
    for j in range(f):
        np.copyto(scratch, src[..., j])
        np.subtract(scratch, mu[..., j], out=scratch)
        np.divide(scratch, sigma[..., j], out=scratch)
        np.copyto(target[..., j], scratch, casting="unsafe")
    return out



def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _restore_target_physical_protected_inplace(
    adversarial: np.ndarray,
    original: np.ndarray,
    feature_names: Sequence[str],
    *,
    immutable_features: Sequence[str] = (),
    valid_mask: np.ndarray | torch.Tensor | None = None,
) -> tuple[int, int]:
    """Last-stage physical immutability projection, in-place without drift."""
    if adversarial.dtype != np.float64 or original.dtype != np.float64:
        raise TypeError("Both authoritative arrays must be float64")
    if adversarial.shape != original.shape or not adversarial.flags.writeable:
        raise ValueError("Physical batches must match and adversarial must be writable")
    b, t, f = layout(adversarial, feature_names)
    names = tuple(str(n).upper() for n in feature_names)
    adv = adversarial.reshape(b, t, f)
    clean = original.reshape(b, t, f)
    valid = None if valid_mask is None else _as_numpy(valid_mask).astype(bool, copy=False)
    if valid is not None and valid.shape != (b, t):
        raise ValueError("Invalid valid_mask shape")
    immutable = {str(n).upper() for n in immutable_features}
    if "DIR" in names:
        immutable.add("DIR")
    if immutable - set(names):
        raise ValueError(f"Unknown immutable features: {sorted(immutable-set(names))}")
    victim_count = 0
    if "DIR" in names:
        victim = np.isclose(clean[..., names.index("DIR")], 1.0, atol=1e-2, rtol=0.0)
        if valid is not None:
            victim &= valid
        victim_count = int(np.count_nonzero(victim))
        np.copyto(adv, clean, where=victim[..., None])
    for key in immutable:
        j = names.index(key)
        np.copyto(adv[..., j], clean[..., j])
    padded = 0
    if valid is not None:
        padded = int(np.count_nonzero(~valid))
        np.copyto(adv, clean, where=(~valid)[..., None])
    return victim_count, padded


def _as_physical_shape(
    raw: np.ndarray | torch.Tensor,
    expected: np.ndarray | torch.Tensor,
    names: Sequence[str],
) -> np.ndarray:
    """Accept flattened/sequence-equivalent physical layouts without copying."""
    physical = _as_numpy(raw)
    if physical.dtype != np.float64:
        raise TypeError("Physical source must be float64, not normalized float32")
    b, t, f = layout(expected, names)
    if physical.shape[0] != b or physical.size != b * t * f:
        raise ValueError("Physical batch and target batch dimensions differ")
    return physical.reshape(tuple(expected.shape))
