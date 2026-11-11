"""Reusable robust-optimization strategies for PyTorch NIDS training."""

from __future__ import annotations

from typing import Callable, Literal

import torch
from torch import nn

from models.wrapper import PyTorchModelWrapper, forward_model

DefenseMode = Literal["none", "pgd", "trades"]
PerturbationMaskFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StepFn = Callable[
    [nn.Module, torch.Tensor, torch.Tensor, PerturbationMaskFn | None, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor],
]


def _resolve_perturbation_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    perturbation_mask_fn: PerturbationMaskFn | None,
) -> torch.Tensor | None:
    if perturbation_mask_fn is None:
        return None
    with torch.no_grad():
        proposed = perturbation_mask_fn(x, y)
        if not isinstance(proposed, torch.Tensor):
            raise TypeError("perturbation_mask_fn must return a torch.Tensor")
        if proposed.shape != x.shape:
            raise ValueError(
                f"Perturbation mask shape {tuple(proposed.shape)} differs from input {tuple(x.shape)}"
            )
        if not torch.all((proposed == 0) | (proposed == 1)).item():
            raise ValueError("Perturbation mask must contain only binary 0/1 values")
        return proposed.detach().to(device=x.device, dtype=x.dtype)


def _classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), y.float().reshape(-1))
    return nn.functional.cross_entropy(logits, y.long())


def _class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1 or logits.shape[-1] == 1:
        positive = torch.sigmoid(logits.reshape(-1))
        return torch.stack((1.0 - positive, positive), dim=-1)
    return torch.softmax(logits, dim=-1)


def _kl_from_logits(clean_logits: torch.Tensor, other_logits: torch.Tensor) -> torch.Tensor:
    clean_prob = _class_probabilities(clean_logits).detach().clamp_min(1e-12)
    other_prob = _class_probabilities(other_logits).clamp_min(1e-12)
    return nn.functional.kl_div(other_prob.log(), clean_prob, reduction="batchmean")


def _pgd_delta(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if epsilon < 0 or alpha <= 0 or steps < 1:
        raise ValueError("PGD requires epsilon >= 0, alpha > 0, and steps >= 1")
    was_training = model.training
    edit_mask = _resolve_perturbation_mask(x, y, perturbation_mask_fn)
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    if edit_mask is not None:
        delta.mul_(edit_mask)
    try:
        model.eval()
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            logits = forward_model(model, x + delta, valid_mask)
            loss = _classification_loss(logits, y)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            with torch.no_grad():
                delta = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon)
                if edit_mask is not None:
                    delta.mul_(edit_mask)
    finally:
        model.train(was_training)
    return delta.detach()


def _trades_loss_and_logits(
    wrapper: PyTorchModelWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    beta: float,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if epsilon < 0 or step_size <= 0 or steps < 1 or beta < 0:
        raise ValueError("TRADES requires epsilon >= 0, step_size > 0, steps >= 1, and beta >= 0")

    model = wrapper.model
    x = x.to(wrapper.device, dtype=torch.float32)
    y = y.to(wrapper.device, dtype=torch.long)
    was_training = model.training
    edit_mask = _resolve_perturbation_mask(x, y, perturbation_mask_fn)
    # TRADES Algorithm 1: Gaussian .001 start, masked/projected for NIDS.
    # Only the inner search is aligned here; outer-loss fidelity belongs to F4.
    delta = (.001 * torch.randn_like(x)).clamp(-epsilon, epsilon)
    if edit_mask is not None:
        delta.mul_(edit_mask)

    try:
        model.eval()
        with torch.no_grad():
            clean_reference_logits = forward_model(model, x, valid_mask)
            clean_reference_prob = _class_probabilities(clean_reference_logits).detach().clamp_min(1e-12)
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            adv_logits = forward_model(model, x + delta, valid_mask)
            adv_prob = _class_probabilities(adv_logits).clamp_min(1e-12)
            inner_kl = nn.functional.kl_div(adv_prob.log(), clean_reference_prob, reduction="batchmean")
            gradient = torch.autograd.grad(inner_kl, delta, only_inputs=True)[0]
            with torch.no_grad():
                delta = (delta + step_size * gradient.sign()).clamp(-epsilon, epsilon)
                if edit_mask is not None:
                    delta.mul_(edit_mask)
    finally:
        model.train(was_training)

    adversarial = (x + delta.detach()).detach()
    joint_mask = None if valid_mask is None else torch.cat((valid_mask, valid_mask), dim=0)
    joint_logits = forward_model(model, torch.cat((x, adversarial), dim=0), joint_mask)
    natural_logits, adversarial_logits = joint_logits.split(x.shape[0], dim=0)
    natural_loss = _classification_loss(natural_logits, y)
    robust_loss = _kl_from_logits(natural_logits, adversarial_logits)
    return natural_loss + beta * robust_loss, natural_logits


def trades_loss(
    wrapper: PyTorchModelWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.04,
    step_size: float = 0.01,
    steps: int = 10,
    beta: float = 6.0,
    *,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
    return_logits: bool = False,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    loss, natural_logits = _trades_loss_and_logits(
        wrapper, x, y, epsilon=epsilon, step_size=step_size, steps=steps,
        beta=beta, perturbation_mask_fn=perturbation_mask_fn, valid_mask=valid_mask,
    )
    return (loss, natural_logits) if return_logits else loss


def resolve_defense_mode(defense_mode: str | None, adversarial_training: bool) -> DefenseMode:
    if defense_mode is None or not str(defense_mode).strip():
        mode = "pgd" if adversarial_training else "none"
    else:
        mode = str(defense_mode).strip().lower()
        if mode == "none" and adversarial_training:
            mode = "pgd"
    if mode not in {"none", "pgd", "trades"}:
        raise ValueError(f"Unsupported defense_mode={mode!r}; expected 'none', 'pgd', or 'trades'")
    return mode  # type: ignore[return-value]


def make_step_fn(
    mode: DefenseMode,
    *,
    wrapper: PyTorchModelWrapper,
    epsilon: float,
    alpha: float,
    steps: int,
    trades_beta: float,
    perturbation_mask_fn: PerturbationMaskFn | None,
) -> StepFn:
    if mode == "pgd":
        def pgd_step(
            model: nn.Module, x: torch.Tensor, y: torch.Tensor,
            batch_mask_fn: PerturbationMaskFn | None = None,
            valid_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            delta = _pgd_delta(
                model, x, y, epsilon, alpha, steps,
                batch_mask_fn if batch_mask_fn is not None else perturbation_mask_fn,
                valid_mask=valid_mask,
            )
            logits = forward_model(model, x + delta, valid_mask)
            return _classification_loss(logits, y), logits
        return pgd_step
    if mode == "trades":
        def trades_step(
            model: nn.Module, x: torch.Tensor, y: torch.Tensor,
            batch_mask_fn: PerturbationMaskFn | None = None,
            valid_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del model
            return _trades_loss_and_logits(
                wrapper, x, y, epsilon=epsilon, step_size=alpha, steps=steps,
                beta=trades_beta, valid_mask=valid_mask,
                perturbation_mask_fn=(
                    batch_mask_fn if batch_mask_fn is not None else perturbation_mask_fn
                ),
            )
        return trades_step

    def natural_step(
        model: nn.Module, x: torch.Tensor, y: torch.Tensor,
        batch_mask_fn: PerturbationMaskFn | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del batch_mask_fn
        logits = forward_model(model, x, valid_mask)
        return _classification_loss(logits, y), logits
    return natural_step


__all__ = [
    "DefenseMode",
    "PerturbationMaskFn",
    "StepFn",
    "_classification_loss",
    "_pgd_delta",
    "_resolve_perturbation_mask",
    "_trades_loss_and_logits",
    "make_step_fn",
    "resolve_defense_mode",
    "trades_loss",
]
