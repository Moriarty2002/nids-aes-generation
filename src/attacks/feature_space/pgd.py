from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from attacks.base import AttackResult, FeatureSpaceAttack
from core.exceptions import IncompatibleThreatModelError
from core.registry import register_attack
from models.wrapper import PyTorchModelWrapper, ensure_gradient_compatible
from ._linf_utils import (classification_loss, immutable_feature_mask,
                         project_linf_delta, validate_target_label)


@dataclass
@register_attack("pgd")
class PGDAttack(FeatureSpaceAttack):
    """Madry random-start projected sign-gradient optimization.

    Necessary NIDS adaptation: the independent surrogate receives exact packet
    masks; protected coordinates and the normalized L∞/declared box restrict
    optimization. Physical acceptance remains independent after generation.

    ``target_label`` turns PGD into a targeted attack automatically. This keeps
    attack optimization independent of the universal gatekeeper while allowing
    AD/AC orchestration to explicitly optimize toward Benign (label 0 by the
    framework convention).
    """
    hard_budget_norm = "linf"

    epsilon: float = 0.03
    alpha: float = 0.005
    steps: int = 20
    targeted: bool = False
    target_label: int | None = None
    clip_min: float | Sequence[float] | np.ndarray | None = None
    clip_max: float | Sequence[float] | np.ndarray | None = None
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()

    attack_name = "pgd"
    requires_gradients = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and >= 0")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and >= 0")
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("steps must be an integer > 0")
        self.feature_names = tuple(str(name).upper() for name in self.feature_names)
        self.immutable_features = tuple(str(name).upper() for name in self.immutable_features)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")
        unknown = sorted(set(self.immutable_features).difference(self.feature_names))
        if unknown and self.feature_names:
            raise ValueError(
                f"Unknown immutable feature name(s): {unknown}; available: {self.feature_names}"
            )
        # Supplying a semantic target is sufficient to select targeted PGD;
        if self.target_label is not None:
            self.targeted = True

    def _projection(
        self,
        x0: torch.Tensor,
        delta: torch.Tensor,
        perturbation_mask: torch.Tensor,
    ) -> torch.Tensor:
        return project_linf_delta(
            x0, delta, perturbation_mask, epsilon=self.epsilon,
            clip_min=self.clip_min, clip_max=self.clip_max, feature_names=self.feature_names,
        )

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Generate bounded adversarial features using projected gradients."""
        model = self._require_model()
        ensure_gradient_compatible(model)
        if not isinstance(model, PyTorchModelWrapper):
            raise IncompatibleThreatModelError("PGD currently requires a PyTorchModelWrapper")

        x0 = (
            x.to(model.device, dtype=torch.float32)
            if isinstance(x, torch.Tensor)
            else torch.as_tensor(np.asarray(x), dtype=torch.float32, device=model.device)
        )
        labels = (
            y.to(model.device, dtype=torch.long)
            if isinstance(y, torch.Tensor)
            else torch.as_tensor(np.asarray(y), dtype=torch.long, device=model.device)
        )
        if x0.ndim < 2:
            raise ValueError(f"PGD expects batch-shaped input, got {tuple(x0.shape)}")
        if len(labels) != len(x0):
            raise ValueError("x and y must contain the same number of samples")

        # Per-call target overrides are useful for research sweeps and preserve
        # backward compatibility with constructor-based configuration.
        effective_target_label = kwargs.get("target_label", self.target_label)
        effective_targeted = self.targeted or effective_target_label is not None
        if effective_targeted and effective_target_label is None:
            raise ValueError("target_label is required for targeted PGD")
        target_label = int(effective_target_label) if effective_target_label is not None else None
        normalization_mean = kwargs.get("normalization_mean")
        normalization_std = kwargs.get("normalization_std")
        perturbation_mask = self.victim_packet_perturbation_mask(
            x0,
            self.feature_names,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            valid_mask=kwargs.get("valid_mask")
        )
        perturbation_mask = perturbation_mask * immutable_feature_mask(
            x0,
            self.feature_names,
            self.immutable_features,
        )

        if effective_targeted:
            assert target_label is not None
            target = torch.full_like(labels, target_label)
        else:
            target = labels

        # Validate the targeted class before entering the optimization loop.
        with torch.no_grad():
            probe_logits = model.forward_differentiable(x0[:1], valid_mask=None if kwargs.get("valid_mask") is None else kwargs["valid_mask"][:1])
        if effective_targeted:
            assert target_label is not None
            validate_target_label(probe_logits, target_label)

        seed = kwargs.get("seed")
        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(device=model.device)
            generator.manual_seed(int(seed))

        delta = torch.empty_like(x0)
        delta.uniform_(-self.epsilon, self.epsilon, generator=generator)
        delta = self._projection(x0, delta, perturbation_mask).detach()

        # Deliberately no gatekeeper, rule projection, or protocol validation is
        # called here. PGD is a pure differentiable optimizer; realizability is
        # measured only by EvaluationEngine after generation.
        for _ in range(self.steps):
            delta.requires_grad_(True)
            logits = model.forward_differentiable(x0 + delta, valid_mask=kwargs.get("valid_mask"))
            loss = classification_loss(logits, target)

            grad = torch.autograd.grad(loss, delta, only_inputs=True, retain_graph=False)[0].detach()
            with torch.no_grad():
                direction = -1.0 if effective_targeted else 1.0
                delta.add_(direction * self.alpha * grad.sign())
                delta.copy_(self._projection(x0, delta, perturbation_mask))
                logits = logits.detach()
                loss = loss.detach()
                logits.zero_()
                loss.zero_()
                grad.zero_()
            delta = delta.detach()
            del logits, loss, grad

        adv = (x0 + delta).detach()
        with torch.no_grad():
            logits = model.forward_differentiable(adv, valid_mask=kwargs.get("valid_mask"))
            if logits.ndim == 1 or logits.shape[-1] == 1:
                pred = (torch.sigmoid(logits) >= 0.5).long().view(-1)
            else:
                pred = logits.argmax(dim=-1)
            logits = logits.detach()
            pred = pred.detach()
            logits.zero_()

        # For targeted AC/AD, success is exactly prediction of the target class.
        # Untargeted mode retains the historical "not the true class" semantics.
        success = (pred.eq(target) if effective_targeted else pred.ne(labels)).detach()
        pred.zero_()
        del logits, pred

        return AttackResult(
            adversarial=adv,
            success_mask=success,
            queries=np.zeros(len(adv), dtype=np.int64),
            intrinsic_feasible_mask=np.ones(len(adv), dtype=bool),
            feature_names=self.feature_names or None,
            metadata={
                "epsilon": float(self.epsilon),
                "alpha": float(self.alpha),
                "steps": int(self.steps),
                "targeted": bool(effective_targeted),
                "target_label": target_label,
            },
        )
