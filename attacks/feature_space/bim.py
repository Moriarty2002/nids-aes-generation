from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from attacks.base import AttackResult, FeatureSpaceAttack
from core.exceptions import IncompatibleThreatModelError
from core.registry import register_attack
from models.wrapper import PyTorchModelWrapper, ensure_gradient_compatible
from ._linf_utils import (
    classification_loss,
    immutable_feature_mask,
    predicted_labels,
    project_linf_delta,
    resolve_target,
    validate_target_label,
)


@dataclass
@register_attack("bim")
class BIMAttack(FeatureSpaceAttack):
    """Basic Iterative Method for masked L-infinity NIDS evasion.

    BIM repeatedly applies sign-gradient updates from the clean sample (no
    random start), projecting every step into the epsilon ball and optional
    normalized-feature box bounds. With a target label, the update minimizes
    target cross entropy so AC attacks must move traffic specifically toward
    Benign rather than merely confusing one attack class with another.

    Each call moves only its batch from CPU to the attack model's device. The
    shared directional mask freezes ``DIR`` metadata and all features belonging
    to victim-sent packets, while ``immutable_features`` can freeze additional
    schema fields. Non-differentiable protocol rules remain outside this loop
    and are measured by the evaluation gatekeeper after generation.
    """

    epsilon: float = 0.01
    alpha: float = 0.002
    steps: int = 10
    targeted: bool = False
    target_label: int | None = None
    clip_min: float | Sequence[float] | np.ndarray | Tensor | None = None
    clip_max: float | Sequence[float] | np.ndarray | Tensor | None = None
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()

    attack_name = "bim"
    requires_gradients = True
    query_based = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be a finite value >= 0")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be a finite value >= 0")
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be an integer >= 1")
        self.feature_names = tuple(str(name) for name in self.feature_names)
        self.immutable_features = tuple(str(name) for name in self.immutable_features)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")
        if self.target_label is not None:
            self.targeted = True

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Generate a deterministic multi-step BIM batch on the bound model device."""
        model = self._require_model()
        ensure_gradient_compatible(model)
        if not isinstance(model, PyTorchModelWrapper):
            raise IncompatibleThreatModelError("BIM requires a PyTorchModelWrapper")

        # Move only this CPU-bound batch to the model device at method entry.
        x0 = (
            x.to(model.device, dtype=torch.float32)
            if isinstance(x, Tensor)
            else torch.as_tensor(np.asarray(x), dtype=torch.float32, device=model.device)
        )
        labels = (
            y.to(model.device, dtype=torch.long)
            if isinstance(y, Tensor)
            else torch.as_tensor(np.asarray(y), dtype=torch.long, device=model.device)
        ).reshape(-1)
        if x0.ndim not in (2, 3) or x0.shape[0] == 0:
            raise ValueError(f"BIM expects a non-empty 2-D or 3-D batch, got {tuple(x0.shape)}")
        if labels.numel() != x0.shape[0]:
            raise ValueError("x and y must contain the same number of samples")

        targeted, target_label = resolve_target(self.target_label, self.targeted, dict(kwargs))
        objective_labels = torch.full_like(labels, int(target_label)) if targeted else labels

        # Required mask: decode normalized DIR using the training-time statistics.
        mask = self.victim_packet_perturbation_mask(
            x0,
            self.feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
        ).to(device=x0.device, dtype=x0.dtype)
        mask = mask * immutable_feature_mask(x0, self.feature_names, self.immutable_features)

        model_module = model.model
        module_modes = [(module, module.training) for module in model_module.modules()]
        model_module.eval()  # Avoid BatchNorm/Dropout state changes during gradient steps.
        try:
            # BIM is deterministic and starts at the clean sample.
            delta = torch.zeros_like(x0) * mask
            with torch.enable_grad():
                for step in range(self.steps):
                    delta = (delta * mask).detach().requires_grad_(True)
                    logits = model_module(x0 + delta)
                    if step == 0 and targeted:
                        assert target_label is not None
                        validate_target_label(logits, target_label)
                    loss = classification_loss(logits, objective_labels)
                    objective = -loss if targeted else loss
                    gradient = torch.autograd.grad(objective, delta, only_inputs=True)[0]

                    with torch.no_grad():
                        delta = delta + float(self.alpha) * gradient.sign()
                        # Mask on every update before and after projection, so a box
                        # clamp cannot perturb DIR or victim-owned packet metadata.
                        delta = delta * mask
                        delta = project_linf_delta(
                            x0,
                            delta,
                            mask,
                            epsilon=self.epsilon,
                            clip_min=self.clip_min,
                            clip_max=self.clip_max,
                            feature_names=self.feature_names,
                        )

                with torch.no_grad():
                    adversarial = (x0 + delta.detach()).detach()
                    adv_logits = model_module(adversarial)
                    predictions = predicted_labels(adv_logits)
                    success = predictions.eq(objective_labels) if targeted else predictions.ne(labels)
        finally:
            for module, was_training in module_modes:
                module.training = was_training

        return AttackResult(
            adversarial=adversarial,
            success_mask=success.detach(),
            queries=np.zeros(len(adversarial), dtype=np.int64),
            intrinsic_feasible_mask=np.ones(len(adversarial), dtype=bool),
            feature_names=self.feature_names or None,
            metadata={
                "attack": self.attack_name,
                "epsilon": float(self.epsilon),
                "alpha": float(self.alpha),
                "steps": int(self.steps),
                "targeted": bool(targeted),
                "target_label": target_label,
                "random_start": False,
            },
        )
