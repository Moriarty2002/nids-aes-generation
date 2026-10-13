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
@register_attack("fgsm")
class FGSMAttack(FeatureSpaceAttack):
    """Fast Gradient Sign Method for feature-space NIDS evasion.

    FGSM takes one deterministic sign-gradient step in normalized feature space.
    If ``target_label`` is supplied (normally Benign class 0), it minimizes the
    loss for that label; otherwise it performs the traditional untargeted step
    that increases loss on the true class.

    The directional mask is security-specific: packet features are editable only
    when the endpoint attacker is allowed to modify that packet. The ``DIR``
    channel itself and configured immutable fields are always frozen. Full RFC
    and physical realizability checks remain post-generation work for the
    EvaluationEngine's UniversalConstraintGatekeeper.
    """

    epsilon: float = 0.01
    targeted: bool = False
    target_label: int | None = None
    clip_min: float | Sequence[float] | np.ndarray | Tensor | None = None
    clip_max: float | Sequence[float] | np.ndarray | Tensor | None = None
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()

    attack_name = "fgsm"
    requires_gradients = True
    query_based = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be a finite value >= 0")
        self.feature_names = tuple(str(name) for name in self.feature_names)
        self.immutable_features = tuple(str(name) for name in self.immutable_features)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")
        if self.target_label is not None:
            self.targeted = True

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Generate one masked FGSM step, allocating only this batch on the model device."""
        model = self._require_model()
        ensure_gradient_compatible(model)
        if not isinstance(model, PyTorchModelWrapper):
            raise IncompatibleThreatModelError("FGSM requires a PyTorchModelWrapper")

        # Move only the current CPU-bound batch to the model device immediately.
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
            raise ValueError(f"FGSM expects a non-empty 2-D or 3-D batch, got {tuple(x0.shape)}")
        if labels.numel() != x0.shape[0]:
            raise ValueError("x and y must contain the same number of samples")

        targeted, target_label = resolve_target(self.target_label, self.targeted, dict(kwargs))
        objective_labels = torch.full_like(labels, int(target_label)) if targeted else labels

        # Required directional/victim-packet mask; normalization stats are needed
        # to decode DIR correctly when the input has been normalized.
        mask = self.victim_packet_perturbation_mask(
            x0,
            self.feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
        ).to(device=x0.device, dtype=x0.dtype)
        mask = mask * immutable_feature_mask(x0, self.feature_names, self.immutable_features)

        model_module = model.model
        module_modes = [(module, module.training) for module in model_module.modules()]
        model_module.eval()  # Avoid BatchNorm/Dropout state changes during attack generation.
        try:
            with torch.enable_grad():
                x_leaf = x0.detach().requires_grad_(True)
                logits = model_module(x_leaf)
                if targeted:
                    assert target_label is not None
                    validate_target_label(logits, target_label)
                loss = classification_loss(logits, objective_labels)
                # Maximizing -CE(target) is equivalent to minimizing CE(target).
                objective = -loss if targeted else loss
                gradient = torch.autograd.grad(objective, x_leaf, only_inputs=True)[0]

                with torch.no_grad():
                    # FGSM starts from delta=0: no random initialization.
                    delta = float(self.epsilon) * gradient.sign()
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
                    adversarial = (x0 + delta).detach()
                    adv_logits = model_module(adversarial)
                    predictions = predicted_labels(adv_logits)
                    success = predictions.eq(objective_labels) if targeted else predictions.ne(labels)
        finally:
            # Restore each submodule's exact prior mode (including custom mixed modes).
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
                "steps": 1,
                "targeted": bool(targeted),
                "target_label": target_label,
                "random_start": False,
            },
        )
