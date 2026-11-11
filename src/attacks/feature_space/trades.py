from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from attacks.base import AttackResult, FeatureSpaceAttack
from core.registry import register_attack
from models.wrapper import PyTorchModelWrapper, ensure_gradient_compatible
from ._optimization_utils import class_scores, restore_module_modes, restore_saved_modes
from ._linf_utils import immutable_feature_mask


@register_attack("trades")
class TRADESAttack(FeatureSpaceAttack):
    """TRADES Algorithm 1 inner KL maximization, used for defense augmentation.

    Original mechanism: .001 Gaussian start and projected sign ascent of
    KL(clean || adversarial). Necessary NIDS adaptation: normalized packet
    coordinates, exact occupancy/edit masks and an independent surrogate replace
    images and the source classifier. Physical acceptance stays external.
    A one-logit NIDS head represents Bernoulli probabilities via scores [0,z].
    This defense primitive is never a primary per-flow evasion method.
    """

    attack_name = "trades"
    hard_budget_norm = "linf"
    requires_gradients = True

    def __init__(
        self,
        epsilon: float = 0.04,
        alpha: float = 0.01,
        steps: int = 10,
        trades_beta: float = 6.0,
        discrete_projection: bool = True,
        immutable_features: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(discrete_projection=discrete_projection, **kwargs)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.trades_beta = float(trades_beta)
        self.immutable_features = tuple(str(f).upper() for f in immutable_features)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and >= 0")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and >= 0")
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be an integer >= 1")

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        model_wrapper = self._require_model()
        ensure_gradient_compatible(model_wrapper)
        if not isinstance(model_wrapper, PyTorchModelWrapper):
            raise TypeError("TRADES inner optimization requires a PyTorchModelWrapper")
        x_clean = torch.as_tensor(x, dtype=torch.float32, device=model_wrapper.device)

        # Build feature mask to respect immutable features
        feature_names = tuple(kwargs.get("feature_names", self.feature_names))
        mask = self.victim_packet_perturbation_mask(
            x_clean,
            feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
            valid_mask=kwargs.get("valid_mask")
        )
        mask = mask * immutable_feature_mask(
            x_clean,
            feature_names,
            self.immutable_features,
        )

        seed = kwargs.get("seed")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=model_wrapper.device).manual_seed(int(seed))
        delta = .001 * torch.randn(x_clean.shape, device=x_clean.device,
                                   dtype=x_clean.dtype, generator=generator)
        delta = delta.clamp(-self.epsilon, self.epsilon) * mask
        saved_modes = restore_module_modes(model_wrapper.model)

        try:
            with torch.no_grad():
                clean_logits = model_wrapper.forward_differentiable(x_clean, valid_mask=kwargs.get("valid_mask"))
                clean_probs = F.softmax(class_scores(clean_logits), dim=-1).detach()
            with torch.enable_grad():
                for _ in range(self.steps):
                    delta = delta.detach().requires_grad_(True)
                    x_adv = x_clean + delta * mask
                    adv_logits = model_wrapper.forward_differentiable(x_adv, valid_mask=kwargs.get("valid_mask"))
                    adv_log_probs = F.log_softmax(class_scores(adv_logits), dim=-1)

                    loss = F.kl_div(adv_log_probs, clean_probs, reduction="batchmean")
                    grad = torch.autograd.grad(loss, delta, only_inputs=True)[0].detach()

                    with torch.no_grad():
                        delta = (delta + self.alpha * grad.sign()).clamp(-self.epsilon, self.epsilon) * mask
        finally:
            restore_saved_modes(saved_modes)

        with torch.no_grad():
            x_final = (x_clean + delta.detach() * mask).detach().cpu().numpy()
        return AttackResult(adversarial=x_final, feature_names=feature_names)
