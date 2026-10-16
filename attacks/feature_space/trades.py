from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.base import AttackResult, FeatureSpaceAttack
from core.registry import register_attack


@register_attack("trades")
class TRADESAttack(FeatureSpaceAttack):
    """TRADES KL-divergence adversarial generator for surrogate augmentation when using Empirical Adversarial Training.
        
        It is implemented as an attack as a workaround to allow for the use of the same interface as other attacks, but it is not a traditional attack in the sense that it does not aim to generate adversarial examples that fool a model. 
        Instead, it generates adversarial examples that are used to augment the training data against a surrogate model to feed them to a non differentiable model.
    """

    attack_name = "trades"
    requires_gradients = True

    def __init__(
        self,
        epsilon: float = 0.04,
        alpha: float = 0.01,
        steps: int = 10,
        trades_beta: float = 6.0,
        discrete_projection: bool = True,
        immutable_features: tuple[str, ...] = ("DIR", "FLG"),
        **kwargs: Any,
    ) -> None:
        super().__init__(discrete_projection=discrete_projection, **kwargs)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.trades_beta = float(trades_beta)
        self.immutable_features = tuple(str(f).upper() for f in immutable_features)

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        model_wrapper = self._require_model()
        x_clean = torch.as_tensor(x, dtype=torch.float32, device=model_wrapper.device)
        
        # Calculate clean surrogate outputs (probabilities)
        with torch.no_grad():
            clean_logits = model_wrapper.forward_differentiable(x_clean)
            clean_probs = F.softmax(clean_logits, dim=-1)
            clean_probs = clean_probs.detach()
            del clean_logits

        # Initialize perturbation with uniform noise
        delta = torch.zeros_like(x_clean).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_(True)

        # Build feature mask to respect immutable features
        feature_names = tuple(kwargs.get("feature_names", self.feature_names))
        mask = self.victim_packet_perturbation_mask(
            x_clean,
            feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
        )
        for idx, name in enumerate(feature_names):
            if name.upper() in self.immutable_features:
                mask[..., idx] = 0.0

        criterion_kl = nn.KLDivLoss(reduction="batchmean")

        # Inner-loop KL divergence maximization
        with torch.enable_grad():
            for _ in range(self.steps):
                x_adv = x_clean + delta * mask
                adv_logits = model_wrapper.forward_differentiable(x_adv)
                adv_log_probs = F.log_softmax(adv_logits, dim=-1)

                # Maximize KL divergence between clean prediction and perturbed prediction
                loss = criterion_kl(adv_log_probs, clean_probs)
                grad = torch.autograd.grad(
                    loss, delta, only_inputs=True, retain_graph=False, create_graph=False
                )[0].detach()

                with torch.no_grad():
                    delta.add_(self.alpha * grad.sign())
                    delta.clamp_(-self.epsilon, self.epsilon)
                delta = delta.detach().requires_grad_(True)
                adv_logits = adv_logits.detach()
                adv_log_probs = adv_log_probs.detach()
                loss = loss.detach()
                del x_adv, adv_logits, adv_log_probs, loss, grad

        with torch.no_grad():
            x_final = (x_clean + delta.detach() * mask).detach().cpu().numpy()
        del delta, clean_probs, mask, criterion_kl
        return AttackResult(adversarial=x_final, feature_names=feature_names)