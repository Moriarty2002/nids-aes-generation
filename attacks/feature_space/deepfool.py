from __future__ import annotations

"""Targeted DeepFool for differentiable tabular/sequence NIDS inputs.

Algorithm adapted conceptually from Moosavi-Dezfooli et al., "DeepFool: a simple
and accurate method to fool deep neural networks" (CVPR 2016). Original author
implementation: https://github.com/LTS4/DeepFool. This is an independent,
framework-specific targeted adaptation; no original repository code is copied.
it intentionally uses only the hyperplane between the supplied ground-truth
class and ``target_label`` (normally Benign=0), rather than searching for the
closest boundary among every class. No gatekeeper rules participate in the
optimization; protocol realizability is assessed afterward by EvaluationEngine.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from attacks.base import AttackResult, FeatureSpaceAttack
from core.exceptions import IncompatibleThreatModelError
from core.registry import register_attack
from models.wrapper import PyTorchModelWrapper, ensure_gradient_compatible
from ._linf_utils import immutable_feature_mask, resolve_target, validate_target_label
from ._optimization_utils import class_scores, input_predictions, project_box_delta, resolve_bounds, restore_module_modes, restore_saved_modes


@dataclass
@register_attack("deepfool")
class DeepFoolAttack(FeatureSpaceAttack):
    """Target each sample's source-to-Benign decision boundary.

    The source class is fixed to the provided ground-truth label for the entire
    optimization. At each iteration, a first-order approximation estimates the
    minimum normal displacement to the source/target logit hyperplane. Gradients
    are obtained from *summed logits*, not a mean-reduced loss, so the effective
    displacement is independent of the current batch size.

    Inputs can be flat ``(N,D)`` arrays or structured ``(N,T,F)`` sequences.
    Only one caller-supplied batch is moved to the model device. The mandatory
    directional mask freezes ``DIR`` and all victim-sent packet features.
    """

    target_label: int | None = None
    max_steps: int = 50
    overshoot: float = 0.02
    gradient_epsilon: float = 1e-12
    clip_min: float | Sequence[float] | np.ndarray | Tensor | None = None
    clip_max: float | Sequence[float] | np.ndarray | Tensor | None = None
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()

    attack_name = "deepfool"
    requires_gradients = True
    query_based = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be an integer >= 1")
        if not np.isfinite(self.overshoot) or self.overshoot < 0:
            raise ValueError("overshoot must be finite and >= 0")
        if not np.isfinite(self.gradient_epsilon) or self.gradient_epsilon <= 0:
            raise ValueError("gradient_epsilon must be finite and > 0")
        self.feature_names = tuple(str(name) for name in self.feature_names)
        self.immutable_features = tuple(str(name) for name in self.immutable_features)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        model = self._require_model()
        ensure_gradient_compatible(model)
        if not isinstance(model, PyTorchModelWrapper):
            raise IncompatibleThreatModelError("DeepFool requires a PyTorchModelWrapper")

        # Per-batch device allocation
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
            raise ValueError(f"DeepFool expects a non-empty 2-D or 3-D batch, got {tuple(x0.shape)}")
        if labels.numel() != x0.shape[0]:
            raise ValueError("x and y must contain the same number of samples")
        if not torch.isfinite(x0).all():
            raise ValueError("DeepFool input contains NaN or infinite values")

        targeted, target_label = resolve_target(self.target_label, True, dict(kwargs))
        assert targeted and target_label is not None

        # Required security-aware mask call. Statistics are in attack-input space
        # (surrogate normalization when this is a transfer attack).
        mask = self.victim_packet_perturbation_mask(
            x0,
            self.feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
        ).to(device=x0.device, dtype=x0.dtype)
        mask = mask * immutable_feature_mask(x0, self.feature_names, self.immutable_features)
        lower, upper = resolve_bounds(x0, self.feature_names, self.clip_min, self.clip_max)

        model_module = model.model
        saved_modes = restore_module_modes(model_module)
        adversarial = x0.detach().clone()
        iterations_executed = 0
        try:
            # Do not let callers' outer no_grad context disable attack gradients.
            with torch.enable_grad():
                for step in range(self.max_steps):
                    iterations_executed = step + 1
                    candidate = adversarial.detach().requires_grad_(True)
                    logits = model_module(candidate)
                    scores = class_scores(logits)
                    if step == 0:
                        validate_target_label(logits, int(target_label))
                        if torch.any((labels < 0) | (labels >= scores.shape[1])):
                            raise ValueError("Ground-truth labels are outside the model's class range")
                    if torch.any((labels < 0) | (labels >= scores.shape[1])):
                        raise ValueError("Ground-truth labels are outside the model's class range")

                    target_scores = scores[:, int(target_label)]
                    source_scores = scores.gather(1, labels.view(-1, 1)).squeeze(1)
                    # Ground-truth class is fixed; samples already across this
                    # particular hyperplane do not need another displacement.
                    active = (labels != int(target_label)) & (source_scores > target_scores)
                    if not bool(active.any()):
                        adversarial = candidate.detach()
                        break

                    active_weight = active.to(dtype=scores.dtype)
                    grad_target = torch.autograd.grad(
                        (target_scores * active_weight).sum(),
                        candidate,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grad_source = torch.autograd.grad(
                        (source_scores * active_weight).sum(),
                        candidate,
                        retain_graph=False,
                        only_inputs=True,
                    )[0]
                    normal = (grad_target - grad_source) * mask
                    normal_sq = normal.reshape(len(x0), -1).square().sum(dim=1)
                    usable = active & (normal_sq > self.gradient_epsilon)
                    if not bool(usable.any()):
                        adversarial = candidate.detach()
                        break

                    margin = (source_scores - target_scores).clamp_min(0.0)
                    scale = torch.zeros_like(margin)
                    scale[usable] = (
                        (1.0 + self.overshoot)
                        * margin[usable]
                        / normal_sq[usable].clamp_min(self.gradient_epsilon)
                    )
                    step_delta = scale.view((-1,) + (1,) * (x0.ndim - 1)) * normal
                    # Explicitly mask both the update and its cumulative projection.
                    step_delta = step_delta * mask
                    with torch.no_grad():
                        proposed_delta = candidate.detach() + step_delta - x0
                        proposed_delta = project_box_delta(x0, proposed_delta * mask, mask, lower, upper)
                        adversarial = (x0 + proposed_delta * mask).detach()
            # Final prediction is still in eval mode; do not allow a caller's
            # training-mode state to update BatchNorm or activate Dropout here.
            with torch.no_grad():
                final_logits = model_module(adversarial)
                predictions = input_predictions(final_logits)
                success = predictions.eq(int(target_label)) & labels.ne(int(target_label))
                distance = (adversarial - x0).reshape(len(x0), -1).norm(p=2, dim=1)
        finally:
            restore_saved_modes(saved_modes)

        return AttackResult(
            adversarial=adversarial.detach(),
            success_mask=success.detach(),
            queries=np.zeros(len(adversarial), dtype=np.int64),
            intrinsic_feasible_mask=np.ones(len(adversarial), dtype=bool),
            feature_names=self.feature_names or None,
            metadata={
                "attack": self.attack_name,
                "targeted": True,
                "target_label": int(target_label),
                "max_steps": int(self.max_steps),
                "overshoot": float(self.overshoot),
                "mean_l2_perturbation": float(distance.mean().detach().cpu()),
                "iterations_executed": int(iterations_executed),
            },
        )
