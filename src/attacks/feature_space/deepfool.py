"""Original untargeted DeepFool nearest-boundary search (Moosavi-Dezfooli et al., CVPR 2016).

Reproduces iterative closest linearized competing-class L2 boundary and overshoot.
Required NIDS changes: the independently pretrained surrogate is the only
optimization oracle; immutable/victim/padded coordinates and the physical
post-finalizer/gate remain authoritative. The evaluator separately defines
benign-label success; crossing into another malicious family is not evasion.
Unverified: original authors' image preprocessing and pixel clipping do not
represent packet-domain constraints. Topology: flat or masked (N,T,F) input.
"""
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
from ._linf_utils import immutable_feature_mask
from ._optimization_utils import (class_scores, input_predictions, project_box_delta,
                                  resolve_bounds, restore_module_modes, restore_saved_modes)


@dataclass
@register_attack('deepfool')
class DeepFoolAttack(FeatureSpaceAttack):
    """Estimate minimum masked L2 distance to every competing class per step."""

    target_label: int | None = None  # evaluation success class; never boundary selection
    max_steps: int = 50
    overshoot: float = 0.02
    gradient_epsilon: float = 1e-12
    clip_min: float | Sequence[float] | np.ndarray | Tensor | None = None
    clip_max: float | Sequence[float] | np.ndarray | Tensor | None = None
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()

    attack_name = 'deepfool'
    requires_gradients = True
    query_based = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError('max_steps must be an integer >= 1')
        if not np.isfinite(self.overshoot) or self.overshoot < 0:
            raise ValueError('overshoot must be finite and >= 0')
        if not np.isfinite(self.gradient_epsilon) or self.gradient_epsilon <= 0:
            raise ValueError('gradient_epsilon must be finite and > 0')
        self.feature_names = tuple(str(name) for name in self.feature_names)
        self.immutable_features = tuple(str(name) for name in self.immutable_features)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('feature_names must not contain duplicates')

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        model = self._require_model()
        ensure_gradient_compatible(model)
        if not isinstance(model, PyTorchModelWrapper):
            raise IncompatibleThreatModelError('DeepFool requires a PyTorchModelWrapper')
        x0 = (x.to(model.device, dtype=torch.float32) if isinstance(x, Tensor)
              else torch.as_tensor(np.asarray(x), dtype=torch.float32, device=model.device))
        labels = (y.to(model.device, dtype=torch.long) if isinstance(y, Tensor)
                  else torch.as_tensor(np.asarray(y), dtype=torch.long, device=model.device)).reshape(-1)
        if x0.ndim not in (2, 3) or len(x0) == 0 or len(labels) != len(x0):
            raise ValueError('DeepFool expects nonempty aligned (N,D) or (N,T,F) arrays')
        if not torch.isfinite(x0).all():
            raise ValueError('DeepFool input contains non-finite values')
        mask = self.victim_packet_perturbation_mask(
            x0, self.feature_names, normalization_mean=kwargs.get('normalization_mean'),
            normalization_std=kwargs.get('normalization_std'), valid_mask=kwargs.get('valid_mask'),
        ).to(x0.device, dtype=x0.dtype)
        mask *= immutable_feature_mask(x0, self.feature_names, self.immutable_features)
        lower, upper = resolve_bounds(x0, self.feature_names, self.clip_min, self.clip_max)
        saved_modes = restore_module_modes(model.model)
        current = x0.detach().clone()
        original_classes = None
        iterations = 0
        try:
            with torch.enable_grad():
                for i in range(self.max_steps):
                    candidate = current.detach().requires_grad_(True)
                    scores = class_scores(model.forward_differentiable(
                        candidate, valid_mask=kwargs.get('valid_mask')))
                    predictions = scores.argmax(dim=1)
                    if original_classes is None:
                        if torch.any((labels < 0) | (labels >= scores.shape[1])):
                            raise ValueError('Ground-truth labels outside model class range')
                        original_classes = predictions.detach().clone()
                    active = predictions.eq(original_classes) & original_classes.eq(labels)
                    if not bool(active.any()):
                        break
                    iterations += 1
                    base_scores = scores.gather(1, original_classes[:, None]).squeeze(1)
                    base_grad = torch.autograd.grad(base_scores.sum(), candidate, retain_graph=True)[0]
                    best_distance = torch.full((len(x0),), torch.inf, device=x0.device)
                    best_normal = torch.zeros_like(candidate)
                    best_margin = torch.zeros(len(x0), device=x0.device)
                    for klass in range(scores.shape[1]):
                        # The current winner cannot be a competing boundary.
                        other_grad = torch.autograd.grad(scores[:, klass].sum(), candidate,
                                                         retain_graph=klass < scores.shape[1]-1)[0]
                        normal = (other_grad - base_grad) * mask
                        margin = scores[:, klass] - base_scores
                        normal_norm = normal.reshape(len(x0), -1).norm(p=2, dim=1)
                        eligible = active & original_classes.ne(klass) & (normal_norm > self.gradient_epsilon)
                        distance = margin.abs() / normal_norm.clamp_min(self.gradient_epsilon)
                        better = eligible & (distance < best_distance)
                        best_distance = torch.where(better, distance, best_distance)
                        best_margin = torch.where(better, margin, best_margin)
                        best_normal = torch.where(better.reshape((-1,) + (1,)*(x0.ndim-1)), normal, best_normal)
                    usable = torch.isfinite(best_distance) & active
                    if not bool(usable.any()):
                        break
                    norm_sq = best_normal.reshape(len(x0), -1).square().sum(dim=1)
                    # Source hyperplane displacement plus overshoot; no benign-class preference.
                    magnitude = ((-best_margin).clamp_min(0.0) + 1e-5) / norm_sq.clamp_min(self.gradient_epsilon)
                    delta = magnitude.reshape((-1,) + (1,)*(x0.ndim-1)) * best_normal * (1 + self.overshoot)
                    proposed = torch.where(usable.reshape((-1,) + (1,)*(x0.ndim-1)), candidate + delta, candidate)
                    with torch.no_grad():
                        total = project_box_delta(x0, (proposed - x0) * mask, mask, lower, upper)
                        current = (x0 + total * mask).detach()
            with torch.no_grad():
                final_predictions = input_predictions(model.forward_differentiable(
                    current, valid_mask=kwargs.get('valid_mask')))
                if original_classes is None:
                    original_classes = final_predictions
                changed_class = final_predictions.ne(original_classes) & original_classes.eq(labels)
                success = (changed_class & final_predictions.eq(int(self.target_label))
                           if self.target_label is not None else changed_class)
                distance = (current - x0).reshape(len(x0), -1).norm(p=2, dim=1)
        finally:
            restore_saved_modes(saved_modes)
        return AttackResult(
            adversarial=current.detach(), success_mask=success.detach(),
            queries=np.zeros(len(x0), dtype=np.int64),
            intrinsic_feasible_mask=np.ones(len(x0), dtype=bool),
            feature_names=self.feature_names or None,
            metadata={'attack': self.attack_name, 'algorithm': 'untargeted_nearest_boundary',
                      'evaluation_benign_label': self.target_label,
                      'overshoot': float(self.overshoot), 'iterations_executed': iterations,
                      'mean_l2_perturbation': float(distance.mean().detach().cpu())},
        )
