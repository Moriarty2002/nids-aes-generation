from __future__ import annotations

"""Carlini-Wagner L2 feature-space attack for network intrusion detection.

Adapted conceptually from the objective in Carlini & Wagner, "Towards Evaluating
the Robustness of Neural Networks" (IEEE S&P 2017). Original authors' code:
https://github.com/carlini/nn_robust_attacks. This is an independent PyTorch
reimplementation, not copied repository code: it uses direct Adam optimization
of a masked perturbation delta with box clipping for standardized tabular data
instead of the tanh change-of-variables commonly used for image inputs.
Protocol/RFC constraints remain exclusively in the post-generation gatekeeper.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from attacks.base import AttackResult, FeatureSpaceAttack
from core.exceptions import IncompatibleThreatModelError
from core.registry import register_attack
from models.wrapper import PyTorchModelWrapper, ensure_gradient_compatible
from ._linf_utils import immutable_feature_mask, resolve_target, validate_target_label
from ._optimization_utils import class_scores, input_predictions, project_box_delta, resolve_bounds, restore_module_modes, restore_saved_modes


@dataclass
@register_attack("cw")
class CWL2Attack(FeatureSpaceAttack):
    """Targeted C&W L2 with per-sample binary search over the trade-off constant.

    It minimizes ``||delta||_2^2 + c * margin`` where targeted margin is
    ``max(max_{i != target} Z_i - Z_target + confidence, 0)``. Unlike L-infinity
    sign attacks, both the squared norm and margin use per-sample sums; no
    ``reduction='mean'`` loss scales the effective objective with batch size.

    Adam updates a delta parameter directly. Every update is projected through
    the configured feature box and the mandatory victim-packet mask. Best-success
    tracking returns the lowest-L2 successful example found per flow, even when
    later optimizer steps move away from the target class.
    """

    target_label: int | None = None
    targeted: bool = True
    confidence: float = 0.0
    learning_rate: float = 1e-2
    max_steps: int = 1000
    binary_search_steps: int = 5
    initial_const: float = 1e-2
    abort_early: bool = True
    abort_check_every: int = 10
    clip_min: float | Sequence[float] | np.ndarray | Tensor | None = None
    clip_max: float | Sequence[float] | np.ndarray | Tensor | None = None
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()

    attack_name = "cw"
    requires_gradients = True
    query_based = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.confidence) or self.confidence < 0:
            raise ValueError("confidence must be finite and >= 0")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be an integer >= 1")
        if not isinstance(self.binary_search_steps, int) or self.binary_search_steps < 1:
            raise ValueError("binary_search_steps must be an integer >= 1")
        if not np.isfinite(self.initial_const) or self.initial_const <= 0:
            raise ValueError("initial_const must be finite and > 0")
        if not isinstance(self.abort_check_every, int) or self.abort_check_every < 1:
            raise ValueError("abort_check_every must be an integer >= 1")
        self.feature_names = tuple(str(name) for name in self.feature_names)
        self.immutable_features = tuple(str(name) for name in self.immutable_features)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")
        if self.target_label is not None:
            self.targeted = True

    @staticmethod
    def _margin_loss(
        scores: Tensor,
        labels: Tensor,
        *,
        targeted: bool,
        target_label: int | None,
        confidence: float,
    ) -> Tensor:
        """Compute the standard non-mean-reduced C&W hinge margin per sample."""
        batch, num_classes = scores.shape
        if targeted:
            if target_label is None or not 0 <= int(target_label) < num_classes:
                raise ValueError(f"target_label must be in [0, {num_classes - 1}]")
            real_labels = torch.full((batch,), int(target_label), device=scores.device, dtype=torch.long)
            real_scores = scores.gather(1, real_labels[:, None]).squeeze(1)
            other_scores = scores.masked_fill(
                torch.nn.functional.one_hot(real_labels, num_classes=num_classes).bool(), -torch.inf
            ).max(dim=1).values
            return (other_scores - real_scores + float(confidence)).clamp_min(0.0)

        real_labels = labels.long().reshape(-1)
        if torch.any((real_labels < 0) | (real_labels >= num_classes)):
            raise ValueError("Ground-truth labels are outside the model's class range")
        real_scores = scores.gather(1, real_labels[:, None]).squeeze(1)
        other_scores = scores.masked_fill(
            torch.nn.functional.one_hot(real_labels, num_classes=num_classes).bool(), -torch.inf
        ).max(dim=1).values
        return (real_scores - other_scores + float(confidence)).clamp_min(0.0)

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        model = self._require_model()
        ensure_gradient_compatible(model)
        if not isinstance(model, PyTorchModelWrapper):
            raise IncompatibleThreatModelError("C&W L2 requires a PyTorchModelWrapper")

        # Allocate only the current CPU-bound batch on the model's device.
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
            raise ValueError(f"C&W expects a non-empty 2-D or 3-D batch, got {tuple(x0.shape)}")
        if labels.numel() != x0.shape[0]:
            raise ValueError("x and y must contain the same number of samples")
        if not torch.isfinite(x0).all():
            raise ValueError("C&W input contains NaN or infinite values")

        targeted, target_label = resolve_target(self.target_label, self.targeted, dict(kwargs))
        if targeted and target_label is None:
            raise ValueError("C&W targeted optimization requires target_label")

        # Exact directional mask call required by the framework threat model.
        mask = self.victim_packet_perturbation_mask(
            x0,
            self.feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
        ).to(device=x0.device, dtype=x0.dtype)
        mask = mask * immutable_feature_mask(x0, self.feature_names, self.immutable_features)
        lower, upper = resolve_bounds(x0, self.feature_names, self.clip_min, self.clip_max)

        batch_size = int(x0.shape[0])
        constants = torch.full((batch_size,), float(self.initial_const), dtype=x0.dtype, device=x0.device)
        lower_const = torch.zeros_like(constants)
        upper_const = torch.full_like(constants, 1e10)
        best_l2_sq = torch.full_like(constants, torch.inf)
        best_adversarial = x0.detach().clone()
        fallback_adversarial = x0.detach().clone()
        total_iterations = 0
        model_module = model.model
        saved_modes = restore_module_modes(model_module)

        try:
            # Some callers wrap the engine in no_grad; C&W still needs input gradients.
            with torch.enable_grad():
                for search_step in range(self.binary_search_steps):
                    delta = nn.Parameter(torch.zeros_like(x0))
                    optimizer = torch.optim.Adam([delta], lr=float(self.learning_rate))
                    round_success = torch.zeros(batch_size, dtype=torch.bool, device=x0.device)
                    previous_loss = float("inf")

                    for step in range(self.max_steps):
                        optimizer.zero_grad(set_to_none=True)
                        applied_delta = project_box_delta(x0, delta, mask, lower, upper)
                        adversarial = x0 + applied_delta
                        logits = model_module(adversarial)
                        scores = class_scores(logits)
                        if search_step == 0 and step == 0:
                            if targeted:
                                assert target_label is not None
                                validate_target_label(logits, int(target_label))
                            if torch.any((labels < 0) | (labels >= scores.shape[1])):
                                raise ValueError("Ground-truth labels are outside the model's class range")

                        margin = self._margin_loss(
                            scores,
                            labels,
                            targeted=targeted,
                            target_label=target_label,
                            confidence=float(self.confidence),
                        )
                        per_sample_l2_sq = applied_delta.reshape(batch_size, -1).square().sum(dim=1)
                        # SUM is essential here: no batch-size-dependent mean loss.
                        per_sample_objective = per_sample_l2_sq + constants * margin
                        objective = per_sample_objective.sum()

                        predictions = input_predictions(logits)
                        successful = (
                            predictions.eq(int(target_label)) & labels.ne(int(target_label))
                            if targeted
                            else predictions.ne(labels)
                        )
                        total_iterations += 1
                        with torch.no_grad():
                            round_success |= successful
                            improved = successful & (per_sample_l2_sq < best_l2_sq)
                            if bool(improved.any()):
                                best_l2_sq[improved] = per_sample_l2_sq.detach()[improved]
                                best_adversarial[improved] = adversarial.detach()[improved]
                            fallback_adversarial = adversarial.detach()

                        gradient = torch.autograd.grad(
                            objective,
                            delta,
                            only_inputs=True,
                            retain_graph=False,
                            create_graph=False,
                        )[0]
                        delta.grad = gradient
                        optimizer.step()
                        # Project the parameter itself after every update. This prevents
                        # optimizer momentum from accumulating edits to frozen features.
                        with torch.no_grad():
                            delta.copy_(project_box_delta(x0, delta, mask, lower, upper))

                        if self.abort_early and (step + 1) % self.abort_check_every == 0:
                            current_loss = float(objective.detach().cpu())
                            if current_loss > previous_loss * 0.9999:
                                break
                            previous_loss = current_loss

                    # Evaluate the post-optimizer state too, so the final Adam update
                    # in each binary-search round is not silently skipped.
                    with torch.no_grad():
                        final_delta = project_box_delta(x0, delta, mask, lower, upper)
                        final_adversarial = x0 + final_delta
                        final_logits = model_module(final_adversarial)
                        final_predictions = input_predictions(final_logits)
                        final_success = (
                            final_predictions.eq(int(target_label)) & labels.ne(int(target_label))
                            if targeted
                            else final_predictions.ne(labels)
                        )
                        final_l2_sq = final_delta.reshape(batch_size, -1).square().sum(dim=1)
                        round_success |= final_success
                        improved = final_success & (final_l2_sq < best_l2_sq)
                        if bool(improved.any()):
                            best_l2_sq[improved] = final_l2_sq[improved]
                            best_adversarial[improved] = final_adversarial[improved]
                        fallback_adversarial = final_adversarial.detach()

                    if search_step + 1 < self.binary_search_steps:
                        upper_const = torch.where(round_success, torch.minimum(upper_const, constants), upper_const)
                        lower_const = torch.where(round_success, lower_const, torch.maximum(lower_const, constants))
                        has_upper = upper_const < 1e9
                        constants = torch.where(
                            has_upper,
                            (lower_const + upper_const) / 2.0,
                            constants * 10.0,
                        )
                # Prefer the smallest-L2 successful candidate for each sample.
                # Failed samples retain the latest optimizer candidate so the
                # evaluator can still report their target-model outcome.
                best_success = torch.isfinite(best_l2_sq)
                adversarial = fallback_adversarial.detach().clone()
                adversarial[best_success] = best_adversarial[best_success]
                with torch.no_grad():
                    final_logits = model_module(adversarial)
                    final_predictions = input_predictions(final_logits)
                    success = (
                        final_predictions.eq(int(target_label)) & labels.ne(int(target_label))
                        if targeted and target_label is not None
                        else final_predictions.ne(labels)
                    )
                    final_delta = adversarial - x0
                    final_l2 = final_delta.reshape(batch_size, -1).norm(p=2, dim=1)
        finally:
            restore_saved_modes(saved_modes)

        return AttackResult(
            adversarial=adversarial.detach(),
            success_mask=success.detach(),
            queries=np.zeros(batch_size, dtype=np.int64),
            intrinsic_feasible_mask=np.ones(batch_size, dtype=bool),
            feature_names=self.feature_names or None,
            metadata={
                "attack": self.attack_name,
                "targeted": bool(targeted),
                "target_label": target_label,
                "confidence": float(self.confidence),
                "learning_rate": float(self.learning_rate),
                "max_steps": int(self.max_steps),
                "binary_search_steps": int(self.binary_search_steps),
                "initial_const": float(self.initial_const),
                "successful_sample_count": int(success.sum().item()),
                "mean_l2_perturbation": float(final_l2.mean().detach().cpu()),
                "total_optimizer_steps": int(total_iterations),
                "margin_reduction": "sum",
            },
        )
