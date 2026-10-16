from __future__ import annotations

"""Batched Jacobian-based Saliency Map Attack (JSMA) for NIDS features.

Conceptual basis: Papernot et al., "The Limitations of Deep Learning in
Adversarial Settings," IEEE European Symposium on Security and Privacy (2016).
This is an independent PyTorch adaptation for tabular and packet-sequence NIDS
inputs; no original repository implementation is copied.

Unlike image-focused JSMA implementations, this version works in the
framework's normalized feature space, supports both (N,D) and (N,T,F) inputs,
and uses a strict endpoint threat model. DIR is always immutable, all features
on victim-sent packets are frozen, and configured immutable fields are removed
from saliency before a feature or feature pair can be selected. Physical and
RFC/protocol checks remain in EvaluationEngine after generation.

The feature-pair search is tensorized: there are no Python loops over feature
pairs. Jacobians are accumulated per class using batched output sums (valid for
sample-independent models in eval mode), while ``jacobian_batch_size`` bounds
the temporary B x D x D saliency matrices for high-dimensional packet flows.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

import numpy as np
import torch
from torch import Tensor

from attacks.base import AttackResult, FeatureSpaceAttack
from core.exceptions import IncompatibleThreatModelError
from core.registry import register_attack
from models.wrapper import PyTorchModelWrapper, ensure_gradient_compatible
from ._linf_utils import immutable_feature_mask, resolve_target, validate_target_label
from ._optimization_utils import (
    input_predictions,
    resolve_bounds,
    restore_module_modes,
    restore_saved_modes,
)


Direction = Literal["auto", "increase", "decrease"]


def _saliency_scores(logits: Tensor) -> Tensor:
    """Return class scores with useful input gradients for one-logit AD models.

    For a one-logit binary classifier, [0,z] is sufficient for argmax but its
    class-0 score has a zero gradient. The symmetric scores [-z/2,+z/2] have
    exactly the same decision boundary and predictions, while giving both
    classes meaningful Jacobians for targeted Benign (class 0) saliency.
    """
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
        binary_logit = logits.reshape(-1)
        return torch.stack((-0.5 * binary_logit, 0.5 * binary_logit), dim=1)
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError(
            "JSMA requires logits shaped (batch, classes) or one binary logit per sample; "
            f"received {tuple(logits.shape)}"
        )
    return logits


def _broadcast_bound(value: float | Tensor, x: Tensor) -> Tensor:
    """Expand a resolved scalar/feature bound to the exact input shape."""
    tensor = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    try:
        return torch.broadcast_to(tensor, x.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Bound shape {tuple(tensor.shape)} cannot broadcast to input shape {tuple(x.shape)}"
        ) from exc


@dataclass
@register_attack("jsma")
class JSMAAttack(FeatureSpaceAttack):
    """Targeted, L0-oriented JSMA for normalized NIDS feature vectors.

    ``theta`` is the magnitude of one feature change in normalized feature
    units. ``direction='auto'`` chooses an increase or decrease independently
    for each selected pair from the target/non-target Jacobian signs. Automatic
    direction is useful for standardized NIDS features, which commonly include
    negative values and do not share the [0,1] domain assumed by many image
    implementations.

    By default at most ``ceil(D * max_perturbation_ratio)`` distinct features
    are selected per flow. Each feature is selected at most once, so the
    selected-feature budget is a transparent upper bound on L0. If one feature
    remains in the budget, a vectorized single-feature saliency fallback handles
    that final coordinate. Pair scoring itself is fully vectorized over D x D.
    """

    theta: float = 0.1
    max_perturbed_features: int | None = None
    max_perturbation_ratio: float = 0.1
    max_iterations: int | None = None
    jacobian_batch_size: int = 4
    direction: Direction = "auto"
    targeted: bool = False
    target_label: int | None = None
    clip_min: float | Sequence[float] | np.ndarray | Tensor | None = None
    clip_max: float | Sequence[float] | np.ndarray | Tensor | None = None
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()
    saliency_epsilon: float = 1e-12

    attack_name = "jsma"
    requires_gradients = True
    query_based = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.theta) or self.theta <= 0:
            raise ValueError("theta must be finite and > 0")
        if not np.isfinite(self.max_perturbation_ratio) or not 0.0 <= self.max_perturbation_ratio <= 1.0:
            raise ValueError("max_perturbation_ratio must be finite and in [0, 1]")
        if self.max_perturbed_features is not None and self.max_perturbed_features < 0:
            raise ValueError("max_perturbed_features must be >= 0")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1 when specified")
        if self.jacobian_batch_size < 1:
            raise ValueError("jacobian_batch_size must be >= 1")
        if self.direction not in {"auto", "increase", "decrease"}:
            raise ValueError("direction must be 'auto', 'increase', or 'decrease'")
        if not np.isfinite(self.saliency_epsilon) or self.saliency_epsilon <= 0:
            raise ValueError("saliency_epsilon must be finite and > 0")
        self.feature_names = tuple(str(name) for name in self.feature_names)
        self.immutable_features = tuple(str(name) for name in self.immutable_features)
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must not contain duplicates")
        if self.target_label is not None:
            self.targeted = True

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Generate a masked JSMA batch on the bound PyTorch model's device."""
        model_ref = self._require_model()
        # Also accept the same wrapper imported via the package-qualified name.
        # Direct execution uses ``models.wrapper`` while package usage may load
        # ``models.wrapper``; module aliasing can make Python's
        # nominal isinstance check false even though the wrapper contract matches.
        capabilities = getattr(model_ref, "capabilities", None)
        compatible_wrapper = (
            getattr(capabilities, "model_family", None) == "pytorch"
            and hasattr(model_ref, "model")
            and hasattr(model_ref, "device")
        )
        if not isinstance(model_ref, PyTorchModelWrapper) and not compatible_wrapper:
            raise IncompatibleThreatModelError("JSMA requires a PyTorchModelWrapper")
        model = cast(PyTorchModelWrapper, model_ref)

        # Required per-batch placement: inputs arriving as CPU NumPy arrays or
        # CPU tensors are moved only for this engine-provided batch.
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

        ensure_gradient_compatible(model)
        if x0.ndim not in (2, 3) or x0.shape[0] == 0:
            raise ValueError(f"JSMA expects a non-empty (N,D) or (N,T,F) batch, got {tuple(x0.shape)}")
        if labels.numel() != len(x0):
            raise ValueError("x and y must contain the same number of samples")
        if not torch.isfinite(x0).all():
            raise ValueError("JSMA input contains NaN or infinite values")

        effective_targeted, target_label = resolve_target(self.target_label, self.targeted, dict(kwargs))

        # Strict required call from FeatureSpaceAttack. The returned mask freezes
        # DIR itself and every feature of each victim-sent packet (DIR == 1).
        mask = self.victim_packet_perturbation_mask(
            x0,
            self.feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
        ).to(device=x0.device, dtype=x0.dtype)
        mask = mask * immutable_feature_mask(x0, self.feature_names, self.immutable_features)
        mask_flat = mask.reshape(len(x0), -1).bool()

        lower_resolved, upper_resolved = resolve_bounds(
            x0, self.feature_names, self.clip_min, self.clip_max
        )
        lower_flat = _broadcast_bound(lower_resolved, x0).reshape(len(x0), -1)
        upper_flat = _broadcast_bound(upper_resolved, x0).reshape(len(x0), -1)

        batch_size = int(x0.shape[0])
        flat_dim = int(x0[0].numel())
        if self.max_perturbed_features is None:
            l0_budget = int(np.ceil(flat_dim * float(self.max_perturbation_ratio)))
        else:
            l0_budget = min(int(self.max_perturbed_features), flat_dim)
        l0_budget = min(l0_budget, flat_dim)
        if l0_budget == 0:
            original_modes = restore_module_modes(model.model)
            try:
                with torch.no_grad():
                    clean_logits = model.model(x0)
                    clean_scores = _saliency_scores(clean_logits)
                    if torch.any((labels < 0) | (labels >= clean_scores.shape[1])):
                        raise ValueError("Ground-truth labels are outside the model's class range")
                    if effective_targeted:
                        assert target_label is not None
                        validate_target_label(clean_logits, int(target_label))
            finally:
                restore_saved_modes(original_modes)
            return AttackResult(
                adversarial=x0.detach().clone(),
                # No perturbation means no attack-induced success, even if the
                # clean model had already misclassified a sample as the target.
                success_mask=torch.zeros(batch_size, dtype=torch.bool, device=x0.device),
                queries=np.zeros(batch_size, dtype=np.int64),
                intrinsic_feasible_mask=np.ones(batch_size, dtype=bool),
                feature_names=self.feature_names or None,
                metadata={
                    "attack": self.attack_name,
                    "targeted": bool(effective_targeted),
                    "target_label": target_label,
                    "l0_budget": 0,
                    "mean_l0_perturbation": 0.0,
                    "iterations_executed": 0,
                },
            )

        model_module = model.model
        saved_modes = restore_module_modes(model_module)
        adversarial = x0.detach().clone()
        clean_predictions = torch.empty(batch_size, dtype=torch.long, device=x0.device)
        target_classes = torch.empty_like(labels)
        iteration_count = 0
        features_selected_total = torch.zeros(batch_size, dtype=torch.long, device=x0.device)
        try:
            # JSMA needs gradients even when the caller wraps evaluation in
            # torch.no_grad(). Eval mode fixes BatchNorm and disables Dropout.
            with torch.enable_grad():
                with torch.no_grad():
                    clean_logits = model_module(x0)
                    clean_scores = _saliency_scores(clean_logits)
                    clean_predictions = input_predictions(clean_logits)
                    class_count = int(clean_scores.shape[1])
                    if torch.any((labels < 0) | (labels >= class_count)):
                        raise ValueError("Ground-truth labels are outside the model's class range")
                    if effective_targeted:
                        assert target_label is not None
                        validate_target_label(clean_logits, int(target_label))
                        target_classes.fill_(int(target_label))
                    else:
                        # Untargeted fallback: target the strongest non-ground-truth
                        # class. NIDS workflow normally supplies Benign explicitly.
                        alternatives = clean_scores.clone()
                        alternatives.scatter_(1, labels[:, None], -torch.inf)
                        target_classes.copy_(alternatives.argmax(dim=1))

                clean_flat = x0.reshape(batch_size, flat_dim)
                adv_flat_all = adversarial.reshape(batch_size, flat_dim)
                triangular = torch.triu(
                    torch.ones((flat_dim, flat_dim), dtype=torch.bool, device=x0.device),
                    diagonal=1,
                )

                for batch_start in range(0, batch_size, self.jacobian_batch_size):
                    batch_stop = min(batch_start + self.jacobian_batch_size, batch_size)
                    clean_chunk = clean_flat[batch_start:batch_stop]
                    mask_chunk = mask_flat[batch_start:batch_stop]
                    lower_chunk = lower_flat[batch_start:batch_stop]
                    upper_chunk = upper_flat[batch_start:batch_stop]
                    labels_chunk = labels[batch_start:batch_stop]
                    target_chunk = target_classes[batch_start:batch_stop]
                    clean_pred_chunk = clean_predictions[batch_start:batch_stop]
                    chunk_size = int(len(clean_chunk))
                    adv_chunk = clean_chunk.detach().clone()
                    selected = torch.zeros_like(mask_chunk, dtype=torch.bool)

                    if effective_targeted:
                        assert target_label is not None
                        eligible = clean_pred_chunk.eq(labels_chunk) & labels_chunk.ne(int(target_label))
                    else:
                        eligible = clean_pred_chunk.eq(labels_chunk)
                    stalled = torch.zeros(chunk_size, dtype=torch.bool, device=x0.device)

                    step_limit = (l0_budget + 1) // 2
                    if self.max_iterations is not None:
                        step_limit = min(step_limit, int(self.max_iterations))

                    for _ in range(step_limit):
                        current = adv_chunk.reshape(chunk_size, *x0.shape[1:]).detach().requires_grad_(True)
                        logits = model_module(current)
                        scores = _saliency_scores(logits)
                        current_predictions = input_predictions(logits)
                        if effective_targeted:
                            assert target_label is not None
                            active = eligible & current_predictions.ne(int(target_label)) & ~stalled
                        else:
                            active = eligible & current_predictions.eq(labels_chunk) & ~stalled
                        if not bool(active.any()):
                            break

                        # Compute the Jacobian for every class in batched form.
                        # The only Python loop is over classes, never over feature
                        # coordinates or feature pairs. Summing each class's
                        # outputs gives the per-example input gradients for a
                        # sample-independent model in eval mode.
                        jacobian_by_class: list[Tensor | None] = [None] * class_count
                        graph_classes = [
                            class_idx
                            for class_idx in range(class_count)
                            if scores[:, class_idx].requires_grad
                        ]
                        for position, class_idx in enumerate(graph_classes):
                            gradient_result = torch.autograd.grad(
                                scores[:, class_idx].sum(),
                                current,
                                retain_graph=position < len(graph_classes) - 1,
                                create_graph=False,
                                only_inputs=True,
                                allow_unused=True,
                            )[0]
                            if gradient_result is not None:
                                jacobian_by_class[class_idx] = gradient_result.reshape(chunk_size, flat_dim)
                        zero_gradient = torch.zeros(
                            (chunk_size, flat_dim), dtype=x0.dtype, device=x0.device
                        )
                        jacobian = torch.stack(
                            [zero_gradient if gradient is None else gradient for gradient in jacobian_by_class],
                            dim=1,
                        )

                        row_idx = torch.arange(chunk_size, device=x0.device)
                        target_gradient = jacobian[row_idx, target_chunk] * mask_chunk.to(x0.dtype)
                        other_gradient = (jacobian.sum(dim=1) - jacobian[row_idx, target_chunk]) * mask_chunk.to(x0.dtype)

                        current_flat = current.reshape(chunk_size, flat_dim).detach()
                        available = mask_chunk & ~selected
                        can_increase = current_flat < (upper_chunk - self.saliency_epsilon)
                        can_decrease = current_flat > (lower_chunk + self.saliency_epsilon)

                        # Feature-pair saliency is calculated on D x D matrices.
                        # The upper triangle prevents duplicate (i,j)/(j,i) pairs
                        # and excludes i == j, without a Python loop over pairs.
                        alpha = target_gradient.unsqueeze(2) + target_gradient.unsqueeze(1)
                        beta = other_gradient.unsqueeze(2) + other_gradient.unsqueeze(1)
                        valid_pair = (
                            available.unsqueeze(2)
                            & available.unsqueeze(1)
                            & triangular.unsqueeze(0)
                            & active.view(-1, 1, 1)
                        )
                        if self.direction in {"auto", "increase"}:
                            valid_inc = (
                                valid_pair
                                & can_increase.unsqueeze(2)
                                & can_increase.unsqueeze(1)
                                & (alpha > 0)
                                & (beta < 0)
                            )
                            inc_saliency = torch.where(valid_inc, alpha * (-beta), 0.0)
                        else:
                            inc_saliency = torch.zeros_like(alpha)

                        if self.direction in {"auto", "decrease"}:
                            valid_dec = (
                                valid_pair
                                & can_decrease.unsqueeze(2)
                                & can_decrease.unsqueeze(1)
                                & (alpha < 0)
                                & (beta > 0)
                            )
                            dec_saliency = torch.where(valid_dec, (-alpha) * beta, 0.0)
                        else:
                            dec_saliency = torch.zeros_like(alpha)

                        inc_best, inc_flat_index = inc_saliency.reshape(chunk_size, -1).max(dim=1)
                        dec_best, dec_flat_index = dec_saliency.reshape(chunk_size, -1).max(dim=1)
                        choose_decrease = dec_best > inc_best
                        pair_saliency = torch.maximum(inc_best, dec_best)
                        pair_flat_index = torch.where(choose_decrease, dec_flat_index, inc_flat_index)
                        has_pair = active & (features_selected_total[batch_start:batch_stop] < l0_budget) & (pair_saliency > self.saliency_epsilon)
                        remaining = l0_budget - features_selected_total[batch_start:batch_stop]
                        has_pair &= remaining >= 2

                        # If exactly one coordinate remains in the L0 budget,
                        # use the same derivative sign rules with a vector saliency map.
                        valid_single_inc = (
                            available & can_increase & active.unsqueeze(1) & (target_gradient > 0) & (other_gradient < 0)
                        ) if self.direction in {"auto", "increase"} else torch.zeros_like(available)
                        valid_single_dec = (
                            available & can_decrease & active.unsqueeze(1) & (target_gradient < 0) & (other_gradient > 0)
                        ) if self.direction in {"auto", "decrease"} else torch.zeros_like(available)
                        single_inc = torch.where(valid_single_inc, target_gradient * (-other_gradient), 0.0)
                        single_dec = torch.where(valid_single_dec, (-target_gradient) * other_gradient, 0.0)
                        single_inc_best, single_inc_index = single_inc.max(dim=1)
                        single_dec_best, single_dec_index = single_dec.max(dim=1)
                        use_single_decrease = single_dec_best > single_inc_best
                        single_best = torch.maximum(single_inc_best, single_dec_best)
                        single_index = torch.where(use_single_decrease, single_dec_index, single_inc_index)
                        has_single = active & (remaining == 1) & (single_best > self.saliency_epsilon)

                        has_candidate = has_pair | has_single
                        stalled |= active & ~has_candidate
                        if not bool(has_candidate.any()):
                            break

                        # Apply the chosen updates. The delta is multiplied by the
                        # endpoint edit mask both at update time and after clipping.
                        with torch.no_grad():
                            current_delta = (adv_chunk - clean_chunk) * mask_chunk
                            update_rows = torch.arange(chunk_size, device=x0.device)
                            if bool(has_pair.any()):
                                pair_rows = update_rows[has_pair]
                                pair_indices = pair_flat_index[has_pair]
                                first = torch.div(pair_indices, flat_dim, rounding_mode="floor")
                                second = pair_indices.remainder(flat_dim)
                                pair_direction = torch.where(
                                    choose_decrease[has_pair],
                                    torch.full_like(pair_indices, -1, dtype=x0.dtype),
                                    torch.ones_like(pair_indices, dtype=x0.dtype),
                                ) * float(self.theta)
                                current_delta[pair_rows, first] += pair_direction
                                current_delta[pair_rows, second] += pair_direction
                                selected[pair_rows, first] = True
                                selected[pair_rows, second] = True
                            if bool(has_single.any()):
                                single_rows = update_rows[has_single]
                                single_direction = torch.where(
                                    use_single_decrease[has_single],
                                    torch.full_like(single_index[has_single], -1, dtype=x0.dtype),
                                    torch.ones_like(single_index[has_single], dtype=x0.dtype),
                                ) * float(self.theta)
                                current_delta[single_rows, single_index[has_single]] += single_direction
                                selected[single_rows, single_index[has_single]] = True

                            current_delta = current_delta * mask_chunk
                            candidate_flat = clean_chunk + current_delta
                            clipped_flat = torch.maximum(torch.minimum(candidate_flat, upper_chunk), lower_chunk)
                            # Only selected coordinates are box-clipped. Untouched
                            # clean inputs remain unchanged even if an input already
                            # falls outside an optional box bound.
                            adv_chunk = torch.where(selected & mask_chunk, clipped_flat, clean_chunk)
                            current_delta = (adv_chunk - clean_chunk) * mask_chunk
                            adv_chunk = (clean_chunk + current_delta).detach()
                            features_selected_total[batch_start:batch_stop] = selected.sum(dim=1)
                        logits = logits.detach()
                        scores = scores.detach()
                        current_predictions = current_predictions.detach()
                        jacobian = jacobian.detach()
                        target_gradient = target_gradient.detach()
                        other_gradient = other_gradient.detach()
                        del (
                            logits,
                            scores,
                            current_predictions,
                            jacobian,
                            target_gradient,
                            other_gradient,
                            alpha,
                            beta,
                            inc_saliency,
                            dec_saliency,
                            current,
                        )
                        iteration_count += 1

                    adv_flat_all[batch_start:batch_stop] = adv_chunk

            with torch.no_grad():
                final_logits = model_module(adversarial)
                final_predictions = input_predictions(final_logits)
                if effective_targeted:
                    assert target_label is not None
                    success = (
                        clean_predictions.eq(labels)
                        & labels.ne(int(target_label))
                        & final_predictions.eq(int(target_label))
                    )
                else:
                    success = clean_predictions.eq(labels) & final_predictions.ne(labels)
                delta = (adversarial - x0).reshape(batch_size, -1)
                actual_l0 = (delta.abs() > self.saliency_epsilon).sum(dim=1)
                mean_l0 = float(actual_l0.float().mean().detach().cpu()) if batch_size else 0.0
                final_logits = final_logits.detach()
                final_predictions = final_predictions.detach()
                del final_logits, final_predictions, delta
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
                "targeted": bool(effective_targeted),
                "target_label": target_label,
                "theta": float(self.theta),
                "direction": self.direction,
                "l0_budget": int(l0_budget),
                "max_iterations": self.max_iterations,
                "jacobian_batch_size": int(self.jacobian_batch_size),
                "iterations_executed": int(iteration_count),
                "mean_l0_perturbation": mean_l0,
                "successful_sample_count": int(success.sum().detach().cpu()),
                "saliency_pair_search": "vectorized_upper_triangle",
            },
        )
