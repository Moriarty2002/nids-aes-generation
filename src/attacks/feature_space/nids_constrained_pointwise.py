"""Constrained random initialization + Pointwise, Brockmann et al., LCN 2025 §IV-C/E.

Original mechanism: obtain a successful decision-query candidate, try reverting
changed features, then binary-search remaining numeric perturbations (Pointwise
is a minimizer and cannot initialize its own adversarial example).
Mandatory NIDS changes: all decisions use the *independently trained surrogate*;
physical float64 packets, ordered T=20 masks, victim-owned/immutable fields and
canonical forward float32 encoding constrain every proposal. The independent
post-generation gatekeeper retains final admission authority.
Unverified: original code's CICIDS flow-feature sampling law does not directly
specify a probability measure for six heterogeneous packet features. The
packet-local coordinate subset and integer lattice are declared adaptations;
continuous values use source-consistent uniform constrained initialization,
not a claim of bitwise author-code parity. The source's ten-query initial cap is a study setting, not a wire limit.
Role: per-record surrogate-decision transfer attack; flat and sequence inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from attacks.base import AttackResult, FeatureSpaceAttack
from core.dual_representation import encode_normalized32, layout, stats_views
from core.registry import register_attack
from constraints.rules import RuleSet
from models.wrapper import PyTorchModelWrapper


@dataclass
@register_attack('nids_constrained_pointwise')
class NIDSConstrainedPointwiseAttack(FeatureSpaceAttack):
    """Budgeted decision search followed by conservative sparse refinement.

    Candidates are authoritative physical float64; no inverse transformation of
    the rounded float32 clean model input occurs. Local proposal validation is
    NOT substituted for the independent, post-finalizer feasibility gate.
    """

    operates_on_physical_features = True
    hard_budget_norm = 'linf'
    query_based = True
    requires_gradients = False
    attack_name = 'nids_constrained_pointwise'

    epsilon: float = 0.03
    max_queries: int = 1000
    init_queries: int = 10
    binary_steps: int = 10
    target_label: int = 0
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()

    def _eligible_coordinates(self, original: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
        """Select only real, attacker-owned coordinates of declared mutable fields."""
        b, t, f = layout(original, self.feature_names)
        names = tuple(str(v).upper() for v in self.feature_names)
        protected = {str(x).upper() for x in self.immutable_features} | {'DIR','FLG','FEAT_PAD_MASK'}
        editable = np.array([name not in protected for name in names], dtype=bool)
        mask = np.broadcast_to(editable, (b,t,f)).copy()
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != (b,t):
                raise ValueError('Pointwise valid_mask does not match packet topology')
            mask &= valid[:,:,None]
        if 'DIR' in names:
            # Victim-owned packets cannot be changed even in editable fields.
            mask &= (~np.isclose(original.reshape(b,t,f)[...,names.index('DIR')],
                                      1., atol=1e-2, rtol=0.))[:,:,None]
        return mask.reshape(original.shape)

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Query surrogate decisions only, counting every evaluated record.

        Init failure produces an unchanged physical candidate. Pointwise acts
        only on a verified successful *surrogate* start and does not query the
        evaluation target or declare the independent target's final success.
        """
        source = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        if source.dtype != np.float64:
            raise TypeError('Pointwise physical generator requires authoritative float64 input')
        if source.ndim not in (2,3):
            raise ValueError('Pointwise expects flat or masked packet-sequence input')
        if not (0 < self.epsilon <= 1) or not (1 <= self.init_queries) or self.max_queries < 1:
            raise ValueError('Pointwise requires positive epsilon, init_queries and query limit')
        if self.binary_steps < 0:
            raise ValueError('Pointwise binary_steps must be nonnegative')
        if kwargs.get('normalization_mean') is None or kwargs.get('normalization_std') is None:
            raise ValueError('Pointwise requires the authoritative surrogate normalization domain')
        mean, std = kwargs['normalization_mean'], kwargs['normalization_std']
        _, time_steps, f = layout(source, self.feature_names)
        if time_steps != 20 or kwargs.get("valid_mask") is None:
            raise ValueError("Pointwise requires fixed T=20 and an authoritative valid_mask")
        _, sigma = stats_views(source, self.feature_names, mean, std)
        b, t, _ = layout(source, self.feature_names)
        sigma = np.broadcast_to(sigma, (b,t,f)).reshape(source.shape)
        bound = sigma * float(self.epsilon)
        valid_mask = kwargs.get('valid_mask')
        mutable = self._eligible_coordinates(source, valid_mask)
        original = source.copy()
        adv = source.copy()
        n = len(source)
        calls = np.zeros(n, dtype=np.int64)
        success = np.zeros(n, dtype=bool)
        attempted = np.zeros(n, dtype=bool)
        rng = np.random.default_rng(int(kwargs.get('seed', 0)))
        labels = np.asarray(y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y).reshape(-1)
        if labels.size != n:
            raise ValueError('Pointwise labels and physical inputs differ in batch length')
        names = tuple(str(z).upper() for z in self.feature_names)
        feature_type = np.array([z for z in names])
        mutable_flat = mutable.reshape(n,-1)
        original_flat = original.reshape(n,-1)
        adv_flat = adv.reshape(n,-1)
        bound_flat = bound.reshape(n,-1)
        # Clipping is an attack proposal restriction; authoritative lattice,
        # restoration and final canonical encoding remain owned by the finalizer.
        ranges = {str(k).upper(): tuple(v) for k,v in self.clip_ranges.items()}
        coords_type = np.tile(feature_type, t)
        source_rules = RuleSet.default()
        mask_np = np.asarray(valid_mask, dtype=bool)

        def propose(i: int, base: np.ndarray, j: int, candidate_value: float) -> np.ndarray | None:
            new = base.copy()
            key = str(coords_type[j])
            if key in ranges:
                lo,hi = ranges[key]
                candidate_value = float(np.clip(candidate_value,lo,hi))
            if key in {'PL','WIN','TTL'}:
                candidate_value = float(np.rint(candidate_value))
            if key == 'IAT' and candidate_value < 0:
                return None
            if not np.isfinite(candidate_value):
                return None
            if abs(candidate_value - original_flat[i,j]) > bound_flat[i,j] + 1e-12 * max(1.,bound_flat[i,j]):
                return None
            new[j] = candidate_value
            if new[j] == base[j]:
                return None
            return new

        def ask(i: int, proposal: np.ndarray) -> bool:
            # Use the existing physical rules to prefilter each query; this is
            # not an independent admission verdict. The final evaluator gate
            # still evaluates all generated candidates after finalization.
            candidate = proposal.reshape(1,t,f)
            reference = original_flat[i].reshape(1,t,f)
            physical_ok, _ = source_rules.evaluate(candidate, self.feature_names,
                                                    valid_mask=mask_np[i:i+1], original=reference)
            if not bool(physical_ok[0]):
                return False
            # Surrogate sees exactly the forward-encoded candidate; caller
            # counts this even if it fails or does not improve the attack.
            shaped = proposal.reshape((1,) + tuple(original.shape[1:]))
            model_input = encode_normalized32(shaped, self.feature_names, mean, std)
            model = self._require_model()
            # Each source decision evaluates one record, so a sequence surrogate
            # needs that record's exact occupancy rather than the engine batch.
            prediction = np.asarray(
                model.predict(model_input, valid_mask=mask_np[i:i+1])
                if isinstance(model, PyTorchModelWrapper) else model.predict(model_input)
            ).reshape(-1)
            if prediction.size != 1:
                raise ValueError('Pointwise surrogate returned an invalid decision batch')
            calls[i] += 1
            return int(prediction[0]) == int(self.target_label)

        # Fixed per-record query budget, including the first clean-surrogate decision.
        for i in range(n):
            baseline_decision = ask(i, original_flat[i])
            if baseline_decision:
                success[i] = True
                continue
            if labels[i] == self.target_label:
                continue  # benign is not eligible for benign-target attack
            cols = np.flatnonzero(mutable_flat[i])
            if cols.size == 0:
                continue
            # Stage 1: constrained randomized initialization (not Pointwise).
            init_limit = min(self.init_queries, self.max_queries - int(calls[i]))
            for _ in range(init_limit):
                proposal = original_flat[i].copy()
                rng.shuffle(cols)
                # Subset size is a local packet-feature sampling decision; the
                # LCN's half-range on its flow schema is not a packet-wire bound.
                count = int(rng.integers(1, cols.size + 1))
                for j in cols[:count]:
                    # Brockmann et al. initialize uniformly *within* each admissible
                    # feature interval, not only on its epsilon-ball endpoints.
                    # Integer packet attributes still pass through the wire lattice.
                    key = str(coords_type[j])
                    low = float(original_flat[i,j] - bound_flat[i,j])
                    high = float(original_flat[i,j] + bound_flat[i,j])
                    if key in ranges:
                        low = max(low, float(ranges[key][0]))
                        high = min(high, float(ranges[key][1]))
                    if key == 'IAT':
                        low = max(low, 0.)
                    if low >= high:
                        continue
                    candidate = propose(i, proposal, int(j), float(rng.uniform(low, high)))
                    if candidate is not None:
                        proposal = candidate
                if np.array_equal(proposal.view(np.uint64), original_flat[i].view(np.uint64)):
                    continue
                if ask(i, proposal):
                    adv_flat[i] = proposal
                    success[i] = True
                    break
                if calls[i] >= self.max_queries:
                    break
            if not success[i]:
                continue
            attempted[i] = True
            # Repeat Pointwise coordinate removal and magnitude reduction until
            # neither operation improves the successful candidate, or queries run out.
            # binary_steps bounds a single coordinate's numerical search; it is
            # not treated as a claim about the original paper's iteration count.
            while calls[i] < self.max_queries:
                improved = False
                different = np.flatnonzero(adv_flat[i].view('u8') != original_flat[i].view('u8'))
                rng.shuffle(different)
                for j in different:
                    if calls[i] >= self.max_queries:
                        break
                    proposal = adv_flat[i].copy()
                    proposal[j] = original_flat[i,j]
                    if ask(i, proposal):
                        adv_flat[i] = proposal
                        improved = True
                different = np.flatnonzero(adv_flat[i].view('u8') != original_flat[i].view('u8'))
                rng.shuffle(different)
                for j in different:
                    if calls[i] >= self.max_queries:
                        break
                    best = float(adv_flat[i,j])
                    closest = float(original_flat[i,j])
                    for _ in range(self.binary_steps):
                        if calls[i] >= self.max_queries:
                            break
                        candidate = propose(i, adv_flat[i], int(j), (best + closest) / 2.)
                        if candidate is None:
                            break  # precision limit or physical lattice
                        if ask(i, candidate):
                            adv_flat[i] = candidate
                            best = float(candidate[j])
                            improved = True
                        else:
                            closest = float(candidate[j])
                if not improved:
                    break
        return AttackResult(
            adversarial=adv,
            success_mask=success,
            queries=calls,
            intrinsic_feasible_mask=np.ones(n, dtype=bool),
            metadata={
                'method': 'constrained_random_init_plus_pointwise',
                'initialization_cap': int(self.init_queries),
                'initialization_distribution': 'uniform_physical_and_native_epsilon_intersection',
                'total_query_budget': int(self.max_queries),
                'targeted': True,
                'target_label': int(self.target_label),
                'pointwise_refinement_attempted': bool(np.any(attempted)),
                'initialization_success_count': int(np.count_nonzero(attempted)),
                'source_oracle': 'independently_pretrained_surrogate',
                'physical_proposal_topology': 'flat' if source.ndim == 2 else 'sequence',
            },
        )
