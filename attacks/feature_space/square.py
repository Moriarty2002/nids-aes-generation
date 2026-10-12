from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from core.registry import register_attack
from attacks.base import AttackResult, FeatureSpaceAttack


@dataclass
@register_attack("square")
class SquareAttack(FeatureSpaceAttack):
    """Deterministic query-based black-box attack using model probabilities.

    Each proposal is accepted only when it improves the relevant source or
    target-class probability, which keeps query accounting explicit and avoids
    requiring gradients from the wrapped model.
    """

    epsilon: float = 0.03
    max_queries: int = 500
    p_init: float = 0.8
    targeted: bool = False
    target_label: int | None = None
    feature_names: tuple[str, ...] = ()

    attack_name = "square"
    query_based = True
    requires_gradients = False

    def _scores(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model = self._require_model()
        pred = model.predict(x)
        proba = model.predict_proba(x)
        return pred, proba

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Search locally around each sample until success or query exhaustion."""
        x0 = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
        labels = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        rng = np.random.default_rng(int(kwargs.get("seed", 0)))
        adv = x0.astype(np.float32, copy=True)
        n, d = adv.shape
        perturbation_mask = self.victim_packet_perturbation_mask(
            torch.as_tensor(x0),
            self.feature_names,
            normalization_mean=kwargs.get("normalization_mean"),
            normalization_std=kwargs.get("normalization_std"),
        ).cpu().numpy()
        queries = np.zeros(n, dtype=np.int64)
        base_pred, base_prob = self._scores(adv)
        if self.targeted and self.target_label is None:
            raise ValueError("target_label is required for targeted Square Attack")
        target_label = int(self.target_label) if self.target_label is not None else None
        target = np.full(n, target_label, dtype=labels.dtype) if self.targeted else labels
        success = (base_pred == target) if self.targeted else (base_pred != labels)
        active = ~success
        for _ in range(self.max_queries):
            if not np.any(active):
                break
            p = max(1.0 / d, self.p_init * (1.0 - _ / max(self.max_queries, 1)))
            for i in np.flatnonzero(active):
                width = max(1, int(round(p * d)))
                mutable_cols = np.flatnonzero(perturbation_mask[i])
                if not len(mutable_cols):
                    continue
                width = min(width, len(mutable_cols))
                cols = rng.choice(mutable_cols, size=width, replace=False)
                proposal = adv[i].copy()
                proposal[cols] += rng.choice((-1.0, 1.0), size=width).astype(np.float32) * self.epsilon
                proposal = np.clip(proposal, x0[i] - self.epsilon, x0[i] + self.epsilon)
                pred, prob = self._scores(proposal[None, :])
                queries[i] += 1
                if self.targeted:
                    current = base_prob[i, target[i]]
                    proposed = prob[0, target[i]]
                    accept = proposed > current
                else:
                    current = base_prob[i, labels[i]]
                    proposed = prob[0, labels[i]]
                    accept = proposed < current
                if bool(accept):
                    adv[i] = proposal
                    base_prob[i] = prob[0]
                    base_pred[i] = pred[0]
                    success[i] = bool(base_pred[i] == target[i]) if self.targeted else bool(base_pred[i] != labels[i])
                    active[i] = not success[i]
        return AttackResult(
            adversarial=adv,
            success_mask=success,
            queries=queries,
            intrinsic_feasible_mask=np.ones(n, dtype=bool),
            metadata={"epsilon": self.epsilon, "max_queries": self.max_queries, "query_budget": self.max_queries, "targeted": self.targeted},
        )
