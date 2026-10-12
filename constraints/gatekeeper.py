from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .rules import RuleSet, RuleViolation

@dataclass(frozen=True)
class GatekeeperResult:
    """Constraint outcome and rule-level telemetry for one generated batch."""
    accepted: np.ndarray
    realizability_rate: float
    violations: tuple[RuleViolation, ...]
    metadata: dict[str, Any]


class UniversalConstraintGatekeeper:
    """Post-generation constraint checker.

    This class never participates in attack optimization. It is intentionally a
    non-differentiable, evaluation-time gate. In strict mode, callers should filter
    rejected samples before NIDS scoring; ``validate`` still returns a mask rather
    than raising so the evaluation engine can report the realizability rate.
    """

    def __init__(
        self,
        rules: RuleSet | None = None,
        *,
        strict: bool = False,
        normalization_mean: np.ndarray | None = None,
        normalization_std: np.ndarray | None = None,
    ) -> None:
        self.rules = rules or RuleSet.default()
        self.strict = bool(strict)
        self.normalization_mean = None if normalization_mean is None else np.asarray(normalization_mean, dtype=np.float32)
        self.normalization_std = None if normalization_std is None else np.asarray(normalization_std, dtype=np.float32)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def validate(
        self,
        samples: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        valid_mask: np.ndarray | torch.Tensor | None = None,
        original_samples: np.ndarray | torch.Tensor | None = None,
    ) -> GatekeeperResult:
        """Validate generated samples after converting them to physical units.

        ``original_samples`` is the clean baseline in the same feature space.
        It enables differential threat-model rules, notably the rule that
        victim response packets must be byte-for-byte immutable for a
        non-Man-in-the-Middle attacker.
        """
        # Keep validation at float32; the rules use tolerances and do not need
        # a second full-size float64 copy of the adversarial batch.
        x = self._to_numpy(samples).astype(np.float32, copy=False)
        if x.ndim not in {2, 3}:
            raise ValueError(f"Constraints require a 2-D or 3-D batch, received shape {x.shape}")
        physical = x
        original_physical = None
        
        if original_samples is not None:
            original = self._to_numpy(original_samples).astype(np.float32, copy=False)
            if original.shape != x.shape:
                raise ValueError("original_samples must have the same shape as samples")
            original_physical = original
            
        if self.normalization_mean is not None or self.normalization_std is not None:
            if self.normalization_mean is None or self.normalization_std is None:   
                raise ValueError("normalization_mean and normalization_std must be configured together")
            expected = x.shape[-1] if x.ndim == 3 else x.shape[1]
            if len(self.normalization_mean) != expected or len(self.normalization_std) != expected:
                raise ValueError("Normalization statistics do not match sample dimensionality")
            physical = x * self.normalization_std + self.normalization_mean
            if original_physical is not None:
                original_physical = (
                    original_physical * self.normalization_std + self.normalization_mean
                )
                
        rule_features = physical
        original_rule_features = original_physical
        rule_mask = None if valid_mask is None else self._to_numpy(valid_mask)
        
        feature_count = len(feature_names)
        if physical.ndim == 2 and feature_count > 0 and physical.shape[1] > feature_count and physical.shape[1] % feature_count == 0:
            timesteps = physical.shape[1] // feature_count
            rule_features = physical.reshape(physical.shape[0], timesteps, feature_count)
            if original_physical is not None:
                original_rule_features = original_physical.reshape(
                    original_physical.shape[0], timesteps, feature_count
                )
            if rule_mask is not None and rule_mask.ndim == 2:
                rule_mask = rule_mask.astype(bool, copy=False)
            else:
                rule_mask = np.ones((physical.shape[0], timesteps), dtype=bool)

        accepted, violations = self.rules.evaluate(
            rule_features,
            feature_names,
            valid_mask=rule_mask,
            original=original_rule_features,
        )
        rate = float(accepted.mean()) if len(accepted) else 1.0
        return GatekeeperResult(
            accepted=accepted,
            realizability_rate=rate,
            violations=tuple(violations),
            metadata={
                "sample_count": int(len(accepted)),
                "violation_count": int((~accepted).sum()),
                "strict": self.strict,
            },
        )

    def filter_valid(
        self,
        samples: np.ndarray | torch.Tensor,
        result: GatekeeperResult,
    ) -> np.ndarray | torch.Tensor:
        """Return only accepted samples while preserving the input container type."""
        mask = result.accepted
        if isinstance(samples, torch.Tensor):
            return samples[torch.as_tensor(mask, device=samples.device)]
        return np.asarray(samples)[mask]

    def gate(
        self,
        samples: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[np.ndarray | torch.Tensor, GatekeeperResult]:
        """Validate and filter in one call for callers that need both outputs."""
        result = self.validate(samples, feature_names, valid_mask=valid_mask)
        return self.filter_valid(samples, result), result

    def measure_realizability(
        self,
        samples: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        return self.validate(samples, feature_names, valid_mask=valid_mask).realizability_rate
