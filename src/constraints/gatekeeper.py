from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from src.core.dual_representation import *

from .rules import RuleSet, RuleViolation, DirectionalImmutabilityRule


@dataclass(frozen=True)
class GatekeeperResult:
    """Independent result of exact physical-space constraint validation."""
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
    
    Never estimates physical truth from normalized float32 data. The caller
    must supply a float64 authoritative candidate and clean reference.
    """

    def __init__(
        self,
        rules: RuleSet | None = None,
        *,
        strict: bool = False,
        normalization_mean: np.ndarray | None = None,
        normalization_std: np.ndarray | None = None,
    ) -> None:
        self.rules = rules if rules is not None else RuleSet.default()
        self.strict = bool(strict)
        self.normalization_mean = None if normalization_mean is None else np.asarray(normalization_mean, dtype=np.float64)
        self.normalization_std = None if normalization_std is None else np.asarray(normalization_std, dtype=np.float64)
        
        if (self.normalization_mean is None) != (self.normalization_std is None):
            raise ValueError("normalization_mean and normalization_std must be supplied together")

    @staticmethod
    def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def validate(
        self,
        samples: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        authoritative_physical: np.ndarray | torch.Tensor,
        original_physical: np.ndarray | torch.Tensor | None = None,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> GatekeeperResult:
        """Check exact float64 physical values and their target float32 encoding.

        Mismatched physical/model pairs fail closed with RuntimeError, not a
        softer per-flow constraint violation. Both physical inputs are read-only.
        """
        normalized = self._to_numpy(samples)
        physical = self._to_numpy(authoritative_physical)
        if normalized.dtype != np.float32:
            raise TypeError("Target model input must be normalized float32")
        if physical.dtype != np.float64:
            raise TypeError("authoritative_physical must be float64")
        if physical.shape != normalized.shape:
            raise ValueError("Physical/model batch shape mismatch")
        b, t, f = layout(physical, feature_names)
        if not np.all(np.isfinite(physical)) or not np.all(np.isfinite(normalized)):
            raise ValueError("Physical and normalized model values must be finite")

        clean = None
        if original_physical is not None:
            clean = self._to_numpy(original_physical)
            if clean.dtype != np.float64 or clean.shape != physical.shape:
                raise TypeError("original_physical must be matching float64 data")
            if not np.all(np.isfinite(clean)):
                raise ValueError("Original physical values must be finite")
        elif "DIR" in feature_names and any(
            isinstance(rule, DirectionalImmutabilityRule) for rule in self.rules.rules
        ):
            raise ValueError("original_physical is mandatory for directional immutability")

        # Exact equality of IEEE-754 float32 bit patterns, including signed 0.
        # Encoding uses shared canonical helper, no float32->float64 inverse.
        expected = encode_normalized32(
            physical, feature_names, self.normalization_mean, self.normalization_std
        )
        if not np.array_equal(
            np.ascontiguousarray(expected).view(np.uint32),
            np.ascontiguousarray(normalized).view(np.uint32),
        ):
            mismatches = int(np.count_nonzero(
                np.ascontiguousarray(expected).view(np.uint32) !=
                np.ascontiguousarray(normalized).view(np.uint32)
            ))
            raise RuntimeError(
                f"Physical/model encoding mismatch in {mismatches} coordinate(s); "
                "evaluation aborted to prevent oracle/model desynchronization"
            )

        mask = None
        if valid_mask is not None:
            mask = self._to_numpy(valid_mask).astype(bool, copy=False)
            if mask.shape != (b, t):
                raise ValueError(f"valid_mask shape {mask.shape} must equal {(b,t)}")

        # RuleSet is packet-oriented. Flat sequences are reshaped as a view.
        rule_features = physical.reshape(b, t, f)
        rule_original = None if clean is None else clean.reshape(b, t, f)
        accepted, violations = self.rules.evaluate(
            rule_features, feature_names, valid_mask=mask, original=rule_original
        )
        rate = float(accepted.mean()) if len(accepted) else 1.0
        return GatekeeperResult(
            accepted=accepted,
            realizability_rate=rate,
            violations=tuple(violations),
            metadata={
                "sample_count": int(b),
                "violation_count": int((~accepted).sum()),
                "strict": self.strict,
                "dual_encoding_verified": True,
            },
        )

    def filter_valid(
        self, samples: np.ndarray | torch.Tensor, result: GatekeeperResult
    ) -> np.ndarray | torch.Tensor:
        """Return only accepted samples while preserving the input container type."""
        if isinstance(samples, torch.Tensor):
            return samples[torch.as_tensor(result.accepted, device=samples.device)]
        return np.asarray(samples)[result.accepted]

    def gate(
        self,
        samples: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        authoritative_physical: np.ndarray | torch.Tensor,
        original_physical: np.ndarray | torch.Tensor | None = None,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[np.ndarray | torch.Tensor, GatekeeperResult]:
        result = self.validate(
            samples, feature_names, authoritative_physical=authoritative_physical,
            original_physical=original_physical, valid_mask=valid_mask,
        )
        return self.filter_valid(samples, result), result

    def measure_realizability(
        self,
        samples: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        authoritative_physical: np.ndarray | torch.Tensor,
        original_physical: np.ndarray | torch.Tensor | None = None,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        return self.validate(
            samples, feature_names, authoritative_physical=authoritative_physical,
            original_physical=original_physical, valid_mask=valid_mask,
        ).realizability_rate
