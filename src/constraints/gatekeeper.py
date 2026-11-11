from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping

import numpy as np
import torch

from core.dual_representation import encode_normalized32, layout
from core.packet_schema import CICIOMT2024_PAD_VALUES
from .budget import FinalAttackBudget, check_final_budget

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
        strict: bool = True,
        normalization_mean: np.ndarray | None = None,
        normalization_std: np.ndarray | None = None,
        padding_values: Mapping[str, float] | None = None,
        padding_indicator_name: str | None = None,
    ) -> None:
        self.rules = rules if rules is not None else RuleSet.default()
        self.strict = bool(strict)
        self.padding_indicator_name = (None if padding_indicator_name is None
                                       else str(padding_indicator_name).upper())
        self.padding_values = None if padding_values is None else {str(k).upper(): float(v) for k,v in padding_values.items()}
        if self.padding_values is not None and not all(np.isfinite(v) for v in self.padding_values.values()):
            raise ValueError("Canonical padding values must be finite")
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
        is_problem_space: bool = False,
        attack_budget: FinalAttackBudget | None = None,
        budget_normalization_mean: np.ndarray | None = None,
        budget_normalization_std: np.ndarray | None = None,
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
        elif "DIR" in feature_names and not is_problem_space and any(
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
            mask = self._to_numpy(valid_mask)
            if mask.dtype != np.bool_ or mask.shape != (b, t):
                raise ValueError(
                    f"valid_mask must be exact boolean with shape {(b,t)}; "
                    f"received {mask.dtype} {mask.shape}"
                )

        # RuleSet is packet-oriented. Flat sequences are reshaped as a view.
        rule_features = physical.reshape(b, t, f)
        rule_original = None if clean is None else clean.reshape(b, t, f)
        rules = (
            tuple(
                rule for rule in self.rules.rules
                if not (is_problem_space and isinstance(rule, DirectionalImmutabilityRule))
            )
            if is_problem_space
            else self.rules.rules
        )
        rule_set = self.rules if rules == self.rules.rules else RuleSet(rules)
        accepted, violations = rule_set.evaluate(
            rule_features, feature_names, valid_mask=mask, original=rule_original
        )
        # Final packet occupancy is a prefix by protocol. Reject malformed
        # records *before* calling a sequence model that assumes prefix masks.
        if mask is not None:
            lengths = mask.sum(axis=1)
            expected_prefix = np.arange(t)[None, :] < lengths[:, None]
            malformed = np.flatnonzero(~np.all(mask == expected_prefix, axis=1))
            if len(malformed):
                accepted[malformed] = False
                violations.append(RuleViolation("nonprefix_packet_occupancy", malformed,
                                                "Packet occupancy must be a contiguous real prefix"))

            # Optional materialized occupancy indicators are physical features:
            # their REAL positions must encode 0.0, not a forged "padded" flag.
            # The corresponding invalid positions are checked below against
            # the schema's canonical 1.0 padding sentinel.
            indicator_name = self.padding_indicator_name
            upper_names = tuple(str(name).upper() for name in feature_names)
            if indicator_name is None and "FEAT_PAD_MASK" in upper_names:
                indicator_name = "FEAT_PAD_MASK"
            if indicator_name is not None:
                if indicator_name not in upper_names:
                    raise ValueError(f"Configured padding indicator {indicator_name!r} absent from feature schema")
                indicator = rule_features[..., upper_names.index(indicator_name)]
                wrong_real = (indicator.view(np.uint64) != np.float64(0.0).view(np.uint64)) & mask
                rejected = np.flatnonzero(np.any(wrong_real, axis=1))
                if len(rejected):
                    accepted[rejected] = False
                    violations.append(RuleViolation("padding_indicator_occupancy", rejected,
                                                    "Real packet positions must have canonical 0.0 occupancy indicator"))

            if np.any(~mask):
                padding_schema = self.padding_values
                if padding_schema is None:
                    # Only the *exact* active CICIoMT2024 schema has known built-in
                    # sentinels; all other padded schemas require explicit values.
                    names = tuple(str(name).upper() for name in feature_names)
                    if set(names) == set(CICIOMT2024_PAD_VALUES):
                        padding_schema = CICIOMT2024_PAD_VALUES
                    elif set(names) == set(CICIOMT2024_PAD_VALUES) | {"FEAT_PAD_MASK"}:
                        padding_schema = {**CICIOMT2024_PAD_VALUES, "FEAT_PAD_MASK": 1.0}
                    else:
                        raise ValueError("Masked packet layouts require explicit canonical padding_values")
                required_names = tuple(str(name).upper() for name in feature_names)
                missing = set(required_names) - set(padding_schema)
                if missing:
                    raise ValueError(f"Missing padding sentinels for fields {sorted(missing)}")
                padded = rule_features.reshape(b, t, f)
                malformed_padding = np.zeros(b, dtype=bool)
                for col, name in enumerate(required_names):
                    expected = np.float64(padding_schema[name])
                    # Raw IEEE-754 bit equality checks signed zero as well.
                    wrong = padded[..., col].view(np.uint64) != expected.view(np.uint64)
                    malformed_padding |= np.any(wrong & ~mask, axis=1)
                rejected = np.flatnonzero(malformed_padding)
                if len(rejected):
                    accepted[rejected] = False
                    violations.append(RuleViolation("noncanonical_padding", rejected,
                                                    "Masked positions must equal schema padding sentinels bit-exactly"))

        if attack_budget is not None:
            if budget_normalization_mean is None or budget_normalization_std is None:
                raise ValueError("Final attack-budget verification requires authorized normalization statistics")
            if clean is None:
                raise ValueError("Independent final-budget validation requires original float64 physical input")
            fits = check_final_budget(
                physical, clean, feature_names, attack_budget,
                budget_normalization_mean, budget_normalization_std,
            )
            rejected = np.flatnonzero(~fits)
            if len(rejected):
                accepted[rejected] = False
                violations.append(RuleViolation(
                    f"attack_budget_{attack_budget.norm}_{attack_budget.domain}",
                    rejected,
                    "Final physical candidate exceeds the attack's declared normalized hard budget",
                ))
        rate = float(accepted.mean()) if len(accepted) else 1.0
        return GatekeeperResult(
            accepted=accepted,
            realizability_rate=rate,
            violations=tuple(violations),
            metadata={
                "sample_count": int(b),
                "violation_count": int((~accepted).sum()),
                "strict": self.strict,
                "is_problem_space": bool(is_problem_space),
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
        is_problem_space: bool = False,
    ) -> tuple[np.ndarray | torch.Tensor, GatekeeperResult]:
        result = self.validate(
            samples, feature_names, authoritative_physical=authoritative_physical,
            original_physical=original_physical, valid_mask=valid_mask,
            is_problem_space=is_problem_space,
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
        is_problem_space: bool = False,
    ) -> float:
        return self.validate(
            samples, feature_names, authoritative_physical=authoritative_physical,
            original_physical=original_physical, valid_mask=valid_mask,
            is_problem_space=is_problem_space,
        ).realizability_rate
