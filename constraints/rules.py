from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RuleViolation:
    rule: str
    indices: np.ndarray
    message: str


class ConstraintRule:
    name: str = "constraint"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        raise NotImplementedError


def _column(x: np.ndarray, feature_index: Mapping[str, int], feature: str) -> tuple[np.ndarray, np.ndarray] | None:
    if feature not in feature_index:
        return None
    idx = feature_index[feature]
    return x[:, idx], np.asarray(idx)


@dataclass(frozen=True)
class NonNegativeRule(ConstraintRule):
    feature: str
    name: str = "non_negative"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        found = _column(x, feature_index, self.feature)
        if found is None:
            return None
        values, _ = found
        bad = np.flatnonzero(~np.isfinite(values) | (values < 0))
        return RuleViolation(self.name, bad, f"{self.feature} must be finite and >= 0") if len(bad) else None


@dataclass(frozen=True)
class RangeRule(ConstraintRule):
    feature: str
    lower: float
    upper: float
    name: str = "range"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        found = _column(x, feature_index, self.feature)
        if found is None:
            return None
        values, _ = found
        bad = np.flatnonzero(~np.isfinite(values) | (values < self.lower) | (values > self.upper))
        return RuleViolation(self.name, bad, f"{self.feature} must be in [{self.lower}, {self.upper}]") if len(bad) else None


@dataclass(frozen=True)
class DiscreteIntegrityRule(ConstraintRule):
    feature: str
    allowed_values: tuple[int, ...] | None = None
    tolerance: float = 1e-6
    name: str = "discrete_integrity"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        found = _column(x, feature_index, self.feature)
        if found is None:
            return None
        values, _ = found
        integral = np.isfinite(values) & (np.abs(values - np.rint(values)) <= self.tolerance)
        if self.allowed_values is not None:
            integral &= np.isin(np.rint(np.nan_to_num(values, nan=0.0)).astype(np.int64), np.asarray(self.allowed_values, dtype=np.int64))
        bad = np.flatnonzero(~integral)
        return RuleViolation(self.name, bad, f"{self.feature} must preserve discrete integrity") if len(bad) else None


@dataclass(frozen=True)
class FlowDurationRule(ConstraintRule):
    duration_feature: str = "flow_duration_s"
    max_duration_s: float = 24.0 * 3600.0
    name: str = "flow_duration"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        return RangeRule(self.duration_feature, 0.0, self.max_duration_s, self.name).check(x, feature_index)


@dataclass(frozen=True)
class PacketByteConsistencyRule(ConstraintRule):
    packet_count_feature: str = "packet_count"
    byte_count_feature: str = "byte_count"
    name: str = "packet_byte_consistency"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        if self.packet_count_feature not in feature_index or self.byte_count_feature not in feature_index:
            return None
        packets = x[:, feature_index[self.packet_count_feature]]
        bytes_ = x[:, feature_index[self.byte_count_feature]]
        bad = np.flatnonzero((~np.isfinite(packets)) | (~np.isfinite(bytes_)) | (packets < 0) | (bytes_ < 0) | ((packets == 0) & (bytes_ > 0)))
        return RuleViolation(self.name, bad, "packet/byte counts must be physically consistent") if len(bad) else None


@dataclass(frozen=True)
class TcpFlagStateRule(ConstraintRule):
    syn: str = "tcp_syn"
    ack: str = "tcp_ack"
    fin: str = "tcp_fin"
    rst: str = "tcp_rst"
    protocol: str = "protocol"
    name: str = "tcp_flag_state"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        required = [self.syn, self.ack, self.fin, self.rst, self.protocol]
        if any(k not in feature_index for k in required):
            return None
        syn = x[:, feature_index[self.syn]]
        ack = x[:, feature_index[self.ack]]
        fin = x[:, feature_index[self.fin]]
        rst = x[:, feature_index[self.rst]]
        proto = x[:, feature_index[self.protocol]]
        is_tcp = np.isclose(proto, 6.0)
        flags = np.stack([syn, ack, fin, rst], axis=1)
        integral = np.all(np.isfinite(flags) & (flags >= 0) & (np.abs(flags - np.rint(flags)) <= 1e-6), axis=1)
        non_tcp_has_flags = (~is_tcp) & np.any(flags > 0, axis=1)
        bad = np.flatnonzero(~integral | non_tcp_has_flags)
        return RuleViolation(self.name, bad, "invalid TCP flag counts or flags on non-TCP flows") if len(bad) else None


@dataclass(frozen=True)
class RuleSet:
    rules: tuple[ConstraintRule, ...]

    @classmethod
    def default(cls) -> "RuleSet":
        return cls(
            rules=(
                NonNegativeRule("packet_count"),
                NonNegativeRule("byte_count"),
                NonNegativeRule("flow_duration_s"),
                RangeRule("protocol", 0, 255),
                DiscreteIntegrityRule("protocol", tuple(range(0, 256))),
                PacketByteConsistencyRule(),
                FlowDurationRule(),
                TcpFlagStateRule(),
                # Sequence-style flow fields. Padding values are ignored when the
                # gatekeeper receives the dataset's valid packet mask.
                NonNegativeRule("PL"),
                NonNegativeRule("IAT"),
                RangeRule("TTL", 0, 255),
                DiscreteIntegrityRule("DIR", (-1, 0, 1)),
                RangeRule("WIN", -1, 65535),
                DiscreteIntegrityRule("FLG", tuple([-1] + list(range(0, 256)))),
            )
        )

    def evaluate(
        self,
        x: np.ndarray,
        feature_names: Sequence[str],
        *,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[RuleViolation]]:
        """Evaluate all rules and map packet-level failures to sample indices.

        Sequence padding is ignored when ``valid_mask`` is supplied, so padded
        values cannot make an otherwise valid flow fail strict evaluation.
        """
        if x.ndim == 2:
            flat = x
            sample_index = np.arange(len(x), dtype=np.int64)
            valid_flat = np.ones(len(x), dtype=bool)
        elif x.ndim == 3:
            n, t, f = x.shape
            if len(feature_names) != f:
                raise ValueError(f"Expected {f} feature names for sequence data, got {len(feature_names)}")
            flat = x.reshape(n * t, f)
            sample_index = np.repeat(np.arange(n, dtype=np.int64), t)
            if valid_mask is None:
                valid_flat = np.ones(n * t, dtype=bool)
            else:
                mask = np.asarray(valid_mask, dtype=bool)
                if mask.shape != (n, t):
                    raise ValueError(f"valid_mask shape {mask.shape} does not match sequence shape {(n, t)}")
                valid_flat = mask.reshape(-1)
        else:
            raise ValueError(f"Constraints require a 2-D or 3-D batch, received shape {x.shape}")

        mapping = {name: i for i, name in enumerate(feature_names)}
        invalid_samples = np.zeros(len(x), dtype=bool)
        violations: list[RuleViolation] = []
        for rule in self.rules:
            checked = rule.check(flat, mapping)
            if checked is None:
                continue
            checked_positions = checked.indices[valid_flat[checked.indices]]
            if len(checked_positions) == 0:
                continue
            affected_samples = np.unique(sample_index[checked_positions])
            violations.append(RuleViolation(checked.rule, affected_samples, checked.message))
            invalid_samples[affected_samples] = True
        return ~invalid_samples, violations


def validate_tcp_flag_sequence(flag_sequence: Sequence[Mapping[str, bool | int]]) -> tuple[bool, str | None]:
    """Conservative TCP sequence validation for raw packet streams."""
    terminated = False
    for position, packet in enumerate(flag_sequence):
        try:
            syn = bool(int(packet.get("syn", 0)))
            ack = bool(int(packet.get("ack", 0)))
            fin = bool(int(packet.get("fin", 0)))
            rst = bool(int(packet.get("rst", 0)))
        except (TypeError, ValueError):
            return False, f"non-integral TCP flag at position {position}"
        if syn and fin:
            return False, f"SYN+FIN at position {position}"
        if syn and rst:
            return False, f"SYN+RST at position {position}"
        if terminated and syn:
            return False, f"SYN after termination at position {position}"
        if rst or fin:
            terminated = True
    return True, None
