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
        bad = np.flatnonzero(~np.isfinite(values) | (values < 0.0))
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
    tolerance: float = 0.0
    name: str = "discrete_integrity"

    def check(self, x: np.ndarray, feature_index: Mapping[str, int]) -> RuleViolation | None:
        found = _column(x, feature_index, self.feature)
        if found is None:
            return None
        values, _ = found
        # Wire-level integer fields are exactly integral in authoritative float64.
        # A tolerance admits unrepresentable packet lengths/window values.
        integral = np.isfinite(values) & (values == np.rint(values))
        if self.allowed_values is not None:
            integral &= np.isin(values, np.asarray(self.allowed_values, dtype=np.float64))
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
        is_tcp = proto == 6.0
        flags = np.stack([syn, ack, fin, rst], axis=1)
        integral = np.all(np.isfinite(flags) & (flags >= 0) & (flags == np.rint(flags)), axis=1)
        non_tcp_has_flags = (~is_tcp) & np.any(flags > 0, axis=1)
        bad = np.flatnonzero(~integral | non_tcp_has_flags)
        return RuleViolation(self.name, bad, "invalid TCP flag counts or flags on non-TCP flows") if len(bad) else None


@dataclass(frozen=True)
class DirectionalImmutabilityRule(ConstraintRule):
    """Reject DIR changes and edits to packets sent by the victim endpoint.

    A non-MitM attacker can generate or modify its own outbound packets, but
    cannot alter the victim's responses while they traverse the network. This
    differential rule compares adversarial and clean feature rows and marks a
    sample invalid when any feature changes on a physical ``DIR == 1`` row.
    """

    direction_feature: str = "DIR"
    victim_direction: float = 1.0
    tolerance: float = 0.0
    name: str = "directional_immutability"

    def check(
        self,
        x: np.ndarray,
        feature_index: Mapping[str, int],
        *,
        original: np.ndarray | None = None,
    ) -> RuleViolation | None:
        if original is None or self.direction_feature not in feature_index:
            return None
        if original.shape != x.shape:
            raise ValueError("Directional immutability requires matching clean and adversarial shapes")
        # Use the clean direction label to establish packet ownership. Using
        # the adversarial DIR value would let an attack relabel a victim row
        # before the immutability check runs.
        direction = original[:, feature_index[self.direction_feature]]
        victim = np.isclose(direction, self.victim_direction, atol=self.tolerance, rtol=0.0)
        # Physical authority is float64: enforce bit-identical values, not a
        # numerical tolerance. Signed zero is also treated as a byte change.
        if x.dtype != np.float64 or original.dtype != np.float64:
            raise TypeError("Directional immutability requires float64 physical arrays")
        direction_changed = (
            x[:, feature_index[self.direction_feature]].view(np.uint64)
            != direction.view(np.uint64)
        )
        packet_changed = np.any(
            x.view(np.uint64) != original.view(np.uint64), axis=1
        )
        bad = np.flatnonzero(direction_changed | (victim & packet_changed))
        return (
            RuleViolation(
                self.name,
                bad,
                "DIR must remain unchanged and all victim-direction packet features must remain unchanged",
            )
            if len(bad)
            else None
        )


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
                # Independent Gatekeeper oracle: these integrality rules are always
                # enforced, regardless of whether an attack uses lattice projection.
                DiscreteIntegrityRule("PL", tuple(range(0, 1501)), name="discrete_packet_length"),
                NonNegativeRule("IAT", name="non_negative_inter_arrival_time"),
                RangeRule("TTL", 0, 255),
                DiscreteIntegrityRule("TTL", tuple(range(0, 256)), name="discrete_ttl_integrity"),
                DiscreteIntegrityRule("DIR", (0, 1)),
                RangeRule("WIN", 0, 65535),
                DiscreteIntegrityRule("WIN", name="discrete_window_integrity"),
                DiscreteIntegrityRule("FLG", tuple(range(0, 256))),
                DirectionalImmutabilityRule(),
            )
        )

    def evaluate(
        self,
        x: np.ndarray,
        feature_names: Sequence[str],
        *,
        valid_mask: np.ndarray | None = None,
        original: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[RuleViolation]]:
        """Evaluate all rules and map packet-level failures to sample indices.

        Physical wire-domain checks ignore positions with ``valid_mask=False``.
        The independent gatekeeper separately enforces exact canonical padding
        sentinels and contiguous real-packet occupancy on those positions.
        """
        if x.ndim == 2:
            flat = x
            original_flat = None if original is None else original
            sample_index = np.arange(len(x), dtype=np.int64)
            valid_flat = np.ones(len(x), dtype=bool)
        elif x.ndim == 3:
            n, t, f = x.shape
            if len(feature_names) != f:
                raise ValueError(f"Expected {f} feature names for sequence data, got {len(feature_names)}")
            flat = x.reshape(n * t, f)
            if original is not None:
                if original.shape != x.shape:
                    raise ValueError("Original and adversarial sequence shapes must match")
                original_flat = original.reshape(n * t, f)
            else:
                original_flat = None
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
            if isinstance(rule, DirectionalImmutabilityRule):
                checked = rule.check(flat, mapping, original=original_flat)
            else:
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


def validate_clean_packet_source(
    physical: np.ndarray,
    feature_names: Sequence[str],
    valid_mask: np.ndarray,
    padding_values: Mapping[str, float],
    sample_ids: Sequence[str],
    *,
    batch_size: int = 256,
) -> None:
    """Fail closed on invalid original float64 packets before any model fitting.

    Applies the *same* fixed wire-domain rules as the independent adversarial
    gatekeeper. No learned statistics, normalized values, or test-informed
    repairs are used. Bounded slices avoid another full-corpus float64 buffer.
    """
    source = np.asarray(physical)
    mask = np.asarray(valid_mask)
    if source.dtype != np.float64 or source.ndim != 3:
        raise TypeError("Clean packet validation requires physical float64 (N,T,F)")
    n, t, f = source.shape
    if mask.dtype != np.bool_ or mask.shape != (n, t) or len(feature_names) != f:
        raise ValueError("Clean packet validity mask/schema does not match physical data")
    if len(sample_ids) != n or batch_size < 1:
        raise ValueError("Clean packet sample IDs/batch size do not match source")
    names = tuple(str(name) for name in feature_names)
    if len(set(names)) != len(names):
        raise ValueError("Clean physical feature names must be unique")
    expected = np.arange(t)[None, :] < mask.sum(axis=1)[:, None]
    nonprefix = np.flatnonzero(np.any(mask != expected, axis=1))
    if len(nonprefix):
        raise ValueError(f"Invalid clean packet occupancy at sample {sample_ids[int(nonprefix[0])]!r}")
    if not all(name in padding_values for name in names):
        raise ValueError("Clean physical schema lacks canonical padding sentinels")
    canonical = np.array([padding_values[name] for name in names], dtype=np.float64)
    if not np.isfinite(canonical).all():
        raise ValueError("Clean physical padding values must be finite")
    rule_set = RuleSet.default()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        chunk, occupancy = source[start:stop], mask[start:stop]
        if not np.isfinite(chunk).all():
            raise ValueError(f"Non-finite clean physical packet at sample {sample_ids[start]!r}")
        accepted, violations = rule_set.evaluate(chunk, names, valid_mask=occupancy)
        if not accepted.all():
            bad = int(np.flatnonzero(~accepted)[0])
            violated = ', '.join(sorted({v.rule for v in violations if bad in v.indices}))
            raise ValueError(
                f"Invalid clean physical packet at sample {sample_ids[start + bad]!r}: {violated}"
            )
        # Exact IEEE-754 padding bytes: do not discard a signed-zero change.
        for feature_index, sentinel in enumerate(canonical):
            column = chunk[:, :, feature_index]
            wrong = (~occupancy) & (column.view(np.uint64) != sentinel.view(np.uint64))
            if np.any(wrong):
                bad = int(np.flatnonzero(np.any(wrong, axis=1))[0])
                raise ValueError(
                    f"Noncanonical clean padding at sample {sample_ids[start + bad]!r}, "
                    f"feature {names[feature_index]!r}"
                )
