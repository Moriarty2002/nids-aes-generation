"""Independent mapping-v1 victim-packet preservation check.

This is a read-only audit over raw packet mappings. It never infers packet
identity by comparing shifted normalized feature rows. Other raw formats must
supply an independently auditable ownership verifier in their own adapters.
"""
from __future__ import annotations

from collections.abc import Mapping
import struct
from itertools import chain, zip_longest
from typing import Any
import numpy as np


def _mapping_flow(flow: Any) -> bool:
    return (isinstance(flow, (list, tuple))
            and all(isinstance(packet, Mapping) for packet in flow))


def _packet_value_bit_identical(a: Any, b: Any) -> bool:
    """Audit exact raw mapping-v1 packet identity, including float signed zero.

    Python mapping == masks the -0.0/+0.0 distinction; this verifier must not.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, Mapping):
        return (set(a) == set(b) and
                all(_packet_value_bit_identical(a[key], b[key]) for key in a))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_packet_value_bit_identical(x,y) for x,y in zip(a,b))
    if isinstance(a, np.ndarray):
        return (a.dtype == b.dtype and a.shape == b.shape and
                a.tobytes(order='C') == b.tobytes(order='C'))
    if isinstance(a, (float, np.floating)):
        if isinstance(a, np.floating):
            return a.dtype == b.dtype and a.tobytes() == b.tobytes()
        return struct.pack('!d', a) == struct.pack('!d', b)
    try:
        outcome = a == b
        return bool(outcome) if isinstance(outcome, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def verify_mapping_v1_victim_packets(clean_flows: Any, generated_flows: Any) -> np.ndarray | None:
    """Return per-flow ownership feasibility, or None for non-mapping traffic.

    Victim-owned packets must occur unchanged and in the same relative order.
    Insertion of attacker-owned packets anywhere is allowed. Victim insertion,
    deletion, alteration, or reordering is rejected. No full packet copies.
    """
    if not isinstance(clean_flows, (list, tuple)) or not isinstance(generated_flows, (list, tuple)):
        return None
    if len(clean_flows) != len(generated_flows):
        raise ValueError('Raw generated batch and clean batch lengths differ')
    if not all(_mapping_flow(flow) for flow in clean_flows):
        return None
    if not all(_mapping_flow(flow) for flow in generated_flows):
        raise TypeError('Mapping-v1 attack must return mapping-v1 packet flows')
    accepted = np.ones(len(clean_flows), dtype=bool)
    for idx, (clean, generated) in enumerate(zip(clean_flows, generated_flows)):
        try:
            if any(type(packet.get('direction')) is not int or
                   packet['direction'] not in (0, 1) for packet in chain(clean, generated)):
                accepted[idx] = False
                continue
            # Iterators keep verifier memory O(1) per flow. The sentinel
            # detects inserted or deleted victim packets without alignment.
            missing = object()
            clean_victim = (packet for packet in clean if packet['direction'] == 1)
            generated_victim = (packet for packet in generated if packet['direction'] == 1)
            accepted[idx] = all(a is not missing and b is not missing and _packet_value_bit_identical(a, b)
                                for a, b in zip_longest(clean_victim, generated_victim, fillvalue=missing))
        except (ValueError, TypeError, KeyError):
            accepted[idx] = False
    return accepted


def mapping_v1_packet_edit_opcodes(clean_flows: Any, generated_flows: Any) -> np.ndarray | None:
    """Descriptive per-flow (insert, delete, replace) sequence-diff opcode counts.

    Only defined for mapping-v1 raw packet lists; these are deterministic
    SequenceMatcher alignments, *not* a guaranteed minimum-cost edit sequence.
    Run independent victim-ownership verification separately before presenting
    these figures as feasible edits.
    """
    from difflib import SequenceMatcher
    if not isinstance(clean_flows, (list, tuple)) or not isinstance(generated_flows, (list, tuple)):
        return None
    if len(clean_flows) != len(generated_flows):
        raise ValueError('Raw flow batch lengths differ')
    if not all(_mapping_flow(flow) for flow in clean_flows):
        return None
    if not all(_mapping_flow(flow) for flow in generated_flows):
        raise TypeError('Raw mapping-v1 attack returned non-mapping flows')

    counts = np.zeros((len(clean_flows), 3), dtype=np.int64)
    for i, (clean, generated) in enumerate(zip(clean_flows, generated_flows)):
        # Stable ordering of mapping keys; repr captures byte payloads and
        # signed zeros in simple mapping-v1 scalar attributes.
        def tokens(flow: Any) -> list[tuple[tuple[str, str], ...]]:
            return [tuple((str(k), repr(p[k])) for k in sorted(p)) for p in flow]
        operations = SequenceMatcher(None, tokens(clean), tokens(generated), autojunk=False)
        for kind, a0, a1, b0, b1 in operations.get_opcodes():
            if kind == 'insert':
                counts[i, 0] += b1-b0
            elif kind == 'delete':
                counts[i, 1] += a1-a0
            elif kind == 'replace':
                replaced = min(a1-a0, b1-b0)
                counts[i, 2] += replaced
                counts[i, 0] += b1-b0-replaced
                counts[i, 1] += a1-a0-replaced
    return counts
