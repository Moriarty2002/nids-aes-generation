"""Deterministic mapping-v1 raw-packet mutation smoke plugin.

This plugin does not perform feature extraction, normalization, or any target
inference. Real PCAP-specific mutators must supply a separate raw-packet
identity verifier for their own representation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np

from core.registry import register_attack
from core.ownership import verify_mapping_v1_victim_packets
from attacks.base import AttackResult, ProblemSpaceAttack


@dataclass
@register_attack('problem_space_dummy')
class DummyPCAPMutator(ProblemSpaceAttack):
    """Append bytes to attacker-owned mapping-v1 packets without touching victims."""

    byte_delta: int = 4
    max_packets: int | None = None
    attack_name = 'problem_space_dummy'
    query_based = False

    def generate_raw(self, traffic: Any, y: Any, **kwargs: Any) -> AttackResult:
        del kwargs
        if self.byte_delta < 0 or (self.max_packets is not None and self.max_packets < 0):
            raise ValueError('byte_delta and max_packets must be non-negative')
        if not isinstance(traffic, (list, tuple)) or not all(
            isinstance(flow, (list, tuple)) and all(isinstance(packet, Mapping) for packet in flow)
            for flow in traffic
        ):
            raise TypeError('DummyPCAPMutator requires a batch of mapping-v1 packet flows')
        if len(traffic) != len(y):
            raise ValueError('Raw traffic/label batch lengths differ')
        generated: list[list[dict[str, Any]]] = []
        mutated = np.zeros(len(traffic), dtype=np.int64)
        for i, flow in enumerate(traffic):
            copied: list[dict[str, Any]] = []
            budget = len(flow) if self.max_packets is None else self.max_packets
            for packet in flow:
                # Never change a caller-owned raw mapping, even for unchanged rows.
                new_packet = dict(packet)
                direction = packet.get('direction')
                payload = packet.get('payload')
                if direction not in (0, 1) or not isinstance(payload, bytes):
                    raise ValueError('Raw packet must provide direction in {0,1} and bytes payload')
                if direction == 0 and mutated[i] < budget and payload and self.byte_delta:
                    new_packet['payload'] = payload + b'N' * self.byte_delta
                    mutated[i] += 1
                copied.append(new_packet)
            generated.append(copied)
        ownership = verify_mapping_v1_victim_packets(traffic, generated)
        assert ownership is not None
        return AttackResult(
            adversarial=None,
            raw_adversarial=generated,
            intrinsic_feasible_mask=ownership,
            queries=np.zeros(len(traffic), dtype=np.int64),
            metadata={'mutation_type': 'attacker_owned_payload_extension',
                      'mutated_packets': int(mutated.sum())},
        )
