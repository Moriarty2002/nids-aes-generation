from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from core.registry import register_attack
from attacks.base import ProblemSpaceAttack

try:
    from scapy.packet import Packet
    from scapy.layers.inet import IP, TCP, UDP
except ImportError:  # pragma: no cover - optional dependency
    Packet = object  # type: ignore[assignment]
    IP = TCP = UDP = None


@dataclass
@register_attack("problem_space_dummy")
class DummyPCAPMutator(ProblemSpaceAttack):
    """Protocol-aware payload mutator for deterministic problem-space smoke runs.

    Packets are deep-copied before mutation so callers retain the original
    capture. Checksums and derived IP length fields are invalidated because
    Scapy must recalculate them when the packet is serialized.
    """

    byte_delta: int = 4
    max_packets: int | None = None

    attack_name = "problem_space_dummy"

    def generate_raw(self, traffic: Any, y: Any, **kwargs: Any) -> Any:
        """Mutate up to ``max_packets`` payload-bearing TCP/UDP packets."""
        del y, kwargs
        if isinstance(traffic, (str, bytes)):
            raise TypeError("DummyPCAPMutator expects an in-memory packet collection for deterministic routing")
        packets = list(deepcopy(traffic))
        budget = self.max_packets if self.max_packets is not None else len(packets)
        if budget < 0:
            raise ValueError("max_packets must be >= 0")
        mutated: list[Any] = []
        mutated_count = 0
        for packet in packets:
            candidate = packet
            if mutated_count < budget and Packet is not object and (packet.haslayer(TCP) or packet.haslayer(UDP)):
                transport = packet[TCP] if packet.haslayer(TCP) else packet[UDP]
                payload = bytes(transport.payload) if transport.payload is not None else b""
                if payload:
                    data = payload + (b"N" * max(0, self.byte_delta))
                    transport.remove_payload()
                    transport.add_payload(data)
                    if hasattr(transport, "chksum"):
                        del transport.chksum
                    if IP is not None and packet.haslayer(IP) and hasattr(packet[IP], "len"):
                        del packet[IP].len
                    if IP is not None and packet.haslayer(IP) and hasattr(packet[IP], "chksum"):
                        del packet[IP].chksum
                    mutated_count += 1
            mutated.append(candidate)
        return mutated
