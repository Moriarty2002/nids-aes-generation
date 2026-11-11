"""Strict, scoped classic-PCAP packet-prefix extractor for *offline fixtures*.

This adapter is NOT the deferred CICIoMT2024 raw-PCAP preprocessor. Its only
six-field contract is IP total length (PL), between-record elapsed time (IAT),
attacker-ownership DIR, TCP advertised WIN/FLG (zero for UDP), and IPv4 TTL.
Those choices are declared, not inferred from original CICIoMT captures.
Only complete unfragmented single-flow established IPv4 TCP/UDP datagrams are
accepted. Valid first-20 masks and exact float64 measurement provenance are
recomputed after each offline mutation before independent NIDS gatekeeping.
No packet is transmitted; no frozen target is available to this adapter.
"""
from __future__ import annotations

from ipaddress import IPv4Address, ip_address
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from core.dual_representation import encode_normalized32
from core.packet_schema import CICIOMT2024_FIELDS, CICIOMT2024_PAD_VALUES
from core.registry import register_preprocessor
from attacks.problem_space.deepred import PcapFlow, _read_pcap, _format, _checksum


def _prefix(capture: PcapFlow, attacker_ip: IPv4Address) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract one capture's original ordered packet prefix without synthesis."""
    fmt = _format(capture.global_header)
    if not capture.records:
        raise ValueError('An offline packet study cannot contain empty flows')
    out = np.zeros((20, 6), dtype=np.float64)
    for field, value in CICIOMT2024_PAD_VALUES.items():
        out[:, CICIOMT2024_FIELDS.index(field)] = value
    mask = np.zeros(20, dtype=np.bool_)
    first_tuple = None
    previous_tick = None
    for index, record in enumerate(capture.records):
        if index >= 20:
            break
        frame = record.frame
        off = 14 if fmt.link_type == 1 else 0
        if fmt.link_type == 1 and (len(frame) < 14 or frame[12:14] != b'\x08\x00'):
            raise ValueError('Only Ethernet IPv4 is supported in offline prefix')
        if len(frame) < off + 20 or frame[off] != 0x45:
            raise ValueError('Only IPv4 without IP options is supported')
        v, dscp, ip_len, ident, frag, ttl, protocol, checksum, src, dst = struct.unpack_from(
            '!BBHHHBBH4s4s', frame, off)
        if frag & 0x3fff or (frag & 0x2000) or ip_len != len(frame) - off:
            raise ValueError('IPv4 fragmentation, truncation or trailing bytes are unsupported')
        if _checksum(frame[off:off + 20]) != 0:
            raise ValueError('Offline source PCAP requires correct IPv4 checksum')
        if protocol not in (6, 17):
            raise ValueError('Offline packet study accepts only IPv4 TCP/UDP')
        transport_offset = off + 20
        if len(frame) < transport_offset + (20 if protocol == 6 else 8):
            raise ValueError('Truncated IPv4 transport header')
        sport, dport = struct.unpack_from('!HH', frame, transport_offset)
        endpoints = (src, sport, dst, dport, protocol)
        if first_tuple is None:
            first_tuple = endpoints
        elif endpoints != first_tuple and endpoints != (
                first_tuple[2], first_tuple[3], first_tuple[0], first_tuple[1], first_tuple[4]):
            raise ValueError('Offline packet study cannot mix independent TCP/UDP flows')
        if src != attacker_ip.packed and dst != attacker_ip.packed:
            raise ValueError('Declared attacker IP is absent from the source flow')
        direction = 0 if src == attacker_ip.packed else 1
        if protocol == 6:
            data_offset_flags, window = struct.unpack_from('!HH', frame, transport_offset + 12)
            header_len = ((data_offset_flags >> 12) & 0xf) * 4
            flags = data_offset_flags & 0x1ff
            if header_len != 20 or flags & (0x01 | 0x02 | 0x04):
                raise ValueError('Offline TCP bridge supports established ACK/PSH, no SYN/FIN/RST/options')
            win, flag = window, flags
        else:
            udp_length, udp_checksum = struct.unpack_from('!HH', frame, transport_offset + 4)
            if udp_length != len(frame) - transport_offset:
                raise ValueError('UDP declared length differs from full packet')
            win, flag = 0, 0
        tick = int(record.seconds) * int(fmt.tick_rate) + int(record.fraction)
        if previous_tick is not None and tick < previous_tick:
            raise ValueError('Offline PCAP packet timestamps must be ordered')
        # Integer tick subtraction avoids cancellation of long epoch times.
        interval = 0. if previous_tick is None else float(tick - previous_tick) / fmt.tick_rate
        out[index] = (float(ip_len), interval, float(direction), float(win), float(flag), float(ttl))
        mask[index] = True
        previous_tick = tick
    return out, mask, len(capture.records)


@register_preprocessor('offline_ipv4_packet_prefix')
class OfflinePcapPacketPrefix:
    """Strict offline PCAP->(N,20,6) adapter; never claim real CICIoMT equivalence."""
    def __init__(self, *, attacker_ip: str):
        address = ip_address(attacker_ip)
        if not isinstance(address, IPv4Address):
            raise ValueError('Offline packet bridge requires an explicit IPv4 attacker address')
        self.attacker_ip = address
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = None if state.get('mean') is None else np.asarray(state['mean'], dtype=np.float32)
        self.std = None if state.get('std') is None else np.asarray(state['std'], dtype=np.float32)

    def transform(self, raw: Sequence[PcapFlow | str | Path]):
        if not isinstance(raw, (tuple, list, np.ndarray)):
            raise ValueError('Offline prefix bridge accepts a batch of named PCAP sources')
        n = len(raw)
        physical = np.empty((n, 20, 6), dtype=np.float64)
        mask = np.empty((n, 20), dtype=np.bool_)
        original_lengths = []
        for i, item in enumerate(raw):
            capture = _read_pcap(item) if isinstance(item, (str,Path)) else item
            if not isinstance(capture, PcapFlow):
                raise ValueError('Each offline capture must be a validated PcapFlow or path')
            physical[i], mask[i], original_len = _prefix(capture, self.attacker_ip)
            original_lengths.append(original_len)
        features = encode_normalized32(physical, CICIOMT2024_FIELDS, self.mean, self.std)
        return SimpleNamespace(features=features, physical_features=physical,
                               valid_mask=mask, feature_names=CICIOMT2024_FIELDS,
                               metadata={'source':'offline_ipv4_packet_prefix_v1',
                                         'raw_packet_counts':original_lengths,
                                         'not_ciciomt_raw_extractor_certification':True})
