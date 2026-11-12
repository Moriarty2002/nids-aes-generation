"""Mirage offline ACK-block DQN (Li et al., IEEE ICC 2025, §IV/Eq. 5).

Original: class-specific BiLSTM DQN, encrypted-flow classifier simulator,
ACK-scoped byte-preserving packet-block redistribution and confidence reward.
NIDS adaptation: offline mapping-v1 TCP block replay, strict topology checks,
train-only simulator, no frozen target queries and no wire-level claim.
Adaptation: local action discretization, payload rather than whole-packet lengths,
fixed first packet/count/endpoints and causal anchored timing. Huber/replay, target
sync, exploration, hidden width and history scaling are local DQN choices.
The malicious-to-benign NIDS goal differs from the original privacy objective.
Topology: raw mapping-v1 TCP flows or strict offline classic-PCAP IPv4/TCP
(complete ACK blocks, valid checksums, no fragments or ambiguous retransmits).
The global raw-to-canonical extractor and actual network delivery remain deferred.
"""



from __future__ import annotations

import gc
import math
import os
import random
import struct
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from attacks.base import AttackResult, ProblemSpaceAttack
from core.registry import DATASETS, register_attack
from core.provenance import (authorized_attack_training_indices, trained_generator_provenance,
                             verify_generator_artifact_provenance)

_FORMAT_VERSION = 2
_FORBIDDEN_PACKET_KEYS = frozenset({"wire_bytes", "raw", "checksum", "ip_checksum", "tcp_checksum",
                                   "len", "packet_len", "packet_length", "payload_len",
                                   "payload_length", "ip_len", "total_length",
                                   "tcp_len", "ip_total_length", "frame_len"})
_REQUIRED = frozenset({"direction", "timestamp", "payload", "seq", "ack", "flags"})


def _pcap_tcp_packets(capture: Any) -> tuple[list[dict[str, Any]], list[tuple[int,int,int,int]], Any]:
    """Decode a complete single-flow IPv4/TCP PCAP into Mirage's ACK-block view.

    The decoder verifies original IP/TCP checksums and rejects fragmentation,
    missing payload bytes, unsupported link types and mixed 5-tuples. These
    are *supported offline capture* boundaries, not claims about all PCAPs.
    """
    from attacks.problem_space.deepred import PcapFlow, _format, _checksum
    if not isinstance(capture, PcapFlow):
        raise TypeError('Expected an immutable classic-PCAP capture')
    fmt = _format(capture.global_header)
    if not capture.records:
        raise ValueError('Mirage cannot infer a sender from empty PCAP')
    parsed, offsets, endpoints = [], [], []
    for record in capture.records:
        frame = record.frame
        offset = 14 if fmt.link_type == 1 else 0
        if fmt.link_type == 1 and (len(frame) < 14 or frame[12:14] != b'\x08\x00'):
            raise ValueError('Mirage supports only Ethernet/IPv4 or raw IPv4 PCAP')
        if len(frame) < offset + 40 or frame[offset] >> 4 != 4:
            raise ValueError('Mirage requires complete IPv4/TCP packets')
        ihl = (frame[offset] & 15) * 4
        if ihl < 20 or len(frame) < offset + ihl + 20:
            raise ValueError('Malformed IPv4/TCP header')
        ip = frame[offset:offset+ihl]
        total_length, fragment = struct.unpack_from('!HH', ip, 2)[0], struct.unpack_from('!H', ip, 6)[0]
        if fragment & 0x3fff or ip[9] != 6 or offset + total_length != len(frame):
            raise ValueError('Mirage requires an unfragmented, untruncated TCP packet without L2 trailer')
        start = offset + ihl
        tcp_hlen = (frame[start+12] >> 4) * 4
        if tcp_hlen < 20 or total_length < ihl + tcp_hlen:
            raise ValueError('Invalid TCP data offset')
        head = frame[start:start+tcp_hlen]
        payload = frame[start+tcp_hlen:]
        tcp = frame[start:]
        pseudo = ip[12:20] + struct.pack('!BBH', 0, 6, len(tcp))
        if _checksum(ip) != 0 or _checksum(pseudo + tcp) != 0:
            raise ValueError('Mirage requires intact original IP/TCP checksums; offloaded captures unsupported')
        sport, dport, seq, ack = struct.unpack_from('!HHII', head, 0)
        if not sport or not dport:
            raise ValueError('TCP ports must be nonzero')
        flags = head[13]
        if flags & ~(0x10 | 0x08):
            raise ValueError('Mirage PCAP supports established ACK/PSH segments only')
        symbols = ('A' if flags & 0x10 else '') + ('P' if flags & 0x08 else '')
        endpoints.append((ip[12:16], ip[16:20], sport, dport))
        parsed.append({'timestamp': record.seconds + record.fraction/fmt.tick_rate,
                       'payload': payload, 'seq': int(seq), 'ack': int(ack),
                       'flags': symbols})
        offsets.append((offset, ihl, start, tcp_hlen))
    senders = [i for i, pkt in enumerate(parsed) if pkt['payload']]
    if not senders:
        raise ValueError('Mirage requires at least one data packet')
    sender = endpoints[senders[0]]
    reverse = (sender[1], sender[0], sender[3], sender[2])
    for i, endpoint in enumerate(endpoints):
        if endpoint not in (sender, reverse):
            raise ValueError('Mirage PCAP contains unrelated TCP flows')
        parsed[i]['direction'] = 0 if endpoint == sender else 1
    return parsed, offsets, fmt


def _pcap_from_mapping(original: Any, before: Sequence[Mapping[str, Any]],
                       changed: Sequence[Mapping[str, Any]], offsets: Sequence[tuple[int,int,int,int]],
                       fmt: Any) -> Any:
    """Write each ACK-scoped Mirage proposal as a checksummed offline packet.

    Every victim packet remains byte-identical. Sender TCP sequence values and
    checksums are recomputed when payload lengths change; total block bytes,
    ACK boundaries, flags, IP endpoints and ordered time are independently
    checked by the source `_topology_verified` and by redecoding this PCAP.
    """
    from attacks.problem_space.deepred import PcapFlow, CaptureRecord, _checksum
    blocks = _parse_blocks(before, 1, 65535)
    if blocks is None or not _topology_verified(before, changed, blocks, 1, 65535):
        raise ValueError('Mirage proposed an invalid ACK-block or byte stream')
    changed_records = []
    previous_tick = -1
    for i, (rec, baseline, candidate, (offset, ihl, start, tcp_hlen)) in enumerate(
            zip(original.records, before, changed, offsets)):
        if candidate == baseline:
            changed_records.append(rec)
            tick = rec.seconds * fmt.tick_rate + rec.fraction
        else:
            if baseline['direction'] != 0:
                raise ValueError('Cannot modify a victim TCP packet')
            stamp = float(candidate['timestamp'])
            if not math.isfinite(stamp) or stamp < 0 or stamp >= 2**32:
                raise ValueError('Invalid offline PCAP timestamp')
            tick = round(stamp * fmt.tick_rate)
            seconds, fraction = divmod(tick, fmt.tick_rate)
            frame = rec.frame
            ip = bytearray(frame[offset:start])
            tcp = bytearray(frame[start:start+tcp_hlen])
            payload = candidate['payload']
            length = len(ip) + len(tcp) + len(payload)
            if length > 65535:
                raise ValueError('Mirage exceeds IPv4 total-length limit')
            struct.pack_into('!H',ip,2,length)
            struct.pack_into('!H',ip,10,0)
            struct.pack_into('!H',ip,10,_checksum(bytes(ip)))
            struct.pack_into('!I',tcp,4,int(candidate['seq']))
            struct.pack_into('!H',tcp,16,0)
            pseudo = ip[12:20] + struct.pack('!BBH',0,6,len(tcp)+len(payload))
            struct.pack_into('!H',tcp,16,_checksum(pseudo+tcp+payload))
            assembled = frame[:offset] + ip + tcp + payload
            changed_records.append(CaptureRecord(int(seconds),int(fraction),
                                                 bytes(assembled),len(assembled)))
        if tick < previous_tick:
            raise ValueError('Mirage would reorder PCAP packet timestamps')
        previous_tick = tick
    result = PcapFlow(original.global_header,tuple(changed_records))
    decoded, _, _ = _pcap_tcp_packets(result)
    if any((a['payload'],a['seq'],a['ack'],a['direction']) !=
           (b['payload'],b['seq'],b['ack'],b['direction'])
           for a,b in zip(decoded,changed)):
        raise ValueError('Mirage TCP PCAP reparse disagrees with constructed payload/sequence')
    return result


def _pristine(flow: Any) -> Any:
    """Return the original flow as-is; an infeasible flow must never be partly edited."""
    return flow


def _packet_fields(packet: Any) -> bool:
    if not isinstance(packet, Mapping) or not _REQUIRED.issubset(packet):
        return False
    if _FORBIDDEN_PACKET_KEYS.intersection(packet):
        return False
    if type(packet["payload"]) is not bytes or packet["direction"] not in (0, 1):
        return False
    if type(packet["seq"]) is not int or type(packet["ack"]) is not int:
        return False
    if not (0 <= packet["seq"] < 2**32 and 0 <= packet["ack"] < 2**32):
        return False
    if not isinstance(packet["flags"], str):
        return False
    try:
        return math.isfinite(float(packet["timestamp"]))
    except (TypeError, ValueError, OverflowError):
        return False


def _parse_blocks(flow: Any, min_payload: int, max_payload: int) -> list[tuple[int, ...]] | None:
    """Prove ACK-scoped, lossless and unidirectional segmentation *before* mutation.

    ACK anchors must acknowledge the entire preceding run. This intentionally
    rejects delayed/partial ACKs and out-of-order/retransmitted packets: without
    TCP reassembly metadata, claiming they are safe would be speculative.
    """
    if not isinstance(flow, (list, tuple, np.ndarray)) or not len(flow):
        return None
    if isinstance(flow, np.ndarray) and flow.ndim != 1:
        return None
    if any(not _packet_fields(p) for p in flow):
        return None
    ts = [float(p["timestamp"]) for p in flow]
    if any(b < a for a, b in zip(ts, ts[1:])):
        return None

    blocks: list[tuple[int, ...]] = []
    active: list[int] = []
    for index, packet in enumerate(flow):
        if packet["direction"] == 0:
            # SYN/FIN/RST consume sequence space or alter control semantics.
            if set(packet["flags"]) - {"A", "P"}:
                return None
            size = len(packet["payload"])
            if not min_payload <= size <= max_payload:
                return None
            if active:
                predecessor = flow[active[-1]]
                expected = (predecessor["seq"] + len(predecessor["payload"])) % (2**32)
                if packet["seq"] != expected:
                    return None
            active.append(index)
            continue
        if not active:
            continue
        if "A" not in packet["flags"]:
            return None
        last = flow[active[-1]]
        end_seq = (last["seq"] + len(last["payload"])) % (2**32)
        if packet["ack"] != end_seq:
            return None
        blocks.append(tuple(active))
        active = []
    if active or not blocks:
        return None
    return blocks


def _sizes_and_iats(flow: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(flow[i]["payload"]) for i in indices), dtype=np.float64)
    times = np.fromiter((float(flow[i]["timestamp"]) for i in indices), dtype=np.float64)
    intervals = np.diff(times, prepend=times[0]) if len(times) else times
    return lengths, intervals


def _summaries(values: np.ndarray, *, temporal: bool) -> list[float]:
    """Statistics are non-negative; zero-filled empty directions are explicit."""
    if not len(values):
        return [0.0] * (6 if temporal else 8)
    result = [float(np.mean(values)), float(np.std(values)), float(np.min(values)), float(np.max(values))]
    if temporal:
        result.extend([float(np.median(values)), float(np.percentile(values, 90))])
    else:
        result.extend([float(np.percentile(values, 10)), float(np.median(values)),
                       float(np.percentile(values, 90)), float(np.sum(values))])
    return result


_STAT_GROUPS = ("all", "attacker", "victim")
_LEN_SUFFIX = ("mean", "std", "min", "max", "p10", "median", "p90", "sum")
_IAT_SUFFIX = ("mean", "std", "min", "max", "median", "p90")
_STAT_NAMES = tuple(
    [f"{part}_length_{stat}" for part in _STAT_GROUPS for stat in _LEN_SUFFIX]
    + [f"{part}_iat_{stat}" for part in _STAT_GROUPS for stat in _IAT_SUFFIX]
    + ["packet_count", "attacker_count", "victim_count", "attacker_ratio",
       "duration", "byte_count", "attacker_bytes", "victim_bytes"]
)


def _statistics(flow: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Extract candidate distribution features directly from packet timestamps/bytes."""
    all_ids = list(range(len(flow)))
    own_ids = [i for i, packet in enumerate(flow) if packet["direction"] == 0]
    peer_ids = [i for i, packet in enumerate(flow) if packet["direction"] == 1]
    length_groups: list[np.ndarray] = []
    iat_groups: list[np.ndarray] = []
    for indices in (all_ids, own_ids, peer_ids):
        lens, iats = _sizes_and_iats(flow, indices)
        length_groups.append(lens)
        iat_groups.append(iats)
    features = []
    for values in length_groups:
        features.extend(_summaries(values, temporal=False))
    for values in iat_groups:
        features.extend(_summaries(values, temporal=True))
    total = len(flow)
    features.extend([
        float(total), float(len(own_ids)), float(len(peer_ids)),
        len(own_ids) / max(1, total),
        max(0.0, float(flow[-1]["timestamp"]) - float(flow[0]["timestamp"])) if total else 0.,
        float(length_groups[0].sum()), float(length_groups[1].sum()),
        float(length_groups[2].sum()),
    ])
    return np.asarray(features, dtype=np.float64)


def _select_twenty(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Class-separability ranking without fitting an extra large tree ensemble."""
    scores = np.zeros(x.shape[1], dtype=np.float64)
    grand_mean = np.mean(x, axis=0, dtype=np.float64)
    denom = np.zeros(x.shape[1], dtype=np.float64)
    for label in np.unique(y):
        group = x[y == label].astype(np.float64, copy=False)
        mean = group.mean(axis=0)
        scores += len(group) * (mean - grand_mean) ** 2
        denom += np.square(group - mean).sum(axis=0)
    rank = scores / np.maximum(denom, 1e-9)
    return np.argsort(-rank, kind="stable")[:20].astype(np.int64)


class _Simulator(nn.Module):
    """Small frozen statistical analyzer used solely to supply RL rewards."""

    def __init__(self, n_classes: int, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, n_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _BiLSTMQ(nn.Module):
    """Q(s,a) is discrete: max_a Q(s,a) is then a well-defined Bellman target."""

    def __init__(self, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden, num_layers=2,
                            bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * hidden, n_actions)

    def forward(self, observations: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            observations, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.head(torch.cat((hidden[-2], hidden[-1]), dim=-1))


def _to_batch(histories: Sequence[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not histories or any(np.asarray(s).ndim != 2 or np.asarray(s).shape[1] != 3
                            or not len(s) or not np.isfinite(s).all() for s in histories):
        raise ValueError("DQN histories must be nonempty finite (T,3) sequences")
    lengths = torch.tensor([len(s) for s in histories], dtype=torch.int64)
    longest = int(lengths.max().item())
    batch = torch.zeros((len(histories), longest, 3), dtype=torch.float32, device=device)
    for i, states in enumerate(histories):
        batch[i, :len(states)] = torch.from_numpy(states).to(device)
    return batch, lengths


def _encode_state(length: int, dt: float, remaining: int, max_payload: int,
                  time_scale: float, max_history: int, history: list[list[float]]) -> np.ndarray:
    if (not math.isfinite(dt) or dt < 0 or length < 0 or remaining < 0
            or max_payload < 1 or not math.isfinite(time_scale) or time_scale <= 0
            or max_history < 1):
        raise ValueError("Invalid physical DQN state")
    history.append([length / max_payload,
                    math.log1p(max(0.0, dt)) / max(1e-8, math.log1p(time_scale)),
                    remaining / max_payload])
    if len(history) > max_history:
        del history[:-max_history]
    return np.asarray(history, dtype=np.float32).copy()


def _propose_lengths(total: int, slots_left: int, desired: int,
                     minimum: int, maximum: int) -> int | None:
    """Reserve min/max capacity for all remaining packets; never strand a byte."""
    lower = max(minimum, total - (slots_left - 1) * maximum)
    upper = min(maximum, total - (slots_left - 1) * minimum)
    if lower > upper:
        return None
    return int(max(lower, min(upper, desired)))


def _planning_blocks(blocks: Sequence[Sequence[int]], valid_mask: Any = None) -> list[tuple[int, ...]]:
    limit = 20
    if valid_mask is not None:
        valid = np.asarray(valid_mask)
        if (valid.dtype != np.bool_ or valid.shape != (20,)
                or not np.array_equal(valid, np.arange(20) < valid.sum())):
            raise ValueError("Mirage requires an authoritative first-20 prefix mask")
        limit = int(valid.sum())
    return [tuple(block) for block in blocks if len(block) >= 2
            and all(0 <= index < limit for index in block)]


def _reconstruct_flow(
    flow: Sequence[Mapping[str, Any]], blocks: Sequence[Sequence[int]],
    qnet: _BiLSTMQ, *, length_actions: tuple[int, ...], time_factors: tuple[float, ...],
    min_payload: int, max_payload: int, max_history: int, time_scale: float,
    rng: random.Random | None = None, explore: float = 0.0,
    device: torch.device = torch.device("cpu"), valid_mask: Any = None,
) -> tuple[list[dict[str, Any]] | None, list[tuple[np.ndarray, int, np.ndarray, bool]]]:
    """Eq1-3 next-packet actions; first packet and endpoint/count anchors are local.

    Causal timing spends only the remaining block span. Later choices never
    revise a state already observed by the policy. A straddling block is frozen.
    """
    if (not length_actions or not time_factors or not 0 <= explore <= 1
            or any(not math.isfinite(f) or f <= 0 for f in time_factors)
            or any(not min_payload <= length <= max_payload for length in length_actions)):
        raise ValueError("Invalid Mirage action domain")
    output = list(flow)
    trace = []
    for block in _planning_blocks(blocks, valid_mask):
        pooled = b"".join(flow[i]["payload"] for i in block)
        sizes = [len(flow[i]["payload"]) for i in block]
        times = [float(flow[i]["timestamp"]) for i in block]
        n, total = len(block), len(pooled)
        if not n * min_payload <= total <= n * max_payload:
            return None, []
        gaps = np.diff(np.asarray(times, dtype=np.float64))
        if not np.isfinite(times).all() or np.any(gaps < 0):
            return None, []
        remaining = total - sizes[0]
        history: list[list[float]] = []
        state = _encode_state(sizes[0], 0., remaining, max_payload, time_scale, max_history, history)
        cursor = sizes[0]
        previous_time = times[0]
        for i in range(1, n):
            obs, lens = _to_batch([state], device)
            with torch.inference_mode():
                quality = qnet(obs, lens)
            if quality.shape != (1, len(length_actions)*len(time_factors)) or not torch.isfinite(quality).all():
                raise ValueError("Invalid Mirage Q action vector")
            action = int(quality[0].argmax().item())
            if rng is not None and rng.random() < explore:
                action = rng.randrange(len(length_actions)*len(time_factors))
            length_index, time_index = divmod(action, len(time_factors))
            length = _propose_lengths(remaining, n-i, length_actions[length_index], min_payload, max_payload)
            if length is None:
                return None, []
            factor = time_factors[time_index]
            if i == n-1:
                timestamp = times[-1]
            elif factor == 1. and previous_time == times[i-1]:
                timestamp = times[i]
            else:
                weight = float(gaps[i-1]) * factor
                future = float(gaps[i:].sum())
                denominator = weight + future
                if not math.isfinite(denominator):
                    raise ValueError("Nonfinite Mirage timing weight")
                dt = (times[-1]-previous_time)*(weight/denominator) if denominator else 0.
                timestamp = previous_time + dt
            packet = dict(flow[block[i]])
            packet.update(payload=pooled[cursor:cursor+length],
                          seq=(int(flow[block[0]]["seq"])+cursor) % 2**32,
                          timestamp=timestamp)
            output[block[i]] = packet
            cursor += length
            remaining -= length
            future_state = _encode_state(length, timestamp-previous_time, remaining,
                                         max_payload, time_scale, max_history, history)
            trace.append((state, action, future_state, i == n-1))
            state, previous_time = future_state, timestamp
        if cursor != total or remaining:
            return None, []
    return output, trace


def _rewarded_transitions(flow: Sequence[Mapping[str, Any]], blocks: Sequence[Sequence[int]],
                         trace: Sequence[tuple], reward: Any) -> list[tuple]:
    """Eq3/5 score reconstructed prefixes; each PB terminates its own episode."""
    endpoints = [index+1 for block in blocks for index in block[1:]]
    if len(endpoints) != len(trace):
        raise ValueError("Mirage replay/prefix trajectory mismatch")
    return [(state, action, reward(flow[:end]), future, bool(done))
            for end, (state, action, future, done) in zip(endpoints, trace, strict=True)]


def _topology_verified(original: Sequence[Mapping[str, Any]], changed: Sequence[Mapping[str, Any]],
                       blocks: Sequence[Sequence[int]], minimum: int, maximum: int) -> bool:
    """Independent-in-plugin invariant check, not positional feature immutability."""
    if len(original) != len(changed):
        return False
    if _parse_blocks(changed, minimum, maximum) != list(blocks):
        return False
    editable = {i for block in blocks for i in block}
    for i, (before, after) in enumerate(zip(original, changed)):
        if i not in editable:
            if before is not after and before != after:
                return False
            continue
        if any(before[key] != after[key] for key in before.keys() - {"payload", "seq", "timestamp"}):
            return False
    for block in blocks:
        # Full byte-string equality catches loss, reordering and synthetic data.
        if b"".join(original[i]["payload"] for i in block) != b"".join(changed[i]["payload"] for i in block):
            return False
        if float(original[block[0]]["timestamp"]) != float(changed[block[0]]["timestamp"]):
            return False
        if float(original[block[-1]]["timestamp"]) != float(changed[block[-1]]["timestamp"]):
            return False
    return True


def _transform_statistics(values: np.ndarray, selected: Sequence[int],
                          mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values, mean, scale = np.asarray(values), np.asarray(mean), np.asarray(scale)
    selected = np.asarray(selected)
    if (values.shape != (len(_STAT_NAMES),) or not np.isfinite(values).all() or np.any(values < 0)
            or selected.shape != (20,) or selected.dtype.kind not in "iu"
            or len(np.unique(selected)) != 20 or np.any(selected < 0) or np.any(selected >= len(values))
            or mean.shape != (20,) or scale.shape != (20,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid simulator selected-feature transform")
    x = np.log1p(values[selected].astype(np.float64))
    return np.asarray((x - mean) / scale, dtype=np.float32)


def _fit_simulator(x: np.ndarray, y: np.ndarray, selected: np.ndarray, labels: list[int],
                   *, epochs: int, batch_size: int, lr: float, seed: int) -> tuple[_Simulator, np.ndarray, np.ndarray, float]:
    if (epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0
            or not np.isfinite(x).all() or len(labels) < 2 or len(set(labels)) != len(labels)):
        raise ValueError("Invalid simulator training domain")
    indexed = np.log1p(np.maximum(x[:, selected], 0.0)).astype(np.float64)
    mean, std = indexed.mean(axis=0), indexed.std(axis=0)
    std = np.maximum(std, 1e-6)
    indexed = np.asarray((indexed - mean) / std, dtype=np.float32)
    mapping = {label: i for i, label in enumerate(labels)}
    targets = np.asarray([mapping[int(v)] for v in y], dtype=np.int64)
    net = _Simulator(len(labels))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    sampler = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        for indices in np.array_split(sampler.permutation(len(x)), max(1, math.ceil(len(x) / batch_size))):
            features = torch.from_numpy(indexed[indices])
            truth = torch.from_numpy(targets[indices])
            loss = F.cross_entropy(net(features), truth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            del features, truth, loss
    net.eval()
    with torch.inference_mode():
        correct = 0
        for start in range(0, len(x), batch_size):
            logits = net(torch.from_numpy(indexed[start:start+batch_size]))
            correct += int((logits.argmax(dim=1).numpy() == targets[start:start+batch_size]).sum())
            del logits
    net.requires_grad_(False)
    accuracy = correct / max(1, len(x))
    del indexed, targets
    return net, mean, std, float(accuracy)


def _confidence_reward(probability: float) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Simulator probability must lie in [0,1]")
    return 2.0 / (1.0 + math.exp(1.0 - probability)) - 1.0


def _reward(simulator: _Simulator, flow: Sequence[Mapping[str, Any]], target_col: int,
            selection: Sequence[int], mean: np.ndarray, std: np.ndarray) -> float:
    features = _statistics(flow)
    scaled = _transform_statistics(features, selection, mean, std)
    with torch.inference_mode():
        probability = float(simulator(torch.from_numpy(scaled).view(1, -1)).softmax(-1)[0, target_col].item())
    del features, scaled
    # Paper Eq. (5): reward is in [2/(1+e)-1, 0], increasing in target confidence.
    return _confidence_reward(probability)


def _dqn_update(online: _BiLSTMQ, target: _BiLSTMQ, optimizer: torch.optim.Optimizer,
                replay: deque, rng: random.Random, batch_size: int, gamma: float) -> float:
    if not replay or batch_size < 1 or not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("Invalid Bellman replay settings")
    examples = rng.sample(list(replay), min(batch_size, len(replay)))
    states, actions, rewards, next_states, terminals = zip(*examples)
    values, lengths = _to_batch(states, torch.device("cpu"))
    future, future_lengths = _to_batch(next_states, torch.device("cpu"))
    chosen = online(values, lengths).gather(1, torch.tensor(actions).view(-1, 1)).flatten()
    with torch.no_grad():
        next_max = target(future, future_lengths).max(dim=1).values
        rhs = torch.tensor(rewards, dtype=torch.float32) + gamma * (
            1.0 - torch.tensor(terminals, dtype=torch.float32)
        ) * next_max
    loss = F.smooth_l1_loss(chosen, rhs)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=5.0)
    optimizer.step()
    result = float(loss.item())
    del examples, values, lengths, future, future_lengths, chosen, rhs, loss
    return result


def _surrogate_labels(dataset: Any, physical: np.ndarray, indices: Sequence[int],
                      config: Mapping[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    from workflow.factory import _build_model
    from workflow.io import _load_checkpoint, _load_checkpoint_metadata, _normalization_from_checkpoint
    from workflow.seeds import role_seed
    from core.dual_representation import encode_normalized32
    from core.provenance import frozen_model_state_sha256
    from core.query_meter import ModelQueryMeter
    from models.wrapper import PyTorchModelWrapper

    source = dict(config.get("surrogate_model") or {})
    if not source.get("enabled", True) or not source.get("checkpoint"):
        raise ValueError("Mirage requires an independently pretrained surrogate")
    source_seed = role_seed(config, "surrogate", seed)
    path = Path(str(source["checkpoint"]).format(seed=source_seed))
    meta = _load_checkpoint_metadata(path)
    if tuple(meta["input_shape"]) != tuple(physical.shape[1:]) or physical.dtype != np.float64:
        raise ValueError("Mirage surrogate physical topology/dtype mismatch")
    norm = _normalization_from_checkpoint(meta, tuple(dataset.feature_names), role="Mirage surrogate")
    wrapper = _build_model(source, physical.shape, device=torch.device("cpu"),
                           num_classes=int(meta["num_classes"]), seed=source_seed)
    _load_checkpoint(wrapper, path, checkpoint_data=meta)
    if frozen_model_state_sha256(wrapper) != meta["model_state_sha256"]:
        raise ValueError("Mirage surrogate learned state mismatch")
    valid = np.asarray(dataset.valid_mask)[list(indices)]
    inputs = encode_normalized32(physical[list(indices)], tuple(dataset.feature_names), norm.mean, norm.std)
    labels, costs = [], np.zeros(len(indices), dtype=np.int64)
    with ModelQueryMeter(wrapper) as meter:
        for start in range(0, len(indices), 64):
            stop = min(start+64, len(indices))
            prediction = (wrapper.predict(inputs[start:stop], valid_mask=valid[start:stop])
                          if isinstance(wrapper, PyTorchModelWrapper) else wrapper.predict(inputs[start:stop]))
            labels.extend(np.asarray(prediction, dtype=np.int64).tolist())
            costs[start:stop] += 1
    if meter.counts.inference_records != int(costs.sum()) or meter.counts.gradient_forward_records:
        raise RuntimeError("Mirage surrogate cost mismatch")
    if not set(labels).issubset(set(dataset.class_mapping.values())):
        raise ValueError("Mirage surrogate class schema mismatch")
    return np.asarray(labels, dtype=np.int64), costs


@register_attack("mirage")
class MirageProblemSpaceAttack(ProblemSpaceAttack):
    """Offline payload-only Mirage approximation with *intrinsic* TCP topology proof.

    Neither raw flow edits nor Q estimates imply an accepted attack. The engine
    re-extracts physical packet features and invokes the independent gatekeeper;
    the supplied mask is an additional, necessary (not sufficient) condition.
    """

    attack_name = "mirage"
    query_based = False
    requires_gradients = False

    def __init__(self, *, checkpoint: str = "runs/checkpoints/mirage_seed{seed}.pt",
                 target_label: int = 0, seed: int = 42, device: str = "cpu",
                 min_payload: int = 1, max_payload: int = 1460,
                 max_history: int = 16, time_scale: float = 1.0,
                 length_actions: Sequence[int] = (128, 256, 512, 768, 1024, 1460),
                 time_factors: Sequence[float] = (0.5, 1.0, 2.0)) -> None:
        super().__init__(model=None)
        self.checkpoint = str(checkpoint)
        self.target_label = int(target_label)
        self.seed = int(seed)
        self.device = torch.device(device if device != "auto" else "cpu")
        self.min_payload, self.max_payload = int(min_payload), int(max_payload)
        self.max_history = int(max_history)
        self.time_scale = float(time_scale)
        self.length_actions = tuple(int(x) for x in length_actions)
        self.time_factors = tuple(float(x) for x in time_factors)
        if (self.min_payload < 1 or self.max_payload < self.min_payload or self.max_history < 1
                or not math.isfinite(self.time_scale) or self.time_scale <= 0 or not self.length_actions or not self.time_factors
                or any(x < self.min_payload or x > self.max_payload for x in self.length_actions)
                or any(not math.isfinite(x) or x <= 0 for x in self.time_factors)):
            raise ValueError("Invalid MIRAGE payload/time/action constraints")
        self._loaded_path: Path | None = None
        self._networks: dict[int, _BiLSTMQ] = {}

    def _load(self, seed: int) -> None:
        """Once per worker; avoid reseeding checkpoints on engine batch increments."""
        if self._loaded_path is not None:
            return
        path = Path(self.checkpoint.format(seed=seed)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MIRAGE checkpoint missing: {path}; first run train_attack")
        data = torch.load(path, map_location="cpu", weights_only=True)
        verify_generator_artifact_provenance(data, getattr(self, 'expected_provenance', None))
        if (data.get("attack_name") != self.attack_name or data.get("format_version") != _FORMAT_VERSION
                or data.get("seed") != seed
                or data.get("action_semantics") != "next_packet_local_discretization_causal_anchored_v2"):
            raise ValueError("Unsupported MIRAGE composite checkpoint/seed/semantics")
        classes = sorted(int(v) for v in data["artifact_provenance"]["class_mapping"].values())
        ids = data.get("training_sample_ids", [])
        if (data.get("simulator_labels") != classes
                or set(data.get("q_networks", {})) != {str(v) for v in classes}
                or not ids or len(set(ids)) != len(ids)
                or data.get("surrogate_inference_costs") != [1]*len(ids)
                or data.get("surrogate_gradient_forward_costs") != [0]*len(ids)):
            raise ValueError("MIRAGE class/training-cost schema mismatch")
        _transform_statistics(np.zeros(len(_STAT_NAMES)), np.asarray(data["selected_indices"]),
                              np.asarray(data["simulator_mean"]), np.asarray(data["simulator_std"]))
        for name in ("min_payload", "max_payload", "max_history", "time_scale"):
            if data[name] != getattr(self, name):
                raise ValueError(f"MIRAGE checkpoint constraint mismatch: {name}")
        if (tuple(data["length_actions"]) != self.length_actions or
                tuple(data["time_factors"]) != self.time_factors):
            raise ValueError("MIRAGE checkpoint action grid mismatch")
        n_actions = len(self.length_actions) * len(self.time_factors)
        networks: dict[int, _BiLSTMQ] = {}
        for label, weights in data["q_networks"].items():
            model = _BiLSTMQ(n_actions)
            model.load_state_dict(weights, strict=True)
            model.to(self.device).eval()
            networks[int(label)] = model
        self._networks = networks
        self._loaded_path = path
        del data
        gc.collect()

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Override base adapter, which otherwise discards intrinsic feasibility!"""
        return self.generate_raw(x, y, **kwargs)

    @torch.inference_mode()
    def generate_raw(self, traffic: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Transactional per-flow repackaging with a 1D boolean feasibility proof.

        Input is a batch of mapping-v1 TCP flow packet lists aligned with y;
        each packet supplies direction, timestamp, payload, seq, ack and flags.
        ACK-scoped payload reconstruction is offline, not a live wire adapter.
        Failed flows are emitted unchanged with mask=False.
        """
        seed = int(kwargs.get("seed", self.seed))
        self._load(seed)
        labels = np.asarray(y).reshape(-1)
        if not isinstance(traffic, (list, tuple, np.ndarray)) or len(traffic) != len(labels):
            raise ValueError("MIRAGE requires a batch of raw flows aligned with y")
        from attacks.problem_space.deepred import PcapFlow, _read_pcap
        capture_inputs = [isinstance(flow, (PcapFlow, str, os.PathLike)) for flow in traffic]
        if any(capture_inputs) and not all(capture_inputs):
            raise ValueError('Mirage cannot mix raw-PCAP and mapping-v1 inputs in a batch')
        converted = []
        raw_origins = []
        raw_offsets = []
        if all(capture_inputs):
            for flow in traffic:
                capture = flow if isinstance(flow, PcapFlow) else _read_pcap(flow)
                mapping, offsets, fmt = _pcap_tcp_packets(capture)
                converted.append(mapping)
                raw_origins.append((capture,fmt))
                raw_offsets.append(offsets)
            traffic = converted
        candidate: list[Any] = []
        feasible = np.zeros(len(traffic), dtype=np.bool_)
        qnet = self._networks.get(self.target_label)
        for i, flow in enumerate(traffic):
            output = _pristine(flow)
            if qnet is not None:
                blocks = _parse_blocks(flow, self.min_payload, self.max_payload)
                if blocks is not None:
                    constructed, trace = _reconstruct_flow(
                        flow, blocks, qnet, length_actions=self.length_actions,
                        time_factors=self.time_factors, min_payload=self.min_payload,
                        max_payload=self.max_payload, max_history=self.max_history,
                        time_scale=self.time_scale, device=self.device,
                        valid_mask=None if kwargs.get("valid_mask") is None else kwargs["valid_mask"][i],
                    )
                    if constructed is not None and _topology_verified(
                            flow, constructed, blocks, self.min_payload, self.max_payload):
                        output = tuple(constructed) if isinstance(flow, tuple) else constructed
                        feasible[i] = True
                    del constructed, trace
            if raw_origins:
                capture, fmt = raw_origins[i]
                if feasible[i]:
                    output = _pcap_from_mapping(capture, flow, output,
                                                raw_offsets[i], fmt)
                else:
                    output = capture
            candidate.append(output)
        # Preserve array/list/tuple batch type for the paired raw preprocessor.
        if isinstance(traffic, np.ndarray):
            result_raw = np.empty(len(candidate), dtype=object)
            result_raw[:] = candidate
        elif isinstance(traffic, tuple):
            result_raw = tuple(candidate)
        else:
            result_raw = candidate
        del candidate
        gc.collect()
        return AttackResult(adversarial=None, raw_adversarial=result_raw,
                            intrinsic_feasible_mask=feasible,
                            queries=np.zeros(len(feasible), dtype=np.int64),
                            metadata={"checkpoint": str(self._loaded_path),
                                      "target_label": self.target_label,
                                      "intrinsically_feasible": int(feasible.sum()),
                                      "raw_adapter": "offline_tcp_pcap" if raw_origins else "mapping-v1"})

    @classmethod
    def train(cls, *, config: Mapping[str, Any], seed: int, gpu: str,
              logger: Any = None) -> Mapping[str, float]:
        """Deterministic, decoupled E -> DQN pipeline on raw training-only flows.

        TRAIN-only statistical features teach E from registered pretrained
        surrogate labels. Frozen E supplies Eq5 prefix rewards to separate
        class Q policies. No evaluation target is loaded or queried.
        """
        del gpu  # CPU is deliberate: repeatable BiLSTM behavior across workers.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        settings = dict(config.get("attack", {}).get("params", {}))
        training = dict(config.get("attack", {}).get("training", {}))
        defaults = cls(**{k: v for k, v in settings.items()
                          if k in {"checkpoint", "target_label", "seed", "device", "min_payload",
                                   "max_payload", "max_history", "time_scale",
                                   "length_actions", "time_factors"}})
        from workflow.factory import _inject_dynamic_dataset_shape
        # Source-checkpoint topology defines this authorized training view;
        # the raw PCAP action itself never consumes normalized model arrays.
        source_config = dict(config)
        source_config['model'] = dict(config['surrogate_model'])
        _inject_dynamic_dataset_shape(source_config)
        dataset_cfg = source_config["dataset"]
        dataset = DATASETS.get(str(dataset_cfg["name"]))(**dict(dataset_cfg.get("params", {})))
        raw_features, raw_labels = dataset.raw_arrays()
        raw_features_shape = raw_features.shape[1:]

        labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
        raw_flows = getattr(dataset, "raw_traffic", None)
        if raw_flows is None or len(raw_flows) != len(labels):
            raise ValueError("MIRAGE training requires aligned dataset.raw_traffic and labels")
        if labels.ndim != 1 or np.any(labels < 0):
            raise ValueError("Training labels must be nonnegative class IDs")
        if len(np.unique(labels)) < 2:
            raise ValueError("Simulator requires at least two traffic classes")

        rng = np.random.default_rng(seed)
        # Never choose a private random train/test split: the source surrogate's
        # exact registered training membership is the sole simulator/DQN fit set.
        indices = authorized_attack_training_indices(
            dataset, labels, config, seed, role='surrogate_model')
        cap = int(training.get("max_train_flows", 2048))
        if cap < 2:
            raise ValueError("max_train_flows must be at least 2")
        if len(indices) > cap:
            indices = np.sort(rng.choice(indices, size=cap, replace=False))
        accepted_flows: list[Any] = []
        accepted_indices: list[int] = []
        stats: list[np.ndarray] = []
        from attacks.problem_space.deepred import PcapFlow, _read_pcap
        for idx in indices:
            flow = raw_flows[int(idx)]
            if isinstance(flow, (PcapFlow, str, os.PathLike)):
                capture = flow if isinstance(flow, PcapFlow) else _read_pcap(flow)
                flow, _, _ = _pcap_tcp_packets(capture)
            if _parse_blocks(flow, defaults.min_payload, defaults.max_payload) is None:
                continue
            accepted_flows.append(flow)
            accepted_indices.append(int(idx))
            limit = min(20, int(np.asarray(dataset.valid_mask)[idx].sum()))
            stats.append(_statistics(flow[:limit]))
        if len(stats) < 2:
            raise ValueError("Insufficient ACK-anchored training flows")
        y, inference_costs = _surrogate_labels(dataset, raw_features, accepted_indices, config, seed)
        del raw_features
        x = np.stack(stats, axis=0)
        observed_labels = sorted(int(v) for v in dataset.class_mapping.values())
        selected = _select_twenty(x, y)
        simulator, mean, std, sim_accuracy = _fit_simulator(
            x, y, selected, observed_labels,
            epochs=int(training.get("simulator_epochs", 12)),
            batch_size=int(training.get("batch_size", 64)),
            lr=float(training.get("simulator_learning_rate", 0.001)), seed=seed,
        )
        del x, stats
        gc.collect()

        n_actions = len(defaults.length_actions) * len(defaults.time_factors)
        q_states: dict[str, dict[str, torch.Tensor]] = {}
        losses: list[float] = []
        simulator_costs = np.zeros(len(accepted_indices), dtype=np.int64)
        episodes = int(training.get("episodes_per_class", 128))
        replay_cap = int(training.get("replay_capacity", 2048))
        batch_size = int(training.get("dqn_batch_size", 16))
        gamma = float(training.get("gamma", 0.95))
        sync_every = int(training.get("target_sync_episodes", 16))
        if min(episodes, replay_cap, batch_size, sync_every) < 1 or not 0 <= gamma < 1:
            raise ValueError("Invalid DQN training settings")
        for target_label in observed_labels:
            online = _BiLSTMQ(n_actions)
            target = _BiLSTMQ(n_actions)
            target.load_state_dict(online.state_dict(), strict=True)
            optimizer = torch.optim.Adam(online.parameters(), lr=float(training.get("dqn_learning_rate", 0.0005)))
            replay: deque = deque(maxlen=replay_cap)
            episode_rng = random.Random(seed * 1000003 + target_label)
            target_column = observed_labels.index(target_label)
            for episode in range(episodes):
                index = episode_rng.randrange(len(accepted_flows))
                flow = accepted_flows[index]
                blocks = _parse_blocks(flow, defaults.min_payload, defaults.max_payload)
                assert blocks is not None
                valid = np.asarray(dataset.valid_mask)[accepted_indices[index]]
                blocks = _planning_blocks(blocks, valid)
                epsilon = max(float(training.get("epsilon_final", 0.05)),
                              float(training.get("epsilon_initial", 0.3)) * (1.0 - episode / episodes))
                online.eval()
                with torch.inference_mode():
                    changed, trace = _reconstruct_flow(
                        flow, blocks, online, length_actions=defaults.length_actions,
                        time_factors=defaults.time_factors, min_payload=defaults.min_payload,
                        max_payload=defaults.max_payload, max_history=defaults.max_history,
                        time_scale=defaults.time_scale, rng=episode_rng, explore=epsilon,
                    )
                if changed is not None and trace and _topology_verified(
                        flow, changed, _parse_blocks(flow, defaults.min_payload, defaults.max_payload),
                        defaults.min_payload, defaults.max_payload):
                    replay.extend(_rewarded_transitions(
                        changed, blocks, trace,
                        lambda prefix: _reward(simulator, prefix, target_column, selected, mean, std)))
                    simulator_costs[index] += len(trace)
                del changed, trace
                if len(replay) >= batch_size:
                    online.train()
                    losses.append(_dqn_update(online, target, optimizer, replay,
                                              episode_rng, batch_size, gamma))
                if (episode + 1) % sync_every == 0:
                    target.load_state_dict(online.state_dict(), strict=True)
                if (episode + 1) % 32 == 0:
                    gc.collect()
            q_states[str(target_label)] = {k: v.detach().cpu().clone()
                                           for k, v in online.state_dict().items()}
            del online, target, optimizer, replay
            gc.collect()

        payload: dict[str, Any] = {
            "attack_name": "mirage", "format_version": _FORMAT_VERSION,
            "seed": int(seed), "raw_adapter": "mapping-v1_or_verified_offline_pcap",
            "artifact_provenance": trained_generator_provenance(
                dataset, labels, config, seed, source_role='surrogate_model',
                input_shape=tuple(raw_features_shape), feature_names=tuple(dataset.feature_names)),
            "min_payload": defaults.min_payload, "max_payload": defaults.max_payload,
            "max_history": defaults.max_history, "time_scale": defaults.time_scale,
            "length_actions": list(defaults.length_actions),
            "time_factors": list(defaults.time_factors),
            "statistic_names": list(_STAT_NAMES), "selected_indices": selected.tolist(),
            "selected_names": [_STAT_NAMES[i] for i in selected],
            "simulator_mean": torch.from_numpy(mean.copy()),
            "simulator_std": torch.from_numpy(std.copy()),
            "simulator_labels": observed_labels,
            "simulator_state": {k: v.detach().cpu().clone()
                                for k, v in simulator.state_dict().items()},
            "q_networks": q_states,
            "action_semantics": "next_packet_local_discretization_causal_anchored_v2",
            "training_sample_ids": [str(dataset.sample_ids[i]) for i in accepted_indices],
            "surrogate_inference_costs": inference_costs.tolist(),
            "simulator_reward_inference_costs": simulator_costs.tolist(),
            "surrogate_gradient_forward_costs": [0] * len(accepted_indices),
            "seed_roles": dict(config.get("seed_roles", {})),
        }
        checkpoint_template = settings.get("checkpoint") or str(
            Path(str(config.get("output_dir", "runs"))) / "checkpoints" / "mirage_seed{seed}.pt")
        path = Path(str(checkpoint_template).format(seed=seed)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        metrics = {"mirage_train_flows": float(len(accepted_flows)),
                   "mirage_trained_classes": float(len(q_states)),
                   "mirage_simulator_accuracy": sim_accuracy,
                   "surrogate_training_inference_records": float(inference_costs.sum()),
                   "surrogate_training_gradient_forward_records": 0.,
                   "mirage_simulator_reward_inference_records": float(simulator_costs.sum()),
                   "mirage_mean_dqn_loss": float(np.mean(losses)) if losses else 0.0}
        if logger:
            logger.info("mirage.train checkpoint=%s metrics=%s", path, metrics)
        del accepted_flows, accepted_indices, simulator, q_states, payload
        gc.collect()
        return metrics
