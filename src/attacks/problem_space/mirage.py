"""MIRAGE: offline, fail-closed TCP packet-block adversarial benchmark plugin.

Raw adapter contract (mapping-v1): dataset.raw_traffic is a batch of flows;
each flow is a list/tuple of mappings with `direction` (0=attacker,
1=victim), `timestamp` (seconds), `payload` (immutable bytes), `seq`
(unsigned TCP sequence number), `ack` (unsigned TCP acknowledgment number),
and `flags` (TCP flags as letters, e.g. 'A', 'PA'). Other keys are retained.
The paired preprocessor MUST accept this same mapping representation, re-extract
physical float64 features, and export its generated `valid_mask`.

This module NEVER transmits traffic or fabricates arbitrary TCP packets. It
resegments existing attacker bytes within acknowledged packet blocks. Unknown
packet encodings, packets needing wire-level checksum repair, retransmissions,
partial ACKs, and unacknowledged blocks fail closed. A separate PCAP/Scapy
adapter must be supplied by the dataset/preprocessor integration; never
silently reinterpret one as mapping-v1.

The paper describes two-output BiLSTM approximation to continuous Q(s,a), but
DQN requires a discrete action for a Bellman max. This implementation uses a
finite, configurable (payload-length, inter-arrival-factor) action grid and a
2-layer bidirectional LSTM Q head over the grid. The simulator uses the 20
most discriminative features selected *from this dataset* by a deterministic
univariate between/within-class ranking (not an undocumented paper list).
"""

from __future__ import annotations

import gc
import math
import os
import random
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

_FORMAT_VERSION = 1
_FORBIDDEN_PACKET_KEYS = frozenset({"wire_bytes", "raw", "checksum", "ip_checksum", "tcp_checksum",
                                   "len", "packet_len", "packet_length", "payload_len",
                                   "payload_length", "ip_len", "total_length",
                                   "tcp_len", "ip_total_length", "frame_len"})
_REQUIRED = frozenset({"direction", "timestamp", "payload", "seq", "ack", "flags"})


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
        max(0.0, float(flow[-1]["timestamp"]) - float(flow[0]["timestamp"])),
        float(length_groups[0].sum()), float(length_groups[1].sum()),
        float(length_groups[2].sum()),
    ])
    return np.asarray(features, dtype=np.float32)


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
    lengths = torch.tensor([len(s) for s in histories], dtype=torch.int64)
    longest = int(lengths.max().item())
    batch = torch.zeros((len(histories), longest, 3), dtype=torch.float32, device=device)
    for i, states in enumerate(histories):
        batch[i, :len(states)] = torch.from_numpy(states).to(device)
    return batch, lengths


def _encode_state(length: int, dt: float, remaining: int, max_payload: int,
                  time_scale: float, max_history: int, history: list[list[float]]) -> np.ndarray:
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


def _reconstruct_flow(
    flow: Sequence[Mapping[str, Any]], blocks: Sequence[Sequence[int]],
    qnet: _BiLSTMQ, *, length_actions: tuple[int, ...], time_factors: tuple[float, ...],
    min_payload: int, max_payload: int, max_history: int, time_scale: float,
    rng: random.Random | None = None, explore: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[list[dict[str, Any]] | None, list[tuple[np.ndarray, int, np.ndarray, bool]]]:
    """Construct complete PB candidates transactionally; never edit victim packets.

    For each PB, sum(payload lengths) and concatenated bytes are invariants.
    The final segment is the exact residual; earlier choices are clamped to
    reserve both minimum and maximum capacities of later packets. First/last
    times anchor the block so that a frozen ACK never moves in time.
    """
    output: list[dict[str, Any]] = list(flow)  # untouched victim objects remain identical
    trace: list[tuple[np.ndarray, int, np.ndarray, bool]] = []
    for block in blocks:
        if len(block) < 2:
            return None, []
        segments = [flow[i]["payload"] for i in block]
        pooled = b"".join(segments)
        sizes = [len(p) for p in segments]
        n, total = len(block), len(pooled)
        if not n * min_payload <= total <= n * max_payload:
            del pooled, segments
            return None, []
        original_times = [float(flow[i]["timestamp"]) for i in block]
        original_iats = [max(0.0, original_times[i] - original_times[i-1])
                         for i in range(1, n)]
        next_sizes: list[int] = []
        weights: list[float] = []
        choices: list[tuple[np.ndarray, int, np.ndarray, bool]] = []
        history: list[list[float]] = []
        remaining = total
        state = _encode_state(sizes[0], 0.0, remaining, max_payload, time_scale,
                              max_history, history)
        for i in range(n - 1):
            obs, lens = _to_batch([state], device)
            with torch.inference_mode():
                quality = qnet(obs, lens)[0]
                action = int(torch.argmax(quality).item())
            del obs, lens, quality
            if rng is not None and rng.random() < explore:
                action = rng.randrange(len(length_actions) * len(time_factors))
            length_index, time_index = divmod(action, len(time_factors))
            proposed = _propose_lengths(total=remaining,
                                        slots_left=n-i, desired=length_actions[length_index],
                                        minimum=min_payload, maximum=max_payload)
            if proposed is None:
                del pooled, segments
                return None, []
            next_sizes.append(proposed)
            remaining -= proposed
            weights.append(max(1e-9, original_iats[i]) * time_factors[time_index])
            next_state = _encode_state(proposed, original_iats[i], remaining,
                                       max_payload, time_scale, max_history, history)
            choices.append((state, action, next_state, i == n - 2))
            state = next_state
        if not min_payload <= remaining <= max_payload:
            del pooled, segments
            return None, []
        next_sizes.append(remaining)
        span = original_times[-1] - original_times[0]
        weight_sum = sum(weights)
        proposed_times = [original_times[0]]
        partial = 0.0
        for weight in weights[:-1]:
            partial += weight
            proposed_times.append(original_times[0] + span * partial / weight_sum)
        proposed_times.append(original_times[-1])

        cursor = 0
        start_seq = int(flow[block[0]]["seq"])
        for i, length, timestamp in zip(block, next_sizes, proposed_times):
            packet = dict(flow[i])
            payload = pooled[cursor:cursor + length]
            if len(payload) != length:
                del pooled, segments
                return None, []
            packet["payload"] = payload
            packet["seq"] = (start_seq + cursor) % (2**32)
            packet["timestamp"] = timestamp
            output[i] = packet
            cursor += length
        if cursor != total:
            del pooled, segments
            return None, []
        trace.extend(choices)
        del pooled, segments, sizes, next_sizes, weights, choices

    return output, trace


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
    x = np.log1p(np.maximum(values[np.asarray(selected, dtype=np.int64)], 0.0))
    return np.asarray((x - mean) / scale, dtype=np.float32)


def _fit_simulator(x: np.ndarray, y: np.ndarray, selected: np.ndarray, labels: list[int],
                   *, epochs: int, batch_size: int, lr: float, seed: int) -> tuple[_Simulator, np.ndarray, np.ndarray, float]:
    indexed = np.log1p(np.maximum(x[:, selected], 0.0)).astype(np.float32)
    mean, std = indexed.mean(axis=0), indexed.std(axis=0)
    std = np.maximum(std, 1e-6)
    indexed = (indexed - mean) / std
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
    accuracy = correct / max(1, len(x))
    del indexed, targets
    return net, mean, std, float(accuracy)


def _reward(simulator: _Simulator, flow: Sequence[Mapping[str, Any]], target_col: int,
            selection: Sequence[int], mean: np.ndarray, std: np.ndarray) -> float:
    features = _statistics(flow)
    scaled = _transform_statistics(features, selection, mean, std)
    with torch.inference_mode():
        probability = float(simulator(torch.from_numpy(scaled).view(1, -1)).softmax(-1)[0, target_col].item())
    del features, scaled
    # Paper Eq. (5): reward is in [2/(1+e)-1, 0], increasing in target confidence.
    return 2.0 / (1.0 + math.exp(1.0 - probability)) - 1.0


def _dqn_update(online: _BiLSTMQ, target: _BiLSTMQ, optimizer: torch.optim.Optimizer,
                replay: deque, rng: random.Random, batch_size: int, gamma: float) -> float:
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
                or self.time_scale <= 0 or not self.length_actions or not self.time_factors
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
        if data.get("attack_name") != self.attack_name or data.get("format_version") != _FORMAT_VERSION:
            raise ValueError("Unsupported MIRAGE composite checkpoint")
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

        Failed flows are emitted precisely as provided, with mask=False. No
        post-hoc row alignment or victim restoration is trusted or required.
        """
        seed = int(kwargs.get("seed", self.seed))
        self._load(seed)
        labels = np.asarray(y).reshape(-1)
        if not isinstance(traffic, (list, tuple, np.ndarray)) or len(traffic) != len(labels):
            raise ValueError("MIRAGE requires a batch of raw flows aligned with y")
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
                    )
                    if constructed is not None and _topology_verified(
                            flow, constructed, blocks, self.min_payload, self.max_payload):
                        output = tuple(constructed) if isinstance(flow, tuple) else constructed
                        feasible[i] = True
                    del constructed, trace
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
                                      "raw_adapter": "mapping-v1"})

    @classmethod
    def train(cls, *, config: Mapping[str, Any], seed: int, gpu: str,
              logger: Any = None) -> Mapping[str, float]:
        """Deterministic, decoupled E -> DQN pipeline on raw training-only flows.

        Phase A selects 20 statistical features on the *raw* packet dataset,
        then fits E. Phase B freezes E, trains one Bellman Q-network per
        destination class using episodes over offline flows, and serializes
        every model into a single state-only .pt checkpoint. No target NIDS
        queries/gradients are used, preserving independent benchmark scoring.
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
        dataset_cfg = config["dataset"]
        dataset = DATASETS.get(str(dataset_cfg["name"]))(**dict(dataset_cfg.get("params", {})))
        raw_features, raw_labels = dataset.raw_arrays()
        del raw_features  # never retain a second corpus-sized feature tensor
        labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
        raw_flows = getattr(dataset, "raw_traffic", None)
        if raw_flows is None or len(raw_flows) != len(labels):
            raise ValueError("MIRAGE training requires aligned dataset.raw_traffic and labels")
        if labels.ndim != 1 or np.any(labels < 0):
            raise ValueError("Training labels must be nonnegative class IDs")
        if len(np.unique(labels)) < 2:
            raise ValueError("Simulator requires at least two traffic classes")

        rng = np.random.default_rng(seed)
        # Honor an explicit training partition, otherwise reserve class-wise 20%.
        provided = getattr(dataset, "train_indices", None)
        if provided is not None:
            indices = np.sort(np.asarray(provided, dtype=np.int64))
        else:
            partitions = []
            for label in np.unique(labels):
                matching = np.flatnonzero(labels == label)
                rng.shuffle(matching)
                partitions.append(matching[:max(1, int(math.floor(0.8 * len(matching))))])
            indices = np.sort(np.concatenate(partitions))
        cap = int(training.get("max_train_flows", 2048))
        if cap < 2:
            raise ValueError("max_train_flows must be at least 2")
        if len(indices) > cap:
            indices = np.sort(rng.choice(indices, size=cap, replace=False))
        accepted_flows: list[Any] = []
        accepted_labels: list[int] = []
        stats: list[np.ndarray] = []
        for idx in indices:
            flow = raw_flows[int(idx)]
            if _parse_blocks(flow, defaults.min_payload, defaults.max_payload) is None:
                continue
            accepted_flows.append(flow)
            accepted_labels.append(int(labels[idx]))
            stats.append(_statistics(flow))
        if len(stats) < 2 or len(set(accepted_labels)) < 2:
            raise ValueError("Insufficient ACK-anchored training flows across >=2 classes")
        x = np.stack(stats, axis=0)
        y = np.asarray(accepted_labels, dtype=np.int64)
        observed_labels = sorted(int(v) for v in np.unique(y))
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
                        flow, changed, blocks, defaults.min_payload, defaults.max_payload):
                    score = _reward(simulator, changed, target_column, selected, mean, std)
                    # Only the completed, physically conserved episode earns E's reward.
                    for j, (s, action, s_next, done) in enumerate(trace):
                        replay.append((s, action, score if j == len(trace)-1 else 0.0,
                                       s_next, float(j == len(trace)-1)))
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
            "seed": int(seed), "raw_adapter": "mapping-v1",
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
                   "mirage_mean_dqn_loss": float(np.mean(losses)) if losses else 0.0}
        if logger:
            logger.info("mirage.train checkpoint=%s metrics=%s", path, metrics)
        del accepted_flows, accepted_labels, simulator, q_states, payload
        gc.collect()
        return metrics
