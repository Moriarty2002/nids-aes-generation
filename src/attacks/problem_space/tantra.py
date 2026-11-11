"""TANTRA: offline, fail-closed LSTM timing reshaping for mapping-v1 flows.

Only attacker-owned packet timestamps may change. This is a *problem-space*
benchmark generator: it neither transmits packets nor modifies payloads or TCP
headers. The paired preprocessor is responsible for extracting authoritative
float64 features and an output valid_mask before independent gatekeeping.

The LSTM estimates a nonnegative next inter-arrival time (IAT) from the prior
W (payload length, IAT) pairs. Inference is forward-only (no packet may be sent
earlier than its baseline), with cumulative delay within each attacker run and
re-anchoring at untouched victim packets. A proposed crossing of an upcoming
victim timestamp INVALIDATES the entire flow; it is never silently repaired.
"""

from __future__ import annotations

import math
import os
import struct
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
_REQUIRED = frozenset({"direction", "timestamp", "payload", "seq", "ack", "flags"})
# Without an adapter capable of repairing/recomputing these derived/wire fields,
# changing the timestamp would make their representation unverifiable.
_FORBIDDEN_PACKET_KEYS = frozenset({
    "wire_bytes", "raw", "checksum", "ip_checksum", "tcp_checksum", "len",
    "packet_len", "packet_length", "payload_len", "payload_length",
    "ip_len", "total_length", "tcp_len", "ip_total_length", "frame_len",
})


class _TantraLSTM(nn.Module):
    """Many-to-one IAT regressor; output is nonnegative, in time_scale units."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 1), nn.Softplus(),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != 2:
            raise ValueError("Expected a (batch, window_size, 2) tensor")
        _, (hidden, _) = self.lstm(windows)
        return self.head(hidden[-1])  # (batch, 1); softplus >= 0


def _packet_valid(packet: Any, max_payload: int) -> bool:
    """Accept only well-defined mapping-v1 metadata and immutable byte payloads."""
    if not isinstance(packet, Mapping) or not _REQUIRED.issubset(packet):
        return False
    if _FORBIDDEN_PACKET_KEYS.intersection(packet):
        return False
    direction = packet["direction"]
    if (isinstance(direction, (bool, np.bool_))
            or not isinstance(direction, (int, np.integer))
            or direction not in (0, 1)):
        return False
    if type(packet["payload"]) is not bytes or len(packet["payload"]) > max_payload:
        return False
    for name in ("seq", "ack"):
        value = packet[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not (0 <= value < (1 << 32))):
            return False
    if not isinstance(packet["flags"], str):
        return False
    raw_ts = packet["timestamp"]
    if (isinstance(raw_ts, (bool, np.bool_))
            or not isinstance(raw_ts, (int, float, np.integer, np.floating))):
        return False
    if isinstance(raw_ts, np.floating) and raw_ts.dtype.itemsize > 8:
        # A long-double timestamp cannot be treated as authoritative float64
        # without potentially silently changing the recorded instant.
        return False
    try:
        value = float(raw_ts)
    except (ValueError, TypeError, OverflowError):
        return False
    if not math.isfinite(value):
        return False
    # The authoritative physical clock is float64. Integers larger than 2**53
    # lose integer exactness when represented in float64 and must fail closed.
    if isinstance(raw_ts, (int, np.integer)) and int(raw_ts) != int(value):
        return False
    return True


def _read_flow(flow: Any, max_payload: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Read exact physical timestamps; reject bad schema/order before mutation."""
    if not isinstance(flow, (list, tuple, np.ndarray)) or len(flow) < 2:
        return None
    if isinstance(flow, np.ndarray) and flow.ndim != 1:
        return None
    if not all(_packet_valid(packet, max_payload) for packet in flow):
        return None
    times = np.fromiter((float(p["timestamp"]) for p in flow), dtype=np.float64,
                        count=len(flow))
    if not np.all(np.isfinite(times)) or np.any(times[1:] < times[:-1]):
        return None
    lengths = np.fromiter((len(p["payload"]) for p in flow), dtype=np.float64,
                          count=len(flow))
    return times, lengths


def _exact_value_equal(left: Any, right: Any) -> bool:
    """No numerical tolerance for protected fields (including -0.0 timestamps)."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, (float, np.floating)):
        try:
            return struct.pack("!d", float(left)) == struct.pack("!d", float(right))
        except (OverflowError, ValueError):
            return False
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _exact_value_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _exact_value_equal(left[k], right[k]) for k in left
        )
    try:
        result = left == right
        return isinstance(result, (bool, np.bool_)) and bool(result)
    except (TypeError, ValueError):
        return False


def _intrinsic_valid(original: Any, candidate: Any, max_payload: int) -> bool:
    """Topological proof independent of the time generation algorithm.

    Every packet remains at the same index with the same keys and payload;
    only the timestamp of a direction=0 packet may differ. Victim anchors,
    complete TCP metadata, and all other keys are byte/exact-value immutable.
    The earliest subsequent victim must not precede any edited attacker packet.
    """
    clean = _read_flow(original, max_payload)
    changed = _read_flow(candidate, max_payload)
    if clean is None or changed is None or len(original) != len(candidate):
        return False
    original_times, _ = clean
    edited_times, _ = changed
    next_victim = math.inf
    for i in range(len(original) - 1, -1, -1):
        src, dst = original[i], candidate[i]
        if src.keys() != dst.keys():
            return False
        for key in src:
            if key != "timestamp" and not _exact_value_equal(src[key], dst[key]):
                return False
        if int(src["direction"]) == 1:
            if not _exact_value_equal(src["timestamp"], dst["timestamp"]):
                return False
            next_victim = original_times[i]
        else:
            if edited_times[i] < original_times[i]:  # delays only
                return False
            if edited_times[i] > next_victim:  # cannot cross a frozen response
                return False
    # _read_flow(candidate) also proved nondecreasing order and finite times.
    return True


def _normalized_series(
    times: np.ndarray, sizes: np.ndarray, *, max_payload: int, time_scale: float,
) -> np.ndarray | None:
    """Feature k is (size_k/max_payload, (time_k-time_{k-1})/time_scale)."""
    iats = np.diff(times, prepend=times[0])
    series64 = np.column_stack((sizes / max_payload, iats / time_scale))
    if not np.all(np.isfinite(series64)):
        return None
    series = series64.astype(np.float32)
    if not np.all(np.isfinite(series)):
        return None
    return series


def _reshape_flow(
    flow: Sequence[Mapping[str, Any]], net: _TantraLSTM, *,
    device: torch.device, window_size: int, max_payload: int, time_scale: float,
) -> list[Mapping[str, Any]] | None:
    """Forward-only, transactionally compute timestamps, or return None.

    For an attacker packet i, with original timestamp t_i and delay carry c,
    let p_i be the LSTM's predicted *IAT* in seconds. The proposed timestamp is

        t'_i = max(t_i + c, t'_(i-1) + p_i).

    Thus delay never decreases within an attacker run. A frozen victim packet
    resets c to 0 *after* the prior run has passed all anchor checks. This is a
    monotone-delay approximation, not a claim that an offline log can model
    all online TCP feedback/response mechanisms.
    """
    parsed = _read_flow(flow, max_payload)
    if parsed is None:
        return None
    original_times, sizes = parsed
    series = _normalized_series(original_times, sizes,
                                max_payload=max_payload, time_scale=time_scale)
    if series is None or len(flow) <= window_size:
        return None

    next_anchor = np.full(len(flow), math.inf, dtype=np.float64)
    closest = math.inf
    for i in range(len(flow) - 1, -1, -1):
        next_anchor[i] = closest
        if int(flow[i]["direction"]) == 1:
            closest = float(original_times[i])

    output: list[Mapping[str, Any]] = list(flow)
    revised_times = original_times.copy()
    carry = 0.0
    any_changed = False
    window_tensor = torch.empty((1, window_size, 2), device=device, dtype=torch.float32)

    for i in range(1, len(flow)):
        base_t = float(original_times[i])
        previous_t = float(revised_times[i - 1])
        if int(flow[i]["direction"]) == 1:
            # Victim timestamp and payload remain exactly as supplied.
            if base_t < previous_t:
                return None
            carry = 0.0
            updated_t = base_t
        elif i < window_size:
            # No full, training-compatible W-step history yet: preserve prefix.
            updated_t = base_t
        else:
            history = np.ascontiguousarray(series[i - window_size:i])
            window_tensor.copy_(torch.from_numpy(history))
            scaled_prediction = float(net(window_tensor).item())
            predicted_iat = scaled_prediction * time_scale
            if not math.isfinite(predicted_iat) or predicted_iat < 0:
                return None
            updated_t = max(base_t + carry, previous_t + predicted_iat)
            if not math.isfinite(updated_t) or updated_t > next_anchor[i]:
                return None
            carry = max(carry, updated_t - base_t)
            if updated_t > base_t:
                edited_packet = dict(flow[i])
                edited_packet["timestamp"] = updated_t
                output[i] = edited_packet
                any_changed = True

        if updated_t < previous_t or not math.isfinite(updated_t):
            return None
        revised_times[i] = updated_t
        scaled_iat = (updated_t - previous_t) / time_scale
        if not math.isfinite(scaled_iat) or scaled_iat > np.finfo(np.float32).max:
            return None
        series[i, 1] = np.float32(scaled_iat)

    return output if any_changed and _intrinsic_valid(flow, output, max_payload) else None


@register_attack("tantra")
class TantraProblemSpaceAttack(ProblemSpaceAttack):
    """Seed-local, read-only checkpoint caching and intrinsic fail-closed mutation."""

    attack_name = "tantra"
    query_based = False
    requires_gradients = False

    def __init__(
        self, *, checkpoint: str = "runs/checkpoints/tantra_seed{seed}.pt",
        window_size: int = 50, target_label: int = 0,
        max_payload: int = 1460, time_scale: float = 1.0,
        hidden_size: int = 64, seed: int = 42, device: str = "cpu",
    ) -> None:
        super().__init__(model=None)
        self.checkpoint = str(checkpoint)
        self.window_size = int(window_size)
        self.target_label = int(target_label)
        self.max_payload = int(max_payload)
        self.time_scale = float(time_scale)
        self.hidden_size = int(hidden_size)
        self.seed = int(seed)
        self.device = torch.device("cpu" if device == "auto" else device)
        if (self.window_size < 1 or self.max_payload < 1 or self.hidden_size < 1
                or not math.isfinite(self.time_scale) or self.time_scale <= 0
                or self.target_label != 0):
            raise ValueError("Invalid TANTRA configuration (target_label must be benign 0)")
        self._loaded_path: Path | None = None
        self._network: _TantraLSTM | None = None

    def _load(self, seed: int) -> None:
        """Load exactly once in the worker, including across engine batch offsets."""
        if self._loaded_path is not None:
            return
        path = Path(self.checkpoint.format(seed=int(seed))).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TANTRA checkpoint missing: {path}; run train_attack first")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data.get("attack_name") != self.attack_name
                or data.get("format_version") != _FORMAT_VERSION
                or data.get("raw_adapter") != "mapping-v1"):
            raise ValueError("Unsupported TANTRA checkpoint/adapter")
        for name in ("window_size", "hidden_size", "max_payload", "time_scale"):
            if data.get(name) != getattr(self, name):
                raise ValueError(f"TANTRA checkpoint normalization/constraint mismatch: {name}")
        required_norm = {
            "payload_divisor": self.max_payload, "iat_divisor": self.time_scale,
        }
        required_constraints = {
            "editable_direction": 0, "victim_direction": 1,
            "forward_only": True, "victim_anchor_immutable": True,
            "packet_count_immutable": True, "payload_immutable": True,
            "tcp_metadata_immutable": True,
        }
        if (data.get("normalization") != required_norm
                or data.get("constraints") != required_constraints):
            raise ValueError("TANTRA checkpoint physical contract mismatch")
        net = _TantraLSTM(hidden_size=self.hidden_size)
        net.load_state_dict(data["model_state"], strict=True)
        self._network = net.to(self.device).eval()
        self._loaded_path = path
        del data

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        # The base ProblemSpaceAttack.generate() would discard the intrinsic mask.
        return self.generate_raw(x, y, **kwargs)

    @torch.inference_mode()
    def generate_raw(self, traffic: Any, y: Any, **kwargs: Any) -> AttackResult:
        self._load(int(kwargs.get("seed", self.seed)))
        if not isinstance(traffic, (list, tuple, np.ndarray)):
            raise TypeError("TANTRA requires a batch of mapping-v1 packet flows")
        if isinstance(traffic, np.ndarray) and traffic.ndim != 1:
            raise ValueError("TANTRA expects a 1D object batch of flows")
        labels = np.asarray(y).reshape(-1)
        if labels.shape != (len(traffic),) or labels.dtype.kind not in "iu":
            raise ValueError("TANTRA requires integer labels aligned with raw traffic")
        assert self._network is not None

        raw: list[Any] = []
        feasible = np.zeros(len(traffic), dtype=np.bool_)
        for i, flow in enumerate(traffic):
            output: Any = flow  # failures must never escape as partially edited flows
            if labels[i] != 0:
                with torch.inference_mode():
                    candidate = _reshape_flow(
                        flow, self._network, device=self.device,
                        window_size=self.window_size, max_payload=self.max_payload,
                        time_scale=self.time_scale,
                    )
                if candidate is not None:
                    output = tuple(candidate) if isinstance(flow, tuple) else candidate
                    feasible[i] = True
            raw.append(output)
        if isinstance(traffic, np.ndarray):
            result_raw = np.empty(len(raw), dtype=object)
            result_raw[:] = raw
        elif isinstance(traffic, tuple):
            result_raw = tuple(raw)
        else:
            result_raw = raw
        return AttackResult(
            adversarial=None,
            raw_adversarial=result_raw,
            intrinsic_feasible_mask=feasible,
            queries=np.zeros(len(traffic), dtype=np.int64),
            metadata={
                "checkpoint": str(self._loaded_path), "raw_adapter": "mapping-v1",
                "target_label": self.target_label,
                "intrinsically_feasible": int(feasible.sum()),
                "failed_or_unmodified": int((~feasible).sum()),
                "timestamp_only": True,
            },
        )

    @classmethod
    def train(
        cls, *, config: Mapping[str, Any], seed: int, gpu: str,
        logger: Any = None,
    ) -> Mapping[str, float]:
        """Fit only on benign, training-partition mapping-v1 flows.

        A bounded, uniformly sampled window reservoir avoids retaining all
        dataset windows or creating extra spawn workers. LSTM training uses CPU
        deliberately; the evaluator may independently select an inference GPU.
        """
        del gpu
        torch.manual_seed(int(seed))
        rng = np.random.default_rng(int(seed))
        settings = dict(config.get("attack", {}).get("params", {}))
        training = dict(config.get("attack", {}).get("training", {}))
        instance = cls(**{name: value for name, value in settings.items() if name in {
            "checkpoint", "window_size", "target_label", "max_payload", "time_scale",
            "hidden_size", "seed", "device",
        }})
        cap_flows = int(training.get("max_train_flows", 2048))
        cap_windows = int(training.get("max_train_windows", 50000))
        epochs = int(training.get("epochs", 12))
        batch_size = int(training.get("batch_size", 64))
        lr = float(training.get("learning_rate", 1e-3))
        if (min(cap_flows, cap_windows, epochs, batch_size) < 1
                or not math.isfinite(lr) or lr <= 0):
            raise ValueError("Invalid TANTRA training hyperparameters")

        dataset_cfg = config["dataset"]
        dataset = DATASETS.get(str(dataset_cfg["name"]))(**dict(dataset_cfg.get("params", {})))
        raw_features, raw_labels = dataset.raw_arrays()
        del raw_features  # never retain the dataset's full continuous feature corpus
        labels = np.asarray(raw_labels).reshape(-1)
        raw_flows = getattr(dataset, "raw_traffic", None)
        if raw_flows is None or len(raw_flows) != len(labels):
            raise ValueError("TANTRA training requires aligned dataset.raw_traffic and labels")

        benign = np.flatnonzero(labels == 0)
        indices = getattr(dataset, "train_indices", None)
        if indices is not None:
            train_indices = np.asarray(indices, dtype=np.int64)
            if (train_indices.ndim != 1 or np.any(train_indices < 0)
                    or np.any(train_indices >= len(labels))):
                raise ValueError("Invalid dataset.train_indices")
            benign = np.intersect1d(benign, train_indices, assume_unique=False)
        else:
            # Match MIRAGE: hold out ~20% of benign flows unless an explicit
            # training split is provided. Never train using validation labels.
            rng.shuffle(benign)
            benign = benign[:max(1, int(math.floor(0.8 * len(benign))))]
        if len(benign) > cap_flows:
            benign = rng.choice(benign, size=cap_flows, replace=False)
        if len(benign) == 0:
            raise ValueError("No benign training flows available for TANTRA")

        X_bank = np.empty((cap_windows, instance.window_size, 2), dtype=np.float32)
        Y_bank = np.empty((cap_windows, 1), dtype=np.float32)
        seen = 0
        accepted_flows = 0
        for index in benign:
            parsed = _read_flow(raw_flows[int(index)], instance.max_payload)
            if parsed is None:
                continue
            times, sizes = parsed
            if len(times) <= instance.window_size:
                continue
            series = _normalized_series(times, sizes, max_payload=instance.max_payload,
                                        time_scale=instance.time_scale)
            if series is None:
                continue
            # sliding_window_view produces (L-W+1, 2, W); transpose to (N,W,2).
            windows = np.lib.stride_tricks.sliding_window_view(
                series, window_shape=instance.window_size, axis=0,
            ).transpose(0, 2, 1)[:-1]
            targets = series[instance.window_size:, 1]
            if not len(windows):
                continue
            accepted_flows += 1
            for x_win, y_iat in zip(windows, targets):
                seen += 1
                if seen <= cap_windows:
                    pos = seen - 1
                else:
                    pos = int(rng.integers(0, seen))
                    if pos >= cap_windows:
                        continue
                X_bank[pos] = x_win
                Y_bank[pos, 0] = y_iat
        n_samples = min(seen, cap_windows)
        if n_samples == 0:
            raise ValueError(
                "No valid benign windows: check mapping-v1 schema and window_size"
            )
        X = torch.from_numpy(X_bank[:n_samples])
        Y = torch.from_numpy(Y_bank[:n_samples])
        net = _TantraLSTM(hidden_size=instance.hidden_size)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        order_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        weighted_loss, examples_seen = 0.0, 0
        net.train()
        for _ in range(epochs):
            order = torch.randperm(n_samples, generator=order_generator)
            for start in range(0, n_samples, batch_size):
                ids = order[start:start + batch_size]
                prediction = net(X[ids])
                loss = F.mse_loss(prediction, Y[ids])
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("Non-finite TANTRA training loss; inspect IAT scaling")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()
                batch_len = len(ids)
                weighted_loss += float(loss.detach().item()) * batch_len
                examples_seen += batch_len
                del prediction, loss, ids
            del order
        net.eval()
        state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        payload: dict[str, Any] = {
            "attack_name": "tantra", "format_version": _FORMAT_VERSION,
            "raw_adapter": "mapping-v1", "seed": int(seed),
            "window_size": instance.window_size,
            "hidden_size": instance.hidden_size,
            "max_payload": instance.max_payload,
            "time_scale": instance.time_scale,
            "normalization": {"payload_divisor": instance.max_payload,
                              "iat_divisor": instance.time_scale},
            "constraints": {
                "editable_direction": 0, "victim_direction": 1,
                "forward_only": True, "victim_anchor_immutable": True,
                "packet_count_immutable": True, "payload_immutable": True,
                "tcp_metadata_immutable": True,
            },
            "model_state": state,
        }
        checkpoint_template = settings.get("checkpoint") or str(
            Path(str(config.get("output_dir", "runs"))) / "checkpoints" / "tantra_seed{seed}.pt"
        )
        path = Path(str(checkpoint_template).format(seed=int(seed))).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

        metrics = {
            "tantra_train_flows": float(accepted_flows),
            "tantra_train_windows": float(n_samples),
            "tantra_mean_mse": float(weighted_loss / examples_seen),
        }
        if logger:
            logger.info("tantra.train checkpoint=%s metrics=%s", path, metrics)
        return metrics
