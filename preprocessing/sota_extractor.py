from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from core.registry import register_preprocessor
from .base import BasePreprocessor, FeatureExtractorStrategy, PreprocessResult

try:
    import scapy.all as scapy
except ImportError:  # pragma: no cover - exercised only without the optional dependency
    scapy = None

IP = getattr(scapy, "IP", None)
IPv6 = getattr(scapy, "IPv6", None)
PcapReader = getattr(scapy, "PcapReader", None)
TCP = getattr(scapy, "TCP", None)
UDP = getattr(scapy, "UDP", None)


@dataclass(frozen=True)
class StandardFlowExtractor(FeatureExtractorStrategy):
    """Convert packet collections into a stable flow-feature schema.

    Keeping extraction deterministic is important because the same schema is
    used when fitting normalization and when evaluating generated traffic.
    """

    include_protocol_one_hot: bool = True

    def extract(self, raw: Any) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
        if isinstance(raw, (str, Path)):
            return self._extract_path(Path(raw))
        if isinstance(raw, Mapping):
            packets = raw.get("packets")
            if packets is None:
                raise ValueError("Mapping input must contain a 'packets' collection")
            return self._extract_packets(packets)
        return self._extract_packets(raw)

    def _extract_path(self, path: Path) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".npz"}:
            with np.load(path) as data:
                return (
                    np.asarray(data["features"], dtype=np.float32),
                    tuple(str(x) for x in data["feature_names"].tolist()),
                    {"source": str(path), "format": suffix[1:]},
                )
        if suffix in {".json", ".jsonl"}:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return self._extract_packets(rows)
        if suffix == ".pcap":
            if PcapReader is None:
                raise RuntimeError("scapy is required for PCAP preprocessing")
            with PcapReader(str(path)) as reader:
                return self._extract_packets(reader)
        raise ValueError(f"Unsupported raw input format: {path.suffix}")

    def _extract_packets(self, packets: Iterable[Any]) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
        flows: dict[tuple[str, str, int, int, int], list[dict[str, float]]] = defaultdict(list)
        packet_count = 0
        for packet in packets:
            packet_count += 1
            row = self._packet_record(packet)
            if row is None:
                continue
            key = (row["src_ip_text"], row["dst_ip_text"], int(row["src_port"]), int(row["dst_port"]), int(row["protocol"]))
            flows[key].append(row)

        feature_names = (
            "flow_duration_s",
            "packet_count",
            "byte_count",
            "packet_rate",
            "byte_rate",
            "mean_packet_size",
            "min_packet_size",
            "max_packet_size",
            "tcp_syn",
            "tcp_ack",
            "tcp_fin",
            "tcp_rst",
            "udp_packet_ratio",
            "protocol",
        )
        features: list[list[float]] = []
        for rows in flows.values():
            if not rows:
                continue
            times = np.asarray([r["timestamp"] for r in rows], dtype=np.float64)
            sizes = np.asarray([r["packet_size"] for r in rows], dtype=np.float64)
            duration = float(max(0.0, times.max() - times.min()))
            packet_count = float(len(rows))
            byte_count = float(sizes.sum())
            denom = max(duration, 1e-9)
            features.append(
                [
                    duration,
                    packet_count,
                    byte_count,
                    packet_count / denom,
                    byte_count / denom,
                    float(sizes.mean()),
                    float(sizes.min()),
                    float(sizes.max()),
                    float(sum(r["tcp_syn"] for r in rows)),
                    float(sum(r["tcp_ack"] for r in rows)),
                    float(sum(r["tcp_fin"] for r in rows)),
                    float(sum(r["tcp_rst"] for r in rows)),
                    float(sum(r["protocol"] == 17 for r in rows) / packet_count),
                    float(rows[0]["protocol"]),
                ]
            )
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2:
            array = np.empty((0, len(feature_names)), dtype=np.float32)
        return array, feature_names, {"packet_count": packet_count, "flow_count": len(features)}

    @staticmethod
    def _packet_record(packet: Any) -> dict[str, Any] | None:
        if isinstance(packet, Mapping):
            required = {"timestamp", "packet_size", "src_ip_text", "dst_ip_text", "src_port", "dst_port", "protocol"}
            if not required.issubset(packet):
                return None
            return {k: packet[k] for k in required} | {
                "tcp_syn": float(packet.get("tcp_syn", 0)),
                "tcp_ack": float(packet.get("tcp_ack", 0)),
                "tcp_fin": float(packet.get("tcp_fin", 0)),
                "tcp_rst": float(packet.get("tcp_rst", 0)),
            }
        if IP is None or not packet.haslayer(IP) and not packet.haslayer(IPv6):
            return None
        ip_layer = packet[IP] if packet.haslayer(IP) else packet[IPv6]
        protocol = int(getattr(ip_layer, "proto", 0))
        src_port = dst_port = 0
        syn = ack = fin = rst = 0.0
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            protocol = 6
            src_port = int(tcp.sport)
            dst_port = int(tcp.dport)
            flags = str(tcp.flags)
            syn = float("S" in flags)
            ack = float("A" in flags)
            fin = float("F" in flags)
            rst = float("R" in flags)
        elif packet.haslayer(UDP):
            udp = packet[UDP]
            protocol = 17
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
        return {
            "timestamp": float(packet.time),
            "packet_size": float(len(packet)),
            "src_ip_text": str(ip_layer.src),
            "dst_ip_text": str(ip_layer.dst),
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "tcp_syn": syn,
            "tcp_ack": ack,
            "tcp_fin": fin,
            "tcp_rst": rst,
        }


@register_preprocessor("sota_extractor")
class SOTAPreprocessor(BasePreprocessor):
    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self.extractor = StandardFlowExtractor()
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, raw: Any) -> None:
        """Learn normalization statistics from unnormalized extracted features."""
        features, _, _ = self.extractor.extract(raw)
        if len(features) == 0:
            raise ValueError("Cannot fit normalization on empty feature set")
        self.mean = features.mean(axis=0)
        self.std = np.clip(features.std(axis=0), 1e-12, None)

    def transform(self, raw: Any, fit: bool = False) -> PreprocessResult:  # type: ignore[override]
        """Extract features and optionally apply the fitted normalization state."""
        features, names, metadata = self.extractor.extract(raw)
        if self.normalize:
            if fit:
                self.mean = features.mean(axis=0)
                self.std = np.clip(features.std(axis=0), 1e-12, None)
            if self.mean is None or self.std is None:
                raise RuntimeError("Preprocessor normalization statistics are not fitted")
            features = (features - self.mean) / self.std
        return PreprocessResult(np.asarray(features, dtype=np.float32), names, metadata)

    def state_dict(self) -> dict[str, Any]:
        return {"normalize": self.normalize, "mean": None if self.mean is None else self.mean.tolist(), "std": None if self.std is None else self.std.tolist()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.normalize = bool(state["normalize"])
        self.mean = None if state["mean"] is None else np.asarray(state["mean"], dtype=np.float32)
        self.std = None if state["std"] is None else np.asarray(state["std"], dtype=np.float32)
