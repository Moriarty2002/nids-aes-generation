"""DeepRed offline C2 flow modification (Hajizadeh et al., WOOT 2025).

Original: WOOT25 Fig3/Algorithms1-2 noise-only positive feature replacement GAN.
Local: BCE means, Adam settings/LeakyReLU slope, two src2dst aggregates and
standardized discriminator inputs; no auxiliary proxy CE/minimality objective.
Replacement-to-action rounding/caps are a separate fail-closed NIDS adaptation.
NIDS adaptation: bounded offline IPv4/UDP PCAP backend with independent
packet checksum/length checks and authorized training-only surrogate fitting.
Remaining deviation: not live C2 transmission, TCP or generic protocol wire
certification; success on mapped features does not certify application meaning.
Topology: classic PCAP/PCAP-flow IPv4 UDP; deferred extractor untouched.
"""



from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, ip_address
import logging
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from attacks.base import AttackResult, ProblemSpaceAttack
from core.registry import register_attack
from core.provenance import (authorized_attack_training_indices, trained_generator_provenance,
                             verify_generator_artifact_provenance)


_NAMES = ("src2dst_bytes", "src2dst_max_ps")
_FORMAT_VERSION = 3
_MAX_CAPTURE_BYTES = 64 * 1024 * 1024
_MAX_PACKETS_PER_FLOW = 100_000


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    """An unmodified PCAP record, including its timestamp and captured bytes."""

    seconds: int
    fraction: int
    frame: bytes
    original_length: int


@dataclass(frozen=True, slots=True)
class PcapFlow:
    """Immutable in-memory PCAP; also usable by an explicitly compatible extractor."""

    global_header: bytes
    records: tuple[CaptureRecord, ...]


@dataclass(frozen=True, slots=True)
class _PcapFormat:
    endian: str
    tick_rate: int
    link_type: int


@dataclass(frozen=True, slots=True)
class _Datagram:
    """Parse-only IPv4/UDP summary. Bytes are never rewritten in this object."""

    src: bytes
    dst: bytes
    sport: int
    dport: int
    ip_length: int
    identification: int
    ttl: int
    tos: int
    l2_header: bytes


def _format(header: bytes) -> _PcapFormat:
    """Accept only classic PCAP; rejecting unknown link types prevents unsafe guesses."""
    magic = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),
        b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
        b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),
        b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
    }
    if len(header) != 24 or header[:4] not in magic:
        raise ValueError("Only classic PCAP files with a recognized endianness are supported")
    endian, tick_rate = magic[header[:4]]
    major, minor, _, _, snaplen, network = struct.unpack(endian + "HHiIII", header[4:])
    link_type = network & 0xFFFF
    if (major, minor) != (2, 4) or snaplen < 28 or link_type not in (1, 101):
        raise ValueError("Unsupported PCAP version, snaplen, or link type")
    return _PcapFormat(endian, tick_rate, link_type)


def _read_pcap(path: str | os.PathLike[str]) -> PcapFlow:
    """Read a bounded capture without trusting arbitrary on-disk record lengths."""
    path = Path(path)
    if path.stat().st_size > _MAX_CAPTURE_BYTES:
        raise ValueError("PCAP exceeds configured hard size cap")
    with path.open("rb") as source:
        header = source.read(24)
        fmt = _format(header)
        snaplen = struct.unpack_from(fmt.endian + "I", header, 16)[0]
        records: list[CaptureRecord] = []
        while record_header := source.read(16):
            if len(record_header) != 16 or len(records) >= _MAX_PACKETS_PER_FLOW:
                raise ValueError("Truncated PCAP record or packet count cap exceeded")
            sec, frac, included, original = struct.unpack(fmt.endian + "IIII", record_header)
            if frac >= fmt.tick_rate or included > original or included > snaplen:
                raise ValueError("Invalid PCAP timestamps or record lengths")
            frame = source.read(included)
            if len(frame) != included:
                raise ValueError("Truncated PCAP packet payload")
            records.append(CaptureRecord(sec, frac, frame, original))
    return PcapFlow(header, tuple(records))


def _write_pcap(path: Path, capture: PcapFlow) -> None:
    """Atomic, private output; never change or overwrite the original capture."""
    fmt = _format(capture.global_header)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".deepred-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as destination:
            destination.write(capture.global_header)
            for rec in capture.records:
                destination.write(struct.pack(
                    fmt.endian + "IIII", rec.seconds, rec.fraction,
                    len(rec.frame), rec.original_length,
                ))
                destination.write(rec.frame)
            destination.flush()
            os.fsync(destination.fileno())
        # Exclusive destination: avoid accidental replacement of earlier trials.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checksum(data: bytes) -> int:
    """Internet one's-complement checksum with RFC 1071 odd-byte padding."""
    if len(data) % 2:
        data += b"\x00"
    words = struct.unpack("!" + "H" * (len(data) // 2), data)
    total = sum(words)
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _parse_datagram(frame: bytes, link_type: int, *, new: bool = False) -> _Datagram:
    """Validate L2/L3/L4 geometry and forbid ambiguous or fragmented packets.

    Original packets can contain checksum offloading artifacts, so their wire
    checksums are not reinterpreted. Synthetic packets MUST have valid checksums.
    """
    if link_type == 1:
        if len(frame) < 14 or frame[12:14] != b"\x08\x00":
            raise ValueError("Only Ethernet/IPv4 is supported")
        offset, l2 = 14, frame[:14]
    else:
        offset, l2 = 0, b""
    if len(frame) < offset + 28:
        raise ValueError("Short IPv4/UDP datagram")
    first = frame[offset]
    if first >> 4 != 4:
        raise ValueError("IPv6/non-IP is not supported")
    ihl = (first & 0x0F) * 4
    if ihl < 20 or len(frame) < offset + ihl + 8:
        raise ValueError("Invalid IPv4 header length")
    ip_head = frame[offset:offset + ihl]
    length, ident, frag, ttl, proto = struct.unpack_from("!HHHBB", ip_head, 2)
    if frag & 0x3FFF or proto != 17 or ttl == 0:
        raise ValueError("Fragments and non-UDP traffic are not supported")
    if length < ihl + 8 or offset + length > len(frame):
        raise ValueError("IPv4 length inconsistent with captured frame")
    udp_start = offset + ihl
    sport, dport, udp_len, udp_sum = struct.unpack_from("!HHHH", frame, udp_start)
    if sport == 0 or dport == 0 or udp_len != length - ihl:
        raise ValueError("Invalid UDP header or inconsistent UDP/IP lengths")
    if new:
        if _checksum(ip_head) != 0 or udp_sum == 0:
            raise ValueError("Synthetic IP/UDP checksum missing or invalid")
        pseudo = ip_head[12:20] + struct.pack("!BBH", 0, 17, udp_len)
        if _checksum(pseudo + frame[udp_start:udp_start + udp_len]) != 0:
            raise ValueError("Synthetic UDP checksum invalid")
    return _Datagram(
        src=ip_head[12:16], dst=ip_head[16:20], sport=sport, dport=dport,
        ip_length=length, identification=ident, ttl=ttl, tos=ip_head[1],
        l2_header=l2,
    )


def _measure(capture: PcapFlow, attacker_ip: bytes) -> tuple[np.ndarray, _Datagram, list[_Datagram]]:
    """Measure *actual* original packet topology, not normalized model tensors.

    Source-to-destination is the first packet's source. To prevent ambiguous
    endpoint attribution, only captures whose first source is the configured
    attacker endpoint can be altered. Every packet must belong to one 5-tuple.
    """
    fmt = _format(capture.global_header)
    if not capture.records:
        raise ValueError("Cannot determine flow direction from an empty capture")
    datagrams = [_parse_datagram(rec.frame, fmt.link_type) for rec in capture.records]
    first = datagrams[0]
    if first.src != attacker_ip or first.dst == attacker_ip:
        raise ValueError("src2dst does not originate at the configured attacker endpoint")
    key = (first.src, first.dst, first.sport, first.dport)
    reverse = (first.dst, first.src, first.dport, first.sport)
    for packet in datagrams:
        state = (packet.src, packet.dst, packet.sport, packet.dport)
        if state not in (key, reverse):
            raise ValueError("The capture contains multiple 5-tuples")
    outbound = [pkt for pkt in datagrams if pkt.src == attacker_ip]
    if not outbound:
        raise ValueError("No attacker-owned packets")
    sizes = np.asarray([pkt.ip_length for pkt in outbound], dtype=np.int64)
    values = np.asarray([int(sizes.sum()), int(sizes.max())], dtype=np.float64)
    return values, outbound[-1], datagrams


def _packet(template: _Datagram, size: int, identification: int) -> bytes:
    """Build one checksummed, inert UDP packet within the existing 5-tuple.

    Unlike modifying TCP payload bytes, this operation never requires rewriting
    a victim-owned acknowledgment or compensating a TCP sequence number.
    """
    if not 28 <= size <= 65535:
        raise ValueError("IPv4/UDP requires at least 28 bytes")
    content = b"\x00" * (size - 28)
    udp_len = size - 20
    udp_zero = struct.pack("!HHHH", template.sport, template.dport, udp_len, 0)
    pseudo = template.src + template.dst + struct.pack("!BBH", 0, 17, udp_len)
    udp_sum = _checksum(pseudo + udp_zero + content) or 0xFFFF
    udp = struct.pack("!HHHH", template.sport, template.dport, udp_len, udp_sum)
    ipv4 = struct.pack(
        "!BBHHHBBH4s4s", 0x45, template.tos, size, identification & 0xFFFF,
        0x4000, template.ttl, 17, 0, template.src, template.dst,
    )
    ip_sum = _checksum(ipv4)
    ipv4 = ipv4[:10] + struct.pack("!H", ip_sum) + ipv4[12:]
    return template.l2_header + ipv4 + udp + content


def _plan_sizes(
    before: np.ndarray,
    delta: np.ndarray,
    selected: tuple[str, ...],
    mtu: int,
    max_packets: int,
) -> list[int]:
    """Solve integer, packet-level geometry; reject rather than approximate.

    Every injected UDP datagram consumes >=28 IP bytes. In the two-feature
    case, a max-size packet must also fit the total requested byte increment.
    If this is impossible, no feasible inverse mapping exists for this flow.
    """
    selected_set = set(selected)
    byte_change = int(math.ceil(float(delta[0]))) if _NAMES[0] in selected_set else None
    size_change = int(math.ceil(float(delta[1]))) if _NAMES[1] in selected_set else None
    if byte_change is not None and byte_change < 1:
        raise ValueError("Byte objective is not a positive increment")
    if size_change is not None and size_change < 1:
        raise ValueError("Max-size objective is not a positive increment")
    desired_max = int(before[1]) + (size_change or 0)
    lengths: list[int] = []
    if size_change is not None:
        if desired_max > mtu:
            raise ValueError("Requested max packet size exceeds MTU")
        lengths.append(desired_max)
    if byte_change is not None:
        remaining = byte_change - sum(lengths)
        if remaining < 0:
            raise ValueError("Max-size insertion exceeds requested byte budget")
        if remaining:
            # Split only if a single datagram exceeds MTU, allocating >=28
            # bytes to each remainder without exceeding a selected max size.
            packet_limit = min(mtu, desired_max) if size_change is not None else mtu
            while remaining:
                if len(lengths) >= max_packets:
                    raise ValueError("Requested budget needs too many packets")
                piece = min(remaining, packet_limit)
                if remaining - piece in range(1, 28):
                    piece -= 28 - (remaining - piece)
                if piece < 28 or remaining - piece < 0:
                    raise ValueError("Byte budget is smaller than valid datagram")
                lengths.append(piece)
                remaining -= piece
    if not lengths or len(lengths) > max_packets:
        raise ValueError("No valid injected packets")
    if any(size < 28 or size > mtu for size in lengths):
        raise ValueError("Invalid packet length in proposed topology")
    return lengths


def _mutate(
    original: PcapFlow,
    attacker_ip: bytes,
    proposed_delta: np.ndarray,
    selected: tuple[str, ...],
    mtu: int,
    max_added_packets: int,
) -> tuple[PcapFlow, dict[str, int]]:
    """Append attacker-originated UDP packets, then verify all invariants.

    Byte-for-byte prefix identity is stronger than row-position comparisons:
    original victim packets cannot be overwritten, deleted, or reordered.
    """
    fmt = _format(original.global_header)
    before, template, parsed = _measure(original, attacker_ip)
    sizes = _plan_sizes(before, proposed_delta, selected, mtu, max_added_packets)
    snaplen = struct.unpack_from(fmt.endian + "I", original.global_header, 16)[0]
    if any(size + len(template.l2_header) > snaplen for size in sizes):
        raise ValueError("Injected datagram exceeds PCAP snaplen")
    last = original.records[-1]
    ticks = last.seconds * fmt.tick_rate + last.fraction
    records = list(original.records)
    for offset, size in enumerate(sizes, 1):
        moment = ticks + offset
        frame = _packet(template, size, template.identification + offset)
        records.append(CaptureRecord(
            seconds=moment // fmt.tick_rate,
            fraction=moment % fmt.tick_rate,
            frame=frame,
            original_length=len(frame),
        ))
    output = PcapFlow(original.global_header, tuple(records))
    _validate_mutation(original, output, parsed, attacker_ip, sizes, proposed_delta, selected)
    return output, {"injected_packets": len(sizes), "injected_ip_bytes": sum(sizes)}


def _validate_mutation(
    clean: PcapFlow,
    output: PcapFlow,
    parsed: list[_Datagram],
    attacker_ip: bytes,
    sizes: Sequence[int],
    delta: np.ndarray,
    selected: tuple[str, ...],
) -> None:
    """Independent topology audit used to produce intrinsic_feasible_mask.

    Any error becomes a rejected flow. Critically, verification is on complete
    capture records, not time-step-aligned model feature arrays.
    """
    count = len(clean.records)
    if output.global_header != clean.global_header or output.records[:count] != clean.records:
        raise ValueError("Original records were overwritten/reordered/deleted")
    if len(output.records) != count + len(sizes):
        raise ValueError("Injected packet count does not match plan")
    fmt = _format(output.global_header)
    new_datagrams = [_parse_datagram(r.frame, fmt.link_type, new=True)
                     for r in output.records[count:]]
    first = parsed[0]
    original_tuple = (first.src, first.dst, first.sport, first.dport)
    for packet, expected in zip(new_datagrams, sizes, strict=True):
        if (packet.src, packet.dst, packet.sport, packet.dport) != original_tuple:
            raise ValueError("Injected packet crossed ownership or flow 5-tuple")
        if packet.src != attacker_ip or packet.ip_length != expected:
            raise ValueError("Injected packet geometry/ownership mismatch")
    earlier, _, _ = _measure(clean, attacker_ip)
    after, _, _ = _measure(output, attacker_ip)
    if int(after[0] - earlier[0]) != sum(sizes):
        raise ValueError("Actual byte increment disagrees with packet plan")
    if _NAMES[0] in selected and int(after[0] - earlier[0]) != math.ceil(float(delta[0])):
        raise ValueError("Actual bytes fail target objective")
    if _NAMES[1] in selected and int(after[1] - earlier[1]) != math.ceil(float(delta[1])):
        raise ValueError("Actual max packet size fails target objective")


class _Generator(nn.Module):
    """Fig3: Gaussian noise -> 64/128/256 BN/LeakyReLU -> positive replacements."""

    def __init__(self, latent_dim: int, targets: int) -> None:
        super().__init__()
        if latent_dim < 1 or targets not in (1, 2):
            raise ValueError("DeepRed generates one or two primary replacements")
        layers: list[nn.Module] = []
        current = latent_dim
        for hidden in (64, 128, 256):
            layers += [nn.Linear(current, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2)]
            current = hidden
        layers += [nn.Linear(current, targets), nn.Softplus()]
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: Tensor) -> Tensor:
        return self.layers(noise)


class _Discriminator(nn.Module):
    """Paper's 128/64/32/1 discriminator, returning BCE-with-logits scores."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x).reshape(-1)


def _normalized(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """GAN-only normalization; never used for physical PCAP feature encoding."""
    return (x - mean) / std


def _create_candidate(x: Tensor, replacements: Tensor, indices: tuple[int, ...]) -> Tensor:
    """Fig3 reassembly replaces selected coordinates, with zero old-coordinate gradient."""
    if (x.ndim != 2 or len(set(indices)) != len(indices) or not 1 <= len(indices) <= 2
            or any(i < 0 or i >= x.shape[1] for i in indices)
            or replacements.shape != (len(x), len(indices))
            or not torch.isfinite(x).all() or not torch.isfinite(replacements).all()
            or torch.any(replacements <= 0)):
        raise ValueError("Invalid DeepRed replacement coordinates/values")
    index = torch.tensor(indices, device=x.device).unsqueeze(0).expand(len(x), -1)
    return x.scatter(1, index, replacements)


def _post_modify(x: Tensor, feature_names: Sequence[str]) -> Tensor:
    """Algorithm1's explicit byte-sum dependency; unsupported schemas fail closed.

    The production pair has no derived aggregates. Synthetic source equations
    may include both byte directions and their bidirectional sum, never a new
    extractor or fabricated packet mapping.
    """
    names = tuple(feature_names)
    supported = {*_NAMES, "dst2src_bytes", "bidirectional_bytes"}
    if (x.ndim != 2 or x.shape[1] != len(names) or len(set(names)) != len(names)
            or set(names) - supported or not torch.isfinite(x).all() or torch.any(x < 0)):
        raise ValueError("Unsupported/nonfinite DeepRed dependency schema")
    if "bidirectional_bytes" not in names:
        return x
    if not {"src2dst_bytes", "dst2src_bytes"}.issubset(names):
        raise ValueError("Both byte directions are required for bidirectional_bytes")
    result = x.clone()
    result[:, names.index("bidirectional_bytes")] = x[:, names.index("src2dst_bytes")] + x[:, names.index("dst2src_bytes")]
    return result


def _discriminator_step(discriminator: nn.Module, optimizer: torch.optim.Optimizer,
                        benign: Tensor, malicious: Tensor, crafted: Tensor) -> float:
    discriminator.train()
    values = [discriminator(batch) for batch in (benign, malicious, crafted.detach())]
    loss = sum(F.binary_cross_entropy_with_logits(value, torch.full_like(value, label))
               for value, label in zip(values, (0., 1., 1.), strict=True)) / 3.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _generator_step(generator: nn.Module, discriminator: nn.Module,
                    optimizer: torch.optim.Optimizer, malicious: Tensor, noise: Tensor,
                    indices: tuple[int, ...], mean: Tensor, std: Tensor,
                    feature_names: Sequence[str] = _NAMES) -> float:
    # Freeze both parameters and buffers, but retain gradients into crafted inputs.
    flags = [p.requires_grad for p in discriminator.parameters()]
    training = discriminator.training
    discriminator.zero_grad(set_to_none=True)
    discriminator.eval().requires_grad_(False)
    generator.train()
    try:
        crafted = _post_modify(_create_candidate(malicious, generator(noise), indices), feature_names)
        values = discriminator(_normalized(crafted, mean, std))
        loss = F.binary_cross_entropy_with_logits(values, torch.zeros_like(values))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach())
    finally:
        for param, flag in zip(discriminator.parameters(), flags, strict=True):
            param.requires_grad_(flag)
        discriminator.train(training)


def _gan_population_mask(truth: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    """Retain known source populations only when the authorized surrogate agrees.

    Ground truth defines TRAIN population membership, not an extra oracle.
    Incorrectly detected malicious records never become benign D examples.
    """
    truth, oracle = np.asarray(truth), np.asarray(oracle)
    if (truth.ndim != 1 or truth.shape != oracle.shape or truth.dtype.kind not in "iu"
            or oracle.dtype.kind not in "iu" or np.any(truth < 0) or np.any(oracle < 0)):
        raise ValueError("Invalid DeepRed source population labels")
    return (truth != 0) == (oracle != 0)


def _fit_gan(physical: np.ndarray, labels: np.ndarray, *, latent_dim: int,
             indices: tuple[int, ...], epochs: int, batch_size: int,
             learning_rate: float, seed: int, device: torch.device,
             feature_names: Sequence[str] = _NAMES) -> tuple[_Generator, np.ndarray, np.ndarray, float, float]:
    """One D then G update per minibatch; no source/target classifier gradients."""
    physical = np.asarray(physical)
    labels = np.asarray(labels)
    if (physical.dtype != np.float64 or physical.ndim != 2 or len(physical) != len(labels)
            or not np.isfinite(physical).all() or np.any(physical < 0)
            or not set(np.unique(labels)).issubset({0, 1}) or batch_size < 2 or epochs < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid DeepRed training population/settings")
    bad_ids, good_ids = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if len(bad_ids) < 2 or not len(good_ids):
        raise ValueError("DeepRed requires benign and >=2 malicious TRAIN examples")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean_np, std_np = physical.mean(0), np.maximum(physical.std(0), 1.)
    data = torch.as_tensor(physical, dtype=torch.float32, device=device)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    generator = _Generator(latent_dim, len(indices)).to(device)
    discriminator = _Discriminator(physical.shape[1]).to(device)
    go = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    do = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    final_d = final_g = 0.
    for _ in range(epochs):
        shuffled = rng.permutation(bad_ids)
        # Merge a singleton tail so every TRAIN malicious row receives an update.
        chunks = np.array_split(shuffled, max(1, len(shuffled)//batch_size))
        for selection in chunks:
            bad = data[torch.as_tensor(selection, device=device)]
            good = data[torch.as_tensor(rng.choice(good_ids, len(selection)), device=device)]
            generator.eval()
            with torch.no_grad():
                crafted = _post_modify(_create_candidate(
                    bad, generator(torch.randn(len(bad), latent_dim, device=device)), indices), feature_names)
            final_d = _discriminator_step(discriminator, do, _normalized(good, mean, std),
                                         _normalized(bad, mean, std), _normalized(crafted, mean, std))
            final_g = _generator_step(generator, discriminator, go, bad,
                                     torch.randn(len(bad), latent_dim, device=device),
                                     indices, mean, std, feature_names)
    return generator.eval(), mean_np, std_np, final_g, final_d


@register_attack("deepred")
class DeepRed(ProblemSpaceAttack):
    """Source-math replacement GAN with a limited offline src2dst action mapping.

    Registered TRAIN canonical tensors supply independent surrogate labels.
    Two separate physical aggregates supply GAN populations; neither is a
    substitute for the source's eight NFStream features or full C2 semantics.
    """

    attack_name = "deepred"
    query_based = False
    requires_gradients = False  # No target-model gradients at evaluation time.

    def __init__(
        self,
        *,
        checkpoint: str = "runs/checkpoints/deepred_seed{seed}.pt",
        training_npz: str | None = None,
        attack_features: Sequence[str] = ("src2dst_bytes",),
        attacker_ip: str | None = None,
        output_dir: str = "runs/deepred_pcaps",
        max_delta_bytes: int = 512,
        max_delta_max_ps: int = 256,
        mtu: int = 1500,
        max_added_packets: int = 4,
        latent_dim: int = 200,
        batch_size: int = 64,
        gan_epochs: int = 100,
        learning_rate: float = 2e-4,
        seed: int = 42,
        device: str = "cpu",
        config: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__()
        selected = tuple(str(name).lower() for name in attack_features)
        if not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected):
            raise ValueError("Select one or two distinct adversarial features")
        if not set(selected).issubset(_NAMES):
            raise ValueError(f"This backend supports only {_NAMES}")
        if attacker_ip is not None and not isinstance(ip_address(attacker_ip), IPv4Address):
            raise ValueError("attacker_ip must be IPv4")
        if not 28 <= mtu <= 65535 or not 1 <= max_added_packets <= 64:
            raise ValueError("Invalid MTU or packet-injection budget")
        if min(max_delta_bytes, max_delta_max_ps, latent_dim, gan_epochs) <= 0 or batch_size < 2:
            raise ValueError("GAN dimensions, training steps, and budgets must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        # Hydra may inject training_fraction/feature_names, but these never
        # change on-wire PCAP semantics; ignore only those known adapters.
        unknown = set(extra) - {"training_fraction", "feature_names", "immutable_features", "variant", "target_label"}
        if unknown:
            raise TypeError(f"Unknown DeepRed parameters: {sorted(unknown)}")
        self.checkpoint = str(checkpoint)
        self.training_npz = training_npz
        self.attack_features = selected
        self.attacker_ip = attacker_ip
        self.output_dir = Path(output_dir)
        self.max_delta_bytes = int(max_delta_bytes)
        self.max_delta_max_ps = int(max_delta_max_ps)
        self.mtu = int(mtu)
        self.max_added_packets = int(max_added_packets)
        self.latent_dim = int(latent_dim)
        self.batch_size = int(batch_size)
        self.gan_epochs = int(gan_epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.config = config
        self._generator: _Generator | None = None
        self._mean: Tensor | None = None
        self._std: Tensor | None = None
        self._loaded_seed: int | None = None

    @classmethod
    def train(
        cls,
        *,
        config: Mapping[str, Any],
        seed: int,
        gpu: str,
        logger: logging.Logger | None = None,
    ) -> Mapping[str, float]:
        """Fit on registered train-only PCAPs using only independent surrogate feedback.

        The pretrained authorized surrogate labels raw training flows through
        canonical six-feature inputs. Their predicted labels define the GAN
        populations; no target is loaded and no unrepresentable crafted flow
        is submitted to the canonical surrogate. Aggregate scaling is internal.
        """
        attack_options = config.get("attack", {})
        if not isinstance(attack_options, Mapping):
            raise TypeError("config.attack must be a mapping")
        # Worker seed/device are authoritative; Hydra may also carry these
        # names as optional fields, so remove them before explicit binding.
        settings = cls(**{key: value for key, value in attack_options.items()
                          if key not in {"name", "seed", "device", "config"}},
                       seed=int(seed), device=gpu, config=config)
        if settings.training_npz is not None:
            raise ValueError('Unregistered training_npz/oracle_labels cannot prove training-only membership; '
                             'use the registered raw-PCAP dataset and independent surrogate')
        if settings.attacker_ip is None:
            raise ValueError('DeepRed fitting requires an explicitly declared attacker IPv4 address')
        from workflow.factory import _build_model, _inject_dynamic_dataset_shape
        from workflow.io import (_load_checkpoint, _load_checkpoint_metadata,
                                 _normalization_from_checkpoint)
        from workflow.seeds import role_seed
        from core.provenance import frozen_model_state_sha256
        from core.query_meter import ModelQueryMeter
        from core.registry import DATASETS
        source_cfg = dict(config.get('surrogate_model') or {})
        if not source_cfg or not source_cfg.get('enabled', True):
            raise ValueError('DeepRed generator requires independently registered surrogate source')
        source_config = dict(config)
        source_config['model'] = source_cfg
        _inject_dynamic_dataset_shape(source_config)
        dataset_cfg = source_config['dataset']
        dataset = DATASETS.get(str(dataset_cfg['name']))(**dict(dataset_cfg.get('params', {})))
        raw, all_labels = dataset.raw_arrays()
        all_labels = np.asarray(all_labels, dtype=np.int64)
        indices = authorized_attack_training_indices(
            dataset, all_labels, config, seed, role='surrogate_model')
        raw_traffic = getattr(dataset, 'raw_traffic', None)
        if raw_traffic is None or len(raw_traffic) != len(all_labels):
            raise ValueError('DeepRed needs registered, aligned training PCAP sources')
        attacker = ip_address(settings.attacker_ip).packed
        physical_rows = []
        for row in indices:
            capture = raw_traffic[int(row)]
            capture = _read_pcap(capture) if isinstance(capture, (str, os.PathLike)) else capture
            if not isinstance(capture, PcapFlow):
                raise ValueError('Raw traffic must be a verified classic-PCAP source')
            aggregates, _, _ = _measure(capture, attacker)
            physical_rows.append(aggregates)
        physical = np.stack(physical_rows).astype(np.float64, copy=False)
        labels = all_labels[indices]
        source_seed = role_seed(config, 'surrogate', seed)
        source_path = Path(str(source_cfg['checkpoint']).format(seed=source_seed))
        source_meta = _load_checkpoint_metadata(source_path)
        if tuple(source_meta['input_shape']) != tuple(raw.shape[1:]):
            raise ValueError('Registered surrogate topology differs from DeepRed training source')
        norm = _normalization_from_checkpoint(source_meta, tuple(dataset.feature_names), role='DeepRed surrogate')
        source_wrapper = _build_model(source_cfg, (len(raw), *source_meta['input_shape']),
                                      device=settings.device, num_classes=int(source_meta['num_classes']),seed=source_seed)
        _load_checkpoint(source_wrapper, source_path, checkpoint_data=source_meta)
        if frozen_model_state_sha256(source_wrapper) != source_meta['model_state_sha256']:
            raise ValueError('DeepRed independent surrogate learned state mismatch')
        from core.dual_representation import encode_normalized32
        from models.wrapper import PyTorchModelWrapper
        if raw.dtype != np.float64:
            raise TypeError("DeepRed requires authoritative physical float64")
        model_input = encode_normalized32(raw[indices], tuple(dataset.feature_names), norm.mean, norm.std)
        valid = np.asarray(dataset.valid_mask)[indices]
        with ModelQueryMeter(source_wrapper) as query_meter:
            oracle = np.asarray(source_wrapper.predict(model_input, valid_mask=valid)
                                if isinstance(source_wrapper, PyTorchModelWrapper)
                                else source_wrapper.predict(model_input), dtype=np.int64).reshape(-1)
        oracle_records = int(query_meter.counts.inference_records)
        if oracle_records != len(indices) or query_meter.counts.gradient_forward_records:
            raise RuntimeError("DeepRed surrogate cost mismatch")
        del source_wrapper, raw, model_input
        if len(physical) < 4:
            raise ValueError('DeepRed requires at least four authorized training-only flows')
        if not np.isfinite(physical).all() or np.any(physical < 0):
            raise ValueError("Training data must contain finite nonnegative physical measurements")
        if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(oracle.dtype, np.integer):
            raise TypeError("Labels and oracle_labels must use integer types")
        if not set(oracle).issubset(set(dataset.class_mapping.values())):
            raise ValueError("DeepRed surrogate class schema mismatch")
        y = (oracle != 0).astype(np.int64)
        population = _gan_population_mask(labels, oracle)
        generator, mean_np, std_np, final_g, final_d = _fit_gan(
            physical[population], y[population], latent_dim=settings.latent_dim,
            indices=tuple(_NAMES.index(name) for name in settings.attack_features),
            epochs=settings.gan_epochs, batch_size=settings.batch_size,
            learning_rate=settings.learning_rate, seed=seed, device=settings.device)

        destination = Path(settings.checkpoint.format(seed=int(seed)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": _FORMAT_VERSION,
            "artifact_provenance": trained_generator_provenance(
                dataset, all_labels, config, seed, source_role='surrogate_model',
                input_shape=tuple(source_meta['input_shape']),
                feature_names=tuple(dataset.feature_names)),
            "authorized_surrogate_inference_records": oracle_records,
            "attack": settings.attack_name,
            "seed": int(seed),
            "feature_names": _NAMES,
            "attack_features": settings.attack_features,
            "latent_dim": settings.latent_dim,
            "max_delta_bytes": settings.max_delta_bytes,
            "max_delta_max_ps": settings.max_delta_max_ps,
            "mean": mean_np.tolist(),
            "std": std_np.tolist(),
            "proposal_semantics": "noise_only_positive_replacement_v3",
            "training_sample_ids": [str(dataset.sample_ids[i]) for i in indices],
            "gan_training_sample_ids": [str(dataset.sample_ids[i]) for i in indices[population]],
            "surrogate_inference_costs": [1] * len(indices),
            "surrogate_gradient_forward_costs": [0] * len(indices),
            "seed_roles": dict(config.get("seed_roles", {})),
            "generator": {key: val.detach().cpu() for key, val in generator.state_dict().items()},
        }
        fd, tmp = tempfile.mkstemp(prefix=".deepred-checkpoint-", suffix=".pt", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, destination)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        if logger:
            logger.info("deepred.train checkpoint=%s count=%d", destination, len(physical))
        return {
            "deepred_surrogate_training_samples": float(len(physical)),
            "deepred_gan_training_samples": float(population.sum()),
            "surrogate_training_inference_records": float(oracle_records),
            "surrogate_training_gradient_forward_records": 0.,
            "deepred_generator_loss": final_g,
            "deepred_discriminator_loss": final_d,
            "deepred_training_epochs": float(settings.gan_epochs),
            "deepred_independent_surrogate_oracle_records": float(oracle_records),
            "deepred_source_prediction_class_count": float(len(np.unique(y))),
        }

    def _load(self, seed: int) -> None:
        """Load a verified, seed-local checkpoint, not a live model object."""
        # The evaluation worker pins the first registered attack-seed checkpoint;
        # per-batch RNG offsets must not silently switch learned generators.
        if self._generator is not None and self._loaded_seed is not None:
            return
        path = Path(self.checkpoint.format(seed=seed))
        if not path.is_file():
            raise FileNotFoundError(f"DeepRed checkpoint missing: {path}; run train_attack first")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        verify_generator_artifact_provenance(checkpoint, getattr(self,'expected_provenance',None))
        if (checkpoint.get("format_version") != _FORMAT_VERSION
                or checkpoint.get("attack") != "deepred"
                or checkpoint.get("proposal_semantics") != "noise_only_positive_replacement_v3"
                or checkpoint.get("seed") != seed
                or tuple(checkpoint.get("feature_names", ())) != _NAMES
                or tuple(checkpoint.get("attack_features", ())) != self.attack_features
                or int(checkpoint.get("latent_dim", -1)) != self.latent_dim
                or int(checkpoint.get("max_delta_bytes", -1)) != self.max_delta_bytes
                or int(checkpoint.get("max_delta_max_ps", -1)) != self.max_delta_max_ps):
            raise ValueError("DeepRed checkpoint schema, budget, or seed mismatch")
        ids = checkpoint.get("training_sample_ids", [])
        if (not ids or len(set(ids)) != len(ids)
                or checkpoint.get("surrogate_inference_costs") != [1]*len(ids)
                or checkpoint.get("surrogate_gradient_forward_costs") != [0]*len(ids)
                or checkpoint.get("authorized_surrogate_inference_records") != len(ids)):
            raise ValueError("DeepRed training cost/membership schema mismatch")
        generator = _Generator(self.latent_dim, len(self.attack_features)).to(self.device)
        generator.load_state_dict(checkpoint["generator"], strict=True)
        generator.eval()
        self._mean = torch.tensor(checkpoint["mean"], device=self.device, dtype=torch.float32)
        self._std = torch.tensor(checkpoint["std"], device=self.device, dtype=torch.float32)
        if (self._mean.shape != (2,) or self._std.shape != (2,)
                or not torch.isfinite(self._mean).all() or not torch.isfinite(self._std).all()
                or torch.any(self._std <= 0)):
            raise ValueError("Checkpoint contains invalid training statistics")
        self._generator = generator
        self._loaded_seed = seed

    def _delta(self, features: np.ndarray, seed: int) -> np.ndarray:
        """Round positive replacement targets, then prove bounded monotone actions."""
        # Checkpoint seed and noise seed are distinct: evaluation batches may
        # offset their RNG seed without changing the trained model identity.
        assert self._generator is not None and self._mean is not None and self._std is not None
        truth = np.asarray(features)
        if (truth.dtype != np.float64 or truth.shape != (2,) or not np.isfinite(truth).all()
                or np.any(truth < 0) or np.any(truth != np.floor(truth))):
            raise ValueError("DeepRed actions require authoritative finite float64 aggregates")
        random = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(1, self.latent_dim, generator=random).to(self.device)
        with torch.inference_mode():
            replacement = self._generator(noise).cpu().numpy()[0].astype(np.float64)
        if replacement.shape != (len(self.attack_features),) or not np.isfinite(replacement).all() or np.any(replacement <= 0):
            raise ValueError("GAN returned invalid positive replacement")
        deltas = np.zeros(2, dtype=np.float64)
        for value, name in zip(replacement, self.attack_features, strict=True):
            index = _NAMES.index(name)
            delta = math.ceil(float(value)) - truth[index]
            budget = self.max_delta_bytes if index == 0 else self.max_delta_max_ps
            if delta <= 0 or delta > budget or (index == 1 and math.ceil(float(value)) > self.mtu):
                raise ValueError("Replacement cannot satisfy monotone local action bounds")
            deltas[index] = delta
        _plan_sizes(truth, deltas, self.attack_features, self.mtu, self.max_added_packets)
        return deltas

    def generate_raw(self, traffic: Sequence[str | os.PathLike[str] | PcapFlow],
                     y: Any, **kwargs: Any) -> AttackResult:
        """Mutate offline PCAPs and return a verified *per-flow* feasibility mask.

        Input contains classic PCAP file paths or PcapFlow records with IPv4/UDP
        packets; TCP, PCAPng and live-C2 traffic are not supported. Each result
        retains its flow index and per-flow feasibility flag, and failures return
        the exact original path/object. Packet payloads are not logged.
        """
        labels = np.asarray(y.detach().cpu().numpy() if isinstance(y, Tensor) else y).reshape(-1)
        if len(traffic) != len(labels):
            raise ValueError("Number of traffic flows and labels differs")
        effective_seed = int(kwargs.get("seed", self.seed))
        endpoint = kwargs.get("attacker_ip", self.attacker_ip)
        output: list[str | os.PathLike[str] | PcapFlow] = list(traffic)
        feasible = np.zeros(len(traffic), dtype=np.bool_)
        summaries: list[dict[str, Any]] = []
        if endpoint is None:
            return AttackResult(adversarial=None, raw_adversarial=output,
                                intrinsic_feasible_mask=feasible,
                                queries=np.zeros(len(traffic), dtype=np.int64),
                                metadata={"rejected_reason": "attacker_ip_required"})
        try:
            attacker = ip_address(endpoint)
            if not isinstance(attacker, IPv4Address):
                raise ValueError("Only IPv4 is supported")
            attacker_bytes = attacker.packed
        except ValueError as exc:
            raise ValueError("attacker_ip must be a valid IPv4 address") from exc
        # Checkpoint/setup errors are fatal, never silently treated as successes.
        self._load(effective_seed)
        for index, (flow, label) in enumerate(zip(traffic, labels, strict=True)):
            note: dict[str, Any] = {"index": index, "feasible": False}
            if int(label) == 0:
                note["reason"] = "benign_flow_not_attacked"
                summaries.append(note)
                continue
            try:
                is_path = isinstance(flow, (str, os.PathLike))
                original = _read_pcap(flow) if is_path else flow
                if not isinstance(original, PcapFlow):
                    raise TypeError("Raw traffic must be PCAP paths or PcapFlow objects")
                before, _, _ = _measure(original, attacker_bytes)
                delta = self._delta(before, effective_seed + index)
                candidate, stats = _mutate(original, attacker_bytes, delta,
                                           self.attack_features, self.mtu, self.max_added_packets)
                if is_path:
                    # One unique file for each candidate. Never modify source PCAP.
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    fd, temporary_path = tempfile.mkstemp(prefix=".deepred-flow-", suffix=".pcap",
                                                          dir=self.output_dir)
                    os.close(fd)
                    os.unlink(temporary_path)
                    destination = Path(temporary_path)
                    _write_pcap(destination, candidate)
                    output[index] = str(destination)
                else:
                    output[index] = candidate
                feasible[index] = True
                note.update({"feasible": True, **stats,
                             "delta_bytes": float(delta[0]),
                             "delta_max_ps": float(delta[1])})
            except (OSError, ValueError, TypeError, RuntimeError, OverflowError, struct.error) as exc:
                note["reason"] = str(exc)
            summaries.append(note)
        return AttackResult(
            adversarial=None,
            raw_adversarial=output,
            intrinsic_feasible_mask=feasible,
            queries=np.zeros(len(traffic), dtype=np.int64),
            metadata={"packet_backend": "offline_ipv4_udp_pcap", "flow_audits": summaries},
        )

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Override base adapter: its default discards intrinsic feasibility.

        EvaluationEngine requires AttackResult.raw_adversarial and a boolean
        intrinsic_feasible_mask; calling the base class implementation would
        wrap this AttackResult incorrectly and omit the authoritative mask.
        """
        result = self.generate_raw(x, y, **kwargs)
        if not isinstance(result, AttackResult) or result.intrinsic_feasible_mask is None:
            raise RuntimeError("DeepRed must return a fail-closed AttackResult")
        return result
