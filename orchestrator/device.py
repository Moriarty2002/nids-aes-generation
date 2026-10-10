from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from core.exceptions import DeviceConfigurationError


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    torch_device: torch.device
    cuda_available: bool
    device_count: int
    device_name: str
    capability: tuple[int, int] | None
    total_memory_bytes: int | None


def resolve_device(requested: str | int | None) -> DeviceInfo:
    """Validate a requested CPU/GPU and return immutable device metadata."""
    value = "cpu" if requested is None else str(requested).strip().lower()
    if value == "cpu":
        return DeviceInfo(value, torch.device("cpu"), torch.cuda.is_available(), torch.cuda.device_count() if torch.cuda.is_available() else 0, "cpu", None, None)
    if not value.startswith("cuda:"):
        value = f"cuda:{value}"
    if not torch.cuda.is_available():
        raise DeviceConfigurationError(f"GPU requested as {value}, but torch.cuda.is_available() is False")
    try:
        index = int(value.split(":", 1)[1])
    except ValueError as exc:
        raise DeviceConfigurationError(f"Invalid GPU specification: {requested!r}") from exc
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise DeviceConfigurationError(f"Requested GPU {index}, available GPU indices: 0..{count - 1}")
    props = torch.cuda.get_device_properties(index)
    torch.cuda.set_device(index)
    return DeviceInfo(value, torch.device(value), True, count, props.name, tuple(props.major_minor_version) if hasattr(props, "major_minor_version") else (props.major, props.minor), props.total_memory)


def configure_process_device(requested: str | int | None) -> DeviceInfo:
    """Configure the current worker's CUDA context after validating the request."""
    info = resolve_device(requested)
    if info.torch_device.type == "cuda":
        torch.cuda.set_device(info.torch_device)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return info


def to_device(tensor: torch.Tensor, device: torch.device, *, pin_cpu_memory: bool = True) -> torch.Tensor:
    """Move a tensor efficiently, optionally pinning eligible CPU memory first."""
    value = tensor
    if device.type == "cuda" and pin_cpu_memory and value.device.type == "cpu" and value.is_floating_point():
        value = value.pin_memory()
    return value.to(device, non_blocking=(device.type == "cuda"))
