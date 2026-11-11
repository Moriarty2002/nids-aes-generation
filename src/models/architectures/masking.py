"""Packet-validity contract for fixed-order sequence NIDS models."""
from __future__ import annotations

import torch


def prefix_valid_mask(x: torch.Tensor, valid_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3:
        raise ValueError("Sequence classifier expects (B,T,F)")
    if valid_mask is None:
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=x.device)
        if tuple(valid.shape) != tuple(x.shape[:2]):
            raise ValueError(f"valid_mask shape {tuple(valid.shape)} != {tuple(x.shape[:2])}")
    lengths = valid.long().sum(dim=1)
    expected = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
    if not torch.equal(expected, valid):
        raise ValueError("Sequence masks must represent a contiguous real-packet prefix")
    return valid, lengths
