from __future__ import annotations

import torch
from torch import nn

from core.registry import register_nids
from models.wrapper import PyTorchModelWrapper


@register_nids("mlp")
class MLPNIDS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: tuple[int, ...] = (256, 128), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.BatchNorm1d(width), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
