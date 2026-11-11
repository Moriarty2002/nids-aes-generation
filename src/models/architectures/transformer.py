from __future__ import annotations

import math
import torch
from torch import nn

from core.registry import register_nids
from .masking import prefix_valid_mask


@register_nids("transformer")
class TransformerNIDS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, sequence_features: int,
                 d_model: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(sequence_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))
        position = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.) / d_model))
        pe = torch.zeros(20, d_model)
        pe[:, 0::2] = torch.sin(position * frequencies)
        pe[:, 1::2] = torch.cos(position * frequencies[:pe[:, 1::2].shape[1]])
        self.register_buffer("packet_positions", pe, persistent=True)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        valid, lengths = prefix_valid_mask(x, valid_mask)
        if x.shape[1] != 20:
            raise ValueError("Transformer checkpoint topology requires T=20")
        masked_x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        states = self.projection(masked_x) + self.packet_positions[None].to(x.dtype)
        # PyTorch attention with all positions masked generates NaNs. Give
        # zero-length examples a deterministic dummy key, then zero their pool.
        attention_valid = valid.clone()
        attention_valid[lengths == 0, 0] = True
        states = self.transformer(states, src_key_padding_mask=~attention_valid)
        pooled = (states * valid.unsqueeze(-1)).sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        return self.classifier(pooled)
