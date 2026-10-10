from __future__ import annotations

import torch
from torch import nn

from core.registry import register_nids


@register_nids("transformer")
class TransformerNIDS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_features: int,  # Injected by workflow.py for 3D datasets
        d_model: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Project the packet feature dimension into the transformer's hidden dimension
        self.projection = nn.Linear(sequence_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Packets, Features)
        x = self.projection(x)         # -> (Batch, Packets, d_model)
        x = self.transformer(x)        # -> (Batch, Packets, d_model)
        
        # Global average pooling across the packet sequence dimension (dim=1)
        x = x.mean(dim=1)              # -> (Batch, d_model)
        return self.classifier(x)