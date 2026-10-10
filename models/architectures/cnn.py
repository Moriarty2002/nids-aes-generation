from __future__ import annotations

import torch
from torch import nn

from core.registry import register_nids


@register_nids("cnn")
class CNN1DNIDS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sequence_features: int,  # Injected by workflow.py
        channels: tuple[int, ...] = (64, 128),
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        layers = []
        
        # For 1D Convolutions across time, the feature count acts as the channel count
        in_ch = sequence_features 
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
                nn.MaxPool1d(2)
            ]
            in_ch = out_ch
            
        self.features = nn.Sequential(*layers)
        
        # Adaptively pool the sequence length dimension down to 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Incoming Parquet x shape: (Batch, Packets, Features)
        # PyTorch Conv1d expects: (Batch, Channels, Length)
        x = x.transpose(1, 2) 
        
        x = self.features(x)
        x = self.pool(x).squeeze(-1) # Remove the trailing length dimension of size 1
        return self.classifier(x)