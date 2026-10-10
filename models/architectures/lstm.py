from __future__ import annotations

import torch
from torch import nn

from core.registry import register_nids


@register_nids("lstm")
class BiLSTMNIDS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
        **kwargs  # Swallow any extra sequence injections cleanly
    ) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=layers, 
            batch_first=True, 
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Bidirectional means we multiply hidden_dim by 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Packets, Features)
        out, _ = self.lstm(x)
        
        # Extract the hidden state from the last packet in the sequence
        out = out[:, -1, :] 
        return self.classifier(out)