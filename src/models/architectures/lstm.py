from __future__ import annotations

import torch
from torch import nn

from core.registry import register_nids
from .masking import prefix_valid_mask


@register_nids("lstm")
class BiLSTMNIDS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128,
                 layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        valid, lengths = prefix_valid_mask(x, valid_mask)
        # Packed recurrence excludes padding in both forward and reverse LSTM.
        # Empty flows use one deterministic zero packet; their pooled state is
        # then zeroed, avoiding pack_padded_sequence's forbidden zero length.
        safe = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        packed = nn.utils.rnn.pack_padded_sequence(
            safe, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        state = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        state = state * (lengths > 0).unsqueeze(-1).to(state.dtype)
        return self.classifier(state)
