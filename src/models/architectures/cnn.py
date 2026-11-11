from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import register_nids
from .masking import prefix_valid_mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """Channel batch normalization over *observed* packet positions only.

    Ordinary BatchNorm1d includes masked zero-padding in its running moments,
    silently binding the model to the occupancy distribution and batch padding.
    Keep the same affine/running-state tensor names but train only on valid
    tokens. Earlier CNN checkpoints must be retrained: their running statistics
    were learned under a different mathematical definition.
    """

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or valid_mask.shape != (x.shape[0], x.shape[2]) or valid_mask.dtype != torch.bool:
            raise ValueError('MaskedBatchNorm1d requires x=(B,C,T) and boolean valid_mask=(B,T)')
        if not self.training and self.track_running_stats:
            mean = self.running_mean
            variance = self.running_var
        else:
            weights = valid_mask[:, None, :].to(x.dtype)
            count = weights.sum()
            if count.item() > 0:
                mean = (x * weights).sum(dim=(0, 2)) / count
                variance = ((x - mean[None, :, None]).square() * weights).sum(dim=(0, 2)) / count
                if self.training and self.track_running_stats and count.item() > 1:
                    with torch.no_grad():
                        self.num_batches_tracked.add_(1)
                        factor = (1.0 / float(self.num_batches_tracked.item())
                                  if self.momentum is None else float(self.momentum))
                        self.running_mean.lerp_(mean.detach(), factor)
                        self.running_var.lerp_((variance.detach() * count / (count - 1)), factor)
            else:
                # Zero-packet batches contribute no learned distribution and
                # cannot poison running statistics with artificial zero tokens.
                if self.track_running_stats:
                    mean, variance = self.running_mean, self.running_var
                else:
                    mean = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
                    variance = torch.ones_like(mean)
        out = (x - mean[None, :, None]) * torch.rsqrt(variance[None, :, None] + self.eps)
        if self.affine:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out.masked_fill(~valid_mask[:, None, :], 0.)


@register_nids("cnn")
class CNN1DNIDS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, sequence_features: int,
                 channels: tuple[int, ...] = (64, 128), dropout: float = 0.2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        in_channels = sequence_features
        for out_channels in channels:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(), MaskedBatchNorm1d(out_channels),
            ))
            in_channels = out_channels
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_channels, num_classes))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        valid, _ = prefix_valid_mask(x, valid_mask)
        states = x.masked_fill(~valid.unsqueeze(-1), 0.).transpose(1, 2)
        position_mask = valid
        for block in self.blocks:
            convolution, activation, masked_norm = block
            states = masked_norm(activation(convolution(states)), position_mask)
            # No padded position can win a max-pool window. Windows without
            # any genuine packet stay invalid and are excluded from final mean.
            states = F.max_pool1d(states.masked_fill(~position_mask[:, None, :], -torch.inf), 2)
            position_mask = F.max_pool1d(position_mask[:, None].float(), 2).squeeze(1) > 0
            states = states.masked_fill(~position_mask[:, None, :], 0.)
        pooled = states.sum(dim=-1) / position_mask.sum(dim=-1).clamp_min(1).unsqueeze(-1)
        return self.classifier(pooled)
