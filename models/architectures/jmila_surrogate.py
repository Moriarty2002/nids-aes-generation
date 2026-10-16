"""Seven-layer fully connected surrogate for tabular NIDS experiments.

The requested topology follows the Jmila & Khedher adversarial-NIDS context
(Computers Networks, 2022, https://doi.org/10.1016/j.comnet.2022.109073), with
layer widths specified by this benchmark's experiment design. The BatchNorm,
Dropout and optimizer defaults are explicit, configurable implementation
choices informed by recent NIDS hyperparameter studies; they should not be
interpreted as a verbatim reproduction of that paper's entire training recipe.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import register_nids


class _BatchNorm1dSafeForSingleton(nn.BatchNorm1d):
    """Use running statistics for a final singleton batch instead of raising.

    BatchNorm1d cannot estimate batch variance from one training example. For
    that uncommon final batch, this subclass uses the learned running values;
    normal batches retain standard BatchNorm training behavior.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.ndim == 2 and input.shape[0] == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input)


@register_nids("jmila_surrogate")
class JmilaSurrogateNIDS(nn.Module):
    """Seven linear layers with consecutively halved hidden dimensions.

    Architecture: ``input_dim -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> C``.
    Every hidden layer uses Linear -> BatchNorm1d -> ReLU -> Dropout. The
    output returns logits (no softmax), which is compatible with cross-entropy
    training and targeted gradient attacks.

    ``requires_flatten=True`` lets workflow.py automatically convert packet
    sequences to tabular vectors when this fully connected model is selected.
    """

    requires_flatten = True
    REGISTRY_NAME = "jmila_surrogate"
    DEFAULT_HIDDEN_DIMS = (1024, 512, 256, 128, 64, 32)

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        widths = tuple(int(width) for width in hidden_dims)
        if not widths or any(width < 1 for width in widths):
            raise ValueError("hidden_dims must be a non-empty sequence of positive integers")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in widths:
            layers.extend(
                (
                    nn.Linear(previous, width),
                    _BatchNorm1dSafeForSingleton(width),
                    nn.ReLU(inplace=False),
                    nn.Dropout(float(dropout)),
                )
            )
            previous = width
        layers.append(nn.Linear(previous, int(num_classes)))
        self.net = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dims = widths
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected batch-first tabular input, got {tuple(x.shape)}")
        return self.net(x.flatten(start_dim=1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final hidden representation before the classifier head."""
        if x.ndim < 2:
            raise ValueError(f"Expected batch-first tabular input, got {tuple(x.shape)}")
        value = x.flatten(start_dim=1)
        for layer in list(self.net.children())[:-1]:
            value = layer(value)
        return value
