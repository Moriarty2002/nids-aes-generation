from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NormalizationStats:
    """Serializable feature-wise statistics shared by train and evaluation paths."""
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]
    version: str = "1.0"

    def __post_init__(self) -> None:
        # Canonical model normalization statistics are stored as float32.
        # Physical traffic remains original float64; the encoder performs
        # float64 arithmetic using these frozen float32 statistics, then one
        # and only one final cast to model float32.
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.shape != std.shape or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError('Normalization statistics must be finite and shape-aligned')
        if np.any(std < 0):
            raise ValueError('Normalization standard deviations cannot be negative')
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'std', np.maximum(std, np.float32(1e-12)))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Map physical features into the model's normalized input space."""
        return (x - self.mean) / np.clip(self.std, 1e-12, None)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Map normalized model features back into physical feature space."""
        return x * np.clip(self.std, 1e-12, None) + self.mean


class BaseDataset(ABC, Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset contract with explicit normalization and tensor conversion."""

    def __init__(self, normalization: NormalizationStats | None = None) -> None:
        self.normalization = normalization

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def raw_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit_normalization(self, x: np.ndarray) -> NormalizationStats:
        stats = NormalizationStats(
            mean=np.asarray(x.mean(axis=0), dtype=np.float64),
            std=np.asarray(x.std(axis=0), dtype=np.float64),
            feature_names=self.feature_names,
        )
        self.normalization = stats
        return stats

    def normalized_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized features, fitting statistics lazily when absent."""
        x, y = self.raw_arrays()
        if self.normalization is None:
            self.fit_normalization(x)
        assert self.normalization is not None
        return self.normalization.normalize(x), y

    def __len__(self) -> int:
        x, _ = self.raw_arrays()
        return int(len(x))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.normalized_arrays()
        return torch.as_tensor(x[idx], dtype=torch.float32), torch.as_tensor(y[idx], dtype=torch.long)
