from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
from torch.utils.data import DataLoader

from models.wrapper import PyTorchModelWrapper


class BaseDetector(ABC):
    """Interface for post-hoc inference-time detectors.

    Detectors learn only from training/in-distribution data and expose an anomaly
    score where the plugin documents whether larger or smaller values indicate
    abnormal traffic. This interface allows future LID, k-distance (KD), or
    confidence-based detectors without changing the EvaluationEngine.
    """

    @abstractmethod
    def fit(
        self,
        dataloader: DataLoader,
        wrapper: PyTorchModelWrapper,
    ) -> "BaseDetector":
        """Fit detector statistics from the supplied training examples."""
        raise NotImplementedError

    @abstractmethod
    def score(self, x: Any, wrapper: PyTorchModelWrapper) -> np.ndarray:
        """Return one anomaly score per sample, where higher means more anomalous."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly state mapping."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore fitted detector state."""
        raise NotImplementedError
