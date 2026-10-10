from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreprocessResult:
    """Feature batch plus schema and provenance metadata from preprocessing."""
    features: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


class FeatureExtractorStrategy(ABC):
    """Strategy interface for converting one raw input source into features."""
    @abstractmethod
    def extract(self, raw: Any) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
        raise NotImplementedError


class BasePreprocessor(ABC):
    """Preprocessor interface shared by CLI and in-memory problem-space routing."""

    @abstractmethod
    def transform(self, raw: Any) -> PreprocessResult:
        raise NotImplementedError

    def __call__(self, raw: Any) -> PreprocessResult:
        return self.transform(raw)
