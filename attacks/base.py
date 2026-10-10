from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from models.wrapper import ModelWrapper


@dataclass
class AttackResult:
    """Common container separating generated data from attack telemetry."""
    adversarial: Any
    success_mask: np.ndarray | torch.Tensor | None = None
    queries: np.ndarray | None = None
    intrinsic_feasible_mask: np.ndarray | None = None
    raw_adversarial: Any | None = None
    feature_names: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def successful(self) -> np.ndarray:
        if self.success_mask is None:
            raise ValueError("AttackResult has no success_mask")
        if isinstance(self.success_mask, torch.Tensor):
            return self.success_mask.detach().cpu().numpy().astype(bool)
        return np.asarray(self.success_mask, dtype=bool)


class BaseAttack(ABC):
    """Strategy interface for threat-model-specific adversarial generation.

    Attacks report their own optimization outcome, but the evaluation engine
    remains responsible for final target-model success and constraint scoring.
    """

    attack_name: str
    query_based: bool = False
    requires_gradients: bool = False

    def __init__(self, model: ModelWrapper | None = None) -> None:
        self.model = model

    @abstractmethod
    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        raise NotImplementedError

    def bind_model(self, model: ModelWrapper) -> "BaseAttack":
        self.model = model
        return self


class FeatureSpaceAttack(BaseAttack):
    """Attack strategy operating on normalized feature tensors or vectors."""

    def _require_model(self) -> ModelWrapper:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} requires a bound target/surrogate model")
        return self.model


class ProblemSpaceAttack(BaseAttack):
    """Mutate raw traffic before the preprocessor converts it into model features."""

    query_based: bool = True

    @abstractmethod
    def generate_raw(self, traffic: Any, y: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Adapt raw generation to the common result interface."""
        raw_adv = self.generate_raw(x, y, **kwargs)
        count = len(raw_adv) if hasattr(raw_adv, "__len__") else 1
        return AttackResult(adversarial=None, raw_adversarial=raw_adv, queries=np.zeros(count, dtype=np.int64))
