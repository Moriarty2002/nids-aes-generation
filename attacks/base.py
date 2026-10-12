from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

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
    """Attack strategy operating on normalized feature tensors or vectors.

    Endpoint threat models are directional: an attacker may control packets it
    transmits, but cannot rewrite packets emitted by the victim without being
    on the network path.  Feature-space attacks therefore need one shared,
    schema-aware mask so that every perturbation mechanism represents the same
    physically realizable capability.
    """

    def _require_model(self) -> ModelWrapper:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} requires a bound target/surrogate model")
        return self.model

    @staticmethod
    def victim_packet_perturbation_mask(
        x: torch.Tensor,
        feature_names: Sequence[str],
        *,
        normalization_mean: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        normalization_std: Sequence[float] | np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a float mask that freezes every feature of victim packets.

        ``DIR == 1`` identifies victim-to-attacker packets in the physical
        feature schema.  Attack inputs are usually z-score normalized, so the
        direction channel is denormalized before comparison when statistics
        are supplied.  The returned tensor has the same shape as ``x`` and is
        zero at victim packet positions, allowing it to be multiplied into
        both random initialization and every projected update.  Both sequence
        tensors ``(N, T, F)`` and flattened tensors ``(N, T*F)`` are supported.
        """
        if x.ndim not in {2, 3}:
            raise ValueError(f"Directional masking requires a 2-D or 3-D tensor, got {tuple(x.shape)}")
        try:
            direction_index = tuple(feature_names).index("DIR")
        except ValueError:
            return torch.ones_like(x)

        feature_count = len(feature_names)
        if feature_count <= direction_index:
            raise ValueError("feature_names must include a valid DIR feature")
        if x.ndim == 3:
            if x.shape[-1] != feature_count:
                raise ValueError(
                    f"Expected {feature_count} features for sequence input, got {x.shape[-1]}"
                )
            direction = x[..., direction_index]
            reshape_back = False
        else:
            if x.shape[-1] % feature_count:
                raise ValueError(
                    f"Flattened input width {x.shape[-1]} is not divisible by {feature_count} features"
                )
            direction = x.reshape(x.shape[0], -1, feature_count)[..., direction_index]
            reshape_back = True

        if normalization_mean is not None or normalization_std is not None:
            if normalization_mean is None or normalization_std is None:
                raise ValueError("normalization_mean and normalization_std must be supplied together")
            mean = torch.as_tensor(normalization_mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(normalization_std, device=x.device, dtype=x.dtype)
            flat_width = x.reshape(x.shape[0], -1, feature_count).shape[1] * feature_count
            if mean.numel() == feature_count and std.numel() == feature_count:
                direction_mean = mean[direction_index]
                direction_std = std[direction_index]
            elif mean.numel() == flat_width and std.numel() == flat_width:
                tiled_mean = mean.reshape(-1, feature_count)
                tiled_std = std.reshape(-1, feature_count)
                direction_mean = tiled_mean[:, direction_index]
                direction_std = tiled_std[:, direction_index]
                if x.ndim == 3:
                    direction_mean = direction_mean.view(1, -1)
                    direction_std = direction_std.view(1, -1)
            else:
                raise ValueError(
                    "Normalization statistics must match either feature_names "
                    "or the flattened input width"
                )
            direction = direction * direction_std + direction_mean

        victim = torch.isclose(direction, torch.ones_like(direction), atol=1e-5, rtol=0.0)
        mask = (~victim).unsqueeze(-1).expand(*victim.shape, feature_count).reshape_as(
            x.reshape(x.shape[0], -1, feature_count)
        )
        return mask.reshape_as(x) if reshape_back else mask


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
