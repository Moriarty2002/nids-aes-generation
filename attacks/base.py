from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
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

    def __post_init__(self) -> None:
        for field_name in ("adversarial", "raw_adversarial", "success_mask"):
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                setattr(self, field_name, value.detach())

    def successful(self) -> np.ndarray:
        # NOTE: success_mask indicates whether the attack succeeded against the bound target/surrogate model
        # DURING generation, BEFORE any post-generation discrete lattice projection is applied.
        # For final, authoritative, post-projection benchmark metrics, rely on the
        # EvaluationEngine's recomputed metrics (e.g., 'asr'), not this internal mask.
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


@dataclass(kw_only=True)
class FeatureSpaceAttack(BaseAttack):
    """Base class for normalized feature-space attacks.

    ``discrete_projection`` is an optional *post-generation* transformation.  It
    is deliberately separate from the attack objective and from the universal
    constraint gatekeeper: attacks optimize their continuous feature-space
    objective first, then this optional adapter snaps selected physical fields
    to integer lattice points before the result reaches evaluation.

    The projector is installed as a small finalization wrapper around concrete
    subclasses' ``generate`` methods.  This centralizes the behavior so newly
    registered feature-space attacks inherit it without embedding gatekeeper or
    lattice logic in every optimizer implementation.  Problem-space attacks do
    not inherit this class and are unaffected.
    """

    discrete_projection: bool = False
    feature_names: tuple[str, ...] = ()
    immutable_features: tuple[str, ...] = ()
    model: ModelWrapper | None = field(default=None, init=False, repr=False, compare=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure each concrete feature attack finalizes output before returning it."""
        super().__init_subclass__(**kwargs)
        implementation = cls.__dict__.get("generate")
        if implementation is None or getattr(implementation, "_discrete_projector_wrapped", False):
            return

        @wraps(implementation)
        def _generate_with_discrete_projection(
            self: "FeatureSpaceAttack", *args: Any, **call_kwargs: Any
        ) -> AttackResult:
            result = implementation(self, *args, **call_kwargs)
            if not isinstance(result, AttackResult) or result.adversarial is None:
                return result

            names = tuple(
                result.feature_names
                or getattr(self, "feature_names", ())
                or call_kwargs.get("feature_names", ())
            )
            normalization_mean = call_kwargs.get("normalization_mean")
            normalization_std = call_kwargs.get("normalization_std")
            result.adversarial = self._apply_discrete_lattice_projection(
                result.adversarial,
                names,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
            )
            result.metadata.setdefault("discrete_projection_enabled", bool(self.discrete_projection))
            result.metadata.setdefault(
                "discrete_projection_applied",
                bool(
                    self.discrete_projection
                    and normalization_mean is not None
                    and normalization_std is not None
                    and {str(name).upper() for name in names}.intersection({"PL", "TTL", "WIN"})
                ),
            )
            return result

        setattr(_generate_with_discrete_projection, "_discrete_projector_wrapped", True)
        setattr(cls, "generate", _generate_with_discrete_projection)

    def _require_model(self) -> ModelWrapper:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} requires a bound target/surrogate model")
        return self.model

    def _apply_discrete_lattice_projection(
        self,
        x_adv: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        normalization_mean: Sequence[float] | np.ndarray | torch.Tensor | None,
        normalization_std: Sequence[float] | np.ndarray | torch.Tensor | None,
    ) -> np.ndarray | torch.Tensor:
        """Optionally snap ``PL``, ``TTL`` and ``WIN`` to integer physical values.

        Option A (default): ``discrete_projection=False`` returns the original
        object unchanged, preserving continuous attack outputs but unfeasible
        for real-world deployment.

        Option B: with projection enabled and both normalization vectors supplied,
        this converts normalized values to physical units, applies ``np.round``
        to only the named discrete fields, then normalizes them again. It handles
        tabular ``(N,F)``, flattened sequence ``(N,T*F)``, and sequence
        ``(N,T,F)`` representations. Normalization vectors may be per-feature
        ``(F,)`` or per-flattened-coordinate ``(T*F,)``.

        This method does not alter any gatekeeper rules and is not a substitute
        for the post-generation realizability check. If normalization metadata
        is absent, the original batch is returned and the independent gatekeeper
        remains responsible for reporting non-integral features.
        """
        if self.discrete_projection:
            if normalization_mean is None or normalization_std is None:
                raise ValueError(
                    "discrete_projection=True requires normalization_mean and normalization_std "
                    "to be provided to the generate() method."
                )

        names = tuple(str(name) for name in feature_names)
        if not names:
            return x_adv
        target_fields = {"PL", "TTL", "WIN"}
        target_indices = [i for i, name in enumerate(names) if name.upper() in target_fields]
        if not target_indices:
            return x_adv

        is_tensor = isinstance(x_adv, torch.Tensor)
        if is_tensor:
            source_device = x_adv.device
            source_dtype = x_adv.dtype
            x_array = x_adv.detach().cpu().numpy()
        else:
            source_device = None
            source_dtype = None
            x_array = np.asarray(x_adv)

        if x_array.ndim == 3:
            batch_size, time_steps, feature_count = x_array.shape
            if feature_count != len(names):
                raise ValueError(
                    f"Feature schema has {len(names)} names but adversarial tensor has {feature_count} features"
                )
            shaped = x_array.reshape(batch_size, time_steps, feature_count).astype(np.float64, copy=True)
        elif x_array.ndim == 2:
            batch_size, width = x_array.shape
            if width % len(names) != 0:
                raise ValueError(
                    f"Adversarial width {width} is not divisible by feature count {len(names)}"
                )
            time_steps = width // len(names)
            feature_count = len(names)
            shaped = x_array.reshape(batch_size, time_steps, feature_count).astype(np.float64, copy=True)
        else:
            raise ValueError(
                f"Discrete lattice projection expects a 2-D or 3-D batch, got shape {x_array.shape}"
            )

        mean = self._normalization_vector(normalization_mean, "normalization_mean")
        std = self._normalization_vector(normalization_std, "normalization_std")
        if mean.size != std.size:
            raise ValueError("normalization_mean and normalization_std must have equal lengths")
        if mean.size == feature_count:
            mean_view = mean.reshape(1, 1, feature_count)
            std_view = std.reshape(1, 1, feature_count)
        elif mean.size == time_steps * feature_count:
            mean_view = mean.reshape(1, time_steps, feature_count)
            std_view = std.reshape(1, time_steps, feature_count)
        else:
            raise ValueError(
                "Normalization vectors must match the feature count or flattened input width; "
                f"got {mean.size} values for shape {tuple(x_array.shape)} and {len(names)} names"
            )
        if not np.all(np.isfinite(mean_view)) or not np.all(np.isfinite(std_view)):
            raise ValueError("Normalization statistics must be finite")
        if np.any(std_view <= 0):
            raise ValueError("normalization_std must be strictly positive")

        for feature_idx in target_indices:
            physical = shaped[..., feature_idx] * std_view[..., feature_idx] + mean_view[..., feature_idx]
            snapped = np.round(physical)
            shaped[..., feature_idx] = (
                snapped - mean_view[..., feature_idx]
            ) / std_view[..., feature_idx]

        projected = shaped.reshape(x_array.shape).astype(x_array.dtype, copy=False)
        if is_tensor:
            assert source_device is not None and source_dtype is not None
            return torch.as_tensor(projected, device=source_device, dtype=source_dtype)
        return projected

    @staticmethod
    def _normalization_vector(
        value: Sequence[float] | np.ndarray | torch.Tensor,
        name: str,
    ) -> np.ndarray:
        """Convert a normalization vector to a validated flat CPU NumPy array."""
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        if array.size == 0:
            raise ValueError(f"{name} cannot be empty")
        return array

    @staticmethod
    def victim_packet_perturbation_mask(
        x: torch.Tensor,
        feature_names: Sequence[str],
        *,
        normalization_mean: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        normalization_std: Sequence[float] | np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a float mask for the features an endpoint attacker may edit.

        ``DIR == 1`` identifies victim-to-attacker packets in the physical
        feature schema.  Attack inputs are usually z-score normalized, so the
        direction channel is denormalized before comparison when statistics
        are supplied.  DIR is always immutable, and every feature is immutable
        on victim packets. Both sequence tensors ``(N, T, F)`` and flattened
        tensors ``(N, T*F)`` are supported.
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

        victim = torch.isclose(direction, torch.ones_like(direction), atol=1e-4, rtol=0.0)
        mask = (~victim).unsqueeze(-1).expand(*victim.shape, feature_count).clone()
        # Direction is part of the traffic metadata, not an attacker-controlled
        # feature, so it must remain unchanged for every packet.
        mask[..., direction_index] = 0.0
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
