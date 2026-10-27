from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Sequence

import numpy as np
import torch

from models.wrapper import ModelWrapper
from core.dual_representation import decode_physical64, encode_normalized32, layout


@dataclass
class AttackResult:
    """Common container separating generated data from attack telemetry."""
    adversarial: np.ndarray | torch.Tensor | None
    success_mask: np.ndarray | torch.Tensor | None = None
    queries: np.ndarray | None = None
    intrinsic_feasible_mask: np.ndarray | None = None
    raw_adversarial: Any | None = None
    feature_names: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    authoritative_physical: np.ndarray | torch.Tensor | None = None

    def __post_init__(self) -> None:
        for field_name in ("adversarial", "raw_adversarial", "success_mask", "authoritative_physical"):
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

    Finalization is installed as a small wrapper around concrete subclasses'
    ``generate`` methods. This centralizes immutable-field and padding
    restoration, followed by optional lattice projection, so newly registered
    feature-space attacks inherit the safety policy. Problem-space attacks do
    not inherit this class and are unaffected.
    """

    discrete_projection: bool = False
    clip_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
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
            # Engine supplies the unquantized physical baseline. Never forward
            # this internal argument to concrete PGD/gradient-attack methods.
            clean_authoritative = call_kwargs.pop("_clean_authoritative_physical", None)
            result = implementation(self, *args, **call_kwargs)
            if not isinstance(result, AttackResult) or result.adversarial is None:
                return result

            original = args[0] if args else call_kwargs.get("x")
            names = tuple(result.feature_names or self.feature_names or call_kwargs.get("feature_names", ()))
            mean = call_kwargs.get("normalization_mean")
            std = call_kwargs.get("normalization_std")
            if mean is None or std is None:
                raise ValueError("Dual-representation feature attacks require normalization statistics")
            if original is None:
                raise ValueError("Dual-representation feature attacks require a clean input")
            source = result.adversarial
            if source.dtype != (torch.float32 if isinstance(source, torch.Tensor) else np.float32):
                raise TypeError("Feature attack must generate normalized float32 samples")
            if tuple(source.shape) != tuple(original.shape):
                raise ValueError("Attack and baseline shapes differ")

            # Exactly one float64 candidate workspace. The separate clean
            # physical reference is necessary for faithful victim restoration.
            if clean_authoritative is None:
                raise ValueError(
                    "Authoritative clean physical baseline is required; "
                    "normalized float32 values cannot recover exact physical truth"
                )
            physical = decode_physical64(source, names, mean, std)
            if isinstance(physical, torch.Tensor):
                clean_physical = torch.as_tensor(
                    clean_authoritative, dtype=torch.float64, device=physical.device
                )
            else:
                clean_physical = np.asarray(clean_authoritative, dtype=np.float64)
            result.metadata["clean_baseline_provenance"] = "pre_normalization_physical"
            if clean_physical.shape != physical.shape:
                raise ValueError("Clean authoritative physical shape differs from attack shape")

            # All post-generation operations are in-place in native physical
            # space. Protected values are restored LAST so clipping and lattice
            # projection cannot touch victim packets or immutable fields.
            self._apply_range_clipping(physical, names, clip_ranges=self.clip_ranges)
            self._apply_discrete_lattice_projection(physical, names)
            self._restore_protected_values(
                physical, clean_physical, names,
                valid_mask=call_kwargs.get("valid_mask"),
            )
            result.authoritative_physical = physical
            result.adversarial = encode_normalized32(physical, names, mean, std)
            result.metadata.update({
                "range_clipping_enabled": bool(self.clip_ranges),
                "range_clipping_applied": bool(self.clip_ranges),
                "discrete_projection_enabled": bool(self.discrete_projection),
                "discrete_projection_applied": bool(self.discrete_projection and
                    {str(n).upper() for n in names}.intersection({"PL", "TTL", "WIN"})),
                "dual_authoritative_representation": True,
            })
            return result

        setattr(_generate_with_discrete_projection, "_discrete_projector_wrapped", True)
        setattr(cls, "generate", _generate_with_discrete_projection)

    def _restore_protected_values(
        self,
        x_adv: np.ndarray | torch.Tensor,
        x_orig: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Restore physical immutable columns, victim packets, and padding IN PLACE."""
        if x_adv.dtype != (torch.float64 if isinstance(x_adv, torch.Tensor) else np.float64):
            raise TypeError("Physical restoration requires float64 adversarial input")
        b, t, f = layout(x_adv, feature_names)
        if tuple(x_adv.shape) != tuple(x_orig.shape):
            raise ValueError("Clean and attack physical shapes must match")
        names = tuple(str(n).upper() for n in feature_names)
        immutable = {str(n).upper() for n in self.immutable_features} | ({"DIR"} if "DIR" in names else set())
        if immutable - set(names):
            raise ValueError(f"Unknown immutable features: {sorted(immutable - set(names))}")
        if isinstance(x_adv, torch.Tensor):
            clean = torch.as_tensor(x_orig, dtype=torch.float64, device=x_adv.device).reshape(b,t,f)
            adv = x_adv.reshape(b,t,f)
            if valid_mask is None:
                valid = None
            else:
                valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=x_adv.device)
                if tuple(valid.shape) != (b,t):
                    raise ValueError("Invalid valid_mask shape")
            with torch.no_grad():
                if "DIR" in names:
                    victim = torch.isclose(clean[...,names.index("DIR")],
                                            torch.tensor(1.0, dtype=torch.float64, device=x_adv.device),
                                            atol=1e-2, rtol=0.0)
                    if valid is not None:
                        victim &= valid
                    torch.where(victim.unsqueeze(-1), clean, adv, out=adv)
                for field_name in immutable:
                    j = names.index(field_name)
                    adv[...,j].copy_(clean[...,j])
                if valid is not None:
                    torch.where((~valid).unsqueeze(-1), clean, adv, out=adv)
            return x_adv
        adv = x_adv.reshape(b,t,f)
        clean = np.asarray(x_orig, dtype=np.float64).reshape(b,t,f)
        if valid_mask is None:
            valid_np = None
        else:
            valid_np = np.asarray(valid_mask, dtype=bool)
            if valid_np.shape != (b,t):
                raise ValueError("Invalid valid_mask shape")
        if "DIR" in names:
            victim_np = np.isclose(clean[...,names.index("DIR")], 1.0, atol=1e-2, rtol=0.0)
            if valid_np is not None:
                victim_np &= valid_np
            np.copyto(adv, clean, where=victim_np[...,None])
        for field_name in immutable:
            j = names.index(field_name)
            np.copyto(adv[...,j], clean[...,j])
        if valid_np is not None:
            np.copyto(adv, clean, where=(~valid_np)[...,None])
        return x_adv

    def _require_model(self) -> ModelWrapper:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} requires a bound model")
        return self.model

    def _apply_range_clipping(
        self,
        x_adv: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
        *,
        clip_ranges: dict[str, tuple[float, float]],
    ) -> np.ndarray | torch.Tensor:
        """In-place float64 physical clipping; no normalized-space conversion."""
        b, t, f = layout(x_adv, feature_names)
        if x_adv.dtype != (torch.float64 if isinstance(x_adv, torch.Tensor) else np.float64):
            raise TypeError("Physical clipping requires float64")
        names = tuple(str(n).upper() for n in feature_names)
        view = x_adv.reshape(b,t,f)
        if clip_ranges:
            for key, bound in clip_ranges.items():
                name = str(key).upper()
                if name not in names:
                    raise ValueError(f"Unknown clip_ranges feature {key!r}")
                if len(bound) != 2:
                    raise ValueError(f"Invalid physical bounds for {key!r}")
                lower, upper = float(bound[0]), float(bound[1])
                if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                    raise ValueError(f"Invalid physical bounds for {key!r}")
                j = names.index(name)
                if isinstance(view, torch.Tensor):
                    with torch.no_grad():
                        view[...,j].clamp_(min=lower, max=upper)
                else:
                    np.clip(view[...,j], lower, upper, out=view[...,j])
        return x_adv

    def _apply_discrete_lattice_projection(
        self,
        x_adv: np.ndarray | torch.Tensor,
        feature_names: Sequence[str],
    ) -> np.ndarray | torch.Tensor:
        """Snap PL/TTL/WIN directly to float64 physical integers IN PLACE.
        
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
        if not self.discrete_projection:
            return x_adv
        b, t, f = layout(x_adv, feature_names)
        if x_adv.dtype != (torch.float64 if isinstance(x_adv, torch.Tensor) else np.float64):
            raise TypeError("Physical lattice projection requires float64")
        view = x_adv.reshape(b,t,f)
        for j, name in enumerate(feature_names):
            if str(name).upper() not in {"PL", "TTL", "WIN"}:
                continue
            if isinstance(view, torch.Tensor):
                with torch.no_grad():
                    view[...,j].round_()
            else:
                np.rint(view[...,j], out=view[...,j])
        return x_adv

    @staticmethod
    def _normalization_vector(
        value: Sequence[float] | np.ndarray | torch.Tensor,
        name: str,
    ) -> np.ndarray:
        """Convert a normalization vector to a validated flat CPU NumPy array."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        out = np.asarray(value, dtype=np.float64).reshape(-1)
        if not out.size:
            raise ValueError(f"{name} cannot be empty")
        return out

    @staticmethod
    def victim_packet_perturbation_mask(
        x: torch.Tensor,
        feature_names: Sequence[str],
        *,
        normalization_mean: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        normalization_std: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        valid_mask: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mark only real, attacker-controlled packet coordinates as perturbable.

        For endpoint (non-MitM) attackers, packets with physical ``DIR == 1``
        are victim-owned and their *entire feature rows* are immutable. The
        direction column is immutable on every packet. Sequence padding is not
        traffic: when ``valid_mask[b, t]`` is false, **all** of its feature
        coordinates must have zero gradient/random-start/PGD updates.

        ``valid_mask`` has shape ``(B, T)`` and is converted to the device of
        ``x``. Both ``(B, T*F)`` and ``(B, T, F)`` layouts, with ``F`` or
        ``T*F`` normalization statistics, are supported. The returned tensor
        has exactly the input's shape, device and dtype (normally float32).
        For a schema without DIR, padding is still enforced.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Directional masking requires a torch.Tensor")
        if x.ndim not in (2, 3):
            raise ValueError(
                f"Directional masking requires (B, T*F) or (B, T, F), got {tuple(x.shape)}"
            )

        names = tuple(feature_names)
        feature_count = len(names)
        batch_size = int(x.shape[0])
        valid: torch.Tensor | None = None
        if valid_mask is not None:
            valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=x.device)
            if valid.ndim != 2 or valid.shape[0] != batch_size:
                raise ValueError(
                    "valid_mask must have shape (B, T) with B equal to x.shape[0]"
                )

        if x.ndim == 3:
            time_steps = int(x.shape[1])
            if feature_count == 0:
                feature_count = int(x.shape[2])
            elif x.shape[2] != feature_count:
                raise ValueError("Sequence width differs from feature_names")
        else:
            width = int(x.shape[1])
            if feature_count == 0:
                # An absent schema historically permits all changes; with a
                # packet mask we can still infer layout and freeze padding.
                if valid is None:
                    return torch.ones_like(x)
                if valid.shape[1] == 0 or width % int(valid.shape[1]):
                    raise ValueError("Cannot infer packet width from valid_mask")
                time_steps = int(valid.shape[1])
                feature_count = width // time_steps
            else:
                if width % feature_count:
                    raise ValueError(
                        f"Flattened width {width} not divisible by {feature_count} features"
                    )
                time_steps = width // feature_count

        if feature_count == 0:
            raise ValueError("At least one feature coordinate is required")
        if valid is not None and tuple(valid.shape) != (batch_size, time_steps):
            raise ValueError(
                f"valid_mask shape {tuple(valid.shape)} does not match "
                f"packet layout {(batch_size, time_steps)}"
            )

        x_packets = x.reshape(batch_size, time_steps, feature_count)
        direction_index = names.index("DIR") if "DIR" in names else None
        if direction_index is None:
            if valid is None:
                return torch.ones_like(x)
            return (
                valid.unsqueeze(-1)
                .expand(batch_size, time_steps, feature_count)
                .reshape_as(x)
                .to(dtype=x.dtype)
            )

        direction = x_packets[..., direction_index]
        if normalization_mean is not None or normalization_std is not None:
            if normalization_mean is None or normalization_std is None:
                raise ValueError("normalization_mean and normalization_std must be supplied together")
            mean = torch.as_tensor(normalization_mean, device=x.device, dtype=x.dtype).reshape(-1)
            std = torch.as_tensor(normalization_std, device=x.device, dtype=x.dtype).reshape(-1)
            if mean.numel() == feature_count and std.numel() == feature_count:
                direction_mean = mean[direction_index]
                direction_std = std[direction_index]
            elif mean.numel() == time_steps * feature_count and std.numel() == time_steps * feature_count:
                direction_mean = mean.reshape(time_steps, feature_count)[:, direction_index]
                direction_std = std.reshape(time_steps, feature_count)[:, direction_index]
            else:
                raise ValueError(
                    "Normalization statistics must have F or T*F elements"
                )
            direction = direction * direction_std + direction_mean

        victim = torch.isclose(direction, torch.ones_like(direction), atol=1e-2, rtol=0.0)
        editable = ~victim
        if valid is not None:
            editable = editable & valid

        # Materialize independent writable storage before zeroing the DIR
        # coordinate; expanded views cannot be written to safely.
        packet_mask = editable.unsqueeze(-1).expand_as(x_packets).clone()
        packet_mask[..., direction_index] = False
        return packet_mask.reshape_as(x).to(dtype=x.dtype)


class ProblemSpaceAttack(BaseAttack):
    """
    Base class for all problem-space (raw traffic/PCAP) attacks.
    
    Mutate raw traffic before the preprocessor converts it into model features. 
    CRITICAL ARCHITECTURAL REQUIREMENT:
    Because problem-space attacks can inject or drop packets, the generated adversarial sequence 
    loses 1:1 positional alignment with the clean baseline. Consequently, the EvaluationEngine 
    disables post-hoc row-by-row directional immutability checks for these attacks, as array 
    differentials are mathematically undefined on unaligned sequences.

    To maintain threat-model integrity, concrete implementations MUST:
    1. Enforce packet ownership intrinsically during the PCAP mutation phase (e.g., strictly 
       verifying that victim packets are never modified).
    2. Explicitly populate the `intrinsic_feasible_mask` field in the returned `AttackResult` 
       with a 1D boolean numpy array of shape `(batch_size,)`.
    
    The framework operates on a strict fail-closed security principle. If `intrinsic_feasible_mask` 
    is returned as `None`, the EvaluationEngine will intercept it and intentionally raise a 
    RuntimeError to prevent unverified topological changes from bypassing the gatekeeper.
    """

    query_based: bool = True

    @abstractmethod
    def generate_raw(self, traffic: Any, y: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Adapt raw generation to the common result interface."""
        raw_adv = self.generate_raw(x, y, **kwargs)
        count = len(raw_adv) if hasattr(raw_adv, "__len__") else 1
        return AttackResult(adversarial=None, raw_adversarial=raw_adv, queries=np.zeros(count, dtype=np.int64))
