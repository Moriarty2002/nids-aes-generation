"""Class-conditional A2PM for normalized feature-space NIDS benchmarking.

The fitted pattern states are *physical float64* training observations; only
normalized float32 candidates are perturbed and sent to the target model.
Physical finalization, immutable restoration, and any optional lattice
projection belong exclusively to FeatureSpaceAttack's generate wrapper.

This is a fixed-state, query-based adaptation of A2PM's interval/combination
patterns, not a drop-in reimplementation of the upstream A2PMethod API (which
also supports online partial_fit, momentum and configurable pattern sequences)
because otherwise the attack would be non-deterministic and not reproducible, depending 
from the order of appearence of the samples in the batch. 
The original A2PMethod is designed for online learning, while this implementation 
is designed for offline training and evaluation.

* **A2PM**
  * **Paper:** *Adaptative Perturbation Patterns: Realistic Adversarial Learning for Robust Intrusion Detection* - Vitorino et al., Future Internet 2022
  * **Repository:** [https://github.com/vitorinojoao/a2pm](https://github.com/vitorinojoao/a2pm)
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from attacks.base import AttackResult, FeatureSpaceAttack
from core.registry import DATASETS, register_attack


_FORMAT_VERSION = 1


def _checkpoint_path(template: str | Path, seed: int) -> Path:
    """Resolve one worker's state artifact independently of the current seed batch."""
    return Path(str(template).format(seed=int(seed))).expanduser()


def _to_numpy(value: Any) -> np.ndarray:
    """Return a CPU view for metadata/labels, not for adversarial computation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _training_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    """Match AMD-GAN's reproducible 80/20 class-stratified fitting subset.

    The dataset registry exposes raw arrays rather than a uniform split API.
    Using the same split policy as AMD-GAN avoids learning from every held-out
    example merely because this attack trains a statistical state, not a network.
    """
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        rng.shuffle(positions)
        count = min(len(positions), max(1, math.ceil(0.8 * len(positions))))
        chosen.append(positions[:count])
    return np.sort(np.concatenate(chosen)).astype(np.int64)


def _read_classifier_schema(
    config: Mapping[str, Any], seed: int
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    """Use the same frozen normalization authority as AMD-GAN.

    Re-fitting a scaler on the attack's training subset would make observed
    categorical values incomparable to the ModelWrapper's inference inputs.
    """
    surrogate = config.get("surrogate_model")
    if isinstance(surrogate, Mapping) and surrogate.get("enabled", True):
        model_cfg = surrogate
    else:
        model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping) or not model_cfg.get("checkpoint"):
        raise ValueError("A2PM training requires model/surrogate_model.checkpoint")
    path = _checkpoint_path(str(model_cfg["checkpoint"]), seed)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
    # Framework model checkpoints can contain NumPy metadata; only load a
    # trusted local checkpoint, in accordance with AMD-GAN's existing loader.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Classifier checkpoint must contain a mapping")
    shape = tuple(int(v) for v in payload["input_shape"])
    if len(shape) not in (1, 2) or min(shape) < 1:
        raise ValueError(f"Unsupported model input_shape: {shape}")
    mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    std = np.asarray(payload["normalization_std"], dtype=np.float32)
    if mean.shape != std.shape or not np.isfinite(mean).all():
        raise ValueError("Invalid classifier normalization mean")
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid classifier normalization standard deviation")
    names = tuple(str(s).upper() for s in payload.get("feature_names", ()))
    return shape, mean, std, names


@register_attack("a2pm")
class A2PMAttack(FeatureSpaceAttack):
    """Learn class-conditioned intervals and locked categorical combinations.

    Feature indices are **packet-local**, not flattened sequence coordinates.
    Combination states are keyed by the *original* packet's locked values, so a
    sampled category can change only within the same protocol/port context.
    A missing class or locked-key match is a no-op rather than an arbitrary
    cross-class or cross-protocol substitution.
    """

    attack_name = "a2pm"
    query_based = True
    requires_gradients = False

    def __init__(
        self,
        *,
        continuous_idx: Sequence[int],
        categorical_idx: Sequence[int] = (),
        locked_features: Sequence[int | str] = (),
        interval_probability: float = 0.6,
        combination_probability: float = 0.4,
        max_iterations: int = 20,
        target_label: int = 0,
        seed: int = 42,
        feature_names: Sequence[str] = (),
        immutable_features: Sequence[str] = (),
        clip_ranges: Mapping[str, tuple[float, float]] | None = None,
        discrete_projection: bool = False,
        checkpoint: str | None = None,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # FeatureSpaceAttack is a keyword-only dataclass; orchestrator-injected
        # auxiliary Hydra keys need the same filtered forwarding as AMD-GAN.
        base_options = {
            "discrete_projection": bool(discrete_projection),
            "clip_ranges": dict(clip_ranges or {}),
            "feature_names": tuple(str(n).upper() for n in feature_names),
            "immutable_features": tuple(str(n).upper() for n in immutable_features),
            **kwargs,
        }
        base_init = super().__init__
        parameters = inspect.signature(base_init).parameters
        supports_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if "config" in parameters or supports_extra:
            base_init(config=config, **base_options)
        else:
            base_init(**{key: val for key, val in base_options.items() if key in parameters})

        self.config = config
        self.checkpoint = checkpoint
        self.seed = int(seed)
        self.target_label = int(target_label)
        self.continuous_idx = tuple(int(i) for i in continuous_idx)
        self.categorical_idx = tuple(int(i) for i in categorical_idx)
        self.locked_features = tuple(locked_features)
        self.interval_probability = float(interval_probability)
        self.combination_probability = float(combination_probability)
        self.max_iterations = int(max_iterations)

        chosen = self.continuous_idx + self.categorical_idx
        if any(i < 0 for i in chosen) or len(chosen) != len(set(chosen)):
            raise ValueError("Pattern indices must be nonnegative and disjoint")
        if not chosen:
            raise ValueError("A2PM requires at least one pattern feature")
        if not 0.0 <= self.interval_probability <= 1.0:
            raise ValueError("interval_probability must be between zero and one")
        if not 0.0 <= self.combination_probability <= 1.0:
            raise ValueError("combination_probability must be between zero and one")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        self._loaded_path: Path | None = None
        self._states: dict[int, dict[str, torch.Tensor]] = {}
        self._normalization_mean: torch.Tensor | None = None
        self._normalization_std: torch.Tensor | None = None
        self._input_shape: tuple[int, ...] = ()

    def _locked_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        """Resolve Hydra name/index locks only after knowing the model schema."""
        resolved: list[int] = []
        for entry in self.locked_features:
            if isinstance(entry, str):
                upper = entry.upper()
                if upper not in names:
                    raise ValueError(f"Unknown locked feature: {entry}")
                resolved.append(names.index(upper))
            else:
                resolved.append(int(entry))
        if len(resolved) != len(set(resolved)):
            raise ValueError("Duplicate locked features")
        if any(i < 0 or i >= len(names) for i in resolved):
            raise ValueError("Locked feature index outside packet width")
        if set(resolved).intersection(self.continuous_idx):
            raise ValueError("A locked feature cannot be interval-perturbable")
        return tuple(resolved)

    def _validate_schema(self, names: Sequence[str]) -> tuple[int, ...]:
        if not names or len(set(names)) != len(names):
            raise ValueError("A2PM requires ordered, unique feature_names")
        if max(self.continuous_idx + self.categorical_idx) >= len(names):
            raise ValueError("Pattern index outside packet width")
        locks = self._locked_indices(names)
        immutable = set(self.immutable_features) | ({"DIR"} if "DIR" in names else set())
        if immutable - set(names):
            raise ValueError(f"Unknown immutable names: {sorted(immutable - set(names))}")
        # Immutable or locked fields may appear in the *conditioning* records,
        # but must never be selected for interval changes or category swaps.
        forbidden = {names.index(name) for name in immutable}
        if forbidden.intersection(self.continuous_idx):
            raise ValueError("Immutable packet fields cannot be continuous attack features")
        return locks

    @classmethod
    def train(
        cls,
        *,
        config: Mapping[str, Any],
        seed: int,
        gpu: str,
        logger: logging.Logger | None = None,
    ) -> dict[str, float]:
        """Acquire per-malicious-class physical pattern states from training data.

        All statistics use original float64 records *before* normalization.
        Padding is excluded and the AMD-GAN stratified split is reused. Unique
        combinations are computed jointly: independent category histograms
        would invent feature pairs never observed in a malicious flow.
        The GPU parameter is accepted for the orchestrator lifecycle; this
        statistical fitting procedure intentionally needs no accelerator.
        """
        del gpu
        attack_cfg = config.get("attack", {})
        if not isinstance(attack_cfg, Mapping):
            raise ValueError("config.attack must be a mapping")
        nested = attack_cfg.get("params", {})
        if not isinstance(nested, Mapping):
            raise ValueError("config.attack.params must be a mapping")
        accepted = {
            "continuous_idx", "categorical_idx", "locked_features",
            "interval_probability", "combination_probability", "max_iterations",
            "target_label", "feature_names", "immutable_features", "clip_ranges",
            "discrete_projection", "checkpoint",
        }
        options = {k: v for k, v in {**nested, **attack_cfg}.items() if k in accepted}
        settings = cls(config=config, seed=seed, **options)
        shape, mean, std, checkpoint_names = _read_classifier_schema(config, seed)

        dataset_cfg = config.get("dataset")
        if not isinstance(dataset_cfg, Mapping) or not dataset_cfg.get("name"):
            raise ValueError("A2PM requires dataset.name and dataset.params")
        constructor = DATASETS.get(str(dataset_cfg["name"]))
        dataset_params = dict(dataset_cfg.get("params", {}))
        try:
            if "flatten" in inspect.signature(constructor).parameters:
                dataset_params["flatten"] = len(shape) == 1
        except (TypeError, ValueError):
            pass
        dataset = constructor(**dataset_params)
        raw, labels = dataset.raw_arrays()
        physical = _to_numpy(raw)
        labels = _to_numpy(labels).reshape(-1)
        if physical.dtype != np.float64:
            raise TypeError("A2PM requires original physical float64 from dataset.raw_arrays()")
        if physical.ndim not in (2, 3) or tuple(physical.shape[1:]) != shape:
            raise ValueError("Training dataset shape disagrees with classifier checkpoint")
        if len(labels) != len(physical) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Training labels must be one integer per flow")
        if (labels < 0).any():
            raise ValueError("Training labels must be nonnegative")
        names = tuple(str(n).upper() for n in getattr(dataset, "feature_names", ()))
        if names and checkpoint_names and names != checkpoint_names:
            raise ValueError("Training data feature order differs from classifier")
        if settings.feature_names and checkpoint_names and settings.feature_names != checkpoint_names:
            raise ValueError("Configured feature order differs from classifier")
        if settings.feature_names and names and settings.feature_names != names:
            raise ValueError("Configured feature order differs from dataset")
        settings.feature_names = settings.feature_names or names or checkpoint_names
        locks = settings._validate_schema(settings.feature_names)
        width = len(settings.feature_names)
        if physical.ndim == 3 and physical.shape[-1] != width:
            raise ValueError("Sequence packet width differs from feature_names")
        if physical.ndim == 2 and physical.shape[-1] % width:
            raise ValueError("Flattened physical width not divisible by feature width")
        packets = physical.reshape(len(physical), -1, width)
        packet_count = packets.shape[1]
        if mean.size not in (width, packet_count * width):
            raise ValueError("Classifier normalization must have F or T*F entries")

        raw_mask = getattr(dataset, "valid_mask", None)
        if raw_mask is None and packet_count > 1:
            raise ValueError("Sequence training data requires valid_mask for padding exclusion")
        valid = (np.ones(packets.shape[:2], dtype=bool) if raw_mask is None
                 else np.asarray(_to_numpy(raw_mask), dtype=bool))
        if valid.shape != packets.shape[:2]:
            raise ValueError("Training valid_mask disagrees with packet layout")

        chosen = _training_indices(labels, seed)
        mutable_categories = tuple(i for i in settings.categorical_idx if i not in locks
                                   and settings.feature_names[i] not in settings.immutable_features
                                   and settings.feature_names[i] != "DIR")
        combination_columns = mutable_categories + tuple(i for i in locks if i not in mutable_categories)
        states: dict[int, dict[str, torch.Tensor]] = {}
        observed_packets = 0
        for label in sorted(int(v) for v in np.unique(labels[chosen]) if int(v) != settings.target_label):
            per_class = chosen[labels[chosen] == label]
            observations = packets[per_class][valid[per_class]]
            if not len(observations):
                continue
            if not np.isfinite(observations).all():
                raise ValueError(f"Non-finite physical training observations in class {label}")
            minimum = np.min(observations[:, settings.continuous_idx], axis=0) if settings.continuous_idx else np.empty(0, dtype=np.float64)
            maximum = np.max(observations[:, settings.continuous_idx], axis=0) if settings.continuous_idx else np.empty(0, dtype=np.float64)
            combinations = (np.unique(observations[:, combination_columns], axis=0)
                            if mutable_categories else np.empty((0, 0), dtype=np.float64))
            states[label] = {
                "minimum": torch.from_numpy(np.ascontiguousarray(minimum)),
                "maximum": torch.from_numpy(np.ascontiguousarray(maximum)),
                "combinations": torch.from_numpy(np.ascontiguousarray(combinations)),
            }
            observed_packets += len(observations)

        if not states:
            raise ValueError("A2PM training found no valid malicious class observations")
        destination = settings._checkpoint_for_seed(seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "attack_name": "a2pm",
            "format_version": _FORMAT_VERSION,
            "seed": int(seed),
            "input_shape": tuple(shape),
            "feature_names": tuple(settings.feature_names),
            "immutable_features": tuple(settings.immutable_features),
            "continuous_idx": settings.continuous_idx,
            "categorical_idx": settings.categorical_idx,
            "locked_idx": locks,
            "mutable_categorical_idx": mutable_categories,
            "combination_columns": combination_columns,
            "normalization_mean": torch.from_numpy(np.ascontiguousarray(mean)),
            "normalization_std": torch.from_numpy(np.ascontiguousarray(std)),
            "states": states,
        }
        temporary = destination.with_suffix(".pt.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        count = sum(int(s["combinations"].shape[0]) for s in states.values())
        if logger:
            logger.info("a2pm.train seed=%d classes=%d packets=%d combinations=%d checkpoint=%s",
                        seed, len(states), observed_packets, count, destination)
        return {
            "a2pm_trained_classes": float(len(states)),
            "a2pm_training_packets": float(observed_packets),
            "a2pm_tracked_combinations": float(count),
        }

    def _checkpoint_for_seed(self, seed: int) -> Path:
        template = self.checkpoint
        if not template:
            root = "runs" if self.config is None else str(self.config.get("output_dir", "runs"))
            template = str(Path(root) / "checkpoints" / "a2pm_seed{seed}.pt")
        return _checkpoint_path(template, seed)

    def _load_patterns(self, seed: int) -> None:
        """Pin a trained worker seed on first use; batch RNG seeds may differ.

        EvaluationEngine adds the batch offset to kwargs['seed']. It must
        reseed sampling, not accidentally select a different learned manifold.
        """
        if self._loaded_path is not None:
            return
        path = self._checkpoint_for_seed(seed)
        if not path.is_file():
            raise FileNotFoundError(f"A2PM checkpoint missing: {path}; run train_attack first")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("A2PM checkpoint must contain a mapping")
        if state.get("attack_name") != "a2pm" or state.get("format_version") != _FORMAT_VERSION:
            raise ValueError("Unsupported A2PM checkpoint format")
        if int(state.get("seed", -1)) != seed:
            raise ValueError("A2PM checkpoint seed mismatch")
        names = tuple(state["feature_names"])
        if self.feature_names and tuple(self.feature_names) != names:
            raise ValueError("A2PM checkpoint feature schema mismatch")
        self.feature_names = names
        locks = self._validate_schema(names)
        for setting, expected in (
            ("continuous_idx", self.continuous_idx),
            ("categorical_idx", self.categorical_idx),
            ("locked_idx", locks),
            ("immutable_features", tuple(self.immutable_features)),
        ):
            if tuple(state[setting]) != expected:
                raise ValueError(f"A2PM checkpoint {setting} mismatch; retrain")
        self._states = state["states"]
        self._normalization_mean = state["normalization_mean"]
        self._normalization_std = state["normalization_std"]
        self._input_shape = tuple(state["input_shape"])
        self._loaded_path = path

    def _normalization_for_positions(
        self, mean: torch.Tensor, std: torch.Tensor, packet_count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align per-feature or per-position normalization to (T,F)."""
        width = len(self.feature_names)
        if mean.numel() == width:
            return mean.reshape(1, width).expand(packet_count, width), std.reshape(1, width).expand(packet_count, width)
        if mean.numel() == packet_count * width:
            return mean.reshape(packet_count, width), std.reshape(packet_count, width)
        raise ValueError("Generation normalization requires F or T*F elements")

    @staticmethod
    def _normalize_state(physical: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Encode checkpoint statistics once, without decoding attack candidates.

        Stored physical state is float64 by design. This conversion uses the
        fitted float32 classifier mean/std as exact binary values in float64
        intermediate arithmetic, then rounds to the float32 inference space.
        Only *pattern states* are converted, never the candidate or its source.
        """
        return ((physical.to(torch.float64) - mean.to(torch.float64)) /
                std.to(torch.float64)).to(torch.float32)

    def _combination_lookup(
        self, state: Mapping[str, torch.Tensor], time: int,
        mean: torch.Tensor, std: torch.Tensor, lock_idx: tuple[int, ...],
        mutable_idx: tuple[int, ...], columns: tuple[int, ...],
    ) -> dict[tuple[float, ...], torch.Tensor]:
        """Group jointly observed categories by an *unmodified* lock tuple."""
        combinations = state["combinations"]
        if not combinations.numel():
            return {}
        normalized = self._normalize_state(combinations, mean[time, list(columns)], std[time, list(columns)])
        lock_positions = [columns.index(i) for i in lock_idx]
        mutable_positions = [columns.index(i) for i in mutable_idx]
        grouped: dict[tuple[float, ...], list[torch.Tensor]] = {}
        physical_locks: dict[tuple[float, ...], set[tuple[float, ...]]] = {}
        for physical, row in zip(combinations, normalized):
            key = tuple(float(row[i]) for i in lock_positions)
            raw_key = tuple(float(physical[i]) for i in lock_positions)
            grouped.setdefault(key, []).append(row[mutable_positions])
            physical_locks.setdefault(key, set()).add(raw_key)
        # A float32 normalization may map two distinct physical port/protocol
        # states to the same key. Such a key is ambiguous; never mix its
        # categorical alternatives across different physical lock states.
        return {
            key: torch.stack(rows)
            for key, rows in grouped.items() if len(physical_locks[key]) == 1
        }

    def _predict(self, candidates: torch.Tensor, original_is_tensor: bool) -> np.ndarray:
        """Query the polymorphic ModelWrapper exactly once per supplied flow."""
        model = self._require_model()
        argument: Any = candidates if original_is_tensor else candidates.detach().cpu().numpy()
        with torch.no_grad():
            prediction = _to_numpy(model.predict(argument)).reshape(-1)
        if len(prediction) != len(candidates):
            raise ValueError("ModelWrapper.predict must return one label per flow")
        return prediction.astype(np.int64, copy=False)

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Search class-specific feasible perturbations using prediction queries.

        The clean baseline costs one query per flow and identifies flows already
        at the target label. Each later query is counted only for a *changed*
        candidate. Successful flows are frozen. When no iteration yields any
        new success, the search stops: without gradients or a proxy objective,
        additional random mutations have no grounded optimization signal.

        The mandatory packet mask is intersected **before model prediction**,
        never deferred to physical postprocessing. Inference, returned inputs,
        and the working candidate remain float32 at all times.
        """
        self._require_model()
        original_is_tensor = isinstance(x, torch.Tensor)
        if original_is_tensor and x.dtype != torch.float32:
            raise TypeError("A2PM requires normalized float32 inputs")
        if not original_is_tensor and np.asarray(x).dtype != np.float32:
            raise TypeError("A2PM requires normalized float32 inputs")
        device = x.device if original_is_tensor else torch.device("cpu")
        x0 = torch.as_tensor(x, dtype=torch.float32, device=device).detach()
        if x0.ndim not in (2, 3) or not torch.isfinite(x0).all():
            raise ValueError("A2PM requires finite (B,T*F)/(B,T,F) inputs")
        labels = np.asarray(_to_numpy(y), dtype=np.int64).reshape(-1)
        if len(labels) != len(x0):
            raise ValueError("A2PM input/label batch mismatch")
        seed = int(kwargs.get("seed", self.seed))
        self._load_patterns(seed)
        if tuple(x0.shape[1:]) != self._input_shape:
            raise ValueError("A2PM generation topology differs from trained checkpoint")
        width = len(self.feature_names)
        if x0.ndim == 3 and x0.shape[-1] != width:
            raise ValueError("A2PM sequence packet width differs from feature_names")
        if x0.ndim == 2 and x0.shape[-1] % width:
            raise ValueError("A2PM flattened feature width is incompatible with input")
        packets = x0.reshape(len(x0), -1, width)
        packet_count = packets.shape[1]
        locks = self._locked_indices(self.feature_names)
        mutable = tuple(i for i in self.categorical_idx if i not in locks
                        and self.feature_names[i] not in self.immutable_features
                        and self.feature_names[i] != "DIR")
        columns = mutable + tuple(i for i in locks if i not in mutable)

        runtime_mean = kwargs.get("normalization_mean")
        runtime_std = kwargs.get("normalization_std")
        if runtime_mean is None or runtime_std is None:
            raise ValueError("A2PM requires normalization_mean/std")
        mean = torch.as_tensor(runtime_mean, dtype=torch.float32, device="cpu").reshape(-1)
        std = torch.as_tensor(runtime_std, dtype=torch.float32, device="cpu").reshape(-1)
        if (not torch.equal(mean, self._normalization_mean.reshape(-1)) or
                not torch.equal(std, self._normalization_std.reshape(-1))):
            raise ValueError("Runtime normalization differs from fitted A2PM checkpoint")
        position_mean, position_std = self._normalization_for_positions(mean, std, packet_count)

        permitted = self.victim_packet_perturbation_mask(
            x0, self.feature_names,
            normalization_mean=runtime_mean, normalization_std=runtime_std,
            valid_mask=kwargs.get("valid_mask"),
        ).reshape_as(packets).bool()
        # Apply the framework's *additional* immutable list. DIR and whole
        # victim/padding rows are already handled by the mandatory mask.
        for name in self.immutable_features:
            permitted[..., self.feature_names.index(name)] = False
        if locks:
            permitted[..., list(locks)] = False

        target = int(kwargs.get("target_label", self.target_label))
        rng = np.random.default_rng(seed)
        adv = x0.clone()
        counts = np.zeros(len(x0), dtype=np.int64)
        baseline = self._predict(adv, original_is_tensor)
        counts += 1
        success = baseline == target
        active = np.logical_and(~success, labels != target)
        attempts = 0
        stop_reason = "all_evaded_or_ineligible" if not active.any() else "max_iterations"
        interval_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        combination_cache: dict[tuple[int, int], dict[tuple[float, ...], torch.Tensor]] = {}

        for _ in range(self.max_iterations):
            if not active.any():
                stop_reason = "all_evaded_or_ineligible"
                break
            candidate = adv.clone()
            proposed = candidate.reshape_as(packets)
            # The baseline direction/locked fields are never used for a swap's
            # *output*, only for choosing a valid category record.
            source = x0.reshape_as(packets)
            for label in sorted(set(int(n) for n in labels[active])):
                state = self._states.get(label)
                if state is None:
                    continue
                ids = np.flatnonzero(active & (labels == label))
                rows = torch.as_tensor(ids, device=device, dtype=torch.long)
                for time in range(packet_count):
                    if not bool(permitted[rows, time].any()):
                        continue
                    if self.continuous_idx and self.interval_probability > 0:
                        selected = rng.random(len(ids)) < self.interval_probability
                        if selected.any():
                            chosen = rows[torch.as_tensor(selected, device=device)]
                            indices = list(self.continuous_idx)
                            cache_key = (label, time)
                            if cache_key not in interval_cache:
                                lower = self._normalize_state(
                                    state["minimum"], position_mean[time, indices], position_std[time, indices]
                                ).to(device)
                                upper = self._normalize_state(
                                    state["maximum"], position_mean[time, indices], position_std[time, indices]
                                ).to(device)
                                interval_cache[cache_key] = (lower, upper)
                            lower, upper = interval_cache[cache_key]
                            before = proposed[chosen, time][:, indices]
                            base = torch.maximum(torch.minimum(before, upper), lower)
                            epsilon = torch.from_numpy(
                                1.0 - rng.random(base.shape, dtype=np.float32)
                            ).to(device)
                            upward = torch.from_numpy(rng.random(base.shape) < 0.5).to(device)
                            after = torch.where(upward, base + epsilon * (upper - base),
                                                base - epsilon * (base - lower))
                            allowed = permitted[chosen, time][:, indices]
                            proposed[chosen[:, None], time, indices] = torch.where(allowed, after, before)

                    if not mutable or self.combination_probability <= 0:
                        continue
                    cache_key = (label, time)
                    if cache_key not in combination_cache:
                        combination_cache[cache_key] = self._combination_lookup(
                            state, time, position_mean, position_std, locks, mutable, columns
                        )
                    lookup = combination_cache[cache_key]
                    if not lookup:
                        continue
                    for row_id in ids:
                        if rng.random() >= self.combination_probability:
                            continue
                        if not bool(permitted[row_id, time, list(mutable)].any()):
                            continue
                        key = tuple(float(v) for v in source[row_id, time, list(locks)].tolist())
                        matching = lookup.get(key)
                        if matching is None:
                            continue
                        chosen_pattern = matching[int(rng.integers(len(matching)))].to(device)
                        before = proposed[row_id, time, list(mutable)]
                        allowed = permitted[row_id, time, list(mutable)]
                        proposed[row_id, time, list(mutable)] = torch.where(allowed, chosen_pattern, before)

            # An explicit last intersection, before calling ModelWrapper, makes
            # the zero-edit invariant independent of individual pattern logic.
            candidate = torch.where(permitted.reshape_as(x0), candidate, adv)
            changed = torch.any(candidate.reshape(len(x0), -1) != adv.reshape(len(x0), -1), dim=1)
            changed_ids = np.flatnonzero(active & changed.detach().cpu().numpy())
            if not len(changed_ids):
                stop_reason = "no_new_evasions"
                break
            proposed_predictions = self._predict(candidate[changed_ids], original_is_tensor)
            counts[changed_ids] += 1
            new_success = proposed_predictions == target
            success[changed_ids] |= new_success
            adv = candidate
            active = np.logical_and(~success, labels != target)
            attempts += 1
            if not new_success.any():
                stop_reason = "no_new_evasions"
                break
            if not active.any():
                stop_reason = "all_evaded_or_ineligible"
                break

        return AttackResult(
            adversarial=adv,
            success_mask=success,
            queries=counts,
            feature_names=tuple(self.feature_names),
            metadata={
                "attack": "a2pm", "iterations": attempts,
                "max_iterations": self.max_iterations,
                "stop_reason": stop_reason,
                "checkpoint": str(self._loaded_path),
                "learned_classes": len(self._states),
                "baseline_target_predictions": int(np.sum(baseline == target)),
                "successes_before_projection": int(np.sum(success)),
                "total_queries": int(counts.sum()),
            },
        )
