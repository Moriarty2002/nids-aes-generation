from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from attacks.base import AttackResult, BaseAttack, FeatureSpaceAttack, ProblemSpaceAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from constraints.rules import DirectionalImmutabilityRule, RuleSet
from defenses.base import BaseDetector
from models.wrapper import ModelWrapper
from preprocessing.base import BasePreprocessor
from core.dual_representation import decode_physical64, encode_normalized32, layout
from .metrics import detection_auroc, evaluate_predictions


@dataclass(frozen=True)
class FeatureNormalization:
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.asarray(self.mean, dtype=np.float32))
        object.__setattr__(self, "std", np.clip(np.asarray(self.std, dtype=np.float32), 1e-12, None))

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Decode for convenience only, not recovery of pre-float32 truth."""
        return decode_physical64(x, self.feature_names, self.mean, self.std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Canonical physical float64 -> model float32 encoding."""
        return encode_normalized32(x, self.feature_names, self.mean, self.std)


@dataclass(frozen=True)
class EvaluationOutput:
    metrics: dict[str, float]
    baseline_predictions: np.ndarray
    adversarial_predictions: np.ndarray
    accepted_mask: np.ndarray
    scored_mask: np.ndarray
    attack_result: AttackResult
    detector_clean_scores: np.ndarray | None = None
    detector_adversarial_scores: np.ndarray | None = None


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sync_if_cuda(model: ModelWrapper) -> None:
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _timed_predict(model: ModelWrapper, x: Any) -> tuple[np.ndarray, float]:
    """Measure full target prediction wall time, including device synchronization."""
    _sync_if_cuda(model)
    start = perf_counter()
    predictions = np.asarray(model.predict(x), dtype=np.int64)
    _sync_if_cuda(model)
    elapsed = perf_counter() - start
    return predictions, max(elapsed, np.finfo(np.float64).eps)


def _slice_batch(value: Any, start: int, stop: int) -> Any:
    """Slice array-like batch inputs without changing their container type."""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value[start:stop]
    if isinstance(value, (list, tuple)):
        return value[start:stop]
    raise TypeError(f"Cannot batch value of type {type(value).__name__}")


PhysicalSource = np.ndarray | torch.Tensor | Callable[[int, int], np.ndarray | torch.Tensor]


def _slice_physical_source(
    source: PhysicalSource, start: int, stop: int
) -> np.ndarray | torch.Tensor:
    # A callable source permits on-demand packet extraction / memory maps with
    # no mandatory full-dataset float64 physical allocation.
    value = source(start, stop) if callable(source) else _slice_batch(source, start, stop)
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError("physical source must return np.ndarray or torch.Tensor")
    if value.dtype != (torch.float64 if isinstance(value, torch.Tensor) else np.float64):
        raise TypeError("physical source must supply original float64 samples")
    return value


def _batched_predict(
    model: ModelWrapper,
    x: Any,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Predict in bounded batches and return predictions in input order."""
    n_samples = len(x)
    predictions: list[np.ndarray] = []
    elapsed = 0.0
    for start in range(0, n_samples, batch_size):
        batch_predictions, batch_elapsed = _timed_predict(
            model,
            _slice_batch(x, start, min(start + batch_size, n_samples)),
        )
        predictions.append(batch_predictions)
        elapsed += batch_elapsed
    if not predictions:
        return np.empty(0, dtype=np.int64), np.finfo(np.float64).eps
    return np.concatenate(predictions), max(elapsed, np.finfo(np.float64).eps)


def _batched_detector_score(
    detector: BaseDetector,
    model: ModelWrapper,
    x: Any,
    batch_size: int,
) -> np.ndarray:
    """Score inputs through the detector in bounded batches to prevent OOM."""
    n_samples = len(x)
    scores: list[np.ndarray] = []
    for start in range(0, n_samples, batch_size):
        batch_scores = detector.score(
            _slice_batch(x, start, min(start + batch_size, n_samples)),
            model,
        )
        scores.append(np.asarray(batch_scores, dtype=np.float64).reshape(-1))
    if not scores:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(scores)


def _chunked_perturbation_distances(
    x: np.ndarray,
    x_adv: np.ndarray,
    sample_mask: np.ndarray,
    *,
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute distances in chunks, avoiding full-corpus float64 copies and deltas."""
    a = np.asarray(x)
    b = np.asarray(x_adv)
    mask = np.asarray(sample_mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"x and x_adv shapes differ: {a.shape} != {b.shape}")
    if a.ndim < 2:
        raise ValueError(f"Expected batch-shaped arrays, received {a.shape}")
    if len(mask) != len(a):
        raise ValueError("sample_mask and batch length must match")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    l0_total = 0.0
    l2_total = 0.0
    linf_total = 0.0
    selected_count = 0
    for start in range(0, len(a), batch_size):
        stop = min(start + batch_size, len(a))
        local_mask = mask[start:stop]
        if not np.any(local_mask):
            continue
        # Boolean indexing creates copies only for one bounded chunk.
        a_chunk = a[start:stop][local_mask]
        b_chunk = b[start:stop][local_mask]
        delta = np.subtract(b_chunk, a_chunk, dtype=np.float64).reshape(int(local_mask.sum()), -1)
        l0_total += float(np.count_nonzero(np.abs(delta) > 1e-12, axis=1).sum(dtype=np.int64))
        l2_total += float(np.linalg.norm(delta, ord=2, axis=1).sum(dtype=np.float64))
        linf_total += float(np.linalg.norm(delta, ord=np.inf, axis=1).sum(dtype=np.float64))
        selected_count += int(local_mask.sum())

    if selected_count == 0:
        return {"mean_l0": 0.0, "mean_l2": 0.0, "mean_linf": 0.0}
    return {
        "mean_l0": l0_total / selected_count,
        "mean_l2": l2_total / selected_count,
        "mean_linf": linf_total / selected_count,
    }


def _restore_target_physical_protected_inplace(
    adversarial: np.ndarray,
    original: np.ndarray,
    feature_names: Sequence[str],
    *,
    immutable_features: Sequence[str] = (),
    valid_mask: np.ndarray | torch.Tensor | None = None,
) -> tuple[int, int]:
    """Last-stage physical immutability projection, in-place without drift."""
    if adversarial.dtype != np.float64 or original.dtype != np.float64:
        raise TypeError("Both authoritative arrays must be float64")
    if adversarial.shape != original.shape or not adversarial.flags.writeable:
        raise ValueError("Physical batches must match and adversarial must be writable")
    b, t, f = layout(adversarial, feature_names)
    names = tuple(str(n).upper() for n in feature_names)
    adv = adversarial.reshape(b, t, f)
    clean = original.reshape(b, t, f)
    valid = None if valid_mask is None else _as_numpy(valid_mask).astype(bool, copy=False)
    if valid is not None and valid.shape != (b, t):
        raise ValueError("Invalid valid_mask shape")
    immutable = {str(n).upper() for n in immutable_features}
    if "DIR" in names:
        immutable.add("DIR")
    if immutable - set(names):
        raise ValueError(f"Unknown immutable features: {sorted(immutable-set(names))}")
    victim_count = 0
    if "DIR" in names:
        victim = np.isclose(clean[..., names.index("DIR")], 1.0, atol=1e-2, rtol=0.0)
        if valid is not None:
            victim &= valid
        victim_count = int(np.count_nonzero(victim))
        np.copyto(adv, clean, where=victim[..., None])
    for key in immutable:
        j = names.index(key)
        np.copyto(adv[..., j], clean[..., j])
    padded = 0
    if valid is not None:
        padded = int(np.count_nonzero(~valid))
        np.copyto(adv, clean, where=(~valid)[..., None])
    return victim_count, padded


def _as_physical_shape(
    raw: np.ndarray | torch.Tensor,
    expected: np.ndarray | torch.Tensor,
    names: Sequence[str],
) -> np.ndarray:
    """Accept flattened/sequence-equivalent physical layouts without copying."""
    physical = _as_numpy(raw)
    if physical.dtype != np.float64:
        raise TypeError("Physical source must be float64, not normalized float32")
    b, t, f = layout(expected, names)
    if physical.shape[0] != b or physical.size != b * t * f:
        raise ValueError("Physical batch and target batch dimensions differ")
    return physical.reshape(tuple(expected.shape))


def _violation_counts(gate_result: Any) -> dict[str, float]:
    """Count rejected samples per concrete gatekeeper rule."""
    counter: Counter[str] = Counter()
    for violation in tuple(getattr(gate_result, "violations", ())):
        rule = str(getattr(violation, "rule", "unknown"))
        indices = getattr(violation, "indices", ())
        try:
            count = len(np.unique(np.asarray(indices, dtype=np.int64)))
        except (TypeError, ValueError):
            count = 0
        counter[rule] += int(count)
    return {f"violation_{rule}": float(count) for rule, count in sorted(counter.items())}


class EvaluationEngine:
    """Evaluate generated traffic without leaking target-model information to attacks.

    The engine first records clean target predictions, then generates traffic
    through the attack or surrogate, and only afterward applies constraints and
    target scoring. This ordering keeps transferability and realizability
    measurements independent from the attack's internal success estimate.
    """

    def __init__(
        self,
        target_model: ModelWrapper,
        attack: BaseAttack,
        gatekeeper: UniversalConstraintGatekeeper,
        *,
        attack_model: ModelWrapper | None = None,
        preprocessor: BasePreprocessor | None = None,
        attack_normalization: FeatureNormalization | None = None,
        target_normalization: FeatureNormalization | None = None,
        class_labels: Sequence[int] | None = None,
        class_names: Sequence[str] | None = None,
        benign_label: int = 0,
        target_label: int | None = None,
        c_fp: float = 1.0,
        c_fn: float = 1.0,
        strict_constraints: bool | None = None,
        strict_multiclass: bool = True,
        detector: BaseDetector | None = None,
        batch_size: int = 256,
        max_samples: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.target_model = target_model
        self.attack_model = attack_model or target_model
        self.attack = attack.bind_model(self.attack_model)
        self.gatekeeper = gatekeeper
        # Differential packet comparisons have no positional meaning after
        # raw packet insertion/deletion. Construct a separate immutable oracle
        # policy instead of mutating the feature-space Gatekeeper's RuleSet.
        # All non-differential physical rules and the exact normalized/physical
        # encoding verification remain active for problem-space candidates.
        if isinstance(self.attack, ProblemSpaceAttack):
            aligned_rules = tuple(
                rule for rule in gatekeeper.rules.rules
                if not isinstance(rule, DirectionalImmutabilityRule)
            )
            self._problem_space_gatekeeper = UniversalConstraintGatekeeper(
                rules=RuleSet(aligned_rules),
                strict=gatekeeper.strict,
                normalization_mean=gatekeeper.normalization_mean,
                normalization_std=gatekeeper.normalization_std,
                logger=gatekeeper.logger,
                discrete_diagnostics=gatekeeper.discrete_diagnostics,
            )
        else:
            self._problem_space_gatekeeper = None
        self.preprocessor = preprocessor
        self.attack_normalization = attack_normalization
        self.target_normalization = target_normalization
        self.class_labels = None if class_labels is None else tuple(int(value) for value in class_labels)
        self.class_names = None if class_names is None else tuple(str(value) for value in class_names)
        self.benign_label = int(benign_label)
        self.target_label = None if target_label is None else int(target_label)
        self.c_fp = float(c_fp)
        self.c_fn = float(c_fn)
        self.strict_constraints = gatekeeper.strict if strict_constraints is None else bool(strict_constraints)
        self.strict_multiclass = bool(strict_multiclass)
        self.detector = detector
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be >= 1 or None")
        self.max_samples = max_samples
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = int(batch_size)
        self.logger = logger

    def run(
        self,
        x: Any,
        y: np.ndarray | torch.Tensor,
        *,
        feature_names: Sequence[str],
        attack_input: Any | None = None,
        attack_input_is_target_normalized: bool = False,
        clean_physical: PhysicalSource | None = None,
        authoritative_physical_output: np.ndarray | None = None,
        raw_traffic: Any | None = None,
        valid_mask: np.ndarray | torch.Tensor | None = None,
        attack_kwargs: Mapping[str, Any] | None = None,
    ) -> EvaluationOutput:
        """Generate, validate, score, and summarize one evaluation batch.

        ``attack_input_is_target_normalized`` permits transfer attacks to convert
        each target-normalized batch to surrogate-normalized space on demand,
        without retaining a second normalized copy of the entire dataset.
        """
        y_np = _as_numpy(y).astype(np.int64, copy=False)
        x_target = x
        if clean_physical is None:
            raise ValueError(
                "Dual-authoritative evaluation requires clean_physical captured BEFORE "
                "normalization/float32 quantization; decoding x cannot recover physical truth"
            )
        if not callable(clean_physical):
            if not isinstance(clean_physical, (np.ndarray, torch.Tensor)):
                raise TypeError("clean_physical must be a float64 batch or a batch supplier")
            if len(clean_physical) != len(x_target):
                raise ValueError("clean_physical and x must have identical batch length")
        if self.max_samples is not None and len(y_np) > self.max_samples:
            raise ValueError(
                "EvaluationEngine received more samples than evaluation.max_samples; "
                "subsample before normalization in evaluate_attack_seed"
            )
        if authoritative_physical_output is not None:
            if (not isinstance(authoritative_physical_output, np.ndarray)
                    or authoritative_physical_output.dtype != np.float64
                    or not authoritative_physical_output.flags.writeable
                    or authoritative_physical_output.shape != tuple(x_target.shape)):
                raise ValueError(
                    "authoritative_physical_output must be a writable float64 "
                    "array or np.memmap with exactly the target batch shape"
                )
        if attack_input_is_target_normalized:
            if isinstance(self.attack, ProblemSpaceAttack):
                raise ValueError("On-demand normalization conversion is only valid for feature-space attacks")
            if self.attack_normalization is None or self.target_normalization is None:
                raise ValueError(
                    "On-demand target-to-attack normalization requires both normalization objects"
                )
        
        # The clean target input is also required to originate from the exact
        # physical baseline; otherwise model and oracle already disagree before
        # the attack runs. The check is chunked, no corpus-sized temporary.
        for verify_start in range(0, len(y_np), self.batch_size):
            verify_stop = min(verify_start + self.batch_size, len(y_np))
            reference = _as_numpy(_slice_batch(x_target, verify_start, verify_stop))
            reference_physical = _as_physical_shape(
                _slice_physical_source(clean_physical, verify_start, verify_stop), reference, feature_names
            )
            canonical = encode_normalized32(
                reference_physical, feature_names,
                None if self.target_normalization is None else self.target_normalization.mean,
                None if self.target_normalization is None else self.target_normalization.std,
            )
            if reference.dtype != np.float32 or not np.array_equal(
                np.ascontiguousarray(reference).view(np.uint32),
                np.ascontiguousarray(canonical).view(np.uint32),
            ):
                raise RuntimeError(
                    f"Clean model/physical encoding mismatch in batch "
                    f"{verify_start}:{verify_stop}; re-encode clean inputs "
                    "from pre-normalization float64 physical source"
                )

        baseline_pred, baseline_elapsed = _batched_predict(
            self.target_model, x_target, self.batch_size
        )
        if len(baseline_pred) != len(y_np):
            raise ValueError("Target baseline predictions and labels have different lengths")

        n_samples = len(y_np)
        if self.logger:
            self.logger.info(
                "evaluation.engine_start samples=%d batch_size=%d attack=%s attack_model=%s "
                "target_model=%s strict_constraints=%s detector=%s",
                n_samples, self.batch_size, type(self.attack).__name__,
                type(self.attack_model).__name__, type(self.target_model).__name__,
                self.strict_constraints, type(self.detector).__name__ if self.detector else "none",
            )
        baseline_latency_ms = baseline_elapsed * 1000.0 / max(n_samples, 1)
        baseline_throughput = n_samples / baseline_elapsed if n_samples else 0.0
        if self.logger:
            self.logger.info(
                "evaluation.baseline_done samples=%d latency_ms_per_sample=%.3f throughput=%.2f",
                n_samples, baseline_latency_ms, baseline_throughput,
            )

        if isinstance(self.attack, ProblemSpaceAttack):
            target_input = raw_traffic if raw_traffic is not None else attack_input
            if target_input is None:
                raise ValueError("Problem-space evaluation requires raw_traffic or attack_input")
        else:
            target_input = x if attack_input is None else attack_input

        effective_attack_kwargs = dict(attack_kwargs or {})
        if self.attack_normalization is not None and not isinstance(self.attack, ProblemSpaceAttack):
            effective_attack_kwargs.setdefault("normalization_mean", self.attack_normalization.mean)
            effective_attack_kwargs.setdefault("normalization_std", self.attack_normalization.std)

        success_batches: list[np.ndarray] = []
        query_batches: list[np.ndarray] = []
        accepted_batches: list[np.ndarray] = []
        scored_batches: list[np.ndarray] = []
        adversarial_prediction_batches: list[np.ndarray] = []
        clean_detector_batches: list[np.ndarray] = []
        adversarial_detector_batches: list[np.ndarray] = []
        y_score_batches: list[np.ndarray] = []
        baseline_score_batches: list[np.ndarray] = []
        query_score_batches: list[np.ndarray] = []
        violation_counts: Counter[str] = Counter()
        first_feature_names: tuple[str, ...] | None = None
        first_feature_shape: tuple[int, ...] | None = None
        x_adv_target: np.ndarray | None = None
        success_available: bool | None = None
        queries_available: bool | None = None
        first_metadata: dict[str, Any] = {}

        if self.detector is not None and not callable(getattr(self.target_model, "extract_features", None)):
            raise TypeError(
                f"Configured detector requires target wrapper {type(self.target_model).__name__} "
                "to implement extract_features(x)"
            )

        for start in range(0, n_samples, self.batch_size):
            stop = min(start + self.batch_size, n_samples)
            batch_kwargs = dict(effective_attack_kwargs)
            if batch_kwargs.get("seed") is not None:
                batch_kwargs["seed"] = int(batch_kwargs["seed"]) + start

            clean_target_physical = _as_physical_shape(
                _slice_physical_source(clean_physical, start, stop),
                _slice_batch(x_target, start, stop),
                feature_names,
            )
            attack_batch_input = _slice_batch(target_input, start, stop)
            if attack_input_is_target_normalized:
                # Preserve physical truth while changing model normalization.
                attack_batch_input = encode_normalized32(
                    clean_target_physical, feature_names,
                    self.attack_normalization.mean, self.attack_normalization.std,
                )

            expected_batch_size = stop - start
            # The authoritative mask is per-batch, not the entire corpus.
            # Explicitly override a stale mask supplied through attack_kwargs.
            # Padding must be invisible to randomized starts and PGD like updates,
            # in addition to the existing final float64 physical restoration.
            valid_mask_batch = None if valid_mask is None else _slice_batch(valid_mask, start, stop)
            if valid_mask_batch is not None:
                _, expected_time_steps, _ = layout(attack_batch_input, feature_names)
                if tuple(valid_mask_batch.shape) != (expected_batch_size, expected_time_steps):
                    raise ValueError(
                        f"valid_mask batch shape {tuple(valid_mask_batch.shape)} != "
                        f"{(expected_batch_size, expected_time_steps)}"
                    )
                batch_kwargs["valid_mask"] = valid_mask_batch
            else:
                # run(valid_mask=None) means no padding mask; never leak one
                # inherited from a different experiment via attack_kwargs.
                batch_kwargs.pop("valid_mask", None)
            if isinstance(self.attack, FeatureSpaceAttack):
                batch_kwargs["_clean_authoritative_physical"] = clean_target_physical
                batch_kwargs["_discrete_diagnostic_logger"] = self.logger
                batch_kwargs["_discrete_diagnostic_config"] = getattr(self, "discrete_diagnostics", None)
            batch_result = self.attack.generate(
                attack_batch_input,
                _slice_batch(y, start, stop),
                **batch_kwargs,
            )
            del attack_batch_input
            if self.logger:
                self.logger.info(
                    "evaluation.attack_batch batch=%d:%d generated=%d success=%s queries=%s",
                    start, stop, expected_batch_size,
                    "yes" if batch_result.success_mask is not None else "no",
                    "yes" if batch_result.queries is not None else "no",
                )

            if not first_metadata:
                first_metadata = dict(batch_result.metadata)

            has_success_mask = batch_result.success_mask is not None
            has_queries = batch_result.queries is not None
            if success_available is None:
                success_available = has_success_mask
            elif has_success_mask != success_available:
                raise RuntimeError("Attack success_mask availability changed between batches")
            if queries_available is None:
                queries_available = has_queries
            elif has_queries != queries_available:
                raise RuntimeError("Attack queries availability changed between batches")

            if batch_result.success_mask is not None:
                success_batch = _as_numpy(batch_result.success_mask).reshape(-1)
                if len(success_batch) != expected_batch_size:
                    raise RuntimeError(
                        "Attack success_mask length changed or does not match its input batch: "
                        f"expected {expected_batch_size}, got {len(success_batch)}"
                    )
                success_batches.append(success_batch)
            if batch_result.queries is not None:
                query_batch = np.asarray(batch_result.queries).reshape(-1)
                if len(query_batch) != expected_batch_size:
                    raise RuntimeError(
                        "Attack queries length changed or does not match its input batch: "
                        f"expected {expected_batch_size}, got {len(query_batch)}"
                    )
                query_batches.append(query_batch)

            x_adv_source: np.ndarray | None = None
            processed: Any | None = None
            # For raw packet attacks, only the generated packet occupancy is
            # authoritative. The clean valid_mask is still passed to the attack
            # above, but must never determine which generated packets are scored.
            gate_valid_mask_batch = valid_mask_batch
            if batch_result.raw_adversarial is not None:
                if self.preprocessor is None:
                    raise RuntimeError("Problem-space evaluation requires a preprocessor")
                processed = self.preprocessor.transform(batch_result.raw_adversarial)
                batch_feature_names = tuple(processed.feature_names)
                # Physical packet features must be supplied by the extractor;
                # decoding its normalized outputs would falsely claim authority.
                raw_physical = getattr(processed, "physical_features", None)
                if raw_physical is None:
                    raise RuntimeError(
                        "Problem-space preprocessor must export float64 physical_features "
                        "before normalization for dual-authoritative evaluation"
                    )
                x_adv_physical = _as_physical_shape(
                    raw_physical, _slice_batch(x_target, start, stop), batch_feature_names
                )
                # Packet insertion can populate previously padded positions.
                # Reusing the clean mask would silently hide those packets from
                # the independent physical constraint oracle. Mask provenance
                # must come from re-extracting the generated raw traffic.
                generated_valid_mask = getattr(processed, "valid_mask", None)
                if generated_valid_mask is None:
                    if valid_mask_batch is not None and not bool(
                        np.all(_as_numpy(valid_mask_batch))
                    ):
                        raise RuntimeError(
                            "Problem-space preprocessing must export processed.valid_mask "
                            "when clean sequences contain padding; original padding "
                            "cannot describe injected packet occupancy"
                        )
                    # A fully populated clean sequence has no padded slots to
                    # recover. Without explicit output metadata, retain that
                    # all-valid mask (or None for unmasked tabular traffic).
                else:
                    generated_valid_mask = _as_numpy(generated_valid_mask)
                    _, generated_timesteps, _ = layout(x_adv_physical, batch_feature_names)
                    expected_mask_shape = (expected_batch_size, generated_timesteps)
                    if (generated_valid_mask.dtype != np.bool_
                            or generated_valid_mask.shape != expected_mask_shape):
                        raise ValueError(
                            "processed.valid_mask must be a boolean array of shape "
                            f"{expected_mask_shape}, got shape "
                            f"{generated_valid_mask.shape} and dtype "
                            f"{generated_valid_mask.dtype}"
                        )
                    gate_valid_mask_batch = generated_valid_mask
                    if self.logger is not None:
                        if valid_mask_batch is not None:
                            original_valid = _as_numpy(valid_mask_batch).astype(bool, copy=False)
                            self.logger.info(
                                "evaluation.problem_space_occupancy batch=%d:%d "
                                "injected_positions=%d removed_positions=%d",
                                start, stop,
                                int(np.count_nonzero(gate_valid_mask_batch & ~original_valid)),
                                int(np.count_nonzero(original_valid & ~gate_valid_mask_batch)),
                            )
                        else:
                            self.logger.info(
                                "evaluation.problem_space_occupancy batch=%d:%d valid_packets=%d",
                                start, stop, int(np.count_nonzero(gate_valid_mask_batch)),
                            )
            else:
                if batch_result.adversarial is None or batch_result.authoritative_physical is None:
                    raise RuntimeError("Feature attack must return BOTH dual representations")
                batch_feature_names = tuple(batch_result.feature_names or feature_names)
                x_adv_source = _as_numpy(batch_result.adversarial)
                if x_adv_source.dtype != np.float32:
                    raise TypeError("Attack model representation must be float32")
                x_adv_physical = _as_physical_shape(
                    batch_result.authoritative_physical, x_adv_source, batch_feature_names
                )
                if not x_adv_physical.flags.writeable or not x_adv_physical.flags.c_contiguous:
                    x_adv_physical = np.array(x_adv_physical, dtype=np.float64, copy=True, order="C")

            if tuple(batch_feature_names) != tuple(feature_names):
                raise ValueError("Physical and target feature schemas differ")
            if isinstance(self.attack, FeatureSpaceAttack):
                # Feature-space adversaries must not rewrite victim packets,
                # immutable metadata, or padding in the physical reference.
                _restore_target_physical_protected_inplace(
                    x_adv_physical, clean_target_physical, batch_feature_names,
                    immutable_features=getattr(self.attack, "immutable_features", ()),
                    valid_mask=valid_mask_batch,
                )
            elif isinstance(self.attack, ProblemSpaceAttack):
                # A raw-traffic attack may legitimately inject into previously
                # padded slots. The extracted physical packets are authoritative;
                # applying the clean mask here would erase real attack output.
                pass
            else:
                raise TypeError("Evaluation attack must be feature-space or problem-space")

            # Both paths encode directly from the resulting float64 physical
            # representation. The Gatekeeper independently checks this encoding.
            x_adv_batch = encode_normalized32(
                x_adv_physical, batch_feature_names,
                None if self.target_normalization is None else self.target_normalization.mean,
                None if self.target_normalization is None else self.target_normalization.std,
            )

            if len(x_adv_batch) != expected_batch_size:
                raise RuntimeError(
                    "Attack output length changed or does not match its input batch: "
                    f"expected {expected_batch_size}, got {len(x_adv_batch)}"
                )
            if first_feature_names is None:
                first_feature_names = batch_feature_names
                first_feature_shape = tuple(x_adv_batch.shape[1:])
                x_adv_target = np.empty(
                    (n_samples, *first_feature_shape),
                    dtype=np.float32,
                )
            else:
                if batch_feature_names != first_feature_names:
                    raise RuntimeError(
                        "Attack feature schema changed between batches: "
                        f"expected {first_feature_names}, got {batch_feature_names}"
                    )
                if tuple(x_adv_batch.shape[1:]) != first_feature_shape:
                    raise RuntimeError(
                        "Attack feature dimensions changed between batches: "
                        f"expected trailing shape {first_feature_shape}, "
                        f"got {tuple(x_adv_batch.shape[1:])}"
                    )
            assert x_adv_target is not None
            # Write each result straight into the final array. Do not retain a list
            # of all batches and then concatenate a second full-size corpus copy.
            x_adv_target[start:stop] = x_adv_batch
            if authoritative_physical_output is not None:
                # Optional disk-backed float64 sink for retaining the entire
                # authoritative physical corpus without using system RAM.
                authoritative_physical_output[start:stop] = x_adv_physical

            x_clean_batch = _slice_batch(x_target, start, stop)
            y_batch = y_np[start:stop]
            baseline_batch = baseline_pred[start:stop]
            # Raw packet mutations can insert/remove packets and shift all later
            # positions. A row-wise comparison against the clean physical tensor
            # therefore rejects valid raw traffic for the wrong reason. The
            # problem-space oracle omits ONLY this unaligned differential rule;
            # all independent physical constraints and byte-exact float32
            # encoding verification are still enforced.
            problem_space = isinstance(self.attack, ProblemSpaceAttack)
            if problem_space:
                # Without an alignment-aware independent validator, packet
                # ownership must be enforced by the raw mutator itself. Require
                # an explicit per-flow feasibility claim: silently interpreting
                # an absent mask as "all feasible" invalidates realizability.
                raw_intrinsic = batch_result.intrinsic_feasible_mask
                if raw_intrinsic is None:
                    raise RuntimeError(
                        "Problem-space attack must provide intrinsic_feasible_mask "
                        "from its packet-ownership/alignment validation; unaligned "
                        "directional immutability cannot be checked by feature rows"
                    )
                intrinsic = _as_numpy(raw_intrinsic)
                if intrinsic.dtype != np.bool_ or intrinsic.shape != (expected_batch_size,):
                    raise ValueError(
                        "Problem-space intrinsic_feasible_mask must be boolean "
                        f"with shape {(expected_batch_size,)}, got "
                        f"{intrinsic.dtype} and {intrinsic.shape}"
                    )
                if self._problem_space_gatekeeper is None:
                    raise RuntimeError("Problem-space oracle was not initialized")
                gatekeeper = self._problem_space_gatekeeper
            else:
                intrinsic = None
                gatekeeper = self.gatekeeper

            gate = gatekeeper.validate(
                x_adv_batch,
                first_feature_names,
                authoritative_physical=x_adv_physical,
                original_physical=None if problem_space else clean_target_physical,
                valid_mask=gate_valid_mask_batch,
            )
            accepted_batch = np.asarray(gate.accepted, dtype=bool).reshape(-1)
            if intrinsic is not None:
                # Intrinsic checks are necessary but not sufficient: a flow
                # must also satisfy every independently checkable physical rule.
                accepted_batch &= intrinsic
                n_intrinsic_violations = int(np.count_nonzero(~intrinsic))
                if n_intrinsic_violations:
                    violation_counts["problem_space_intrinsic_feasibility"] += n_intrinsic_violations
            if len(accepted_batch) != expected_batch_size:
                raise RuntimeError(
                    "Gatekeeper accepted mask length does not match the generated batch: "
                    f"expected {expected_batch_size}, got {len(accepted_batch)}"
                )
            accepted_batches.append(accepted_batch)
            for violation in gate.violations:
                rule = str(getattr(violation, "rule", "unknown"))
                indices = getattr(violation, "indices", ())
                try:
                    violation_counts[rule] += len(np.unique(np.asarray(indices, dtype=np.int64)))
                except (TypeError, ValueError):
                    continue

            if self.detector is not None:
                clean_scores = np.asarray(
                    self.detector.score(x_clean_batch, self.target_model),
                    dtype=np.float64,
                ).reshape(-1)
                adversarial_scores = np.asarray(
                    self.detector.score(x_adv_batch, self.target_model),
                    dtype=np.float64,
                ).reshape(-1)
                if len(clean_scores) != expected_batch_size or len(adversarial_scores) != expected_batch_size:
                    raise RuntimeError("Detector must return exactly one score per evaluation sample")
                clean_detector_batches.append(clean_scores)
                adversarial_detector_batches.append(adversarial_scores)

            score_mask_batch = (
                accepted_batch if self.strict_constraints else np.ones(expected_batch_size, dtype=bool)
            )
            scored_batches.append(score_mask_batch)
            if queries_available:
                query_score_batches.append(query_batch[score_mask_batch])
            if np.any(score_mask_batch):
                adversarial_predictions_batch = _batched_predict(
                    self.target_model,
                    x_adv_batch[score_mask_batch],
                    self.batch_size,
                )[0]
                adversarial_prediction_batches.append(adversarial_predictions_batch)
                y_score_batches.append(y_batch[score_mask_batch])
                baseline_score_batches.append(baseline_batch[score_mask_batch])

            del batch_result, x_adv_batch, x_adv_physical, clean_target_physical
            del x_clean_batch, y_batch, baseline_batch
            if x_adv_source is not None:
                del x_adv_source
            if processed is not None:
                del processed

        if x_adv_target is None or first_feature_names is None or first_feature_shape is None:
            raise ValueError("Evaluation requires at least one generated sample")

        accepted = np.concatenate(accepted_batches)
        score_mask = np.concatenate(scored_batches)
        adversarial_pred_full = np.full(n_samples, -1, dtype=np.int64)
        scored_predictions = (
            np.concatenate(adversarial_prediction_batches)
            if adversarial_prediction_batches
            else np.empty(0, dtype=np.int64)
        )
        adversarial_pred_full[score_mask] = scored_predictions
        y_score = np.concatenate(y_score_batches) if y_score_batches else np.empty(0, dtype=np.int64)
        base_score = (
            np.concatenate(baseline_score_batches)
            if baseline_score_batches
            else np.empty(0, dtype=np.int64)
        )
        adv_score = scored_predictions
        queries = np.concatenate(query_score_batches) if query_score_batches else None
        detector_clean_scores = (
            np.concatenate(clean_detector_batches) if clean_detector_batches else None
        )
        detector_adversarial_scores = (
            np.concatenate(adversarial_detector_batches) if adversarial_detector_batches else None
        )
        actual_feature_names = first_feature_names
        attack_result = AttackResult(
            adversarial=x_adv_target,
            success_mask=np.concatenate(success_batches) if success_available else None,
            queries=np.concatenate(query_batches) if queries_available else None,
            feature_names=actual_feature_names,
            metadata={**first_metadata, "physical_authority": (
                "retained_in_caller_sink" if authoritative_physical_output is not None
                else "batch_local_not_retained"
            )},
            authoritative_physical=authoritative_physical_output,
        )

        if len(x_adv_target) != n_samples or len(accepted) != n_samples:
            raise ValueError(f"Attack produced an invalid sample count for a batch of {n_samples} labels")

        desired_label = self.benign_label if self.target_label is None else self.target_label
        successful = (base_score == y_score) & (y_score != desired_label) & (adv_score == desired_label)
        attack_to_attack = (
            (base_score == y_score)
            & (y_score != self.benign_label)
            & (adv_score != self.benign_label)
            & (adv_score != desired_label)
        )
        successful_full = np.zeros(n_samples, dtype=bool)
        successful_full[score_mask] = successful

        metric_labels = self.class_labels
        if metric_labels is None:
            observed = (
                np.unique(np.concatenate([y_score, base_score, adv_score]))
                if len(y_score)
                else np.asarray([], dtype=np.int64)
            )
            metric_labels = tuple(int(value) for value in observed)

        # Avoid full-size x_score/x_adv_score Boolean-indexed copies. Distances
        # are calculated directly from the two full arrays in bounded chunks.
        metrics = evaluate_predictions(
            y_score,
            adv_score,
            labels=metric_labels,
            class_names=self.class_names,
            queries=queries,
            success_mask=successful,
            query_budget=(
                int(attack_result.metadata["query_budget"])
                if "query_budget" in attack_result.metadata
                else None
            ),
            accepted_mask=accepted[score_mask] if self.strict_constraints else accepted,
            baseline_pred=base_score,
            benign_label=self.benign_label,
            target_label=self.target_label,
            c_fp=self.c_fp,
            c_fn=self.c_fn,
        )
        metrics.update(
            _chunked_perturbation_distances(
                _as_numpy(x_target),
                x_adv_target,
                successful_full,
                batch_size=self.batch_size,
            )
        )

        metrics["strict_constraints"] = float(self.strict_constraints)
        metrics["strict_multiclass"] = float(self.strict_multiclass)
        metrics["attack_to_attack_failure_count"] = float(np.sum(attack_to_attack))
        metrics["attack_to_attack_failure_rate"] = float(
            np.mean(attack_to_attack) if len(attack_to_attack) else 0.0
        )
        metrics["scored_sample_count"] = float(score_mask.sum())
        metrics["rejected_sample_count"] = float((~accepted).sum())
        metrics["realizability_rate"] = float(accepted.mean()) if len(accepted) else 1.0
        metrics["inference_latency_ms_per_sample"] = float(baseline_latency_ms)
        metrics["throughput_flows_per_second"] = float(baseline_throughput)
        violation_metrics = {
            f"violation_{rule}": float(count)
            for rule, count in sorted(violation_counts.items())
        }
        metrics.update(violation_metrics)
        metrics["gatekeeper_violation_total"] = float(sum(violation_metrics.values()))
        metrics["successful_sample_count"] = float(np.sum(successful))
        if self.detector is not None:
            assert detector_clean_scores is not None and detector_adversarial_scores is not None
            metrics["detector_enabled"] = 1.0
            metrics["detection_auroc"] = detection_auroc(
                detector_clean_scores, detector_adversarial_scores
            )
            metrics["detector_clean_score_mean"] = float(np.mean(detector_clean_scores))
            metrics["detector_adversarial_score_mean"] = float(np.mean(detector_adversarial_scores))
            metrics["detection_auroc_realizable"] = detection_auroc(
                detector_clean_scores, detector_adversarial_scores[accepted]
            )
            metrics["detector_adversarial_score_mean_realizable"] = (
                float(np.mean(detector_adversarial_scores[accepted])) if np.any(accepted) else float("nan")
            )
        else:
            metrics["detector_enabled"] = 0.0
        if self.attack_model is not self.target_model:
            metrics["transfer_attack"] = 1.0
        if self.logger:
            self.logger.info(
                "evaluation.done samples=%d scored=%d accepted=%d successful=%d asr=%.2f%% "
                "realizability=%.2f%% rejected=%d",
                n_samples, int(metrics["scored_sample_count"]),
                int(np.sum(accepted)), int(metrics["successful_sample_count"]),
                100.0 * metrics.get("asr", 0.0),
                100.0 * metrics["realizability_rate"],
                int(metrics["rejected_sample_count"]),
            )

        return EvaluationOutput(
            metrics=metrics,
            baseline_predictions=np.asarray(baseline_pred, dtype=np.int64),
            adversarial_predictions=adversarial_pred_full,
            accepted_mask=accepted,
            scored_mask=score_mask,
            attack_result=attack_result,
            detector_clean_scores=detector_clean_scores,
            detector_adversarial_scores=detector_adversarial_scores,
        )

