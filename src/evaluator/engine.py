from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import inspect
from dataclasses import dataclass
import warnings
import logging
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from attacks.base import AttackResult, BaseAttack, FeatureSpaceAttack, ProblemSpaceAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from defenses.base import BaseDetector
from models.wrapper import ModelWrapper, PyTorchModelWrapper
from orchestrator.progress import ProgressCallback
from preprocessing.base import BasePreprocessor
from core.ownership import verify_mapping_v1_victim_packets, mapping_v1_packet_edit_opcodes
from core.query_meter import ModelQueryCounts, ModelQueryMeter
from core.provenance import frozen_model_state_sha256
from core.dual_representation import (
    _as_physical_shape,
    _restore_target_physical_protected_inplace,
    decode_physical64,
    encode_normalized32,
    layout,
)
from .metrics import detection_auroc, evaluate_predictions, classification_metrics, classification_support, PhysicalDistanceAccumulator


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
    defended_predictions: np.ndarray | None = None


@dataclass(frozen=True)
class BatchContext:
    """Synchronized clean batch views used throughout one evaluation iteration."""

    x_batch: Any
    y_batch: np.ndarray
    clean_physical_batch: np.ndarray
    valid_mask_batch: np.ndarray | torch.Tensor | None


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sync_if_cuda(model: ModelWrapper) -> None:
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _timed_predict(model: ModelWrapper, x: Any, valid_mask: Any = None) -> tuple[np.ndarray, float]:
    """Measure full target prediction wall time, including device synchronization."""
    _sync_if_cuda(model)
    start = perf_counter()
    predictions = np.asarray(
        model.predict(x, valid_mask=valid_mask) if isinstance(model, PyTorchModelWrapper)
        else model.predict(x), dtype=np.int64
    )
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
    valid_mask: Any = None,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, float]:
    """Predict in bounded batches and return predictions in input order."""
    n_samples = len(x)
    predictions: list[np.ndarray] = []
    elapsed = 0.0
    for start in range(0, n_samples, batch_size):
        batch_predictions, batch_elapsed = _timed_predict(
            model,
            _slice_batch(x, start, min(start + batch_size, n_samples)),
            valid_mask=None if valid_mask is None else _slice_batch(valid_mask, start, min(start + batch_size, n_samples)),
        )
        predictions.append(batch_predictions)
        elapsed += batch_elapsed
        if progress is not None:
            progress('clean_inference', completed=min(start + batch_size, n_samples),
                     total=n_samples, unit='flows')
    if not predictions:
        return np.empty(0, dtype=np.int64), np.finfo(np.float64).eps
    return np.concatenate(predictions), max(elapsed, np.finfo(np.float64).eps)


def _batched_detector_score(
    detector: BaseDetector,
    model: ModelWrapper,
    x: Any,
    batch_size: int,
    valid_mask: Any = None,
) -> np.ndarray:
    """Score inputs through the detector in bounded batches to prevent OOM."""
    n_samples = len(x)
    scores: list[np.ndarray] = []
    for start in range(0, n_samples, batch_size):
        kwargs = {}
        if "valid_mask" in inspect.signature(detector.score).parameters:
            kwargs["valid_mask"] = (
                None if valid_mask is None else _slice_batch(valid_mask, start, min(start + batch_size, n_samples))
            )
        batch_scores = detector.score(
            _slice_batch(x, start, min(start + batch_size, n_samples)), model, **kwargs,
        )
        scores.append(np.asarray(batch_scores, dtype=np.float64).reshape(-1))
    if not scores:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(scores)








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
        surrogate_input_shape: Sequence[int] | None = None,
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
        # Enforce the benchmark-wide target-isolation policy even when the
        # engine is called directly, bypassing workflow configuration.
        if (getattr(attack, 'requires_gradients', False) or (isinstance(attack, FeatureSpaceAttack) and getattr(attack, 'query_based', False))) and attack_model is None:
            raise ValueError('Query- and gradient-based attacks require a separately authorized surrogate')
        if attack_model is target_model:
            raise ValueError('Attack generation may never bind the target model')
        if attack_model is not None:
            # Independent models may have exactly the same architecture and
            # hyperparameters. Only *shared live storage* is forbidden: it
            # would expose target-owned weights to attacker-side autograd.
            target_nn = getattr(target_model, 'model', None)
            surrogate_nn = getattr(attack_model, 'model', None)
            if isinstance(target_nn, torch.nn.Module) and isinstance(surrogate_nn, torch.nn.Module):
                def tensor_storages(module: torch.nn.Module) -> set[tuple[str, int]]:
                    return {
                        (str(t.device), int(t.untyped_storage().data_ptr()))
                        for t in (*module.parameters(), *module.buffers())
                        if t.numel() and t.untyped_storage().data_ptr()
                    }

                if tensor_storages(target_nn) & tensor_storages(surrogate_nn):
                    raise ValueError('Attacker and target share underlying PyTorch parameter or buffer storage')
            for name in ('model', 'estimator'):
                underlying = getattr(target_model, name, None)
                if underlying is not None and underlying is getattr(attack_model, name, None):
                    raise ValueError('Attacker and target share underlying model storage')
            # A distinct wrapper/checkpoint path does not confer independence:
            # reject byte-identical learned states regardless of path or seed.
            if (isinstance(getattr(target_model, 'model', None), torch.nn.Module) or
                    getattr(target_model, 'estimator', None) is not None) and (
                    isinstance(getattr(attack_model, 'model', None), torch.nn.Module) or
                    getattr(attack_model, 'estimator', None) is not None):
                if frozen_model_state_sha256(target_model) == frozen_model_state_sha256(attack_model):
                    raise ValueError('Attack surrogate has exactly the frozen target model state')
        if attack_model is None and getattr(attack, 'model', None) is not None:
            raise ValueError('Model-independent attack must not arrive pre-bound to a model')
        self.attack_model = attack_model
        self.attack = attack.bind_model(attack_model) if attack_model is not None else attack
        self.surrogate_input_shape = tuple(int(dim) for dim in surrogate_input_shape) if surrogate_input_shape else None
        self.gatekeeper = gatekeeper
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
        if not self.strict_constraints:
            warnings.warn(
                "strict_constraints=false enables NON-STRICT DIAGNOSTIC scoring of "
                "physically rejected adversarial candidates. Its ungated outputs "
                "are not valid SOTA feasible-attack evidence.",
                UserWarning, stacklevel=2,
            )
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
        progress: ProgressCallback | None = None,
    ) -> EvaluationOutput:
        """Generate, validate, score, and summarize one evaluation batch.

        ``attack_input_is_target_normalized`` permits transfer attacks to convert
        each target-normalized batch to surrogate-normalized space on demand,
        without retaining a second normalized copy of the entire dataset.
        """
        y_np = _as_numpy(y).astype(np.int64, copy=False)
        x_target = x
        if valid_mask is not None:
            valid_mask = _as_numpy(valid_mask)
            if valid_mask.dtype != np.bool_ or valid_mask.ndim != 2 or valid_mask.shape[0] != len(y_np):
                raise ValueError("valid_mask must be a boolean (N,T) packet occupancy array")
        if valid_mask is not None:
            _, clean_t, _ = layout(x_target, feature_names)
            if valid_mask.shape != (len(y_np), clean_t):
                raise ValueError("Clean packet occupancy has the wrong (N,T) shape")
            lengths = valid_mask.sum(axis=1)
            if not np.all(valid_mask == (np.arange(clean_t)[None, :] < lengths[:, None])):
                raise ValueError("Clean valid_mask must contain a contiguous real-packet prefix")
        if not self.strict_multiclass:
            raise ValueError("strict_multiclass=false is unsupported: benign-target evasion is always strict")
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

        # Fail preflight before querying even the clean evaluation target.
        final_attack_budget = (
            self.attack.final_budget_contract()
            if isinstance(self.attack, FeatureSpaceAttack) else None
        )
        if final_attack_budget is not None:
            if self.attack_normalization is None:
                provided = dict(attack_kwargs or {})
                if provided.get("normalization_mean") is None or provided.get("normalization_std") is None:
                    raise ValueError("A bounded attack must declare its authorized normalization")

        if progress is not None:
            progress('clean_inference', completed=0, total=len(y_np), unit='flows')
        with ModelQueryMeter(self.target_model) as clean_meter:
            baseline_pred, baseline_elapsed = _batched_predict(
                self.target_model, x_target, self.batch_size, valid_mask=valid_mask,
                progress=progress,
            )
        target_clean_counts = clean_meter.counts
        target_adv_counts = ModelQueryCounts()
        target_detector_clean_counts = ModelQueryCounts()
        target_detector_adv_counts = ModelQueryCounts()
        surrogate_counts = ModelQueryCounts()
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
            # Attack kwargs cannot override checkpoint-authorized surrogate normalization.
            for name, correct in (("normalization_mean", self.attack_normalization.mean),
                                  ("normalization_std", self.attack_normalization.std)):
                if name in effective_attack_kwargs and not np.array_equal(
                    np.asarray(effective_attack_kwargs[name]), np.asarray(correct)
                ):
                    raise ValueError(f"Attack-supplied {name} differs from authorized surrogate normalization")
                effective_attack_kwargs[name] = correct

        success_batches: list[np.ndarray] = []
        query_batches: list[np.ndarray] = []
        distance_accumulator: PhysicalDistanceAccumulator | None = (
            PhysicalDistanceAccumulator(feature_names)
            if isinstance(self.attack, FeatureSpaceAttack) else None
        )
        accepted_batches: list[np.ndarray] = []
        scored_batches: list[np.ndarray] = []
        adversarial_prediction_batches: list[np.ndarray] = []
        clean_detector_batches: list[np.ndarray] = []
        adversarial_detector_batches: list[np.ndarray] = []
        y_score_batches: list[np.ndarray] = []
        baseline_score_batches: list[np.ndarray] = []
        query_score_batches: list[np.ndarray] = []
        violation_counts: Counter[str] = Counter()
        verified_raw_edit_totals = np.zeros(3, dtype=np.int64)
        verified_raw_edit_flow_count = 0
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

        if progress is not None:
            progress('generate_and_gate', completed=0, total=n_samples, unit='flows')
        for start in range(0, n_samples, self.batch_size):
            stop = min(start + self.batch_size, n_samples)
            batch_kwargs = dict(effective_attack_kwargs)
            if batch_kwargs.get("seed") is not None:
                batch_kwargs["seed"] = int(batch_kwargs["seed"]) + start

            x_batch = _slice_batch(x_target, start, stop)
            y_batch = y_np[start:stop]
            clean_target_physical = _as_physical_shape(
                _slice_physical_source(clean_physical, start, stop),
                x_batch,
                feature_names,
            )
            valid_mask_batch = None if valid_mask is None else _slice_batch(valid_mask, start, stop)
            batch_context = BatchContext(
                x_batch=x_batch,
                y_batch=y_batch,
                clean_physical_batch=clean_target_physical,
                valid_mask_batch=valid_mask_batch,
            )
            
            physical_feature_attack = (
                isinstance(self.attack, FeatureSpaceAttack)
                and bool(getattr(
                    self.attack,
                    "operates_on_physical_features",
                    False,
                ))
            )

            if physical_feature_attack:
                # Give the physical generator its own float64 batch workspace.
                # In-place generator mutations must never corrupt the pristine baseline
                # or a shared/memory-mapped physical dataset.
                attack_batch_input = np.array(
                    batch_context.clean_physical_batch,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
            else:
                # Existing normalized-float32 behavior remains unchanged.
                # ProblemSpaceAttack continues through its existing separate path.
                attack_batch_input = _slice_batch(
                    target_input,
                    start,
                    stop,
                )

                if attack_input_is_target_normalized:
                    # Transfer attacks may require surrogate-specific normalization.
                    # Physical generators do not need this intermediate conversion.
                    attack_batch_input = encode_normalized32(
                        batch_context.clean_physical_batch,
                        feature_names,
                        self.attack_normalization.mean,
                        self.attack_normalization.std,
                    )

            # Bijective model-topology adaptation is a view, not re-extraction
            # or a float32->float64 inverse. Attack results are reshaped back
            # to the fixed target topology before final physical validation.
            if (self.surrogate_input_shape is not None
                    and isinstance(self.attack, FeatureSpaceAttack)):
                if int(np.prod(self.surrogate_input_shape)) != int(np.prod(x_batch.shape[1:])):
                    raise ValueError('Target and surrogate input widths do not match')
                attack_batch_input = attack_batch_input.reshape(
                    (stop - start, *self.surrogate_input_shape)
                )

            expected_batch_size = stop - start
            # The authoritative mask is per-batch, not the entire corpus.
            # Explicitly override a stale mask supplied through attack_kwargs.
            # Padding must be invisible to randomized starts and PGD like updates,
            # in addition to the existing final float64 physical restoration.
            if batch_context.valid_mask_batch is not None:
                _, expected_time_steps, _ = layout(
                    x_batch if isinstance(self.attack, ProblemSpaceAttack) else attack_batch_input,
                    feature_names,
                )
                if tuple(batch_context.valid_mask_batch.shape) != (expected_batch_size, expected_time_steps):
                    raise ValueError(
                        f"valid_mask batch shape {tuple(batch_context.valid_mask_batch.shape)} != "
                        f"{(expected_batch_size, expected_time_steps)}"
                    )
                batch_kwargs["valid_mask"] = batch_context.valid_mask_batch
            else:
                # run(valid_mask=None) means no padding mask; never leak one
                # inherited from a different experiment via attack_kwargs.
                batch_kwargs.pop("valid_mask", None)
            if isinstance(self.attack, FeatureSpaceAttack):
                batch_kwargs["_clean_authoritative_physical"] = batch_context.clean_physical_batch.reshape(
                    attack_batch_input.shape
                )
                batch_kwargs["_discrete_diagnostic_logger"] = self.logger
                batch_kwargs["_discrete_diagnostic_config"] = getattr(self, "discrete_diagnostics", None)
            attack_batch_raw = attack_batch_input if isinstance(self.attack, ProblemSpaceAttack) else None
            # Surrogate-only scoped mask: all optimizer forward paths, including
            # direct attack.model.forward_differentiable(), see correct occupancy.
            mask_context = (
                self.attack_model.attack_mask_context(batch_context.valid_mask_batch)
                if isinstance(self.attack_model, PyTorchModelWrapper)
                else nullcontext()
            )
            # Independent metering is scoped strictly to generation. The
            # target guard is active even for an attack retaining a hidden
            # reference to the target; the evaluator does not supply one.
            with ModelQueryMeter(self.target_model, forbidden=True):
                with ModelQueryMeter(self.attack_model) as attack_meter:
                    with mask_context:
                        batch_result = self.attack.generate(
                            attack_batch_input, batch_context.y_batch, **batch_kwargs,
                        )
            surrogate_counts.add(attack_meter.counts)
            # Per-sample plugin counts are *inference record evaluations*.
            # Gradient forwards have their own category and are not disguised
            # as black-box score queries. Validate query-based plugin claims
            # against independently measured inference counts.
            if (self.attack_model is not None and
                    (self.attack.query_based or getattr(self.attack, 'attack_name', '') == 'tdtce')):
                claimed = batch_result.queries
                if claimed is None or np.asarray(claimed).ndim != 1 or (
                    int(np.asarray(claimed).sum()) != attack_meter.counts.inference_records
                ):
                    raise ValueError('Attack query telemetry contradicts independently metered surrogate inference')
            del attack_batch_input
            if self.logger:
                self.logger.debug(
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
                if query_batch.dtype.kind not in "iu" or np.any(query_batch < 0):
                    raise ValueError("Attack queries must be nonnegative integers for every attempt")
                query_batches.append(query_batch.astype(np.int64, copy=False))

            x_adv_source: np.ndarray | None = None
            processed: Any | None = None
            # For raw packet attacks, only the generated packet occupancy is
            # authoritative. The clean valid_mask is still passed to the attack
            # above, but must never determine which generated packets are scored.
            gate_valid_mask_batch = batch_context.valid_mask_batch
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
                    raw_physical, batch_context.x_batch, batch_feature_names
                )
                # Packet insertion can populate previously padded positions.
                # Reusing the clean mask would silently hide those packets from
                # the independent physical constraint oracle. Mask provenance
                # must come from re-extracting the generated raw traffic.
                generated_valid_mask = getattr(processed, "valid_mask", None)
                if generated_valid_mask is None:
                    if batch_context.valid_mask_batch is not None and not bool(
                        np.all(_as_numpy(batch_context.valid_mask_batch))
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
                        if batch_context.valid_mask_batch is not None:
                            original_valid = _as_numpy(batch_context.valid_mask_batch).astype(bool, copy=False)
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
                    batch_result.authoritative_physical, x_batch, batch_feature_names
                )
                if not x_adv_physical.flags.writeable or not x_adv_physical.flags.c_contiguous:
                    x_adv_physical = np.array(x_adv_physical, dtype=np.float64, copy=True, order="C")

            if tuple(batch_feature_names) != tuple(feature_names):
                raise ValueError("Physical and target feature schemas differ")
            if isinstance(self.attack, FeatureSpaceAttack):
                # Feature-space adversaries must not rewrite victim packets,
                # immutable metadata, or padding in the physical reference.
                _restore_target_physical_protected_inplace(
                    x_adv_physical, batch_context.clean_physical_batch, batch_feature_names,
                    immutable_features=getattr(self.attack, "immutable_features", ()),
                    valid_mask=batch_context.valid_mask_batch,
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

            x_clean_batch = batch_context.x_batch
            y_batch = batch_context.y_batch
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
                # For mapping-v1 inputs, independently verify victim packet
                # identity/order at the raw level, not by shifted feature rows.
                mapping_ownership = verify_mapping_v1_victim_packets(
                    attack_batch_raw, batch_result.raw_adversarial
                )
                if mapping_ownership is not None:
                    intrinsic = intrinsic & mapping_ownership
                    violation_counts['problem_space_victim_ownership'] += int(
                        np.count_nonzero(~mapping_ownership)
                    )

            else:
                raw_intrinsic = batch_result.intrinsic_feasible_mask
                intrinsic = None
                if raw_intrinsic is not None:
                    intrinsic = _as_numpy(raw_intrinsic)
                    if intrinsic.dtype != np.bool_ or intrinsic.shape != (expected_batch_size,):
                        raise ValueError(
                            "Feature-space intrinsic_feasible_mask must be boolean "
                            f"with shape {(expected_batch_size,)}"
                        )

            gate = self.gatekeeper.validate(
                x_adv_batch,
                first_feature_names,
                authoritative_physical=x_adv_physical,
                original_physical=None if problem_space else batch_context.clean_physical_batch,
                valid_mask=gate_valid_mask_batch,
                is_problem_space=problem_space,
                attack_budget=final_attack_budget,
                budget_normalization_mean=batch_kwargs.get("normalization_mean"),
                budget_normalization_std=batch_kwargs.get("normalization_std"),
            )
            accepted_batch = np.asarray(gate.accepted, dtype=bool).reshape(-1)
            if intrinsic is not None:
                # Intrinsic checks are necessary but not sufficient: a flow
                # must also satisfy every independently checkable physical rule.
                accepted_batch &= intrinsic
                n_intrinsic_violations = int(np.count_nonzero(~intrinsic))
                if n_intrinsic_violations:
                    violation_counts[
                        "problem_space_intrinsic_feasibility" if problem_space
                        else "feature_space_intrinsic_feasibility"
                    ] += n_intrinsic_violations
            if len(accepted_batch) != expected_batch_size:
                raise RuntimeError(
                    "Gatekeeper accepted mask length does not match the generated batch: "
                    f"expected {expected_batch_size}, got {len(accepted_batch)}"
                )
            accepted_batches.append(accepted_batch)
            if problem_space and mapping_ownership is not None:
                raw_edits = mapping_v1_packet_edit_opcodes(attack_batch_raw, batch_result.raw_adversarial)
                if raw_edits is not None:
                    # Only independently verified, gatekeeper-accepted flows
                    # contribute to feasible mapping-v1 opcode counts.
                    verified_raw_edit_totals += raw_edits[accepted_batch].sum(axis=0)
                    verified_raw_edit_flow_count += int(accepted_batch.sum())
            for violation in gate.violations:
                rule = str(getattr(violation, "rule", "unknown"))
                indices = getattr(violation, "indices", ())
                try:
                    violation_counts[rule] += len(np.unique(np.asarray(indices, dtype=np.int64)))
                except (TypeError, ValueError):
                    continue

            # Gatekeeper admission comes BEFORE target-dependent adversarial
            # detector scoring. Rejected samples retain NaN score placeholders
            # to keep all reporting arrays aligned with the original sample IDs.
            score_mask_batch = (
                accepted_batch if self.strict_constraints else np.ones(expected_batch_size, dtype=bool)
            )
            if self.detector is not None:
                with ModelQueryMeter(self.target_model) as detector_clean_meter:
                    clean_scores = _batched_detector_score(
                        self.detector, self.target_model, x_clean_batch, self.batch_size,
                        valid_mask=batch_context.valid_mask_batch,
                    )
                target_detector_clean_counts.add(detector_clean_meter.counts)
                adversarial_scores = np.full(expected_batch_size, np.nan, dtype=np.float64)
                if np.any(score_mask_batch):
                    with ModelQueryMeter(self.target_model) as detector_adv_meter:
                        adversarial_scores[score_mask_batch] = _batched_detector_score(
                            self.detector, self.target_model,
                            x_adv_batch[score_mask_batch], self.batch_size,
                            valid_mask=(None if gate_valid_mask_batch is None
                                        else gate_valid_mask_batch[score_mask_batch]),
                        )
                    target_detector_adv_counts.add(detector_adv_meter.counts)
                if len(clean_scores) != expected_batch_size:
                    raise RuntimeError('Detector must score every clean evaluation example')
                clean_detector_batches.append(clean_scores)
                adversarial_detector_batches.append(adversarial_scores)
            scored_batches.append(score_mask_batch)
            if queries_available:
                query_score_batches.append(query_batch[score_mask_batch])
            if np.any(score_mask_batch):
                with ModelQueryMeter(self.target_model) as target_adv_meter:
                    adversarial_predictions_batch = _batched_predict(
                        self.target_model,
                        x_adv_batch[score_mask_batch],
                        self.batch_size,
                        valid_mask=(None if gate_valid_mask_batch is None
                                    else gate_valid_mask_batch[score_mask_batch]),
                    )[0]
                target_adv_counts.add(target_adv_meter.counts)
                adversarial_prediction_batches.append(adversarial_predictions_batch)
                y_score_batches.append(y_batch[score_mask_batch])
                baseline_score_batches.append(baseline_batch[score_mask_batch])

            if distance_accumulator is not None:
                # A rejected sample has no target prediction. It still counts
                # as an eligible generation attempt, but never as a success.
                batch_adv_pred = np.full(expected_batch_size, -1, dtype=np.int64)
                if np.any(score_mask_batch):
                    batch_adv_pred[score_mask_batch] = adversarial_predictions_batch
                desired = self.benign_label if self.target_label is None else self.target_label
                batch_eligible = (baseline_batch == y_batch) & (y_batch != desired)
                batch_success = batch_eligible & accepted_batch & (batch_adv_pred == desired)
                distance_accumulator.update(
                    batch_context.clean_physical_batch, x_adv_physical,
                    batch_context.valid_mask_batch,
                    eligible=batch_eligible, accepted=accepted_batch,
                    successful=batch_success,
                )

            del batch_result, x_adv_batch, x_adv_physical, batch_context, attack_batch_raw
            del x_clean_batch, y_batch, baseline_batch
            if x_adv_source is not None:
                del x_adv_source
            if processed is not None:
                del processed
            # Only count fully generated, independently gated, permissibly scored batches.
            if progress is not None:
                progress('generate_and_gate', completed=stop, total=n_samples, unit='flows')

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
            metadata={**first_metadata, "final_budget": (
                None if final_attack_budget is None else {
                    "epsilon": final_attack_budget.epsilon,
                    "norm": final_attack_budget.norm,
                    "domain": final_attack_budget.domain,
                    "normalization_owner": "authorized_attack_or_surrogate",
                }
            ), "physical_authority": (
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
        eligible_full = (baseline_pred == y_np) & (y_np != desired_label)
        # Independent physical admission is necessary for benchmark success.
        # Rejected attempts remain in the denominator as unsuccessful attempts.
        successful_full = eligible_full & accepted & (adversarial_pred_full == desired_label)

        metric_labels = self.class_labels
        if metric_labels is None:
            # Direct engine users without a declared schema still require a
            # valid universe when every candidate is rejected. Use the clean
            # full-test population, never only the scored accepted subset.
            observed = np.unique(np.concatenate((y_np, baseline_pred, adv_score)))
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
        if distance_accumulator is not None:
            metrics.update(distance_accumulator.summary())
            metrics["distance_domain_physical_aligned"] = 1.0
        else:
            # Variable-length raw topology is not aligned edit cost. Do not
            # silently label normalized coordinate distances as physical edits.
            metrics["distance_domain_physical_aligned"] = 0.0
            if verified_raw_edit_flow_count:
                metrics['raw_mapping_v1_verified_accepted_flow_count'] = float(verified_raw_edit_flow_count)
                for name, value in zip(('inserted', 'deleted', 'replaced'), verified_raw_edit_totals):
                    metrics[f'raw_mapping_v1_sequence_opcode_{name}_packet_count'] = float(value)
                    metrics[f'raw_mapping_v1_sequence_opcode_{name}_per_accepted_flow'] = float(value/verified_raw_edit_flow_count)
            else:
                metrics['raw_mapping_v1_verified_accepted_flow_count'] = 0.0
            for key in ("mean_l0", "mean_l2", "mean_linf"):
                metrics[key] = float("nan")


        # Gated full-test defended outcomes use *clean* target predictions for
        # rejected candidates. They are NOT adversarial predictions.
        defended_predictions = baseline_pred.copy()
        defended_predictions[accepted & score_mask] = adversarial_pred_full[accepted & score_mask]
        groups = {
            "full_test_clean": (y_np, baseline_pred),
            "full_test_constraint_gated_defended": (y_np, defended_predictions),
            "accepted_subset_clean": (y_np[accepted], baseline_pred[accepted]),
            "accepted_subset_adversarial": (
                y_np[accepted], adversarial_pred_full[accepted]
            ),
        }
        for prefix, (true_subset, pred_subset) in groups.items():
            if len(true_subset) and np.any(pred_subset < 0):
                raise RuntimeError("Accepted population lacks target predictions")
            summary = classification_metrics(true_subset, pred_subset, metric_labels).as_dict()
            for key, value in summary.items():
                metrics[f"{prefix}_{key}"] = float(value)
            # Persist all schema class counts for this exact metric population,
            # including an empty accepted-only subset. These are descriptive
            # counts, never alternative ASR denominators.
            support = classification_support(true_subset, pred_subset, metric_labels)
            metrics[f"{prefix}_schema_complete"] = float(support['schema_complete'])
            metrics[f"{prefix}_predicted_only_class_count"] = float(len(support['predicted_only_classes']))
            for label, count in support['ground_truth_support'].items():
                metrics[f"{prefix}_support_class_{label}"] = float(count)
            for label, count in support['predicted_counts'].items():
                metrics[f"{prefix}_predicted_count_class_{label}"] = float(count)
        metrics["full_test_sample_count"] = float(n_samples)
        metrics["accepted_subset_sample_count"] = float(accepted.sum())
        # Independent oracle metering is authoritative for model operations;
        # attack-provided per-record counts are separately validated where
        # exact attribution is available (Square/A2PM/TDTCE).
        metrics.update(surrogate_counts.metrics('surrogate_generation'))
        metrics.update(target_clean_counts.metrics('target_clean'))
        metrics.update(target_adv_counts.metrics('target_adversarial_scored'))
        metrics.update(target_detector_clean_counts.metrics('target_detector_clean'))
        metrics.update(target_detector_adv_counts.metrics('target_detector_adversarial_scored'))
        metrics['query_accounting_contract_version'] = 2.0
        # The authoritative scalar query denominator is ALL generated records.
        # A batch of B forward-evaluated records costs B example queries, while
        # one call remains one invocation. Gradient-bearing forwards are never
        # silently added to black-box score queries.
        metrics['all_attempts_total_queries'] = float(surrogate_counts.inference_records)
        metrics['all_attempts_mean_queries'] = float(
            surrogate_counts.inference_records / n_samples if n_samples else float('nan')
        )
        metrics['query_telemetry_available'] = float(queries_available is True)
        metrics['per_sample_inference_query_attribution_available'] = float(queries_available is True)
        if queries_available:
            all_queries = np.asarray(attack_result.queries, dtype=np.int64)
            metrics['all_attempts_plugin_reported_inference_records'] = float(all_queries.sum(dtype=np.int64))
        else:
            metrics['all_attempts_plugin_reported_inference_records'] = float('nan')
        # The lower-level evaluator's legacy fields describe *plugin* arrays,
        # not independently metered model operations; keep their population
        # explicit instead of presenting them as authoritative generic queries.
        for old, new in (
            ('mean_queries', 'scored_subset_plugin_reported_mean_inference_queries'),
            ('mean_queries_to_evasion', 'scored_subset_plugin_reported_mean_inference_queries_to_evasion'),
        ):
            if old in metrics:
                metrics[new] = metrics.pop(old)
        metrics["final_hard_budget_checked"] = float(final_attack_budget is not None)
        if final_attack_budget is not None:
            metrics["final_hard_budget_epsilon"] = float(final_attack_budget.epsilon)
            metrics["final_hard_budget_failure_count"] = float(
                violation_counts.get(
                    f"attack_budget_{final_attack_budget.norm}_{final_attack_budget.domain}", 0
                )
            )
        metrics["strict_constraints"] = float(self.strict_constraints)
        metrics["strict_feasible_benchmark"] = float(self.strict_constraints)
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
        eligible_count = int(np.count_nonzero(eligible_full))
        feasible_eligible_count = int(np.count_nonzero(eligible_full & accepted))
        feasible_success_count = int(np.count_nonzero(successful_full))
        # The primary ASR includes all initially-correct non-target attempts,
        # not only the easy subset accepted by the constraint oracle.
        if not self.strict_constraints:
            metrics['asr_unconstrained_diagnostic'] = float(metrics.get('asr', 0.0))
        metrics['asr'] = float(feasible_success_count / eligible_count) if eligible_count else float('nan')
        metrics['asr_conditional_on_feasibility'] = (
            float(feasible_success_count / feasible_eligible_count)
            if feasible_eligible_count else float('nan')
        )
        metrics['attack_eligible_sample_count'] = float(eligible_count)
        metrics['feasible_eligible_sample_count'] = float(feasible_eligible_count)
        metrics['realizability_rate_eligible'] = (
            float(feasible_eligible_count / eligible_count) if eligible_count else float('nan')
        )
        for label in metric_labels:
            name = (self.class_names[label] if self.class_names is not None
                    and 0 <= label < len(self.class_names) else f'class_{label}')
            class_eligible = eligible_full & (y_np == label)
            denominator = int(np.count_nonzero(class_eligible))
            metrics[f'asr_eligible_count_class_{label}'] = float(denominator)
            metrics[f'asr_feasible_eligible_count_class_{label}'] = float(np.count_nonzero(class_eligible & accepted))
            metrics[f'asr_success_count_class_{label}'] = float(np.count_nonzero(successful_full & (y_np == label)))
            metrics[f'asr_{name}'] = (
                float(np.count_nonzero(successful_full & (y_np == label)) / denominator)
                if denominator else float('nan')
            )
        metrics["gatekeeper_violation_total"] = float(sum(violation_metrics.values()))
        metrics["successful_sample_count"] = float(np.count_nonzero(successful_full))
        if self.detector is not None:
            assert detector_clean_scores is not None and detector_adversarial_scores is not None
            metrics["detector_enabled"] = 1.0
            # Matched pairs are the only defensible detector comparison.
            metrics["detection_auroc"] = detection_auroc(
                detector_clean_scores[accepted], detector_adversarial_scores[accepted]
            ) if np.any(accepted) else float('nan')
            metrics['detection_auroc_realizable'] = metrics['detection_auroc']
            metrics["detector_clean_score_mean"] = float(np.mean(detector_clean_scores))
            metrics["detector_adversarial_score_mean"] = (
                float(np.mean(detector_adversarial_scores[accepted])) if np.any(accepted) else float('nan')
            )
            metrics['detector_scored_adversarial_count'] = float(np.count_nonzero(score_mask))
            metrics["detector_adversarial_score_mean_realizable"] = (
                float(np.mean(detector_adversarial_scores[accepted])) if np.any(accepted) else float("nan")
            )
        else:
            metrics["detector_enabled"] = 0.0
        if self.attack_model is not None and self.attack_model is not self.target_model:
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
            defended_predictions=defended_predictions,
        )
