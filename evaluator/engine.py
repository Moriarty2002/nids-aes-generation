from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from attacks.base import AttackResult, BaseAttack, ProblemSpaceAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from defenses.base import BaseDetector
from models.wrapper import ModelWrapper
from preprocessing.base import BasePreprocessor
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
        """Denormalize into one output buffer rather than chaining full-array temporaries."""
        out = np.array(x, dtype=np.float32, copy=True)
        np.multiply(out, self.std, out=out)
        np.add(out, self.mean, out=out)
        return out

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize into one output buffer rather than chaining full-array temporaries."""
        out = np.array(x, dtype=np.float32, copy=True)
        np.subtract(out, self.mean, out=out)
        np.divide(out, self.std, out=out)
        return out


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
        benign_label: int = 0,
        target_label: int | None = None,
        c_fp: float = 1.0,
        c_fn: float = 1.0,
        strict_constraints: bool | None = None,
        strict_multiclass: bool = True,
        detector: BaseDetector | None = None,
        batch_size: int = 256,
        max_samples: int | None = None,
    ) -> None:
        self.target_model = target_model
        self.attack_model = attack_model or target_model
        self.attack = attack.bind_model(self.attack_model)
        self.gatekeeper = gatekeeper
        self.preprocessor = preprocessor
        self.attack_normalization = attack_normalization
        self.target_normalization = target_normalization
        self.class_labels = None if class_labels is None else tuple(int(value) for value in class_labels)
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

    def run(
        self,
        x: Any,
        y: np.ndarray | torch.Tensor,
        *,
        feature_names: Sequence[str],
        attack_input: Any | None = None,
        attack_input_is_target_normalized: bool = False,
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
        if self.max_samples is not None and len(y_np) > self.max_samples:
            raise ValueError(
                "EvaluationEngine received more samples than evaluation.max_samples; "
                "subsample before normalization in evaluate_attack_seed"
            )
        if attack_input_is_target_normalized:
            if isinstance(self.attack, ProblemSpaceAttack):
                raise ValueError("On-demand normalization conversion is only valid for feature-space attacks")
            if self.attack_normalization is None or self.target_normalization is None:
                raise ValueError(
                    "On-demand target-to-attack normalization requires both normalization objects"
                )

        baseline_pred, baseline_elapsed = _batched_predict(
            self.target_model, x_target, self.batch_size
        )
        if len(baseline_pred) != len(y_np):
            raise ValueError("Target baseline predictions and labels have different lengths")

        n_samples = len(y_np)
        baseline_latency_ms = baseline_elapsed * 1000.0 / max(n_samples, 1)
        baseline_throughput = n_samples / baseline_elapsed if n_samples else 0.0

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

            attack_batch_input = _slice_batch(target_input, start, stop)
            if attack_input_is_target_normalized:
                attack_batch_input = self._convert_target_space_to_attack(
                    _as_numpy(attack_batch_input)
                )

            batch_result = self.attack.generate(
                attack_batch_input,
                _slice_batch(y, start, stop),
                **batch_kwargs,
            )
            del attack_batch_input

            if not first_metadata:
                first_metadata = dict(batch_result.metadata)

            expected_batch_size = stop - start
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
            if batch_result.raw_adversarial is not None:
                if self.preprocessor is None:
                    raise RuntimeError("Problem-space evaluation requires a preprocessor")
                processed = self.preprocessor.transform(batch_result.raw_adversarial)
                x_adv_batch = np.asarray(processed.features, dtype=np.float32)
                batch_feature_names = tuple(processed.feature_names)
            else:
                if batch_result.adversarial is None:
                    raise RuntimeError(
                        "Feature-space attack returned neither adversarial nor raw_adversarial data"
                    )
                x_adv_source = _as_numpy(batch_result.adversarial).astype(np.float32, copy=False)
                batch_feature_names = tuple(batch_result.feature_names or feature_names)
                x_adv_batch = self._convert_attack_space_to_target(
                    x_adv_source, batch_feature_names
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

            x_clean_batch = _slice_batch(x_target, start, stop)
            y_batch = y_np[start:stop]
            baseline_batch = baseline_pred[start:stop]
            valid_mask_batch = None if valid_mask is None else _slice_batch(valid_mask, start, stop)
            gate = self.gatekeeper.validate(
                x_adv_batch,
                first_feature_names,
                valid_mask=valid_mask_batch,
                original_samples=x_clean_batch,
            )
            accepted_batch = np.asarray(gate.accepted, dtype=bool).reshape(-1)
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

            del batch_result, x_adv_batch, x_clean_batch, y_batch, baseline_batch
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
            metadata=first_metadata,
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

    def _convert_target_space_to_attack(self, x_target: np.ndarray) -> np.ndarray:
        """Convert one target-normalized batch into attack-model normalization in-place."""
        if self.target_normalization is None or self.attack_normalization is None:
            raise ValueError("Target-to-attack conversion requires both normalization objects")
        if self.target_normalization.feature_names != self.attack_normalization.feature_names:
            raise ValueError("Target and attack feature schemas differ")
        out = np.array(x_target, dtype=np.float32, copy=True)
        np.multiply(out, self.target_normalization.std, out=out)
        np.add(out, self.target_normalization.mean, out=out)
        np.subtract(out, self.attack_normalization.mean, out=out)
        np.divide(out, self.attack_normalization.std, out=out)
        return out

    def _convert_attack_space_to_target(self, x_adv: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        if self.attack_normalization is None or self.target_normalization is None:
            return x_adv
        if tuple(feature_names) != self.attack_normalization.feature_names:
            raise ValueError(
                "Attack feature schema does not match surrogate normalization schema: "
                f"{tuple(feature_names)} != {self.attack_normalization.feature_names}"
            )
        if tuple(feature_names) != self.target_normalization.feature_names:
            raise ValueError(
                "Surrogate and target feature schemas must match for transfer evaluation: "
                f"{tuple(feature_names)} != {self.target_normalization.feature_names}"
            )
        # Convert directly in one batch-sized output instead of allocating an
        # intermediate denormalized physical-space matrix.
        out = np.array(x_adv, dtype=np.float32, copy=True)
        np.multiply(out, self.attack_normalization.std, out=out)
        np.add(out, self.attack_normalization.mean, out=out)
        np.subtract(out, self.target_normalization.mean, out=out)
        np.divide(out, self.target_normalization.std, out=out)
        return out
