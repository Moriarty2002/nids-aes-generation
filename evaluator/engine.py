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
from models.wrapper import ModelWrapper, PyTorchModelWrapper
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
        return np.asarray(x, dtype=np.float32) * self.std + self.mean

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std


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

    def run(
        self,
        x: Any,
        y: np.ndarray | torch.Tensor,
        *,
        feature_names: Sequence[str],
        attack_input: Any | None = None,
        raw_traffic: Any | None = None,
        valid_mask: np.ndarray | torch.Tensor | None = None,
        attack_kwargs: Mapping[str, Any] | None = None,
    ) -> EvaluationOutput:
        """Generate, validate, score, and summarize one evaluation batch."""
        y_np = _as_numpy(y).astype(np.int64, copy=False)
        x_target = x

        # Baseline target-model inference is the primary operational-speed metric.
        baseline_pred, baseline_elapsed = _timed_predict(self.target_model, x_target)
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

        # Crucially, the gatekeeper is not available to the attack at any point.
        attack_result = self.attack.generate(target_input, y, **dict(attack_kwargs or {}))

        if attack_result.raw_adversarial is not None:
            if self.preprocessor is None:
                raise RuntimeError("Problem-space evaluation requires a preprocessor")
            processed = self.preprocessor.transform(attack_result.raw_adversarial)
            x_adv_target = np.asarray(processed.features, dtype=np.float32)
            actual_feature_names = tuple(processed.feature_names)
        else:
            if attack_result.adversarial is None:
                raise RuntimeError("Feature-space attack returned neither adversarial nor raw_adversarial data")
            x_adv_source = _as_numpy(attack_result.adversarial).astype(np.float32, copy=False)
            actual_feature_names = tuple(attack_result.feature_names or feature_names)
            x_adv_target = self._convert_attack_space_to_target(x_adv_source, actual_feature_names)

        if len(x_adv_target) != n_samples:
            raise ValueError(f"Attack produced {len(x_adv_target)} samples for a batch of {n_samples} labels")

        # Detector scoring is deliberately separate from NIDS success scoring and
        # from the universal constraint gatekeeper. Score all generated samples
        # so detector AUROC is not biased by strict-mode filtering.
        detector_clean_scores: np.ndarray | None = None
        detector_adversarial_scores: np.ndarray | None = None
        if self.detector is not None:
            if not isinstance(self.target_model, PyTorchModelWrapper):
                raise TypeError("Configured detectors require a PyTorchModelWrapper target model")
            detector_clean_scores = np.asarray(
                self.detector.score(x_target, self.target_model), dtype=np.float64
            ).reshape(-1)
            detector_adversarial_scores = np.asarray(
                self.detector.score(x_adv_target, self.target_model), dtype=np.float64
            ).reshape(-1)
            if len(detector_clean_scores) != n_samples or len(detector_adversarial_scores) != n_samples:
                raise ValueError("Detector must return exactly one score per clean/adversarial sample")

        # Strict architectural boundary: this is the first and only gatekeeper
        # call, so attacks cannot optimize against post-generation rules.
        gate = self.gatekeeper.validate(x_adv_target, actual_feature_names, valid_mask=valid_mask)
        accepted = np.asarray(gate.accepted, dtype=bool)
        score_mask = accepted.copy() if self.strict_constraints else np.ones(n_samples, dtype=bool)

        adversarial_pred_full = np.full(n_samples, -1, dtype=np.int64)
        if np.any(score_mask):
            adversarial_pred_full[score_mask] = np.asarray(
                self.target_model.predict(x_adv_target[score_mask]), dtype=np.int64
            )

        y_score = y_np[score_mask]
        base_score = baseline_pred[score_mask]
        adv_score = adversarial_pred_full[score_mask]
        queries = None if attack_result.queries is None else np.asarray(attack_result.queries)[score_mask]
        x_score = _as_numpy(x_target)[score_mask]
        x_adv_score = x_adv_target[score_mask]

        # Target-model success is recomputed here; surrogate/attack-reported
        # success is deliberately not used for transferability scoring.
        desired_label = self.benign_label if self.target_label is None else self.target_label
        successful = (base_score == y_score) & (y_score != desired_label) & (adv_score == desired_label)
        attack_to_attack = (
            (base_score == y_score)
            & (y_score != self.benign_label)
            & (adv_score != self.benign_label)
            & (adv_score != desired_label)
        )

        metric_labels = self.class_labels
        if metric_labels is None:
            observed = (
                np.unique(np.concatenate([y_score, base_score, adv_score]))
                if len(y_score)
                else np.asarray([], dtype=np.int64)
            )
            metric_labels = tuple(int(value) for value in observed)

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
            x=x_score,
            x_adv=x_adv_score,
            accepted_mask=accepted,
            baseline_pred=base_score,
            benign_label=self.benign_label,
            target_label=self.target_label,
            c_fp=self.c_fp,
            c_fn=self.c_fn,
        )

        metrics["strict_constraints"] = float(self.strict_constraints)
        metrics["strict_multiclass"] = float(self.strict_multiclass)
        metrics["attack_to_attack_failure_count"] = float(np.sum(attack_to_attack))
        metrics["attack_to_attack_failure_rate"] = float(
            np.mean(attack_to_attack) if len(attack_to_attack) else 0.0
        )
        metrics["scored_sample_count"] = float(score_mask.sum())
        metrics["rejected_sample_count"] = float((~accepted).sum())
        metrics["realizability_rate"] = float(gate.realizability_rate)
        metrics["inference_latency_ms_per_sample"] = float(baseline_latency_ms)
        metrics["throughput_flows_per_second"] = float(baseline_throughput)
        violation_metrics = _violation_counts(gate)
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
        physical = self.attack_normalization.denormalize(x_adv)
        return self.target_normalization.normalize(physical).astype(np.float32, copy=False)
