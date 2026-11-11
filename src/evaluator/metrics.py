from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


def _class_metric_key(
    prefix: str,
    label: int,
    class_names: Iterable[str] | None,
) -> str:
    """Build a stable metric key using the human-readable class when available."""
    names = None if class_names is None else tuple(str(name) for name in class_names)
    class_name = names[label] if names is not None and 0 <= label < len(names) else f"class_{label}"
    return f"{prefix}_{class_name}"


# A class missing from ground truth has undefined recall, not zero recall.
# Aggregate observed-support measures separately from complete-schema measures.
CLASSIFICATION_METRIC_POLICY_VERSION = "support_aware_multiclass_v1"


@dataclass(frozen=True)
class ClassificationMetrics:
    mcc: float
    balanced_accuracy: float
    gmean: float
    far: float
    dr: float
    f1_macro: float
    f1_micro: float
    f1_macro_observed_support: float
    f1_macro_full_schema: float
    gmean_full_schema: float
    dr_full_schema: float
    benign_false_alarm_rate: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def _integer_labels(values: Iterable[int], name: str) -> np.ndarray:
    """Reject accidental label truncation or noninteger IDs before computing metrics."""
    raw = np.asarray(list(values))
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain 1-D integer class IDs')
    result = raw.astype(np.int64)
    if not np.array_equal(result, raw):
        raise ValueError(f'{name} contains integer IDs outside int64 range')
    return result


def _classification_inputs(
    y_true: Iterable[int], y_pred: Iterable[int], labels: Iterable[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt, yp = _integer_labels(y_true, 'y_true'), _integer_labels(y_pred, 'y_pred')
    if len(yt) != len(yp):
        raise ValueError('y_true and y_pred must have equal length')
    declared = (
        np.unique(np.concatenate((yt, yp))) if labels is None
        else _integer_labels(labels, 'labels')
    )
    if len(set(declared.tolist())) != len(declared):
        raise ValueError('Class schema must have unique labels')
    if not np.isin(yt, declared).all() or not np.isin(yp, declared).all():
        raise ValueError('Observed class ID outside the declared class schema')
    return yt, yp, declared


def classification_support(
    y_true: Iterable[int], y_pred: Iterable[int], labels: Iterable[int] | None = None,
) -> dict[str, object]:
    """Explicit ground-truth/prediction counts for the *given* metric population.

    All authoritative class IDs appear even if the accepted subset is empty.
    This is descriptive metadata, never a way to change feasible-ASR membership.
    """
    yt, yp, declared = _classification_inputs(y_true, y_pred, labels)
    truth_counts = {str(int(k)): int(np.count_nonzero(yt == k)) for k in declared}
    pred_counts = {str(int(k)): int(np.count_nonzero(yp == k)) for k in declared}
    missing = [int(k) for k in declared if truth_counts[str(int(k))] == 0]
    predicted_only = [k for k in missing if pred_counts[str(k)] > 0]
    true_positive = {str(int(k)): int(np.count_nonzero((yt == k) & (yp == k)))
                     for k in declared}
    recall = {
        k: (true_positive[k] / count if count else float('nan'))
        for k, count in truth_counts.items()
    }
    fpr = {
        k: ((pred_counts[k] - true_positive[k]) / (len(yt) - count)
            if len(yt) > count else float('nan'))
        for k, count in truth_counts.items()
    }
    return {
        'n': int(len(yt)),
        'ground_truth_support': truth_counts,
        'predicted_counts': pred_counts,
        'true_positive_counts': true_positive,
        'per_class_recall': recall,  # NaN in Python / null in strict JSON if unsupported
        'per_class_ovr_false_positive_rate': fpr,
        'missing_ground_truth_classes': missing,
        'predicted_only_classes': predicted_only,
        'schema_complete': bool(len(declared) and not missing),
        'metric_policy_version': CLASSIFICATION_METRIC_POLICY_VERSION,
    }


def _confusion(yt: np.ndarray, yp: np.ndarray, declared: np.ndarray) -> np.ndarray:
    # sklearn warns on a valid one-class population despite explicit labels.
    # There is no need to call it for the trivial 1x1 count.
    if len(declared) == 1:
        return np.array([[len(yt)]], dtype=np.int64)
    return confusion_matrix(yt, yp, labels=declared)


def _macro_gmean(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    yt, yp, declared = _classification_inputs(y_true, y_pred, labels)
    if not len(yt):
        return float('nan')
    cm = _confusion(yt, yp, declared)
    support = cm.sum(axis=1)
    observed = support > 0
    if not np.any(observed):
        return float('nan')
    recalls = cm.diagonal()[observed] / support[observed]
    return float(np.prod(recalls) ** (1. / len(recalls)))


def _macro_dr(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    yt, yp, declared = _classification_inputs(y_true, y_pred, labels)
    if not len(yt):
        return float('nan')
    cm = _confusion(yt, yp, declared)
    support = cm.sum(axis=1)
    observed = support > 0
    return float(np.mean(cm.diagonal()[observed] / support[observed])) if np.any(observed) else float('nan')


def _macro_far(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    """Mean one-vs-rest FPR across classes with a nonempty *negative* group.

    This is not necessarily the NIDS benign false-alarm rate. When the
    negative denominator is zero, the class FPR is undefined and omitted.
    """
    yt, yp, declared = _classification_inputs(y_true, y_pred, labels)
    if not len(yt):
        return float('nan')
    cm = _confusion(yt, yp, declared)
    positives = cm.sum(axis=1)
    negatives = cm.sum() - positives
    defined = negatives > 0
    if not np.any(defined):
        return float('nan')
    fp = cm.sum(axis=0) - cm.diagonal()
    return float(np.mean(fp[defined] / negatives[defined]))


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Iterable[int] | None = None,
) -> ClassificationMetrics:
    yt, yp, declared = _classification_inputs(y_true, y_pred, labels)
    if not len(yt):
        return ClassificationMetrics(*(float('nan') for _ in range(12)))
    cm = _confusion(yt, yp, declared)
    support = cm.sum(axis=1)
    supported = declared[support > 0]
    complete = bool(len(declared) and np.all(support > 0))
    f1_observed = float(f1_score(yt, yp, average='macro', zero_division=0, labels=supported))
    full_f1 = (
        float(f1_score(yt, yp, average='macro', zero_division=0, labels=declared))
        if complete else float('nan')
    )
    # MCC is undefined if the entire population has one class in both
    # truth and prediction; avoid sklearn's single-label warning.
    if len(np.unique(np.concatenate((yt, yp)))) < 2:
        mcc = float('nan')
    else:
        try:
            mcc = float(matthews_corrcoef(yt, yp))
        except ValueError:
            mcc = float('nan')
    dr = _macro_dr(yt, yp, declared)
    gmean = _macro_gmean(yt, yp, declared)
    return ClassificationMetrics(
        mcc=mcc,
        balanced_accuracy=dr,  # sklearn's balanced accuracy = observed-class macro recall
        gmean=gmean,
        far=_macro_far(yt, yp, declared),
        dr=dr,
        f1_macro=f1_observed,
        f1_micro=float(f1_score(yt, yp, average='micro', zero_division=0, labels=declared)),
        f1_macro_observed_support=f1_observed,
        f1_macro_full_schema=full_f1,
        gmean_full_schema=gmean if complete else float('nan'),
        dr_full_schema=dr if complete else float('nan'),
        benign_false_alarm_rate=(
            float(np.mean(yp[yt == 0] != 0)) if np.any(yt == 0)
            else float('nan')
        ),
    )


def detection_auroc(clean_scores: Iterable[float], adversarial_scores: Iterable[float]) -> float:
    """Compute AUROC for detector scores, treating clean=0 and adversarial=1.

    The detector contract is that larger scores indicate more anomalous samples.
    A NaN is returned when either population is empty, because AUROC is undefined
    without both negative and positive examples. Non-finite scores are rejected
    rather than silently dropped so a malformed evaluation cannot look valid.
    """
    clean = np.asarray(list(clean_scores), dtype=np.float64).reshape(-1)
    adversarial = np.asarray(list(adversarial_scores), dtype=np.float64).reshape(-1)
    if clean.size == 0 or adversarial.size == 0:
        return float("nan")
    if not np.isfinite(clean).all() or not np.isfinite(adversarial).all():
        raise ValueError("Detection AUROC scores must all be finite")
    labels = np.concatenate((np.zeros(clean.size, dtype=np.int64), np.ones(adversarial.size, dtype=np.int64)))
    scores = np.concatenate((clean, adversarial))
    return float(roc_auc_score(labels, scores))


def _attack_eligibility(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    *,
    benign_label: int,
    target_label: int | None,
) -> np.ndarray:
    """Return source flows for strict evasion measurement.

    A sample is attack-eligible only when the clean NIDS correctly identifies
    its true class and the true class is not already the requested target.
    """
    desired = benign_label if target_label is None else target_label
    return (baseline_pred == y_true) & (y_true != desired)


def attack_success_mask(
    y_true: Iterable[int],
    baseline_pred: Iterable[int],
    adversarial_pred: Iterable[int],
    *,
    benign_label: int = 0,
    target_label: int | None = None,
) -> np.ndarray:
    """Return the canonical, strict target-label success mask.

    In multi-class AC, predicting one attack class as another attack class is
    explicitly a failure. Success requires the target model to predict the
    configured target label (Benign by default).
    """
    yt = np.asarray(list(y_true), dtype=np.int64)
    base = np.asarray(list(baseline_pred), dtype=np.int64)
    adv = np.asarray(list(adversarial_pred), dtype=np.int64)
    if not (len(yt) == len(base) == len(adv)):
        raise ValueError("y_true, baseline_pred, and adversarial_pred must have equal length")
    desired = benign_label if target_label is None else target_label
    eligible = _attack_eligibility(yt, base, benign_label=benign_label, target_label=target_label)
    return eligible & (adv == desired)


def attack_success_rate(
    y_true: Iterable[int],
    baseline_pred: Iterable[int],
    adversarial_pred: Iterable[int],
    *,
    benign_label: int = 0,
    target_label: int | None = None,
) -> float:
    yt = np.asarray(list(y_true), dtype=np.int64)
    base = np.asarray(list(baseline_pred), dtype=np.int64)
    adv = np.asarray(list(adversarial_pred), dtype=np.int64)
    eligible = _attack_eligibility(yt, base, benign_label=benign_label, target_label=target_label)
    if not np.any(eligible):
        return float('nan')
    desired = benign_label if target_label is None else target_label
    return float(np.mean(adv[eligible] == desired))


def attack_success_rate_by_class(
    y_true: Iterable[int],
    baseline_pred: Iterable[int],
    adversarial_pred: Iterable[int],
    *,
    labels: Iterable[int] | None = None,
    class_names: Iterable[str] | None = None,
    benign_label: int = 0,
    target_label: int | None = None,
) -> dict[str, float]:
    """Return per-source-class ASR using human-readable metric keys."""
    yt = np.asarray(list(y_true), dtype=np.int64)
    base = np.asarray(list(baseline_pred), dtype=np.int64)
    adv = np.asarray(list(adversarial_pred), dtype=np.int64)
    if not (len(yt) == len(base) == len(adv)):
        raise ValueError("y_true, baseline_pred, and adversarial_pred must have equal length")
    class_labels = np.unique(yt) if labels is None else np.asarray(list(labels), dtype=np.int64)
    eligible = _attack_eligibility(yt, base, benign_label=benign_label, target_label=target_label)
    desired = benign_label if target_label is None else target_label

    result: dict[str, float] = {}
    for label in class_labels:
        class_mask = eligible & (yt == int(label))
        result[_class_metric_key("asr", int(label), class_names)] = (
            float(np.mean(adv[class_mask] == desired)) if np.any(class_mask) else float('nan')
        )
    return result


def perturbation_distances(
    x: np.ndarray,
    x_adv: np.ndarray,
    *,
    sample_mask: Iterable[bool] | None = None,
) -> dict[str, float]:
    """Return mean L0/L2/Linf distances, optionally over successful evasions."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(x_adv, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"x and x_adv shapes differ: {a.shape} != {b.shape}")
    if a.ndim < 2:
        raise ValueError(f"Expected batch-shaped arrays, received {a.shape}")
    if sample_mask is not None:
        mask = np.asarray(list(sample_mask), dtype=bool)
        if len(mask) != len(a):
            raise ValueError("sample_mask and batch length must match")
        a = a[mask]
        b = b[mask]
    if len(a) == 0:
        return {"mean_l0": float('nan'), "mean_l2": float('nan'), "mean_linf": float('nan')}

    delta = (b - a).reshape(len(a), -1)
    return {
        "mean_l0": float(np.mean(np.count_nonzero(delta != 0, axis=1))),
        "mean_l2": float(np.mean(np.linalg.norm(delta, ord=2, axis=1))),
        "mean_linf": float(np.mean(np.linalg.norm(delta, ord=np.inf, axis=1))),
    }


def mean_queries(queries: Iterable[int | float] | None) -> float:
    if queries is None:
        return float('nan')  # unavailable telemetry is not a zero-query attack
    values = np.asarray(list(queries), dtype=np.float64)
    return float(values.mean()) if len(values) else float('nan')


def mean_queries_to_success(
    queries: Iterable[int | float] | None,
    success_mask: Iterable[bool] | None,
) -> float:
    if queries is None or success_mask is None:
        return float('nan')
    q = np.asarray(list(queries), dtype=np.float64)
    success = np.asarray(list(success_mask), dtype=bool)
    if len(q) != len(success):
        raise ValueError("queries and success_mask must have equal length")
    values = q[success]
    return float(values.mean()) if len(values) else float('nan')


def realizability_rate(accepted_mask: Iterable[bool]) -> float:
    mask = np.asarray(list(accepted_mask), dtype=bool)
    return float(mask.mean()) if len(mask) else float('nan')


def adversarial_expected_operating_cost(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    benign_label: int = 0,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
) -> float:
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_pred), dtype=np.int64)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length")
    benign = yt == benign_label
    malicious = ~benign
    # Both populations are necessary to estimate this two-group cost.
    if not np.any(benign) or not np.any(malicious):
        return float('nan')
    fp_rate = float(np.mean(yp[benign] != benign_label))
    fn_rate = float(np.mean(yp[malicious] == benign_label))
    return c_fp * fp_rate + c_fn * fn_rate


def evaluate_predictions(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    labels: Iterable[int] | None = None,
    class_names: Iterable[str] | None = None,
    queries: Iterable[int | float] | None = None,
    success_mask: Iterable[bool] | None = None,
    query_budget: int | None = None,
    x: np.ndarray | None = None,
    x_adv: np.ndarray | None = None,
    accepted_mask: Iterable[bool] | None = None,
    baseline_pred: Iterable[int] | None = None,
    benign_label: int = 0,
    target_label: int | None = None,
    c_fp: float = 1.0,
    c_fn: float = 1.0,
) -> dict[str, float]:
    """Compute classification and adversarial metrics.

    When ``baseline_pred`` is present, it is authoritative for attack success:
    the supplied ``success_mask`` is ignored for ASR, perturbation distances,
    and mean-queries-to-evasion so an attack plugin cannot redefine what counts
    as an evasion. The canonical definition requires ``adv_pred == target``.
    """
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_pred), dtype=np.int64)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length")
    label_values = None if labels is None else tuple(int(v) for v in labels)
    classification = classification_metrics(yt, yp, label_values)
    result = classification.as_dict()
    result["mean_queries"] = mean_queries(queries)
    if query_budget is not None:
        result["query_budget"] = float(query_budget)
    result["realizability_rate"] = 1.0 if accepted_mask is None else realizability_rate(accepted_mask)

    provided_success = None if success_mask is None else np.asarray(list(success_mask), dtype=bool)
    if provided_success is not None and len(provided_success) != len(yt):
        raise ValueError("success_mask and batch length must match")

    if baseline_pred is not None:
        baseline = np.asarray(list(baseline_pred), dtype=np.int64)
        if len(baseline) != len(yt):
            raise ValueError("baseline_pred and y_true must have equal length")
        strict_success = attack_success_mask(
            yt,
            baseline,
            yp,
            benign_label=benign_label,
            target_label=target_label,
        )
        result["asr"] = attack_success_rate(
            yt,
            baseline,
            yp,
            benign_label=benign_label,
            target_label=target_label,
        )
        result.update(
            attack_success_rate_by_class(
                yt,
                baseline,
                yp,
                labels=label_values,
                class_names=class_names,
                benign_label=benign_label,
                target_label=target_label,
            )
        )
        effective_success = strict_success
    else:
        effective_success = provided_success
        result["asr"] = float("nan")
        class_labels = np.unique(yt) if label_values is None else np.asarray(label_values, dtype=np.int64)
        result.update({
            _class_metric_key("asr", int(label), class_names): float("nan")
            for label in class_labels
        })

    result["mean_queries_to_evasion"] = mean_queries_to_success(queries, effective_success)
    if x is not None and x_adv is not None:
        result.update(perturbation_distances(x, x_adv, sample_mask=effective_success))
    else:
        result.update({"mean_l0": float('nan'), "mean_l2": float('nan'), "mean_linf": float('nan')})

    result["successful_sample_count"] = float(np.sum(effective_success)) if effective_success is not None else 0.0
    result["adveoc"] = adversarial_expected_operating_cost(
        yt,
        yp,
        benign_label=benign_label,
        c_fp=c_fp,
        c_fn=c_fn,
    )
    return result


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
        l0_total += float(np.count_nonzero(delta != 0, axis=1).sum(dtype=np.int64))
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


class PhysicalDistanceAccumulator:
    """Accumulate exact float64 physical perturbations inside existing batches.

    Distances across mixed physical units are DESCRIPTIVE, not a normalized edit
    cost. `mean_l0` counts exact nonzero physical deltas, including tiny IATs.
    Each population has an explicit sample count and per-feature packet count.
    """

    def __init__(self, feature_names: Iterable[str]) -> None:
        self.feature_names = tuple(str(name) for name in feature_names)
        self.records: dict[str, dict[str, object]] = {
            group: {
                "n": 0, "l0": 0., "l2": 0., "linf": 0.,
                "abs_sum": np.zeros(len(self.feature_names), dtype=np.float64),
                "changed": np.zeros(len(self.feature_names), dtype=np.int64),
                "valid_packets": 0,
            }
            for group in ("eligible", "accepted", "successful")
        }

    def update(
        self, clean: np.ndarray, adversarial: np.ndarray,
        valid_mask: np.ndarray | None,
        *, eligible: np.ndarray, accepted: np.ndarray, successful: np.ndarray,
    ) -> None:
        clean = np.asarray(clean)
        adversarial = np.asarray(adversarial)
        if clean.dtype != np.float64 or adversarial.dtype != np.float64 or clean.shape != adversarial.shape:
            raise TypeError("Aligned physical distances require equally shaped authoritative float64 inputs")
        n, f = len(clean), len(self.feature_names)
        if clean.size != n * f * (clean.size // (n * f)):
            raise ValueError("Physical batch width must be divisible by ordered feature count")
        t = clean.size // (n * f)
        real = np.ones((n, t), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if real.shape != (n, t):
            raise ValueError("Physical-distance valid_mask shape mismatch")
        delta = np.subtract(adversarial.reshape(n, t, f), clean.reshape(n, t, f), dtype=np.float64)
        delta[~real] = 0.0
        flat = delta.reshape(n, -1)
        absolute = np.abs(flat)
        l0 = np.count_nonzero(flat != 0., axis=1)
        l2 = np.hypot.reduce(absolute, axis=1)
        linf = absolute.max(axis=1)
        for group, selection in (("eligible", eligible), ("accepted", accepted), ("successful", successful)):
            selected = np.asarray(selection, dtype=bool)
            if selected.shape != (n,):
                raise ValueError("Physical-distance group mask shape mismatch")
            if not selected.any():
                continue
            record = self.records[group]
            record["n"] += int(selected.sum())
            record["l0"] += float(l0[selected].sum(dtype=np.float64))
            record["l2"] += float(l2[selected].sum(dtype=np.float64))
            record["linf"] += float(linf[selected].sum(dtype=np.float64))
            record["abs_sum"] += np.abs(delta[selected]).sum(axis=(0, 1), dtype=np.float64)
            record["changed"] += np.count_nonzero(delta[selected], axis=(0, 1))
            record["valid_packets"] += int(real[selected].sum())

    def summary(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for group, r in self.records.items():
            n = int(r["n"])
            result[f"physical_{group}_sample_count"] = float(n)
            for metric in ("l0", "l2", "linf"):
                result[f"physical_{group}_mean_{metric}"] = (
                    float(r[metric]) / n if n else float("nan")
                )
            packet_count = int(r["valid_packets"])
            for j, field in enumerate(self.feature_names):
                result[f"physical_{group}_{field}_mean_abs_change"] = (
                    float(r["abs_sum"][j]) / packet_count if packet_count else float("nan")
                )
                result[f"physical_{group}_{field}_changed_packet_count"] = float(r["changed"][j])
        result.update({
            f"mean_{metric}": result[f"physical_accepted_mean_{metric}"]
            for metric in ("l0", "l2", "linf")
        })
        return result
