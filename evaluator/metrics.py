from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


def _class_metric_key(
    prefix: str,
    label: int,
    class_names: Iterable[str] | None,
) -> str:
    """Build a stable metric key using the human-readable class when available."""
    names = None if class_names is None else tuple(str(name) for name in class_names)
    class_name = names[label] if names is not None and 0 <= label < len(names) else f"class_{label}"
    return f"{prefix}_{class_name}"


@dataclass(frozen=True)
class ClassificationMetrics:
    mcc: float
    balanced_accuracy: float
    gmean: float
    far: float
    dr: float
    f1_macro: float
    f1_micro: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mcc": self.mcc,
            "balanced_accuracy": self.balanced_accuracy,
            "gmean": self.gmean,
            "far": self.far,
            "dr": self.dr,
            "f1_macro": self.f1_macro,
            "f1_micro": self.f1_micro,
        }


def _macro_gmean(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    if len(labels) == 0:
        return 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls: list[float] = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recalls.append(float(tp / (tp + fn)) if tp + fn else 0.0)
    return float(np.prod(np.clip(recalls, 0.0, 1.0)) ** (1.0 / len(recalls)))


def _macro_far(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    if len(labels) == 0:
        return 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fars: list[float] = []
    total = cm.sum()
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        fars.append(float(fp / (fp + tn)) if fp + tn else 0.0)
    return float(np.mean(fars)) if fars else 0.0


def _macro_dr(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    if len(labels) == 0:
        return 0.0
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rates: list[float] = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        rates.append(float(tp / (tp + fn)) if tp + fn else 0.0)
    return float(np.mean(rates)) if rates else 0.0


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Iterable[int] | None = None,
) -> ClassificationMetrics:
    yt = np.asarray(list(y_true), dtype=np.int64)
    yp = np.asarray(list(y_pred), dtype=np.int64)
    if len(yt) == 0:
        return ClassificationMetrics(*(0.0 for _ in range(7)))
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length")
    label_array = None if labels is None else np.asarray(list(labels), dtype=np.int64)
    try:
        mcc = float(matthews_corrcoef(yt, yp))
    except ValueError:
        mcc = 0.0
    return ClassificationMetrics(
        mcc=mcc,
        balanced_accuracy=float(balanced_accuracy_score(yt, yp)),
        gmean=_macro_gmean(yt, yp, label_array),
        far=_macro_far(yt, yp, label_array),
        dr=_macro_dr(yt, yp, label_array),
        f1_macro=float(f1_score(yt, yp, average="macro", zero_division=0, labels=label_array)),
        f1_micro=float(f1_score(yt, yp, average="micro", zero_division=0, labels=label_array)),
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
        return 0.0
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
            float(np.mean(adv[class_mask] == desired)) if np.any(class_mask) else 0.0
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
        return {"mean_l0": 0.0, "mean_l2": 0.0, "mean_linf": 0.0}

    delta = (b - a).reshape(len(a), -1)
    return {
        "mean_l0": float(np.mean(np.count_nonzero(np.abs(delta) > 1e-12, axis=1))),
        "mean_l2": float(np.mean(np.linalg.norm(delta, ord=2, axis=1))),
        "mean_linf": float(np.mean(np.linalg.norm(delta, ord=np.inf, axis=1))),
    }


def mean_queries(queries: Iterable[int | float] | None) -> float:
    if queries is None:
        return 0.0
    values = np.asarray(list(queries), dtype=np.float64)
    return float(values.mean()) if len(values) else 0.0


def mean_queries_to_success(
    queries: Iterable[int | float] | None,
    success_mask: Iterable[bool] | None,
) -> float:
    if queries is None or success_mask is None:
        return 0.0
    q = np.asarray(list(queries), dtype=np.float64)
    success = np.asarray(list(success_mask), dtype=bool)
    if len(q) != len(success):
        raise ValueError("queries and success_mask must have equal length")
    values = q[success]
    return float(values.mean()) if len(values) else 0.0


def realizability_rate(accepted_mask: Iterable[bool]) -> float:
    mask = np.asarray(list(accepted_mask), dtype=bool)
    return float(mask.mean()) if len(mask) else 1.0


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
    if len(yt) == 0:
        return 0.0
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length")
    benign = yt == benign_label
    malicious = ~benign
    fp_rate = float(np.sum((yp != benign_label) & benign) / max(np.sum(benign), 1))
    fn_rate = float(np.sum((yp == benign_label) & malicious) / max(np.sum(malicious), 1))
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
        result["asr"] = 0.0
        class_labels = np.unique(yt) if label_values is None else np.asarray(label_values, dtype=np.int64)
        result.update({
            _class_metric_key("asr", int(label), class_names): 0.0
            for label in class_labels
        })

    result["mean_queries_to_evasion"] = mean_queries_to_success(queries, effective_success)
    if x is not None and x_adv is not None:
        result.update(perturbation_distances(x, x_adv, sample_mask=effective_success))
    else:
        result.update({"mean_l0": 0.0, "mean_l2": 0.0, "mean_linf": 0.0})

    result["successful_sample_count"] = float(np.sum(effective_success)) if effective_success is not None else 0.0
    result["adveoc"] = adversarial_expected_operating_cost(
        yt,
        yp,
        benign_label=benign_label,
        c_fp=c_fp,
        c_fn=c_fn,
    )
    return result
