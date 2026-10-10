from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    std: float
    n: int

    def as_dict(self, metric: str) -> dict[str, float | int]:
        return {f"{metric}_mean": self.mean, f"{metric}_std": self.std, f"{metric}_n": self.n}


def aggregate_by_seed(seed_metrics: Sequence[Mapping[str, float]]) -> dict[str, AggregateMetric]:
    if not seed_metrics:
        return {}
    keys = sorted({key for row in seed_metrics for key in row})
    aggregated: dict[str, AggregateMetric] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in seed_metrics if key in row], dtype=np.float64)
        aggregated[key] = AggregateMetric(float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0), int(len(values)))
    return aggregated


def paired_wilcoxon(
    baseline_values: Iterable[float],
    attack_values: Iterable[float],
    *,
    alternative: str = "two-sided",
) -> tuple[float, int]:
    base = np.asarray(list(baseline_values), dtype=np.float64)
    attack = np.asarray(list(attack_values), dtype=np.float64)
    if base.shape != attack.shape:
        raise ValueError("Baseline and attack arrays must contain paired seed observations")
    if len(base) < 2:
        return float("nan"), 0
    diff = attack - base
    if np.allclose(diff, 0.0):
        return 1.0, int(len(diff))
    result = wilcoxon(base, attack, alternative=alternative, zero_method="wilcox", method="auto")
    return float(getattr(result, "pvalue")), int(len(diff) - np.count_nonzero(diff == 0))


def compare_against_baseline(
    baseline_seed_metrics: Sequence[Mapping[str, float]],
    attack_seed_metrics: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float | int]]:
    baseline_by_seed = {int(row["seed"]): row for row in baseline_seed_metrics}
    attack_by_seed = {int(row["seed"]): row for row in attack_seed_metrics}
    common = sorted(set(baseline_by_seed).intersection(attack_by_seed))
    if len(common) < 2:
        return {}
    metric_names = sorted(
        set().union(*(baseline_by_seed[s].keys() for s in common))
        .intersection(set().union(*(attack_by_seed[s].keys() for s in common)))
        - {"seed"}
    )
    results: dict[str, dict[str, float | int]] = {}
    for metric in metric_names:
        base = [float(baseline_by_seed[s][metric]) for s in common]
        attack = [float(attack_by_seed[s][metric]) for s in common]
        p, n_pairs = paired_wilcoxon(base, attack)
        results[metric] = {"p_value": p, "n_pairs": n_pairs}
    return results


def summarize(seed_metrics: Sequence[Mapping[str, float]], baseline_seed_metrics: Sequence[Mapping[str, float]] | None = None) -> dict[str, object]:
    aggregates = aggregate_by_seed(seed_metrics)
    summary: dict[str, object] = {"metrics": {k: v.as_dict(k) for k, v in aggregates.items()}}
    if baseline_seed_metrics is not None:
        summary["wilcoxon_vs_baseline"] = compare_against_baseline(baseline_seed_metrics, seed_metrics)
    return summary
