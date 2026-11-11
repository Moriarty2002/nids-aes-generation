"""Matched comparisons, descriptive seed summaries and evidence-gated inference.

The default experiment repeatedly splits one corpus. Its test observations and
trained models can overlap across seeds: neither a Student-t confidence interval
nor a Wilcoxon p-value automatically has an independent-replicate interpretation.
The mathematical tests remain available for *declared independent acquisitions*
with distinct, verifiable source hashes and predeclared paired units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import re

import numpy as np
from scipy.stats import t as student_t, wilcoxon


@dataclass(frozen=True)
class IndependentAcquisitionDesign:
    """Evidence identifying independent acquisitions, not RNG seeds.

    Distinct file hashes alone do not prove capture independence. ``evidence``
    records the external acquisition protocol supporting that assumption; all
    inference using this design is explicitly conditional on that declaration.
    """

    unit_ids: Mapping[int, str]
    source_sha256: Mapping[int, str]
    evidence: str

    def verify(self, seeds: Sequence[int]) -> None:
        selected = tuple(int(seed) for seed in seeds)
        if len(selected) < 5 or len(set(selected)) != len(selected):
            raise ValueError('Independent-replicate inference requires >=5 distinct paired seeds')
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError('Independent acquisitions require an explicit evidence description')
        try:
            units = [str(self.unit_ids[seed]) for seed in selected]
            sources = [str(self.source_sha256[seed]) for seed in selected]
        except KeyError as exc:
            raise ValueError('Missing independent acquisition provenance for a seed') from exc
        if any(not unit.strip() for unit in units) or len(set(units)) != len(units):
            raise ValueError('Acquisition units must be nonempty and unique per seed')
        if (len(set(sources)) != len(sources)
                or any(re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None for digest in sources)):
            raise ValueError('Independent acquisitions need distinct valid source SHA-256 hashes')


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    std: float
    n: int
    ci95_low: float = float('nan')
    ci95_high: float = float('nan')
    inference_status: str = 'descriptive_repeated_splits'

    def as_dict(self, metric: str) -> dict[str, float | int | str]:
        return {
            f'{metric}_mean': self.mean,
            f'{metric}_std': self.std,
            f'{metric}_n': self.n,
            f'{metric}_ci95_low': self.ci95_low,
            f'{metric}_ci95_high': self.ci95_high,
            f'{metric}_inference_status': self.inference_status,
        }


def aggregate_by_seed(
    seed_metrics: Sequence[Mapping[str, float]],
    *,
    independent_design: IndependentAcquisitionDesign | None = None,
) -> dict[str, AggregateMetric]:
    if not seed_metrics:
        return {}
    if independent_design is not None:
        try:
            independent_design.verify([int(row['seed']) for row in seed_metrics])
        except (KeyError, TypeError) as exc:
            raise ValueError('Independent analysis requires a seed for every row') from exc
        for row in seed_metrics:
            seed = int(row['seed'])
            if row.get('dataset_source_sha256') != independent_design.source_sha256[seed]:
                raise ValueError('Independent acquisition must match the source hash on every metric row')
    keys = sorted({key for row in seed_metrics for key in row if key != 'seed'})
    aggregated: dict[str, AggregateMetric] = {}
    for key in keys:
        try:
            values = np.asarray([float(row[key]) for row in seed_metrics if key in row], dtype=np.float64)
        except (TypeError, ValueError):
            # Pairing/acquisition metadata is not a numeric outcome.
            continue
        values = values[np.isfinite(values)]
        n = len(values)
        if not n:
            aggregated[key] = AggregateMetric(float('nan'), float('nan'), 0)
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        low = high = float('nan')
        status = 'descriptive_repeated_splits'
        if independent_design is not None:
            # A missing/undefined metric can eliminate an acquisition. Inference
            # requires at least five actually observed units, not five seed IDs.
            if n >= 5:
                margin = float(student_t.ppf(0.975, n - 1) * std / np.sqrt(n))
                low, high = mean - margin, mean + margin
                status = 'conditional_on_declared_independent_acquisitions'
            else:
                status = 'insufficient_independent_observations'
        aggregated[key] = AggregateMetric(mean, std, n, low, high, status)
    return aggregated


def paired_wilcoxon(
    baseline_values: Iterable[float], attack_values: Iterable[float],
    *, alternative: str = 'two-sided',
) -> tuple[float, int]:
    """Low-level signed-rank calculation; caller certifies paired independent units."""
    base = np.asarray(list(baseline_values), dtype=np.float64)
    attack = np.asarray(list(attack_values), dtype=np.float64)
    if base.shape != attack.shape:
        raise ValueError('Paired arrays require matching shape, IDs and protocol')
    if not np.isfinite(base).all() or not np.isfinite(attack).all():
        raise ValueError('Paired Wilcoxon cannot use NaN/inf observations')
    if len(base) < 2:
        return float('nan'), 0
    diff = attack - base
    nonzero = int(np.count_nonzero(diff != 0))
    if not nonzero:
        return 1.0, 0
    result = wilcoxon(base, attack, alternative=alternative, zero_method='wilcox', method='auto')
    return float(result.pvalue), nonzero


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm family-wise correction for a predeclared set of simultaneous tests."""
    items = sorted(((key, float(p)) for key, p in p_values.items()), key=lambda x: x[1])
    if any(not 0 <= p <= 1 for _, p in items):
        raise ValueError('Holm correction requires finite p-values in [0,1]')
    corrected: dict[str, float] = {}
    floor = 0.0
    for idx, (key, p) in enumerate(items):
        floor = max(floor, min(1.0, p * (len(items) - idx)))
        corrected[key] = floor
    return corrected


def compare_against_baseline(
    baseline_seed_metrics: Sequence[Mapping[str, float]],
    attack_seed_metrics: Sequence[Mapping[str, float]],
    *,
    independent_design: IndependentAcquisitionDesign | None = None,
    primary_metrics: Sequence[str] = ('asr',),
) -> dict[str, dict[str, float | int | str]]:
    """Paired Wilcoxon for explicitly independent acquisitions only.

    Matching seed numbers and pairing fingerprints establish within-seed
    comparability, *not* independence across resamples of one corpus.
    """
    if independent_design is None:
        return {}
    if not primary_metrics or len(set(primary_metrics)) != len(primary_metrics):
        raise ValueError('Predeclared primary metric names must be unique and nonempty')
    base_by_seed = {int(row['seed']): row for row in baseline_seed_metrics}
    attack_by_seed = {int(row['seed']): row for row in attack_seed_metrics}
    if (len(base_by_seed) != len(baseline_seed_metrics)
            or len(attack_by_seed) != len(attack_seed_metrics)):
        raise ValueError('Duplicate seed records cannot be paired')
    common = sorted(set(base_by_seed).intersection(attack_by_seed))
    common = [seed for seed in common if
              isinstance(base_by_seed[seed].get('pairing_fingerprint'), str)
              and base_by_seed[seed]['pairing_fingerprint']
              and base_by_seed[seed]['pairing_fingerprint'] == attack_by_seed[seed].get('pairing_fingerprint')]
    independent_design.verify(common)
    # If source IDs are recorded on rows, they must match the acquisition claim.
    for seed in common:
        for arm in (base_by_seed, attack_by_seed):
            recorded = arm[seed].get('dataset_source_sha256')
            if recorded != independent_design.source_sha256[seed]:
                raise ValueError('Declared independent acquisition does not match row source hash')
    results: dict[str, dict[str, float | int | str]] = {}
    for metric in primary_metrics:
        try:
            pairs = [(float(base_by_seed[s][metric]), float(attack_by_seed[s][metric])) for s in common]
        except (KeyError, TypeError, ValueError):
            continue
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        if len(pairs) < 5:
            continue
        p, nonzero = paired_wilcoxon((a for a, _ in pairs), (b for _, b in pairs))
        diff = np.asarray([b - a for a, b in pairs], dtype=np.float64)
        results[metric] = {
            'p_value': p, 'n_nonzero_pairs': nonzero, 'n_matched_seeds': len(pairs),
            'paired_delta_mean': float(diff.mean()),
            'paired_delta_std': float(diff.std(ddof=1)),
            'effect_size_dz': (
                float(diff.mean() / diff.std(ddof=1))
                if float(diff.std(ddof=1)) > 0 else float('nan')
            ),
            'inference_status': 'conditional_on_declared_independent_acquisitions',
        }
    adjusted = holm_adjust({k: float(v['p_value']) for k, v in results.items()})
    for metric, adjusted_p in adjusted.items():
        results[metric]['p_value_holm'] = adjusted_p
    return results


def paired_record_delta(
    baseline_ids: Sequence[str], baseline_values: Sequence[float],
    attack_ids: Sequence[str], attack_values: Sequence[float],
    *, group_ids: Sequence[str] | None = None,
    replicates: int = 2000, seed: int = 0,
) -> dict[str, float | int | str]:
    """Matched-record differences, with only conditional capture-group intervals.

    The caller verifies exact target, budget and protocol pairing. Without
    independently established capture groups an IID-flow bootstrap is NOT used.
    """
    a_ids, b_ids = tuple(map(str, baseline_ids)), tuple(map(str, attack_ids))
    if len(set(a_ids)) != len(a_ids) or a_ids != b_ids:
        raise ValueError('Paired records require the same unique ordered sample IDs')
    a, b = np.asarray(baseline_values, dtype=np.float64), np.asarray(attack_values, dtype=np.float64)
    if a.shape != b.shape or a.shape != (len(a_ids),) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired record values must be finite 1-D, aligned arrays')
    if replicates < 2:
        raise ValueError('replicates must be >= 2')
    if len(a) == 0:
        return {'delta_mean': float('nan'), 'ci95_low': float('nan'), 'ci95_high': float('nan'),
                'n': 0, 'n_groups': 0, 'interval_status': 'undefined_empty_population'}
    deltas = b - a
    if group_ids is None:
        return {'delta_mean': float(deltas.mean()), 'ci95_low': float('nan'),
                'ci95_high': float('nan'), 'n': len(a), 'n_groups': 0,
                'interval_status': 'undefined_without_capture_groups'}
    if len(group_ids) != len(a):
        raise ValueError('group_ids must align exactly with paired samples')
    values = np.asarray(group_ids, dtype=str)
    if any(not value.strip() for value in values):
        raise ValueError('Capture group IDs must be nonempty')
    _, groups = np.unique(values, return_inverse=True)
    distinct = np.unique(groups)
    if len(distinct) < 2:
        return {'delta_mean': float(deltas.mean()), 'ci95_low': float('nan'),
                'ci95_high': float('nan'), 'n': len(a), 'n_groups': len(distinct),
                'interval_status': 'undefined_insufficient_capture_groups'}
    sample_sets = [np.flatnonzero(groups == g) for g in distinct]
    rng = np.random.default_rng(seed)
    boots = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        indices = np.concatenate([sample_sets[j] for j in rng.integers(len(sample_sets), size=len(sample_sets))])
        boots[i] = float(deltas[indices].mean())
    low, high = np.percentile(boots, [2.5, 97.5])
    return {'delta_mean': float(deltas.mean()), 'ci95_low': float(low), 'ci95_high': float(high),
            'n': len(a), 'n_groups': len(distinct),
            'interval_status': 'conditional_on_declared_capture_groups'}


def summarize(
    seed_metrics: Sequence[Mapping[str, float]],
    baseline_seed_metrics: Sequence[Mapping[str, float]] | None = None,
    *, independent_design: IndependentAcquisitionDesign | None = None,
) -> dict[str, object]:
    aggregates = aggregate_by_seed(seed_metrics, independent_design=independent_design)
    summary: dict[str, object] = {'metrics': {k: v.as_dict(k) for k, v in aggregates.items()}}
    if baseline_seed_metrics is not None:
        summary['wilcoxon_vs_baseline'] = compare_against_baseline(
            baseline_seed_metrics, seed_metrics, independent_design=independent_design
        )
    return summary


def corrected_repeated_holdout_t(
    paired_deltas: Sequence[float], *, train_size: int, test_size: int,
) -> dict[str, float | int | str]:
    """Nadeau--Bengio corrected resampled t, conditional on repeated holdouts.

    This accounts *approximately* for overlapping random train/test partitions
    of one corpus. It is not a replacement for independent capture-level
    inference, nor is it applicable to arbitrary/grouped splitting protocols.
    The caller must independently verify matched per-split outcomes and sizes.
    """
    values = np.asarray(paired_deltas, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 5 or not np.isfinite(values).all()
            or isinstance(train_size, (bool, np.bool_))
            or isinstance(test_size, (bool, np.bool_))
            or int(train_size) != train_size or int(test_size) != test_size
            or train_size <= 0 or test_size <= 0):
        raise ValueError('Corrected repeated-holdout t requires >=5 finite paired '
                         'differences and positive integer split sizes')
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    result: dict[str, float | int | str] = {
        'method': 'nadeau_bengio_corrected_resampled_t',
        'inference_status': 'approximate_conditional_repeated_random_holdouts',
        'n_paired_splits': n, 'train_size': int(train_size),
        'test_size': int(test_size), 'paired_delta_mean': mean,
        'paired_delta_std': std, 'p_value': float('nan'),
        'ci95_low': float('nan'), 'ci95_high': float('nan'),
    }
    # A constant observed difference provides no empirical variance estimate.
    if std == 0.0:
        result['inference_status'] = 'undefined_zero_between_split_variance'
        return result
    # Nadeau & Bengio (2003): Var(mean_d) ~= (1/n + n_test/n_train) * s_d^2.
    # Validation records are not included in n_train: only classifier fitting
    # members are counted, so the correction is conservative when validation
    # also influenced hyperparameter selection.
    variance_factor = 1.0 / n + float(test_size) / float(train_size)
    se = float(np.sqrt(variance_factor) * std)
    half_width = float(student_t.ppf(.975, n - 1) * se)
    result.update({
        'standard_error_corrected': se,
        'p_value': float(2.0 * student_t.sf(abs(mean / se), n - 1)),
        'ci95_low': mean - half_width,
        'ci95_high': mean + half_width,
        'variance_correction_factor': variance_factor,
    })
    return result
