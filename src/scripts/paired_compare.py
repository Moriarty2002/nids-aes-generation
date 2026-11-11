#!/usr/bin/env python3
"""Fail-closed paired comparison of two frozen NIDS attack-evaluation runs.

Usage:
 python src/scripts/paired_compare.py RUN_A RUN_B --out paired.json

Both runs must have a user-certified identical comparable_budget_id in their
resolved evaluation configurations. Attack seed equality alone never qualifies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from evaluator.statistical import (
    IndependentAcquisitionDesign, aggregate_by_seed, paired_record_delta,
    paired_wilcoxon, corrected_repeated_holdout_t,
)
from workflow.io import _strict_json_values

_COMMON_PROTOCOL_KEYS = (
    'source_revision_sha256', 'dataset_source_sha256', 'target_checkpoint_sha256',
    'surrogate_checkpoint_sha256', 'manifest_sha256',
    'selected_test_ids_sha256', 'class_mapping_sha256',
    'physical_transformation_sha256', 'attack_access', 'strict_constraints',
    'target_label', 'sequence_policy', 'threat_model_id',
    'budget_profile', 'comparable_budget_id',
    'final_budget_policy_version', 'padding_occupancy_policy_version',
    'query_accounting_policy_version',
    'classification_metric_policy_version', 'split_policy_version',
)


def _seed_reports(root: Path) -> dict[int, Path]:
    parent = root / 'per_seed'
    result = {}
    for path in sorted(parent.glob('evaluate_seed_*.json')):
        try:
            seed = int(path.stem.split('_')[-1])
        except ValueError:
            continue
        result[seed] = path
    return result


def _defense_seed_outcome(seed: int, lhs, rhs, target_label: int,
                          *, replicates: int, bootstrap_seed: int) -> dict:
    """Paired full-test defense impact; each target has its own ASR eligibility."""
    y = lhs['true_labels']
    accepted_a, accepted_b = lhs['accepted_mask'].astype(bool), rhs['accepted_mask'].astype(bool)
    clean_a, clean_b = lhs['clean_predictions'], rhs['clean_predictions']
    defended_a = lhs['constraint_gated_defended_predictions']
    defended_b = rhs['constraint_gated_defended_predictions']
    n = len(y)
    if not n:
        raise ValueError('Defense comparison requires a nonempty full-test population')
    for prediction in (clean_a, clean_b, defended_a, defended_b):
        if prediction.shape != (n,) or np.any(prediction < 0):
            raise ValueError('Defense comparison has missing or invalid full-test predictions')
    eligible_a = (clean_a == y) & (y != target_label)
    eligible_b = (clean_b == y) & (y != target_label)
    successes_a = eligible_a & accepted_a & (defended_a == target_label)
    successes_b = eligible_b & accepted_b & (defended_b == target_label)
    asr_a = float(successes_a.sum() / eligible_a.sum()) if eligible_a.any() else float('nan')
    asr_b = float(successes_b.sum() / eligible_b.sum()) if eligible_b.any() else float('nan')
    c_a, c_b = float(np.mean(clean_a == y)), float(np.mean(clean_b == y))
    d_a, d_b = float(np.mean(defended_a == y)), float(np.mean(defended_b == y))
    groups = lhs['group_ids']
    group_ids = groups if len(groups) == n else None
    paired = paired_record_delta(
        lhs['sample_ids'], (defended_a == y).astype(float),
        rhs['sample_ids'], (defended_b == y).astype(float),
        group_ids=group_ids, replicates=replicates, seed=bootstrap_seed + seed,
    )
    return {
        'seed': seed,
        'clean_accuracy_left': c_a, 'clean_accuracy_right': c_b,
        'defended_accuracy_left': d_a, 'defended_accuracy_right': d_b,
        'delta_clean_accuracy': c_b - c_a,
        'delta_defended_accuracy': d_b - d_a,
        'asr_left': asr_a, 'asr_right': asr_b,
        'delta_feasible_asr': asr_b - asr_a,
        'eligible_left': int(eligible_a.sum()),
        'eligible_right': int(eligible_b.sum()),
        'accepted_eligible_left': int((eligible_a & accepted_a).sum()),
        'accepted_eligible_right': int((eligible_b & accepted_b).sum()),
        'ci95_low': paired['ci95_low'], 'ci95_high': paired['ci95_high'],
        'n_groups': paired['n_groups'],
        'interval_status': paired['interval_status'],
    }


def _validated_repeated_holdouts(
    directory: Path, per_seed: list[dict], source_hashes: dict[int, str],
    split_seed_by_run: dict[int, int], sample_ids_by_run: dict[int, tuple[str, ...]],
) -> tuple[int, int]:
    """Check actual held-out IDs and physical corpus before corrected resampled t.

    This option supports *ungrouped*, repeated random research holdouts of the
    same corpus, not capture-group splits or arbitrary K-fold experiments.
    """
    if len(per_seed) < 5:
        raise ValueError('Corrected repeated-holdout inference requires >=5 paired splits')
    registry_path = directory.parent / 'experiment_manifest.json'
    if not registry_path.is_file():
        raise ValueError('Corrected holdout requires the registered experiment split manifest')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    registered_splits = registry.get('splits', {})
    sizes: set[tuple[int, int]] = set()
    complete_identities = set()
    corpus_memberships = set()
    test_memberships = set()
    for row in per_seed:
        seed = int(row['seed'])
        split_seed = split_seed_by_run[seed]
        path = directory / f'seed{split_seed}.json'
        if not path.is_file():
            raise ValueError(f'Missing registered canonical split for seed {seed}: {path}')
        split = json.loads(path.read_text(encoding='utf-8'))
        registered = registered_splits.get(str(split_seed), {})
        canonical_bytes = json.dumps(
            split, sort_keys=True, separators=(',', ':'), allow_nan=False,
        ).encode('utf-8')
        if (registered.get('path') != path.relative_to(directory.parent).as_posix()
                or registered.get('sha256') != hashlib.sha256(canonical_bytes).hexdigest()):
            raise ValueError('Canonical split bytes are not registered by the experiment manifest')
        if int(split['seed']) != split_seed:
            raise ValueError('Canonical split seed disagrees with evaluation protocol')
        identity = split['physical_identity']
        if identity.get('source_sha256') != source_hashes[seed]:
            raise ValueError('Canonical split source differs from evaluated source')
        if identity.get('group_ids_sha256') is not None:
            raise ValueError('Corrected random-holdout t is not valid for grouped capture splits')
        complete_identities.add(json.dumps(identity, sort_keys=True))
        train = tuple(map(str, split['train_ids']))
        valid = tuple(map(str, split['validation_ids']))
        test = tuple(map(str, split['test_ids']))
        if not train or not test or len(set(train + valid + test)) != len(train + valid + test):
            raise ValueError('Canonical split contains overlapping or empty partitions')
        if test != sample_ids_by_run[seed]:
            raise ValueError('Corrected holdout requires the complete ordered test population')
        sizes.add((len(train), len(test)))
        corpus_memberships.add(frozenset(train + valid + test))
        test_memberships.add(test)
    if (len(complete_identities) != 1 or len(corpus_memberships) != 1
            or len(sizes) != 1 or len(test_memberships) < 2):
        raise ValueError('Corrected holdout requires the same physical corpus, fixed '
                         'full corpus membership, train/test sizes and distinct random test partitions')
    return next(iter(sizes))


def compare_run_dirs(
    left: Path, right: Path, *, replicates: int = 2000,
    bootstrap_seed: int = 0,
    independent_design: IndependentAcquisitionDesign | None = None,
    mode: str = 'attack',
    resampled_splits_dir: Path | None = None,
) -> dict:
    if mode not in {'attack', 'defense'}:
        raise ValueError('Comparison mode must be attack or defense')
    if independent_design is not None and resampled_splits_dir is not None:
        raise ValueError('Choose either independent-acquisition or corrected repeated-holdout inference')
    left, right = Path(left), Path(right)
    left_reports, right_reports = _seed_reports(left), _seed_reports(right)
    if set(left_reports) != set(right_reports) or not left_reports:
        raise ValueError('Run directories must have the same nonempty seed set')
    per_seed = []
    evaluation_is_strict: bool | None = None
    observed_sources: dict[int, str] = {}
    seen_independent_ids: set[str] = set()
    split_seed_by_run: dict[int, int] = {}
    sample_ids_by_run: dict[int, tuple[str, ...]] = {}
    for seed in sorted(left_reports):
        left_report = json.loads(left_reports[seed].read_text(encoding='utf-8'))
        right_report = json.loads(right_reports[seed].read_text(encoding='utf-8'))
        a, b = left_report['protocol'], right_report['protocol']
        common_keys = (_COMMON_PROTOCOL_KEYS if mode == 'attack' else
                       tuple(key for key in _COMMON_PROTOCOL_KEYS
                             if key != 'target_checkpoint_sha256'))
        for key in common_keys:
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f'Cannot pair seed {seed}: protocol {key} differs or is missing')
        if mode == 'defense':
            if left_report.get('target_model') != right_report.get('target_model'):
                raise ValueError('Controlled defense comparison requires the same target architecture')
            for key in ('attack_name', 'target_defense_mode', 'target_architecture_params_sha256', 'final_hard_budget'):
                if key not in a or key not in b:
                    raise ValueError(f'Defense comparison requires protocol {key}')
            if a['target_architecture_params_sha256'] != b['target_architecture_params_sha256']:
                raise ValueError('Defense comparison requires matching target architecture hyperparameters')
            if a['attack_name'] != b['attack_name'] or a['final_hard_budget'] != b['final_hard_budget']:
                raise ValueError('Defense comparison requires the same attack and mathematical budget')
        observed_sources[seed] = str(a['dataset_source_sha256'])
        if evaluation_is_strict is None:
            evaluation_is_strict = a['strict_constraints'] is True
        elif evaluation_is_strict != (a['strict_constraints'] is True):
            raise ValueError('Mixed strict and non-strict seeds cannot be paired')
        if resampled_splits_dir is not None:
            if a['strict_constraints'] is not True or a['budget_profile'] != 'research':
                raise ValueError('Corrected repeated holdout inference requires strict research evaluations')
            split_roles = a.get('seed_roles')
            if not isinstance(split_roles, dict) or 'split' not in split_roles:
                raise ValueError('Missing actual split seed in evaluation provenance')
            split_seed_by_run[seed] = int(split_roles['split'])
        if independent_design is not None and a['strict_constraints'] is not True:
            raise ValueError('Inferential SOTA reporting requires strict feasibility admission')
        if not isinstance(a['comparable_budget_id'], str) or not a['comparable_budget_id']:
            raise ValueError('A nonempty matching comparable_budget_id must certify comparable attack budgets')
        with (np.load(left / 'per_seed' / f'matched_outcomes_seed{seed}.npz', allow_pickle=False) as lhs,
              np.load(right / 'per_seed' / f'matched_outcomes_seed{seed}.npz', allow_pickle=False) as rhs):
            keys = (('sample_ids', 'true_labels', 'clean_predictions', 'group_ids')
                    if mode == 'attack' else ('sample_ids', 'true_labels', 'group_ids'))
            for key in keys:
                if not np.array_equal(lhs[key], rhs[key]):
                    raise ValueError(f'Cannot pair seed {seed}: matched record {key} differs')
            ids = lhs['sample_ids']
            if resampled_splits_dir is not None:
                sample_ids_by_run[seed] = tuple(map(str, ids.tolist()))
            if independent_design is not None:
                # Independent acquisitions cannot recycle the same evaluated flow IDs.
                overlap = seen_independent_ids.intersection(map(str, ids.tolist()))
                if overlap:
                    raise ValueError('Independent-acquisition analysis contains overlapping test sample IDs')
                seen_independent_ids.update(map(str, ids.tolist()))
            y = lhs['true_labels']
            clean = lhs['clean_predictions']
            if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
                raise ValueError('Samples must have unique, nonempty held-out IDs')
            target_label = int(a['target_label'])
            if mode == 'defense':
                if evaluation_is_strict and (
                    not np.array_equal(lhs['scored_mask'], lhs['accepted_mask'])
                    or not np.array_equal(rhs['scored_mask'], rhs['accepted_mask'])
                ):
                    raise ValueError('Strict defense comparisons cannot score rejected candidates')
                per_seed.append(_defense_seed_outcome(
                    seed, lhs, rhs, target_label,
                    replicates=replicates, bootstrap_seed=bootstrap_seed,
                ))
                continue
            eligible = (clean == y) & (y != target_label)
            aa = lhs['accepted_mask'].astype(bool)
            bb = rhs['accepted_mask'].astype(bool)
            if not np.any(eligible):
                per_seed.append({'seed':seed,'asr_left':float('nan'), 'asr_right':float('nan'),
                                 'delta_asr':float('nan'), 'eligible':0, 'ci95_low':float('nan'),
                                 'ci95_high':float('nan'), 'n_groups':0})
                continue
            scored_a = lhs['scored_mask'].astype(bool)
            scored_b = rhs['scored_mask'].astype(bool)
            if evaluation_is_strict and (
                    not np.array_equal(scored_a, aa) or not np.array_equal(scored_b, bb)):
                raise ValueError('Strict paired inference requires strict constraint admission')
            success_a = (aa & (lhs['constraint_gated_defended_predictions'] == target_label))[eligible].astype(float)
            success_b = (bb & (rhs['constraint_gated_defended_predictions'] == target_label))[eligible].astype(float)
            groups = lhs['group_ids']
            paired_groups = groups[eligible] if len(groups) == len(ids) else None
            estimate = paired_record_delta(
                ids[eligible], success_a, ids[eligible], success_b,
                group_ids=paired_groups, replicates=replicates, seed=bootstrap_seed+seed,
            )
            # In the absence of at least two documented capture groups, an
            # IID flow bootstrap cannot certify capture-dependent precision.
            conditional_group_ci = (paired_groups is not None and
                                    len(np.unique(paired_groups)) >= 2)
            if not conditional_group_ci:
                estimate['ci95_low'] = float('nan')
                estimate['ci95_high'] = float('nan')
            per_seed.append({
                'seed':seed, 'asr_left':float(success_a.mean()), 'asr_right':float(success_b.mean()),
                'delta_asr':float(success_b.mean()-success_a.mean()),
                'eligible':int(eligible.sum()), 'ci95_low':estimate['ci95_low'],
                'ci95_high':estimate['ci95_high'], 'n_groups':estimate['n_groups'],
                'interval_status': ('conditional_on_declared_capture_groups' if conditional_group_ci
                                    else 'undefined_without_at_least_two_capture_groups'),
            })
    if independent_design is not None:
        independent_design.verify(sorted(observed_sources))
        for seed, source in observed_sources.items():
            if source != independent_design.source_sha256[seed]:
                raise ValueError(f'Independent acquisition source hash mismatch for seed {seed}')
    primary = 'delta_asr' if mode == 'attack' else 'delta_defended_accuracy'
    metric_names = ((primary,) if mode == 'attack' else
                    ('delta_clean_accuracy', 'delta_defended_accuracy', 'delta_feasible_asr'))
    aggregation_rows = [
        {'seed': r['seed'], 'dataset_source_sha256': observed_sources[r['seed']],
         **{key: r[key] for key in metric_names}}
        for r in per_seed
    ]
    aggregated = aggregate_by_seed(aggregation_rows, independent_design=independent_design)
    descriptive = (aggregated[primary].as_dict(primary) if mode == 'attack'
                   else {key: aggregated[key].as_dict(key) for key in metric_names})
    inference: dict[str, object] = {
        'status': ('descriptive_repeated_splits' if evaluation_is_strict
                   else 'non_strict_diagnostic_no_inferential_claim'),
        'wilcoxon_p_value': float('nan'), 'p_value_holm': float('nan'),
        'n_paired_units': 0,
    }
    if independent_design is not None:
        valid = [r for r in per_seed if np.isfinite(r[primary])]
        if len(valid) >= 5:
            left_metric = 'asr_left' if mode == 'attack' else 'defended_accuracy_left'
            right_metric = 'asr_right' if mode == 'attack' else 'defended_accuracy_right'
            p_value, nonzero = paired_wilcoxon(
                [r[left_metric] for r in valid], [r[right_metric] for r in valid]
            )
            inference = {
                'status': 'conditional_on_declared_independent_acquisitions',
                'wilcoxon_p_value': p_value,
                'p_value_holm': p_value,  # One prespecified primary hypothesis.
                'n_paired_units': len(valid), 'n_nonzero_pairs': nonzero,
                'evidence': independent_design.evidence,
                'test': 'paired_two_sided_wilcoxon',
                'primary_metric': primary,
            }
    elif resampled_splits_dir is not None:
        train_size, test_size = _validated_repeated_holdouts(
            Path(resampled_splits_dir), per_seed, observed_sources,
            split_seed_by_run, sample_ids_by_run,
        )
        selected = [r[primary] for r in per_seed]
        if not all(np.isfinite(x) for x in selected):
            raise ValueError('Corrected repeated-holdout inference needs a defined '
                             'primary metric for every split')
        corrected = corrected_repeated_holdout_t(
            selected, train_size=train_size, test_size=test_size,
        )
        inference = {
            'status': corrected['inference_status'],
            'test': corrected['method'], 'primary_metric': primary,
            'p_value': corrected['p_value'], 'n_paired_units': len(selected),
            'wilcoxon_p_value': float('nan'), 'p_value_holm': corrected['p_value'],
            'train_size': train_size, 'test_size': test_size,
            'note': ('Approximate Nadeau--Bengio random-subsampling correction; '
                     'conditional on the same IID corpus and independently drawn '
                     'random partitions. Not a new-capture generalization claim.'),
        }
        if mode == 'attack':
            descriptive[f'{primary}_ci95_low'] = corrected['ci95_low']
            descriptive[f'{primary}_ci95_high'] = corrected['ci95_high']
        else:
            descriptive[primary][f'{primary}_ci95_low'] = corrected['ci95_low']
            descriptive[primary][f'{primary}_ci95_high'] = corrected['ci95_high']
            descriptive[primary][f'{primary}_inference_status'] = corrected['inference_status']
    return {
        'mode': mode,
        'status': (
            ('verified_paired' if mode == 'attack' else 'verified_paired_defense')
            if evaluation_is_strict else
            ('paired_non_strict_diagnostic_not_sota_certified' if mode == 'attack'
             else 'paired_non_strict_defense_diagnostic_not_sota_certified')
        ),
        'direction': 'right_minus_left',
        'uncertainty': (
            'Within-seed paired capture-group intervals are conditional on declared groups; '
            'across-seed inference is disabled by default for overlapping resamples. '
            'An explicitly validated corrected repeated-holdout t is approximate '
            'and not an independent-capture claim.'
        ),
        'left_run': str(left), 'right_run': str(right), 'per_seed': per_seed,
        'inference': inference, 'across_seed_delta': descriptive,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--mode', choices=('attack', 'defense'), default='attack')
    parser.add_argument(
        '--resampled-splits-dir', type=Path,
        help='Opt-in approximate corrected t-test for verified ungrouped random research holdouts; '
             'pass experiment_dir/splits. Cannot combine with independent acquisitions.',
    )
    parser.add_argument(
        '--independent-acquisitions-manifest', type=Path,
        help='Optional predeclared independent-acquisition evidence. No inferential '
             'CI/p-value is computed for ordinary overlapping repeated splits.',
    )
    options = parser.parse_args()
    design = None
    if options.independent_acquisitions_manifest is not None:
        declaration = json.loads(options.independent_acquisitions_manifest.read_text(encoding='utf-8'))
        units = declaration['units']
        design = IndependentAcquisitionDesign(
            unit_ids={int(seed): str(value['unit_id']) for seed, value in units.items()},
            source_sha256={int(seed): str(value['source_sha256']) for seed, value in units.items()},
            evidence=str(declaration['evidence']),
        )
    report = compare_run_dirs(
        options.left, options.right, replicates=options.replicates,
        bootstrap_seed=options.bootstrap_seed, independent_design=design,
        mode=options.mode, resampled_splits_dir=options.resampled_splits_dir,
    )
    serialized = json.dumps(_strict_json_values(report), indent=2, sort_keys=True, allow_nan=False)
    if options.out:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)


if __name__ == '__main__':
    main()
