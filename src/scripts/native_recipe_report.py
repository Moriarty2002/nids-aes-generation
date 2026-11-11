#!/usr/bin/env python3
"""Read-only, descriptive comparison of *native-configured* attack recipes.

Unlike paired_compare.py, this tool does NOT demand a shared epsilon/query
budget, and it NEVER certifies equal-resource algorithms or inferential p-values.
It requires genuine research reports with the same registered split, frozen
target and ordered held-out IDs *within* each seed. No checkpoints are loaded,
models are trained, or experimental results are changed.

Example:
  python src/scripts/native_recipe_report.py --experiment-dir RUNS/EXPERIMENT \
    --run pgd=RUNS/PGD_EXECUTION --run fgsm=RUNS/FGSM_EXECUTION \
    --out native_comparison.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from core.provenance import _digest_strings
from workflow.io import _strict_json_values

# These properties must be identical for an attack-recipe comparison of a
# common frozen target. Attack-specific normalization/budget/queries differ.
_MATCHED_PROTOCOL = (
    'source_revision_sha256', 'experiment_id', 'dataset_source_sha256',
    'target_checkpoint_sha256', 'manifest_sha256', 'selected_test_ids_sha256',
    'class_mapping_sha256', 'physical_transformation_sha256',
    'target_architecture_params_sha256', 'target_label', 'sequence_policy',
    'threat_model_id', 'final_budget_policy_version',
    'padding_occupancy_policy_version', 'query_accounting_policy_version',
    'classification_metric_policy_version', 'split_policy_version',
    'strict_constraints', 'budget_profile',
)


def _method_publication_evidence(method_name: str) -> dict[str, str]:
    """Read the versioned primary-method inventory; do not infer fidelity from an ASR.

    Evidence status is distinct from experiment role. A functioning attack
    may be useful in the native descriptive report without being a verified
    original method, and the reporter never retrains or repeats any inference.
    """
    inventory = Path(__file__).resolve().parents[1] / 'attacks' / 'PRIMARY_METHOD_FIDELITY.csv'
    with inventory.open(newline='', encoding='utf-8') as stream:
        matches = [row for row in csv.DictReader(stream) if row['method'] == method_name]
    if len(matches) != 1:
        raise ValueError(f'Method-fidelity inventory missing/ambiguous for {method_name}')
    item = matches[0]
    return {'publication_role': item['publication_role'],
            'source_fidelity_status': item.get('source_fidelity_status') or 'source_fidelity_unverified'}


def _publication_role(method_name: str) -> str:
    """Return a method's scientific role without claiming original parity."""
    return _method_publication_evidence(method_name)['publication_role']


_ADMITTED_SOURCE_STATUSES = frozenset({'paper_core_verified', 'source_supported_nids_adaptation'})


def _paper_admitted(arm: Mapping[str, object]) -> bool:
    status = arm.get('source_fidelity_status')
    # Direct legacy callers passing only publication_role retain their old
    # admission-view interface; registered comparison arms always carry status.
    return (arm['publication_role'] == 'primary_evasion'
            and (status is None or status in _ADMITTED_SOURCE_STATUSES))


def _primary_original_attack_view(arms: Mapping[str, dict], labels: Sequence[str]) -> dict:
    """Give an explicit, evidence-filtered *view*, never an equal-budget ranking.

    The full native descriptive attack comparison still reports every selected
    completed research arm, including custom or different-purpose methods.
    """
    included = [label for label in labels if _paper_admitted(arms[label])]
    excluded = {label: arms[label]['publication_role'] for label in labels
                if not _paper_admitted(arms[label])}
    return {'included_arm_labels': included, 'excluded_arms': excluded,
            'ranking_emitted': False,
            'status': ('two_or_more_eligible_native_recipes_descriptive_only'
                       if len(included) >= 2 else 'insufficient_primary_arms'),
            'note': 'Admission is not original-paper equivalence, equal-budget ranking, or SOTA certification.'}


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f'Required research analysis input missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _registered_split(experiment_dir: Path, registry: dict, split_seed: int) -> dict:
    entry = registry.get('splits', {}).get(str(split_seed))
    if not entry:
        raise ValueError(f'Missing registered canonical split seed {split_seed}')
    # Exact relative paths are part of the existing experiment registry.
    path = (experiment_dir / entry['path']).resolve()
    if not path.is_relative_to(experiment_dir.resolve()):
        raise ValueError('Registered split path escapes the selected experiment')
    split = _read_json(path)
    digest = hashlib.sha256(json.dumps(
        split, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')).hexdigest()
    if digest != entry['sha256'] or int(split['seed']) != split_seed:
        raise ValueError('Canonical split membership is not registered or has changed')
    return split


def _required_arrays(path: Path) -> dict[str, np.ndarray]:
    required = ('sample_ids', 'true_labels', 'clean_predictions',
                'constraint_gated_defended_predictions', 'accepted_mask', 'scored_mask')
    if not path.is_file():
        raise ValueError(f'Missing per-record matched outcomes: {path}')
    with np.load(path, allow_pickle=False) as result:
        if any(key not in result.files for key in required):
            raise ValueError(f'Matched outcomes are incomplete: {path}')
        return {key: np.asarray(result[key]).copy() for key in required}



def _binary_operational_metrics(y: np.ndarray, clean: np.ndarray,
                                gated: np.ndarray, benign_label: int,
                                saved_metrics: dict) -> dict:
    """Binary malicious operational indicators on the *full* matched test set.

    The saved protocol target_label is the benign class ID by the explicit
    workflow/evaluate.py producer contract, not an attack plugin's objective.
    Wrong malicious-family predictions are binary detections, not evasion.
    """
    malicious = y != benign_label
    total_malicious = int(np.count_nonzero(malicious))
    total_benign = len(y) - total_malicious

    def rates(pred):
        if not total_malicious:
            return None, None
        detected = int(np.count_nonzero(malicious & (pred != benign_label)))
        return detected / total_malicious, (total_malicious - detected) / total_malicious

    clean_dr, clean_fnr = rates(clean)
    gated_dr, gated_fnr = rates(gated)

    def finite(key):
        value = saved_metrics.get(key)
        return float(value) if isinstance(value, (float, int)) and np.isfinite(value) else None

    clean_mcc = finite('full_test_clean_mcc')
    gated_mcc = finite('full_test_constraint_gated_defended_mcc')
    return {
        'full_test_malicious_support': total_malicious,
        'full_test_benign_support': total_benign,
        'full_test_clean_malicious_detection_rate': clean_dr,
        'full_test_clean_malicious_false_negative_rate': clean_fnr,
        'full_test_constraint_gated_defended_malicious_detection_rate': gated_dr,
        'full_test_constraint_gated_defended_malicious_false_negative_rate': gated_fnr,
        'full_test_clean_to_gated_accuracy_drop': (float(np.mean(clean == y))
                                                   - float(np.mean(gated == y))),
        'full_test_clean_to_gated_mcc_drop': (clean_mcc - gated_mcc
                                              if clean_mcc is not None and gated_mcc is not None
                                              else None),
    }


def _seed_outcome(report: dict, arrays: dict[str, np.ndarray], seed: int) -> dict:
    """Reconstruct strict benign-evasion outcomes from matched saved predictions.

    Eligible initially correct malicious attempts remain in feasible ASR even
    when rejected; accepted eligible attempts define a separate conditional
    denominator. Never rescore rejected candidates.
    """
    protocol = report['protocol']
    n = len(arrays['sample_ids'])
    for key, value in arrays.items():
        if value.shape != (n,):
            raise ValueError(f'Seed {seed}: nonaligned per-record array {key}')
    if n < 1 or len(set(map(str, arrays['sample_ids']))) != n:
        raise ValueError(f'Seed {seed}: empty or duplicate held-out record IDs')
    if arrays['accepted_mask'].dtype != np.bool_ or arrays['scored_mask'].dtype != np.bool_:
        raise ValueError('Strict acceptance/scoring masks must be Boolean')
    if not np.array_equal(arrays['accepted_mask'], arrays['scored_mask']):
        raise ValueError('Strict run scored independently rejected adversarial candidates')
    y = arrays['true_labels']
    clean = arrays['clean_predictions']
    defended = arrays['constraint_gated_defended_predictions']
    if any(np.any(a < 0) for a in (y, clean, defended)):
        raise ValueError('Full-test true/clean/defended classes cannot be missing')
    accepted = arrays['accepted_mask']
    if not np.array_equal(defended[~accepted], clean[~accepted]):
        raise ValueError('Rejected candidate must use the clean prediction in gated full-test metrics')
    target_label = int(protocol['target_label'])
    eligible = (y != target_label) & (clean == y)
    success = eligible & accepted & (defended == target_label)
    a = int(eligible.sum())
    b = int((eligible & accepted).sum())
    s = int(success.sum())
    metrics = report.get('metrics') or {}
    for key, actual in (
        ('full_test_sample_count', n), ('attack_eligible_sample_count', a),
        ('feasible_eligible_sample_count', b), ('successful_sample_count', s),
    ):
        if key in metrics and int(metrics[key]) != actual:
            raise ValueError(f'Seed {seed}: evaluator metric {key} disagrees with aligned records')
    primary = s / a if a else None
    conditional = s / b if b else None
    for key, expected in (("scored_sample_count", int(arrays['scored_mask'].sum())),
                          ("rejected_sample_count", int((~accepted).sum())),
                          ("accepted_subset_sample_count", int(accepted.sum()))):
        if key in metrics and metrics[key] is not None and int(metrics[key]) != expected:
            raise ValueError(f'Seed {seed}: saved {key} disagrees with actual gatekeeper mask')
    for key, values in (
        ('asr_eligible_count_class_', eligible),
        ('asr_feasible_eligible_count_class_', eligible & accepted),
        ('asr_success_count_class_', success),
    ):
        for metric, saved in metrics.items():
            if metric.startswith(key) and metric[len(key):].isdigit():
                actual = int(np.count_nonzero(values & (y == int(metric[len(key):]))))
                if saved is None or int(saved) != actual:
                    raise ValueError(f'Seed {seed}: saved class ASR support {metric} differs from records')
    if metrics.get('asr') is not None and (primary is None or
            not np.isclose(float(metrics['asr']), primary, atol=1e-12, rtol=0)):
        raise ValueError(f'Seed {seed}: saved feasible ASR disagrees with eligible denominator')
    # Expose all existing evaluator scalar metrics verbatim (with non-finite
    # measurements serialized as null), never infer unrecorded generation cost.
    observed = _strict_json_values(metrics)
    for key, value in (("asr_conditional_on_feasibility", conditional),
                       ("realizability_rate_eligible", b / a if a else None),
                       ("realizability_rate", float(accepted.mean()))):
        saved = observed.get(key)
        if saved is not None and (value is None or not np.isclose(float(saved), value,
                                                         atol=1e-12, rtol=0)):
            raise ValueError(f"Seed {seed}: evaluator {key} disagrees with aligned outcomes")
    # The evaluation producer stores its resolved benign_label as protocol.target_label;
    # plugin targetedness is separate and cannot redefine the true benign class.
    derived = _binary_operational_metrics(y, clean, defended, target_label, observed)
    return {
        'derived_operational_metrics': derived, **derived,
        'seed': seed, 'test_count': n, 'malicious_true': int(np.count_nonzero(y != target_label)),
        'malicious_true_count': int(np.count_nonzero(y != target_label)),
        'eligible': a, 'eligible_count': a, 'accepted': int(accepted.sum()),
        'accepted_all_count': int(accepted.sum()), 'rejected': int((~accepted).sum()),
        'rejected_all_count': int((~accepted).sum()), 'scored_count': int(arrays['scored_mask'].sum()),
        'accepted_eligible': b, 'accepted_eligible_count': b,
        'successes': s, 'success_count': s, 'asr': primary, 'feasible_asr': primary,
        'conditional_asr': conditional, 'conditional_feasible_asr': conditional,
        'realizability': float(accepted.mean()),
        'realizability_rate_all_test': float(accepted.mean()),
        'realizability_rate_eligible': b / a if a else None,
        'full_test_clean_accuracy': float(np.mean(clean == y)),
        'full_test_constraint_gated_defended_accuracy': float(np.mean(defended == y)),
        'reported_metrics': observed,
    }


# ---------------------------------------------------------------------------
# Registered, completed execution selection (read-only; no ExperimentStore.read
# because a reporter must remain usable on an intact relocated old experiment).
# ---------------------------------------------------------------------------

def _seeds(expected_seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(int(s) for s in expected_seeds))
    if not values or len(set(values)) != len(values):
        raise ValueError('Research seed set must be nonempty and contain no duplicates')
    return values


def _within(experiment_dir: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('Registered execution path must be relative and stay inside experiment')
    full = (experiment_dir / candidate).resolve()
    if not full.is_relative_to(experiment_dir.resolve()):
        raise ValueError('Registered execution path escapes selected experiment')
    return full


def _registered_run(root: Path, registry: dict, path: Path,
                    seeds: tuple[int, ...]) -> tuple[str, dict]:
    directory = Path(path).resolve()
    if not directory.is_relative_to(root.resolve()):
        raise ValueError(f'Unregistered or out-of-experiment execution: {directory}')
    found = [(key, entry) for key, entry in registry.get('executions', {}).items()
             if _within(root, entry.get('path', '')) == directory]
    if len(found) != 1:
        raise ValueError(f'Execution is not uniquely registered: {directory}')
    execution_id, entry = found[0]
    if (entry.get('action') != 'evaluate_attack' or entry.get('status') != 'completed'
            or tuple(sorted(map(int, entry.get('seeds', [])))) != seeds
            or len(entry.get('seeds', [])) != len(seeds)):
        raise ValueError(f'Execution {execution_id} must be completed evaluate_attack with exact seeds {seeds}')
    return execution_id, entry


def select_completed_executions(experiment_dir: Path, *,
                                expected_seeds: Sequence[int] = (1, 2, 3, 4, 5)) -> list[dict]:
    """Discover only research-configured, completed, registered evaluations.

    No date-based selection; duplicate logical arms are checked at comparison
    grouping time. Non-evaluation/unfinished entries are never candidates.
    """
    root = Path(experiment_dir).resolve()
    registry = _read_json(root / 'experiment_manifest.json')
    seeds = _seeds(expected_seeds)
    selections = []
    for execution_id, entry in sorted(registry.get('executions', {}).items()):
        if entry.get('action') != 'evaluate_attack' or entry.get('status') != 'completed':
            continue
        if tuple(sorted(map(int, entry.get('seeds', [])))) != seeds or len(entry.get('seeds', [])) != len(seeds):
            # Explicitly preserve incomplete/mismatched runs as ineligible.
            continue
        directory = _within(root, entry['path'])
        config = _read_json(directory / 'resolved_config.json')
        if (config.get('budget', {}).get('name') != 'research'
                or config.get('evaluation', {}).get('strict_constraints') is not True
                or config.get('evaluation', {}).get('max_samples') is not None):
            continue
        selections.append({'execution_id': execution_id, 'path': directory})
    return selections


def _load_run(label: str, directory: Path, root: Path, registry: dict,
              seeds: tuple[int, ...]) -> dict:
    execution_id, _ = _registered_run(root, registry, directory, seeds)
    config = _read_json(directory / 'resolved_config.json')
    if (config.get('budget', {}).get('name') != 'research'
            or config.get('evaluation', {}).get('strict_constraints') is not True
            or config.get('evaluation', {}).get('max_samples') is not None
            or tuple(sorted(map(int, config.get('seeds', [])))) != seeds
            or len(config.get('seeds', [])) != len(seeds)):
        raise ValueError(f'Run {label}: only complete strict research configuration/seeds are admissible')
    # A historical absolute experiment_dir is a locator, not experiment identity.
    if not config.get('experiment_dir'):
        raise ValueError(f'Run {label}: resolved experiment locator missing')
    paths = set((directory / 'per_seed').glob('evaluate_seed_*.json'))
    wanted = {directory / 'per_seed' / f'evaluate_seed_{seed}.json' for seed in seeds}
    if paths != wanted:
        raise ValueError(f'Run {label}: incomplete or unexpected per-seed evaluation report set')
    reports = {}
    for seed in seeds:
        report = _read_json(directory / 'per_seed' / f'evaluate_seed_{seed}.json')
        p = report.get('protocol', {})
        if (int(report.get('seed', -1)) != seed
                or p.get('experiment_id') != registry.get('experiment_id')
                or p.get('source_revision_sha256') != registry.get('source_revision_sha256')
                or p.get('budget_profile') != 'research'
                or p.get('strict_constraints') is not True
                or p.get('evaluation_status') != 'strict_feasible_benchmark'
                or p.get('attack_name') != config.get('attack', {}).get('name')):
            raise ValueError(f'Run {label} seed {seed}: incomplete, non-strict or mismatched research protocol')
        model_config = config.get('model') or {}
        if model_config.get('name') is not None and model_config.get('name') != report.get('target_model'):
            raise ValueError(f'Run {label} seed {seed}: resolved model differs from reported target architecture')
        requested_training = (model_config.get('training') or {}).get('defense_mode')
        if requested_training is not None and requested_training != p.get('target_defense_mode', 'none'):
            raise ValueError(f'Run {label} seed {seed}: configured training defense differs from fitted target')
        if 'target_label' not in p or int(p['target_label']) != int(config.get('evaluation', {}).get('benign_label', 0)):
            raise ValueError(f'Run {label} seed {seed}: saved benign-label protocol differs from resolved evaluation')
        roles = p.get('seed_roles', {})
        if (int(roles.get('worker_replica', -1)) != seed or 'attack' not in roles
                or roles['attack'] is None):
            raise ValueError(f'Run {label} seed {seed}: effective worker/attack seed roles absent or inconsistent')
        if p.get('attack_access') not in {'transfer', 'model_independent'}:
            raise ValueError(f'Run {label} seed {seed}: frozen-target oracle attack access not permitted')
        if p['attack_access'] == 'transfer' and not p.get('surrogate_checkpoint_sha256'):
            raise ValueError(f'Run {label} seed {seed}: authorized surrogate checkpoint identity missing')
        if any(field not in p for field in _MATCHED_PROTOCOL):
            raise ValueError(f'Run {label} seed {seed}: scientific protocol missing mandatory identity')
        reports[seed] = report
    digest = hashlib.sha256(json.dumps(config, sort_keys=True,
                        separators=(',', ':'), default=str).encode('utf-8')).hexdigest()
    environment = (_read_json(directory / 'environment.json')
                   if (directory / 'environment.json').is_file() else None)
    return {'label': label, 'directory': directory, 'execution_id': execution_id,
            'environment': environment,
            'config': config, 'config_sha256': digest, 'reports': reports,
            'attack': config.get('attack', {}).get('name'),
            'attack_recipe_digest': hashlib.sha256(json.dumps(config.get('attack'), sort_keys=True,
                                 separators=(',', ':'), default=str).encode()).hexdigest()}


def _detector_identity(report: dict) -> str:
    detector = report['protocol'].get('detector_artifact')
    if not detector:
        return 'no_detector'
    return str(detector.get('name', 'detector')) + ':' + str(detector.get('checkpoint_sha256'))


def _defense_label(run: dict) -> str:
    p = run['reports'][next(iter(run['reports']))]['protocol']
    detector = p.get('detector_artifact')
    return (str(run['reports'][next(iter(run['reports']))].get('target_model') or
                run['config'].get('model', {}).get('name')) + '/' +
            str(p.get('target_defense_mode', 'none')) + '/' +
            (str(detector.get('name', 'detector')) if detector else 'no_detector'))


def _attack_key(run: dict, seeds: tuple[int, ...]) -> tuple:
    first = run['reports'][seeds[0]]
    return (first.get('target_model'),
            tuple(run['reports'][s]['protocol'].get('target_defense_mode') for s in seeds),
            tuple(_detector_identity(run['reports'][s]) for s in seeds),
            tuple(run['reports'][s]['protocol'].get('target_checkpoint_sha256') for s in seeds),
            tuple(run['reports'][s]['protocol'].get('target_architecture_params_sha256') for s in seeds))


def _defense_key(run: dict, seeds: tuple[int, ...]) -> tuple:
    return (run['attack'], run['attack_recipe_digest'],
            tuple(run['reports'][s].get('target_model') for s in seeds),
            tuple(run['reports'][s]['protocol'].get('target_architecture_params_sha256') for s in seeds))


def _population_metrics(report: dict) -> dict:
    # The evaluator already computed all classification, class-support, detector,
    # query, physical-distance and constraint details. Never recompute them here.
    return _strict_json_values(report.get('metrics') or {})


def _validated_seed(run: dict, seed: int, root: Path, registry: dict) -> tuple[dict, dict]:
    report = run['reports'][seed]
    p = report['protocol']
    roles = p['seed_roles']
    split_seed = int(roles.get('split', -1))
    canonical = _registered_split(root, registry, split_seed)
    if canonical['physical_identity']['source_sha256'] != p['dataset_source_sha256']:
        raise ValueError(f'Run {run["label"]} seed {seed}: registered source/split identity mismatch')
    if not p.get('target_checkpoint_sha256') or not p.get('manifest_sha256'):
        raise ValueError('Target and split hashes required for research reporting')
    if not any(entry.get('kind') == 'model'
               and entry.get('sha256') == p['target_checkpoint_sha256']
               and int(entry.get('split_seed', -1)) == split_seed
               and int(entry.get('seed', -1)) == int(roles.get('target_model', -1))
               for entry in registry.get('artifacts', {}).values()):
        raise ValueError(f'Run {run["label"]} seed {seed}: target is not registered for declared role/split')
    detector = p.get('detector_artifact')
    if detector:
        detector_sha = detector.get('checkpoint_sha256')
        if (not detector_sha or int(roles.get('detector') or -1) != int(roles.get('target_model') or -2)
                or not any(e.get('kind') == 'detector' and e.get('sha256') == detector_sha
                           and int(e.get('seed', -1)) == int(roles['detector'])
                           for e in registry.get('artifacts', {}).values())):
            raise ValueError(f'Run {run["label"]} seed {seed}: fitted detector checkpoint not registered')
    arrays = _required_arrays(run['directory'] / 'per_seed' / f'matched_outcomes_seed{seed}.npz')
    ids = tuple(map(str, arrays['sample_ids'].tolist()))
    if ids != tuple(map(str, canonical['test_ids'])) or _digest_strings(ids) != p.get('selected_test_ids_sha256'):
        raise ValueError(f'Run {run["label"]} seed {seed}: ordered held-out IDs mismatch')
    outcome = _seed_outcome(report, arrays, seed)
    outcome.update({
        'seed_roles': dict(roles),
        'target_checkpoint_sha256': p['target_checkpoint_sha256'],
        'manifest_sha256': p['manifest_sha256'],
        'selected_test_ids_sha256': p['selected_test_ids_sha256'],
        'dataset_source_sha256': p['dataset_source_sha256'],
        'surrogate_checkpoint_sha256': p.get('surrogate_checkpoint_sha256'),
        'detector_artifact': p.get('detector_artifact'),
        'final_hard_budget': p.get('final_hard_budget'),
        'budget': p.get('final_hard_budget'),
        'attack_access': p['attack_access'],
        'attack_access_description': ('surrogate-query; no frozen-target query'
             if p['attack_access'] == 'query'
             else 'authorized surrogate-gradient transfer; no frozen-target gradient'
             if p['attack_access'] == 'transfer' else 'model-independent; no frozen-target query'),
        'attack_name': p['attack_name'],
        'target_defense_mode': p.get('target_defense_mode', 'none'),
        'target_model': report.get('target_model'),
        'target_architecture_params_sha256': p['target_architecture_params_sha256'],
        'class_support_by_population': _strict_json_values(report.get('class_support_by_population')),
        'reported_metrics': _population_metrics(report),
        'offline_generator_training_cost': None,
        'generator_training_cost_observed': None,
        'execution_id': run['execution_id'],
        'resolved_config_sha256': run['config_sha256'],
        'resolved_attack': run['config']['attack'],
        'class_mapping_sha256': p.get('class_mapping_sha256'),
        'physical_transformation_sha256': p.get('physical_transformation_sha256'),
        'threat_model_id': p.get('threat_model_id'),
        'query_accounting_policy_version': p.get('query_accounting_policy_version'),
        'classification_metric_policy_version': p.get('classification_metric_policy_version'),
        'target_label': p.get('target_label'),
    })
    return outcome, arrays


def _comparison_identity(a: dict, b: dict, *, mode: str, seed: int) -> None:
    pa, pb = a['protocol'], b['protocol']
    exceptions = {'target_checkpoint_sha256'} if mode == 'defense' else set()
    different = [field for field in _MATCHED_PROTOCOL if field not in exceptions
                 and pa[field] != pb[field]]
    if different:
        raise ValueError(f'Seed {seed}: {mode} identity/policy differs: {different}')
    if int(pa['seed_roles']['attack']) != int(pb['seed_roles']['attack']):
        raise ValueError(f'Seed {seed}: attack RNG seed differs across arms: '
                         f'{pa["seed_roles"]["attack"]} versus {pb["seed_roles"]["attack"]}')
    if mode == 'defense':
        if pa['attack_name'] != pb['attack_name']:
            raise ValueError(f'Seed {seed}: defense requires identical attack recipe')
        for key in ('surrogate_checkpoint_sha256', 'attack_access', 'final_hard_budget'):
            if pa.get(key) != pb.get(key):
                raise ValueError(f'Seed {seed}: defense attack/surrogate/budget mismatch: {key}')
        if pa['seed_roles'].get('surrogate') != pb['seed_roles'].get('surrogate'):
            raise ValueError(f'Seed {seed}: defense surrogate role mismatch')


def _stats(values: Sequence[float | None]) -> dict:
    finite = np.asarray([float(v) for v in values
                         if v is not None and np.isfinite(float(v))], dtype=np.float64)
    return {'n_finite': len(finite), 'mean': float(finite.mean()) if len(finite) else None,
            'std': float(finite.std(ddof=1)) if len(finite) > 1 else None}


def _delta(a: float | None, b: float | None) -> float | None:
    return float(a - b) if a is not None and b is not None else None


def _seed_metric(o: dict, key: str) -> float | None:
    if key in o:
        value = o[key]
    else:
        value = o.get('reported_metrics', {}).get(key)
    return float(value) if isinstance(value, (float, int)) and np.isfinite(value) else None


_DEFENSE_DELTA_METRICS = ('feasible_asr_policy_specific',
    'full_test_clean_accuracy', 'full_test_constraint_gated_defended_accuracy',
    'full_test_clean_mcc', 'full_test_constraint_gated_defended_mcc',
    'full_test_clean_f1_macro', 'full_test_constraint_gated_defended_f1_macro',
    'full_test_clean_malicious_detection_rate',
    'full_test_clean_malicious_false_negative_rate',
    'full_test_constraint_gated_defended_malicious_detection_rate',
    'full_test_constraint_gated_defended_malicious_false_negative_rate',
    'full_test_clean_to_gated_accuracy_drop', 'full_test_clean_to_gated_mcc_drop')


def _compare_group(runs: list[dict], *, mode: str, root: Path, registry: dict,
                   seeds: tuple[int, ...], reference: str | None) -> dict:
    """Compare only registered research outputs with matching seed populations.

    Attack arms require the same frozen target; training-defense arms may use
    different target weights but must match held-out IDs and attack policy.
    Pairwise differences remain descriptive without independent sampling units.
    """
    if len(runs) < 2:
        raise ValueError('At least two distinct completed arms are needed for a comparison group')
    by_label = {r['label']: r for r in runs}
    if len(by_label) != len(runs):
        raise ValueError('Duplicate report arm labels; select unique executions explicitly')
    if mode == 'attack':
        native = [(r['attack'], r['attack_recipe_digest']) for r in runs]
        if len(set(native)) != len(native):
            ids = [r['execution_id'] for r in runs]
            raise ValueError(f'Duplicate completed logical attack arm: executions {ids}; select explicitly')
        if len({r['attack'] for r in runs}) != len(runs):
            raise ValueError('Different configurations of same attack name require separate, explicitly labeled studies')
        if reference is None:
            # Do not silently adopt input/insertion/latest order. A canonical
            # unambiguous PGD reference is a documented convenience, else fail.
            if 'pgd' in by_label and sum(r['attack'] == 'pgd' for r in runs) == 1:
                reference = 'pgd'
            else:
                raise ValueError('Multiple attack arms require an explicit --reference (e.g. --reference pgd)')
    else:
        if len({r['attack_recipe_digest'] for r in runs}) != 1 or len({r['attack'] for r in runs}) != 1:
            raise ValueError('Defense comparison requires identical resolved attack recipe/configuration')
        if reference is None:
            baselines = [r['label'] for r in runs if all(
                r['reports'][s]['protocol'].get('target_defense_mode', 'none') == 'none'
                and not r['reports'][s]['protocol'].get('detector_artifact') for s in seeds)]
            if len(baselines) != 1:
                raise ValueError('Defense reference is ambiguous; pass --reference explicitly')
            reference = baselines[0]
    if reference not in by_label:
        raise ValueError(f'Reference {reference!r} not present in comparison group: {sorted(by_label)}')
    ordered_labels = sorted(by_label)
    per_seed = []
    paired = []
    all_outcomes: dict[str, list[dict]] = {name: [] for name in ordered_labels}
    deltas: dict[str, list] = {}
    for seed in seeds:
        loaded = {name: _validated_seed(by_label[name], seed, root, registry)
                  for name in ordered_labels}
        reference_run = by_label[reference]
        ref_o, ref_a = loaded[reference]
        for name in ordered_labels:
            run = by_label[name]
            o, arrays = loaded[name]
            _comparison_identity(reference_run['reports'][seed], run['reports'][seed],
                                 mode=mode, seed=seed)
            checks = ('sample_ids', 'true_labels', 'clean_predictions') if mode == 'attack' else (
                'sample_ids', 'true_labels')
            for key in checks:
                if not np.array_equal(ref_a[key], arrays[key]):
                    raise ValueError(f'Seed {seed}: {mode} pairing has mismatched {key}')
            if mode == 'attack':
                if _detector_identity(reference_run['reports'][seed]) != _detector_identity(run['reports'][seed]):
                    raise ValueError('Attack-comparison detector context mismatch')
                if reference_run['reports'][seed]['protocol'].get('target_defense_mode', 'none') != run['reports'][seed]['protocol'].get('target_defense_mode', 'none'):
                    raise ValueError('Attack-comparison training defense context differs')
            else:
                pa = reference_run['reports'][seed]['protocol']
                pb = run['reports'][seed]['protocol']
                if pa['target_checkpoint_sha256'] == pb['target_checkpoint_sha256'] and (
                    pa.get('target_defense_mode', 'none') == pb.get('target_defense_mode', 'none')):
                    # Detector-only comparisons cannot change classifier decisions.
                    if (not np.array_equal(ref_a['accepted_mask'], arrays['accepted_mask'])
                            or not np.array_equal(ref_a['clean_predictions'], arrays['clean_predictions'])
                            or not np.array_equal(ref_a['constraint_gated_defended_predictions'], arrays['constraint_gated_defended_predictions'])):
                        raise ValueError('Detector-only comparison changed classifier outputs; detector does not relabel')
            all_outcomes[name].append(o)
        per_seed.append({'seed': seed, **{n: loaded[n][0] for n in ordered_labels}})
        differences = {'seed': seed}
        for name in ordered_labels:
            if name == reference:
                continue
            if mode == 'attack':
                key = f'{name}_minus_{reference}'
                diff = _delta(loaded[name][0]['asr'], ref_o['asr'])
                differences[key] = diff
                deltas.setdefault(key, []).append(diff)
            else:
                key = f'{name}_minus_{reference}'
                diff_values = {}
                for metric in _DEFENSE_DELTA_METRICS:
                    actual = 'asr' if metric == 'feasible_asr_policy_specific' else metric
                    diff_values[metric] = _delta(_seed_metric(loaded[name][0], actual),
                                                 _seed_metric(ref_o, actual))
                    deltas.setdefault(key + ':' + metric, []).append(diff_values[metric])
                differences[key] = diff_values
        paired.append(differences)
        del loaded  # release matched NPZ arrays before next seed
    descriptive = {}
    for name in ordered_labels:
        outcomes = all_outcomes[name]
        numeric_keys = set().union(*(set(o['reported_metrics']) for o in outcomes))
        numeric_keys.update(('asr', 'conditional_asr', 'realizability_rate_all_test',
                             'realizability_rate_eligible', 'full_test_clean_accuracy',
                             'full_test_constraint_gated_defended_accuracy'))
        derived_keys = set(outcomes[0]['derived_operational_metrics'])
        numeric_keys.update(derived_keys)
        summaries = {key: _stats([_seed_metric(o, key) for o in outcomes])
                     for key in sorted(numeric_keys)
                     if any(_seed_metric(o, key) is not None for o in outcomes)
                     or key in {'asr', 'conditional_asr', 'realizability_rate_eligible'}
                     or key in derived_keys}
        descriptive[name] = {'metrics': summaries,
          'total_test_count': sum(o['test_count'] for o in outcomes),
          'per_seed_eligible_counts': [o['eligible_count'] for o in outcomes],
          'per_seed_accepted_eligible_counts': [o['accepted_eligible_count'] for o in outcomes],
          'resolved_attack': by_label[name]['config']['attack'],
          **_method_publication_evidence(by_label[name]['attack']),
          'primary_original_attack_ranking_admitted': _paper_admitted(
              _method_publication_evidence(by_label[name]['attack'])),
          'resolved_config_sha256': by_label[name]['config_sha256'],
          'execution_id': by_label[name]['execution_id'],
          'environment': by_label[name].get('environment'),
          'access_description': outcomes[0]['attack_access_description'],
          'native_budget_by_seed': [o['final_hard_budget'] for o in outcomes],
          'defense_context': _defense_label(by_label[name]),
          'detector_artifacts_by_seed': [o['detector_artifact'] for o in outcomes]}
    delta_summaries = {}
    for key, values in sorted(deltas.items()):
        stat = _stats(values)
        delta_summaries[key] = {'n_finite_pairs': stat['n_finite'],
                                'mean': stat['mean'], 'std': stat['std']}
    first = by_label[reference]
    group_text = (_defense_label(first) if mode == 'attack' else first['attack'] + '/' +
                  str(first['reports'][seeds[0]].get('target_model')))
    group_id = (mode + ':' + group_text + ':' +
                hashlib.sha256(json.dumps(_attack_key(first, seeds) if mode == 'attack'
                    else _defense_key(first, seeds), default=str).encode()).hexdigest()[:12])
    return {
        'group_id': group_id, 'mode': mode, 'context': group_text,
        'reference': reference, 'reference_attack' if mode == 'attack' else 'reference_defense': reference,
        'seeds': list(seeds), 'arm_labels': ordered_labels,
        'arms': descriptive, 'per_seed': per_seed, 'paired_deltas': paired,
        **({'primary_original_attack_view': _primary_original_attack_view(descriptive, ordered_labels)}
           if mode == 'attack' else {}),
        'across_seed_descriptive': delta_summaries,
        'denominator_policy': ('same_frozen_target_same_eligible_population' if mode == 'attack'
                               else 'different_targets_may_have_different_eligible_populations'),
        'detector_classification_effect': 'no_detector_interception_or_relabeling',
        'comparison_scope': ('native_heterogeneous_resources_not_equal_budget' if mode == 'attack'
                              else 'matched_attack_configuration_training_and_auxiliary_detector_defense'),
        'inference': {'status': 'descriptive_overlapping_or_unverified_units',
                      'p_value': None, 'ci95': None, 'n_finite_pairs': {
                          key: val['n_finite_pairs'] for key, val in delta_summaries.items()},
                      'reason': 'Aligned worker seeds/overlapping test records do not prove independent acquisitions'},
    }


def _assign_reference(items: list[dict], requested: str | None, mode: str) -> str | None:
    if requested is None:
        return None
    if requested in {run['label'] for run in items}:
        return requested
    # Multiple automatic groups: explicit reference may target just one group;
    # all other groups still need an unambiguous documented baseline.
    return None


def summarize_comparisons(*, experiment_dir: Path,
                          expected_seeds: Sequence[int] = (1, 2, 3, 4, 5),
                          runs: Mapping[str, Path] | None = None,
                          discover: bool = False, mode: str = 'attack',
                          reference: str | None = None) -> dict:
    """Read registered completed results, never models or traffic features.

    Mode A matches exact frozen target within seed; mode D intentionally permits
    independently trained targets and differing clean eligibility, while fixing
    the attack recipe, surrogate and held-out membership.
    """
    if mode not in {'attack', 'defense', 'all'}:
        raise ValueError(f'Unrecognized comparison mode: {mode}')
    if bool(discover) == bool(runs):
        raise ValueError('Specify either --discover or at least two explicit --run selections')
    root = Path(experiment_dir).resolve()
    registry = _read_json(root / 'experiment_manifest.json')
    seeds = _seeds(expected_seeds)
    if discover:
        chosen = select_completed_executions(root, expected_seeds=seeds)
        if not chosen:
            raise ValueError(f'No registered completed strict research evaluation runs for seeds {seeds}')
        descriptors = [_load_run('execution:' + x['execution_id'], x['path'], root, registry, seeds)
                       for x in chosen]
        # For discovery, use stable attack display labels where unique within
        # a target context; defense gets its human readable training/detector
        # context. Duplicate logical arms always require explicit selection.
        for item in descriptors:
            item['label'] = item['attack'] if mode == 'attack' else _defense_label(item)
    else:
        if len(runs) < 2:
            raise ValueError('At least two explicitly selected completed runs required')
        descriptors = [_load_run(name, Path(path).resolve(), root, registry, seeds)
                       for name, path in sorted(runs.items())]
    results = {'status': 'descriptive_registered_research_comparisons',
               'experiment_id': registry.get('experiment_id'),
               'experiment_dir': str(root),
               'source_revision_sha256': registry.get('source_revision_sha256'),
               'requested_seeds': list(seeds), 'mode': mode,
               'attack_groups': [], 'defense_groups': [], 'skipped_groups': [],
               'inference': {'status': 'descriptive_overlapping_or_unverified_units',
                             'p_value': None, 'ci95': None,
                             'reason': 'No independent-capture acquisition or IID repeat-unit evidence'},
               'scientific_label': 'native_configuration_descriptive_not_equal_budget_not_primary_paper_certification'}
    # An explicitly requested reference must occur in an *emitted* comparable
    # group, never only in a skipped/one-arm group. All-mode may use documented
    # defaults for its other, separately scoped comparison groups.
    emitted_arm_labels: set[str] = set()
    explicit_reference_matched = False
    for selected_mode in (('attack', 'defense') if mode == 'all' else (mode,)):
        partitions: dict[tuple, list[dict]] = {}
        key_fn = _attack_key if selected_mode == 'attack' else _defense_key
        for item in descriptors:
            partitions.setdefault(key_fn(item, seeds), []).append(item)
        for key, members in sorted(partitions.items(), key=lambda x: str(x[0])):
            if len(members) < 2:
                results['skipped_groups'].append({'mode': selected_mode,
                    'reason': 'fewer_than_two_compatible_completed_arms',
                    'executions': sorted(r['execution_id'] for r in members),
                    'attack': members[0]['attack']})
                continue
            if discover:
                # Mode all can contain several attack and defense groups. Names
                # are unique per group, and reference is decided group-locally.
                for r in members:
                    r['label'] = r['attack'] if selected_mode == 'attack' else _defense_label(r)
                if len({r['label'] for r in members}) < len(members):
                    raise ValueError(f'Duplicate completed logical {selected_mode} arm; execution IDs: '
                                     + ', '.join(sorted(r['execution_id'] for r in members)))
            group_labels = {r['label'] for r in members}
            emitted_arm_labels.update(group_labels)
            if reference is not None and reference in group_labels:
                explicit_reference_matched = True
            elif reference is not None and mode != 'all':
                raise ValueError(f'Unknown explicit reference {reference!r}; '
                                 f'available arms: {sorted(group_labels)}')
            chosen_reference = _assign_reference(members, reference, selected_mode)
            group = _compare_group(members, mode=selected_mode, root=root,
                                   registry=registry, seeds=seeds, reference=chosen_reference)
            results[selected_mode + '_groups'].append(group)
    if discover:
        # Mismatched seed sets, smoke runs and uncompleted evaluations are
        # accounted for explicitly; none enters a scientific group.
        for execution_id, entry in sorted(registry.get('executions', {}).items()):
            if entry.get('action') != 'evaluate_attack':
                continue
            if entry.get('status') != 'completed':
                reason = 'evaluation_not_completed'
            elif (tuple(sorted(map(int, entry.get('seeds', [])))) != seeds
                  or len(entry.get('seeds', [])) != len(seeds)):
                reason = 'requested_research_seeds_incomplete_or_mismatched'
            elif execution_id not in {r['execution_id'] for r in descriptors}:
                reason = 'nonresearch_or_diagnostic_configuration'
            else:
                continue
            results['skipped_groups'].append({'mode': 'execution', 'execution_id': execution_id,
                                              'reason': reason})
    if reference is not None and not explicit_reference_matched:
        raise ValueError(f'Unknown explicit reference {reference!r}; available arms '
                         f'in emitted comparison groups: {sorted(emitted_arm_labels)}')
    if not results['attack_groups'] and not results['defense_groups']:
        raise ValueError('No compatible completed research comparison group; ' +
                         json.dumps(results['skipped_groups'], sort_keys=True))
    return _strict_json_values(results)


def summarize_native_runs(runs: Mapping[str, Path], *, experiment_dir: Path,
                          expected_seeds: Sequence[int] = (1, 2, 3, 4, 5),
                          reference: str | None = None) -> dict:
    """Backward-compatible explicit two-or-more attack comparison interface."""
    group = summarize_comparisons(experiment_dir=experiment_dir,
                  expected_seeds=expected_seeds, runs=runs, reference=reference,
                  mode='attack')['attack_groups'][0]
    recipes = {name: {'resolved_attack': group['arms'][name]['resolved_attack'],
                      'resolved_config_sha256': group['arms'][name]['resolved_config_sha256'],
                      'paper_fidelity': 'unverified_until_primary_study_recipe_is_traced',
                      'access_description': group['arms'][name]['access_description']}
               for name in group['arm_labels']}
    return _strict_json_values({
        'status': 'descriptive_native_configuration_comparison',
        'label': 'native-configuration attack recipe comparison (different epsilon/query/training allowances)',
        'experiment_id': _read_json(Path(experiment_dir) / 'experiment_manifest.json').get('experiment_id'),
        'reference_attack': group['reference'], 'recipes': recipes,
        'per_seed': group['per_seed'], 'paired_deltas': group['paired_deltas'],
        'across_seed_descriptive': group['across_seed_descriptive'],
        'arms': group['arms'], 'inference': group['inference'],
    })


_SEED_CSV = ('experiment_id', 'mode', 'group_id', 'context', 'arm', 'seed',
             'execution_id', 'resolved_config_sha256', 'resolved_attack',
             'target_architecture_params_sha256', 'target_checkpoint_sha256',
             'manifest_sha256', 'selected_test_ids_sha256', 'attack_name',
             'class_mapping_sha256', 'physical_transformation_sha256',
             'threat_model_id', 'query_accounting_policy_version',
             'classification_metric_policy_version', 'target_label',
             'attack_access', 'target_defense_mode', 'surrogate_checkpoint_sha256',
             'attack_seed', 'split_seed', 'target_seed', 'detector_seed',
             'test_count', 'malicious_true_count', 'eligible_count',
             'accepted_eligible_count', 'success_count', 'accepted_all_count',
             'rejected_all_count', 'scored_count', 'feasible_asr',
             'conditional_feasible_asr', 'realizability_rate_all_test',
             'realizability_rate_eligible', 'full_test_clean_accuracy',
             'full_test_constraint_gated_defended_accuracy',
             'full_test_clean_mcc', 'full_test_constraint_gated_defended_mcc',
             'accepted_subset_clean_f1_macro', 'accepted_subset_adversarial_f1_macro',
             'full_test_clean_f1_macro', 'full_test_constraint_gated_defended_f1_macro',
             'mean_l0', 'mean_l2', 'mean_linf',
             'surrogate_generation_inference_records', 'surrogate_generation_inference_calls',
             'surrogate_generation_gradient_forward_records',
             'surrogate_generation_gradient_forward_calls',
             'all_attempts_total_queries', 'all_attempts_mean_queries',
             'all_attempts_plugin_reported_inference_records',
             'per_sample_inference_query_attribution_available',
             'query_telemetry_available', 'target_clean_inference_records',
             'target_adversarial_scored_inference_records',
             'detection_auroc', 'detection_auroc_realizable',
             'detector_checkpoint_sha256', 'detector_fitted_params', 'detector_scoring_params',
             'final_hard_budget_checked', 'final_hard_budget_norm', 'final_hard_budget_epsilon',
             'final_hard_budget_domain', 'final_hard_budget_failure_count',
             'offline_generator_training_cost', 'inference_latency_ms_per_sample',
             'throughput_flows_per_second', 'class_support_by_population', 'status',
             'full_test_malicious_support', 'full_test_benign_support',
             'full_test_clean_malicious_detection_rate',
             'full_test_clean_malicious_false_negative_rate',
             'full_test_constraint_gated_defended_malicious_detection_rate',
             'full_test_constraint_gated_defended_malicious_false_negative_rate',
             'full_test_clean_to_gated_accuracy_drop', 'full_test_clean_to_gated_mcc_drop')
_SUMMARY_CSV = ('experiment_id', 'mode', 'group_id', 'context', 'reference', 'arm',
                'execution_id', 'metric', 'n_finite', 'mean', 'std',
                'eligible_counts_by_seed', 'native_hard_budget_by_seed',
                'scientific_label')


def _format_csv_value(value):
    if value is None:
        return ''
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return value


def _md_value(value, *, signed: bool = False) -> str:
    """Display finite scalars without rounding tiny nonzero values to zero."""
    if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
        return 'undefined'
    number = float(value)
    ordinary = f'{number:+.4f}' if signed else f'{number:.4f}'
    if number != 0 and f'{number:.4f}' in {'0.0000', '-0.0000'}:
        return f'{number:+.3e}' if signed else f'{number:.3e}'
    return ordinary


def _md_stat(stats: dict | None, *, signed: bool = False, paired: bool = False) -> str:
    stats = stats or {}
    n = stats.get('n_finite_pairs' if paired else 'n_finite', 0)
    if stats.get('mean') is None:
        return f'undefined (n={n})'
    return f"{_md_value(stats['mean'], signed=signed)} ± {_md_value(stats.get('std'))} (n={n})"


def _md_metric(arm: dict, name: str) -> str:
    return _md_stat(arm['metrics'].get(name))


def _md_delta(group: dict, arm: str, metric: str | None = None) -> str:
    if arm == group['reference']:
        return 'reference'
    key = f"{arm}_minus_{group['reference']}" + ((':' + metric) if metric else '')
    return _md_stat(group['across_seed_descriptive'].get(key), signed=True, paired=True)


def _md_text(value: object) -> str:
    """Escape presentation-only Markdown table delimiters."""
    return str(value).replace('|', r'\|').replace('\n', ' ')


def _md_bound(arm: dict, seeds: Sequence[int]) -> str:
    budgets = arm['native_budget_by_seed']
    def describe(b):
        if b is None:
            return 'no declared hard norm'
        return f"{b.get('norm', 'undefined')} ε={_md_value(b.get('epsilon'))} ({b.get('domain', 'undefined')})"
    unique = list(dict.fromkeys(describe(b) for b in budgets))
    if len(budgets) != len(seeds):
        raise ValueError('Native budget rows do not align with actual worker seeds')
    return _md_text(unique[0] if len(unique) == 1 else '; '.join(
        f'seed {seed}: {describe(budget)}' for seed, budget in zip(seeds, budgets)))


def _md_pair_metrics(arm: dict, stem: str) -> str:
    return (_md_metric(arm, 'full_test_clean_' + stem) + ' → ' +
            _md_metric(arm, 'full_test_constraint_gated_defended_' + stem))


def _markdown_group(group: dict, mode: str) -> list[str]:
    """Presentation projection of *already validated* arm and paired statistics.

    No ASR, MCC, recall, quantile, detection threshold or group statistic is
    recomputed here; all numerical cells use the JSON aggregate directly.
    """
    lines = [f"### {_md_text(group['context'])}", '',
             f"Reference: `{_md_text(group['reference'])}`; seeds: `{group['seeds']}`; "
             f"finite paired differences are descriptive (sample SD; n shown).", '']
    arms = group['arm_labels']
    if mode == 'attack':
        lines += ['#### Native attack recipes — strict feasible benign evasion', '',
                  'Native heterogeneous budgets — not equal-resource ranking. '
                  'A separate explicit publication-admission view excludes method-inspired, '
                  'distributional-stress, training-only and diagnostic roles; '
                  'neither view asserts an original-method ranking. '
                  'Eligible support includes rejected initially-correct malicious attempts.', '',
                  '| Arm | Feasible ASR mean ± sample SD | Eligible / accepted-eligible by seed | Eligible feasibility | Δ feasible ASR vs reference (paired) | Native hard bound | Surrogate access / method |',
                  '|---|---|---|---|---|---|---|']
        for name in arms:
            a = group['arms'][name]
            method = a['resolved_attack']
            access = (str(method.get('name', name)) + '; ' + a['access_description']
                      + '; publication_role=' + a['publication_role'])
            if method.get('targeted') is not None:
                access += '; targeted=' + str(bool(method['targeted'])).lower()
            for declared in ('steps', 'max_queries', 'restarts'):
                if method.get(declared) is not None:
                    access += f'; {declared}={method[declared]}'
            lines.append('| ' + ' | '.join(map(_md_text, (
                name, _md_metric(a, 'asr'),
                str(a['per_seed_eligible_counts']) + ' / ' + str(a['per_seed_accepted_eligible_counts']),
                _md_metric(a, 'realizability_rate_eligible'), _md_delta(group, name),
                _md_bound(a, group['seeds']), access))) + ' |')
        view = group['primary_original_attack_view']
        lines += ['', '#### Original-method publication admission (not a ranking)', '',
                  'Eligible labels: ' + (', '.join(map(str, view['included_arm_labels'])) or 'none') + '.',
                  'Excluded from primary original-method analysis: ' +
                  (', '.join(f'{label} ({role})' for label, role in view['excluded_arms'].items())
                   or 'none') + '.',
                  'The complete native-recipe table above remains descriptive and includes all arms. '
                  'No equal-resource ranking or publication-grade fidelity is inferred.', '']
        lines += ['', 'Paired-delta JSON keys: ' + ', '.join(
            f'`{name}_minus_{group["reference"]}`' for name in arms
            if name != group['reference']) + '.', '']
        lines += ['#### Attack physical feasibility and separately metered costs', '',
                  'Successful-population float64 physical L0/L2/L∞ exclude padding; '
                  'accepted-population `mean_l*` is **not** substituted for absent successful distances. '
                  'Columns use `physical_successful_mean_l0`, `physical_successful_mean_l2`, '
                  'and `physical_successful_mean_linf`.', '',
                  '| Arm | Physical-success support | Successful physical L0 / L2 / L∞ | Accepted all-test feasibility | Surrogate inference records / calls | Surrogate gradient-forward records / calls | All-attempt surrogate queries | Per-record query attribution |',
                  '|---|---|---|---|---|---|---|---|']
        for name in arms:
            a = group['arms'][name]
            fields = (name,
                      _md_metric(a, 'physical_successful_sample_count'),
                      ' / '.join(_md_metric(a, 'physical_successful_mean_' + k)
                                 for k in ('l0', 'l2', 'linf')),
                      _md_metric(a, 'realizability_rate_all_test'),
                      _md_metric(a, 'surrogate_generation_inference_records') + ' / ' +
                      _md_metric(a, 'surrogate_generation_inference_calls'),
                      _md_metric(a, 'surrogate_generation_gradient_forward_records') + ' / ' +
                      _md_metric(a, 'surrogate_generation_gradient_forward_calls'),
                      _md_metric(a, 'all_attempts_total_queries'),
                      _md_metric(a, 'per_sample_inference_query_attribution_available'))
            lines.append('| ' + ' | '.join(map(_md_text, fields)) + ' |')
        lines += ['', 'surrogate inference records are not gradient-forward records, and neither '
                  'is a frozen-target generation query. Query attribution can be unavailable. '
                  'Measured target inference latency, when present, is **not** end-to-end attack runtime.', '']
    else:
        lines += ['#### Training-defense impact — full held-out clean / constraint-gated attacked', '',
                  'Every defense retains its **own distinct eligible denominator**; a feasible-ASR '
                  'contrast across different trained targets is policy-specific, not a paired '
                  'per-record success rate. Rejected attempts use clean-prediction fallback.', '',
                  '| Defense arm | Clean → gated accuracy | Clean → gated MCC | Clean → gated macro-F1 | Clean → gated binary malicious DR | Clean−gated accuracy / MCC drop | Policy-specific feasible ASR | Eligible counts by seed |',
                  '|---|---|---|---|---|---|---|---|']
        for name in arms:
            a = group['arms'][name]
            fields = (name, _md_pair_metrics(a, 'accuracy'), _md_pair_metrics(a, 'mcc'),
                      _md_pair_metrics(a, 'f1_macro'),
                      _md_pair_metrics(a, 'malicious_detection_rate'),
                      _md_metric(a, 'full_test_clean_to_gated_accuracy_drop') + ' / ' +
                      _md_metric(a, 'full_test_clean_to_gated_mcc_drop'),
                      _md_metric(a, 'asr'), a['per_seed_eligible_counts'])
            lines.append('| ' + ' | '.join(map(_md_text, fields)) + ' |')
        lines += ['', '#### Signed reference-relative training-defense differences', '',
                  '| Arm | Δ clean accuracy | Δ gated accuracy | Δ clean MCC | Δ gated MCC | Δ clean macro-F1 | Δ gated macro-F1 | Δ policy-specific ASR |',
                  '|---|---|---|---|---|---|---|---|']
        for name in arms:
            diff_metrics = ('full_test_clean_accuracy', 'full_test_constraint_gated_defended_accuracy',
                            'full_test_clean_mcc', 'full_test_constraint_gated_defended_mcc',
                            'full_test_clean_f1_macro', 'full_test_constraint_gated_defended_f1_macro',
                            'feasible_asr_policy_specific')
            lines.append('| ' + ' | '.join(map(_md_text,
                (name, *(_md_delta(group, name, m) for m in diff_metrics)))) + ' |')
        lines += ['', '#### Signed operational binary differences (defense vs reference)', '',
                  '| Arm | Δ clean malicious DR | Δ gated malicious DR | Δ gated malicious FNR | Δ accuracy drop | Δ MCC drop |',
                  '|---|---|---|---|---|---|']
        for name in arms:
            diff_metrics = ('full_test_clean_malicious_detection_rate',
                            'full_test_constraint_gated_defended_malicious_detection_rate',
                            'full_test_constraint_gated_defended_malicious_false_negative_rate',
                            'full_test_clean_to_gated_accuracy_drop',
                            'full_test_clean_to_gated_mcc_drop')
            lines.append('| ' + ' | '.join(map(_md_text,
                (name, *(_md_delta(group, name, m) for m in diff_metrics)))) + ' |')
    if mode == 'attack':
        lines += ['', '#### Full-test classifier performance — clean → constraint-gated attacked', '',
                  '| Arm | Clean → gated accuracy | Clean → gated MCC | Clean → gated macro-F1 | Clean → gated binary malicious DR |',
                  '|---|---|---|---|---|']
        for name in arms:
            a = group['arms'][name]
            fields = (name, _md_pair_metrics(a, 'accuracy'), _md_pair_metrics(a, 'mcc'),
                      _md_pair_metrics(a, 'f1_macro'),
                      _md_pair_metrics(a, 'malicious_detection_rate'))
            lines.append('| ' + ' | '.join(map(_md_text, fields)) + ' |')
    lines += ['', '#### Full-test NIDS classification and operational indicators', '',
              'Rates below use the **full held-out test population**, not accepted-only '
              'samples. Binary malicious DR counts wrong-family malicious predictions '
              'as detected; binary malicious DR is not macro family recall (`dr`) '
              'or benign-target evasion ASR.', '',
              '| Arm | Clean → gated benign FAR | Clean → gated binary malicious FNR | Malicious / benign support by seed | Clean−gated accuracy drop | Clean−gated MCC drop |',
              '|---|---|---|---|---|---|']
    for name in arms:
        a = group['arms'][name]
        seed_rows = [row[name] for row in group['per_seed']]
        counts = ([r.get('full_test_malicious_support') for r in seed_rows],
                  [r.get('full_test_benign_support') for r in seed_rows])
        fields = (name, _md_pair_metrics(a, 'benign_false_alarm_rate'),
                  _md_pair_metrics(a, 'malicious_false_negative_rate'),
                  str(counts[0]) + ' / ' + str(counts[1]),
                  _md_metric(a, 'full_test_clean_to_gated_accuracy_drop'),
                  _md_metric(a, 'full_test_clean_to_gated_mcc_drop'))
        lines.append('| ' + ' | '.join(map(_md_text, fields)) + ' |')
    if any(artifact for name in arms for artifact in group['arms'][name]['detector_artifacts_by_seed']):
        lines += ['', '#### Auxiliary detector discrimination (not classifier mitigation)', '',
                  '| Arm | Detector checkpoint (abbreviated) | Matched-accepted AUROC mean ± sample SD | Accepted / scored per seed |',
                  '|---|---|---|---|']
        for name in arms:
            a = group['arms'][name]
            shas = [(x or {}).get('checkpoint_sha256') for x in a['detector_artifacts_by_seed']]
            compact = ', '.join(sorted({str(s)[:16] for s in shas if s})) or 'not applicable'
            seed_rows = [row[name] for row in group['per_seed']]
            support = ([r['accepted_all_count'] for r in seed_rows],
                       [r['scored_count'] for r in seed_rows])
            fields = (name, compact, _md_metric(a, 'detection_auroc'),
                      str(support[0]) + ' / ' + str(support[1]))
            lines.append('| ' + ' | '.join(map(_md_text, fields)) + ' |')
        lines += ['', 'Auxiliary score only — no detector threshold, blocking, '
                  'interception or classifier relabeling. Detection AUROC compares '
                  'matched accepted clean/adversarial score populations, not classifier accuracy.', '']
    lines += ['', 'Full per-class support/recall, accepted-subset classifications, '
              'native attack training/query caps, detector identities and all evaluator '
              'metrics remain in the JSON and CSV. Descriptive means and sample SD '
              'are calculated across finite seed observations, not pooled records.', '']
    return lines

def write_report(result: dict, out: Path) -> dict[str, Path]:
    """Emit JSON and stdlib CSV/Markdown projections from one validated aggregate."""
    out = Path(out).resolve()
    root = result.get('experiment_dir')
    if root and out.is_relative_to(Path(root).resolve()):
        raise ValueError('Comparison outputs must be outside the immutable experiment directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Only these newly requested report files are touched; no experiment files.
    out.write_text(json.dumps(result, sort_keys=True, indent=2,
                              allow_nan=False) + '\n', encoding='utf-8')
    produced = {'json': out}
    md = ['# NIDS registered native-configuration research comparison', '',
          f"Experiment: `{result['experiment_id']}`", '',
          'Descriptive only; heterogeneous native attack resources are not equal-budget rankings.',
          f"{len(result['requested_seeds'])} aligned replicate seeds; statistical independence unverified; "
          'no inferential p-values or confidence intervals.',
          'Float64 aligned-feature physical distances are descriptive mixed-unit values,',
          'not normalized mathematical epsilon or raw packet-edit cost.',
          'Constraint-gated attacked full-test classification uses clean prediction fallback for rejections.',
          'Detector AUROC is auxiliary discrimination only; no deployed classifier blocking.', '']
    for mode in ('attack', 'defense'):
        groups = sorted(result.get(mode + '_groups', []), key=lambda g: g['group_id'])
        if not groups:
            continue
        per_rows, summary_rows = [], []
        md.append('## ' + mode.capitalize() + ' comparisons')
        md.append('')
        for group in groups:
            md.extend(_markdown_group(group, mode))
            for arm in group['arm_labels']:
                data = group['arms'][arm]
                for metric, stat in sorted(data['metrics'].items()):
                    summary_rows.append({
                        'experiment_id': result['experiment_id'], 'mode': mode,
                        'group_id': group['group_id'], 'context': group['context'],
                        'reference': group['reference'], 'arm': arm,
                        'execution_id': data['execution_id'], 'metric': metric,
                        'n_finite': stat['n_finite'], 'mean': stat['mean'], 'std': stat['std'],
                        'eligible_counts_by_seed': data['per_seed_eligible_counts'],
                        'native_hard_budget_by_seed': data['native_budget_by_seed'],
                        'scientific_label': group['comparison_scope']})
            for seed_row in group['per_seed']:
                for arm in group['arm_labels']:
                    o = seed_row[arm]
                    p = o.get('reported_metrics', {})
                    hard = o.get('final_hard_budget') or {}
                    roles = o['seed_roles']
                    row = {
                        'experiment_id': result['experiment_id'], 'mode': mode,
                        'group_id': group['group_id'], 'context': group['context'],
                        'arm': arm, 'seed': seed_row['seed'],
                        'execution_id': o['execution_id'],
                        'resolved_config_sha256': o['resolved_config_sha256'],
                        'target_checkpoint_sha256': o['target_checkpoint_sha256'],
                        'manifest_sha256': o['manifest_sha256'],
                        'selected_test_ids_sha256': o['selected_test_ids_sha256'],
                        'attack_name': o['attack_name'], 'attack_access': o['attack_access'],
                        'target_defense_mode': o['target_defense_mode'],
                        'surrogate_checkpoint_sha256': o['surrogate_checkpoint_sha256'],
                        'attack_seed': roles.get('attack'), 'split_seed': roles.get('split'),
                        'target_seed': roles.get('target_model'), 'detector_seed': roles.get('detector'),
                        'detector_checkpoint_sha256': (o['detector_artifact'] or {}).get('checkpoint_sha256'),
                        'detector_fitted_params': (o['detector_artifact'] or {}).get('fitted_params'),
                        'detector_scoring_params': (o['detector_artifact'] or {}).get('scoring_params'),
                        'final_hard_budget_norm': hard.get('norm'),
                        'final_hard_budget_epsilon': hard.get('epsilon'),
                        'final_hard_budget_domain': hard.get('domain'),
                        'offline_generator_training_cost': o['offline_generator_training_cost'],
                        'status': 'descriptive_strict_registered_research',
                    }
                    row.update({field: o.get(field, p.get(field)) for field in _SEED_CSV if field not in row})
                    per_rows.append(row)
        for suffix, headers, rows in [('per_seed', _SEED_CSV, per_rows),
                                       ('summary', _SUMMARY_CSV, summary_rows)]:
            path = out.parent / f'{mode}_{suffix}.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=headers)
                writer.writeheader()
                writer.writerows({key: _format_csv_value(row.get(key)) for key in headers} for row in rows)
            produced[f'{mode}_{suffix}_csv'] = path
    if result.get('skipped_groups'):
        md.extend(['## Skipped groups', '',
                   '```json', json.dumps(result['skipped_groups'], indent=2, sort_keys=True), '```', ''])
    path = out.parent / 'comparison_summary.md'
    path.write_text('\n'.join(md) + '\n', encoding='utf-8')
    produced['markdown'] = path
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--run', action='append', metavar='LABEL=RUN_DIR')
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--mode', choices=('attack', 'defense', 'all'), default='attack')
    parser.add_argument('--reference', help='Explicit baseline arm label for signed differences')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for item in args.run or []:
        label, separator, path = item.partition('=')
        if not separator or not label or not path or label in runs:
            parser.error('--run must use unique LABEL=RUN_DIR selections')
        runs[label] = Path(path)
    if args.discover and runs:
        parser.error('--discover cannot be combined with explicit --run; use one mode of selection')
    if not args.discover and not runs:
        parser.error('Supply --discover or explicit --run entries')
    result = summarize_comparisons(experiment_dir=args.experiment_dir,
                                   expected_seeds=args.seeds,
                                   discover=args.discover, runs=runs or None,
                                   mode=args.mode, reference=args.reference)
    outputs = write_report(result, args.out)
    for name, path in outputs.items():
        print(name + ': ' + str(path))


if __name__ == '__main__':
    main()
