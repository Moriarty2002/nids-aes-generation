"""Registered AMD-GAN class-distribution study (Lara-Gutierrez et al., 2026, §5).

The original AMD-GAN is a class-decoupled WGAN-GP for distribution synthesis,
minority-class TSTR and volumetric stress, not per-record benign-target ASR.
This CICIoMT2024 adaptation samples physical packet-feature rows from the
registered train-only specialists and measures TSTR on *held-out real packets*.
Its declared sample unit differs from the paper's source flow representation;
no target or surrogate classifier is queried by the distribution sampler.
The deferred raw-PCAP bridge is not implied by a valid packet-feature sample.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.exceptions import ConfigurationError
from core.provenance import (
    _file_sha256, generator_artifact_provenance, load_split_manifest,
    manifest_hash, split_indices, validate_research_split_manifest,
)
from orchestrator.device import configure_process_device
from orchestrator.runner import seed_everything
from .experiment import ExperimentStore
from .factory import _cfg, _inject_dynamic_dataset_shape, _instantiate_attack, _instantiate_dataset
from .io import _load_checkpoint_metadata, _save_json
from .seeds import role_seed


def distribution_stress_seed(*, config: Mapping[str, Any], seed: int, gpu: str,
                             logger: logging.Logger | None = None,
                             progress: Any = None) -> Mapping[str, float]:
    """Read-only registered AMD study with split-verified held-out populations.

    Requires an immutable registered independently trained surrogate and its
    generator. This task never queries the frozen evaluation target; it
    produces TSTR diagnostic metrics, not feasible-ASR or target scores.
    """
    cfg = _cfg(config)
    if cfg.get('attack', {}).get('name') != 'amd_gan':
        raise ConfigurationError('Distribution stress requires the AMD-GAN class-specific recipe')
    configure_process_device(gpu)
    attack_seed = role_seed(cfg, 'attack', seed)
    source_seed = role_seed(cfg, 'surrogate', seed)
    target_seed = role_seed(cfg, 'target', seed)
    seed_everything(attack_seed)
    _inject_dynamic_dataset_shape(cfg)
    if progress is not None:
        progress('load_registered_distribution_source')
    dataset = _instantiate_dataset(cfg)
    physical, labels = dataset.raw_arrays()
    physical = np.asarray(physical)
    labels = np.asarray(labels, dtype=np.int64)
    if physical.dtype != np.float64:
        raise ConfigurationError('Distribution stress requires physical float64 train/test truth')
    store = ExperimentStore(cfg.get('experiment_dir'), dataset=cfg['dataset']['name'])
    source_cfg = dict(cfg.get('surrogate_model') or {})
    if not source_cfg or not source_cfg.get('enabled', True):
        raise ConfigurationError('AMD specialist source requires a registered independent surrogate')
    source_path = store.model_path(source_cfg['name'], 'none', source_seed, role='surrogates')
    store.validate_template(source_cfg.get('checkpoint'), source_path, source_seed,
                            role='surrogate_model')
    store.require('model', source_path, seed=source_seed)
    source_meta = _load_checkpoint_metadata(source_path)
    target_cfg = dict(cfg['model'])
    target_path = store.model_path(target_cfg['name'],
                                   target_cfg.get('training', {}).get('defense_mode', 'none'),
                                   target_seed)
    store.validate_template(target_cfg.get('checkpoint'), target_path, target_seed,
                            role='model')
    store.require('model', target_path, seed=target_seed)
    target_meta = _load_checkpoint_metadata(target_path)
    if (source_path == target_path
            or source_meta.get('model_state_sha256') == target_meta.get('model_state_sha256')):
        raise ConfigurationError('AMD distribution specialists need independently trained source weights')
    if (source_meta.get('split_manifest_sha256') != target_meta.get('split_manifest_sha256')
            or source_meta.get('class_mapping') != target_meta.get('class_mapping')
            or source_meta.get('dataset_identity') != target_meta.get('dataset_identity')):
        raise ConfigurationError('Distributional study surrogate and target must share source/split/schema')
    split_seed = int((target_meta.get('seed_roles') or {}).get('split', target_seed))
    manifest = load_split_manifest(target_path.parent / 'split_manifest.json',
                                   dataset, labels, split_seed,
                                   expected_hash=target_meta['split_manifest_sha256'])
    if manifest_hash(manifest) != source_meta.get('split_manifest_sha256'):
        raise ConfigurationError('Distributional training and held-out split differ')
    if cfg.get('seed_roles', {}).get('split') is not None and role_seed(cfg, 'split', seed) != split_seed:
        raise ConfigurationError('Distribution study declared split seed differs')
    if str(cfg.get('budget', {}).get('name', 'smoke')) == 'research':
        validate_research_split_manifest(manifest, labels, dataset)
        if target_meta.get('training_budget_profile') != 'research':
            raise ConfigurationError('Distributional research study requires registered research-trained target identity')
    train_ids = tuple(map(str, manifest['train_ids']))
    test_ids = tuple(map(str, manifest['test_ids']))
    if set(train_ids) & set(test_ids):
        raise ConfigurationError('Distributional study train/test records overlap')
    selected = split_indices(manifest, dataset, 'test')
    if tuple(map(str, np.asarray(dataset.sample_ids)[selected])) != test_ids:
        raise ConfigurationError('Distribution study held-out record order differs from manifest')
    names = tuple(dataset.feature_names)
    attack_cfg = dict(cfg['attack'])
    generator_path = store.generator_path(attack_cfg['name'], attack_cfg.get('variant', 'default'), attack_seed)
    store.validate_template(attack_cfg.get('checkpoint'), generator_path, attack_seed, role='attack')
    store.require('generator', generator_path, seed=attack_seed, configuration=attack_cfg)
    attack_cls, attack_params = _instantiate_attack(cfg, names)
    attack = attack_cls(**attack_params)
    if attack.attack_name != 'amd_gan':
        raise ConfigurationError('Wrong registered distribution sampler')
    attack.expected_provenance = generator_artifact_provenance(
        manifest=manifest, source_state_sha256=source_meta['model_state_sha256'],
        source_checkpoint_sha256=_file_sha256(source_path), source_role='surrogate_model',
        attack_config=attack_cfg, seed=attack_seed,
        input_shape=source_meta['input_shape'], feature_names=names,
        class_mapping=target_meta['class_mapping'], access_mode='transfer',
    )
    mask = getattr(dataset, 'valid_mask', None)
    if mask is None:
        raise ConfigurationError('TSTR real-packet membership requires the registered real-packet occupancy mask')
    mask = np.asarray(mask, dtype=bool)
    if physical.ndim == 2:
        physical = physical.reshape(len(labels), -1, len(names))
    if physical.ndim != 3 or physical.shape[:2] != mask.shape:
        raise ConfigurationError('AMD physical packets / sequence mask do not align')
    study_cfg = dict(cfg.get('distribution_stress') or {})
    samples_per_class = int(study_cfg.get('samples_per_class', 64))
    classifier_trees = int(study_cfg.get('classifier_trees', 100))
    if samples_per_class < 1 or classifier_trees < 1:
        raise ConfigurationError('Distribution study size and diagnostic classifier trees must be positive')
    classes = tuple(sorted(map(int, np.unique(labels))))
    if progress is not None:
        progress('amd_registered_class_sampling_tstr')
    report = attack.synthetic_train_test_real(
        train_sample_ids=train_ids, test_sample_ids=test_ids,
        real_test_packets=physical[selected], real_test_labels=labels[selected],
        real_test_valid_mask=mask[selected], classes=classes,
        samples_per_class=samples_per_class, seed=attack_seed,
        classifier_trees=classifier_trees,
    )
    report.update({
        'source_revision_sha256': store.read()['source_revision_sha256'],
        'experiment_id': store.read()['experiment_id'],
        'generator_checkpoint_sha256': _file_sha256(generator_path),
        'independent_surrogate_checkpoint_sha256': _file_sha256(source_path),
        'split_manifest_sha256': manifest_hash(manifest),
        'ordered_heldout_sample_ids': list(test_ids),
        'effective_attack_seed': attack_seed,
        'effective_surrogate_seed': source_seed,
        'effective_target_seed_identity_only': target_seed,
        'source_attacker_target_oracle_queries': 0,
        'paper_stress_volume_certified': False,
        'original_paper_equivalence': 'packet-record TSTR adaptation, not original volumetric/capture study',
    })
    output_dir = cfg.get('output_dir')
    if output_dir:
        _save_json(Path(str(output_dir)) / 'per_seed' / f'amd_distribution_seed{seed}.json', report)
    if logger:
        logger.info('AMD distribution TSTR seed=%d heldout_packet_count=%d source=%s',
                    seed, report['heldout_real_packet_count'], generator_path)
    if progress is not None:
        progress('amd_registered_distribution_complete', status='completed')
    return {
        'amd_tstr_real_accuracy': float(report['tstr_real_accuracy']),
        'amd_tstr_real_macro_f1': float(report['tstr_real_macro_f1']),
        'amd_heldout_real_packet_count': float(report['heldout_real_packet_count']),
        'amd_target_oracle_queries': 0.,
    }
