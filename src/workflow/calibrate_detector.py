"""Registered Lee detector preparation after independently trained target and surrogate.

Lee et al., NeurIPS 2018 Sec. 2.2/Algorithm 1: fit class Gaussian
statistics on clean train embeddings, then combine layer scores on disjoint
validation clean/adversarial examples. The validation attack is constructed
with a registered *independent surrogate*, not the frozen target. The existing
physical float64 finalizer and independent gate decide admission; rejected
validation proposals fall back to clean samples and are never treated as
realized attacks. The held-out test partition is never read for calibration.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from attacks.feature_space.fgsm import FGSMAttack
from constraints.gatekeeper import UniversalConstraintGatekeeper
from core.dual_representation import encode_normalized32
from core.exceptions import ConfigurationError
from core.provenance import (
    _digest_strings, _file_sha256, frozen_model_state_sha256,
    load_split_manifest, manifest_hash, split_indices,
)
from core.query_meter import ModelQueryMeter
from defenses.multilayer_mahalanobis import LeeMultilayerMahalanobis
from models.wrapper import PyTorchModelWrapper
from orchestrator.device import configure_process_device
from orchestrator.runner import seed_everything
from .experiment import ExperimentStore
from .factory import _build_model, _cfg, _inject_dynamic_dataset_shape, _instantiate_dataset
from .io import _load_checkpoint, _load_checkpoint_metadata, _normalization_from_checkpoint
from .seeds import role_seed


class _ObservedRows(Dataset):
    """Return the actual registered row ID alongside target input/occupancy."""
    def __init__(self, x: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                 ids: Sequence[str], allowed: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(labels, dtype=torch.long)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.ids = tuple(map(str, ids))
        self.allowed = torch.as_tensor(allowed, dtype=torch.bool)
        if len(self.x) != len(self.y) or len(self.x) != len(self.ids) or len(self.x) != len(self.mask):
            raise ValueError('Observed detector population has mismatched sample lengths')
        if tuple(self.allowed.shape) != tuple(self.x.shape):
            raise ValueError('Detector gradient edit mask must match normalized target input')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.mask[idx], self.ids[idx], self.allowed[idx]


def _ordered_indices(dataset: Any, ordered_ids: Sequence[str]) -> np.ndarray:
    positions = {str(value): i for i, value in enumerate(dataset.sample_ids)}
    if len(positions) != len(dataset.sample_ids) or any(str(k) not in positions for k in ordered_ids):
        raise ConfigurationError('Detector membership IDs must resolve uniquely in the authoritative dataset')
    return np.asarray([positions[str(k)] for k in ordered_ids], dtype=np.int64)


def _editable_mask(x: np.ndarray, norm: Any, names: tuple[str, ...], valid: np.ndarray,
                   immutable: Sequence[str]) -> np.ndarray:
    # Reuse exactly the attack's directional packet-ownership rules for Lee's
    # optional virtual score perturbation; no physical data is mutated.
    candidate = FGSMAttack(epsilon=0., feature_names=names, immutable_features=tuple(immutable))
    editable = candidate.victim_packet_perturbation_mask(
        torch.as_tensor(x), names, normalization_mean=norm.mean,
        normalization_std=norm.std, valid_mask=valid,
    ).cpu().numpy().astype(bool)
    for idx, name in enumerate(names):
        if name.upper() in {str(v).upper() for v in immutable} | {'DIR', 'FLG'}:
            editable.reshape(len(x), -1, len(names))[:, :, idx] = False
    return editable


def calibrate_detector_seed(*, config: Mapping[str, Any], seed: int, gpu: str,
                            logger: logging.Logger | None = None,
                            progress: Any = None) -> Mapping[str, float]:
    """Fit/validate/publish a *complete* Lee state under the existing registry.

    Requires the independently registered target and surrogate beforehand.
    Publication is atomic under ExperimentStore and does not edit either
    classifier checkpoint. This is a detector-only experiment action.
    """
    cfg = _cfg(config)
    detector_cfg = dict(cfg.get('detector') or {})
    if not detector_cfg.get('enabled') or detector_cfg.get('name') != LeeMultilayerMahalanobis.REGISTRY_NAME:
        raise ConfigurationError('calibrate_detector requires enabled Lee multilayer detector')
    if str((cfg.get('evaluation') or {}).get('strict_constraints', True)).lower() == 'false':
        raise ConfigurationError('Detector calibration requires strict physical admission')
    dc = dict(detector_cfg.get('params') or {})
    calibration_epsilon = float(detector_cfg.get('calibration_epsilon', 0.01))
    if not np.isfinite(calibration_epsilon) or not 0 < calibration_epsilon <= 1:
        raise ConfigurationError('Declare positive finite normalized validation FGSM epsilon <= 1')
    device = configure_process_device(gpu).torch_device
    target_seed = role_seed(cfg, 'target', seed)
    source_seed = role_seed(cfg, 'surrogate', seed)
    seed_everything(role_seed(cfg, 'detector', target_seed))
    _inject_dynamic_dataset_shape(cfg)
    dataset = _instantiate_dataset(cfg)
    x_raw, labels = dataset.raw_arrays()
    x_raw = np.asarray(x_raw)
    labels = np.asarray(labels, dtype=np.int64)
    names = tuple(dataset.feature_names)
    if x_raw.dtype != np.float64:
        raise ConfigurationError('Lee calibration requires authoritative float64 physical inputs')
    store = ExperimentStore(cfg.get('experiment_dir'), dataset=cfg['dataset']['name'])
    model_cfg = dict(cfg['model'])
    target_path = store.model_path(model_cfg['name'], model_cfg.get('training', {}).get('defense_mode', 'none'), target_seed)
    store.validate_template(model_cfg.get('checkpoint'), target_path, target_seed, role='model')
    store.require('model', target_path, seed=target_seed)
    saved_target = _load_checkpoint_metadata(target_path)
    if saved_target.get('experiment_id') != store.read()['experiment_id']:
        raise ConfigurationError('Detector target is outside this experiment')
    split_seed = int(saved_target.get('seed_roles', {}).get('split', target_seed))
    manifest = load_split_manifest(target_path.parent / 'split_manifest.json', dataset, labels, split_seed,
                                   expected_hash=saved_target['split_manifest_sha256'])
    if saved_target.get('dataset_identity') != manifest['identity'] or saved_target.get('class_mapping') != dataset.class_mapping:
        raise ConfigurationError('Detector training dataset/split differs from the target checkpoint')
    if manifest_hash(manifest) != saved_target['split_manifest_sha256']:
        raise ConfigurationError('Detector split hash differs from target checkpoint')
    surrogate_cfg = dict(cfg.get('surrogate_model') or {})
    if not surrogate_cfg or not surrogate_cfg.get('enabled', True):
        raise ConfigurationError('Lee calibration needs independently registered surrogate')
    source_path = store.model_path(surrogate_cfg['name'], 'none', source_seed, role='surrogates')
    store.validate_template(surrogate_cfg.get('checkpoint'), source_path, source_seed, role='surrogate_model')
    store.require('model', source_path, seed=source_seed)
    if source_path == target_path:
        raise ConfigurationError('Lee calibration cannot train an attack with its frozen target')
    saved_source = _load_checkpoint_metadata(source_path)
    if (saved_source.get('split_manifest_sha256') != saved_target.get('split_manifest_sha256')
            or saved_source.get('class_mapping') != saved_target.get('class_mapping')
            or saved_source.get('dataset_identity') != saved_target.get('dataset_identity')):
        raise ConfigurationError('Surrogate and target memberships/schema must share this registered split')
    target_norm = _normalization_from_checkpoint(saved_target, names, role='Lee target')
    source_norm = _normalization_from_checkpoint(saved_source, names, role='Lee surrogate')
    target = _build_model(model_cfg, (len(labels), *saved_target['input_shape']),
                          device=device, num_classes=int(saved_target['num_classes']), seed=target_seed)
    _load_checkpoint(target, target_path, checkpoint_data=saved_target)
    if not isinstance(target, PyTorchModelWrapper) or frozen_model_state_sha256(target) != saved_target['model_state_sha256']:
        raise ConfigurationError('Lee requires exact registered PyTorch target state')
    source_model_cfg = {'name': surrogate_cfg['name'], 'params': surrogate_cfg.get('params', {})}
    source = _build_model(source_model_cfg, (len(labels), *saved_source['input_shape']),
                          device=device, num_classes=int(saved_source['num_classes']), seed=source_seed)
    _load_checkpoint(source, source_path, checkpoint_data=saved_source)
    if not isinstance(source, PyTorchModelWrapper) or frozen_model_state_sha256(source) != saved_source['model_state_sha256']:
        raise ConfigurationError('Lee calibration requires independently fitted differentiable surrogate')
    if np.prod(saved_source['input_shape']) != np.prod(saved_target['input_shape']):
        raise ConfigurationError('Surrogate/target input topology cannot be bijectively aligned')
    train_ids = tuple(map(str, manifest['train_ids']))
    val_ids = tuple(map(str, manifest['validation_ids']))
    if set(train_ids) & set(val_ids) or set(val_ids) & set(manifest['test_ids']):
        raise ConfigurationError('Detector train/validation/test memberships overlap')
    tr = _ordered_indices(dataset, train_ids)
    va = _ordered_indices(dataset, val_ids)
    raw_mask = getattr(dataset, 'valid_mask', None)
    mask = (np.ones((len(labels), 1), dtype=bool) if raw_mask is None else np.asarray(raw_mask, dtype=bool))
    immutable = tuple(cfg['dataset'].get('immutable_features', ('DIR', 'FLG')))
    batch_size = int(dc.get('score_batch_size', 1024))
    norm_train = target_norm.normalize(x_raw[tr])
    norm_val = target_norm.normalize(x_raw[va])
    allowed_train = _editable_mask(norm_train, target_norm, names, mask[tr], immutable)
    allowed_val = _editable_mask(norm_val, target_norm, names, mask[va], immutable)
    def rows(arr: np.ndarray, positions: np.ndarray, ids: Sequence[str], editable: np.ndarray) -> DataLoader:
        return DataLoader(_ObservedRows(arr, labels[positions], mask[positions], ids, editable),
                          batch_size=batch_size, shuffle=False, num_workers=0)
    detector = LeeMultilayerMahalanobis(**dc)
    if progress is not None:
        progress('lee_clean_train_fit')
    detector.fit(rows(norm_train, tr, train_ids, allowed_train), target,
                 registered_train_ids=train_ids,
                 registered_split_manifest_sha256=manifest_hash(manifest))

    # Source independently generates admissible validation proposals. The
    # frozen target is guarded even if a plugin tries to access it transitively.
    attack = FGSMAttack(epsilon=calibration_epsilon, targeted=True, target_label=0,
                        feature_names=names, immutable_features=immutable,
                        clip_ranges=dict(cfg['dataset'].get('clip_ranges') or {}),
                        discrete_projection=True).bind_model(source)
    proposals = []
    accepted_count = 0
    attack_counts = {'gradient_forward_records': 0, 'inference_records': 0}
    if progress is not None:
        progress('lee_surrogate_validation_generation')
    with ModelQueryMeter(target, forbidden=True), ModelQueryMeter(source) as source_meter:
        for start in range(0, len(va), batch_size):
            stop = min(start + batch_size, len(va))
            source_input = source_norm.normalize(x_raw[va[start:stop]]).reshape(
                (stop-start, *saved_source['input_shape']))
            pristine = np.ascontiguousarray(x_raw[va[start:stop]].reshape(source_input.shape), dtype=np.float64)
            vm = mask[va[start:stop]]
            with source.attack_mask_context(vm):
                result = attack.generate(source_input, labels[va[start:stop]],
                                         normalization_mean=source_norm.mean,
                                         normalization_std=source_norm.std,
                                         valid_mask=vm, _clean_authoritative_physical=pristine)
            physical = np.asarray(result.authoritative_physical, dtype=np.float64).reshape(
                (stop-start, *saved_target['input_shape']))
            pristine_target = x_raw[va[start:stop]]
            proposed_encoded = encode_normalized32(physical, names, target_norm.mean, target_norm.std)
            gate = UniversalConstraintGatekeeper(
                strict=True, normalization_mean=target_norm.mean, normalization_std=target_norm.std,
                padding_values=getattr(dataset, 'pad_values', None),
                padding_indicator_name=(getattr(dataset, 'padding_indicator_name', None)
                                        if getattr(dataset, 'include_padding_indicator', False) else None),
            ).validate(proposed_encoded, names, authoritative_physical=physical,
                       original_physical=pristine_target, valid_mask=vm,
                       attack_budget=attack.final_budget_contract(),
                       budget_normalization_mean=source_norm.mean,
                       budget_normalization_std=source_norm.std)
            accepted_count += int(np.count_nonzero(gate.accepted))
            # Gated full-validation population has untouched clean fallback.
            encoded = target_norm.normalize(pristine_target)
            encoded[gate.accepted] = proposed_encoded[gate.accepted]
            proposals.append(encoded)
    attack_counts.update({
        'gradient_forward_records': source_meter.counts.gradient_forward_records,
        'inference_records': source_meter.counts.inference_records,
    })
    # If none passes the physical gate, calibration would manufacture an
    # adversarial distribution from identical clean samples: fail closed.
    if not accepted_count:
        raise ConfigurationError('No physically valid independent-surrogate validation examples; Lee calibration refused')
    adversarial = np.concatenate(proposals, axis=0)
    if progress is not None:
        progress('lee_validation_calibration')
    detector.calibrate(rows(norm_val, va, val_ids, allowed_val),
                       rows(adversarial, va, val_ids, allowed_val), target,
                       partition='validation', membership_sha256=_digest_strings(val_ids),
                       registered_validation_ids=val_ids,
                       registered_split_manifest_sha256=manifest_hash(manifest),
                       attack_source=(f'registered_independent_surrogate:{source_path.name}:'
                                      f'{_file_sha256(source_path)}:targeted_fgsm_eps{calibration_epsilon:g}'))
    detector_path = store.detector_path(target_path)
    if detector_cfg.get('checkpoint'):
        raise ConfigurationError('External detector paths are not registered research calibration')
    payload = {'name': detector.REGISTRY_NAME, 'params': dc, 'state_dict': detector.state_dict(),
               'seed': target_seed, 'fit_split': 'train',
               'target_model_state_sha256': saved_target['model_state_sha256'],
               'split_manifest_sha256': manifest_hash(manifest),
               'train_membership_sha256': _digest_strings(train_ids),
               'validation_membership_sha256': _digest_strings(val_ids),
               'surrogate_checkpoint_sha256': _file_sha256(source_path),
               'calibration_epsilon': calibration_epsilon,
               'calibration_accepted_count': accepted_count,
               'calibration_population_count': len(va),
               'surrogate_gradient_forward_records': attack_counts['gradient_forward_records'],
               'surrogate_inference_records': attack_counts['inference_records'],
               'detector_only': True}
    store.publish_detector(detector_path, payload, seed=target_seed,
                           target_state_sha256=saved_target['model_state_sha256'],
                           split_hash=manifest_hash(manifest))
    if progress is not None:
        progress('lee_detector_registered', status='completed')
    if logger is not None:
        logger.info('Lee calibrated seed=%s valid=%s/%s checkpoint=%s',
                    target_seed, accepted_count, len(va), detector_path)
    return {'lee_train_sample_count': float(len(tr)),
            'lee_validation_sample_count': float(len(va)),
            'lee_calibration_accepted_count': float(accepted_count),
            'lee_surrogate_gradient_forward_records': float(attack_counts['gradient_forward_records']),
            'lee_surrogate_inference_records': float(attack_counts['inference_records']),
            'lee_detector_only': 1.0}
