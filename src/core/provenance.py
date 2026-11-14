"""Immutable, seed-scoped train/validation/test provenance for NIDS benchmarks.

The manifest stores IDs and hashes, never feature matrices. File fingerprints are
streamed from disk; no additional whole-corpus float64 buffers are allocated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from core.exceptions import ConfigurationError

SPLIT_VERSION = 1


def _digest_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ConfigurationError(f'Dataset source file missing: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(dataset: Any, count: int) -> tuple[str, ...]:
    values = getattr(dataset, 'sample_ids', None)
    if values is None:
        raise ConfigurationError('Dataset must expose stable, unique sample_ids for benchmark provenance')
    values = tuple(str(value) for value in values)
    if len(values) != count or len(set(values)) != count:
        raise ConfigurationError('Dataset sample_ids must be unique and match sample count')
    return values


def dataset_identity(dataset: Any, labels: np.ndarray) -> dict[str, Any]:
    """Validate record identity without copying feature arrays."""
    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ConfigurationError('Benchmark labels must be 1-D integer values')
    source = getattr(dataset, 'path', None)
    if source is None:
        raise ConfigurationError('Dataset requires a verifiable source path')
    ids = _ids(dataset, len(values))
    names = tuple(str(name) for name in dataset.feature_names)
    adapter_state = getattr(dataset, 'state_dict', None)
    transformation = None
    if callable(adapter_state):
        state = adapter_state()
        # Do not bind fit-time normalization (computed after split creation)
        # or corpus-dependent descriptive truncation counts to the identity.
        stable_keys = (
            'fields', 'feature_names', 'sequence_length', 'sequence_policy',
            'feature_count', 'flatten', 'include_padding_indicator',
            'padding_indicator_name', 'pad_values', 'truncate',
            'strict_feat_pad', 'feat_pad_column', 'label_column',
            'benign_string', 'task_mode', 'label_mode', 'classes',
            'class_mapping', 'id_column', 'clean_physical_validation_policy',
        )
        transformation = {k: state[k] for k in stable_keys if k in state}
    return {
        'transformation': transformation,
        'group_ids_sha256': (
            _digest_strings(tuple(str(g) for g in dataset.group_ids))
            if getattr(dataset, 'group_ids', None) is not None else None
        ),
        'source_sha256': _file_sha256(Path(source)),
        'sample_ids_sha256': _digest_strings(ids),
        'labels_sha256': hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest(),
        'sample_count': len(values),
        'feature_names': list(names),
    }


# An immutable split manifest records what the splitter *actually did*.
# A research split may not silently fall back to unstratified rows.
SPLIT_POLICY_VERSION = 'audited_class_coverage_v1'


def _split_rows(rows: np.ndarray, labels: np.ndarray, fraction: float, seed: int,
                groups: np.ndarray | None, *, research: bool = False
                ) -> tuple[np.ndarray, np.ndarray, str]:
    if not 0 < fraction < 1:
        raise ConfigurationError('Split fraction must be strictly between 0 and 1')
    if groups is not None:
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
            a, b = next(splitter.split(rows, labels[rows], groups[rows]))
        except ValueError as exc:
            raise ConfigurationError(f'Group split is impossible: {exc}') from exc
        return rows[a], rows[b], 'group_shuffle'
    try:
        a, b = train_test_split(rows, test_size=fraction, random_state=seed,
                                stratify=labels[rows])
        return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), 'stratified'
    except ValueError as exc:
        if research:
            raise ConfigurationError(
                'Research stratification is impossible with the supplied class '
                f'counts/split fractions: {exc}. Choose a declared viable split '
                'policy; unstratified fallback is forbidden.'
            ) from exc
        # Permitted ONLY for smoke/diagnostic work, and explicitly recorded.
        try:
            a, b = train_test_split(rows, test_size=fraction, random_state=seed)
        except ValueError as fallback_exc:
            raise ConfigurationError(f'Smoke split is impossible: {fallback_exc}') from fallback_exc
        return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), 'random_fallback'


def create_split_manifest(dataset: Any, labels: np.ndarray, seed: int,
                          config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create the deterministic train/validation/test membership for a seed.

    Research splits require stratification or declared nonoverlapping capture
    groups; their identities bind all training/evaluation artifacts and must not
    be reconstructed independently by individual attack workflows.
    """
    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ConfigurationError('Benchmark split requires a 1-D integer label array')
    values = values.astype(np.int64, copy=False)
    if len(values) < 10:
        raise ConfigurationError('At least 10 records required for 3-way benchmark split')
    options = dict(config or {})
    policy = str(options.get('policy', 'smoke'))
    if policy not in {'research', 'smoke'}:
        raise ConfigurationError('data_split.policy must be research or smoke')
    research = policy == 'research'
    test_fraction = float(options.get('test_fraction', .2))
    validation_fraction = float(options.get('validation_fraction', .2))
    groups_raw = getattr(dataset, 'group_ids', None)
    groups = None if groups_raw is None else np.asarray(groups_raw, dtype=str)
    if groups is not None and groups.shape != (len(values),):
        raise ConfigurationError('group_ids must have shape (N,)')
    rows = np.arange(len(values), dtype=np.int64)
    train_validation, test, test_algorithm = _split_rows(
        rows, values, test_fraction, seed, groups, research=research,
    )
    train, validation, validation_algorithm = _split_rows(
        train_validation, values, validation_fraction, seed + 1, groups, research=research,
    )
    if not len(train) or not len(validation) or not len(test):
        raise ConfigurationError('Split produced an empty training, validation, or test partition')
    required_classes = set(np.unique(values).tolist())
    declared_classes = getattr(dataset, 'classes', None)
    if research and declared_classes is not None:
        declared_ids = set(range(len(declared_classes)))
        if required_classes != declared_ids:
            raise ConfigurationError(
                'Research source labels do not cover the complete declared class schema'
            )
    class_coverage = {
        part: sorted(int(v) for v in np.unique(values[index]))
        for part, index in (('train', train), ('validation', validation), ('test', test))
    }
    if required_classes - set(class_coverage['train']):
        raise ConfigurationError('Training partition is missing a class; adjust group/split settings')
    if research:
        for name in ('validation', 'test'):
            missing = sorted(required_classes - set(class_coverage[name]))
            if missing:
                raise ConfigurationError(
                    f'Research {name} partition missing classes {missing}; '
                    'full-schema NIDS comparisons require declared class coverage. '
                    'Choose a viable predeclared split/group policy.'
                )
    if groups is not None:
        memberships = [set(groups[partition].tolist()) for partition in (train, validation, test)]
        if any(memberships[i] & memberships[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ConfigurationError('Group overlap across partitions')
    ids = _ids(dataset, len(values))
    return {
        'version': SPLIT_VERSION,
        'seed': int(seed),
        'identity': dataset_identity(dataset, values),
        'split_policy': {
            'version': SPLIT_POLICY_VERSION,
            'policy': policy,
            'test_fraction': test_fraction,
            'validation_fraction': validation_fraction,
            'grouped': groups is not None,
            'test_split_algorithm': test_algorithm,
            'validation_split_algorithm': validation_algorithm,
            'class_coverage': class_coverage,
            'class_support': {
                name: {str(int(k)): int(np.count_nonzero(values[index] == k))
                       for k in sorted(required_classes)}
                for name, index in (('train', train), ('validation', validation), ('test', test))
            },
        },
        'train_ids': [ids[int(i)] for i in np.sort(train)],
        'validation_ids': [ids[int(i)] for i in np.sort(validation)],
        'test_ids': [ids[int(i)] for i in np.sort(test)],
    }


def validate_research_split_manifest(
    manifest: Mapping[str, Any], labels: np.ndarray, dataset: Any,
) -> None:
    """Verify research suitability from actual, immutable membership, not claims.

    A grouped partition is allowed if the provided groups truly do not cross
    partitions; its per-class coverage is checked without inventing groups.
    """
    policy = manifest.get('split_policy', {})
    if (policy.get('version') != SPLIT_POLICY_VERSION
            or policy.get('policy') != 'research'):
        raise ConfigurationError('Research evaluation requires an audited research split manifest; retrain')
    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ConfigurationError('Research labels must be 1-D integers')
    actual = {part: split_indices(manifest, dataset, part)
              for part in ('train', 'validation', 'test')}
    required = set(np.unique(values).tolist())
    declared_classes = getattr(dataset, 'classes', None)
    if declared_classes is not None and required != set(range(len(declared_classes))):
        raise ConfigurationError('Research source missing declared class schema members')
    for name, idx in actual.items():
        observed = set(np.unique(values[idx]).tolist())
        if observed != required:
            raise ConfigurationError(f'Research split {name} missing required class support')
        saved = policy.get('class_coverage', {}).get(name)
        if saved != sorted(int(v) for v in observed):
            raise ConfigurationError(f'Research split {name} has inconsistent class coverage metadata')
        reported_counts = policy.get('class_support', {}).get(name)
        expected_counts = {str(int(k)): int(np.count_nonzero(values[idx] == k))
                           for k in sorted(required)}
        if reported_counts != expected_counts:
            raise ConfigurationError(f'Research split {name} class support metadata mismatch')
    if policy.get('grouped'):
        group_ids = getattr(dataset, 'group_ids', None)
        if group_ids is None:
            raise ConfigurationError('Research split declares groups but dataset has none')
        groups = np.asarray(group_ids, dtype=str)
        memberships = [set(groups[actual[part]].tolist()) for part in actual]
        if any(memberships[i] & memberships[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ConfigurationError('Research capture-group overlap across partitions')
        expected_algorithm = 'group_shuffle'
    else:
        expected_algorithm = 'stratified'
    for name in ('test', 'validation'):
        if policy.get(f'{name}_split_algorithm') != expected_algorithm:
            raise ConfigurationError('Research split does not use its declared splitting algorithm')


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def save_split_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_split_manifest(path: Path, dataset: Any, labels: np.ndarray, seed: int,
                        expected_hash: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f'Missing mandatory target split manifest: {path}; retrain target')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('version') != SPLIT_VERSION or manifest.get('seed') != int(seed):
        raise ConfigurationError('Split manifest version/seed mismatch')
    if manifest.get('identity') != dataset_identity(dataset, np.asarray(labels)):
        raise ConfigurationError('Dataset bytes, sample IDs, labels, or feature schema changed since target training')
    if expected_hash is not None and manifest_hash(manifest) != expected_hash:
        raise ConfigurationError('Target checkpoint / split manifest hash mismatch')
    ids = _ids(dataset, len(labels))
    selected = list(manifest['train_ids']) + list(manifest['validation_ids']) + list(manifest['test_ids'])
    if len(selected) != len(ids) or set(selected) != set(ids) or len(set(selected)) != len(ids):
        raise ConfigurationError('Split manifest does not partition the full dataset')
    return manifest


def split_indices(manifest: Mapping[str, Any], dataset: Any, partition: str) -> np.ndarray:
    if partition not in {'train', 'validation', 'test'}:
        raise ValueError('Unknown partition')
    ids = _ids(dataset, int(manifest['identity']['sample_count']))
    index_by_id = {value: index for index, value in enumerate(ids)}
    try:
        return np.asarray([index_by_id[value] for value in manifest[f'{partition}_ids']], dtype=np.int64)
    except KeyError as exc:
        raise ConfigurationError(f'Split contains unknown sample ID {exc}') from exc


def validate_registered_split_protocol(manifest: Mapping[str, Any],
                                       config: Mapping[str, Any], worker_seed: int) -> None:
    """Check caller claims against an identity-verified checkpoint split."""
    from workflow.seeds import role_seed
    saved = manifest['split_policy']
    options = dict(config.get('data_split') or {})
    budget = str((config.get('budget') or {}).get('name', 'smoke'))
    research = saved.get('policy') == 'research' or budget == 'research'
    if (budget not in {'research', 'smoke'} or options.get('policy', budget) != budget
            or (research and budget != saved.get('policy'))):
        raise ConfigurationError('Caller budget/data_split.policy disagrees with registered split policy')
    # Research claims include resolved defaults. Smoke retains its existing
    # explicit-only diagnostic overrides, without being relabeled research.
    if research or (config.get('seed_roles') or {}).get('split') is not None:
        if role_seed(config, 'split', worker_seed) != int(manifest['seed']):
            raise ConfigurationError('Configured split seed disagrees with registered split manifest')
    for name in ('test_fraction', 'validation_fraction'):
        if not research and name not in options:
            continue
        try:
            effective = float(options.get(name, .2))
            recorded = float(saved[name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f'Invalid split fraction {name}; expected a number strictly between 0 and 1') from exc
        if not 0 < effective < 1 or not 0 < recorded < 1:
            raise ConfigurationError(f'Split fraction {name} must be finite and strictly between 0 and 1')
        if effective != recorded:
            raise ConfigurationError(f'Effective {name} disagrees with registered split manifest')


def authorized_attack_training_indices(dataset: Any, labels: np.ndarray,
                                       config: Mapping[str, Any], seed: int,
                                       *, role: str = 'model') -> np.ndarray:
    """Bind offline generator fitting to the *trained classifier's* partition.

    In particular, a generator MUST NOT independently split a full dataset;
    this is the enforcement boundary for standalone train_attack workers.
    """
    config_block = config.get(role)
    if not isinstance(config_block, Mapping) or not config_block.get('checkpoint'):
        raise ConfigurationError(f'{role}.checkpoint is required for train_attack provenance')
    from workflow.seeds import role_seed
    source_seed = role_seed(config, 'surrogate' if role == 'surrogate_model' else 'target', seed)
    checkpoint = Path(str(config_block['checkpoint']).format(seed=int(source_seed)))
    if not checkpoint.is_file():
        raise ConfigurationError(f'Missing fitted classifier checkpoint: {checkpoint}')
    # The classifier checkpoint's small metadata sidecar binds attack fitting
    # to the exact split selected for model training. Loading the checkpoint
    # itself here would waste large estimator/optimizer allocations.
    sidecar = checkpoint.parent / 'training_provenance.json'
    if not sidecar.is_file():
        raise ConfigurationError(f'Missing classifier training provenance: {sidecar}; retrain classifier')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    if metadata.get('seed') != int(source_seed) or metadata.get('benchmark_protocol_version') != 2:
        raise ConfigurationError('Classifier training provenance version/seed mismatch')
    expected = metadata.get('split_manifest_sha256')
    if not isinstance(expected, str) or len(expected) != 64:
        raise ConfigurationError('Classifier training provenance has no valid split digest')
    split_seed = int((metadata.get('seed_roles') or {}).get('split', seed))
    manifest = load_split_manifest(
        checkpoint.parent / 'split_manifest.json', dataset, labels, split_seed,
        expected_hash=expected,
    )
    if metadata.get('dataset_identity') != manifest['identity']:
        raise ConfigurationError('Classifier training provenance dataset identity mismatch')
    validate_registered_split_protocol(manifest, config, seed)
    if manifest['split_policy']['policy'] == 'research':
        validate_research_split_manifest(manifest, labels, dataset)
    return split_indices(manifest, dataset, 'train')


def source_revision_sha256(root: Path | None = None) -> str:
    """Hash the shipped Python/config source tree when there is no Git metadata.

    The hash is provenance metadata; a different source hash is not itself
    proof of an incompatible checkpoint without a semantic schema comparison.
    """
    root = (Path(__file__).resolve().parents[1] if root is None else Path(root))
    digest = hashlib.sha256()
    for source in sorted(path for path in root.rglob('*') if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.sh')):
        relative = source.relative_to(root).as_posix().encode('utf-8')
        digest.update(len(relative).to_bytes(8, 'little'))
        digest.update(relative)
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def _update_fitted_estimator_digest(digest: Any, value: Any, ancestors: set[int]) -> None:
    """Canonical, content-only digest of fitted sklearn estimator object graphs.

    Unlike pickle/joblib's object-graph encodings, this is invariant under
    deepcopy and checkpoint reloading: sklearn tree nodes are represented by
    their explicit __getstate__ arrays, not Python object addresses. Numeric
    buffers are hashed in bounded pieces; any unsupported state fails closed.
    """
    import struct

    cls = type(value)
    typ = (cls.__module__ + '.' + cls.__qualname__).encode('utf-8')
    digest.update(len(typ).to_bytes(4, 'big') + typ)
    if value is None:
        return
    if isinstance(value, bool):
        digest.update(b'1' if value else b'0')
        return
    if isinstance(value, int):
        digest.update(str(value).encode('ascii') + b';')
        return
    if isinstance(value, float):
        digest.update(struct.pack('!d', value))
        return
    if isinstance(value, (str, bytes)):
        raw = value.encode('utf-8') if isinstance(value, str) else value
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
        return
    if isinstance(value, np.dtype):
        digest.update(repr(value.descr if value.names else value.str).encode('utf-8'))
        return
    if isinstance(value, np.generic):
        digest.update(value.dtype.str.encode('ascii') + b'\0')
        digest.update(value.tobytes())
        return
    if isinstance(value, np.ndarray):
        digest.update(repr((value.shape, value.dtype.descr if value.dtype.names else value.dtype.str)).encode('utf-8'))
        if value.dtype.names:
            # sklearn's Cython Tree.nodes has *uninitialized structure padding*.
            # The void bytes between named fields change under deepcopy despite
            # identical learned decision nodes. Hash only typed field values.
            for name in value.dtype.names:
                _update_fitted_estimator_digest(digest, name, ancestors)
                _update_fitted_estimator_digest(digest, value[name], ancestors)
        elif value.dtype.hasobject:
            for element in value.flat:
                _update_fitted_estimator_digest(digest, element, ancestors)
        elif value.flags.c_contiguous:
            data = memoryview(value).cast('B')
            for start in range(0, data.nbytes, 4 * 1024 * 1024):
                digest.update(data[start:start + 4 * 1024 * 1024])
        elif not value.ndim:
            digest.update(value.tobytes())
        else:
            # A strided field/array is copied only in <= 4 MiB row chunks.
            row_bytes = int(value.dtype.itemsize * np.prod(value.shape[1:], dtype=np.int64))
            rows = max(1, (4 * 1024 * 1024) // max(row_bytes, 1))
            for start in range(0, value.shape[0], rows):
                data = np.ascontiguousarray(value[start:start + rows])
                digest.update(memoryview(data).cast('B'))
        return
    object_id = id(value)
    if object_id in ancestors:
        raise ConfigurationError('Cyclic fitted estimator state cannot be certified')
    ancestors.add(object_id)
    try:
        if isinstance(value, dict):
            if not all(isinstance(k, (str, int)) for k in value):
                raise ConfigurationError('Unsupported fitted estimator mapping key')
            keys = sorted(value, key=lambda k: (type(k).__name__, str(k)))
            digest.update(len(keys).to_bytes(8, 'big'))
            for key in keys:
                _update_fitted_estimator_digest(digest, key, ancestors)
                _update_fitted_estimator_digest(digest, value[key], ancestors)
        elif isinstance(value, (tuple, list)):
            digest.update(len(value).to_bytes(8, 'big'))
            for element in value:
                _update_fitted_estimator_digest(digest, element, ancestors)
        elif isinstance(value, np.random.RandomState):
            _update_fitted_estimator_digest(digest, value.get_state(), ancestors)
        elif isinstance(value, np.random.Generator):
            _update_fitted_estimator_digest(digest, value.bit_generator.state, ancestors)
        elif hasattr(value, '__dict__'):
            _update_fitted_estimator_digest(digest, vars(value), ancestors)
        elif callable(getattr(value, '__getstate__', None)):
            state = value.__getstate__()
            if state is None:
                raise ConfigurationError('Opaque fitted estimator state cannot be certified')
            _update_fitted_estimator_digest(digest, state, ancestors)
        else:
            raise ConfigurationError(f'Unsupported fitted estimator state type {cls.__name__}')
    finally:
        ancestors.remove(object_id)


def frozen_model_state_sha256(wrapper: Any) -> str:
    """Hash fitted classifier *state*, not its path, split, seed or wrapper object.

    Hashes PyTorch parameter/buffer bytes in deterministic name order; sklearn
    Pickler writes to a digest sink so it never materializes an extra giant
    serialized estimator in RAM. Must be called after fit, before detector fit.
    """
    import torch

    digest = hashlib.sha256()
    if isinstance(getattr(wrapper, 'model', None), torch.nn.Module):
        digest.update(b'nids:pytorch-state-v1\0')
        state = wrapper.model.state_dict()
        for name, tensor in sorted(state.items()):
            if not isinstance(tensor, torch.Tensor):
                raise ConfigurationError(f'Non-tensor model-state entry: {name}')
            name_bytes = name.encode('utf-8')
            digest.update(len(name_bytes).to_bytes(4, 'little'))
            digest.update(name_bytes)
            data = tensor.detach().to(device='cpu').contiguous().numpy()
            dtype_bytes = data.dtype.str.encode('ascii')
            digest.update(dtype_bytes + b'\0')
            digest.update(json.dumps(data.shape).encode('ascii') + b'\0')
            # memoryview is read-only here; hash in bounded blocks.
            byte_view = memoryview(data).cast('B')
            for start in range(0, byte_view.nbytes, 4 * 1024 * 1024):
                digest.update(byte_view[start:start + 4 * 1024 * 1024])
    elif getattr(wrapper, 'estimator', None) is not None:
        # Hash the fitted logical state rather than pickle/joblib object-graph
        # bytes: a deep-copied target must never masquerade as an independent
        # surrogate. This changes the sklearn fingerprint contract and requires
        # retraining existing sklearn checkpoints and dependent detectors.
        digest.update(b'nids:sklearn-fitted-state-v2\0')
        _update_fitted_estimator_digest(digest, wrapper.estimator, set())
    else:
        raise ConfigurationError('Cannot fingerprint an unfitted or unsupported classifier')
    return digest.hexdigest()


def generator_artifact_provenance(
    *, manifest: Mapping[str, Any], source_state_sha256: str,
    source_role: str, attack_config: Mapping[str, Any], seed: int,
    source_checkpoint_sha256: str | None = None,
    input_shape: Sequence[int], feature_names: Sequence[str],
    class_mapping: Mapping[str, int], access_mode: str,
) -> dict[str, Any]:
    """Small immutable generator contract, independent of log directories.

    `source_state_sha256` refers only to the *authorized surrogate* when
    source_role=surrogate_model. A model-independent generator may use its
    permitted partition's manifest but never receives target model weights.
    """
    if source_role not in {'surrogate_model', 'model'} or access_mode not in {'transfer', 'model_independent'}:
        raise ConfigurationError('Unsupported generator provenance source/access mode')
    if (source_role == 'model') != (access_mode == 'model_independent'):
        raise ConfigurationError('Generator source role and model-access policy contradict')
    if source_role == 'surrogate_model' and (not isinstance(source_state_sha256, str) or len(source_state_sha256) != 64):
        raise ConfigurationError('Generator requires exact authorized surrogate-state fingerprint')
    if not manifest.get('train_ids'):
        raise ConfigurationError('Generator provenance requires explicit nonempty training IDs')
    attack = dict(attack_config)
    attack.pop('checkpoint', None)
    attack.pop('output_dir', None)  # relocatable raw-PCAP output, not an algorithm input
    attack.pop('variant', None)
    params = attack.get('params')
    if isinstance(params, Mapping):
        params = dict(params)
        params.pop('checkpoint', None)
        attack['params'] = params
    algorithm_json = json.dumps(attack, sort_keys=True, separators=(',', ':'), default=str)
    identity = manifest.get('identity')
    if not isinstance(identity, dict):
        raise ConfigurationError('Generator provenance requires verified dataset transformation identity')
    return {
        'contract_version': 1,
        'seed': int(seed),
        'attack_name': str(attack.get('name', '')),
        'access_mode': access_mode,
        'source_role': source_role,
        'authorized_source_state_sha256': source_state_sha256 if source_role == 'surrogate_model' else None,
        'authorized_source_checkpoint_sha256': source_checkpoint_sha256 if source_role == 'surrogate_model' else None,
        'split_manifest_sha256': manifest_hash(manifest),
        'training_ids_sha256': _digest_strings(tuple(str(v) for v in manifest['train_ids'])),
        'dataset_identity': identity,
        'class_mapping': dict(class_mapping),
        'feature_names': list(feature_names),
        'input_shape': [int(v) for v in input_shape],
        'algorithm_config_sha256': hashlib.sha256(algorithm_json.encode('utf-8')).hexdigest(),
    }


def verify_generator_artifact_provenance(payload: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
    """Missing metadata is invalid: incompatible generators require retraining."""
    if expected is None:
        raise ConfigurationError('Generator has no independently verified runtime provenance; use evaluate_attack workflow')
    actual = payload.get('artifact_provenance')
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ConfigurationError('Generator artifact provenance differs from permitted training IDs, source, schema, hyperparameters or seed; retrain generator')


def trained_generator_provenance(
    dataset: Any, labels: np.ndarray, config: Mapping[str, Any], seed: int,
    *, source_role: str, input_shape: Sequence[int], feature_names: Sequence[str],
) -> dict[str, Any]:
    """Derive training contract from the already-authorized classifier sidecar."""
    # Performs a fresh manifest/source and membership audit before writing.
    authorized_attack_training_indices(dataset, labels, config, seed, role=source_role)
    source_cfg = config[source_role]
    checkpoint = Path(str(source_cfg['checkpoint']).format(seed=int(seed)))
    metadata = json.loads((checkpoint.parent / 'training_provenance.json').read_text(encoding='utf-8'))
    manifest = json.loads((checkpoint.parent / 'split_manifest.json').read_text(encoding='utf-8'))
    source_hash = metadata.get('model_state_sha256') if source_role == 'surrogate_model' else ''
    if source_role == 'surrogate_model' and (not isinstance(source_hash, str) or len(source_hash) != 64):
        raise ConfigurationError('Authorized surrogate missing final-state fingerprint; retrain surrogate')
    return generator_artifact_provenance(
        manifest=manifest, source_state_sha256=source_hash, source_role=source_role,
        source_checkpoint_sha256=(_file_sha256(checkpoint) if source_role == 'surrogate_model' else None),
        attack_config=config['attack'], seed=seed, input_shape=input_shape,
        feature_names=feature_names, class_mapping=dict(dataset.class_mapping),
        access_mode='transfer' if source_role == 'surrogate_model' else 'model_independent',
    )
