"""Versioned experiment storage, immutable checkpoint identities and seed pairing.

The directory is the *location*, not the scientific identity: registered SHA-256
hashes and split/transform provenance remain independently verified at load time.
No fallback to the latest timestamp, another seed, or a legacy artifact root exists.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping
from uuid import uuid4

from core.exceptions import ConfigurationError
from core.provenance import _file_sha256, source_revision_sha256

_VERSION = 1
_SAFE = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _part(value: Any) -> str:
    value = str(value)
    if not value or value in {'.', '..'} or not _SAFE.fullmatch(value):
        raise ConfigurationError(f"Unsafe experiment artifact path component: {value!r}")
    return value


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _generator_configuration_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Hash scientific generator settings, not the relocatable artifact locator."""
    result = {key: value for key, value in config.items()
              if key not in {'checkpoint', 'output_dir'}}
    if isinstance(result.get('params'), Mapping):
        result['params'] = {key: value for key, value in result['params'].items()
                            if key != 'checkpoint'}
    return result


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_experiment_dir(root: str | Path = 'runs') -> Path:
    """Create a collision-resistant version under the common runs directory."""
    base = Path(root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        candidate = base / f'experiment_{stamp}_{uuid4().hex[:8]}'
        try:
            candidate.mkdir(mode=0o750)
            return candidate
        except FileExistsError:
            continue


class ExperimentStore:
    """Single canonical registry for training, generation and frozen evaluation."""

    def __init__(self, directory: str | Path, *, dataset: str | None = None,
                 create: bool = False) -> None:
        if not directory:
            raise ConfigurationError('experiment_dir is required; select an existing experiment or create one for training')
        self.root = Path(directory).expanduser().resolve()
        self.manifest_file = self.root / 'experiment_manifest.json'
        self.lock_file = self.root / '.experiment.lock'
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            with self._locked():
                if not self.manifest_file.exists():
                    _write_json(self.manifest_file, {
                        'schema_version': _VERSION,
                        'experiment_id': self.root.name,
                        'created_utc': datetime.now(timezone.utc).isoformat(),
                        'source_revision_sha256': source_revision_sha256(),
                        'dataset': str(dataset),
                        'splits': {}, 'artifacts': {}, 'executions': {},
                    })
        self.read(dataset=dataset)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if not self.root.is_dir():
            raise ConfigurationError(f'Experiment directory not found: {self.root}')
        with self.lock_file.open('a+b') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def read(self, *, dataset: str | None = None) -> dict[str, Any]:
        if not self.manifest_file.is_file():
            raise ConfigurationError(f'Missing experiment_manifest.json in {self.root}; no implicit artifact discovery')
        try:
            result = json.loads(self.manifest_file.read_text(encoding='utf-8'))
        except (ValueError, OSError) as exc:
            raise ConfigurationError(f'Invalid experiment manifest: {self.manifest_file}') from exc
        if result.get('schema_version') != _VERSION:
            raise ConfigurationError('Unsupported experiment manifest version; explicit verified migration required')
        if dataset is not None and result.get('dataset') != str(dataset):
            raise ConfigurationError('Experiment dataset differs from selected dataset')
        if result.get('source_revision_sha256') != source_revision_sha256():
            raise ConfigurationError('Experiment source revision differs; create a new experiment version')
        return result

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ConfigurationError(f'Artifact escapes experiment directory: {path}') from exc

    @staticmethod
    def model_role(name: str) -> str:
        return 'surrogates' if str(name) == 'jmila_surrogate' else 'targets'

    def model_path(self, name: str, defense: str, seed: int, *, role: str | None = None) -> Path:
        family = role or self.model_role(name)
        if family not in {'targets', 'surrogates'}:
            raise ConfigurationError('Model artifact role must be targets or surrogates')
        return (self.root / 'checkpoints' / family / _part(name) /
                _part(defense) / f'seed{int(seed)}' / 'nids.pt')

    def generator_path(self, name: str, variant: str, seed: int) -> Path:
        return (self.root / 'checkpoints' / 'generators' / _part(name) /
                _part(variant) / f'seed{int(seed)}.pt')

    def detector_path(self, target_path: Path) -> Path:
        return target_path.parent / 'detector.pt'

    def validate_template(self, configured: str | None, expected: Path, seed: int,
                          *, role: str) -> Path:
        if configured is None or not str(configured):
            raise ConfigurationError(f'Missing {role}.checkpoint in experiment configuration')
        actual = Path(str(configured).format(seed=seed)).expanduser().resolve()
        if actual != expected.resolve():
            raise ConfigurationError(
                f'{role}.checkpoint must resolve to registered experiment path {expected}, got {actual}. '
                'Explicit external, legacy or fixed-seed checkpoint paths are not supported.'
            )
        return actual

    def record_key(self, kind: str, path: Path) -> str:
        return kind + ':' + self.relative(path)

    def preflight_new(self, kind: str, path: Path) -> None:
        key = self.record_key(kind, path)
        if path.exists() or key in self.read()['artifacts']:
            raise ConfigurationError(f'Immutable experiment artifact already exists: {path}; create a new experiment version')

    def new_staging_directory(self, final_directory: Path) -> Path:
        self.preflight_new('model', final_directory / 'nids.pt')
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix='.staging-', dir=final_directory.parent))

    @staticmethod
    def _canonical_split(manifest: Mapping[str, Any]) -> dict[str, Any]:
        identity = dict(manifest.get('identity', {}))
        transform = identity.get('transformation')
        if isinstance(transform, Mapping):
            identity['transformation'] = {k: v for k, v in transform.items() if k != 'flatten'}
        return {
            'seed': int(manifest['seed']),
            'train_ids': list(manifest['train_ids']),
            'validation_ids': list(manifest['validation_ids']),
            'test_ids': list(manifest['test_ids']),
            'physical_identity': identity,
        }

    def validate_partition(self, manifest: Mapping[str, Any]) -> None:
        """Reject a mismatched same-seed data split *before* fitting a classifier."""
        canonical = self._canonical_split(manifest)
        registered = self.read()['splits'].get(str(canonical['seed']))
        if registered is None:
            return
        path = self.root / registered['path']
        if not path.is_file():
            raise ConfigurationError('Registered experiment split is missing')
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != canonical or _digest_json(existing) != registered['sha256']:
            raise ConfigurationError('Configured split membership or physical identity differs from this experiment seed')

    def publish_model(self, staged: Path, final_path: Path, split_manifest: Mapping[str, Any],
                      *, state_sha256: str, seed: int, split_hash: str) -> None:
        """Publish a fully trained directory, then register it under a short lock."""
        final_directory = final_path.parent
        if not (staged / 'nids.pt').is_file() or not (staged / 'split_manifest.json').is_file():
            raise ConfigurationError('Incomplete staged model; refusing to publish')
        canonical = self._canonical_split(split_manifest)
        split_seed = str(canonical['seed'])
        split_relative = f'splits/seed{split_seed}.json'
        key = self.record_key('model', final_path)
        with self._locked():
            data = self.read()
            if key in data['artifacts'] or final_directory.exists():
                raise ConfigurationError(f'Immutable model path already occupied: {final_path}')
            stored = data['splits'].get(split_seed)
            if stored is not None:
                existing = json.loads((self.root / stored['path']).read_text(encoding='utf-8'))
                if existing != canonical or stored['sha256'] != _digest_json(existing):
                    raise ConfigurationError('Incompatible split IDs/source/physical schema for this experiment seed')
            else:
                _write_json(self.root / split_relative, canonical)
                data['splits'][split_seed] = {'path': split_relative, 'sha256': _digest_json(canonical)}
            staged.replace(final_directory)  # same filesystem; never expose partial weights
            data['artifacts'][key] = {
                'kind': 'model', 'path': self.relative(final_path),
                'sha256': _file_sha256(final_path),
                'model_state_sha256': str(state_sha256),
                'seed': int(seed), 'split_seed': int(split_seed),
                'split_manifest_sha256': str(split_hash),
                'split_membership_sha256': data['splits'][split_seed]['sha256'],
                'training_provenance_sha256': _file_sha256(final_directory / 'training_provenance.json'),
                'model_split_file_sha256': _file_sha256(final_directory / 'split_manifest.json'),
            }
            detector = final_directory / 'detector.pt'
            if detector.is_file():
                data['artifacts'][self.record_key('detector', detector)] = {
                    'kind': 'detector', 'path': self.relative(detector),
                    'sha256': _file_sha256(detector), 'seed': int(seed),
                    'target_model_state_sha256': str(state_sha256),
                    'split_manifest_sha256': str(split_hash),
                }
            _write_json(self.manifest_file, data)

    @contextmanager
    def claim_generator(self, path: Path) -> Iterator[None]:
        """Prevent two generator workers writing the same seed/variant artifact."""
        self.preflight_new('generator', path)
        path.parent.mkdir(parents=True, exist_ok=True)
        claim = path.with_suffix(path.suffix + '.claim')
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ConfigurationError(f'Generator checkpoint has another writer: {path}') from exc
        os.close(fd)
        try:
            yield
        finally:
            claim.unlink(missing_ok=True)

    def publish_generator(self, path: Path, *, seed: int, configuration: Mapping[str, Any]) -> None:
        key = self.record_key('generator', path)
        if not path.is_file():
            raise ConfigurationError(f'Generator training did not produce configured artifact: {path}')
        entry = {'kind': 'generator', 'path': self.relative(path),
                 'sha256': _file_sha256(path), 'seed': int(seed),
                 'configuration_sha256': _digest_json(_generator_configuration_identity(configuration))}
        with self._locked():
            data = self.read()
            if key in data['artifacts']:
                raise ConfigurationError(f'Generator already registered: {path}')
            data['artifacts'][key] = entry
            _write_json(self.manifest_file, data)

    def publish_detector(self, path: Path, payload: Mapping[str, Any], *, seed: int,
                         target_state_sha256: str, split_hash: str) -> None:
        """Atomically register only a *fully calibrated* Lee detector artifact.

        The model checkpoint stays immutable. A new detector is installed only
        after its complete state and target/split binding have been verified.
        """
        import torch
        state = dict(payload.get('state_dict') or {})
        if (not state.get('layers') or state.get('calibration_coef') is None
                or state.get('calibration_membership_sha256') is None
                or state.get('target_model_state_sha256') != target_state_sha256
                or state.get('split_manifest_sha256') != split_hash):
            raise ConfigurationError('Cannot publish uncalibrated or misbound Lee detector')
        target_path = path.parent / 'nids.pt'
        self.require('model', target_path, seed=seed)
        self.preflight_new('detector', path)
        temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
        try:
            torch.save(dict(payload), temporary)
            with self._locked():
                data = self.read()
                key = self.record_key('detector', path)
                if key in data['artifacts'] or path.exists():
                    raise ConfigurationError('Immutable detector checkpoint already registered')
                model_entry = data['artifacts'].get(self.record_key('model', target_path))
                if (model_entry is None
                        or model_entry.get('model_state_sha256') != target_state_sha256
                        or model_entry.get('split_manifest_sha256') != split_hash):
                    raise ConfigurationError('Lee checkpoint target/split differs from registered classifier')
                temporary.replace(path)
                data['artifacts'][key] = {
                    'kind': 'detector', 'path': self.relative(path), 'sha256': _file_sha256(path),
                    'seed': int(seed), 'target_model_state_sha256': target_state_sha256,
                    'split_manifest_sha256': split_hash,
                    'validation_membership_sha256': payload.get('validation_membership_sha256'),
                    'surrogate_checkpoint_sha256': payload.get('surrogate_checkpoint_sha256'),
                }
                _write_json(self.manifest_file, data)
        finally:
            temporary.unlink(missing_ok=True)

    def require(self, kind: str, path: Path, *, seed: int,
                configuration: Mapping[str, Any] | None = None) -> dict[str, Any]:
        entry = self.read()['artifacts'].get(self.record_key(kind, path))
        if entry is None:
            raise ConfigurationError(f'Unregistered {kind} checkpoint: {path}; complete training in this experiment first')
        if entry['kind'] != kind or int(entry['seed']) != int(seed) or not path.is_file():
            raise ConfigurationError(f'{kind} checkpoint missing or seed mismatch: {path}')
        if _file_sha256(path) != entry['sha256']:
            raise ConfigurationError(f'{kind} checkpoint bytes changed after registration: {path}')
        if kind == 'model':
            if _file_sha256(path.parent / 'training_provenance.json') != entry.get('training_provenance_sha256'):
                raise ConfigurationError('Model provenance sidecar changed after registration')
            if _file_sha256(path.parent / 'split_manifest.json') != entry.get('model_split_file_sha256'):
                raise ConfigurationError('Model split-manifest bytes changed after registration')
            split = self.read()['splits'].get(str(entry.get('split_seed')))
            if split is None or not (self.root / split['path']).is_file():
                raise ConfigurationError('Registered model has no canonical split record')
            canonical = json.loads((self.root / split['path']).read_text(encoding='utf-8'))
            if _digest_json(canonical) != split['sha256'] or split['sha256'] != entry.get('split_membership_sha256'):
                raise ConfigurationError('Canonical experiment split was modified or does not match registered model')
        if configuration is not None and entry.get('configuration_sha256') != _digest_json(_generator_configuration_identity(configuration)):
            raise ConfigurationError(f'{kind} checkpoint configuration differs from training: {path}')
        return dict(entry)

    def record_execution(self, execution_id: str, *, action: str, seeds: list[int],
                         output_dir: Path) -> None:
        with self._locked():
            data = self.read()
            if execution_id in data['executions']:
                raise ConfigurationError(f'Duplicate execution ID: {execution_id}')
            data['executions'][execution_id] = {
                'action': str(action), 'seeds': list(map(int, seeds)),
                'path': self.relative(output_dir), 'status': 'running',
            }
            _write_json(self.manifest_file, data)

    def finish_execution(self, execution_id: str, *, success: bool) -> None:
        with self._locked():
            data = self.read()
            if execution_id not in data['executions']:
                raise ConfigurationError(f'Execution ID not registered: {execution_id}')
            entry = data['executions'][execution_id]
            entry['status'] = 'completed' if success else 'failed'
            if success:
                execution_dir = self.root / entry['path']
                output_files = [execution_dir / name for name in
                                ('resolved_config.json', 'summary.json', 'seed_metrics.csv')]
                for pattern in ('per_seed/*.json', 'per_seed/*.npz', 'reports/*.csv'):
                    output_files.extend(sorted(execution_dir.glob(pattern)))
                entry['output_sha256'] = {
                    path.relative_to(execution_dir).as_posix(): _file_sha256(path)
                    for path in output_files if path.is_file()
                }
            _write_json(self.manifest_file, data)


def preflight_action(store: ExperimentStore, config: Mapping[str, Any],
                     action: str, seeds: list[int]) -> None:
    """Validate all seed-to-artifact mappings before costly worker dispatch.

    This is an early failure check, not a replacement for the worker's full
    split, physical-source, model-state and generator-provenance validation.
    """
    from workflow.seeds import role_seed

    model = config['model']
    attack = config.get('attack') or {}
    surrogate = config.get('surrogate_model') or {}
    enabled_detector = bool((config.get('detector') or {}).get('enabled', False))
    for seed in seeds:
        if action == 'train_nids':
            path = store.model_path(model['name'],
                                    model.get('training', {}).get('defense_mode', 'none'), seed,
                                    role=model.get('artifact_role'))
            store.validate_template(model.get('checkpoint'), path, seed, role='model')
            store.preflight_new('model', path)
        elif action == 'evaluate_attack':
            target_seed = role_seed(config, 'target', seed)
            path = store.model_path(model['name'],
                                    model.get('training', {}).get('defense_mode', 'none'),
                                    target_seed, role=model.get('artifact_role'))
            store.validate_template(model.get('checkpoint'), path, target_seed, role='model')
            store.require('model', path, seed=target_seed)
            if enabled_detector:
                detector_seed = role_seed(config, 'detector', target_seed)
                if detector_seed != target_seed:
                    raise ConfigurationError('Detector seed must identify this frozen target')
                store.require('detector', store.detector_path(path), seed=target_seed)
            if str((config.get('evaluation') or {}).get('attack_access', 'transfer')) == 'transfer':
                if not surrogate or not surrogate.get('enabled', True):
                    raise ConfigurationError('Transfer evaluation requires an authorized independent surrogate')
                source_seed = role_seed(config, 'surrogate', seed)
                src_path = store.model_path(surrogate['name'], 'none', source_seed, role='surrogates')
                store.validate_template(surrogate.get('checkpoint'), src_path,
                                        source_seed, role='surrogate_model')
                store.require('model', src_path, seed=source_seed)
            if str(attack.get('name')) in {'amd_gan', 'transformer_gan', 'nids_gan', 'tdtce', 'a2pm', 'mirage', 'deepred'}:
                attack_seed = role_seed(config, 'attack', seed)
                generator = store.generator_path(attack['name'], attack.get('variant', 'default'), attack_seed)
                configured = attack.get('checkpoint', (attack.get('params') or {}).get('checkpoint'))
                store.validate_template(configured, generator, attack_seed, role='attack')
                store.require('generator', generator, seed=attack_seed, configuration=attack)
        elif action == 'evaluate_distribution_stress':
            if str(attack.get('name')) != 'amd_gan':
                raise ConfigurationError('AMD distribution stress cannot use a per-record attack profile')
            target_seed = role_seed(config, 'target', seed)
            path = store.model_path(model['name'], model.get('training', {}).get('defense_mode', 'none'), target_seed)
            store.validate_template(model.get('checkpoint'), path, target_seed, role='model')
            store.require('model', path, seed=target_seed)
            source_seed = role_seed(config, 'surrogate', seed)
            if not surrogate or not surrogate.get('enabled', True):
                raise ConfigurationError('Registered AMD study requires an independently trained surrogate')
            source_path = store.model_path(surrogate['name'], 'none', source_seed, role='surrogates')
            store.validate_template(surrogate.get('checkpoint'), source_path, source_seed, role='surrogate_model')
            store.require('model', source_path, seed=source_seed)
            attack_seed = role_seed(config, 'attack', seed)
            generator = store.generator_path('amd_gan', attack.get('variant', 'default'), attack_seed)
            store.validate_template(attack.get('checkpoint'), generator, attack_seed, role='attack')
            store.require('generator', generator, seed=attack_seed, configuration=attack)
        elif action == 'calibrate_detector':
            if not enabled_detector or (config['detector'].get('name') != 'mahalanobis_multilayer_val_calibrated'):
                raise ConfigurationError('Detector calibration requires the selected Lee multilayer profile')
            target_seed = role_seed(config, 'target', seed)
            if role_seed(config, 'detector', target_seed) != target_seed:
                raise ConfigurationError('Detector must use this exact frozen target seed')
            path = store.model_path(model['name'], model.get('training', {}).get('defense_mode', 'none'), target_seed)
            store.validate_template(model.get('checkpoint'), path, target_seed, role='model')
            store.require('model', path, seed=target_seed)
            source_seed = role_seed(config, 'surrogate', seed)
            if not surrogate or not surrogate.get('enabled', True):
                raise ConfigurationError('Lee validation attack requires independent registered surrogate')
            source_path = store.model_path(surrogate['name'], 'none', source_seed, role='surrogates')
            store.validate_template(surrogate.get('checkpoint'), source_path, source_seed, role='surrogate_model')
            store.require('model', source_path, seed=source_seed)
            if path == source_path:
                raise ConfigurationError('Lee calibration source cannot be its target checkpoint')
            store.preflight_new('detector', store.detector_path(path))
        elif action == 'train_attack':
            if not surrogate or not surrogate.get('enabled', True):
                raise ConfigurationError('Offline training requires an independent authorized surrogate')
            source_seed = role_seed(config, 'surrogate', seed)
            src_path = store.model_path(surrogate['name'], 'none', source_seed, role='surrogates')
            store.validate_template(surrogate.get('checkpoint'), src_path,
                                    source_seed, role='surrogate_model')
            store.require('model', src_path, seed=source_seed)
            generator = store.generator_path(attack['name'], attack.get('variant', 'default'), seed)
            configured = attack.get('checkpoint', (attack.get('params') or {}).get('checkpoint'))
            store.validate_template(configured, generator, seed, role='attack')
            store.preflight_new('generator', generator)
