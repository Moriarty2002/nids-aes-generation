"""A2PM author ordered class patterns and generate/patience controller.

Mandatory differences: frozen authorized TRAIN fitting, physical float64 packet
rows, exact ownership/occupancy protection and surrogate-only forward encoding.
Unknown classes fail; an unavailable legal locked combination is a no-op. No
cross-topology/global RNG identity or author-optimal packet settings is claimed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import torch

from attacks.base import AttackResult, FeatureSpaceAttack
from core.dual_representation import encode_normalized32
from core.exceptions import ConfigurationError
from core.provenance import (authorized_attack_training_indices, trained_generator_provenance,
                             verify_generator_artifact_provenance)
from core.registry import DATASETS, register_attack

_FORMAT_VERSION = 5


def _checkpoint_path(template: str | Path, seed: int) -> Path:
    return Path(str(template).format(seed=int(seed))).expanduser()


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _fit_patterns(observations: np.ndarray, specs: Sequence[Mapping[str, Any]]) -> list[dict[str, np.ndarray]]:
    states = []
    for spec in specs:
        values = observations[:, spec['features']]
        if spec['type'] == 'interval':
            states.append({'minimum': values.min(0), 'maximum': values.max(0)})
        else:
            states.append({'combinations': np.unique(values, axis=0)})
    return states


def _transform_patterns(values: np.ndarray, specs: Sequence[Mapping[str, Any]],
                        states: Sequence[Mapping[str, Any]], rngs: Sequence[Any],
                        permitted: np.ndarray) -> np.ndarray:
    """Author feature/row draw order; protect packet truth after every pattern."""
    out = values.copy()
    for spec, state, rng in zip(specs, states, rngs):
        columns = list(spec['features'])
        probability = spec.get('probability', .5)
        if spec['type'] == 'interval':
            for j, column in enumerate(columns):
                low, high = float(state['minimum'][j]), float(state['maximum'][j])
                if low == high:
                    continue
                for row in range(len(out)):
                    if not permitted[row, column] or rng.random() >= probability:
                        continue
                    value = out[row, column]
                    if spec.get('missing_value') is not None and value == spec['missing_value']:
                        continue
                    ratio = spec.get('ratio', .1)
                    maximum = spec.get('max_ratio')
                    if maximum is not None and maximum != ratio:
                        ratio += (maximum - ratio) * rng.random()
                    magnitude = (high - low) * ratio
                    direction = rng.random()
                    if (direction >= .5 and value < high) or (direction < .5 and value <= low):
                        proposed = min(value + magnitude, high)
                    else:
                        proposed = max(value - magnitude, low)
                    if column in spec.get('integer_features', ()):
                        proposed = round(proposed)
                    out[row, column] = proposed
        else:
            combinations = np.asarray(state['combinations'])
            pattern_locks = [columns.index(i) for i in spec.get('locked_features', ())]
            mutable = [i for i in columns if i not in spec.get('locked_features', ())]
            for row in range(len(out)):
                if not any(permitted[row, i] for i in mutable) or rng.random() >= probability:
                    continue
                # Immutable coordinates become additional conditional locks;
                # restoring them after an unconditional draw could invent a tuple.
                locks = sorted(set(pattern_locks) | {j for j,column in enumerate(columns) if not permitted[row,column]})
                if locks:
                    matching = np.flatnonzero(np.all(combinations[:, locks] == out[row, np.asarray(columns)[locks]], axis=1))
                    if not len(matching):
                        continue
                    selected = int(rng.choice(matching, size=None, replace=False))
                else:
                    selected = int(rng.integers(0, len(combinations), endpoint=False))
                for column in mutable:
                    if permitted[row, column]:
                        out[row, column] = combinations[selected, columns.index(column)]
    return out
def _read_classifier_schema(
    config: Mapping[str, Any], seed: int
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    """Use the same frozen normalization authority as AMD-GAN.

    Re-fitting a scaler on the attack's training subset would make observed
    categorical values incomparable to the ModelWrapper's inference inputs.
    """
    surrogate = config.get("surrogate_model")
    if isinstance(surrogate, Mapping) and surrogate.get("enabled", True):
        model_cfg = surrogate
    else:
        model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping) or not model_cfg.get("checkpoint"):
        raise ValueError("A2PM training requires model/surrogate_model.checkpoint")
    path = _checkpoint_path(str(model_cfg["checkpoint"]), seed)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
    # Framework model checkpoints can contain NumPy metadata; only load a
    # trusted local checkpoint, in accordance with AMD-GAN's existing loader.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Classifier checkpoint must contain a mapping")
    shape = tuple(int(v) for v in payload["input_shape"])
    if len(shape) not in (1, 2) or min(shape) < 1:
        raise ValueError(f"Unsupported model input_shape: {shape}")
    mean = np.asarray(payload["normalization_mean"], dtype=np.float32)
    std = np.asarray(payload["normalization_std"], dtype=np.float32)
    if mean.shape != std.shape or not np.isfinite(mean).all():
        raise ValueError("Invalid classifier normalization mean")
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid classifier normalization standard deviation")
    names = tuple(str(s).upper() for s in payload.get("feature_names", ()))
    return shape, mean, std, names


@register_attack('a2pm')
@dataclass(kw_only=True)
class A2PMAttack(FeatureSpaceAttack):
    attack_name: ClassVar[str] = 'a2pm'
    operates_on_physical_features: ClassVar[bool] = True
    query_based: ClassVar[bool] = True
    requires_gradients: ClassVar[bool] = False
    patterns: Sequence[Mapping[str, Any]] = field(default_factory=lambda: [
        dict(type='interval', features=[1], probability=.6, ratio=.1, max_ratio=.3),
        dict(type='combination', features=[0,2,3,4,5], locked_features=[2,4], probability=.4),
    ])
    class_patterns: Mapping[str, Any] = field(default_factory=dict)
    max_iterations: int = 20
    patience: int = 2
    target_label: int = 0
    seed: int = 42
    checkpoint: str | None = None
    _loaded_path: Path | None = field(default=None, init=False, repr=False)
    _states: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _class_specs: dict[int, Any] = field(default_factory=dict, init=False, repr=False)
    _normalization_mean: torch.Tensor | None = field(default=None, init=False, repr=False)
    _normalization_std: torch.Tensor | None = field(default=None, init=False, repr=False)
    _input_shape: tuple[int, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or int(self.max_iterations) != self.max_iterations or self.max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')
        if isinstance(self.patience, bool) or int(self.patience) != self.patience or not 0 <= self.patience < self.max_iterations:
            raise ValueError('patience must be 0 or lower than max_iterations')
        self.patterns = self._validated_patterns(self.patterns)
        self.class_patterns = {str(k): self._validated_patterns(v) if v is not None else None
                               for k,v in self.class_patterns.items()}
        if self.feature_names:
            self._validate_schema(self.feature_names)

    @staticmethod
    def _validated_patterns(patterns: Any) -> list[dict[str, Any]]:
        if not isinstance(patterns, (list,tuple)) or not patterns:
            raise ValueError('patterns must be a nonempty ordered sequence')
        result=[]
        for raw in patterns:
            spec=deepcopy(dict(raw))
            kind=spec.get('type')
            allowed={'type','features','probability'} | ({'ratio','max_ratio','integer_features','missing_value'} if kind=='interval' else {'locked_features'})
            if kind not in {'interval','combination'} or set(spec)-allowed:
                raise ValueError('Unknown pattern type or setting')
            features=spec.get('features')
            if not features or any(isinstance(i,bool) or not isinstance(i,int) or i<0 for i in features) or len(set(features))!=len(features):
                raise ValueError('Pattern features must be distinct nonnegative indices')
            spec['features']=sorted(features)
            probability=spec.get('probability',.5)
            if not np.isfinite(probability) or not 0 < probability <= 1:
                raise ValueError('Pattern probability must be in (0,1]')
            subset=spec.get('integer_features' if kind=='interval' else 'locked_features',())
            if not set(subset).issubset(features):
                raise ValueError('Integer/locked features must be a subset of pattern features')
            if kind=='interval':
                ratio=spec.get('ratio',.1)
                maximum=spec.get('max_ratio')
                if not np.isfinite(ratio) or ratio<=0 or (maximum is not None and (not np.isfinite(maximum) or maximum<ratio)):
                    raise ValueError('Invalid interval ratio/max_ratio')
            elif len(subset)>=len(features):
                raise ValueError('Combination needs an unlocked feature')
            result.append(spec)
        return result

    def _validate_schema(self, names: Sequence[str]) -> None:
        if not names or len(set(names))!=len(names) or set(names)-{'PL','IAT','DIR','WIN','FLG','TTL','FEAT_PAD_MASK'}:
            raise ValueError('Unknown or duplicate packet schema')
        if set(self.immutable_features)-set(names):
            raise ValueError('Unknown immutable feature')
        for specs in [self.patterns,*self.class_patterns.values()]:
            for spec in specs or ():
                if max(spec['features'])>=len(names):
                    raise ValueError('Pattern feature outside schema')
                for index in spec.get('integer_features',()):
                    if names[index] not in {'PL','WIN','TTL'}:
                        raise ValueError('Integer treatment must select an integral physical field')

    def _checkpoint_for_seed(self, seed: int) -> Path:
        if not self.checkpoint:
            raise ConfigurationError('A2PM requires a registered checkpoint')
        return _checkpoint_path(self.checkpoint,seed)

    @classmethod
    def train(cls, *, config: Mapping[str,Any], seed: int, gpu: str, logger: Any=None) -> dict[str,float]:
        del gpu
        cfg=dict(config['attack'])
        params={**dict(cfg.get('params',{})),**{k:v for k,v in cfg.items() if k not in {'name','variant','params','training'}}}
        settings=cls(**params,seed=seed)
        source = config.get('surrogate_model')
        if not isinstance(source, Mapping) or source.get('enabled', True) is False:
            raise ConfigurationError('A2PM requires an independently trained surrogate_model')
        from workflow.seeds import role_seed
        shape,mean,std,names=_read_classifier_schema(config,role_seed(config,'surrogate',seed))
        dataset_cfg=config['dataset']
        constructor=DATASETS.get(dataset_cfg['name'])
        options=dict(dataset_cfg.get('params',{}))
        if 'flatten' in inspect.signature(constructor).parameters:
            options['flatten']=len(shape)==1
        dataset=constructor(**options)
        raw,labels=dataset.raw_arrays()
        physical=np.asarray(raw)
        labels=np.asarray(labels,dtype=np.int64)
        names=tuple(dataset.feature_names)
        if physical.dtype!=np.float64 or tuple(physical.shape[1:])!=shape:
            raise ValueError('A2PM requires matching physical float64 input shape')
        if settings.feature_names and tuple(settings.feature_names)!=names:
            raise ValueError('A2PM feature schema mismatch')
        settings.feature_names=names
        settings._validate_schema(names)
        packets=physical.reshape(len(labels),-1,len(names))
        valid=np.asarray(dataset.valid_mask,dtype=bool)
        if valid.shape!=packets.shape[:2] or packets.shape[1]>20:
            raise ValueError('A2PM requires authoritative first-20 occupancy')
        chosen=authorized_attack_training_indices(dataset,labels,config,seed,role='surrogate_model')
        mapping=dict(dataset.class_mapping)
        overrides={}
        for key,value in settings.class_patterns.items():
            if key in mapping:
                label=int(mapping[key])
            elif key.isdecimal() and int(key) in mapping.values():
                label=int(key)
            else:
                raise ValueError(f'Unknown class pattern assignment: {key}')
            if label in overrides:
                raise ValueError('Duplicate class assignment')
            overrides[label]=value
        class_specs={}
        states={}
        for label in np.unique(labels[chosen]):
            label=int(label)
            specs=overrides.get(label,None if label==settings.target_label else settings.patterns)
            class_specs[label]=specs
            observations=packets[chosen[labels[chosen]==label]][valid[chosen[labels[chosen]==label]]]
            if not len(observations):
                if specs is not None:
                    raise ValueError(f'Class {label} has no training packet support')
                states[label]=None
            elif not np.isfinite(observations).all():
                raise ValueError('Nonfinite training packets')
            else:
                states[label]=None if specs is None else _fit_patterns(observations,specs)
        origin=trained_generator_provenance(dataset,labels,config,seed,source_role='surrogate_model',input_shape=shape,feature_names=names)
        tensor_states={label:None if state is None else [{k:torch.from_numpy(v.copy()) for k,v in p.items()} for p in state]
                       for label,state in states.items()}
        payload=dict(attack_name='a2pm',format_version=_FORMAT_VERSION,seed=seed,artifact_provenance=origin,
                     feature_names=names,input_shape=shape,patterns=settings.patterns,class_patterns=settings.class_patterns,
                     class_specs=class_specs,states=tensor_states,normalization_mean=torch.from_numpy(mean),normalization_std=torch.from_numpy(std))
        path=settings._checkpoint_for_seed(seed)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        torch.save(payload,temporary)
        temporary.replace(path)
        return {'a2pm_trained_classes':float(sum(s is not None for s in states.values())),
                'a2pm_training_packets':float(valid[chosen].sum()),
                'a2pm_tracked_combinations':float(sum(len(p.get('combinations',())) for s in states.values() if s for p in s))}

    def _load_patterns(self, seed: int) -> None:
        if self._loaded_path is not None:
            return
        path=self._checkpoint_for_seed(seed)
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state.get('attack_name')!='a2pm' or state.get('format_version')!=_FORMAT_VERSION:
            raise ValueError('Unsupported A2PM checkpoint format; retrain')
        verify_generator_artifact_provenance(state,getattr(self,'expected_provenance',None))
        if state.get('seed')!=seed or (self.feature_names and tuple(state['feature_names'])!=tuple(self.feature_names)):
            raise ValueError('A2PM seed/schema mismatch')
        if state['patterns']!=self.patterns or state['class_patterns']!=self.class_patterns:
            raise ValueError('A2PM pattern configuration mismatch; retrain')
        self.feature_names=tuple(state['feature_names'])
        self._validate_schema(self.feature_names)
        self._states=state['states']
        self._class_specs=state['class_specs']
        self._normalization_mean=state['normalization_mean']
        self._normalization_std=state['normalization_std']
        self._input_shape=tuple(state['input_shape'])
        self._loaded_path=path

    def _predict(self, physical: np.ndarray, mean: Any, std: Any, valid: np.ndarray) -> np.ndarray:
        encoded=encode_normalized32(physical,self.feature_names,mean,std)
        model=self._require_model()
        with torch.no_grad():
            if 'valid_mask' in inspect.signature(model.predict).parameters:
                prediction=model.predict(encoded,valid_mask=valid)
            else:
                prediction=model.predict(encoded)
        prediction=_to_numpy(prediction).reshape(-1)
        if len(prediction)!=len(physical):
            raise ValueError('One prediction per record required')
        return prediction

    def generate(self,x:Any,y:Any,**kwargs:Any)->AttackResult:
        self._require_model()
        original=_to_numpy(x)
        if original.dtype!=np.float64 or original.ndim not in (2,3) or not np.isfinite(original).all():
            raise ValueError('A2PM requires finite physical float64 inputs')
        seed=int(kwargs.get('seed',self.seed))
        self._load_patterns(seed)
        if tuple(original.shape[1:])!=self._input_shape:
            raise ValueError('A2PM checkpoint topology mismatch')
        labels=_to_numpy(y).reshape(-1)
        if len(labels)!=len(original) or not np.issubdtype(labels.dtype,np.integer):
            raise ValueError('Invalid class labels')
        for label in labels:
            if int(label) not in self._states:
                raise ValueError(f'Unknown class {label}')
        mean=kwargs.get('normalization_mean')
        std=kwargs.get('normalization_std')
        if mean is None or std is None or not torch.equal(torch.as_tensor(mean,dtype=torch.float32).reshape(-1),self._normalization_mean.reshape(-1)) or not torch.equal(torch.as_tensor(std,dtype=torch.float32).reshape(-1),self._normalization_std.reshape(-1)):
            raise ValueError('Runtime normalization differs from fitted A2PM checkpoint')
        packets=original.reshape(len(original),-1,len(self.feature_names))
        valid=np.asarray(kwargs.get('valid_mask'),dtype=bool)
        if valid.shape!=packets.shape[:2] or packets.shape[1]>20:
            raise ValueError('A2PM requires authoritative first-20 valid_mask')
        normalized=torch.as_tensor(encode_normalized32(original,self.feature_names,mean,std))
        permitted=self.victim_packet_perturbation_mask(normalized,self.feature_names,normalization_mean=mean,normalization_std=std,valid_mask=valid).numpy().reshape(packets.shape).astype(bool)
        for j,name in enumerate(self.feature_names):
            if name in set(self.immutable_features)|{'DIR','FLG','FEAT_PAD_MASK'}:
                permitted[...,j]=False
        adv=original.copy()
        baseline=self._predict(adv,mean,std,valid)
        counts=np.ones(len(labels),np.int64)
        target=int(kwargs.get('target_label',self.target_label))
        success=baseline==target
        active=~success & np.array([self._states[int(label)] is not None for label in labels])
        rngs={label:[np.random.default_rng(seed) for _ in specs] for label,specs in self._class_specs.items() if specs is not None}
        attempts=stagnant=0
        stop='max_iterations'
        for _ in range(self.max_iterations):
            if not active.any():
                stop='all_evaded_or_ineligible'
                break
            for label,specs in self._class_specs.items():
                ids=np.flatnonzero(active & (labels==label))
                if specs is None or not len(ids):
                    continue
                view=adv.reshape(packets.shape)
                # Rows retain source record/packet order. Protected rows do not
                # consume author draws: their presence is a packet adaptation.
                allowed=permitted[ids].reshape(-1,len(self.feature_names))
                values=view[ids].reshape(-1,len(self.feature_names))
                view[ids]=_transform_patterns(values,specs,self._states[label],rngs[label],allowed).reshape(view[ids].shape)
            ids=np.flatnonzero(active)
            predicted=self._predict(adv[ids],mean,std,valid[ids])
            counts[ids]+=1
            new=predicted==target
            success[ids]|=new
            active[ids]=~new
            attempts+=1
            stagnant=0 if new.any() else stagnant+1
            if self.patience and stagnant>=self.patience:
                stop='no_new_evasions'
                break
        if not active.any():
            stop='all_evaded_or_ineligible'
        return AttackResult(adversarial=adv,success_mask=success,queries=counts,feature_names=tuple(self.feature_names),
            metadata={'attack':'a2pm','iterations':attempts,'stop_reason':stop,'patience':self.patience,
                      'checkpoint':str(self._loaded_path),'total_queries':int(counts.sum())})
