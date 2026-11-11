"""Apply small smoke-budget caps without homogenizing heterogeneous attacks.

Research mode leaves the attack/model's own configured parameters unchanged.
Configuration is modified BEFORE Hydra's resolved config is saved.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from core.exceptions import ConfigurationError


def apply_budget(cfg: DictConfig) -> None:
    budget = cfg.get('budget')
    if budget is None:
        return
    profile = str(budget.get('name', 'research'))
    if profile not in {'smoke', 'research'}:
        raise ConfigurationError('budget.name must be smoke or research')
    if profile == 'research':
        return
    cap = budget.get('max_test_samples')
    if cap is not None:
        configured = cfg.evaluation.get('max_samples')
        cfg.evaluation.max_samples = min(int(cap), int(configured)) if configured is not None else int(cap)
    if str(cfg.action) == 'train_nids':
        epoch_cap = budget.get('max_target_epochs')
        if epoch_cap is not None and cfg.model.training.get('epochs') is not None:
            cfg.model.training.epochs = min(int(epoch_cap), int(cfg.model.training.epochs))
        batch_cap = budget.get('max_target_batch_size')
        if batch_cap is not None:
            cfg.model.training.batch_size = min(int(batch_cap), int(cfg.model.training.batch_size))
    name = str(cfg.attack.name)
    overrides = OmegaConf.to_container(budget.get('attack_overrides', {}).get(name, {}), resolve=True)
    for key, value in dict(overrides or {}).items():
        path = f'attack.{key}'
        if OmegaConf.select(cfg, path) is None and '.' not in str(key):
            # null is a legitimate native parameter such as max_iterations.
            if key not in cfg.attack:
                raise ConfigurationError(f'Smoke budget refers to unknown {path}')
        elif '.' in str(key):
            parts = str(key).split('.')
            cursor = cfg.attack
            for item in parts:
                if item not in cursor:
                    raise ConfigurationError(f'Smoke budget refers to unknown {path}')
                cursor = cursor[item]
        OmegaConf.update(cfg, path, value)
