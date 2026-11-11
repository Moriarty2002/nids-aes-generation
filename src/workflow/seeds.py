"""Explicit seed roles for reproducible paired research comparisons.

The worker seed identifies an independent replicate and its report. A role with
no override uses that replicate seed. Fixed target and surrogate role seeds
allow multiple *attack* replicates on exactly the same frozen classifiers and
held-out sample IDs, without silently changing model or split membership.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.exceptions import ConfigurationError

_ROLES = frozenset({'split', 'target', 'surrogate', 'attack', 'detector'})


def role_seed(config: Mapping[str, Any], role: str, worker_seed: int) -> int:
    if role not in _ROLES:
        raise ValueError(f'Unknown deterministic seed role {role!r}')
    roles = config.get('seed_roles', {}) or {}
    if not isinstance(roles, Mapping) or set(roles) - _ROLES:
        raise ConfigurationError(f'Invalid seed_roles; supported keys: {sorted(_ROLES)}')
    value = roles.get(role)
    if value is None:
        result = worker_seed
    elif isinstance(value, bool) or isinstance(value, float) or not str(value).isdigit():
        raise ConfigurationError(f'seed_roles.{role} must be a nonnegative integer or null')
    else:
        result = int(value)
    if int(result) < 0:
        raise ConfigurationError(f'seed_roles.{role} must be nonnegative')
    return int(result)


def validated_worker_seeds(values: Any) -> list[int]:
    """Reject fractional, negative and duplicate replicate IDs before dispatch."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ConfigurationError('seeds must be a nonempty list of nonnegative integers')
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ConfigurationError('Each seed must be a nonnegative integer; fractional seeds are invalid')
    seeds = list(values)
    if len(seeds) != len(set(seeds)):
        raise ConfigurationError('Worker seeds must be unique within an execution')
    return seeds
