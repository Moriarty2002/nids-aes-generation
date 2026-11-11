"""Public NIDS workflow entry points with package facade to avoid circular imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "evaluate_attack_seed",
    "preprocess_seed",
    "train_attack_seed",
    "train_nids_seed",
]

_EXPORT_MODULES: dict[str, str] = {
    "evaluate_attack_seed": ".evaluate",
    "preprocess_seed": ".train",
    "train_attack_seed": ".train",
    "train_nids_seed": ".train",
}


def __getattr__(name: str) -> Any:
    """Resolve workers lazily to avoid circular package initialization."""
    module_name = _EXPORT_MODULES.get(name)

    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    worker = getattr(
        import_module(module_name, package=__name__),
        name,
    )
    globals()[name] = worker
    return worker


def __dir__() -> list[str]:
    """Expose deferred worker entry points during introspection."""
    return sorted(set(globals()) | set(__all__))