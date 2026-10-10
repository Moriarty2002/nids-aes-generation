from __future__ import annotations

import importlib
import pkgutil
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Generic, Iterable, TypeVar

from .exceptions import DuplicateRegistrationError, RegistryError, UnknownComponentError

T = TypeVar("T")


@dataclass(frozen=True)
class RegisteredComponent(Generic[T]):
    name: str
    component: T
    module: str
    qualname: str


class ComponentRegistry(Generic[T]):
    """Thread-safe plugin registry used by Hydra configuration names.

    Import-time decorators populate the registry; discovery is therefore
    performed explicitly in both the parent and spawned workers before lookup.
    """

    def __init__(self, category: str) -> None:
        self.category = category
        self._lock = threading.RLock()
        self._items: dict[str, RegisteredComponent[T]] = {}

    def register(self, name: str, component: T) -> T:
        """Register a component while rejecting conflicting duplicate names."""
        normalized = name.strip().lower().replace("-", "_")
        if not normalized:
            raise RegistryError(f"Cannot register empty {self.category} name")
        module = getattr(component, "__module__", type(component).__module__)
        qualname = getattr(component, "__qualname__", type(component).__qualname__)
        record = RegisteredComponent(normalized, component, module, qualname)
        with self._lock:
            previous = self._items.get(normalized)
            if previous is not None and previous.component is not component:
                raise DuplicateRegistrationError(
                    f"{self.category!r} name {normalized!r} is already registered by "
                    f"{previous.module}.{previous.qualname}"
                )
            self._items[normalized] = record
        return component

    def decorator(self, name: str | None = None) -> Callable[[T], T]:
        def _decorator(component: T) -> T:
            component_name = name or getattr(component, "REGISTRY_NAME", None) or getattr(
                component, "__name__", type(component).__name__
            )
            return self.register(str(component_name), component)

        return _decorator

    def get(self, name: str) -> T:
        """Resolve a configured name and report available alternatives on failure."""
        normalized = name.strip().lower().replace("-", "_")
        with self._lock:
            record = self._items.get(normalized)
        if record is None:
            available = ", ".join(self.names()) or "<none>"
            raise UnknownComponentError(
                f"Unknown {self.category} {name!r}. Available: {available}"
            )
        return record.component

    def metadata(self) -> tuple[RegisteredComponent[T], ...]:
        with self._lock:
            return tuple(sorted(self._items.values(), key=lambda x: x.name))

    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.metadata())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


ATTACKS: ComponentRegistry[Any] = ComponentRegistry("attack")
NIDS: ComponentRegistry[Any] = ComponentRegistry("nids")
DATASETS: ComponentRegistry[Any] = ComponentRegistry("dataset")
PREPROCESSORS: ComponentRegistry[Any] = ComponentRegistry("preprocessor")
DETECTORS: ComponentRegistry[Any] = ComponentRegistry("detector")

def register_attack(name: str | None = None) -> Callable[[T], T]:
    return ATTACKS.decorator(name)


def register_nids(name: str | None = None) -> Callable[[T], T]:
    return NIDS.decorator(name)


def register_dataset(name: str | None = None) -> Callable[[T], T]:
    return DATASETS.decorator(name)


def register_preprocessor(name: str | None = None) -> Callable[[T], T]:
    return PREPROCESSORS.decorator(name)


def register_detector(name: str | None = None) -> Callable[[T], T]:
    """Register an inference-time detector plugin (Mahalanobis, LID, KD, ...)."""
    return DETECTORS.decorator(name)


def _iter_submodules(package: ModuleType) -> Iterable[str]:
    yield package.__name__
    if hasattr(package, "__path__"):
        prefix = package.__name__ + "."
        for module_info in pkgutil.walk_packages(package.__path__, prefix):
            yield module_info.name


def discover_modules(package_names: Iterable[str]) -> tuple[str, ...]:
    """Import plugin packages so their decorators can populate registries."""
    imported: list[str] = []
    for package_name in package_names:
        package = importlib.import_module(package_name)
        for module_name in _iter_submodules(package):
            if module_name == package.__name__:
                continue
            importlib.import_module(module_name)
            imported.append(module_name)
    return tuple(imported)


def discover_all() -> tuple[str, ...]:
    return discover_modules(
        (
            "data",
            "preprocessing",
            "models",
            "attacks",
            "defenses",
        )
    )


def get_registry(category: str) -> ComponentRegistry[Any]:
    registries = {
        "attack": ATTACKS,
        "nids": NIDS,
        "dataset": DATASETS,
        "preprocessor": PREPROCESSORS,
        "detector": DETECTORS,
    }
    try:
        return registries[category]
    except KeyError as exc:
        raise RegistryError(f"Unsupported registry category: {category}") from exc
