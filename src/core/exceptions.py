from __future__ import annotations


class BenchmarkError(RuntimeError):
    """Base exception for framework failures."""


class RegistryError(BenchmarkError):
    """Raised when registration or discovery fails."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a registry name is already owned by a different object."""


class UnknownComponentError(RegistryError):
    """Raised when a requested component does not exist."""


class IncompatibleThreatModelError(BenchmarkError):
    """Raised when an attack's threat model is incompatible with the target model."""


class ConstraintViolationError(BenchmarkError):
    """Raised when strict constraint enforcement rejects a batch."""


class DeviceConfigurationError(BenchmarkError):
    """Raised for invalid or unavailable accelerator configuration."""


class ConfigurationError(BenchmarkError):
    """Raised for invalid experiment configuration."""
