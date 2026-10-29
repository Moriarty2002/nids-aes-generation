"""Composite-manifold AMD-GAN over authoritative physical NIDS features.

A specialist WGAN-GP models each (class, physical immutable state) manifold.
Nonnegative continuous measurements use log1p before per-cluster GAN-domain
scaling; generated continuous coordinates return to physical units via expm1.
Categorical codebooks independently map GAN states to exact physical protocol
states. The raw float64 attack candidate is not projected or normalized here:
FeatureSpaceAttack performs final physical clipping, lattice projection,
immutable/victim/padding restoration, and canonical normalized float32 encoding.

The attack generates fixed-shape packet-feature matrices rather than modifying
PCAP packet identities or lengths. It does not enforce packet ownership or
padding itself. Its fit path excludes invalid padding rows from statistical
training so that placeholder measurements are not learned as network traffic.

Training data must be captured in float64 before model normalization.

* **AMD-GAN**
  * **Paper:** *Adversarial red teaming and imbalance correction in heterogeneous NIDS: A class-specific adaptive GAN framework* - Lara et al., Journal of Information Security and Applications 2026
  * **Repository:** [https://github.com/antoniol00/AMD-GAN](https://github.com/antoniol00/AMD-GAN)
"""

from __future__ import annotations

import gc
import inspect
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import shap
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.nn import functional as F

from attacks.base import AttackResult, FeatureSpaceAttack
from core.registry import DATASETS, register_attack
from orchestrator.device import configure_process_device


@dataclass(frozen=True)
class _Regime:
    name: str
    hidden: tuple[int, int, int]
    batch_size: int
    gp_weight: float
    noise_sigma: float
    oversample: int
    critic_steps: int
    learning_rate: float
    training_steps: int


class _Generator(nn.Module):
    """Generate a joint feature vector with continuous and discrete heads.

    Each categorical *coordinate* is modeled using its empirical codebook,
    rather than treating unrelated categorical columns as one one-hot group.
    Straight-through Gumbel-Softmax keeps each such coordinate differentiable
    during training and returns an observed category value at inference.
    Codebooks are in the GAN's [-1, 1] MinMax training domain.
    """

    def __init__(
        self,
        latent_dim: int,
        width: int,
        hidden: Sequence[int],
        continuous_idx: Sequence[int],
        category_values: dict[int, np.ndarray],
        tau: float,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.continuous_idx = tuple(int(j) for j in continuous_idx)
        self.categorical_idx = tuple(sorted(int(j) for j in category_values))
        self.tau = float(tau)
        layers: list[nn.Module] = []
        incoming = latent_dim
        for layer_number, units in enumerate(hidden):
            layers.append(nn.Linear(incoming, units))
            # Batch normalization is restricted to the Generator's first two
            # hidden layers. It must NEVER be added to the WGAN-GP Critic.
            if layer_number < 2:
                layers.append(nn.BatchNorm1d(units))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            incoming = units
        self.trunk = nn.Sequential(*layers)
        self.continuous_head = (
            nn.Linear(incoming, len(self.continuous_idx))
            if self.continuous_idx else None
        )
        self.categorical_heads = nn.ModuleDict()
        for j in self.categorical_idx:
            codes = np.asarray(category_values[j], dtype=np.float32)
            if codes.ndim != 1 or not codes.size:
                raise ValueError(f"Invalid categorical codebook for coordinate {j}")
            self.categorical_heads[str(j)] = nn.Linear(incoming, int(codes.size))
            self.register_buffer(f"codes_{j}", torch.from_numpy(codes.copy()))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(z)
        coords: dict[int, torch.Tensor] = {}
        if self.continuous_head is not None:
            continuous = torch.tanh(self.continuous_head(hidden))
            for offset, j in enumerate(self.continuous_idx):
                coords[j] = continuous[:, offset]
        for j in self.categorical_idx:
            logits = self.categorical_heads[str(j)](hidden)
            one_hot = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
            codes = getattr(self, f"codes_{j}")
            coords[j] = (one_hot * codes.unsqueeze(0)).sum(dim=-1)
        # torch.stack preserves autograd through every head, and also avoids
        # filling a zero tensor through potentially fragile in-place indexing.
        return torch.stack([coords[j] for j in range(self.width)], dim=-1)


class _Critic(nn.Module):
    """Unbounded scalar WGAN critic, without normalization or sigmoid.

    Gradient penalty acts on interpolated inputs to softly constrain the
    critic's gradient norm; omitting BatchNorm prevents batch-composition
    effects from corrupting this per-sample gradient-penalty calculation.
    """

    def __init__(self, width: int, hidden: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        incoming = width
        for units in hidden:
            layers.extend((nn.Linear(incoming, units), nn.LeakyReLU(0.2)))
            incoming = units
        layers.append(nn.Linear(incoming, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).reshape(-1)


# The second key component contains exact physical immutable-state values
# in canonical feature-name order; no classifier normalization is involved.
ClusterKey = tuple[int, tuple[float, ...]]




def _checkpoint_path(template: str | Path, seed: int) -> Path:
    """Resolve a seed template without relying on the process working directory."""
    return Path(str(template).format(seed=int(seed))).expanduser()


def _classifier_schema(
    config: Mapping[str, Any], seed: int
) -> tuple[dict[str, Any], tuple[int, ...], tuple[str, ...]]:
    """Read classifier topology and feature ordering, but never its scaling.

    A physical-space generator must learn from the dataset's pre-normalization
    measurements; the checkpoint is consulted only for structural compatibility.
    """
    raw_surrogate = config.get("surrogate_model")
    if isinstance(raw_surrogate, Mapping) and raw_surrogate.get("enabled", True):
        model_cfg = dict(raw_surrogate)
        role = "surrogate_model"
    else:
        raw_model = config.get("model")
        if not isinstance(raw_model, Mapping):
            raise ValueError("AMD-GAN requires model or surrogate_model configuration")
        model_cfg = dict(raw_model)
        role = "model"

    checkpoint_template = model_cfg.get("checkpoint")
    if not checkpoint_template:
        raise ValueError(f"{role}.checkpoint must reference a trained NIDS checkpoint")
    path = _checkpoint_path(str(checkpoint_template), seed)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Classifier checkpoint {path} must be a mapping")
    input_shape = tuple(int(v) for v in checkpoint.get("input_shape", ()))
    if len(input_shape) not in (1, 2) or any(v < 1 for v in input_shape):
        raise ValueError(f"Classifier checkpoint has invalid input_shape {input_shape}")
    names = tuple(str(n).upper() for n in checkpoint.get("feature_names", ()))
    return model_cfg, input_shape, names


def _training_dataset(config: Mapping[str, Any], input_shape: tuple[int, ...]) -> Any:
    """Use the DATASETS registry, matching TDTCE's flatten-layout contract."""
    raw = config.get("dataset")
    if not isinstance(raw, Mapping) or not raw.get("name"):
        raise ValueError("AMD-GAN requires dataset.name and dataset.params")
    constructor = DATASETS.get(str(raw["name"]))
    params = dict(raw.get("params", {}))
    try:
        signature = inspect.signature(constructor)
        if "flatten" in signature.parameters:
            params["flatten"] = len(input_shape) == 1
    except (TypeError, ValueError):
        pass
    return constructor(**params)


def _train_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    """Hold out a class-wise subset for generator fitting.

    The split is deterministic but does not automatically match the classifier's
    own training split. Benchmark orchestration must restrict raw_arrays() to
    permissible training samples to avoid inadvertent test-set leakage.
    """
    rng = np.random.default_rng(seed)
    selections: list[np.ndarray] = []
    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        training_count = min(len(indices), max(1, int(math.ceil(0.8 * len(indices)))))
        selections.append(indices[:training_count])
    return np.sort(np.concatenate(selections)).astype(np.int64)





@register_attack("amd_gan")
class AMDGANFeatureAttack(FeatureSpaceAttack):
    """Class-specific WGAN-GP generator of float64 physical packet features.

    It emits raw candidate packets and delegates all threat-model constraints
    to the inherited feature-space finalizer and independent gatekeeper.
    """

    attack_name = "amd_gan"
    operates_on_physical_features = True
    query_based = False
    requires_gradients = False

    def __init__(
        self,
        *,
        continuous_idx: Sequence[int],
        categorical_idx: Sequence[int] = (),
        latent_dim: int = 100,
        gumbel_tau: float = 0.7,
        training_steps_large: int = 15000,
        training_steps_small: int = 25000,
        training_steps_nano: int = 30000,
        critic_steps_large: int = 5,
        critic_steps_small: int = 3,
        critic_steps_nano: int = 2,
        learning_rate_large: float = 1e-4,
        learning_rate_small: float = 5e-5,
        learning_rate_nano: float = 2e-5,
        xai_threshold: float = 0.85,
        xai_samples: int = 256,
        xai_trees: int = 100,
        max_categories: int = 128,
        seed: int = 42,
        target_label: int = 0,
        device: str = "auto",
        feature_names: Sequence[str] = (),
        immutable_features: Sequence[str] = (),
        clip_ranges: Mapping[str, tuple[float, float]] | None = None,
        discrete_projection: bool = False,
        checkpoint: str | None = None,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # The supplied FeatureSpaceAttack is a kw-only dataclass whose __init__
        # accepts neither config nor **kwargs. Newer base classes can accept
        # both. Forward every supported argument, without suppressing errors
        # from unrelated constructor failures or silently discarding settings.
        self.config = config
        self.checkpoint = checkpoint
        base_kwargs = {
            "discrete_projection": bool(discrete_projection),
            "clip_ranges": dict(clip_ranges or {}),
            "feature_names": tuple(str(name).upper() for name in feature_names),
            "immutable_features": tuple(str(name).upper() for name in immutable_features),
            **kwargs,
        }
        base_init = super().__init__
        signature = inspect.signature(base_init)
        accepts_extra = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if "config" in signature.parameters or accepts_extra:
            base_init(config=config, **base_kwargs)
        else:
            # Safely filter out injected orchestrator arguments (like training_fraction)
            # that the strict base class does not natively support, preventing crashes.
            filtered_kwargs = {
                key: value for key, value in base_kwargs.items() 
                if key in signature.parameters
            }
            base_init(**filtered_kwargs)

        self.continuous_idx = tuple(int(j) for j in continuous_idx)
        self.categorical_idx = tuple(int(j) for j in categorical_idx)
        all_indices = self.continuous_idx + self.categorical_idx
        if len(all_indices) != 6 or set(all_indices) != set(range(6)):
            raise ValueError("continuous_idx and categorical_idx must partition indices 0..5")
        if len(set(all_indices)) != 6:
            raise ValueError("Feature indices must not repeat")
        self.width = 6
        if self.feature_names and len(self.feature_names) != 6:
            raise ValueError("feature_names must contain exactly six ordered names")
        if not 0 <= xai_threshold < 1 or xai_samples < 2 or xai_trees < 1:
            raise ValueError("Invalid mandatory SHAP correlation gate settings")
        if latent_dim < 1 or gumbel_tau <= 0 or max_categories < 2:
            raise ValueError("Invalid generator architecture or categorical settings")
        if min(
            training_steps_large, training_steps_small, training_steps_nano,
            critic_steps_large, critic_steps_small, critic_steps_nano,
        ) < 1:
            raise ValueError("Training/critic step counts must be positive")
        if min(learning_rate_large, learning_rate_small, learning_rate_nano) <= 0:
            raise ValueError("Learning rates must be positive")

        self.latent_dim = int(latent_dim)
        self.gumbel_tau = float(gumbel_tau)
        self.xai_threshold = float(xai_threshold)
        self.xai_samples = int(xai_samples)
        self.xai_trees = int(xai_trees)
        self.max_categories = int(max_categories)
        self.seed = int(seed)
        self.target_label = int(target_label)
        self.device_name = str(device)
        self._regimes = (
            _Regime("large", (256, 512, 256), 64, 10.0, 0.0, 1,
                    critic_steps_large, learning_rate_large, training_steps_large),
            _Regime("small", (128, 256, 128), 16, 15.0, 0.01, 1,
                    critic_steps_small, learning_rate_small, training_steps_small),
            _Regime("nano", (64, 128, 64), 8, 15.0, 0.05, 5,
                    critic_steps_nano, learning_rate_nano, training_steps_nano),
        )
        # Only `_load_generator` may populate this on evaluation. The classmethod
        # training path writes CPU weights straight to a disk artifact.
        self._checkpoints: dict[ClusterKey, dict[str, Any]] = {}
        self.training_summary: dict[ClusterKey, dict[str, Any]] = {}
        self._loaded_path: Path | None = None

    def _device(self) -> torch.device:
        if self.device_name == "auto":
            assigned = getattr(self.model, "device", None)
            return torch.device(assigned) if assigned is not None else torch.device("cpu")
        result = torch.device(self.device_name)
        if result.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device unavailable: {result}")
        return result

    def _regime(self, n: int) -> _Regime:
        if n >= 126:
            return self._regimes[0]
        return self._regimes[1] if n >= 29 else self._regimes[2]

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    def _packets(self, x: Any) -> np.ndarray:
        """View unnormalized float64 data as (batch, time, six features).

        Reject float32 inputs rather than silently losing physical precision.
        Both flat and sequence model topologies have the same ordered packets.
        """
        data = self._numpy(x)
        if data.dtype != np.float64 or data.ndim not in (2, 3):
            raise TypeError("AMD-GAN requires 2-D or 3-D physical float64 input")
        if data.ndim == 2 and data.shape[-1] % self.width:
            raise ValueError("Flattened width must be divisible by six")
        if data.ndim == 3 and data.shape[-1] != self.width:
            raise ValueError("Packet width must equal six")
        if not np.isfinite(data).all():
            raise ValueError("Physical input contains non-finite values")
        return data.reshape(data.shape[0], -1, self.width)

    def _immutable_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        """Use physical categorical protocol states as manifold conditions.

        This is a conditioning schema check, not an immutability projection.
        The inherited interceptor is solely responsible for restoration.
        """
        names = tuple(str(name).upper() for name in names)
        if len(names) != self.width or len(set(names)) != self.width:
            raise ValueError("An ordered, unique six-feature schema is required")
        protected = set(self.immutable_features)
        if "DIR" in names:
            protected.add("DIR")
        unknown = protected.difference(names)
        if unknown:
            raise ValueError(f"Unknown immutable feature names: {sorted(unknown)}")
        indices = tuple(j for j, name in enumerate(names) if name in protected)
        conflicts = [names[j] for j in indices if j in self.continuous_idx]
        if conflicts:
            raise ValueError(
                "Immutable columns must be categorical, not continuous/log-smoothed: "
                f"{conflicts}"
            )
        return indices

    @staticmethod
    def _cluster_key(label: int, row: np.ndarray, immutable_idx: Sequence[int]) -> ClusterKey:
        return (int(label), tuple(float(row[j]) for j in immutable_idx))

    def _categorical_codebooks(
        self,
        physical: np.ndarray,
        minimum: np.ndarray,
        safe_span: np.ndarray,
        constant: np.ndarray,
        scaled: np.ndarray,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Represent discrete physical states by categorical GAN-domain tokens.

        Gumbel-Softmax samples tokens. Their physical lookup tables are kept in
        float64 and are NEVER inverse-log-transformed or reconstructed through
        a lossy float32 min/max inverse. Continuous scaling is independent.
        """
        scaled_codes: dict[int, np.ndarray] = {}
        physical_states: dict[int, np.ndarray] = {}
        for j in self.categorical_idx:
            states = np.unique(physical[:, j])
            if states.size > self.max_categories:
                raise ValueError(
                    f"Categorical coordinate {j} has {states.size} states, "
                    f"more than max_categories={self.max_categories}"
                )
            if not np.equal(states, np.rint(states)).all():
                raise ValueError(f"Categorical coordinate {j} must have integral physical states")
            if constant[j]:
                tokens = np.zeros(states.size, dtype=np.float32)
            else:
                tokens = (
                    2.0 * (states - minimum[j]) / safe_span[j] - 1.0
                ).astype(np.float32)
            if np.unique(tokens).size != states.size:
                raise ValueError(
                    f"Physical categorical states for feature {j} collide "
                    "in the generator's float32 token domain"
                )
            # Make every real categorical coordinate exactly a codebook token.
            # This guarantees the critic and categorical Gumbel head see the
            # same discretized manifold, including when raw states are large.
            scaled[:, j] = tokens[np.searchsorted(states, physical[:, j])]
            scaled_codes[j] = tokens
            physical_states[j] = states.astype(np.float64, copy=True)
        return scaled_codes, physical_states


    @staticmethod
    def _gradient_penalty(
        critic: _Critic, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Use WGAN-GP on independently interpolated real and fake examples."""
        mixing = torch.rand(real.shape[0], 1, device=real.device)
        midpoint = (mixing * real + (1.0 - mixing) * fake).requires_grad_(True)
        scores = critic(midpoint)
        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=midpoint,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.norm(p=2, dim=1).sub(1).square().mean()

    def _xai_correlation(
        self,
        real_scaled: np.ndarray,
        generator: _Generator,
        device: torch.device,
        rng: np.random.Generator,
        label: int,
    ) -> float:
        """Reject generators that change RandomForest attribution rank order."""
        count = min(self.xai_samples, max(2, len(real_scaled)))
        selected = rng.choice(len(real_scaled), count, replace=len(real_scaled) < count)
        actual = np.asarray(real_scaled[selected], dtype=np.float32)
        decoy = np.clip(rng.standard_normal(actual.shape), -1.0, 1.0).astype(np.float32)
        forest = RandomForestClassifier(
            n_estimators=self.xai_trees,
            max_depth=8,
            n_jobs=1,
            class_weight="balanced",
            random_state=self.seed + label,
        )
        forest.fit(
            np.concatenate((actual, decoy)),
            np.concatenate((np.ones(count, np.int8), np.zeros(count, np.int8))),
        )
        with torch.inference_mode():
            latent = torch.randn(count, self.latent_dim, device=device)
            generated = generator(latent).cpu().numpy()
        explainer = shap.TreeExplainer(forest)
        values = explainer.shap_values(np.concatenate((actual, generated)))
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        if values.shape != (2 * count, self.width):
            raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}")
        real_rank = np.abs(values[:count]).mean(axis=0)
        fake_rank = np.abs(values[count:]).mean(axis=0)
        rho = float(spearmanr(real_rank, fake_rank).statistic)
        return rho if math.isfinite(rho) else float("-inf")

    def _fit_clusters(
        self,
        x: np.ndarray,
        y: np.ndarray,
        valid_mask: np.ndarray,
        device: torch.device,
        logger: logging.Logger | None,
    ) -> tuple[dict[ClusterKey, dict[str, Any]], dict[ClusterKey, dict[str, Any]]]:
        """Fit WGAN-GP specialists on physical, non-padding packet samples.

        Nonnegative continuous measurements are mapped by log1p to tame heavy
        tails, then scaled into the bounded GAN domain. Discrete protocol
        states are represented by independent categorical lookup tables.
        """
        packets = self._packets(x)
        labels = np.asarray(y).reshape(-1)
        if labels.shape[0] != packets.shape[0]:
            raise ValueError("Training feature/label sample counts differ")
        if not np.isfinite(labels).all() or (labels != np.floor(labels)).any():
            raise ValueError("Training labels must be finite integral values")
        labels = labels.astype(np.int64, copy=False)
        names = tuple(self.feature_names)
        immutable_idx = self._immutable_indices(names)
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != packets.shape[:2] or not valid.any():
            raise ValueError("Training valid_mask must mark real packet rows")

        flat = packets.reshape(-1, self.width)
        flat_labels = np.broadcast_to(labels[:, None], valid.shape).reshape(-1)
        groups: dict[ClusterKey, list[int]] = {}
        for position in np.flatnonzero(valid.reshape(-1)):
            key = self._cluster_key(flat_labels[position], flat[position], immutable_idx)
            groups.setdefault(key, []).append(int(position))

        accepted: dict[ClusterKey, dict[str, Any]] = {}
        summary: dict[ClusterKey, dict[str, Any]] = {}
        for group_id, (key, positions) in enumerate(sorted(groups.items())):
            label, _ = key
            cluster_seed = self.seed + group_id
            torch.manual_seed(cluster_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(cluster_seed)
            rng = np.random.default_rng(cluster_seed)
            physical_data = np.array(flat[positions], dtype=np.float64, copy=True)
            class_data = physical_data.copy()
            n = len(positions)
            regime = self._regime(n)
            if self.continuous_idx:
                cols = list(self.continuous_idx)
                if (class_data[:, cols] < 0.0).any():
                    raise ValueError(
                        "Physical continuous measurements must be nonnegative "
                        "for log1p; revise continuous_idx for signed features"
                    )
                class_data[:, cols] = np.log1p(class_data[:, cols])

            minimum = class_data.min(axis=0)
            maximum = class_data.max(axis=0)
            span = maximum - minimum
            constant = span <= 1e-12
            safe_span = np.where(constant, 1.0, span)
            if not np.isfinite(safe_span).all():
                raise ValueError("Physical feature range cannot be scaled safely")
            scaled = np.clip(
                2.0 * (class_data - minimum) / safe_span - 1.0,
                -1.0, 1.0,
            ).astype(np.float32)
            scaled[:, constant] = 0.0
            if not np.isfinite(scaled).all():
                raise ValueError("Non-finite GAN training coordinates")
            codebooks, physical_categories = self._categorical_codebooks(
                physical_data, minimum, safe_span, constant, scaled,
            )

            generator = _Generator(
                self.latent_dim, self.width, regime.hidden,
                self.continuous_idx, codebooks, self.gumbel_tau,
            ).to(device)
            critic = _Critic(self.width, regime.hidden).to(device)
            generator.train()
            critic.train()
            opt_g = torch.optim.Adam(
                generator.parameters(), lr=regime.learning_rate, betas=(0.0, 0.9)
            )
            opt_d = torch.optim.Adam(
                critic.parameters(), lr=regime.learning_rate, betas=(0.0, 0.9)
            )
            cpu_training = torch.from_numpy(np.ascontiguousarray(scaled))
            population = np.arange(n)
            if regime.oversample > 1:
                population = rng.choice(population, n * regime.oversample, replace=True)

            last_g_loss = float("nan")
            last_d_loss = float("nan")
            try:
                for _ in range(regime.training_steps):
                    for _ in range(regime.critic_steps):
                        selected = rng.choice(population, regime.batch_size, replace=True)
                        real = cpu_training[selected].to(device)
                        if regime.noise_sigma:
                            real = real + torch.randn_like(real) * regime.noise_sigma
                        latent = torch.randn(regime.batch_size, self.latent_dim, device=device)
                        with torch.no_grad():
                            fake = generator(latent)
                        opt_d.zero_grad(set_to_none=True)
                        gp = self._gradient_penalty(critic, real, fake)
                        loss_d = critic(fake).mean() - critic(real).mean() + regime.gp_weight * gp
                        loss_d.backward()
                        opt_d.step()
                        last_d_loss = float(loss_d.detach().item())
                        del real, latent, fake, gp, loss_d

                    critic.requires_grad_(False)
                    opt_g.zero_grad(set_to_none=True)
                    latent = torch.randn(regime.batch_size, self.latent_dim, device=device)
                    fake = generator(latent)
                    loss_g = -critic(fake).mean()
                    loss_g.backward()
                    opt_g.step()
                    last_g_loss = float(loss_g.detach().item())
                    critic.requires_grad_(True)
                    del latent, fake, loss_g

                generator.eval()
                rho = self._xai_correlation(scaled, generator, device, rng, cluster_seed)
                passed = bool(rho > self.xai_threshold)
                summary[key] = {
                    "accepted": passed,
                    "xai_spearman_rho": rho,
                    "regime": regime.name,
                    "sample_count": n,
                    "generator_loss": last_g_loss,
                    "critic_loss": last_d_loss,
                    "cluster_key": key,
                }
                if passed:
                    accepted[key] = {
                        "state": {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in generator.state_dict().items()
                        },
                        "min": torch.from_numpy(minimum.copy()),
                        "span": torch.from_numpy(safe_span.copy()),
                        "constant": torch.from_numpy(constant.copy()),
                        "categories": {
                            j: torch.from_numpy(codes.copy())
                            for j, codes in codebooks.items()
                        },
                        "physical_categories": {
                            j: torch.from_numpy(states.copy())
                            for j, states in physical_categories.items()
                        },
                        "hidden": tuple(regime.hidden),
                        "cluster_key": key,
                    }
                if logger:
                    logger.info(
                        "amd_gan.cluster seed=%d key=%s N=%d regime=%s "
                        "g_loss=%.6f d_loss=%.6f rho=%.4f accepted=%s",
                        self.seed, key, n, regime.name, last_g_loss, last_d_loss,
                        rho, passed,
                    )
            finally:
                del generator, critic, opt_g, opt_d, cpu_training
                del class_data, physical_data, scaled, codebooks, physical_categories
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        return accepted, summary

    @classmethod
    def train(
        cls,
        *,
        config: Mapping[str, Any],
        seed: int,
        gpu: str,
        logger: logging.Logger | None = None,
    ) -> Mapping[str, float]:
        """Train on pre-normalization physical float64 dataset measurements.

        Classifier checkpoint metadata supplies only input topology and the
        ordered feature schema. No classifier normalization is performed,
        fitted, stored, or applied inside this generator.
        """
        device = configure_process_device(gpu).torch_device
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        attack_cfg_raw = config.get("attack", {})
        if not isinstance(attack_cfg_raw, Mapping):
            raise TypeError("config.attack must be a mapping")
        attack_cfg = dict(attack_cfg_raw)
        supported = (
            "continuous_idx", "categorical_idx", "latent_dim", "gumbel_tau",
            "training_steps_large", "training_steps_small", "training_steps_nano",
            "critic_steps_large", "critic_steps_small", "critic_steps_nano",
            "learning_rate_large", "learning_rate_small", "learning_rate_nano",
            "xai_threshold", "xai_samples", "xai_trees", "max_categories",
            "target_label", "immutable_features", "feature_names",
            "clip_ranges", "discrete_projection", "checkpoint",
        )
        params = {name: attack_cfg[name] for name in supported if name in attack_cfg}
        params["seed"] = int(seed)
        params["device"] = str(device)
        settings = cls(config=config, **params)

        classifier_cfg, input_shape, checkpoint_names = _classifier_schema(config, seed)
        dataset = _training_dataset(config, input_shape)
        raw, labels = dataset.raw_arrays()
        raw = np.asarray(raw)
        if raw.dtype != np.float64:
            raise TypeError(
                "Dataset raw_arrays() must expose pre-normalization float64 "
                "physical data; converting normalized/quantized float32 is unsafe"
            )
        labels = np.asarray(labels).reshape(-1)
        if len(raw) != len(labels) or tuple(raw.shape[1:]) != input_shape:
            raise ValueError(
                "Dataset does not match classifier checkpoint: "
                f"data={raw.shape}, labels={labels.shape}, input_shape={input_shape}"
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("Dataset labels must be integer-encoded")
        if np.any(labels < 0):
            raise ValueError("Dataset class labels must be nonnegative")

        dataset_names = tuple(str(name).upper() for name in getattr(dataset, "feature_names", ()))
        if dataset_names and checkpoint_names and dataset_names != checkpoint_names:
            raise ValueError("Dataset feature ordering differs from classifier checkpoint")
        if settings.feature_names and dataset_names and settings.feature_names != dataset_names:
            raise ValueError("Attack feature ordering differs from dataset feature ordering")
        if settings.feature_names and checkpoint_names and settings.feature_names != checkpoint_names:
            raise ValueError("Attack feature ordering differs from classifier checkpoint")
        settings.feature_names = settings.feature_names or dataset_names or checkpoint_names
        settings._immutable_indices(settings.feature_names)

        packets = settings._packets(raw)
        input_mask = getattr(dataset, "valid_mask", None)
        if input_mask is None and packets.shape[1] > 1:
            raise ValueError("Sequence datasets must provide a valid_mask")
        valid_mask = (
            np.ones(packets.shape[:2], dtype=bool)
            if input_mask is None else np.asarray(input_mask, dtype=bool)
        )
        if valid_mask.shape != packets.shape[:2]:
            raise ValueError("Dataset valid_mask shape disagrees with packet topology")
        training_indices = _train_indices(labels, int(seed))
        training_mask = np.zeros(packets.shape[:2], dtype=bool)
        training_mask[training_indices] = valid_mask[training_indices]

        if logger:
            logger.info(
                "amd_gan.train.start seed=%d device=%s train_flows=%d "
                "train_packets=%d checkpoint_model=%s",
                seed, device, len(training_indices), int(training_mask.sum()),
                classifier_cfg.get("name", "unknown"),
            )
        accepted, summary = settings._fit_clusters(raw, labels, training_mask, device, logger)

        # Explicit checkpoint schema: no live Modules, only CPU tensors and
        # small immutable metadata. Composite keys remain exact tuple keys.
        payload: dict[str, Any] = {
            # Version 5 separates physical continuous and categorical domains.
            # Older normalized-domain generators MUST be retrained.
            "format_version": 5,
            "data_domain": "physical_float64",
            "continuous_transform": "log1p_expm1",
            "attack_name": "amd_gan",
            "seed": int(seed),
            "input_shape": tuple(input_shape),
            "feature_names": tuple(settings.feature_names),
            "immutable_features": tuple(settings.immutable_features),
            "continuous_idx": tuple(settings.continuous_idx),
            "categorical_idx": tuple(settings.categorical_idx),
            "latent_dim": int(settings.latent_dim),
            "gumbel_tau": float(settings.gumbel_tau),
            "width": int(settings.width),
            "checkpoints": accepted,
            "training_summary": summary,
            "training_flow_count": int(len(training_indices)),
            "training_packet_count": int(training_mask.sum()),
        }
        directory = Path(str(config.get("output_dir", "runs"))).expanduser() / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"amd_gan_seed{int(seed)}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        g_losses = [record["generator_loss"] for record in summary.values()]
        g_loss = float(np.mean(g_losses)) if g_losses else 0.0
        distinct_labels = {key[0] for key in accepted}
        scores = [record["xai_spearman_rho"] for record in summary.values()]
        finite_scores = [score for score in scores if math.isfinite(score)]
        metrics = {
            "amd_gan_train_loss": g_loss,
            "amd_gan_accepted_classes": float(len(distinct_labels)),
            "amd_gan_accepted_clusters": float(len(accepted)),
            "amd_gan_rejected_clusters": float(len(summary) - len(accepted)),
            "amd_gan_train_samples": float(len(training_indices)),
            "amd_gan_train_packets": float(training_mask.sum()),
            "amd_gan_xai_mean_rho": float(np.mean(finite_scores)) if finite_scores else 0.0,
        }
        if logger:
            logger.info(
                "amd_gan.train.done seed=%d checkpoint=%s accepted=%d/%d",
                seed, destination, len(accepted), len(summary),
            )
        return metrics

    def _resolve_checkpoint_path(self, seed: int) -> Path:
        if self.checkpoint:
            template = self.checkpoint
        else:
            base = "runs" if self.config is None else str(self.config.get("output_dir", "runs"))
            template = str(Path(base) / "checkpoints" / "amd_gan_seed{seed}.pt")
        return _checkpoint_path(template, seed)

    def _load_generator(self, device: torch.device, seed: int) -> None:
        """Load accepted weights only into CPU cache; no GPU model is created.

        The requested composite generator is instantiated on GPU *inside*
        generate(), and released before the next specialist. This keeps
        checkpoint initialization independent of class count and GPU VRAM.
        """
        # EvaluationEngine increments its stochastic seed for each evaluation
        # batch. That increment must change z sampling, NOT the trained model
        # checkpoint. A per-worker attack instance therefore pins the first
        # successfully resolved seed-specific checkpoint, like TDTCE.
        if self._loaded_path is not None:
            return
        path = self._resolve_checkpoint_path(seed)
        if not path.is_file():
            raise FileNotFoundError(
                f"AMD-GAN checkpoint not found: {path}. Run action=train_attack "
                "with this seed, or configure attack.checkpoint explicitly."
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("AMD-GAN checkpoint must be a mapping")
        if (
            payload.get("attack_name") != "amd_gan"
            or payload.get("format_version") != 5
            or payload.get("data_domain") != "physical_float64"
            or payload.get("continuous_transform") != "log1p_expm1"
        ):
            raise ValueError(
                "Unsupported AMD-GAN checkpoint or transform; "
                "retrain old normalized-domain checkpoints for physical float64"
            )
        if int(payload.get("seed", -1)) != seed:
            raise ValueError("AMD-GAN checkpoint seed does not match generation seed")
        if int(payload.get("width", -1)) != self.width:
            raise ValueError("AMD-GAN checkpoint has incompatible feature width")
        if int(payload["latent_dim"]) != self.latent_dim:
            raise ValueError("AMD-GAN latent_dim mismatch")
        if float(payload["gumbel_tau"]) != self.gumbel_tau:
            raise ValueError("AMD-GAN Gumbel temperature mismatch")
        if tuple(payload["continuous_idx"]) != self.continuous_idx:
            raise ValueError("AMD-GAN continuous_idx mismatch")
        if tuple(payload["categorical_idx"]) != self.categorical_idx:
            raise ValueError("AMD-GAN categorical_idx mismatch")
        trained_names = tuple(payload["feature_names"])
        if self.feature_names and self.feature_names != trained_names:
            raise ValueError("AMD-GAN feature schema differs from trained checkpoint")
        self.feature_names = trained_names
        if tuple(payload["immutable_features"]) != self.immutable_features:
            raise ValueError("AMD-GAN immutable feature set differs from checkpoint")
        self._immutable_indices(self.feature_names)
        loaded = payload.get("checkpoints", {})
        if not isinstance(loaded, Mapping):
            raise ValueError("Malformed AMD-GAN composite checkpoint dictionary")
        self._checkpoints = dict(loaded)
        self.training_summary = dict(payload.get("training_summary", {}))
        self._checkpoint_input_shape = tuple(int(value) for value in payload["input_shape"])
        self._loaded_path = path

    def generate(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        target_label: int | None = None,
        **kwargs: Any,
    ) -> AttackResult:
        """Generate float64 physical candidates without projecting constraints.

        The wrapper owns victim/immutable/padding restoration, range clipping,
        lattice projection, and normalized float32 encoding. This generator
        only conditions on physical class/protocol states and samples entire
        packet vectors from accepted WGAN-GP specialists. Rows without a
        trained specialist are left unchanged, not synthesized arbitrarily.
        """
        input_shape = self._numpy(x).shape
        original = self._packets(x)
        batch_size, packet_count, _ = original.shape
        if len(self._numpy(y)) != batch_size:
            raise ValueError("generate() feature/label batch size mismatch")

        requested_label = self.target_label if target_label is None else int(target_label)
        seed = int(kwargs.get("seed", self.seed))
        device = self._device()
        self._load_generator(device, seed)
        if tuple(input_shape[1:]) != self._checkpoint_input_shape:
            raise ValueError("AMD-GAN generation shape differs from trained input shape")

        names = tuple(self.feature_names)
        immutable_idx = self._immutable_indices(names)
        flat = original.reshape(-1, self.width)
        groups: dict[ClusterKey, list[int]] = {}
        missing_cluster_count = 0
        missing_keys: set[ClusterKey] = set()
        for row_id in range(len(flat)):
            key = self._cluster_key(requested_label, flat[row_id], immutable_idx)
            if key not in self._checkpoints:
                missing_cluster_count += 1
                missing_keys.add(key)
                continue
            groups.setdefault(key, []).append(row_id)

        # Detached physical candidate. Never alias the baseline that the
        # interceptor needs for exact protected-coordinate restoration.
        candidate = np.array(original, dtype=np.float64, copy=True, order="C")
        output_rows = candidate.reshape(-1, self.width)
        sampler = torch.Generator(device=device)
        sampler.manual_seed(seed)
        generated_packet_count = 0

        for key, row_ids in sorted(groups.items()):
            info = self._checkpoints[key]
            generator = _Generator(
                self.latent_dim, self.width, info["hidden"],
                self.continuous_idx, info["categories"], self.gumbel_tau,
            ).to(device)
            try:
                generator.load_state_dict(info["state"], strict=True)
                generator.eval()
                with torch.inference_mode():
                    latent = torch.randn(
                        len(row_ids), self.latent_dim,
                        generator=sampler, device=device,
                    )
                    scaled = generator(latent)
                    minimum = torch.as_tensor(
                        info["min"], dtype=torch.float64, device=device,
                    )
                    span = torch.as_tensor(
                        info["span"], dtype=torch.float64, device=device,
                    )
                    proposals = (
                        (scaled.to(torch.float64) + 1.0) * (span / 2.0)
                        + minimum
                    )
                    constant = torch.as_tensor(
                        info["constant"], dtype=torch.bool, device=device,
                    )
                    proposals[:, constant] = minimum[constant]
                    if self.continuous_idx:
                        cols = list(self.continuous_idx)
                        proposals[:, cols] = torch.expm1(proposals[:, cols])

                    # Never treat categorical codes as continuous measurements.
                    # Map Gumbel-selected GAN tokens to exact physical float64
                    # states, without log/exp or float32 quantization.
                    for j in self.categorical_idx:
                        codes = torch.as_tensor(
                            info["categories"][j],
                            dtype=torch.float32, device=device,
                        )
                        states = torch.as_tensor(
                            info["physical_categories"][j],
                            dtype=torch.float64, device=device,
                        )
                        indices = torch.abs(
                            scaled[:, j, None] - codes[None, :]
                        ).argmin(dim=1)
                        proposals[:, j] = states[indices]

                    if not torch.isfinite(proposals).all():
                        raise RuntimeError(
                            f"Generator {key} emitted non-finite physical proposals"
                        )
                    output_rows[row_ids] = proposals.cpu().numpy()
                    generated_packet_count += len(row_ids)
            finally:
                del generator
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if generated_packet_count + missing_cluster_count != batch_size * packet_count:
            raise RuntimeError("AMD-GAN packet accounting invariant failed")

        return AttackResult(
            adversarial=candidate.reshape(input_shape),
            success_mask=np.zeros(batch_size, dtype=bool),
            queries=np.zeros(batch_size, dtype=np.int64),
            feature_names=names,
            metadata={
                "generator_class": requested_label,
                "composite_keys": tuple(sorted(groups)),
                "per_cluster_xai_rho": {
                    key: self.training_summary.get(key, {}).get("xai_spearman_rho")
                    for key in groups
                },
                "generated_packet_count": generated_packet_count,
                "missing_cluster_count": missing_cluster_count,
                "missing_cluster_keys": tuple(sorted(missing_keys)),
                "synthesis_domain": "physical_float64",
                "source_feature_blending": False,
                "packet_independent_synthesis": True,
                "continuous_transform": "log1p_expm1",
                "checkpoint_path": str(self._loaded_path),
            },
        )
