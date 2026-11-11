"""NIDSGAN: Zolbayar et al., arXiv:2203.06694v1, §3/Table 3.

Original: adversarial classification loss plus α-weighted GAN realism and
β-weighted L2 hinge under constrained malicious-flow evasion.
Mandatory NIDS adaptation: authorized independently trained surrogate,
fixed T=20 masks, strict physical finalizer/budget gate and no target oracle.
Source experiment: Table 3 reports 800 training epochs on CICIDS/NSL-KDD;
other architecture and loss-setting values remain declared native choices.
Transfer access is not the paper's direct-query case; 800 is not a
demonstrated CICIoMT-optimal value. Research YAML is authoritative and
smoke budget may override epochs to one for CPU tests.
Topology: flat (B,T*F) or masked sequence (B,T,F) via the local wrapper.
"""



from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from attacks.base import AttackResult, FeatureSpaceAttack
from core.registry import DATASETS, register_attack
from models.wrapper import PyTorchModelWrapper


class _Generator(nn.Module):
    """Deterministic conditional mapping x -> additive delta, flat internally."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.LeakyReLU(0.2)))
            width = hidden_dim
        final = nn.Linear(width, input_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(x.shape[0], -1)).reshape_as(x)


class _Critic(nn.Module):
    """Unbounded scalar Wasserstein critic (no sigmoid or batch statistics)."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(width, hidden_dim), nn.LeakyReLU(0.2)))
            width = hidden_dim
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(x.shape[0], -1)).reshape(-1)


def _radially_bounded(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Scale each sample onto/inside the L2 ball, never clip any feature.

    Unlike the hinge penalty (a soft constraint), this guarantees a hard L2
    bound on the *masked* perturbation during both training and inference.
    The scale factor is in (0, 1]; a zero perturbation remains exactly zero.
    """
    norm = torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), ord=2, dim=1)
    factor = (epsilon / norm.clamp_min(torch.finfo(delta.dtype).tiny)).clamp(max=1.0)
    return delta * factor.reshape((-1,) + (1,) * (delta.ndim - 1))


def _gradient_penalty(
    critic: _Critic,
    clean: torch.Tensor,
    generated: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP penalty on straight-line real/fake interpolates."""
    interpolation_weight = torch.rand(
        (clean.shape[0],) + (1,) * (clean.ndim - 1),
        device=clean.device,
        dtype=clean.dtype,
    )
    interpolated = (
        interpolation_weight * clean
        + (1.0 - interpolation_weight) * generated
    ).detach().requires_grad_(True)
    score = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=score.sum(),
        inputs=interpolated,
        create_graph=True,
        only_inputs=True,
    )[0]
    slopes = torch.linalg.vector_norm(gradients.reshape(len(clean), -1), ord=2, dim=1)
    penalty = ((slopes - 1.0) ** 2).mean()
    del score, gradients, slopes, interpolated, interpolation_weight
    return penalty


@register_attack("nids_gan")
@dataclass(kw_only=True)
class NIDSGANAttack(FeatureSpaceAttack):
    """Generator-only feature attack with offline WGAN-GP training.

    ``checkpoint`` contains ``{seed}`` so separate worker seeds never collide.
    The evaluation engine changes the ``seed`` kwarg on later batches; weights
    are consequently loaded *once*, from the first batch's original seed.

    For all schemas with FLG, its coordinates are frozen in addition to the
    DIR and padding protections supplied by the native mask.  The inherited
    float64 postprocessor restores the same immutable fields after projecting.
    """

    attack_name: ClassVar[str] = "nids_gan"
    operates_on_physical_features: ClassVar[bool] = False
    requires_gradients: ClassVar[bool] = True
    query_based: ClassVar[bool] = False
    hard_budget_norm = "l2"

    epsilon: float = 0.3
    target_label: int = 0
    hidden_dim: int = 256
    depth: int = 2
    epochs: int = 800
    batch_size: int = 128
    critic_steps: int = 5
    generator_lr: float = 1e-3
    critic_lr: float = 1e-3
    adam_beta1: float = 0.5
    adam_beta2: float = 0.9
    alpha: float = 0.1
    beta: float = 0.2
    gradient_penalty: float = 10.0
    checkpoint: str = "checkpoints/nids_gan/seed_{seed}.pt"
    surrogate_requires_flatten: bool | None = None

    _generator: _Generator | None = field(default=None, init=False, repr=False)
    _loaded_seed: int | None = field(default=None, init=False, repr=False)
    _input_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # The framework's native mask handles DIR and padding.  FLG and any
        # caller-declared immutable coordinates are additional exclusions.
        names = tuple(str(name).upper() for name in self.feature_names)
        protected = tuple(dict.fromkeys(
            (*self.immutable_features, *(name for name in ("DIR", "FLG") if name in names))
        ))
        self.immutable_features = tuple(str(name).upper() for name in protected)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("nids_gan epsilon must be finite and strictly positive")
        for name in ("hidden_dim", "depth", "epochs", "batch_size", "critic_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"nids_gan {name} must be >= 1")
        for name in ("generator_lr", "critic_lr"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"nids_gan {name} must be finite and > 0")
        for name in ("alpha", "beta", "gradient_penalty"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"nids_gan {name} must be finite and >= 0")
        for name in ("adam_beta1", "adam_beta2"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"nids_gan {name} must be in [0, 1)")
        if not self.checkpoint:
            raise ValueError("nids_gan checkpoint is required")

    def _mask(
        self,
        x: torch.Tensor,
        feature_names: Sequence[str],
        *,
        normalization_mean: Any,
        normalization_std: Any,
        valid_mask: Any = None,
    ) -> torch.Tensor:
        """Refine the *native* mask; never derive ownership from zero padding.

        The base-class mask is authoritative for DIR and packet validity.
        Setting any additional immutable feature columns to zero extends that
        existing mask, rather than constructing a competing manual mask.
        """
        if normalization_mean is None or normalization_std is None:
            raise ValueError("nids_gan requires surrogate normalization statistics")
        names = tuple(str(name).upper() for name in feature_names)
        native = self.victim_packet_perturbation_mask(
            x,
            names,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            valid_mask=valid_mask,
        )
        extra_protected = set(self.immutable_features) | ({"FLG"} & set(names))
        unknown = extra_protected.difference(names)
        if unknown:
            raise ValueError(f"Unknown immutable features: {sorted(unknown)}")
        if extra_protected:
            packet_view = native.reshape(x.shape[0], -1, len(names))
            for name in extra_protected:
                packet_view[..., names.index(name)] = 0.0
        return native

    def _surrogate_logits(
        self, adversarial: torch.Tensor, valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Backprop through the *bound* PyTorch surrogate, never a proxy model."""
        wrapper = self._require_model()
        if not isinstance(wrapper, PyTorchModelWrapper):
            raise TypeError("NIDSGAN requires a bound PyTorchModelWrapper surrogate")
        if self.surrogate_requires_flatten is None:
            # Model topology is configured explicitly for ambiguous cases.  A
            # 2D input is already flat.  3D inputs are passed through unchanged.
            model_input = adversarial
        elif self.surrogate_requires_flatten:
            model_input = adversarial.reshape(adversarial.shape[0], -1)
        elif adversarial.ndim == 2:
            feature_count = len(self.feature_names)
            if not feature_count or adversarial.shape[1] % feature_count:
                raise ValueError("Cannot reshape flattened attack input to sequence surrogate")
            model_input = adversarial.reshape(adversarial.shape[0], -1, feature_count)
        else:
            model_input = adversarial
        logits = wrapper.forward_differentiable(model_input, valid_mask=valid_mask)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise TypeError("Surrogate must return class logits with shape (B, C)")
        return logits

    def _critic_step(
        self,
        generator: _Generator,
        critic: _Critic,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Train only the critic; the fake branch contains no generator graph."""
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            masked = generator(x) * mask
            fake = x + _radially_bounded(masked, self.epsilon)
        real_scores = critic(x)
        fake_scores = critic(fake)
        gp = _gradient_penalty(critic, x.detach(), fake.detach())
        loss = fake_scores.mean() - real_scores.mean() + self.gradient_penalty * gp
        loss.backward()
        optimizer.step()
        scalar = loss.detach()
        del masked, fake, real_scores, fake_scores, gp, loss
        return scalar

    def _generator_step(
        self,
        generator: _Generator,
        critic: _Critic,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        mask: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint targeted CE + Wasserstein fooling + masked L2 hinge loss."""
        optimizer.zero_grad(set_to_none=True)
        raw_delta = generator(x)
        masked_delta = raw_delta * mask
        norms = torch.linalg.vector_norm(masked_delta.reshape(len(x), -1), ord=2, dim=1)
        perturbation_loss = F.relu(norms - self.epsilon).mean()
        bounded_delta = _radially_bounded(masked_delta, self.epsilon)
        adversarial = x + bounded_delta
        logits = self._surrogate_logits(adversarial, valid_mask)
        target = torch.full(
            (len(x),), self.target_label, dtype=torch.long, device=x.device,
        )
        adversarial_loss = F.cross_entropy(logits, target)
        wgan_loss = -critic(adversarial).mean()
        total = adversarial_loss + self.alpha * wgan_loss + self.beta * perturbation_loss
        total.backward()
        optimizer.step()
        metrics = (
            total.detach(), adversarial_loss.detach(),
            wgan_loss.detach(), perturbation_loss.detach(),
        )
        del raw_delta, masked_delta, norms, bounded_delta, adversarial
        del logits, target, adversarial_loss, wgan_loss, perturbation_loss, total
        return metrics

    @staticmethod
    def _configured_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
        block = dict(config["attack"])
        params = {key: value for key, value in block.items()
                  if key not in {"name", "params", "training", "variant"}}
        params.update(dict(block.get("params", {})))
        return params

    @classmethod
    def train(
        cls,
        *,
        config: Mapping[str, Any],
        seed: int,
        gpu: str,
    ) -> dict[str, float]:
        """Train from the configured PyTorch *surrogate* and its normalizer.

        ``workflow.train_attack_seed`` does not supply a bound model, so the
        existing workflow factory restores the configured surrogate checkpoint.
        Only that wrapper is bound to the attack; no other classifier is built.
        Dataset physical truth is read once, canonically normalized to float32
        with surrogate checkpoint statistics, then dropped before optimization.
        """
        # Local imports avoid an import cycle with workflow's plugin registry.
        from workflow.factory import (
            _build_model,
            _cfg,
            _inject_dynamic_dataset_shape,
            _model_requires_flatten,
        )
        from workflow.io import (
            _load_checkpoint,
            _load_checkpoint_metadata,
            _normalization_from_checkpoint,
        )
        from orchestrator.device import configure_process_device
        from orchestrator.runner import seed_everything
        from workflow.experiment import ExperimentStore
        from workflow.seeds import role_seed

        cfg = _cfg(config)
        surrogate_cfg = dict(cfg.get("surrogate_model") or {})
        if not surrogate_cfg or not bool(surrogate_cfg.get("enabled", True)):
            raise ValueError("train_attack nids_gan requires an enabled surrogate_model")
        if not surrogate_cfg.get("checkpoint"):
            raise ValueError("surrogate_model.checkpoint must point to a trained surrogate")
        device = configure_process_device(gpu).torch_device
        seed_everything(seed)
        # Offline training belongs to the authorized surrogate's topology,
        # not the frozen evaluation target's. The engine later reshapes the
        # same ordered coordinates when target and surrogate layouts differ.
        cfg["model"] = dict(surrogate_cfg)
        _inject_dynamic_dataset_shape(cfg)
        dataset_cfg = dict(cfg["dataset"])
        dataset = DATASETS.get(str(dataset_cfg["name"]))(
            **dict(dataset_cfg.get("params", {}))
        )
        feature_names = tuple(str(name) for name in dataset.feature_names)
        store = ExperimentStore(cfg.get("experiment_dir"), dataset=dataset_cfg["name"])
        source_seed = role_seed(cfg, "surrogate", seed)
        surrogate_path = store.model_path(
            surrogate_cfg["name"], "none", source_seed, role="surrogates",
        )
        store.validate_template(
            surrogate_cfg["checkpoint"], surrogate_path, source_seed,
            role="surrogate_model",
        )
        store.require("model", surrogate_path, seed=source_seed)
        surrogate_metadata = _load_checkpoint_metadata(surrogate_path)
        normalization = _normalization_from_checkpoint(
            surrogate_metadata, feature_names, role="surrogate model",
        )
        x_physical, labels = dataset.raw_arrays()
        labels = np.asarray(labels, dtype=np.int64)
        from core.provenance import authorized_attack_training_indices
        permitted_indices = authorized_attack_training_indices(dataset, labels, cfg, seed, role='surrogate_model')
        malicious_indices = permitted_indices[labels[permitted_indices] != int(cls._configured_parameters(cfg).get('target_label', 0))]
        if not len(malicious_indices):
            raise ValueError("NIDSGAN training requires non-benign flows")
        from core.dual_representation import encode_normalized32
        if np.asarray(x_physical).dtype != np.float64:
            raise TypeError('NIDSGAN requires original pre-normalization float64 features')
        # Encode only permitted training records, never test/validation data.
        x_physical = x_physical[malicious_indices]
        normalized = encode_normalized32(x_physical, feature_names, normalization.mean, normalization.std)
        x_tensor = torch.from_numpy(np.ascontiguousarray(normalized))
        valid = getattr(dataset, "valid_mask", None)
        valid_tensor = None if valid is None else torch.as_tensor(np.asarray(valid)[malicious_indices], dtype=torch.bool)
        if valid_tensor is not None:
            if valid_tensor.shape[0] != len(x_tensor) or valid_tensor.ndim != 2:
                raise ValueError("dataset.valid_mask must be a (N,T) packet mask")
            training_samples = TensorDataset(x_tensor, valid_tensor)
        else:
            training_samples = TensorDataset(x_tensor)
        attack_params = cls._configured_parameters(cfg)
        attack_params["feature_names"] = feature_names
        attack_params["surrogate_requires_flatten"] = _model_requires_flatten(surrogate_cfg)
        attack = cls(**attack_params)
        num_classes = int(surrogate_metadata["num_classes"])
        if not 0 <= attack.target_label < num_classes:
            raise ValueError("NIDSGAN target_label is outside surrogate class indices")
        input_shape = tuple(int(n) for n in x_tensor.shape[1:])
        # Use the surrogate's original model topology, not the target dataset's
        # potentially flattened representation, to construct its checkpointed
        # architecture.  _surrogate_logits reshapes each batch correspondingly.
        surrogate_input_shape = tuple(int(d) for d in surrogate_metadata["input_shape"])
        if int(np.prod(surrogate_input_shape)) != int(np.prod(input_shape)):
            raise ValueError("Surrogate input width and normalized dataset disagree")
        wrapper = _build_model(
            surrogate_cfg,
            (len(x_tensor), *surrogate_input_shape),
            device=device,
            num_classes=num_classes,
            seed=seed,
        )
        if not isinstance(wrapper, PyTorchModelWrapper):
            raise TypeError("NIDSGAN surrogate_model must be a PyTorch model")
        _load_checkpoint(wrapper, surrogate_path, checkpoint_data=surrogate_metadata)
        attack.bind_model(wrapper)
        wrapper.model.eval()
        wrapper.model.requires_grad_(False)
        # Capture verified, compact training provenance *before* releasing source
        # arrays and dataset references. The actual generator checkpoint seed
        # may differ from the authorized surrogate's model-training seed.
        from core.provenance import trained_generator_provenance
        artifact_provenance = trained_generator_provenance(
            dataset, labels, cfg, seed, source_role="surrogate_model",
            input_shape=surrogate_input_shape, feature_names=feature_names,
        )
        # The original authoritative x_physical remains solely in the dataset.
        # There is no data-dependent physical reconstruction in this trainer.
        del x_physical, normalized, labels
        release = getattr(dataset, "release_raw_features", None)
        if callable(release):
            release()
        del dataset

        training_loader = DataLoader(
            training_samples,
            batch_size=attack.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        dimension = int(np.prod(input_shape))
        generator = _Generator(dimension, attack.hidden_dim, attack.depth).to(device)
        critic = _Critic(dimension, attack.hidden_dim, attack.depth).to(device)
        generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=attack.generator_lr,
            betas=(attack.adam_beta1, attack.adam_beta2),
        )
        critic_optimizer = torch.optim.Adam(
            critic.parameters(),
            lr=attack.critic_lr,
            betas=(attack.adam_beta1, attack.adam_beta2),
        )
        aggregate = {"loss": 0.0, "adversarial_loss": 0.0,
                     "wgan_loss": 0.0, "perturbation_loss": 0.0, "critic_loss": 0.0}
        steps = 0
        for _epoch in range(attack.epochs):
            generator.train()
            critic.train()
            for sample in training_loader:
                x = sample[0].to(device=device, dtype=torch.float32, non_blocking=True)
                batch_valid = None if len(sample) == 1 else sample[1].to(device, non_blocking=True)
                mask = attack._mask(
                    x, feature_names,
                    normalization_mean=normalization.mean,
                    normalization_std=normalization.std,
                    valid_mask=batch_valid,
                )
                critic.requires_grad_(True)
                for _ in range(attack.critic_steps):
                    critic_value = attack._critic_step(
                        generator, critic, critic_optimizer, x, mask,
                    )
                critic.requires_grad_(False)
                losses = attack._generator_step(
                    generator, critic, generator_optimizer, x, mask, batch_valid,
                )
                for name, value in zip(
                    ("loss", "adversarial_loss", "wgan_loss", "perturbation_loss"),
                    losses,
                ):
                    aggregate[name] += float(value.item())
                aggregate["critic_loss"] += float(critic_value.item())
                steps += 1
                # Keep no live graph, logits or batch on the attack object.
                del x, batch_valid, mask, critic_value, losses, sample
                generator_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
            critic.requires_grad_(True)

        checkpoint = store.generator_path(
            cls.attack_name, str(cfg["attack"].get("variant", "default")), seed,
        )
        store.validate_template(attack.checkpoint, checkpoint, seed, role="attack")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_provenance": artifact_provenance,
            "attack_name": cls.attack_name,
            "seed": int(seed),
            "generator_state": {key: tensor.detach().cpu()
                                for key, tensor in generator.state_dict().items()},
            "feature_names": feature_names,
            "input_shape": input_shape,
            "hidden_dim": attack.hidden_dim,
            "depth": attack.depth,
            "epsilon": attack.epsilon,
            "target_label": attack.target_label,
            "surrogate_requires_flatten": attack.surrogate_requires_flatten,
        }
        from uuid import uuid4
        temporary = checkpoint.with_name(checkpoint.name + f".tmp-{uuid4().hex}")
        try:
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        finally:
            temporary.unlink(missing_ok=True)
        # The optimizer states and critic are deliberately not part of inference.
        del payload, generator, critic, wrapper, attack
        return {
            **{key: value / max(1, steps) for key, value in aggregate.items()},
            "training_steps": float(steps),
            "training_samples": float(len(malicious_indices)),
        }

    def _load_generator_once(self, seed: int, shape: tuple[int, ...], device: torch.device) -> _Generator:
        """Lazy checkpoint restore; subsequent batches ignore their offset seed."""
        if self._generator is not None:
            if shape != self._input_shape:
                raise ValueError("NIDSGAN input shape differs from loaded checkpoint")
            return self._generator
        checkpoint = Path(self.checkpoint.format(seed=seed))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"NIDSGAN generator checkpoint not found: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("attack_name") != self.attack_name or int(payload.get("seed", -1)) != seed:
            raise ValueError("NIDSGAN checkpoint attack or seed mismatch")
        from core.provenance import verify_generator_artifact_provenance
        verify_generator_artifact_provenance(payload, getattr(self, 'expected_provenance', None))
        if tuple(payload["feature_names"]) != tuple(self.feature_names):
            raise ValueError("NIDSGAN checkpoint feature schema mismatch")
        stored_shape = tuple(int(d) for d in payload["input_shape"])
        if stored_shape != shape:
            raise ValueError(f"NIDSGAN input shape {shape} != checkpoint shape {stored_shape}")
        if float(payload["epsilon"]) != float(self.epsilon):
            raise ValueError("NIDSGAN epsilon differs from trained checkpoint")
        if int(payload.get("target_label", self.target_label)) != self.target_label:
            raise ValueError("NIDSGAN target_label differs from trained checkpoint")
        if (int(payload["hidden_dim"]) != self.hidden_dim or
                int(payload["depth"]) != self.depth):
            raise ValueError("NIDSGAN generator architecture differs from checkpoint")
        generator = _Generator(int(np.prod(stored_shape)), self.hidden_dim, self.depth)
        generator.load_state_dict(payload["generator_state"], strict=True)
        generator.to(device=device, dtype=torch.float32).eval()
        generator.requires_grad_(False)
        self._generator = generator
        self._loaded_seed = seed
        self._input_shape = shape
        return generator

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Deterministic inference in float32; base class handles float64 finality."""
        del y  # Targets are learned in train(); generation never queries models.
        seed = kwargs.get("seed")
        if self._generator is None and seed is None:
            raise ValueError("First NIDSGAN generation batch requires the worker seed")
        if self._generator is None:
            worker_seed = int(seed)
        else:
            worker_seed = int(self._loaded_seed)
        wrapper = self._require_model()
        if not isinstance(wrapper, PyTorchModelWrapper):
            raise TypeError("NIDSGAN requires a bound PyTorch surrogate")
        device = torch.device(wrapper.device)
        normalized = torch.as_tensor(x, dtype=torch.float32, device=device)
        if normalized.ndim not in (2, 3):
            raise ValueError("NIDSGAN expects normalized (B,F) or (B,T,F) features")
        generator = self._load_generator_once(worker_seed, tuple(normalized.shape[1:]), device)
        names = tuple(kwargs.get("feature_names") or self.feature_names)
        with torch.no_grad():
            mask = self._mask(
                normalized, names,
                normalization_mean=kwargs.get("normalization_mean"),
                normalization_std=kwargs.get("normalization_std"),
                valid_mask=kwargs.get("valid_mask"),
            )
            delta = generator(normalized) * mask
            delta = _radially_bounded(delta, self.epsilon)
            adversarial = (normalized + delta).detach()
        del delta, mask, normalized
        return AttackResult(adversarial=adversarial, feature_names=names)
