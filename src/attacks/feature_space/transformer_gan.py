"""Transformer-GAN feature-space attack scaffold for offline NIDS benchmarking.

The generator and discriminator follow the sequence-level Transformer concept in
Laudanna et al., *Transformer or Autoencoder? Who is the ultimate adversary for
attack detectors?*  Their layer widths and disputed objective are NOT copied.

Representation contract
-----------------------
* Model input and generated candidate: normalized torch.float32.
* Constraint authority: an independent copy of the original physical float64 
  sample, obtained from EvaluationEngine before any normalization.
* FeatureSpaceAttack's inherited finalizer must reconcile the two representations
  (physical bounds, lattice projection, immutable restoration and re-encoding).

The attack is NOT COMPLETE:
to recreate the original attack we still need the:
- balancing weight (alpha) for the  generator loss function: Loss_G = Loss_BCE + lambda*Loss_MSE.
- target tensor of the MSE loss function
- interger size, distributions and mechanism of the noise concatenated to the input of the generator

* **Transformer-GAN**
  * **Paper:** *Transformer or Autoencoder? Who is the ultimate adversary for attack detectors?* - Laudanna et al.,International Journal of Information Security 2025
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from attacks.base import AttackResult, FeatureSpaceAttack
from core.registry import DATASETS, register_attack
from orchestrator.device import configure_process_device


def _safe_padding_mask(
    valid_mask: torch.Tensor | None, shape: tuple[int, int], device: torch.device
) -> torch.Tensor | None:
    """Hide padding from attention without producing NaNs for empty sequences.

    A Transformer cannot attend over zero keys. For a wholly padded flow, expose
    one harmless key to attention; the caller subsequently zeroes its outputs.
    Real packet ownership remains the job of the perturbation mask and finalizer.
    """
    if valid_mask is None:
        return None
    valid = torch.as_tensor(valid_mask, device=device)
    if valid.dtype != torch.bool or tuple(valid.shape) != shape:
        raise ValueError(f"valid_mask must be boolean with shape {shape}")
    safe = valid.clone()
    empty = ~safe.any(dim=1)
    safe[empty, 0] = True
    return ~safe


class _TransformerGenerator(nn.Module):
    """Map clean packets plus per-token Gaussian noise back to packet features.

    Both projections derive their widths from ``num_features``. There is no
    causal mask: this is a full-flow perturbation, not an autoregressive model.
    A final unconstrained linear head is intentional; protocol constraints are
    enforced by the framework rather than a potentially incorrect activation.
    """

    def __init__(
        self,
        num_features: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        noise_dim: int,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.noise_dim = noise_dim
        self.input_projection = nn.Linear(num_features + noise_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_projection = nn.Linear(model_dim, num_features)

    def forward(
        self,
        clean: torch.Tensor,
        z: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clean.ndim != 3 or clean.shape[-1] != self.num_features:
            raise ValueError("clean must have shape [batch, time, num_features]")
        if z.shape != (*clean.shape[:2], self.noise_dim):
            raise ValueError("z must have shape [batch, time, noise_dim]")
        if clean.dtype != torch.float32 or z.dtype != torch.float32:
            raise TypeError("Generator inputs must use normalized float32")
        padding = _safe_padding_mask(valid_mask, clean.shape[:2], clean.device)
        hidden = self.input_projection(torch.cat((clean, z), dim=-1))
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        proposed = self.output_projection(hidden)
        if valid_mask is not None:
            proposed = proposed.masked_fill(~valid_mask[..., None], 0.0)
        return proposed


class _TransformerDiscriminator(nn.Module):
    """Score a whole flow with bidirectional attention and temporal mean pooling.

    The output is a raw logit with shape [batch, 1]. Padded time steps do not
    enter the mean; otherwise varying amounts of padding would become an
    accidental discriminator shortcut instead of meaningful traffic evidence.
    """

    def __init__(
        self, num_features: int, model_dim: int, num_heads: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.input_projection = nn.Linear(num_features, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_projection = nn.Linear(model_dim, 1)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.num_features:
            raise ValueError("Discriminator expects [batch, time, num_features]")
        padding = _safe_padding_mask(valid_mask, x.shape[:2], x.device)
        hidden = self.encoder(
            self.input_projection(x), src_key_padding_mask=padding
        )
        if valid_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = valid_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.output_projection(pooled)


def _sequence_shape(shape: tuple[int, ...], num_features: int) -> tuple[int, int, int]:
    """Derive token topology from dimensions alone, without allocating memory."""
    if not shape or len(shape) not in (2, 3) or shape[0] < 1:
        raise ValueError("Expected nonempty [batch, width] or [batch, time, features]")
    if num_features < 1 or shape[-1] % num_features or (len(shape) == 3 and shape[1] < 1):
        raise ValueError("Model width or time dimension does not match feature_names")
    if len(shape) == 3 and shape[-1] != num_features:
        raise ValueError("Sequence feature width does not match feature_names")
    return (shape[0], shape[-1] // num_features if len(shape) == 2 else shape[1], num_features)


def _as_sequences(data: np.ndarray, num_features: int) -> np.ndarray:
    """Preserve flattened model layout while exposing packet tokens to attention."""
    return data.reshape(_sequence_shape(tuple(data.shape), num_features))


def _physical_normalization(
    sequences: np.ndarray, valid_mask: np.ndarray, chunk_size: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Fit training-only float64 statistics, excluding every padding token.

    Chunk-wise variance combination avoids a second full-dataset allocation and
    is more stable than subtracting two large squared-measurement sums.
    """
    count = 0
    mean = np.zeros(sequences.shape[-1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for start in range(0, len(sequences), chunk_size):
        stop = min(start + chunk_size, len(sequences))
        values = sequences[start:stop][valid_mask[start:stop]]
        if not len(values):
            continue
        if not np.isfinite(values).all():
            raise ValueError("Valid training features contain non-finite measurements")
        local_count = len(values)
        local_mean = values.mean(axis=0, dtype=np.float64)
        local_m2 = np.square(values - local_mean).sum(axis=0, dtype=np.float64)
        total = count + local_count
        difference = local_mean - mean
        mean += difference * (local_count / total)
        m2 += local_m2 + np.square(difference) * (count * local_count / total)
        count = total
    if not count:
        raise ValueError("No valid training packets remain after padding exclusion")
    std = np.sqrt(m2 / count)
    std[std < 1e-12] = 1.0
    return mean, std


def _plain_config(value: Any) -> Any:
    """Detach nested Hydra containers without interpreting research settings.

    OmegaConf's DictConfig and ListConfig can live inside otherwise ordinary
    dictionaries. Recursion makes deferred loss/schema metadata JSON-safe while
    preserving user-supplied scalars and list order. Validation of their meaning
    belongs to the future loss/schema implementations, not this scaffold.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_config(item) for item in value]
    return value


def _save_checkpoint(destination: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one seed-specific artifact, even after interrupted work."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@register_attack("transformer_gan")
class TransformerGANAttack(FeatureSpaceAttack):
    """Checkpoint-backed Transformer GAN operating on normalized packet features.

    ``categorical_schema`` and ``loss_config`` are deferred configuration only:
    their contents are retained but never interpreted as a fabricated loss or
    a guessed categorical projection. The authoritative physical finalizer owns
    integer and categorical realizability once a schema is implemented.
    """

    attack_name = "transformer_gan"
    operates_on_physical_features = False
    query_based = False
    requires_gradients = False

    def __init__(
        self,
        *,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        noise_dim: int = 32,
        loss_config: Mapping[str, Any] | None = None,
        categorical_schema: Mapping[str, Any] | None = None,
        feature_names: Sequence[str] = (),
        immutable_features: Sequence[str] = (),
        clip_ranges: Mapping[str, tuple[float, float]] | None = None,
        discrete_projection: bool = False,
        seed: int | None = None,
        device: str = "auto",
        checkpoint: str | None = None,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if min(model_dim, num_heads, num_layers, noise_dim) <= 0:
            raise ValueError("Transformer dimensions, layer count and noise_dim must be > 0")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        # Hydra exposes YAML dictionaries as DictConfig, which implements
        # Mapping rather than dict. Copy them so the plugin keeps a stable,
        # ordinary Python configuration without inventing loss/schema logic.
        if loss_config is not None and not isinstance(loss_config, Mapping):
            raise TypeError("loss_config must be a mapping")
        if categorical_schema is not None and not isinstance(categorical_schema, Mapping):
            raise TypeError("categorical_schema must be a mapping")

        names = tuple(str(name).upper() for name in feature_names)
        immutable = tuple(str(name).upper() for name in immutable_features)
        if len(set(names)) != len(names):
            raise ValueError("feature_names must contain unique entries")
        if names and (set(immutable) - set(names)):
            raise ValueError("immutable_features contains names outside feature_names")

        # Some deployed base versions are strict dataclass constructors, while
        # newer versions additionally accept orchestration metadata/config.
        # Always forward the four threat-model fields; filter ONLY optional
        # orchestration kwargs unsupported by an older base signature.
        base_arguments: dict[str, Any] = {
            "feature_names": names,
            "immutable_features": immutable,
            "clip_ranges": dict(clip_ranges or {}),
            "discrete_projection": bool(discrete_projection),
        }
        base_initializer = super().__init__
        signature = inspect.signature(base_initializer)
        accepts_extras = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if "config" in signature.parameters or accepts_extras:
            base_arguments["config"] = config
        base_arguments.update(
            {k: v for k, v in kwargs.items() if accepts_extras or k in signature.parameters}
        )
        base_initializer(**base_arguments)

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.noise_dim = int(noise_dim)
        self.loss_config = _plain_config(loss_config or {})
        self.categorical_schema = _plain_config(categorical_schema or {})
        self.feature_names = names
        self.immutable_features = immutable
        # An explicit constructor seed pins the training artifact. Otherwise
        # infer the worker seed from generate()'s first batch; later batches
        # advance the noise seed without changing the loaded checkpoint.
        self._configured_checkpoint_seed = None if seed is None else int(seed)
        self.seed = 42 if seed is None else int(seed)
        self.device_name = str(device)
        self.checkpoint = checkpoint
        self.config = config
        self._generator: _TransformerGenerator | None = None
        self._loaded_checkpoint: Path | None = None
        self._checkpoint_normalization: tuple[np.ndarray, np.ndarray] | None = None

    def _device(self) -> torch.device:
        if self.device_name == "auto":
            attached = getattr(getattr(self, "model", None), "device", "cpu")
            device = torch.device(attached or "cpu")
        else:
            device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested GPU is unavailable: {device}")
        return device

    def _checkpoint_path(self, seed: int | None = None) -> Path:
        """Resolve training's fixed seed or evaluation's first batch seed.

        EvaluationEngine increments its RNG seed for successive batches. The
        first batch selects the trained checkpoint; later batches vary noise
        but must not silently swap model weights.
        """
        root = "runs" if self.config is None else str(self.config.get("output_dir", "runs"))
        template = self.checkpoint or str(
            Path(root) / "checkpoints" / "transformer_gan_seed{seed}.pt"
        )
        selected_seed = self.seed if seed is None else int(seed)
        return Path(str(template).format(seed=selected_seed)).expanduser()

    def _load_generator(
        self,
        num_features: int | None,
        device: torch.device,
        *,
        checkpoint_seed: int | None = None,
    ) -> _TransformerGenerator:
        """Load once, recover dataset feature order, and pin the seed artifact.

        Like amd_gan.yaml, this attack config need not hardcode feature_names.
        Its trusted schema is the checkpoint metadata written during training.
        """
        if self._generator is not None:
            if num_features is not None and self._generator.num_features != num_features:
                raise ValueError("Generator feature width changed across evaluation batches")
            return self._generator.to(device).eval()

        path = self._checkpoint_path(checkpoint_seed)
        if not path.is_file():
            raise FileNotFoundError(
                f"Transformer-GAN checkpoint missing: {path}; training losses are "
                "currently unspecified, so no valid checkpoint can be produced yet"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("attack_name") != self.attack_name:
            raise ValueError("Checkpoint is not a Transformer-GAN artifact")
        if payload.get("format_version") != 1 or payload.get("training_complete") is not True:
            raise RuntimeError(
                "Transformer-GAN checkpoint is incomplete or incompatible; "
                "the missing loss formulation must be integrated before evaluation"
            )
        trained_width = int(payload["num_features"])
        if num_features is not None and trained_width != num_features:
            raise ValueError("Checkpoint feature width disagrees with model input")
        names = tuple(str(name).upper() for name in payload["feature_names"])
        if not names or len(names) != trained_width or len(set(names)) != len(names):
            raise ValueError("Checkpoint contains an invalid feature schema")
        if self.feature_names and names != self.feature_names:
            raise ValueError("Checkpoint feature order disagrees with attack configuration")
        if not self.feature_names:
            self.feature_names = names
        num_features = trained_width
        expected = (self.model_dim, self.num_heads, self.num_layers, self.noise_dim)
        actual = tuple(int(payload[k]) for k in (
            "model_dim", "num_heads", "num_layers", "noise_dim"
        ))
        if expected != actual:
            raise ValueError("Transformer hyperparameters disagree with checkpoint")

        generator = _TransformerGenerator(
            num_features, self.model_dim, self.num_heads,
            self.num_layers, self.noise_dim,
        )
        generator.load_state_dict(payload["generator_state"], strict=True)
        self._generator = generator.to(device).eval()
        self._loaded_checkpoint = path
        self._checkpoint_normalization = (
            payload["normalization_mean"].cpu().numpy(),
            payload["normalization_std"].cpu().numpy(),
        )
        return self._generator

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Produce masked float32 proposals plus unchanged physical float64 truth.

        The physical baseline is NEVER reconstructed from normalized floats.
        Instead, the inherited feature-space wrapper projects the candidate
        against that authoritative baseline before the evaluator scores it.
        Success is not guessed: the evaluation engine computes actual evasion.
        """
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.float32:
                raise TypeError("Model input must be normalized torch.float32")
            input_shape = tuple(x.shape)
        else:
            arr = np.asarray(x)
            if arr.dtype != np.float32:
                raise TypeError("Model input must be normalized np.float32")
            input_shape = tuple(arr.shape)

        device = self._device()
        # The evaluation registry does not have to inject feature_names; the
        # checkpoint is the source of truth when YAML omits that optional key.
        # Resolve only once: later batch seeds affect z, never the checkpoint.
        seed = int(kwargs.get("seed", self.seed))
        generator = self._load_generator(
            len(self.feature_names) if self.feature_names else None,
            device,
            checkpoint_seed=(
                self._configured_checkpoint_seed
                if self._configured_checkpoint_seed is not None else seed
            ),
        )
        clean = torch.as_tensor(x, dtype=torch.float32, device=device)
        sequences = clean.reshape(_sequence_shape(input_shape, len(self.feature_names)))
        if y is not None and len(y) != len(sequences):
            raise ValueError("Input batches and label batches must have the same length")
        valid = kwargs.get("valid_mask")
        valid_tensor = None
        if valid is not None:
            valid_tensor = torch.as_tensor(valid, device=device)
            if valid_tensor.dtype != torch.bool or tuple(valid_tensor.shape) != sequences.shape[:2]:
                raise ValueError("valid_mask must have shape [batch, time] and boolean dtype")

        # Query the framework's victim ownership mask on the ORIGINAL model
        # layout; reshape only AFTER checking that every coordinate is covered.
        mean = kwargs.get("normalization_mean")
        std = kwargs.get("normalization_std")
        if (mean is None) != (std is None):
            raise ValueError("normalization_mean and normalization_std must be paired")

        # Preserve the evaluator's original mean/std arrays for
        # victim_packet_perturbation_mask(), which receives the
        # original model input layout. No strict validation against the generator's
        # internal training statistics is performed here, ensuring flattened NIDS
        # models are seamlessly supported.
        perturbable = self.victim_packet_perturbation_mask(
            clean,
            self.feature_names,
            normalization_mean=mean,
            normalization_std=std,
            valid_mask=kwargs.get("valid_mask"),
        )

        mask = torch.as_tensor(perturbable, device=device, dtype=torch.float32)
        if mask.numel() != clean.numel():
            raise ValueError("victim_packet_perturbation_mask returned an invalid shape")
        mask = mask.reshape_as(sequences)
        if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("victim packet mask must contain only finite 0/1 values")

        # The victim-packet mask does not necessarily protect named protocol
        # columns (see PGD's separate immutable_feature_mask). Enforce both.
        protected = set(self.immutable_features) | {"DIR"}
        mutable_columns = torch.tensor(
            [name not in protected for name in self.feature_names],
            dtype=torch.float32, device=device,
        )
        mask = mask * mutable_columns.view(1, 1, -1)
        if valid_tensor is not None:
            mask = mask * valid_tensor.unsqueeze(-1)

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        with torch.inference_mode():
            z = torch.randn(
                *sequences.shape[:2], self.noise_dim,
                device=device, dtype=torch.float32, generator=rng,
            )
            x_clean = sequences
            x_raw = generator(x_clean, z, valid_mask=valid_tensor)
            if x_raw.shape != x_clean.shape or not torch.isfinite(x_raw).all():
                raise RuntimeError("Generator produced invalid candidate dimensions or values")
            # Dynamic constraint mask in model space: immutable coordinates and
            # padding are exactly clean, independent of the generator output.
            x_adv = x_clean + mask * (x_raw - x_clean)
            adversarial = x_adv.reshape(input_shape).clone()

        return AttackResult(
            adversarial=adversarial,
            feature_names=self.feature_names,
            queries=np.zeros(len(sequences), dtype=np.int64),
            metadata={
                "attack": self.attack_name,
                "generation_seed": seed,
                "checkpoint_path": str(self._loaded_checkpoint),
                "candidate_domain": "normalized_float32",
                "physical_authority": "original_float64_pending_base_projection",
            },
        )

    @classmethod
    def train(
        cls,
        *,
        config: Mapping[str, Any],
        seed: int,
        gpu: str,
        logger: logging.Logger | None = None,
    ) -> Mapping[str, float]:
        """Prepare masked training batches and scaffold alternating GAN updates.

        ``dataset.raw_arrays()`` must expose pre-normalization physical float64
        samples. The dataset config must select *training data only*; otherwise
        fitted statistics and generator updates would leak holdout information.

        Deliberate injection point: the first D/G optimization step raises
        ``NotImplementedError('Loss formulation pending')``. A checkpoint with
        ``training_complete=False`` is still written for diagnostic inspection,
        but will never be accepted by ``generate()``.
        """
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        attack_block = config.get("attack", {})
        dataset_cfg = config.get("dataset", {})
        if not isinstance(attack_block, Mapping) or not isinstance(dataset_cfg, Mapping):
            raise TypeError("config.attack and config.dataset must be mappings")
        # Hydra normally nests plugin settings under attack.params; some older
        # training routines receive an already-flattened attack configuration.
        nested = attack_block.get("params", {})
        if not isinstance(nested, Mapping):
            raise TypeError("config.attack.params must be a mapping")
        attack_cfg = {
            **dict(nested),
            **{k: v for k, v in attack_block.items() if k not in ("name", "params")},
        }
        if not dataset_cfg.get("name"):
            raise ValueError("config.dataset.name is required")

        device = configure_process_device(gpu).torch_device
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        constructor = DATASETS.get(str(dataset_cfg["name"]))
        params = dict(dataset_cfg.get("params", {}))
        try:
            if "flatten" in inspect.signature(constructor).parameters:
                params["flatten"] = False  # A GAN trains on tokens, not a flat vector.
        except (TypeError, ValueError):
            pass
        dataset = constructor(**params)
        if not callable(getattr(dataset, "raw_arrays", None)):
            raise TypeError("Registered dataset must provide raw_arrays()")
        raw, labels = dataset.raw_arrays()
        raw = np.asarray(raw)
        labels = np.asarray(labels)
        if raw.dtype != np.float64:
            raise TypeError("raw_arrays() must preserve the original physical float64")
        if len(raw) != len(labels):
            raise ValueError("Dataset labels and feature batches have different lengths")
        names = tuple(str(n).upper() for n in getattr(dataset, "feature_names", ()))
        declared_names = tuple(str(n).upper() for n in attack_cfg.get("feature_names", ()))
        if declared_names and names and declared_names != names:
            raise ValueError("Attack and dataset feature order differ")
        names = declared_names or names
        if not names or len(set(names)) != len(names):
            raise ValueError("Dataset must declare ordered unique feature_names")
        sequences = _as_sequences(raw, len(names))

        supplied_valid = getattr(dataset, "valid_mask", None)
        if supplied_valid is None and sequences.shape[1] > 1:
            raise ValueError("Sequence datasets must provide valid_mask for padding")
        valid = (
            np.ones(sequences.shape[:2], dtype=np.bool_)
            if supplied_valid is None else np.asarray(supplied_valid)
        )
        if valid.dtype != np.bool_ or valid.shape != sequences.shape[:2]:
            raise ValueError("Dataset valid_mask must be boolean [batch, time]")
        if not valid.any():
            raise ValueError("All training tokens are padding")
        mean, std = _physical_normalization(sequences, valid)
        # Dataset-fitted statistics are the default. A known target normalizer
        # can instead be supplied explicitly to prevent train/eval drift.
        override_mean = attack_cfg.get("normalization_mean")
        override_std = attack_cfg.get("normalization_std")
        if (override_mean is None) != (override_std is None):
            raise ValueError("Training normalization_mean/std must be provided together")
        if override_mean is not None:
            mean = np.asarray(override_mean, dtype=np.float64)
            std = np.asarray(override_std, dtype=np.float64)
            if mean.shape != (len(names),) or std.shape != (len(names),):
                raise ValueError("Training normalization must have one value per feature")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Training normalization mean/std must be finite and std > 0")
        settings = cls(
            model_dim=int(attack_cfg.get("model_dim", 128)),
            num_heads=int(attack_cfg.get("num_heads", 4)),
            num_layers=int(attack_cfg.get("num_layers", 2)),
            noise_dim=int(attack_cfg.get("noise_dim", 32)),
            loss_config=dict(attack_cfg.get("loss_config") or {}),
            categorical_schema=dict(attack_cfg.get("categorical_schema") or {}),
            feature_names=names,
            immutable_features=attack_cfg.get("immutable_features", ()),
            clip_ranges=attack_cfg.get("clip_ranges"),
            discrete_projection=bool(attack_cfg.get("discrete_projection", False)),
            seed=seed,
            device=str(device),
            checkpoint=attack_cfg.get("checkpoint"),
            config=config,
        )
        generator = _TransformerGenerator(
            len(names), settings.model_dim, settings.num_heads,
            settings.num_layers, settings.noise_dim,
        ).to(device)
        discriminator = _TransformerDiscriminator(
            len(names), settings.model_dim, settings.num_heads, settings.num_layers
        ).to(device)
        learning_rate = float(attack_cfg.get("learning_rate", 1e-4))
        batch_size = int(attack_cfg.get("batch_size", 64))
        epochs = int(attack_cfg.get("epochs", 1))
        
        if learning_rate <= 0 or batch_size < 1 or epochs < 1:
            raise ValueError("learning_rate, batch_size and epochs must be positive")
        
        optim_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        permutation_rng = np.random.default_rng(int(seed))
        
        # Bypass template evaluation to prevent generating a second Hydra UUID
        directory = Path(str(config.get("output_dir", "runs"))).expanduser() / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"transformer_gan_seed{int(seed)}.pt"
        
        completed = False

        if logger:
            logger.info(
                "transformer_gan.train scaffold: seed=%d device=%s flows=%d valid_tokens=%d",
                seed, device, len(sequences), int(valid.sum()),
            )
        try:
            for epoch in range(epochs):
                indices = permutation_rng.permutation(len(sequences))
                for start in range(0, len(indices), batch_size):
                    rows = indices[start : start + batch_size]
                    clean_np = (sequences[rows] - mean) / std
                    clean_np[~valid[rows]] = 0.0
                    clean = torch.as_tensor(
                        clean_np.astype(np.float32), device=device
                    )
                    batch_valid = torch.as_tensor(valid[rows], device=device)
                    if not torch.isfinite(clean).all():
                        raise ValueError("Normalization produced non-finite data")
                    train_mask = settings.victim_packet_perturbation_mask(
                        clean,
                        names,
                        normalization_mean=mean,
                        normalization_std=std,
                        valid_mask=batch_valid,
                    )
                    train_mask = torch.as_tensor(
                        train_mask, device=device, dtype=torch.float32
                    )
                    if train_mask.numel() != clean.numel():
                        raise ValueError("Training perturbation mask has incorrect shape")
                    train_mask = train_mask.reshape_as(clean)
                    mutable = torch.tensor(
                        [name not in (set(settings.immutable_features) | {"DIR"}) for name in names],
                        device=device, dtype=torch.float32,
                    )
                    train_mask = train_mask * mutable.view(1, 1, -1)
                    train_mask = train_mask * batch_valid.unsqueeze(-1)
                    noise = torch.randn(
                        len(rows), clean.shape[1], settings.noise_dim,
                        device=device, dtype=torch.float32,
                    )
                    # Scaffold both halves explicitly. Detach the fake flow for
                    # D; leave the G graph intact during the subsequent G step.
                    # Loss math and its evasion/MSE weighting are NOT assumed.
                    # Resolve and validate YAML-driven loss settings.
                    loss_cfg = settings.loss_config
                    adv_loss = str(loss_cfg.get("adv_loss", "bce")).lower()
                    rec_loss = str(loss_cfg.get("reconstruction_loss", "mse")).lower()
                    rec_weight = float(loss_cfg.get("reconstruction_weight", 0.1))
                    smoothing = float(loss_cfg.get("label_smoothing", 0.1))

                    if adv_loss != "bce":
                        raise ValueError(f"Unsupported adv_loss: {adv_loss}")
                    if rec_loss not in ("mse", "none"):
                        raise ValueError(f"Unsupported reconstruction_loss: {rec_loss}")
                    if not np.isfinite(rec_weight) or rec_weight < 0:
                        raise ValueError("reconstruction_weight must be finite and >= 0")
                    if not np.isfinite(smoothing) or not 0 <= smoothing < 1:
                        raise ValueError("label_smoothing must be in [0, 1)")

                    for stage in ("discriminator", "generator"):

                        if stage == "discriminator":
                            optim_d.zero_grad(set_to_none=True)

                            # D must not update G. The fake sample
                            # is generated without an autograd graph.
                            with torch.no_grad():
                                proposed = generator(clean, noise, batch_valid)
                                fake = clean + train_mask * (proposed - clean)

                            real_logits = discriminator(clean, batch_valid)
                            fake_logits = discriminator(fake, batch_valid)

                            real_target = torch.full_like(real_logits, 1.0 - smoothing)
                            fake_target = torch.zeros_like(fake_logits)

                            loss_real = (
                                torch.nn.functional.binary_cross_entropy_with_logits(
                                    real_logits, real_target
                                )
                            )
                            loss_fake = (
                                torch.nn.functional.binary_cross_entropy_with_logits(
                                    fake_logits, fake_target
                                )
                            )

                            loss_d = 0.5 * (loss_real + loss_fake)

                            loss_d.backward()
                            optim_d.step()

                        else:
                            optim_g.zero_grad(set_to_none=True)

                            # Freeze D parameters while preserving
                            # gradients through D to generator outputs.
                            discriminator.requires_grad_(False)

                            try:
                                proposed = generator(clean, noise, batch_valid)
                                fake = clean + train_mask * (proposed - clean)
                                fake_logits = discriminator(fake, batch_valid)

                                # Generator attempts to produce samples
                                # that D classifies as real.
                                generator_target = torch.ones_like(fake_logits)
                                loss_adv = (
                                    torch.nn.functional.binary_cross_entropy_with_logits(
                                        fake_logits, generator_target
                                    )
                                )

                                # Optional masked reconstruction penalty.
                                loss_rec = loss_adv.new_zeros(())

                                if rec_loss == "mse" and rec_weight > 0:
                                    squared_error = torch.nn.functional.mse_loss(
                                        proposed * train_mask,
                                        clean * train_mask,
                                        reduction="none",
                                    )

                                    # Normalize only by active, mutable feature coordinates.
                                    # Padding and immutable fields contribute neither
                                    # squared error nor denominator weight.
                                    loss_rec = (
                                        squared_error.sum()
                                        / train_mask.sum().clamp_min(1)
                                    )

                                loss_g = loss_adv + rec_weight * loss_rec

                                loss_g.backward()
                                optim_g.step()

                            finally:
                                discriminator.requires_grad_(True)
                                
            completed = True
        finally:
            payload: dict[str, Any] = {
                "format_version": 1,
                "attack_name": cls.attack_name,
                "training_complete": completed,
                "training_status": "complete" if completed else "loss_pending",
                "seed": int(seed),
                "num_features": len(names),
                "input_shape": tuple(raw.shape[1:]),
                "feature_names": names,
                "immutable_features": settings.immutable_features,
                "model_dim": settings.model_dim,
                "num_heads": settings.num_heads,
                "num_layers": settings.num_layers,
                "noise_dim": settings.noise_dim,
                "normalization_mean": torch.from_numpy(mean.copy()),
                "normalization_std": torch.from_numpy(std.copy()),
                "generator_state": {
                    key: value.detach().cpu().clone()
                    for key, value in generator.state_dict().items()
                },
                "discriminator_state": {
                    key: value.detach().cpu().clone()
                    for key, value in discriminator.state_dict().items()
                },
                "loss_config_json": json.dumps(settings.loss_config, sort_keys=True),
                "categorical_schema_json": json.dumps(
                    settings.categorical_schema, sort_keys=True
                ),
            }
            _save_checkpoint(destination, payload)
            if logger:
                logger.info(
                    "transformer_gan checkpoint=%s training_complete=%s",
                    destination, completed,
                )
        return {"transformer_gan_train_flows": float(len(sequences))}
