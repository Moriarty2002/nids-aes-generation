"""TDTCE: Sui et al., The Computer Journal (2026), Algorithms 1–2.

Original: VAE latent diffusion with AdaLN-G/TDiT and five losses: reconstruction,
KL, adversarial, noise prediction and Mahalanobis regularization.
Mandatory NIDS adaptation: ordered T=20 packet masks, independent train-fitted
surrogate, physical finalizer; no original target-query/SVGP imitation.
Source recipe: Table 1 AdaLN-G I, 6 blocks/p=4, but patches now group
ordered packet tokens rather than the paper's 16x16 image cells. The paper's
classifier-query SVGP and its full PCAP mapping are prohibited/unavailable;
loss weights, beta schedule and training duration are declared NIDS choices.
Topology: flat (B,T*F) and masked (B,T,F), no PCAP wire certification.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from attacks.base import AttackResult, FeatureSpaceAttack
from core.dual_representation import encode_normalized32
from core.provenance import authorized_attack_training_indices
from core.registry import DATASETS, NIDS, register_attack
from models.wrapper import PyTorchModelWrapper
from orchestrator.device import configure_process_device


# TDTCE follows the latent-diffusion formulation described in Sui et al.,
# "TDTCE: generating adversarial instances against NIDS via diffusion model",
# The Computer Journal, 2026, Section 4.3 and Algorithms 1-2.
# https://github.com/HHLinlife/TDTCE


def _sinusoidal_embedding(timesteps: Tensor, dimension: int, max_period: float = 10_000.0) -> Tensor:
    """Build timestep/position embeddings without a fixed sequence length."""
    if dimension < 1:
        raise ValueError("Embedding dimension must be positive")
    half = dimension // 2
    if half == 0:
        return timesteps.to(torch.float32).reshape(-1, 1)
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.to(torch.float32).reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def ddpm_forward_noising(clean_latent: Tensor, noise: Tensor, alpha_bar: Tensor) -> Tensor:
    """Sui et al. Eq. 1 forward marginal (DDPM): exact latent/noise shape.

    Valid packet masking is applied by the caller before and after noising;
    this function never creates physical features or performs model queries.
    """
    if clean_latent.shape != noise.shape:
        raise ValueError('Latent and sampled noise must have identical shape')
    if not torch.isfinite(alpha_bar).all() or torch.any((alpha_bar < 0) | (alpha_bar > 1)):
        raise ValueError('DDPM alpha_bar must be finite in [0,1]')
    if not torch.isfinite(clean_latent).all() or not torch.isfinite(noise).all():
        raise ValueError('DDPM latent/noise must be finite')
    return torch.sqrt(alpha_bar) * clean_latent + torch.sqrt(1.0 - alpha_bar) * noise


def ddpm_reverse_step(noisy_latent: Tensor, predicted_noise: Tensor, *,
                      alpha: Tensor, beta: Tensor, alpha_bar: Tensor,
                      posterior_sigma: Tensor, posterior_noise: Tensor | None) -> Tensor:
    """Sui et al. Eq. 2, with DDPM posterior sigma for t>1.

    The caller provides its independently seeded noise sample and implements
    the final noiseless t=1 step. No classifier oracle is consulted here.
    """
    if noisy_latent.shape != predicted_noise.shape:
        raise ValueError('Latent and predicted noise shapes must agree')
    if (not all(bool(torch.isfinite(v).all()) for v in (alpha,beta,alpha_bar,posterior_sigma))
            or torch.any((alpha <= 0) | (alpha > 1)) or torch.any((beta < 0) | (beta >= 1))
            or torch.any((alpha_bar <= 0) | (alpha_bar > 1)) or torch.any(posterior_sigma < 0)):
        raise ValueError('Invalid DDPM reverse coefficients')
    if not torch.isfinite(noisy_latent).all() or not torch.isfinite(predicted_noise).all():
        raise ValueError('DDPM reverse latent/noise must be finite')
    if torch.any(posterior_sigma > 0) and posterior_noise is None:
        raise ValueError('Nonzero posterior variance requires an explicit noise draw')
    if posterior_noise is not None and posterior_noise.shape != noisy_latent.shape:
        raise ValueError('Posterior noise must match latent shape')
    mean = (noisy_latent - beta / torch.sqrt(1.0-alpha_bar).clamp_min(1e-8)
            * predicted_noise) / torch.sqrt(alpha)
    return mean if posterior_noise is None else mean + posterior_sigma * posterior_noise


class SparseMultiheadAttention(nn.Module):
    """Local multi-head self-attention with bounded temporal neighborhoods.

    Unlike a dense attention mask, keys and values are gathered only from the
    local window, so the attention intermediate memory scales with batch size,
    sequence length, and window width rather than sequence-length squared.
    All operations use standard PyTorch tensor primitives.
    """

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int = 16) -> None:
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.window_size = int(window_size)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        if sequence_length == 0:
            return x

        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        radius = max(0, self.window_size // 2)
        offsets = torch.arange(
            -radius, radius + 1, device=x.device, dtype=torch.long
        )
        positions = torch.arange(sequence_length, device=x.device, dtype=torch.long)
        indices = positions[:, None] + offsets[None, :]
        valid = (indices >= 0) & (indices < sequence_length)
        safe_indices = indices.clamp(0, sequence_length - 1)

        # Advanced indexing allocates only the local neighborhood tensor:
        # (B, H, L, window, Dh).
        local_k = k[:, :, safe_indices, :]
        local_v = v[:, :, safe_indices, :]
        local_valid = valid.reshape(1, sequence_length, -1).expand(batch_size, -1, -1)
        if valid_mask is not None:
            if valid_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"attention valid_mask must be {(batch_size, sequence_length)}, "
                    f"got {tuple(valid_mask.shape)}"
                )
            local_key_valid = valid_mask[:, safe_indices]
            local_valid = local_valid & local_key_valid
            # Linear projection biases must not re-introduce padded values.
            local_v = local_v * local_key_valid[:, None, :, :, None].to(local_v.dtype)
        scores = (q.unsqueeze(-2) * local_k).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~local_valid[:, None, :, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        attended = (weights.unsqueeze(-1) * local_v).sum(dim=-2)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_dim)
        output = self.output(attended)
        if valid_mask is not None:
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        return output


class AdaLNGBlock(nn.Module):
    """Transformer block with adaptive LayerNorm and Gaussian residual gates.

    The AdaLN-G residual scale parameters alpha are initialized from
    N(0, 0.01^2), making the block initially close to an identity mapping.
    Time and class conditions jointly predict per-channel scale and shift.
    """

    def __init__(
        self,
        hidden_dim: int,
        condition_dim: int,
        *,
        num_heads: int = 4,
        attention_window: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Fig4 regresses modulation from t+c with a GRU. Source widths and
        # recurrence length are unspecified: one hidden-width condition step
        # is the declared packet interpretation, not an author setting.
        self.condition_gru = nn.GRU(condition_dim, hidden_dim, batch_first=True)
        self.attention_modulation = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.mlp_modulation = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.attention = SparseMultiheadAttention(
            hidden_dim, num_heads, window_size=attention_window
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.alpha_attention = nn.Parameter(torch.empty(hidden_dim))
        self.alpha_mlp = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.alpha_attention, mean=0.0, std=0.01)
        nn.init.normal_(self.alpha_mlp, mean=0.0, std=0.01)
        nn.init.zeros_(self.attention_modulation.weight)
        nn.init.zeros_(self.attention_modulation.bias)
        nn.init.zeros_(self.mlp_modulation.weight)
        nn.init.zeros_(self.mlp_modulation.bias)

    @staticmethod
    def _adaptive_norm(x: Tensor, norm: nn.Module, modulation: Tensor) -> Tensor:
        scale, shift = modulation.chunk(2, dim=-1)
        return norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self, x: Tensor, condition: Tensor, valid_mask: Tensor | None = None
    ) -> Tensor:
        condition, _ = self.condition_gru(condition.unsqueeze(1))
        condition = condition[:, 0]
        token_mask = None if valid_mask is None else valid_mask.unsqueeze(-1).to(x.dtype)
        attention_input = self._adaptive_norm(
            x, self.norm_attention, self.attention_modulation(condition)
        )
        x = x + self.alpha_attention.reshape(1, 1, -1) * self.attention(
            attention_input, valid_mask=valid_mask
        )
        if token_mask is not None:
            x = x * token_mask
        mlp_input = self._adaptive_norm(x, self.norm_mlp, self.mlp_modulation(condition))
        x = x + self.alpha_mlp.reshape(1, 1, -1) * self.mlp(mlp_input)
        return x if token_mask is None else x * token_mask


class TDiTBackbone(nn.Module):
    """Feature-dimension-agnostic, class- and timestep-conditioned diffusion net.

    ``latent_tokens`` has shape ``(B, T, latent_dim)``. Adjacent temporal tokens
    are grouped according to ``patch_size``; sequence lengths are padded and
    cropped dynamically, not reshaped to a fixed image grid.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_classes: int,
        timesteps: int,
        *,
        patch_size: int = 1,
        num_heads: int = 4,
        adaln_mode: str = "adaln_g",
        attention_window: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        normalized_mode = adaln_mode.strip().lower().replace("-", "_")
        if normalized_mode not in {"adaln_g", "adalng"}:
            raise ValueError(
                f"Unsupported adaln_mode={adaln_mode!r}; this implementation supports 'adaln_g'."
            )
        if latent_dim < 1 or hidden_dim < 1 or num_blocks < 1 or num_classes < 1 or timesteps < 1:
            raise ValueError("latent_dim, hidden_dim, num_blocks, num_classes, and timesteps must be positive")
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.timesteps = int(timesteps)
        self.input_projection = nn.Linear(latent_dim * patch_size, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNGBlock(
                    hidden_dim,
                    hidden_dim,
                    num_heads=num_heads,
                    attention_window=attention_window,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        # Fig3 output head uses RMS normalization, unlike the LayerNorm
        # inside Fig4 residual blocks. Epsilon is a local numerical choice.
        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.output_projection = nn.Linear(hidden_dim, latent_dim * patch_size)

    def forward(
        self,
        latent_tokens: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if latent_tokens.ndim != 3 or latent_tokens.shape[-1] != self.latent_dim:
            raise ValueError(
                "TDiTBackbone expects latent tokens shaped (batch, sequence, latent_dim); "
                f"got {tuple(latent_tokens.shape)}"
            )
        batch_size, sequence_length, _ = latent_tokens.shape
        if timesteps.ndim != 1 or labels.ndim != 1:
            raise ValueError("timesteps and labels must be one-dimensional batch tensors")
        if len(timesteps) != batch_size or len(labels) != batch_size:
            raise ValueError("latent tokens, timesteps, and labels must have matching batch sizes")

        if valid_mask is not None:
            if valid_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"TDiT valid_mask must be {(batch_size, sequence_length)}, "
                    f"got {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask.to(device=latent_tokens.device, dtype=torch.bool)
            latent_tokens = latent_tokens * valid_mask.unsqueeze(-1).to(latent_tokens.dtype)

        pad_length = (-sequence_length) % self.patch_size
        if pad_length:
            latent_tokens = F.pad(latent_tokens, (0, 0, 0, pad_length))
            if valid_mask is not None:
                valid_mask = F.pad(valid_mask, (0, pad_length), value=False)
        padded_length = latent_tokens.shape[1]
        token_count = padded_length // self.patch_size
        patch_valid_mask = (
            None if valid_mask is None else valid_mask.reshape(batch_size, token_count, self.patch_size).any(dim=-1)
        )
        patch_tokens = latent_tokens.reshape(
            batch_size, token_count, self.patch_size * self.latent_dim
        )
        hidden = self.input_projection(patch_tokens)

        positions = torch.arange(token_count, device=hidden.device, dtype=torch.float32)
        position_embedding = _sinusoidal_embedding(positions, self.hidden_dim).unsqueeze(0)
        hidden = hidden + position_embedding.to(dtype=hidden.dtype)
        time_embedding = self.time_embedding(
            _sinusoidal_embedding(timesteps, self.hidden_dim).to(dtype=hidden.dtype)
        )
        condition = time_embedding + self.class_embedding(labels.long())

        if patch_valid_mask is not None:
            hidden = hidden * patch_valid_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, condition, valid_mask=patch_valid_mask)

        output = self.output_projection(self.final_norm(hidden))
        output = output.reshape(batch_size, token_count, self.patch_size, self.latent_dim)
        output = output.reshape(batch_size, padded_length, self.latent_dim)
        output = output[:, :sequence_length]
        if valid_mask is not None:
            output = output * valid_mask[:, :sequence_length].unsqueeze(-1).to(output.dtype)
        return output


class TrafficVAE(nn.Module):
    """Packet-feature VAE that preserves 2-D and 3-D input tensor structure.

    For tabular data the input is ``(B, F)``; for sequences it is ``(B, T, F)``.
    The encoder and decoder operate over the last feature axis and do not assume
    a fixed number of packets or a fixed feature width.
    """

    def __init__(self, feature_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if feature_dim < 1 or latent_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim, latent_dim, and hidden_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_projection = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_projection = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def _as_sequence(self, x: Tensor) -> tuple[Tensor, int]:
        if x.ndim not in {2, 3}:
            raise ValueError(f"TrafficVAE expects (B,F) or (B,T,F), got {tuple(x.shape)}")
        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"TrafficVAE was initialized for {self.feature_dim} features, got {x.shape[-1]}"
            )
        input_ndim = x.ndim
        return (x.unsqueeze(1) if x.ndim == 2 else x), input_ndim

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        sequence, _ = self._as_sequence(x)
        hidden = self.encoder(sequence)
        return self.mean_projection(hidden), self.log_variance_projection(hidden).clamp(-20.0, 10.0)

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_variance)
        return mean + std * torch.randn_like(std)

    def decode(self, latent: Tensor, *, output_ndim: int = 3) -> Tensor:
        decoded = self.decoder(latent)
        if output_ndim == 2:
            if decoded.shape[1] != 1:
                raise ValueError("2-D output requires a single latent token")
            return decoded[:, 0]
        if output_ndim != 3:
            raise ValueError(f"Unsupported output rank {output_ndim}")
        return decoded

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sequence, input_ndim = self._as_sequence(x)
        hidden = self.encoder(sequence)
        mean = self.mean_projection(hidden)
        log_variance = self.log_variance_projection(hidden).clamp(-20.0, 10.0)
        latent = self.reparameterize(mean, log_variance)
        reconstruction = self.decode(latent, output_ndim=input_ndim)
        return reconstruction, mean, log_variance, latent


class _NormalizedArrayDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Normalize raw samples lazily, avoiding a second full-corpus feature matrix."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        valid_mask: np.ndarray | None = None,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self.features = features
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)
        self.feature_names = feature_names or tuple(f"F{i}" for i in range(features.shape[-1]))
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid checkpoint normalizer")
        self.valid_mask = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        sample_index = int(self.indices[index])
        raw_sample = self.features[sample_index:sample_index+1]
        normalized = encode_normalized32(raw_sample, self.feature_names, self.mean, self.std)[0]
        features = torch.as_tensor(np.asarray(normalized, dtype=np.float32))
        label = torch.tensor(int(self.labels[sample_index]), dtype=torch.long)
        if self.valid_mask is None:
            packet_mask = torch.ones(features.shape[-2] if features.ndim == 3 else 1, dtype=torch.bool)
        else:
            packet_mask = torch.as_tensor(self.valid_mask[sample_index], dtype=torch.bool)
        return features, label, packet_mask, torch.tensor(sample_index)


def _expanded_valid_mask(valid_mask: Tensor, reference: Tensor) -> Tensor:
    """Expand a (B,T) packet mask to a tensor used by a TDTCE loss."""
    if valid_mask.ndim != 2 or valid_mask.shape[0] != reference.shape[0]:
        raise ValueError("valid_mask must have shape (batch, sequence_length)")
    if reference.ndim == 3:
        if reference.shape[1] == 1 and valid_mask.shape[1] != 1:
            return valid_mask.any(dim=1).view(-1, 1, 1).expand_as(reference)
        if reference.shape[1] != valid_mask.shape[1]:
            raise ValueError("valid_mask sequence length does not match tensor")
        return valid_mask.unsqueeze(-1).expand_as(reference)
    if reference.ndim == 2:
        if reference.shape[1] % valid_mask.shape[1]:
            raise ValueError("Flattened tensor width is not divisible by valid_mask sequence length")
        return valid_mask.repeat_interleave(reference.shape[1] // valid_mask.shape[1], dim=1)
    raise ValueError(f"Unsupported loss tensor rank {reference.ndim}")


def _canonical_feature_names(names: Any) -> tuple[str, ...]:
    return tuple(str(name).upper() for name in names)


def _validate_immutable_features(
    feature_names: tuple[str, ...],
    immutable_features: tuple[str, ...],
) -> tuple[str, ...]:
    canonical_names = _canonical_feature_names(feature_names)
    canonical_immutable = _canonical_feature_names(immutable_features)
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError("TDTCE feature_names must not contain duplicates")
    unknown = sorted(set(canonical_immutable).difference(canonical_names))
    if unknown:
        raise ValueError(
            f"Unknown TDTCE immutable feature name(s): {unknown}; "
            f"available: {canonical_names}"
        )
    return canonical_immutable


def _estimate_train_mahalanobis(
    raw_features: np.ndarray, train_indices: np.ndarray, mean: np.ndarray,
    std: np.ndarray, valid_mask: np.ndarray, *, batch_size: int,
    full_covariance_max_dim: int, ridge: float, feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, str]:
    """Eq13 clean TRAIN only; masked centered residual covariance stays PSD.

    Missing packet coordinates contribute zero centered residual, not a fake
    physical observation. Ridge and diagonal high-dimensional fallback are
    declared packet adaptations; neither uses held-out or generated records.
    """
    if not len(train_indices) or batch_size < 1 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('Invalid TRAIN Gaussian population/settings')
    dimension = int(np.prod(raw_features.shape[1:]))
    counts = np.zeros(dimension, dtype=np.float64)
    totals = np.zeros(dimension, dtype=np.float64)
    def batches():
        for start in range(0, len(train_indices), batch_size):
            idx = train_indices[start:start+batch_size]
            encoded = encode_normalized32(raw_features[idx], feature_names, mean, std).reshape(len(idx), dimension)
            occupancy = np.repeat(valid_mask[idx], dimension // valid_mask.shape[1], axis=1)
            yield encoded.astype(np.float64), occupancy
    for values, valid in batches():
        totals += np.where(valid, values, 0.).sum(0)
        counts += valid.sum(0)
    center = totals / np.maximum(counts, 1.)
    full = dimension <= full_covariance_max_dim
    covariance = np.zeros((dimension, dimension) if full else dimension, dtype=np.float64)
    for values, valid in batches():
        residual = np.where(valid, values-center, 0.)
        covariance += residual.T @ residual if full else np.square(residual).sum(0)
    covariance /= max(len(train_indices)-1, 1)
    if full:
        covariance.flat[::dimension+1] += ridge
        inverse = np.linalg.pinv(covariance, hermitian=True)
    else:
        inverse = 1. / (covariance+ridge)
    if not np.isfinite(inverse).all():
        raise FloatingPointError('Nonfinite TRAIN Gaussian precision')
    return center.astype(np.float32), inverse.astype(np.float32), 'full' if full else 'diagonal'


def _checkpoint_path(template: str | Path, seed: int) -> Path:
    return Path(str(template).format(seed=int(seed))).expanduser()


def _load_classifier_checkpoint(
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], PyTorchModelWrapper]:
    """Use the explicitly trained surrogate; never load target weights offline."""
    surrogate_raw = config.get('surrogate_model')
    if not isinstance(surrogate_raw, Mapping) or not surrogate_raw.get('enabled', True):
        raise ValueError('TDTCE offline training requires a separate enabled surrogate_model')
    classifier_config = dict(surrogate_raw)
    role = 'surrogate_model'

    checkpoint_value = classifier_config.get("checkpoint")
    if not checkpoint_value:
        raise ValueError(
            f"TDTCE training requires {role}.checkpoint pointing to a trained PyTorch NIDS checkpoint. "
            "For a non-differentiable target, configure an enabled PyTorch surrogate_model."
        )
    checkpoint_path = _checkpoint_path(str(checkpoint_value), seed)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TDTCE classifier checkpoint does not exist: {checkpoint_path}")
    checkpoint = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    if checkpoint.get("model_family", "pytorch") != "pytorch" or not isinstance(
        checkpoint.get("model_state"), Mapping
    ):
        raise ValueError(
            f"TDTCE adversarial training requires a PyTorch classifier checkpoint; got {checkpoint_path}"
        )

    model_name = str(classifier_config.get("name", checkpoint.get("model_name", "")))
    checkpoint_model_name = str(checkpoint.get("model_name", model_name))
    if model_name and checkpoint_model_name and model_name != checkpoint_model_name:
        raise ValueError(
            f"Configured {role} name {model_name!r} does not match checkpoint model "
            f"{checkpoint_model_name!r}"
        )
    model_name = checkpoint_model_name or model_name
    input_shape = tuple(int(value) for value in checkpoint.get("input_shape", ()))
    if not input_shape or len(input_shape) not in {1, 2}:
        raise ValueError(
            "TDTCE supports classifier checkpoint input_shape [F] or [T, F]; "
            f"got {input_shape}"
        )
    num_classes = int(checkpoint.get("num_classes", classifier_config.get("num_classes", 0)))
    if num_classes < 2:
        raise ValueError(f"Classifier checkpoint has invalid num_classes={num_classes}")

    try:
        architecture = NIDS.get(model_name)
    except Exception as exc:
        raise ValueError(f"Classifier architecture {model_name!r} is not registered") from exc
    # The checkpoint is authoritative for architecture dimensions; the config
    # remains a fallback for checkpoints created by older workflow versions.
    model_params_raw = checkpoint.get("model_params")
    if not isinstance(model_params_raw, Mapping):
        model_params_raw = classifier_config.get("params", {})
    model_params = dict(model_params_raw)
    if len(input_shape) == 1:
        input_dim = input_shape[0]
    else:
        input_dim = input_shape[-1]
        normalized_name = model_name.strip().lower().replace("-", "_")
        if normalized_name in {"cnn", "transformer"}:
            model_params.setdefault("sequence_features", input_dim)
    classifier = architecture(input_dim=input_dim, num_classes=num_classes, **model_params)
    wrapper = PyTorchModelWrapper(classifier, device)
    wrapper.model.load_state_dict(checkpoint["model_state"], strict=True)
    wrapper.model.eval()
    for parameter in wrapper.model.parameters():
        parameter.requires_grad_(False)

    if "normalization_mean" not in checkpoint or "normalization_std" not in checkpoint:
        raise ValueError(f"Classifier checkpoint is missing normalization statistics: {checkpoint_path}")
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.clip(np.asarray(checkpoint["normalization_std"], dtype=np.float32), 1e-12, None)
    if mean.shape != std.shape:
        raise ValueError("Classifier checkpoint normalization_mean and normalization_std shapes differ")
    return classifier_config, checkpoint, wrapper


def _instantiate_training_dataset(
    config: Mapping[str, Any],
    expected_input_shape: tuple[int, ...],
) -> Any:
    dataset_config_raw = config.get("dataset")
    if not isinstance(dataset_config_raw, Mapping):
        raise ValueError("TDTCE training requires a dataset configuration block")
    dataset_config = dict(dataset_config_raw)
    dataset_name = str(dataset_config.get("name", ""))
    if not dataset_name:
        raise ValueError("dataset.name is required for TDTCE training")
    constructor = DATASETS.get(dataset_name)
    params = dict(dataset_config.get("params", {}))

    # Sequence plugins that expose a flatten parameter must mirror the trained
    # classifier's saved per-sample input rank.
    try:
        signature = inspect.signature(constructor)
        if "flatten" in signature.parameters:
            params["flatten"] = len(expected_input_shape) == 1
    except (TypeError, ValueError):
        pass
    return constructor(**params)


def _forward_targeted_loss(logits: Tensor, labels: Tensor, target_label: int) -> Tensor:
    batch_size = logits.shape[0]
    if logits.ndim == 1 or logits.shape[-1] == 1:
        target = torch.full(
            (batch_size,), float(target_label), device=logits.device, dtype=logits.dtype
        )
        return F.binary_cross_entropy_with_logits(logits.reshape(-1), target)
    target = torch.full((batch_size,), int(target_label), device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, target)


def _record_sum_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Sum valid coordinates, average nonempty records; empty is differentiable zero."""
    expanded = torch.broadcast_to(mask.bool(), values.shape)
    totals = torch.where(expanded, values, torch.zeros_like(values)).reshape(len(values), -1).sum(1)
    nonempty = expanded.reshape(len(values), -1).any(1)
    return totals.sum() / nonempty.sum().clamp_min(1)


def _vae_losses(reconstruction: Tensor, clean: Tensor, latent_mean: Tensor,
                log_variance: Tensor, predicted_noise: Tensor, noise: Tensor,
                valid_mask: Tensor) -> dict[str, Tensor]:
    # Eq8 prime/star mapping is ambiguous; clean-reference reconstruction is
    # the declared packet interpretation. Eq9's coefficient is weight_kl,
    # without silently adding the conventional extra factor 1/2.
    return {
        'rec': _record_sum_mean((reconstruction-clean).square(), _expanded_valid_mask(valid_mask, reconstruction)),
        'kl': _record_sum_mean(-(1.+log_variance-latent_mean.square()-log_variance.exp()),
                               _expanded_valid_mask(valid_mask, latent_mean)),
        'pre': _record_sum_mean((predicted_noise-noise).square(), _expanded_valid_mask(valid_mask, predicted_noise)),
    }


def _mahalanobis_loss(features: Tensor, train_mean: Tensor, covariance_inverse: Tensor,
                      covariance_mode: str, valid_mask: Tensor | None = None) -> Tensor:
    flattened = features.reshape(len(features), -1)
    difference = flattened - train_mean.reshape(1, -1)
    if valid_mask is None:
        coordinate_mask = torch.ones_like(difference, dtype=torch.bool)
    else:
        if valid_mask.ndim != 2 or len(valid_mask) != len(features) or flattened.shape[1] % valid_mask.shape[1]:
            raise ValueError('Invalid Mahalanobis occupancy')
        coordinate_mask = valid_mask.repeat_interleave(flattened.shape[1] // valid_mask.shape[1], dim=1)
        difference = torch.where(coordinate_mask, difference, torch.zeros_like(difference))
    if covariance_mode == 'full':
        contributions = (difference @ covariance_inverse) * difference
    elif covariance_mode == 'diagonal':
        contributions = difference.square() * covariance_inverse.reshape(1, -1)
    else:
        raise ValueError('Unknown covariance mode')
    return _record_sum_mean(contributions, coordinate_mask)


@dataclass
@register_attack("tdtce")
class TDTCEAttack(FeatureSpaceAttack):
    """Latent diffusion feature-space attack conditioned toward the benign class.

    The generator checkpoint is trained with ``action=train_attack`` and loaded
    lazily during ``action=evaluate_attack``.
    """

    attack_name: ClassVar[str] = "tdtce"
    query_based: ClassVar[bool] = True  # includes post-generation surrogate predictions
    requires_gradients: ClassVar[bool] = False
    hard_budget_norm = "linf"

    epsilon: float = 0.04
    source_recipe_status: str = "paper_table1_GI_packet_adaptation"
    timesteps: int = 50
    hidden_dim: int = 64
    num_blocks: int = 6
    patch_size: int = 4
    adaln_mode: str = "adaln_g"
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 256
    latent_dim: int | None = None
    num_heads: int = 4
    attention_window: int = 16
    dropout: float = 0.0
    weight_rec: float = 1.0
    weight_kl: float = 0.01
    weight_adv: float = 1.0
    weight_pre: float = 1.0
    weight_mahal: float = 0.1
    targeted: bool = True
    target_label: int = 0
    discrete_projection: bool = False
    immutable_features: tuple[str, ...] = ()
    checkpoint: str | None = None
    mahalanobis_full_cov_max_dim: int = 512
    mahalanobis_ridge: float = 1e-3
    gradient_clip_norm: float = 1.0

    _vae: TrafficVAE | None = field(default=None, init=False, repr=False, compare=False)
    _tdiT: TDiTBackbone | None = field(default=None, init=False, repr=False, compare=False)
    _generator_checkpoint: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _generator_device: torch.device | None = field(default=None, init=False, repr=False, compare=False)
    _betas: Tensor | None = field(default=None, init=False, repr=False, compare=False)
    _alphas: Tensor | None = field(default=None, init=False, repr=False, compare=False)
    _alpha_bars: Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.source_recipe_status != 'paper_table1_GI_packet_adaptation':
            raise ValueError('TDTCE source recipe identity must be the declared paper-GI packet adaptation')
        if not self.targeted:
            raise ValueError("TDTCE training implements the source benign-target objective only")
        numeric = (self.epsilon, self.learning_rate, self.weight_rec, self.weight_kl, self.weight_adv, self.weight_pre, self.weight_mahal, self.mahalanobis_ridge, self.gradient_clip_norm, self.dropout)
        if not np.isfinite(numeric).all():
            raise ValueError("TDTCE settings must be finite")
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if self.timesteps < 1 or self.hidden_dim < 1 or self.num_blocks < 1:
            raise ValueError("timesteps, hidden_dim, and num_blocks must be positive")
        if self.patch_size < 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError("patch_size, batch_size, and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if any(weight < 0 for weight in (self.weight_rec, self.weight_kl, self.weight_adv, self.weight_pre, self.weight_mahal)):
            raise ValueError("TDTCE loss weights must be non-negative")
        if self.num_heads < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.target_label < 0:
            raise ValueError("target_label must be >= 0")
        if self.mahalanobis_full_cov_max_dim < 1 or self.mahalanobis_ridge <= 0:
            raise ValueError("Invalid Mahalanobis covariance parameters")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.latent_dim is None:
            self.latent_dim = max(8, self.hidden_dim // 2)
        self.immutable_features = tuple(dict.fromkeys((*[str(name).upper() for name in self.immutable_features],
            *[name for name in ("DIR","FLG") if name in self.feature_names])))

    @staticmethod
    def _diffusion_schedule(
        timesteps: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(timesteps, bool) or int(timesteps) != timesteps or timesteps < 1:
            raise ValueError("timesteps must be a positive integer")
        betas = torch.linspace(1e-4, 2e-2, timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    @classmethod
    def train(
        cls,
        *,
        config: Mapping[str, Any],
        seed: int,
        gpu: str,
        logger: logging.Logger | None = None,
    ) -> Mapping[str, float]:
        """Fit only authorized TRAIN with an independent pretrained surrogate.

        Physical inputs use the existing canonical forward encoder and the
        checkpoint normalizer; the evaluation target is never a fallback.
        """
        device_info = configure_process_device(gpu)
        device = device_info.torch_device
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        attack_cfg_raw = config.get("attack", {})
        attack_cfg = dict(attack_cfg_raw) if isinstance(attack_cfg_raw, Mapping) else {}
        params = {
            name: attack_cfg[name]
            for name in (
                "epsilon", "source_recipe_status", "timesteps", "hidden_dim", "num_blocks", "patch_size",
                "adaln_mode", "learning_rate", "epochs", "batch_size", "latent_dim",
                "num_heads", "attention_window", "dropout", "weight_rec", "weight_kl",
                "weight_adv", "weight_pre", "weight_mahal",                 "targeted", "target_label",
                "immutable_features",
                "mahalanobis_full_cov_max_dim", "mahalanobis_ridge", "gradient_clip_norm",
            )
            if name in attack_cfg
        }
        settings = cls(**params)

        classifier_cfg, classifier_checkpoint, surrogate = _load_classifier_checkpoint(
            config, seed, device
        )
        input_shape = tuple(int(value) for value in classifier_checkpoint["input_shape"])
        dataset = _instantiate_training_dataset(config, input_shape)
        raw_features, labels = dataset.raw_arrays()
        raw_features = np.asarray(raw_features)
        if raw_features.dtype != np.float64:
            raise TypeError('TDTCE requires pre-normalization float64 physical data')
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(raw_features) != len(labels):
            raise ValueError("Dataset features and labels have different sample counts")
        if tuple(raw_features.shape[1:]) != input_shape:
            raise ValueError(
                "TDTCE dataset sample shape does not match the selected classifier checkpoint: "
                f"dataset={tuple(raw_features.shape[1:])}, checkpoint={input_shape}"
            )

        feature_names = tuple(str(name) for name in getattr(dataset, "feature_names", ()))
        canonical_feature_names = _canonical_feature_names(feature_names)
        immutable_features = _validate_immutable_features(
            canonical_feature_names, tuple(dict.fromkeys((*settings.immutable_features, *[n for n in ("DIR","FLG") if n in canonical_feature_names]))),
        )
        settings.feature_names = canonical_feature_names
        settings.immutable_features = immutable_features
        checkpoint_feature_names = tuple(
            str(name) for name in classifier_checkpoint.get("feature_names", feature_names)
        )
        if (
            canonical_feature_names
            and checkpoint_feature_names
            and canonical_feature_names != _canonical_feature_names(checkpoint_feature_names)
        ):
            raise ValueError(
                "TDTCE dataset feature names do not match the classifier checkpoint: "
                f"{feature_names} != {checkpoint_feature_names}"
            )
        mean = np.asarray(classifier_checkpoint["normalization_mean"], dtype=np.float32)
        std = np.asarray(classifier_checkpoint["normalization_std"], dtype=np.float32)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Invalid checkpoint normalizer")
        # Validate broadcasting before the training loop.
        try:
            _ = (raw_features[:1] - mean) / std
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint normalization shape mean={mean.shape}, std={std.shape} "
                f"is incompatible with dataset samples {input_shape}"
            ) from exc

        source_role = 'surrogate_model'  # _load_classifier_checkpoint admits only an authorized surrogate
        train_indices = authorized_attack_training_indices(dataset, labels, config, seed, role=source_role)
        target_label = int(settings.target_label)
        classifier_num_classes = int(classifier_checkpoint["num_classes"])
        if not 0 <= target_label < classifier_num_classes:
            raise ValueError(
                f"target_label={target_label} is outside classifier class range "
                f"[0, {classifier_num_classes - 1}]"
            )
        if np.any(labels < 0) or np.any(labels >= classifier_num_classes):
            raise ValueError("Dataset labels fall outside the classifier checkpoint's class range")
        malicious_count = int(np.sum(labels[train_indices] != target_label))
        if malicious_count == 0:
            raise ValueError(
                "TDTCE adversarial training requires at least one non-target sample in the training split"
            )

        dataset_valid_mask = getattr(dataset, "valid_mask", None)
        if dataset_valid_mask is None and raw_features.ndim == 3:
            raise ValueError("TDTCE sequence training requires the dataset valid_mask")
        valid_mask = np.asarray(dataset_valid_mask, dtype=bool) if dataset_valid_mask is not None else None
        if valid_mask is None:
            valid_mask = np.ones((len(raw_features), 1), dtype=bool)
        elif valid_mask.ndim != 2 or valid_mask.shape[0] != len(raw_features):
            raise ValueError("Dataset valid_mask must have shape (sample_count, sequence_length)")

        train_mean_np, covariance_inverse_np, covariance_mode = _estimate_train_mahalanobis(
            raw_features,
            train_indices,
            mean,
            std,
            valid_mask=valid_mask,
            batch_size=settings.batch_size,
            full_covariance_max_dim=settings.mahalanobis_full_cov_max_dim,
            ridge=settings.mahalanobis_ridge, feature_names=feature_names,
        )
        # ``feature_names`` is the canonical per-packet schema for sequence datasets.
        # Keep the classifier's externally visible shape unchanged; only generator
        # components see the restored (B,T,F) representation.
        feature_count = len(canonical_feature_names) if canonical_feature_names else int(raw_features.shape[-1])
        is_flattened_sequence = (
            raw_features.ndim == 2
            and feature_count > 0
            and raw_features.shape[-1] > feature_count
            and raw_features.shape[-1] % feature_count == 0
        )
        if is_flattened_sequence:
            feature_dim = feature_count
            sequence_length = int(raw_features.shape[-1] // feature_count)
            if valid_mask.shape[1] != sequence_length:
                raise ValueError(
                    "Flattened sequence width and dataset valid_mask disagree: "
                    f"width={raw_features.shape[-1]}, per_packet_features={feature_count}, "
                    f"inferred_T={sequence_length}, valid_mask_T={valid_mask.shape[1]}"
                )
        elif raw_features.ndim == 3:
            feature_dim = int(raw_features.shape[-1])
            sequence_length = int(raw_features.shape[1])
            if feature_count and feature_count != feature_dim:
                raise ValueError(
                    f"Per-packet feature_names has {feature_count} names but tensor has {feature_dim} features"
                )
            if valid_mask.shape[1] != sequence_length:
                raise ValueError("3-D sequence length and dataset valid_mask disagree")
        else:
            feature_dim = int(raw_features.shape[-1])
            sequence_length = 1
            if feature_count and feature_count != feature_dim:
                raise ValueError(
                    f"Tabular feature_names has {feature_count} names but tensor width is {feature_dim}"
                )
            if valid_mask.shape[1] != 1:
                raise ValueError("Tabular input requires a single-token valid_mask")

        if sequence_length > 20:
            raise ValueError("TDTCE supports the first 20 packet positions")
        training_dataset = _NormalizedArrayDataset(
            raw_features, labels, train_indices, mean, std, valid_mask, feature_names=feature_names
        )
        loader_generator = torch.Generator().manual_seed(int(seed))
        data_loader = DataLoader(
            training_dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            generator=loader_generator,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        latent_dim = int(settings.latent_dim or max(8, settings.hidden_dim // 2))
        vae = TrafficVAE(feature_dim, latent_dim, settings.hidden_dim).to(device)
        tdit = TDiTBackbone(
            latent_dim,
            settings.hidden_dim,
            settings.num_blocks,
            classifier_num_classes,
            settings.timesteps,
            patch_size=settings.patch_size,
            num_heads=settings.num_heads,
            adaln_mode=settings.adaln_mode,
            attention_window=settings.attention_window,
            dropout=settings.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            [*vae.parameters(), *tdit.parameters()],
            lr=settings.learning_rate,
        )
        betas, alphas, alpha_bars = cls._diffusion_schedule(settings.timesteps, device)
        train_mean = torch.as_tensor(train_mean_np, device=device, dtype=torch.float32)
        covariance_inverse = torch.as_tensor(
            covariance_inverse_np, device=device, dtype=torch.float32
        )

        if logger:
            logger.info(
                "tdtce.train.start seed=%d samples=%d train_samples=%d input_shape=%s "
                "latent_dim=%d feature_dim=%d sequence_length=%d flattened_sequence=%s "
                "timesteps=%d batch_size=%d covariance=%s device=%s surrogate=%s",
                seed, len(labels), len(train_indices), input_shape, latent_dim,
                feature_dim, sequence_length, is_flattened_sequence,
                settings.timesteps, settings.batch_size, covariance_mode, device,
                classifier_cfg.get("name", classifier_checkpoint.get("model_name", "unknown")),
            )

        surrogate.model.eval()
        surrogate.model.requires_grad_(False)
        vae.train()
        tdit.train()
        epoch_totals: dict[str, float] = {}
        samples_seen_total = 0
        gradient_costs = np.zeros(len(labels), dtype=np.int64)

        for epoch in range(1, settings.epochs + 1):
            epoch_totals = {
                "loss": 0.0,
                "rec": 0.0,
                "kl": 0.0,
                "adv": 0.0,
                "pre": 0.0,
                "mahal": 0.0,
            }
            samples_seen = 0
            for x_batch, y_batch, valid_mask_batch, record_indices in data_loader:
                x_batch = x_batch.to(device, dtype=torch.float32, non_blocking=True)
                y_batch = y_batch.to(device, dtype=torch.long, non_blocking=True)
                valid_mask_batch = valid_mask_batch.to(device, dtype=torch.bool, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                x_batch_generator = (
                    x_batch.reshape(len(x_batch), sequence_length, feature_dim)
                    if is_flattened_sequence
                    else x_batch
                )
                token_valid_mask = valid_mask_batch
                reconstruction, latent_mean, latent_log_variance, latent = vae(x_batch_generator)
                token_mask_3d = token_valid_mask.unsqueeze(-1).to(latent.dtype)
                # Remove padded tokens before they can enter the diffusion attention blocks.
                latent = latent * token_mask_3d
                sampled_timesteps = torch.randint(
                    1, settings.timesteps + 1, (len(x_batch),), device=device
                )
                noise = torch.randn_like(latent) * token_mask_3d
                alpha_bar_t = alpha_bars[sampled_timesteps - 1].reshape(-1, 1, 1)
                noisy_latent = ddpm_forward_noising(latent, noise, alpha_bar_t) * token_mask_3d
                predicted_noise = tdit(
                    noisy_latent, sampled_timesteps, y_batch, valid_mask=token_valid_mask
                )
                estimated_clean_latent = (
                    noisy_latent - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
                ) / torch.sqrt(alpha_bar_t).clamp_min(1e-5)
                estimated_clean_latent = estimated_clean_latent * token_mask_3d
                generated_generator = vae.decode(
                    estimated_clean_latent, output_ndim=x_batch_generator.ndim
                )

                reconstruction_mask = _expanded_valid_mask(token_valid_mask, reconstruction)
                components = _vae_losses(reconstruction, x_batch_generator, latent_mean,
                                         latent_log_variance, predicted_noise, noise, token_valid_mask)
                reconstruction_loss, kl_loss, noise_loss = (components[k] for k in ('rec','kl','pre'))

                perturbation_mask = settings.victim_packet_perturbation_mask(
                    x_batch_generator, canonical_feature_names, normalization_mean=mean,
                    normalization_std=std, valid_mask=token_valid_mask,
                ).bool()
                for j, name in enumerate(canonical_feature_names):
                    if name in immutable_features:
                        perturbation_mask[..., j] = False
                generated_generator = torch.where(perturbation_mask, generated_generator, x_batch_generator)
                generated_for_classifier = (
                    generated_generator.reshape(len(x_batch), -1)
                    if is_flattened_sequence
                    else generated_generator
                )

                malicious_mask = (y_batch != target_label) & token_valid_mask.any(1)
                if bool(malicious_mask.any()):
                    surrogate_logits = surrogate.forward_differentiable(
                        generated_for_classifier[malicious_mask],
                        valid_mask=token_valid_mask[malicious_mask],
                    )
                    gradient_costs[record_indices.numpy()[malicious_mask.cpu().numpy()]] += 1
                    adversarial_loss = _forward_targeted_loss(
                        surrogate_logits,
                        y_batch[malicious_mask],
                        target_label,
                    )
                else:
                    adversarial_loss = generated_for_classifier.sum() * 0.0

                mahalanobis_loss = _mahalanobis_loss(
                    generated_for_classifier,
                    train_mean,
                    covariance_inverse,
                    covariance_mode,
                    valid_mask=valid_mask_batch,
                )
                total_loss = (
                    settings.weight_rec * reconstruction_loss
                    + settings.weight_kl * kl_loss
                    + settings.weight_pre * noise_loss
                    + settings.weight_adv * adversarial_loss
                    + settings.weight_mahal * mahalanobis_loss
                )
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError(
                        f"Non-finite TDTCE training loss at epoch {epoch}; reduce learning_rate or loss weights"
                    )
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    [*vae.parameters(), *tdit.parameters()],
                    max_norm=settings.gradient_clip_norm,
                )
                optimizer.step()

                batch_n = len(x_batch)
                epoch_totals["loss"] += float(total_loss.detach()) * batch_n
                epoch_totals["rec"] += float(reconstruction_loss.detach()) * batch_n
                epoch_totals["kl"] += float(kl_loss.detach()) * batch_n
                epoch_totals["adv"] += float(adversarial_loss.detach()) * batch_n
                epoch_totals["pre"] += float(noise_loss.detach()) * batch_n
                epoch_totals["mahal"] += float(mahalanobis_loss.detach()) * batch_n
                samples_seen += batch_n
                del (
                    x_batch, x_batch_generator, y_batch, reconstruction, latent_mean,
                    latent_log_variance, latent, token_mask_3d,
                    noise, noisy_latent, predicted_noise, estimated_clean_latent,
                    generated_generator, generated_for_classifier, reconstruction_loss,
                    kl_loss, noise_loss, adversarial_loss, mahalanobis_loss, total_loss,
                )

            if samples_seen == 0:
                raise ValueError("TDTCE training DataLoader yielded no samples")
            samples_seen_total = samples_seen
            for name in epoch_totals:
                epoch_totals[name] /= samples_seen
            if logger:
                logger.info(
                    "tdtce.train.epoch seed=%d epoch=%d/%d loss=%.6f rec=%.6f kl=%.6f "
                    "adv=%.6f pre=%.6f mahal=%.6f",
                    seed, epoch, settings.epochs, epoch_totals["loss"],
                    epoch_totals["rec"], epoch_totals["kl"], epoch_totals["adv"],
                    epoch_totals["pre"], epoch_totals["mahal"],
                )

        output_dir_value = config.get("output_dir", "runs")
        output_dir = Path(str(output_dir_value)).expanduser()
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        configured_checkpoint = config.get("attack", {}).get("checkpoint")
        checkpoint_path = (
            _checkpoint_path(str(configured_checkpoint), int(seed))
            if configured_checkpoint else checkpoint_dir / f"tdtce_seed{int(seed)}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        from core.provenance import trained_generator_provenance
        payload: dict[str, Any] = {
            "artifact_provenance": trained_generator_provenance(
                dataset, labels, config, seed, source_role='surrogate_model',
                input_shape=input_shape, feature_names=feature_names,
            ),
            "format_version": 3,
            "gaussian_population": "authorized_clean_train",
            "training_sample_ids": [str(dataset.sample_ids[i]) for i in train_indices],
            "surrogate_gradient_forward_costs": gradient_costs[train_indices].tolist(),
            "surrogate_inference_costs": [0] * len(train_indices),
            "source_recipe_status": settings.source_recipe_status,
            "attack_name": "tdtce",
            "seed": int(seed),
            "input_shape": list(input_shape),
            "feature_names": list(feature_names),
            "feature_dim": int(feature_dim),
            "sequence_length": int(sequence_length),
            "is_flattened_sequence": bool(is_flattened_sequence),
            "immutable_features": list(immutable_features),
            "num_classes": classifier_num_classes,
            "target_label": target_label,
            "normalization_mean": mean.astype(np.float32),
            "normalization_std": std.astype(np.float32),
            "vae_state_dict": {
                key: value.detach().cpu() for key, value in vae.state_dict().items()
            },
            "tdit_state_dict": {
                key: value.detach().cpu() for key, value in tdit.state_dict().items()
            },
            "train_mean": train_mean_np,
            "covariance_inverse": covariance_inverse_np,
            "covariance_mode": covariance_mode,
            "model_config": {
                "feature_dim": feature_dim,
                "source_recipe_status": settings.source_recipe_status,
                "latent_dim": latent_dim,
                "hidden_dim": settings.hidden_dim,
                "num_blocks": settings.num_blocks,
                "patch_size": settings.patch_size,
                "num_heads": settings.num_heads,
                "attention_window": settings.attention_window,
                "dropout": settings.dropout,
                "timesteps": settings.timesteps,
                "adaln_mode": settings.adaln_mode,
            },
            "training_config": {
                "learning_rate": settings.learning_rate,
                "epochs": settings.epochs,
                "batch_size": settings.batch_size,
                "weight_rec": settings.weight_rec,
                "weight_kl": settings.weight_kl,
                "weight_adv": settings.weight_adv,
                "weight_pre": settings.weight_pre,
                "weight_mahal": settings.weight_mahal,
                "targeted": settings.targeted,
                "immutable_features": list(immutable_features),
            },
            "last_epoch_metrics": dict(epoch_totals),
        }
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        if logger:
            logger.info(
                "tdtce.train.done seed=%d checkpoint=%s epochs=%d samples_per_epoch=%d",
                seed, checkpoint_path, settings.epochs, samples_seen_total,
            )
        return {
            "surrogate_training_gradient_forward_records": float(gradient_costs.sum()),
            "surrogate_training_inference_records": 0.,
            "tdtce_train_loss": float(epoch_totals["loss"]),
            "tdtce_reconstruction_loss": float(epoch_totals["rec"]),
            "tdtce_kl_loss": float(epoch_totals["kl"]),
            "tdtce_adversarial_loss": float(epoch_totals["adv"]),
            "tdtce_noise_loss": float(epoch_totals["pre"]),
            "tdtce_mahalanobis_loss": float(epoch_totals["mahal"]),
            "tdtce_train_samples": float(len(train_indices)),
            "tdtce_epochs": float(settings.epochs),
        }

    def _resolve_checkpoint_path(self, seed: int) -> Path:
        template = self.checkpoint or "runs/checkpoints/tdtce_seed{seed}.pt"
        return _checkpoint_path(template, seed)

    def _load_generator(self, device: torch.device, seed: int) -> None:
        if self._generator_checkpoint is not None:
            if self._generator_device != device:
                assert self._vae is not None and self._tdiT is not None
                self._vae.to(device)
                self._tdiT.to(device)
                self._generator_device = device
                self._betas, self._alphas, self._alpha_bars = self._diffusion_schedule(
                    int(self._generator_checkpoint["model_config"]["timesteps"]), device
                )
            return

        path = self._resolve_checkpoint_path(seed)
        if not path.is_file():
            raise FileNotFoundError(
                f"TDTCE generator checkpoint missing: {path}; train_attack and evaluate_attack must use the same stable attack.checkpoint"
            )
        checkpoint = dict(torch.load(path, map_location=device, weights_only=False))
        if checkpoint.get("attack_name") != "tdtce":
            raise ValueError(f"Checkpoint {path} is not a TDTCE generator checkpoint")
        if (int(checkpoint.get('format_version', -1)) != 3
                or checkpoint.get('source_recipe_status') != self.source_recipe_status):
            raise ValueError('TDTCE checkpoint predates the versioned GI packet algorithm or has different recipe identity; retrain')
        from core.provenance import verify_generator_artifact_provenance
        verify_generator_artifact_provenance(checkpoint, getattr(self, 'expected_provenance', None))
        input_shape = tuple(int(value) for value in checkpoint.get("input_shape", ()))
        model_cfg = dict(checkpoint.get("model_config", {}))
        feature_dim = int(checkpoint.get("feature_dim", model_cfg.get("feature_dim", 0)))
        if not input_shape or feature_dim < 1:
            raise ValueError(f"TDTCE checkpoint has invalid architecture metadata: {path}")
        num_classes = int(checkpoint["num_classes"])
        vae = TrafficVAE(
            feature_dim=feature_dim,
            latent_dim=int(model_cfg["latent_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
        ).to(device)
        tdit = TDiTBackbone(
            latent_dim=int(model_cfg["latent_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_blocks=int(model_cfg["num_blocks"]),
            num_classes=num_classes,
            timesteps=int(model_cfg["timesteps"]),
            patch_size=int(model_cfg["patch_size"]),
            num_heads=int(model_cfg["num_heads"]),
            adaln_mode=str(model_cfg["adaln_mode"]),
            attention_window=int(model_cfg.get("attention_window", 16)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        ).to(device)
        vae.load_state_dict(checkpoint["vae_state_dict"], strict=True)
        tdit.load_state_dict(checkpoint["tdit_state_dict"], strict=True)
        vae.eval()
        tdit.eval()
        self._generator_checkpoint = checkpoint
        self._vae = vae
        self._tdiT = tdit
        self._generator_device = device
        self._betas, self._alphas, self._alpha_bars = self._diffusion_schedule(
            int(model_cfg["timesteps"]), device
        )

    def generate(self, x: Any, y: Any, **kwargs: Any) -> AttackResult:
        """Generate bounded adversarial traffic with reverse latent diffusion.

        Reverse diffusion and decoding run entirely under ``torch.no_grad``.
        A per-call generator makes sampling reproducible, while the cache avoids
        loading model weights for every EvaluationEngine batch. Inputs remain on
        CPU and only one bounded slice is transferred to the generator device.
        """
        bound_model = self._require_model()
        x_array = (
            x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)
        ).astype(np.float32, copy=False)
        y_array = (
            y.detach().cpu().numpy() if isinstance(y, Tensor) else np.asarray(y)
        ).astype(np.int64, copy=False).reshape(-1)
        if x_array.ndim not in {2, 3}:
            raise ValueError(f"TDTCE expects input shape (B,F) or (B,T,F), got {x_array.shape}")
        if len(x_array) != len(y_array):
            raise ValueError("TDTCE inputs and labels must have the same batch size")
        if len(x_array) == 0:
            return AttackResult(
                adversarial=x_array.copy(),
                success_mask=np.empty((0,), dtype=bool),
                queries=np.empty((0,), dtype=np.int64),
                feature_names=self.feature_names or None,
                metadata={"attack": "tdtce", "empty_batch": True},
            )

        device = (
            bound_model.device
            if isinstance(bound_model, PyTorchModelWrapper)
            else torch.device("cpu")
        )
        seed_value = int(kwargs.get("seed", 0))
        self._load_generator(device, seed_value)
        assert self._generator_checkpoint is not None
        assert self._vae is not None and self._tdiT is not None
        assert self._betas is not None and self._alphas is not None and self._alpha_bars is not None

        expected_shape = tuple(int(value) for value in self._generator_checkpoint["input_shape"])
        if tuple(x_array.shape[1:]) != expected_shape:
            raise ValueError(
                "TDTCE input shape differs from its trained checkpoint: "
                f"runtime={tuple(x_array.shape[1:])}, trained={expected_shape}"
            )
        checkpoint_names = tuple(
            str(name) for name in self._generator_checkpoint.get("feature_names", ())
        )
        runtime_names = _canonical_feature_names(self.feature_names)
        canonical_checkpoint_names = _canonical_feature_names(checkpoint_names)
        if runtime_names and canonical_checkpoint_names and runtime_names != canonical_checkpoint_names:
            raise ValueError(
                "TDTCE feature schema differs from its checkpoint: "
                f"runtime={tuple(self.feature_names)}, trained={checkpoint_names}"
            )
        checkpoint_immutable = _validate_immutable_features(
            canonical_checkpoint_names or runtime_names,
            _canonical_feature_names(self._generator_checkpoint['immutable_features']),
        )
        if checkpoint_immutable != _canonical_feature_names(self.immutable_features):
            raise ValueError('TDTCE immutable_features differ from trained checkpoint')

        configured_target = int(kwargs.get("target_label", self.target_label))
        num_classes = int(self._generator_checkpoint["num_classes"])
        if not 0 <= configured_target < num_classes:
            raise ValueError(
                f"target_label={configured_target} is outside TDTCE class range [0, {num_classes - 1}]"
            )
        if np.any(y_array < 0) or np.any(y_array >= num_classes):
            raise ValueError("TDTCE labels fall outside the trained class-conditioning range")

        model_cfg = dict(self._generator_checkpoint["model_config"])
        timesteps = int(model_cfg["timesteps"])
        feature_dim = int(self._generator_checkpoint["feature_dim"])
        is_flattened_sequence = bool(self._generator_checkpoint["is_flattened_sequence"])
        sequence_length = int(self._generator_checkpoint["sequence_length"])
        if feature_dim < 1 or sequence_length < 1:
            raise ValueError("TDTCE checkpoint has invalid feature/sequence metadata")
        if is_flattened_sequence:
            if len(expected_shape) != 1 or expected_shape[0] != sequence_length * feature_dim:
                raise ValueError("TDTCE flattened sequence metadata conflicts with input_shape")
            if self.feature_names and len(self.feature_names) != feature_dim:
                raise ValueError(
                    "TDTCE flattened sequence feature_names must describe exactly one packet: "
                    f"expected {feature_dim}, got {len(self.feature_names)}"
                )
        valid_mask_array = kwargs.get("valid_mask")
        expected_packets = (
            sequence_length if is_flattened_sequence
            else (x_array.shape[1] if x_array.ndim == 3 else 1)
        )
        if expected_packets > 20:
            raise ValueError("TDTCE supports the first 20 packet positions")
        if valid_mask_array is None and (expected_packets > 1 or x_array.ndim == 3):
            raise ValueError("TDTCE sequence generation requires valid_mask")
        if valid_mask_array is None:
            valid_mask_array = np.ones((len(x_array), expected_packets), dtype=bool)
        valid_mask_array = np.asarray(valid_mask_array, dtype=bool)
        if valid_mask_array.shape != (len(x_array), expected_packets):
            raise ValueError(
                f"valid_mask shape {valid_mask_array.shape} does not match "
                f"expected {(len(x_array), expected_packets)}"
            )
        torch_generator = torch.Generator(device=device)
        torch_generator.manual_seed(seed_value)
        adversarial = np.empty_like(x_array, dtype=np.float32)

        for start in range(0, len(x_array), self.batch_size):
            stop = min(start + self.batch_size, len(x_array))
            x_batch = torch.as_tensor(x_array[start:stop], device=device, dtype=torch.float32)
            labels_batch = torch.as_tensor(y_array[start:stop], device=device, dtype=torch.long)
            valid_batch = torch.as_tensor(
                valid_mask_array[start:stop],
                device=device,
                dtype=torch.bool,
            )
            # Train and inference use the original malicious class condition;
            # benign is the optimization target, not a substituted class prior.
            condition_labels = labels_batch
            x_batch_generator = (
                x_batch.reshape(len(x_batch), sequence_length, feature_dim)
                if is_flattened_sequence
                else x_batch
            )
            token_mask_3d = valid_batch.unsqueeze(-1).to(dtype=x_batch.dtype)
            with torch.no_grad():
                latent_mean, _ = self._vae.encode(x_batch_generator)
                latent_mean = latent_mean * token_mask_3d
                initial_noise = torch.randn(
                    latent_mean.shape,
                    device=device,
                    dtype=latent_mean.dtype,
                    generator=torch_generator,
                ) * token_mask_3d
                alpha_bar_terminal = self._alpha_bars[-1]
                latent = (
                    torch.sqrt(alpha_bar_terminal) * latent_mean
                    + torch.sqrt(1.0 - alpha_bar_terminal) * initial_noise
                ) * token_mask_3d

                for timestep in range(timesteps, 0, -1):
                    timestep_batch = torch.full(
                        (len(x_batch),), timestep, dtype=torch.long, device=device
                    )
                    predicted_noise = self._tdiT(
                        latent, timestep_batch, condition_labels, valid_mask=valid_batch
                    )
                    beta_t = self._betas[timestep - 1]
                    alpha_t = self._alphas[timestep - 1]
                    alpha_bar_t = self._alpha_bars[timestep - 1]
                    if timestep > 1:
                        alpha_bar_previous = self._alpha_bars[timestep - 2]
                        posterior_sigma = torch.sqrt((
                            beta_t * (1.0 - alpha_bar_previous)
                            / (1.0 - alpha_bar_t).clamp_min(1e-8)
                        ).clamp_min(0.0))
                        reverse_noise = torch.randn(
                            latent.shape, device=device, dtype=latent.dtype,
                            generator=torch_generator,
                        )
                    else:
                        posterior_sigma = torch.zeros_like(beta_t)
                        reverse_noise = None
                    latent = ddpm_reverse_step(
                        latent, predicted_noise, alpha=alpha_t, beta=beta_t,
                        alpha_bar=alpha_bar_t, posterior_sigma=posterior_sigma,
                        posterior_noise=reverse_noise,
                    ).detach() * token_mask_3d
                    del predicted_noise, timestep_batch

                decoded_generator = self._vae.decode(
                    latent, output_ndim=x_batch_generator.ndim
                )
                # epsilon bounds the per-feature perturbation in normalized feature space.
                decoded_generator = torch.maximum(
                    torch.minimum(
                        decoded_generator, x_batch_generator + self.epsilon
                    ),
                    x_batch_generator - self.epsilon,
                )
                allowed = self.victim_packet_perturbation_mask(
                    x_batch_generator, self.feature_names,
                    normalization_mean=kwargs.get('normalization_mean'),
                    normalization_std=kwargs.get('normalization_std'), valid_mask=valid_batch,
                ).bool()
                for j, name in enumerate(self.feature_names):
                    if name in set(self.immutable_features) | {'DIR','FLG','FEAT_PAD_MASK'}:
                        allowed[..., j] = False
                decoded_generator = torch.where(allowed, decoded_generator, x_batch_generator)
                decoded_output = (
                    decoded_generator.reshape(len(x_batch), -1)
                    if is_flattened_sequence
                    else decoded_generator
                )
                adversarial[start:stop] = decoded_output.detach().cpu().numpy().astype(
                    np.float32, copy=False
                )
                del (
                    x_batch, x_batch_generator, labels_batch, valid_batch, condition_labels,
                    token_mask_3d, latent_mean, initial_noise, latent, decoded_generator,
                    decoded_output,
                )

        # Keep classifier inference bounded as well when generate() is called
        # directly with more samples than the evaluation engine's batch size.
        prediction_batches: list[np.ndarray] = []
        for start in range(0, len(adversarial), self.batch_size):
            predictions_batch = np.asarray(
                bound_model.predict(adversarial[start : start + self.batch_size],
                                    valid_mask=valid_mask_array[start:start+self.batch_size]),
                dtype=np.int64,
            ).reshape(-1)
            prediction_batches.append(predictions_batch)
        predictions = (
            np.concatenate(prediction_batches)
            if prediction_batches
            else np.empty((0,), dtype=np.int64)
        )
        if len(predictions) != len(y_array):
            raise RuntimeError("Bound classifier returned a different number of predictions than TDTCE inputs")
        success = predictions == configured_target if self.targeted else predictions != y_array
        return AttackResult(
            adversarial=adversarial,
            success_mask=success.astype(bool, copy=False),
            queries=np.ones(len(adversarial), dtype=np.int64),
            feature_names=self.feature_names or None,
            metadata={
                "attack": "tdtce",
                "timesteps": timesteps,
                "epsilon": float(self.epsilon),
                "targeted": bool(self.targeted),
                "target_label": configured_target,
                "checkpoint": str(self._resolve_checkpoint_path(seed_value)),
                "diffusion_sampling": "ddpm_reverse",
            },
        )
