from __future__ import annotations

import copy
import csv
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Literal, overload

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader

from models.wrapper import PyTorchModelWrapper

DefenseMode = Literal["none", "pgd", "trades"]
PerturbationMaskFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TrainingBatch = (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


def _resolve_perturbation_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    perturbation_mask_fn: PerturbationMaskFn | None,
) -> torch.Tensor | None:
    """Return a detached binary mask on the model device.

    A zero means that a coordinate is outside the attacker's feasible edit
    set. Rejecting fractional/shape-incompatible masks prevents an ostensibly
    constrained inner maximization from changing protected packet coordinates.
    The trainer deliberately does not import any concrete attack implementation.
    """
    if perturbation_mask_fn is None:
        return None
    with torch.no_grad():
        proposed = perturbation_mask_fn(x, y)
        if not isinstance(proposed, torch.Tensor):
            raise TypeError("perturbation_mask_fn must return a torch.Tensor")
        if proposed.shape != x.shape:
            raise ValueError(
                f"Perturbation mask shape {tuple(proposed.shape)} differs from input {tuple(x.shape)}"
            )
        if not torch.all((proposed == 0) | (proposed == 1)).item():
            raise ValueError("Perturbation mask must contain only binary 0/1 values")
        return proposed.detach().to(device=x.device, dtype=x.dtype)


def _batch_packet_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    base_mask_fn: PerturbationMaskFn | None,
    packet_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Intersect the endpoint edit set with true (unpadded) packet positions.

    ``packet_valid_mask`` travels with each shuffled training example as a
    third dataset tensor. Deriving it from the numeric padding sentinel would
    confuse zero-valued real packets with padding and violate the threat model.
    """
    if x.ndim not in (2, 3):
        raise ValueError("Packet masking requires (B,T*F) or (B,T,F) input")
    valid = torch.as_tensor(packet_valid_mask, device=x.device, dtype=torch.bool)
    if valid.ndim != 2 or valid.shape[0] != x.shape[0]:
        raise ValueError("packet_valid_mask must have shape (B,T)")
    t = valid.shape[1]
    if t < 1:
        raise ValueError("packet_valid_mask must have at least one packet position")
    if x.ndim == 3:
        if t != x.shape[1]:
            raise ValueError("packet_valid_mask packet count differs from sequence length")
        f = x.shape[2]
    else:
        if x.shape[1] % t:
            raise ValueError("Flattened feature width is not divisible by packet count")
        f = x.shape[1] // t
    if base_mask_fn is None:
        full_mask = torch.ones_like(x)
    else:
        full_mask = _resolve_perturbation_mask(x, y, base_mask_fn)
        assert full_mask is not None
    with torch.no_grad():
        return full_mask * valid.unsqueeze(-1).expand(-1, -1, f).reshape_as(x)




@dataclass(frozen=True)
class TrainingSummary:
    epochs: int
    train_loss: float
    validation_loss: float | None
    best_epoch: int
    history: tuple[dict[str, float | str], ...] = field(default_factory=tuple)
    defense_mode: str = "none"


def _classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross entropy for multiclass/two-logit models and BCE for one-logit binary models."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        return nn.functional.binary_cross_entropy_with_logits(logits.reshape(-1), y.float().reshape(-1))
    return nn.functional.cross_entropy(logits, y.long())


def _class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return categorical probabilities; represent a one-logit model as two classes."""
    if logits.ndim == 1 or logits.shape[-1] == 1:
        positive = torch.sigmoid(logits.reshape(-1))
        return torch.stack((1.0 - positive, positive), dim=-1)
    return torch.softmax(logits, dim=-1)


def _kl_from_logits(clean_logits: torch.Tensor, other_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_clean || p_other), with probability handling for one-logit binary models."""
    clean_prob = _class_probabilities(clean_logits).detach().clamp_min(1e-12)
    other_prob = _class_probabilities(other_logits).clamp_min(1e-12)
    return nn.functional.kl_div(other_prob.log(), clean_prob, reduction="batchmean")


def _pgd_delta(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
) -> torch.Tensor:
    """Generate Linf PGD perturbations inside the endpoint attacker's edit set.

    Mask random starts and every ascent/projection step, so victim-to-attacker
    packets and immutable metadata are never modified by the inner maximizer.
    BatchNorm running statistics remain unchanged during adversarial search.
    """
    if epsilon < 0 or alpha <= 0 or steps < 1:
        raise ValueError("PGD requires epsilon >= 0, alpha > 0, and steps >= 1")
    was_training = model.training
    edit_mask = _resolve_perturbation_mask(x, y, perturbation_mask_fn)
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    if edit_mask is not None:
        delta.mul_(edit_mask)
    try:
        model.eval()
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            logits = model(x + delta)
            loss = _classification_loss(logits, y)
            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
            with torch.no_grad():
                delta = (delta + alpha * grad.sign()).clamp(-epsilon, epsilon)
                if edit_mask is not None:
                    delta.mul_(edit_mask)
    finally:
        model.train(was_training)
    return delta.detach()


def _write_training_curve_atomic(output_dir: Path | None, history: list[dict[str, float | str]]) -> None:
    """Atomically publish the complete per-epoch history for live monitoring."""
    if output_dir is None or not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "training_curve.csv"
    fieldnames = (
        "epoch", "trn_loss", "trn_acc", "trn_f1", "val_loss", "val_acc", "val_f1",
        "learning_rate", "defense_mode",
    )
    with NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=output_dir,
        prefix="training_curve_", suffix=".tmp", delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": int(row["epoch"]),
                "trn_loss": f"{row['trn_loss']:.6f}",
                "trn_acc": f"{row['trn_acc']:.6f}",
                "trn_f1": f"{row['trn_f1']:.6f}",
                "val_loss": "" if not np.isfinite(row["val_loss"]) else f"{row['val_loss']:.6f}",
                "val_acc": "" if not np.isfinite(row["val_acc"]) else f"{row['val_acc']:.6f}",
                "val_f1": "" if not np.isfinite(row["val_f1"]) else f"{row['val_f1']:.6f}",
                "learning_rate": f"{row['learning_rate']:.3e}",
                "defense_mode": row["defense_mode"],
            })
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(destination)


def _resolve_defense_mode(defense_mode: str | None, adversarial_training: bool) -> DefenseMode:
    """Map the legacy boolean to PGD while accepting the explicit defense_mode API."""
    if defense_mode is None or not str(defense_mode).strip():
        mode = "pgd" if adversarial_training else "none"
    else:
        mode = str(defense_mode).strip().lower()
        # Backward compatibility: an old config with adversarial_training=true still
        # enables PGD even if it inherits the new default defense_mode='none'.
        if mode == "none" and adversarial_training:
            mode = "pgd"
    if mode not in {"none", "pgd", "trades"}:
        raise ValueError(f"Unsupported defense_mode={mode!r}; expected 'none', 'pgd', or 'trades'")
    return mode  # type: ignore[return-value]


def _trades_loss_and_logits(
    wrapper: PyTorchModelWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    beta: float,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute TRADES objective and the natural logits used for training metrics.

    Algorithmic reference: Zhang et al., *Theoretically Principled Trade-off between
    Robustness and Accuracy*, ICML 2019. Reference implementation:
    https://github.com/yaodongyu/TRADES

    This is a framework-specific reimplementation, not copied source. The inner
    maximization runs in eval mode, avoiding BatchNorm running-statistic drift. The
    outer loss is evaluated in train mode on a combined clean/adversarial batch so
    BatchNorm statistics are updated only once per optimizer step.
    """
    if epsilon < 0 or step_size <= 0 or steps < 1 or beta < 0:
        raise ValueError("TRADES requires epsilon >= 0, step_size > 0, steps >= 1, and beta >= 0")

    model = wrapper.model
    x = x.to(wrapper.device, dtype=torch.float32)
    y = y.to(wrapper.device, dtype=torch.long)
    was_training = model.training
    edit_mask = _resolve_perturbation_mask(x, y, perturbation_mask_fn)
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    if edit_mask is not None:
        delta.mul_(edit_mask)

    try:
        model.eval()
        with torch.no_grad():
            clean_reference_logits = model(x)
            clean_reference_prob = _class_probabilities(clean_reference_logits).detach().clamp_min(1e-12)

        # Maximize KL(clean || adversarial) under the Linf budget.
        for _ in range(steps):
            delta = delta.detach().requires_grad_(True)
            adv_logits = model(x + delta)
            adv_prob = _class_probabilities(adv_logits).clamp_min(1e-12)
            inner_kl = nn.functional.kl_div(adv_prob.log(), clean_reference_prob, reduction="batchmean")
            gradient = torch.autograd.grad(inner_kl, delta, only_inputs=True)[0]
            with torch.no_grad():
                delta = (delta + step_size * gradient.sign()).clamp(-epsilon, epsilon)
                if edit_mask is not None:
                    delta.mul_(edit_mask)
    finally:
        # Restore the exact mode that was active on entry even if inner optimization fails.
        model.train(was_training)

    adversarial = (x + delta.detach()).detach()
    # A single forward over both views limits BatchNorm updates to one outer pass.
    joint_logits = model(torch.cat((x, adversarial), dim=0))
    natural_logits, adversarial_logits = joint_logits.split(x.shape[0], dim=0)
    natural_loss = _classification_loss(natural_logits, y)
    robust_loss = _kl_from_logits(natural_logits, adversarial_logits)
    return natural_loss + beta * robust_loss, natural_logits


@overload
def trades_loss(
    wrapper: PyTorchModelWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.04,
    step_size: float = 0.01,
    steps: int = 10,
    beta: float = 6.0,
    *,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
    return_logits: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def trades_loss(
    wrapper: PyTorchModelWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.04,
    step_size: float = 0.01,
    steps: int = 10,
    beta: float = 6.0,
    *,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
    return_logits: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


def trades_loss(
    wrapper: PyTorchModelWrapper,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.04,
    step_size: float = 0.01,
    steps: int = 10,
    beta: float = 6.0,
    *,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """TRADES loss: natural classification loss + beta * KL regularizer.

    See also the official research implementation at https://github.com/yaodongyu/TRADES.
    Defaults follow the requested NIDS-safe starting point (epsilon=0.04, K=10,
    alpha=epsilon/4, beta=6); tune and report these values for each dataset.
    """
    loss, natural_logits = _trades_loss_and_logits(
        wrapper, x, y, epsilon=epsilon, step_size=step_size, steps=steps, beta=beta,
        perturbation_mask_fn=perturbation_mask_fn,
    )
    return (loss, natural_logits) if return_logits else loss


def _resolve_early_stopping_monitor(monitor: str) -> tuple[str, bool]:
    """Resolve a validation metric name and its optimization direction.

    Returns:
        A pair containing the canonical training-history key and whether
        smaller values are better.

    Supported metrics:
        val_loss, val_acc, val_f1, val_mcc, including common aliases.
    """
    normalized = monitor.strip().lower().replace("-", "_").replace(" ", "_")

    aliases: dict[str, str] = {
        "loss": "val_loss",
        "val_loss": "val_loss",
        "validation_loss": "val_loss",
        "acc": "val_acc",
        "accuracy": "val_acc",
        "val_acc": "val_acc",
        "val_accuracy": "val_acc",
        "validation_acc": "val_acc",
        "validation_accuracy": "val_acc",
        "f1": "val_f1",
        "macro_f1": "val_f1",
        "val_f1": "val_f1",
        "val_macro_f1": "val_f1",
        "validation_f1": "val_f1",
        "validation_macro_f1": "val_f1",
        "mcc": "val_mcc",
        "val_mcc": "val_mcc",
        "validation_mcc": "val_mcc",
    }

    metric_key = aliases.get(normalized)
    if metric_key is None:
        supported = "val_loss, val_acc, val_f1, val_mcc"
        raise ValueError(
            f"Unsupported early_stopping_monitor={monitor!r}. "
            f"Supported validation metrics: {supported}."
        )

    is_minimize = "loss" in normalized
    return metric_key, is_minimize


def train_classifier(
    wrapper: PyTorchModelWrapper,
    train_loader: DataLoader[TrainingBatch],
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    adversarial_training: bool = False,
    defense_mode: str | None = None,
    pgd_epsilon: float = 0.04,
    pgd_alpha: float | None = None,
    pgd_steps: int = 10,
    trades_beta: float = 6.0,
    early_stopping_patience: int | None = 0,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val_f1",
    output_dir: str | Path | None = None,
    seed: int | None = None,
    logger: logging.Logger | None = None,
    perturbation_mask_fn: PerturbationMaskFn | None = None,
) -> TrainingSummary:
    """Train a NIDS using natural, PGD adversarial, or TRADES training.

    Optional early stopping monitors a validation metric. When enabled, the
    best model state is deep-copied on improvement and restored before the
    function returns, including when all configured epochs are completed.

    The legacy ``adversarial_training=True`` interface remains supported.
    Validation is performed without optimizer updates, and robust inner loops
    temporarily use evaluation mode to avoid contaminating BatchNorm statistics.

    Args:
        early_stopping_patience: Number of consecutive non-improving epochs
            tolerated before stopping. None or any value <= 0 disables
            early stopping.
        early_stopping_min_delta: Minimum improvement required to reset
            the patience counter. Must be finite and non-negative when enabled.
        early_stopping_monitor: Validation metric to monitor. Supported names
            include val_loss, val_acc, val_f1, and val_mcc.
        perturbation_mask_fn: Optional attack-agnostic callback returning a
            binary per-feature edit mask for ``(x, y)`` on each robust batch.
            A third training-batch tensor, when present, supplies true packet
            validity (B,T), independent of shuffle order. Both restrictions are
            intersected before random starts and at every PGD/TRADES step.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    epsilon = float(pgd_epsilon)
    alpha = float(pgd_alpha) if pgd_alpha is not None else epsilon / 4.0
    mode = _resolve_defense_mode(defense_mode, adversarial_training)

    if mode in {"pgd", "trades"} and (epsilon < 0 or alpha <= 0 or pgd_steps < 1):
        raise ValueError(
            "PGD/TRADES require epsilon >= 0, alpha > 0, and pgd_steps >= 1"
        )
    if trades_beta < 0:
        raise ValueError("trades_beta must be >= 0")

    # Normalize the optional patience value without requiring it to be present
    # in existing Hydra configurations.
    patience = 0 if early_stopping_patience is None else int(early_stopping_patience)
    early_stopping_enabled = patience > 0

    # Required tracking state. Model snapshots are only created when the
    # feature is enabled, so normal training has no deep-copy overhead.
    best_metric_val = float("inf") if "loss" in early_stopping_monitor.lower() else -float("inf")
    epochs_no_improve = 0
    best_state_dict = None
    is_minimize = "loss" in early_stopping_monitor.lower()
    monitor_key: str | None = None
    min_delta = 0.0

    if early_stopping_enabled:
        if validation_loader is None:
            raise ValueError(
                "Early stopping requires a validation_loader."
            )

        min_delta = float(early_stopping_min_delta)
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError(
                "early_stopping_min_delta must be finite and >= 0."
            )

        monitor_key, is_minimize = _resolve_early_stopping_monitor(early_stopping_monitor)

    if logger:
        logger.info(
            "training.start framework=pytorch seed=%s epochs=%d lr=%.3g "
            "weight_decay=%.3g defense=%s epsilon=%.4g alpha=%.4g "
            "steps=%d trades_beta=%.3g batches=%d",
            seed,
            epochs,
            lr,
            weight_decay,
            mode,
            epsilon,
            alpha,
            pgd_steps,
            trades_beta,
            len(train_loader),
        )
        if early_stopping_enabled:
            logger.info(
                "training.early_stopping.start seed=%s monitor=%s "
                "direction=%s patience=%d min_delta=%.6g",
                seed,
                monitor_key,
                "minimize" if is_minimize else "maximize",
                patience,
                min_delta,
            )

    model = wrapper.model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = float("inf")
    best_epoch = 1
    train_loss = float("nan")
    val_loss: float | None = None

    history: list[dict[str, float | str]] = []
    curve_dir = None if output_dir is None else Path(output_dir)

    if curve_dir is not None:
        curve_dir.mkdir(parents=True, exist_ok=True)
        (curve_dir / "training_curve.csv").unlink(missing_ok=True)

    epochs_completed = 0

    for epoch in range(1, epochs + 1):
        epochs_completed = epoch
        model.train()

        running_loss = 0.0
        count = 0
        trn_preds_list: list[np.ndarray] = []
        trn_labels_list: list[np.ndarray] = []

        for train_batch in train_loader:
            if not isinstance(train_batch, (tuple, list)) or len(train_batch) not in (2, 3):
                raise ValueError("Training batches must be (x,y) or (x,y,packet_valid_mask)")
            x, y = train_batch[:2]
            batch_valid_mask = train_batch[2] if len(train_batch) == 3 else None
            x = x.to(
                wrapper.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            y = y.to(
                wrapper.device,
                dtype=torch.long,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            batch_mask_fn = perturbation_mask_fn
            if mode in {"pgd", "trades"} and batch_valid_mask is not None:
                # A shuffled loader must transport padding metadata alongside
                # each example; callbacks alone cannot recover its sample ID.
                batch_mask_fn = partial(
                    _batch_packet_mask,
                    base_mask_fn=perturbation_mask_fn,
                    packet_valid_mask=batch_valid_mask,
                )

            if mode == "pgd":
                delta = _pgd_delta(
                    model,
                    x,
                    y,
                    epsilon,
                    alpha,
                    pgd_steps,
                    perturbation_mask_fn=batch_mask_fn,
                )
                logits = model(x + delta)
                loss = _classification_loss(logits, y)

            elif mode == "trades":
                trades_result = trades_loss(
                    wrapper,
                    x,
                    y,
                    epsilon=epsilon,
                    step_size=alpha,
                    steps=pgd_steps,
                    beta=trades_beta,
                    perturbation_mask_fn=batch_mask_fn,
                    return_logits=True,
                )
                loss, logits = trades_result

            else:
                logits = model(x)
                loss = _classification_loss(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach()) * len(x)
            count += len(x)

            if logits.ndim == 1 or logits.shape[-1] == 1:
                preds = (torch.sigmoid(logits) >= 0.5).long().view(-1)
            else:
                preds = logits.argmax(dim=-1)

            trn_preds_list.append(preds.detach().cpu().numpy())
            trn_labels_list.append(y.detach().cpu().numpy())

        if count == 0:
            raise ValueError("Training DataLoader yielded no samples")

        train_loss = running_loss / count
        train_labels_all = np.concatenate(trn_labels_list)
        train_preds_all = np.concatenate(trn_preds_list)

        trn_acc = float(accuracy_score(train_labels_all, train_preds_all))
        trn_f1 = float(
            f1_score(
                train_labels_all,
                train_preds_all,
                average="macro",
                zero_division=0,
            )
        )

        # Calculate the existing validation metrics, plus MCC for users who
        # explicitly choose it as the early-stopping monitor.
        val_loss = None
        val_acc = float("nan")
        val_f1 = float("nan")
        val_mcc = float("nan")

        if validation_loader is not None:
            model.eval()
            total_val_loss = 0.0
            n_val = 0

            val_preds_list: list[np.ndarray] = []
            val_labels_list: list[np.ndarray] = []

            with torch.no_grad():
                for x, y in validation_loader:
                    x = x.to(
                        wrapper.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    y = y.to(
                        wrapper.device,
                        dtype=torch.long,
                        non_blocking=True,
                    )

                    logits = model(x)
                    loss = _classification_loss(logits, y)

                    total_val_loss += float(loss) * len(x)
                    n_val += len(x)

                    if logits.ndim == 1 or logits.shape[-1] == 1:
                        preds = (
                            (torch.sigmoid(logits) >= 0.5)
                            .long()
                            .view(-1)
                        )
                    else:
                        preds = logits.argmax(dim=-1)

                    val_preds_list.append(preds.detach().cpu().numpy())
                    val_labels_list.append(y.detach().cpu().numpy())

            if n_val:
                val_loss = total_val_loss / n_val
                val_labels_all = np.concatenate(val_labels_list)
                val_preds_all = np.concatenate(val_preds_list)

                val_acc = float(
                    accuracy_score(val_labels_all, val_preds_all)
                )
                val_f1 = float(
                    f1_score(
                        val_labels_all,
                        val_preds_all,
                        average="macro",
                        zero_division=0,
                    )
                )
                val_mcc = float(
                    matthews_corrcoef(val_labels_all, val_preds_all)
                )

                # Preserve the legacy best-validation-loss behavior when early
                # stopping is disabled. With early stopping enabled, best_epoch
                # instead corresponds to the selected monitor metric.
                if val_loss < best_val:
                    best_val = val_loss
                    if not early_stopping_enabled:
                        best_epoch = epoch

        else:
            # Backward-compatible behavior for training without validation.
            best_epoch = epoch

        # Evaluate the chosen metric after validation has completed.
        if early_stopping_enabled:
            assert monitor_key is not None

            monitored_values: dict[str, float | None] = {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_mcc": val_mcc,
            }
            raw_current_val = monitored_values[monitor_key]

            if raw_current_val is None or not np.isfinite(raw_current_val):
                raise ValueError(
                    f"Early-stopping metric {monitor_key!r} is unavailable "
                    f"or non-finite at epoch {epoch}. Check the validation "
                    "loader and validation data."
                )

            current_val = float(raw_current_val)

            if is_minimize:
                improved = current_val < best_metric_val - min_delta
            else:
                improved = current_val > best_metric_val + min_delta

            if improved:
                best_metric_val = current_val
                epochs_no_improve = 0
                best_epoch = epoch

                # Snapshot the full state, including persistent model buffers.
                best_state_dict = copy.deepcopy(
                    wrapper.model.state_dict()
                )
            else:
                epochs_no_improve += 1

        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": float(epoch),
            "trn_loss": float(train_loss),
            "trn_acc": float(trn_acc),
            "trn_f1": float(trn_f1),
            "val_loss": (
                float("nan") if val_loss is None else float(val_loss)
            ),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "val_mcc": float(val_mcc),
            "learning_rate": current_lr,
            "defense_mode": mode,
        })
        
        _write_training_curve_atomic(curve_dir, history)

        if logger:
            logger.info(
                "training.epoch seed=%s epoch=%d/%d defense=%s "
                "train_loss=%.5f train_acc=%.4f train_f1=%.4f "
                "val_loss=%s val_acc=%s val_f1=%s lr=%.3e",
                seed,
                epoch,
                epochs,
                mode,
                train_loss,
                trn_acc,
                trn_f1,
                "n/a" if val_loss is None else f"{val_loss:.5f}",
                "n/a" if not np.isfinite(val_acc) else f"{val_acc:.4f}",
                "n/a" if not np.isfinite(val_f1) else f"{val_f1:.4f}",
                current_lr,
            )

            if early_stopping_enabled:
                logger.info(
                    "training.early_stopping.epoch seed=%s epoch=%d "
                    "monitor=%s value=%.6f best=%.6f stale_epochs=%d/%d",
                    seed,
                    epoch,
                    monitor_key,
                    current_val,
                    best_metric_val,
                    epochs_no_improve,
                    patience,
                )

        if (
            early_stopping_enabled
            and epochs_no_improve >= patience
        ):
            if logger:
                logger.info(
                    "training.early_stopping.stop seed=%s epoch=%d "
                    "best_epoch=%d monitor=%s best_value=%.6f "
                    "patience=%d",
                    seed,
                    epoch,
                    best_epoch,
                    monitor_key,
                    best_metric_val,
                    patience,
                )
            break

    # Restore best weights on both early termination and normal completion.
    # This happens before the wrapper is returned to downstream detector fitting
    # and checkpoint serialization in workflow.py.
    if best_state_dict is not None:
        wrapper.model.load_state_dict(best_state_dict)
        if logger:
            logger.info(
                "training.early_stopping.restored seed=%s best_epoch=%d "
                "monitor=%s best_value=%.6f",
                seed,
                best_epoch,
                monitor_key,
                best_metric_val,
            )

    model.eval()

    if logger:
        logger.info(
            "training.done seed=%s epochs=%d best_epoch=%d "
            "train_loss=%.5f val_loss=%s defense=%s",
            seed,
            epochs_completed,
            best_epoch,
            train_loss,
            "n/a" if val_loss is None else f"{val_loss:.5f}",
            mode,
        )

    return TrainingSummary(
        epochs=epochs_completed,
        train_loss=float(train_loss),
        validation_loss=(
            None if val_loss is None else float(val_loss)
        ),
        best_epoch=int(best_epoch),
        history=tuple(history),
        defense_mode=mode,
    )
