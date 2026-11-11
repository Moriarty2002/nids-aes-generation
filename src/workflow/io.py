from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from evaluator.engine import FeatureNormalization
from core.exceptions import ConfigurationError
from models.wrapper import ModelWrapper, PyTorchModelWrapper, SklearnModelWrapper

def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_strict_json_values(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _strict_json_values(value: Any) -> Any:
    """Preserve undefined numeric rates as JSON null (never fake zero)."""
    if isinstance(value, np.ndarray):
        return _strict_json_values(value.tolist())
    if isinstance(value, torch.Tensor):
        return _strict_json_values(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return _strict_json_values(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _strict_json_values(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json_values(v) for v in value]
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _load_checkpoint(
    wrapper: ModelWrapper,
    checkpoint: str | Path,
    *,
    checkpoint_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore torch state or a fitted sklearn estimator checkpoint.

    Passing metadata that was already loaded avoids deserializing a potentially
    large Random Forest twice during evaluation.
    """
    if checkpoint_data is None:
        map_location: Any = wrapper.device if isinstance(wrapper, PyTorchModelWrapper) else "cpu"
        data = torch.load(str(checkpoint), map_location=map_location, weights_only=False)
    else:
        data = dict(checkpoint_data)
    if isinstance(wrapper, PyTorchModelWrapper):
        model_state = data.get("model_state")
        if model_state is None:
            raise ConfigurationError(f"Checkpoint {checkpoint} has no PyTorch model_state")
        wrapper.model.load_state_dict(model_state, strict=True)
        wrapper.model.eval()
    elif isinstance(wrapper, SklearnModelWrapper):
        estimator = data.get("estimator")
        if estimator is None:
            raise ConfigurationError(
                f"Checkpoint {checkpoint} has no serialized sklearn estimator. "
                "Re-train this model with the updated workflow."
            )
        if not callable(getattr(estimator, "fit", None)) or not callable(getattr(estimator, "predict", None)):
            raise ConfigurationError(f"Checkpoint {checkpoint} contains an invalid sklearn estimator")
        wrapper.estimator = estimator
        wrapper.capabilities = type(wrapper.capabilities)(
            differentiable=False,
            gradient_input=False,
            probabilistic_output=callable(getattr(estimator, "predict_proba", None)),
            model_family="sklearn",
        )
    else:
        raise ConfigurationError(f"Unsupported model wrapper type {type(wrapper).__name__}")
    return data


def _normalization_from_checkpoint(
    checkpoint_data: Mapping[str, Any],
    dataset_feature_names: tuple[str, ...],
    *,
    role: str,
) -> FeatureNormalization:
    feature_names = tuple(str(x) for x in checkpoint_data["feature_names"])
    if feature_names != dataset_feature_names:
        raise ConfigurationError(
            f"Dataset feature schema differs from {role} checkpoint schema: "
            f"{dataset_feature_names} != {feature_names}"
        )
    mean = np.asarray(checkpoint_data["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint_data["normalization_std"], dtype=np.float32)
    return FeatureNormalization(mean=mean, std=std, feature_names=feature_names)


def _load_checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return dict(torch.load(checkpoint, map_location="cpu", weights_only=False))


def _write_sklearn_training_curve(
    output_dir: Path,
    *,
    train_loss: float,
    validation_loss: float,
    train_accuracy: float,
    validation_accuracy: float,
    train_f1: float,
    validation_f1: float,
    defense_mode: str,
) -> None:
    """Write a one-row training curve for estimators trained in a single fit call.

    The column names match ``models.trainer`` so existing cross-seed plotting
    does not fail when a Random Forest seed has no neural-network epoch loop.
    The file is atomically replaced to avoid partial reads by monitoring jobs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "training_curve.csv"
    temporary = output_dir / f"training_curve.{os.getpid()}.tmp"
    fields = (
        "epoch", "trn_loss", "trn_acc", "trn_f1", "val_loss", "val_acc",
        "val_f1", "learning_rate", "defense_mode",
    )
    row = {
        "epoch": 1,
        "trn_loss": train_loss,
        "trn_acc": train_accuracy,
        "trn_f1": train_f1,
        "val_loss": validation_loss,
        "val_acc": validation_accuracy,
        "val_f1": validation_f1,
        "learning_rate": 0.0,
        "defense_mode": defense_mode,
    }
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
