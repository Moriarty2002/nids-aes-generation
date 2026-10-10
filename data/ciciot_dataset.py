from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from core.registry import register_dataset
from .base import BaseDataset, NormalizationStats


@register_dataset("ciciot2023")
@register_dataset("ciciot2024")
@register_dataset("nsl_kdd")
@dataclass
class CICFlowDataset(BaseDataset):
    """CSV/Parquet flow dataset adapter supporting binary or multiclass labels."""

    path: str = ""
    label_column: str = "label"
    drop_columns: tuple[str, ...] = ()
    multiclass: bool = True
    normalization_path: str | None = None

    def __post_init__(self) -> None:
        BaseDataset.__init__(self, None)
        if not self.path:
            raise ValueError("Dataset path must be configured")
        self._frame = self._load_frame(Path(self.path))
        if self.label_column not in self._frame.columns:
            raise ValueError(f"Missing label column {self.label_column!r}")
        self._encode_labels()
        if self.normalization_path:
            self.normalization = self._load_normalization(Path(self.normalization_path))

    @staticmethod
    def _load_frame(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() in {".json", ".jsonl"}:
            return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    def _encode_labels(self) -> None:
        target = self._frame[self.label_column]
        if pd.api.types.is_numeric_dtype(target):
            labels = target.to_numpy(dtype=np.int64)
            classes = tuple(str(v) for v in sorted(set(labels.tolist())))
            self._frame[self.label_column] = labels
        else:
            values = sorted(target.astype(str).unique().tolist())
            mapping = {name: i for i, name in enumerate(values)}
            self._frame[self.label_column] = target.astype(str).map(mapping).astype(np.int64)
            classes = tuple(values)
        self.classes = classes

    @property
    def feature_names(self) -> tuple[str, ...]:
        excluded = {self.label_column, *self.drop_columns}
        return tuple(c for c in self._frame.columns if c not in excluded)

    def raw_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        x = self._frame.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(
            dtype=np.float32
        )
        y = self._frame[self.label_column].to_numpy(dtype=np.int64)
        return x, y

    @staticmethod
    def _load_normalization(path: Path) -> NormalizationStats:
        data = cast(dict[str, Any], pd.read_json(path, typ="series").to_dict())
        return NormalizationStats(
            mean=np.asarray(data["mean"], dtype=np.float64),
            std=np.asarray(data["std"], dtype=np.float64),
            feature_names=tuple(data["feature_names"]),
            version=str(data.get("version", "1.0")),
        )
