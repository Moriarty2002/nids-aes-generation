from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from core.registry import register_dataset
from .base import BaseDataset, NormalizationStats


_DEFAULT_FIELDS: tuple[str, ...] = ("PL", "IAT", "DIR", "WIN", "FLG", "TTL")
_DEFAULT_PAD_VALUES: dict[str, float] = {
    "PL": 0.0,
    "IAT": 0.0,
    "DIR": -1.0,
    "WIN": -1.0,
    "FLG": -1.0,
    "TTL": 0.0,
}


@register_dataset("ciciomt2024")
class SequenceParquetDataset(BaseDataset):
    """Variable-length packet-sequence dataset backed by a Parquet dataframe.

    Every dataframe row represents one flow. Sequence-valued packet fields are
    expanded to a common ``(N, T, F)`` representation. ``FEAT_PAD`` is treated
    as the number of padded packet positions when available; otherwise the
    maximum observed sequence length or explicit ``max_length`` determines T.

    Labels support both binary anomaly detection and string-valued multi-class
    attack classification. For string labels, ``benign_string`` is always
    assigned integer label 0, independent of LabelEncoder's alphabetical order;
    all attack classes are assigned deterministic labels 1..K.
    """

    REGISTRY_NAME = "ciciomt2024"

    def __init__(
        self,
        path: str | Path,
        *,
        fields: Sequence[str] = _DEFAULT_FIELDS,
        label_column: str = "LABEL-bin",
        benign_string: str = "Benign",
        task_mode: str = "auto",
        max_length: int | None = None,
        feat_pad_column: str = "FEAT_PAD",
        flatten: bool = False,
        include_padding_indicator: bool = False,
        padding_indicator_name: str = "FEAT_PAD_MASK",
        pad_values: Mapping[str, float] | None = None,
        strict_feat_pad: bool = True,
        truncate: bool = True,
        normalization_path: str | Path | None = None,
        id_column: str | None = "Index",
    ) -> None:
        """Load and canonicalize all flows once for deterministic worker access.

        The dataset keeps the padded canonical array and lazily materializes
        raw/normalized views. This avoids repeated Parquet parsing while
        keeping derived arrays out of memory until a workflow requests them.
        """
        super().__init__(None)
        self.path = Path(path)
        self.fields = tuple(str(field) for field in fields)
        self.label_column = str(label_column)
        self.benign_string = str(benign_string)
        self.task_mode = str(task_mode).strip().lower()
        self.max_length = max_length
        self.feat_pad_column = str(feat_pad_column)
        self.flatten = bool(flatten)
        self.include_padding_indicator = bool(include_padding_indicator)
        self.padding_indicator_name = str(padding_indicator_name)
        self.pad_values = {**_DEFAULT_PAD_VALUES, **dict(pad_values or {})}
        self.strict_feat_pad = bool(strict_feat_pad)
        self.truncate = bool(truncate)
        self.normalization_path = Path(normalization_path) if normalization_path else None
        self.id_column = id_column
        self.class_mapping: dict[str, int] = {}
        self.classes: tuple[str, ...] = ()
        self.label_mode: str = ""
        self.label_encoder: LabelEncoder | None = None

        if self.task_mode not in {"auto", "binary", "multiclass"}:
            raise ValueError("task_mode must be one of: auto, binary, multiclass")
        if not self.path.exists():
            raise FileNotFoundError(f"Parquet dataset does not exist: {self.path}")
        if self.path.suffix.lower() not in {".parquet", ".pq"}:
            raise ValueError(f"SequenceParquetDataset requires a Parquet file, got {self.path.suffix!r}")
        if not self.fields:
            raise ValueError("fields must contain at least one sequence feature")

        frame = pd.read_parquet(self.path)
        missing = [field for field in self.fields if field not in frame.columns]
        if missing:
            raise ValueError(f"Missing required sequence feature columns: {missing}")
        if self.label_column not in frame.columns:
            raise ValueError(f"Missing label column {self.label_column!r}")

        if self.id_column and self.id_column in frame.columns:
            self.sample_ids = tuple(str(value) for value in frame[self.id_column].tolist())
        else:
            self.sample_ids = tuple(str(value) for value in frame.index.tolist())

        self.lengths = np.empty(len(frame), dtype=np.int64)
        self._canonical, self.valid_mask = self._pad_sequences(frame)
        self._labels = self._load_labels(frame[self.label_column])
        self.normalization = self._load_normalization(self.normalization_path) if self.normalization_path else None
        self._raw_cache: np.ndarray | None = None
        self._normalized_cache: np.ndarray | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = list(self.fields)
        if self.include_padding_indicator:
            names.append(self.padding_indicator_name)
        return tuple(names)

    @property
    def sequence_shape(self) -> tuple[int, int]:
        return int(self._canonical.shape[1]), int(self._canonical.shape[2])

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        if self.flatten:
            return (len(self), self._canonical.shape[1] * len(self.feature_names))
        return tuple(self._canonical.shape[:-1]) + (len(self.feature_names),)

    def raw_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return cached features and a copy of labels for safe caller use."""
        if self._raw_cache is None:
            features = self._canonical
            if self.include_padding_indicator:
                pad_channel = 1.0 - self.valid_mask.astype(np.float32)[..., None]
                features = np.concatenate([features, pad_channel], axis=-1)
            if self.flatten:
                self._raw_cache = features.reshape(features.shape[0], -1).astype(np.float32, copy=False)
            else:
                self._raw_cache = features.astype(np.float32, copy=False)
        return self._raw_cache, self._labels.copy()

    def fit_normalization(
        self,
        x: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> NormalizationStats:
        """Fit per-feature statistics while excluding padded packet positions."""
        x_array = np.asarray(x, dtype=np.float64)
        n_samples = x_array.shape[0]
        sequence_length = self.sequence_shape[0]
        feature_count = len(self.feature_names)

        if self.flatten:
            if x_array.ndim != 2:
                raise ValueError(f"Expected a 2-D flattened tensor, got {x_array.shape}")
            expected = sequence_length * feature_count
            if x_array.shape[1] != expected:
                raise ValueError(f"Expected flattened width {expected}, got {x_array.shape[1]}")
            x_sequence = x_array.reshape(n_samples, sequence_length, feature_count)
        else:
            if x_array.ndim != 3:
                raise ValueError(f"Expected a 3-D sequence tensor, got {x_array.shape}")
            if x_array.shape[1:] != (sequence_length, feature_count):
                raise ValueError(
                    f"Expected sequence shape {(sequence_length, feature_count)}, got {x_array.shape[1:]}"
                )
            x_sequence = x_array

        mask_source = self.valid_mask if valid_mask is None else np.asarray(valid_mask, dtype=bool)
        if mask_source.shape != (n_samples, sequence_length):
            raise ValueError(
                f"valid_mask shape {mask_source.shape} does not match data shape {(n_samples, sequence_length)}"
            )
        valid = mask_source[..., None]

        means = np.zeros(feature_count, dtype=np.float64)
        stds = np.ones(feature_count, dtype=np.float64)
        for feature_idx in range(feature_count):
            values = x_sequence[..., feature_idx][valid[..., 0]]
            if values.size:
                means[feature_idx] = float(values.mean())
                stds[feature_idx] = max(float(values.std()), 1e-12)

        if self.flatten:
            mean = np.tile(means, sequence_length)
            std = np.tile(stds, sequence_length)
        else:
            mean, std = means, stds

        self.normalization = NormalizationStats(
            mean=mean,
            std=std,
            feature_names=self.feature_names,
            version="sequence-parquet-2.0",
        )
        self._normalized_cache = None
        return self.normalization

    def normalized_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._normalized_cache is None:
            raw, labels = self.raw_arrays()
            if self.normalization is None:
                self.fit_normalization(raw)
            assert self.normalization is not None
            self._normalized_cache = self.normalization.normalize(raw).astype(np.float32)
            return self._normalized_cache, labels
        return self._normalized_cache, self._labels.copy()

    def __len__(self) -> int:
        return int(len(self._labels))

    def __getitem__(self, idx: int):  # type: ignore[override]
        x, y = self.normalized_arrays()
        return torch.as_tensor(x[idx], dtype=torch.float32), torch.as_tensor(y[idx], dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "fields": list(self.fields),
            "label_column": self.label_column,
            "benign_string": self.benign_string,
            "task_mode": self.task_mode,
            "label_mode": self.label_mode,
            "classes": list(self.classes),
            "class_mapping": dict(self.class_mapping),
            "sequence_length": int(self.sequence_shape[0]),
            "feature_count": int(self.sequence_shape[1]),
            "flatten": self.flatten,
            "include_padding_indicator": self.include_padding_indicator,
            "padding_indicator_name": self.padding_indicator_name,
            "id_column": self.id_column,
            "pad_values": dict(self.pad_values),
            "normalization": None
            if self.normalization is None
            else {
                "mean": self.normalization.mean.tolist(),
                "std": self.normalization.std.tolist(),
                "feature_names": list(self.normalization.feature_names),
                "version": self.normalization.version,
            },
        }

    def _load_labels(self, series: pd.Series) -> np.ndarray:
        if series.isna().any():
            raise ValueError(f"Label column {self.label_column!r} contains missing values")

        # Parquet may store stringified numeric labels as object dtype. Treat a
        # fully numeric column as numeric before falling back to string labels.
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = not numeric.isna().any()
        if is_numeric:
            numeric_values = numeric.to_numpy(dtype=np.int64)
            unique = sorted(set(numeric_values.tolist()))
            inferred_binary = set(unique).issubset({0, 1})
            mode = "binary" if self.task_mode == "auto" and inferred_binary else self.task_mode
            if mode == "binary":
                self._validate_binary_numeric(numeric_values)
                self.label_mode = "binary"
                self.classes = ("0", "1")
                self.class_mapping = {"0": 0, "1": 1}
                return numeric_values
            if mode == "multiclass" or (self.task_mode == "auto" and not inferred_binary):
                if 0 not in unique:
                    raise ValueError("Multi-class numeric labels require benign class value 0")
                ordered = [0] + [value for value in unique if value != 0]
                mapping = {str(value): idx for idx, value in enumerate(ordered)}
                labels = np.asarray([mapping[str(value)] for value in numeric_values], dtype=np.int64)
                self.label_mode = "multiclass"
                self.classes = tuple(str(value) for value in ordered)
                self.class_mapping = mapping
                return labels
            raise ValueError(f"Unsupported label mode {self.task_mode!r}")

        values = series.astype(str).str.strip().to_numpy(dtype=object)
        if any(not value for value in values):
            raise ValueError("String labels cannot be empty")

        # LabelEncoder supplies deterministic canonical class discovery. Its
        # alphabetical index is deliberately NOT exposed as the benchmark label:
        # Benign is always remapped to zero for attack-evasion evaluation.
        encoder = LabelEncoder()
        encoder.fit(values)
        encoder_classes = [str(value) for value in encoder.classes_.tolist()]
        if self.benign_string not in encoder_classes:
            raise ValueError(
                f"benign_string={self.benign_string!r} was not found in {self.label_column!r}. "
                "Benign must be explicitly present for AD/AC evaluation."
            )

        ordered = [self.benign_string] + sorted(
            value for value in encoder_classes if value != self.benign_string
        )
        mapping = {value: idx for idx, value in enumerate(ordered)}
        labels = np.asarray([mapping[str(value)] for value in values], dtype=np.int64)

        self.label_encoder = encoder
        self.label_mode = "binary" if len(ordered) == 2 else "multiclass"
        self.classes = tuple(ordered)
        self.class_mapping = dict(mapping)
        return labels

    @staticmethod
    def _validate_binary_numeric(values: np.ndarray) -> None:
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"Binary labels must be 0/1; observed values: {sorted(unique)}")

    def _pad_sequences(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        parsed: dict[str, list[list[float]]] = {field: [] for field in self.fields}
        row_observed_lengths: list[int] = []
        declared_lengths: list[int | None] = []

        for row_index, row in frame.iterrows():
            row_lengths: list[int] = []
            for field in self.fields:
                sequence = self._coerce_sequence(row[field], field=field, row_index=row_index)
                parsed[field].append(sequence)
                row_lengths.append(len(sequence))
            observed = max(row_lengths, default=0)
            row_observed_lengths.append(observed)
            if self.feat_pad_column in frame.columns:
                feat_pad = row[self.feat_pad_column]
                if pd.isna(feat_pad):
                    declared_lengths.append(None)
                else:
                    pad_count = int(feat_pad)
                    if pad_count < 0:
                        raise ValueError(f"{self.feat_pad_column} cannot be negative at row {row_index}")
                    declared_lengths.append(observed + pad_count)
            else:
                declared_lengths.append(None)

        if self.max_length is not None:
            target_length = int(self.max_length)
            if target_length <= 0:
                raise ValueError("max_length must be > 0")
        elif any(value is not None for value in declared_lengths):
            target_length = max(int(value) for value in declared_lengths if value is not None)
        else:
            target_length = max(row_observed_lengths, default=0)
        if target_length <= 0:
            raise ValueError("The Parquet dataset contains no packet observations")

        canonical = np.empty((len(frame), target_length, len(self.fields)), dtype=np.float32)
        valid_mask = np.zeros((len(frame), target_length), dtype=bool)

        for row_idx in range(len(frame)):
            observed = row_observed_lengths[row_idx]
            declared = declared_lengths[row_idx]
            if declared is not None and self.strict_feat_pad and self.max_length is None and declared != target_length:
                raise ValueError(
                    f"Inconsistent {self.feat_pad_column} at row {row_idx}: "
                    f"observed_length={observed}, declared_length={declared}, global_length={target_length}"
                )

            if observed > target_length and not self.truncate:
                raise ValueError(
                    f"Row {row_idx} has {observed} packets, exceeding configured max_length={target_length}"
                )
            valid_length = min(observed, target_length)
            valid_mask[row_idx, :valid_length] = True
            self.lengths[row_idx] = valid_length

            for feature_idx, field in enumerate(self.fields):
                values = parsed[field][row_idx]
                if len(values) > target_length:
                    values = values[:target_length]
                pad_value = float(self.pad_values.get(field, 0.0))
                padded = np.full(target_length, pad_value, dtype=np.float32)
                if values:
                    padded[: len(values)] = np.asarray(values, dtype=np.float32)
                canonical[row_idx, :, feature_idx] = padded

        return canonical, valid_mask

    @staticmethod
    def _coerce_sequence(value: Any, *, field: str, row_index: Any) -> list[float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Cannot parse sequence field {field!r} at row {row_index}") from exc
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return [float(value)]
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric sequence value in field {field!r} at row {row_index}") from exc

    @staticmethod
    def _load_normalization(path: Path | None) -> NormalizationStats | None:
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Normalization file does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                payload = {
                    "mean": data["mean"].tolist(),
                    "std": data["std"].tolist(),
                    "feature_names": [str(value) for value in data["feature_names"].tolist()],
                    "version": str(data["version"].tolist()) if "version" in data else "1.0",
                }
        else:
            raise ValueError("normalization_path must point to .json or .npz")
        return NormalizationStats(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.clip(np.asarray(payload["std"], dtype=np.float64), 1e-12, None),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            version=str(payload.get("version", "1.0")),
        )
