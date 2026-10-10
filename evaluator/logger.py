from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import matplotlib
matplotlib.use("Agg")  # Enforce headless backend to prevent X11/GUI display errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


class FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes and fsyncs every record to survive server crashes/OOMs."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        try:
            stream = self.stream
            if stream is not None:
                os.fsync(stream.fileno())
        except (AttributeError, OSError):
            pass


def setup_logger(name: str, log_file: Path | str, level: int = logging.INFO) -> logging.Logger:
    """Configures an isolated logger writing to a dedicated, crash-resilient file."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid adding duplicate handlers on multi-run processes
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = FlushFileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def install_crash_handler(logger: logging.Logger) -> None:
    """Registers an uncaught exception hook to write catastrophic failures to disk."""
    def _hook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Uncaught fatal exception encountered:", exc_info=(exc_type, exc_value, exc_tb))
        logger.critical("Process aborted unexpectedly.")

    sys.excepthook = _hook


class ExperimentReporter:
    """Persist reproducible reports, predictions, matrices, and training curves.

    Reporting is intentionally file-based: workers can finish independently,
    while the parent process later combines their seed-level artifacts.
    """

    def __init__(self, run_dir: Path | str, class_names: Sequence[str] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.reports_dir = self.run_dir / "reports"
        self.curves_dir = self.run_dir / "curves"
        self.preds_dir = self.run_dir / "predictions"
        for folder in (self.reports_dir, self.curves_dir, self.preds_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.class_names = [str(c) for c in class_names] if class_names is not None else None

    def save_predictions(
        self,
        seed: int,
        labels: np.ndarray,
        baseline_preds: np.ndarray,
        adversarial_preds: np.ndarray | None = None,
        logits: np.ndarray | None = None,
    ) -> None:
        """Saves raw predictions and labels to .npz for post-hoc analysis."""
        payload: dict[str, np.ndarray] = {
            "labels": np.asarray(labels, dtype=np.int64),
            "baseline_preds": np.asarray(baseline_preds, dtype=np.int64),
        }
        if adversarial_preds is not None:
            payload["adversarial_preds"] = np.asarray(adversarial_preds, dtype=np.int64)
        if logits is not None:
            payload["logits"] = np.asarray(logits, dtype=np.float32)

        file_path = self.preds_dir / f"seed_{seed}_preds.npz"
        np.savez_compressed(file_path, **cast(dict[str, Any], payload))

    def save_seed_report(
        self,
        seed: int,
        metrics: Mapping[str, Any],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        phase: str = "adversarial",
    ) -> None:
        """Save one seed's metrics and classification report.

        Empty scoring masks are recorded explicitly instead of being passed to
        scikit-learn, because strict constraints may reject every sample.
        """
        report_path = self.reports_dir / f"report_seed_{seed}_{phase}.txt"
        observed = np.concatenate([np.asarray(y_true), np.asarray(y_pred)])
        unique_classes = (
            np.arange(len(self.class_names), dtype=np.int64)
            if observed.size == 0 and self.class_names
            else np.unique(observed)
        )
        target_names = (
            [self.class_names[int(c)] for c in unique_classes]
            if self.class_names and max(unique_classes) < len(self.class_names)
            else [f"Class_{c}" for c in unique_classes]
        )
        clf_report: str = (
            "No samples passed the evaluation scoring mask."
            if observed.size == 0
            else cast(
                str,
                classification_report(
                    y_true,
                    y_pred,
                    labels=unique_classes,
                    target_names=target_names,
                    zero_division=0,
                    output_dict=False,
                ),
            )
        )

        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Metrics Report - Seed {seed} ({phase.upper()})\n")
            f.write("=" * 60 + "\n\n")
            f.write("Classification Report:\n")
            f.write(clf_report + "\n\n")
            f.write("Evaluation Metrics:\n")
            f.write("-" * 30 + "\n")
            for k, v in sorted(metrics.items()):
                f.write(f"{k}: {v}\n")

    def generate_confusion_matrix(
        self,
        seed: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        phase: str = "adversarial",
    ) -> np.ndarray:
        """Write CSV/PDF confusion matrices and return the plotted matrix."""
        observed = np.concatenate([np.asarray(y_true), np.asarray(y_pred)])
        labels = (
            np.arange(len(self.class_names), dtype=np.int64)
            if self.class_names
            else np.unique(observed)
        )
        classes = self.class_names if self.class_names else [f"Class_{i}" for i in labels]

        if observed.size == 0:
            cm_raw = np.zeros((len(labels), len(labels)), dtype=np.int64)
            cm_norm = cm_raw.astype(np.float64)
        else:
            cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
            if phase == "successful_evasions":
                cm_norm = confusion_matrix(y_true, y_pred, labels=labels)
            else:
                cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        pd.DataFrame(cm_raw, index=classes, columns=classes).to_csv(
            self.reports_dir / f"cm_seed_{seed}_{phase}.csv"
        )

        n_classes = len(classes)
        plt.figure(figsize=(max(6, n_classes * 0.8), max(4, n_classes * 0.6)))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            xticklabels=classes,
            yticklabels=classes,
            cmap="viridis",
            cbar=True,
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(f"Normalized Confusion Matrix (Seed {seed} - {phase})")
        plt.savefig(self.reports_dir / f"cm_seed_{seed}_{phase}.pdf", bbox_inches="tight")
        plt.close()

        return cm_norm

    def save_aggregated_report(
        self,
        seed_metrics: Sequence[Mapping[str, Any]],
        cms: Sequence[np.ndarray],
    ) -> None:
        """Save mean/std metrics and the cross-seed confusion-matrix summary."""
        if not seed_metrics:
            return

        scalar_keys = sorted(
            {k for row in seed_metrics for k, v in row.items() if isinstance(v, (int, float)) and k not in {"seed", "gpu"}}
        )

        report_path = self.reports_dir / "report_aggregated.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Aggregated Metrics Report ({len(seed_metrics)} Seeds)\n")
            f.write("=" * 60 + "\n\n")
            for key in scalar_keys:
                vals = [float(row[key]) for row in seed_metrics if key in row]
                f.write(f"{key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Per-Seed Breakdown:\n")
            for row in seed_metrics:
                f.write(f"\nSeed {row.get('seed', 'N/A')}:\n")
                for key in scalar_keys:
                    if key in row:
                        f.write(f"  {key}: {row[key]:.4f}\n")

        # Mean Confusion Matrix Plot
        if cms:
            mean_cm = np.mean(np.stack(cms), axis=0)
            classes = self.class_names or [f"Class_{i}" for i in range(mean_cm.shape[0])]
            pd.DataFrame(mean_cm, index=classes, columns=classes).to_csv(
                self.reports_dir / "cm_mean.csv", float_format="%.4f"
            )

            classes = self.class_names or [f"Class_{i}" for i in range(mean_cm.shape[0])]
            n_classes = len(classes)
            plt.figure(figsize=(max(6, n_classes * 0.8), max(4, n_classes * 0.6)))
            sns.heatmap(
                mean_cm,
                annot=True,
                fmt=".2f",
                xticklabels=classes,
                yticklabels=classes,
                cmap="viridis",
                cbar=True,
            )
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.title(f"Mean Normalized Confusion Matrix ({len(cms)} Seeds)")
            plt.savefig(self.reports_dir / "cm_mean.pdf", bbox_inches="tight")
            plt.close()

    def plot_multi_seed_curves(self, curve_files: Sequence[Path | str]) -> None:
        """Plots training curves with individual seed trajectories and bold mean lines."""
        dfs = []
        for idx, file_path in enumerate(curve_files):
            p = Path(file_path)
            if p.exists():
                df = pd.read_csv(p)
                df["seed_idx"] = idx
                dfs.append(df)

        if not dfs:
            return

        combined = pd.concat(dfs, ignore_index=True)
        max_epoch = int(combined["epoch"].max())
        metric_cols = [
            c
            for c in combined.columns
            if c not in {"epoch", "seed_idx", "learning_rate", "defense_mode"}
        ]

        loss_cols = [c for c in metric_cols if "loss" in c.lower()]
        other_cols = [c for c in metric_cols if c not in loss_cols]

        for group_cols, filename, ylabel in [
            (loss_cols, "epoch_loss.pdf", "Loss Value"),
            (other_cols, "epoch_metrics.pdf", "Metric Score"),
        ]:
            if not group_cols:
                continue

            plt.figure(figsize=(max(8.0, max_epoch * 0.25), 5.0))
            colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]

            for i, col in enumerate(group_cols):
                color = colors[i % len(colors)]
                # Faint lines for individual seeds
                for seed_idx in combined["seed_idx"].unique():
                    subset = combined[combined["seed_idx"] == seed_idx]
                    valid = subset.dropna(subset=[col])
                    plt.plot(
                        valid["epoch"],
                        valid[col],
                        color=color,
                        alpha=0.25,
                        linewidth=1.2,
                        marker="o",
                        markersize=2,
                    )

                # Bold mean line across seeds
                epoch_means = combined.groupby("epoch")[col].mean()
                plt.plot(
                    epoch_means.index,
                    epoch_means.values,
                    color=color,
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    label=f"{col} (mean, N={len(dfs)})",
                )

            plt.xlabel("Epochs")
            plt.ylabel(ylabel)
            plt.grid(linestyle="--", alpha=0.6)
            plt.xticks(np.arange(1, max_epoch + 1, max(1, max_epoch // 20)))
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.savefig(self.curves_dir / filename, bbox_inches="tight")
            plt.close()