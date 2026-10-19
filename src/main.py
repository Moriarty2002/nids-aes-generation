from __future__ import annotations

import csv
import json
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

# The requested repository layout keeps this package directory flat. When main.py is executed
# directly from inside that directory, add its parent so the absolute package imports resolve.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from core.registry import discover_all
from evaluator.logger import ExperimentReporter, install_crash_handler, setup_logger
from evaluator.statistical import aggregate_by_seed, compare_against_baseline
from orchestrator.device import resolve_device
from orchestrator.runner import run_seed_jobs


def _normalize_cli(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in {"--action", "--gpu"} and i + 1 < len(argv):
            out.append(f"{token[2:]}={argv[i + 1]}")
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _resolved_config_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert Hydra's config object into a plain, serializable mapping."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra configuration must resolve to a mapping")
    return cast(dict[str, Any], resolved)


def _metric_rows(rows: list[dict[str, Any]], *, baseline: bool) -> list[dict[str, float | int]]:
    """Extract numeric metrics while preserving seed pairing."""
    extracted: list[dict[str, float | int]] = []
    for row in rows:
        metrics: dict[str, float | int] = {"seed": int(row["seed"])}
        for key, value in row.items():
            if key in {"seed", "gpu"}:
                continue
            if baseline:
                if not key.startswith("baseline_"):
                    continue
                metric_name = key.removeprefix("baseline_")
            else:
                if key.startswith("baseline_"):
                    continue
                metric_name = key
            metrics[metric_name] = float(value)
        extracted.append(metrics)
    return extracted


def _write_summary(output_dir: Path, seed_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = sorted({key for row in seed_rows for key in row})
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(seed_rows)
    aggregate = aggregate_by_seed(_metric_rows(seed_rows, baseline=False))
    payload: dict[str, Any] = {"seed_metrics": seed_rows, "mean_std": {k: v.as_dict(k) for k, v in aggregate.items()}}
    if baseline_rows is not None:
        attack_metrics = _metric_rows(seed_rows, baseline=False)
        base_metrics = _metric_rows(baseline_rows, baseline=True)
        if not any(row for row in base_metrics if len(row) > 1):
            base_metrics = _metric_rows(baseline_rows, baseline=False)
        payload["wilcoxon_vs_baseline"] = compare_against_baseline(base_metrics, attack_metrics)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _action(cfg: DictConfig) -> None:
    discover_all()
    action = str(cfg.action)
    gpu = str(cfg.gpu)
    resolve_device(gpu)
    seeds = [int(x) for x in cfg.get("seeds", [1, 2, 3, 4, 5])]
    
    if action == "preprocess":
        seeds = seeds[:1]
        
    workers = int(cfg.get("parallel_workers", min(len(seeds), 5)))
    devices = [str(x) for x in cfg.get("gpu_partitions", [gpu])]
    
    target = {
        "train_nids": "workflow:train_nids_seed",
        "train_attack": "workflow:train_attack_seed",
        "evaluate_attack": "workflow:evaluate_attack_seed",
        "preprocess": "workflow:preprocess_seed",
    }.get(action)
    
    if target is None:
        raise ValueError(f"Unsupported action {action!r}")
        
    # 1. Resolve output_dir exactly once
    resolved_output_dir = str(cfg.get("output_dir", "../runs"))
    output_dir = Path(resolved_output_dir)
    
    # 2. Create the unified directory manually
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logger = setup_logger(
        "orchestrator",
        output_dir / "logs" / "orchestrator.log",
    )
    install_crash_handler(run_logger)
    run_logger.info(
        "run.start action=%s model=%s dataset=%s attack=%s seeds=%s workers=%d devices=%s",
        action,
        cfg.model.get("name"),
        cfg.dataset.get("name"),
        cfg.attack.get("name"),
        seeds,
        workers,
        devices,
    )
    
    # 3. Manually dump the exact Hydra config for reproducibility
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # 4. Freeze the path into the dictionary passed to workers
    config_dict = _resolved_config_dict(cfg)
    config_dict["output_dir"] = resolved_output_dir 
    
    # Run the multiprocessing jobs
    seed_rows = run_seed_jobs(
        target=target,
        config=config_dict,
        seeds=seeds,
        gpus=devices,
        max_workers=workers,
        logger=run_logger,
    )
    
    _write_summary(output_dir, seed_rows, seed_rows if action == "evaluate_attack" else None)

    # Extract true class names from the first seed's checkpoint
    class_names = None
    if action == "evaluate_attack":
        # Read the external checkpoint path defined in the model config
        ckpt_template = str(cfg.model.get("checkpoint", ""))
        if ckpt_template:
            first_ckpt = Path(ckpt_template.format(seed=seeds[0]))
            if first_ckpt.exists():
                meta = torch.load(first_ckpt, map_location="cpu", weights_only=False)
                class_names = meta.get("classes")
    elif action == "train_nids":
        # Read the checkpoint generated in the current output directory
        first_ckpt = output_dir / "checkpoints" / f"{cfg.model.name}_seed{seeds[0]}" / "nids.pt"
        if first_ckpt.exists():
            meta = torch.load(first_ckpt, map_location="cpu", weights_only=False)
            class_names = meta.get("classes")

    reporter = ExperimentReporter(output_dir, class_names=class_names)
    
    if action == "evaluate_attack":
        cms = []
        for row in seed_rows:
            s = int(row["seed"])
            cm_path = output_dir / "reports" / f"cm_seed_{s}_adversarial.csv"
            if cm_path.exists():
                # Load raw counts so the aggregate remains an average of seed counts.
                df = pd.read_csv(cm_path, index_col=0)
                raw_cm = df.to_numpy(dtype=float)
                cms.append(raw_cm)
                
        reporter.save_aggregated_report(seed_rows, cms)
        
    elif action == "train_nids":
        curve_files = [output_dir / "checkpoints" / f"{cfg.model.name}_seed{s}" / "training_curve.csv" for s in seeds]
        reporter.plot_multi_seed_curves(curve_files)
        
    print(json.dumps({"action": action, "output_dir": str(output_dir), "seed_count": len(seed_rows)}, indent=2))
    run_logger.info("run.done action=%s output=%s seed_count=%d", action, output_dir, len(seed_rows))


OmegaConf.register_new_resolver("uuid", lambda: uuid.uuid4().hex[:8])
OmegaConf.register_new_resolver(
    "eval_path", 
    lambda action, name, variant: f"{name}_{variant}" if action == "evaluate_attack" else "baseline",
    use_cache=True
)
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _action(cfg)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_normalize_cli(sys.argv[1:])]
    main()
