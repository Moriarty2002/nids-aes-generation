from __future__ import annotations

import csv
from datetime import datetime, timezone
import platform
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path
from typing import Any, cast

import pandas as pd

# Canonical application imports use top-level core/attacks/workflow modules.
# Support both `python src/main.py` and `python -m src.main` without creating
# duplicate src.core versus core module identities.
_source_root = str(Path(__file__).resolve().parent)
if _source_root not in sys.path:
    sys.path.insert(0, _source_root)

import hydra
from omegaconf import DictConfig, OmegaConf

from core.registry import discover_all
from evaluator.logger import ExperimentReporter, install_crash_handler, setup_logger
from evaluator.statistical import aggregate_by_seed, compare_against_baseline
from core.provenance import source_revision_sha256
from orchestrator.device import resolve_device
from orchestrator.runner import run_seed_jobs
from workflow.budget import apply_budget
from workflow.io import _save_json
from workflow.seeds import role_seed, validated_worker_seeds
from workflow.experiment import ExperimentStore, create_experiment_dir, preflight_action


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
    _save_json(output_dir / "summary.json", payload)


def _action(cfg: DictConfig) -> None:
    discover_all()
    apply_budget(cfg)
    action = str(cfg.action)
    gpu = str(cfg.gpu)
    resolve_device(gpu)
    seeds = validated_worker_seeds(list(cfg.get('seeds', [1, 2, 3, 4, 5])))
    
    if action == "preprocess":
        seeds = seeds[:1]

    # One versioned scientific experiment spans independently timed commands.
    # Only the initial target/surrogate training action may create an experiment.
    requested_experiment = cfg.get('experiment_dir')
    if requested_experiment is None:
        if action not in {'train_nids', 'preprocess'}:
            raise ValueError('Select an existing experiment_dir for this action; no latest-run fallback')
        experiment_dir = create_experiment_dir(Path.cwd() / 'runs')
    else:
        experiment_dir = Path(str(requested_experiment)).expanduser().resolve()
    store = ExperimentStore(experiment_dir, dataset=str(cfg.dataset.name), create=(action in {'train_nids', 'preprocess'}))
    cfg.experiment_dir = str(store.root)
    execution_id = f'{action}_{cfg.model.name}_{cfg.attack.name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    output_dir = store.root / 'executions' / execution_id
    cfg.output_dir = str(output_dir)
        
    workers = int(cfg.get("parallel_workers", min(len(seeds), 5)))
    devices = [str(x) for x in cfg.get("gpu_partitions", [gpu])]
    
    target = {
        "train_nids": "workflow.train:train_nids_seed",
        "train_attack": "workflow.train:train_attack_seed",
        "calibrate_detector": "workflow.calibrate_detector:calibrate_detector_seed",
        "evaluate_distribution_stress": "workflow.distribution_stress:distribution_stress_seed",
        "evaluate_attack": "workflow.evaluate:evaluate_attack_seed",
        "preprocess": "workflow.train:preprocess_seed",
    }.get(action)
    
    if target is None:
        raise ValueError(f"Unsupported action {action!r}")
        
    # Resolve the execution path once; all workers share it for this command.
    resolved_output_dir = str(output_dir)
    
    # 2. Create the unified directory manually
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logger = setup_logger(
        "orchestrator",
        output_dir / "logs" / "orchestrator.log",
        console=False,  # Only the parent-owned progress renderer reaches terminal.
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
    
    # Preserve original interpolated config and the fully resolved, effective
    # smoke/research settings actually supplied to every worker process.
    OmegaConf.save(cfg, output_dir / "config.yaml")
    resolved_snapshot = _resolved_config_dict(cfg)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )
    libraries = {}
    for package in ("numpy", "pandas", "torch", "scikit-learn", "scipy", "hydra-core", "omegaconf"):
        try:
            libraries[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            libraries[package] = None
    (output_dir / "environment.json").write_text(
        json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                    "source_revision_sha256": source_revision_sha256(),
                    "libraries": libraries, "seed_roles": {"seed_list": seeds,
                    "configured": resolved_snapshot.get("seed_roles", {})}},
                   indent=2, sort_keys=True), encoding="utf-8"
    )
    
    # 4. Freeze the path into the dictionary passed to workers
    config_dict = _resolved_config_dict(cfg)
    config_dict["output_dir"] = resolved_output_dir
    preflight_action(store, config_dict, action, seeds)

    # Log and identify even failed invocations, without registering partial weights.
    store.record_execution(execution_id, action=action, seeds=seeds, output_dir=output_dir)
    try:
        seed_rows = run_seed_jobs(
            target=target, config=config_dict, seeds=seeds, gpus=devices,
            max_workers=workers, logger=run_logger,
        )
    except BaseException:
        store.finish_execution(execution_id, success=False)
        raise

    try:
        # No hypothesis test on scalar seed numbers without a verified matching
        # population, checkpoint, threat, preprocessing, and comparable budget.
        _write_summary(output_dir, seed_rows)

        # Extract true class names from the first seed's checkpoint
        class_names = None
        if action == "evaluate_attack":
            first_ckpt = store.model_path(
                str(cfg.model.name), str(cfg.model.training.defense_mode),
                role_seed(config_dict, 'target', seeds[0]),
            )
            store.require('model', first_ckpt, seed=role_seed(config_dict, 'target', seeds[0]))
            meta = json.loads((first_ckpt.parent / 'training_provenance.json').read_text(encoding='utf-8'))
            class_names = meta.get('classes')
        elif action == "train_nids":
            first_ckpt = store.model_path(str(cfg.model.name), str(cfg.model.training.defense_mode), seeds[0])
            store.require('model', first_ckpt, seed=seeds[0])
            meta = json.loads((first_ckpt.parent / 'training_provenance.json').read_text(encoding='utf-8'))
            class_names = meta.get('classes')

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
            curve_files = [store.model_path(str(cfg.model.name), str(cfg.model.training.defense_mode), s).parent / "training_curve.csv" for s in seeds]
            reporter.plot_multi_seed_curves(curve_files)
        
    except BaseException:
        store.finish_execution(execution_id, success=False)
        raise
    else:
        store.finish_execution(execution_id, success=True)

    print(json.dumps({"action": action, "experiment_dir": str(store.root),
                      "output_dir": str(output_dir), "seed_count": len(seed_rows)}, indent=2))
    run_logger.info("run.done action=%s output=%s seed_count=%d", action, output_dir, len(seed_rows))


OmegaConf.register_new_resolver("uuid", lambda: uuid.uuid4().hex[:8])
OmegaConf.register_new_resolver(
    "eval_path", 
    lambda action, name, variant: f"{name}_{variant}" if action in ["evaluate_attack", "train_attack"] else "baseline",
    use_cache=True
)
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _action(cfg)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_normalize_cli(sys.argv[1:])]
    main()
