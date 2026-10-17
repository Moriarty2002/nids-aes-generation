from __future__ import annotations

import gc
import importlib
import logging
import multiprocessing as mp
import random
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from core.registry import discover_all
from evaluator.logger import setup_logger, install_crash_handler
from .device import configure_process_device

@dataclass(frozen=True)
class SeedJob:
    """Serializable description of one independent worker invocation."""

    seed: int
    gpu: str
    target: str
    config: dict[str, Any]


def seed_everything(seed: int) -> None:
    """Configure deterministic Python, NumPy, and Torch randomness for a seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _invoke_dotted(target: str, config: dict[str, Any], seed: int, gpu: str) -> Mapping[str, float]:
    """Import and execute one Hydra-style ``module:function`` target in a worker."""
    out_dir = Path(config.get("output_dir", "runs"))
    worker_log_file = out_dir / "logs" / f"seed_{seed}.log"
    logger = setup_logger(f"Seed_{seed}", worker_log_file)
    install_crash_handler(logger)

    action = target.rsplit(":", 1)[-1].removesuffix("_seed")
    logger.info(
        "worker.start action=%s seed=%d device=%s target=%s",
        action, seed, gpu, target,
    )
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)

    seed_everything(seed)
    torch.set_num_threads(1)
    discover_all()
    device_info = configure_process_device(gpu)
    logger.info(
        "worker.ready action=%s seed=%d device=%s torch_device=%s threads=%d",
        action, seed, gpu, device_info.torch_device, torch.get_num_threads(),
    )

    try:
        # Pass the logger so workflow functions can use it
        result = func(config=config, seed=seed, gpu=gpu, logger=logger)
        logger.info("worker.done action=%s seed=%d metrics=%s", action, seed, sorted(result))
        return {str(k): float(v) for k, v in dict(result).items()}
    except Exception as exc:
        logger.error(f"Fatal crash in seed {seed}: {exc}", exc_info=True)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for handler in logger.handlers:
            handler.flush()


def run_seed_jobs(
    *,
    target: str,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    gpus: Sequence[str],
    max_workers: int | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, float | int | str]]:
    """Run independent seeds with ``spawn`` and one parent-owned progress bar.

    ``spawn`` avoids inheriting initialized Torch or CUDA state from Hydra's
    parent process. The parent owns progress reporting so worker loggers never
    compete with the progress bar on standard output.
    """
    if not seeds:
        raise ValueError("At least one seed is required")
    if not gpus:
        raise ValueError("At least one device assignment is required")

    workers = min(max_workers or len(seeds), len(seeds))
    if workers < 1:
        raise ValueError("max_workers must be >= 1")

    context = mp.get_context("spawn")
    jobs = [
        SeedJob(int(seed), str(gpus[i % len(gpus)]), target, dict(config))
        for i, seed in enumerate(seeds)
    ]
    output: list[dict[str, float | int | str]] = []
    if logger:
        logger.info(
            "jobs.dispatch target=%s seeds=%s workers=%d devices=%s",
            target, list(seeds), workers, list(gpus),
        )

    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        future_to_job: dict[Future[Mapping[str, float]], SeedJob] = {
            pool.submit(_invoke_dotted, job.target, job.config, job.seed, job.gpu): job
            for job in jobs
        }

        with tqdm(
            total=len(future_to_job),
            desc="Seeds",
            unit="seed",
            dynamic_ncols=True,
            leave=True,
        ) as progress:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result: dict[str, float | int | str] = dict(future.result())
                    result["seed"] = job.seed
                    result["gpu"] = job.gpu
                    output.append(result)
                    if logger:
                        logger.info(
                            "jobs.seed_complete seed=%d device=%s completed=%d/%d",
                            job.seed, job.gpu, len(output), len(jobs),
                        )
                    progress.set_postfix_str(f"seed={job.seed}", refresh=False)
                except Exception:
                    if logger:
                        logger.exception(
                            "jobs.seed_failed seed=%d device=%s target=%s",
                            job.seed, job.gpu, target,
                        )
                    raise
                finally:
                    progress.update(1)

    ordered = sorted(output, key=lambda row: int(row["seed"]))
    if logger:
        logger.info("jobs.done target=%s completed=%d/%d", target, len(ordered), len(jobs))
    return ordered
