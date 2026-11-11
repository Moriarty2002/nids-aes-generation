from __future__ import annotations

import gc
import importlib
import inspect
import logging
import multiprocessing as mp
import random
from concurrent.futures import Future, ProcessPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from core.registry import discover_all
from evaluator.logger import setup_logger, install_crash_handler
from .device import configure_process_device
from .progress import ProgressPublisher, ProgressRenderer

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


def _progress_mode(config: Mapping[str, Any]) -> str:
    """Normalize the one supported progress switch at both process boundaries."""
    mode = str(config.get("progress", "auto")).strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("progress must be auto, on or off")
    return mode


def _invoke_dotted(target: str, config: dict[str, Any], seed: int, gpu: str) -> Mapping[str, float]:
    """Import and execute one Hydra-style ``module:function`` target in a worker."""
    progress_mode = _progress_mode(config)
    out_dir = Path(config.get("output_dir", "../runs"))
    worker_log_file = out_dir / "logs" / f"seed_{seed}.log"
    logger = setup_logger(f"Seed_{seed}", worker_log_file, console=False)
    action = target.rsplit(":", 1)[-1].removesuffix("_seed")
    publisher = (ProgressPublisher(out_dir / "logs", seed, action, logger=logger)
                 if progress_mode != "off" else None)
    if publisher is not None:
        publisher("worker_start")
    install_crash_handler(logger)

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
        # Older research plugins use the four-keyword workflow signature.
        # Inspect the contract once; never retry a failed invocation on TypeError,
        # because a plugin may already have performed scientific side effects.
        signature = inspect.signature(func)
        accepts_progress = ("progress" in signature.parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in signature.parameters.values()
        ))
        keywords = {"config": config, "seed": seed, "gpu": gpu, "logger": logger}
        if accepts_progress:
            keywords["progress"] = publisher
        result = func(**keywords)
        if publisher is not None:
            publisher("worker_done", status="completed")
        logger.info("worker.done action=%s seed=%d metrics=%s", action, seed, sorted(result))
        return {str(k): float(v) for k, v in dict(result).items()}
    except Exception as exc:
        if publisher is not None:
            publisher.mark_failed()
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

    progress_mode = _progress_mode(config)
    logs_dir = Path(str(config.get('output_dir', '../runs'))) / 'logs'
    # Status snapshots are replaceable UI hints, never immutable execution data.
    # Discard stale hints if a diagnostic invocation reuses an output directory;
    # the real workflow already assigns a fresh unique execution directory.
    for job in jobs:
        try:
            (logs_dir / f'progress_seed_{job.seed}.json').unlink(missing_ok=True)
        except OSError as exc:
            if logger:
                logger.warning('jobs.progress_reset_failed seed=%d error=%s', job.seed, exc)
    renderer = ProgressRenderer(
        seeds=[job.seed for job in jobs], logs_dir=logs_dir,
        action=target.rsplit(':', 1)[-1].removesuffix('_seed'),
        model=str(config.get('model', {}).get('name', 'unknown')),
        attack=str(config.get('attack', {}).get('name', 'none')),
        enabled=progress_mode != 'off',
    )
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            future_to_job: dict[Future[Mapping[str, float]], SeedJob] = {
                pool.submit(_invoke_dotted, job.target, job.config, job.seed, job.gpu): job
                for job in jobs
            }
            pending = set(future_to_job)
            while pending:
                done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                renderer.refresh([future_to_job[future].seed for future in pending])
                for future in sorted(done, key=lambda f: future_to_job[f].seed):
                    job = future_to_job[future]
                    log_path = logs_dir / f'seed_{job.seed}.log'
                    try:
                        result: dict[str, float | int | str] = dict(future.result())
                        result['seed'] = job.seed
                        result['gpu'] = job.gpu
                        output.append(result)
                        if logger:
                            logger.info('jobs.seed_complete seed=%d device=%s completed=%d/%d',
                                        job.seed, job.gpu, len(output), len(jobs))
                        renderer.finish_seed(job.seed, ok=True, log_path=log_path)
                    except Exception as exc:
                        if logger:
                            logger.error('jobs.seed_failed seed=%d device=%s target=%s log=%s error=%s',
                                         job.seed, job.gpu, target, log_path, exc)
                        renderer.finish_seed(job.seed, ok=False, log_path=log_path, error=exc)
                        raise
    finally:
        renderer.close()

    ordered = sorted(output, key=lambda row: int(row["seed"]))
    if logger:
        logger.info("jobs.done target=%s completed=%d/%d", target, len(ordered), len(jobs))
    return ordered
