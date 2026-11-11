"""Best-effort, per-execution progress snapshots; never scientific artifacts.

Workers publish one bounded, replaceable JSON status per seed. Only the parent
renders terminal output; no progress event is a model inference or a query.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO

from tqdm import tqdm


class ProgressCallback(Protocol):
    """Optional stage/completed-unit notification, with no model access."""

    def __call__(self, phase: str, *, completed: int | None = None,
                 total: int | None = None, unit: str | None = None,
                 status: str = 'running', **details: Any) -> None: ...


class ProgressPublisher:
    """Write real completed work units, or an unmetered phase without a percent.

    Publication is throttled except on phase transitions and terminal states.
    File-system failures disable the display, never the worker's science.
    """

    def __init__(self, logs_dir: Path, seed: int, action: str, *,
                 logger: logging.Logger | None = None,
                 min_interval: float = 1.0) -> None:
        self.path = Path(logs_dir) / f'progress_seed_{int(seed)}.json'
        self.seed = int(seed)
        self.action = str(action)
        self.logger = logger
        self.min_interval = max(0.0, float(min_interval))
        self._last_written = float('-inf')
        self._last_phase: str | None = None
        self._last_completed: int | None = None
        self._last_total: int | None = None
        self._last_unit: str | None = None
        self._disabled = False

    def __call__(self, phase: str, *, completed: int | None = None,
                 total: int | None = None, unit: str | None = None,
                 status: str = 'running', **details: Any) -> None:
        if self._disabled:
            return
        if not isinstance(phase, str) or not phase:
            raise ValueError('Progress requires a named phase')
        if status not in {'running', 'completed', 'failed'}:
            raise ValueError('Unknown progress status')
        if total is None:
            if completed is not None:
                raise ValueError('Progress without a total cannot claim completed work')
        elif (not isinstance(total, int) or not isinstance(completed, int)
              or total < 0 or not 0 <= completed <= total):
            raise ValueError('Progress must measure real completed units in [0,total]')
        same_phase = phase == self._last_phase
        if same_phase and (total != self._last_total or
                           (completed is not None and self._last_completed is not None
                            and completed < self._last_completed)):
            raise ValueError('Progress counters must be monotonic within a fixed phase')
        self._last_phase = phase
        self._last_completed = completed
        self._last_total = total
        self._last_unit = None if total is None else (unit or 'units')
        now = time.monotonic()
        if (same_phase and status == 'running'
                and now - self._last_written < self.min_interval):
            return
        payload: dict[str, Any] = {
            'version': 1, 'seed': self.seed, 'action': self.action,
            'phase': phase, 'completed': completed, 'total': total,
            'unit': None if total is None else (unit or 'units'),
            'status': status,
            'updated_utc': datetime.now(timezone.utc).isoformat(),
        }
        for key in ('epoch', 'epochs', 'note'):
            if key in details and isinstance(details[key], (str, int, float)):
                payload[key] = details[key]
        temporary = self.path.with_name(f'{self.path.name}.{os.getpid()}.tmp')
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
            os.replace(temporary, self.path)
            self._last_written = now
        except OSError as exc:
            self._disabled = True
            if self.logger is not None:
                self.logger.warning('progress.disabled seed=%d reason=%s', self.seed, exc)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def mark_failed(self) -> None:
        """Retain the last genuinely completed phase count on worker failure."""
        self(self._last_phase or 'worker_failed', completed=self._last_completed,
             total=self._last_total, unit=self._last_unit, status='failed')


def read_snapshot(path: Path, *, seed: int) -> dict[str, Any] | None:
    """Ignore missing, torn, or stale observability files; never fail execution."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('version') != 1 or data.get('seed') != seed:
            return None
        completed, total = data.get('completed'), data.get('total')
        if total is not None and (not isinstance(total, int) or not isinstance(completed, int)
                                  or total < 0 or not 0 <= completed <= total):
            return None
        if total is None and completed is not None:
            return None
        if not isinstance(data.get('phase'), str):
            return None
        return data
    except (OSError, ValueError, TypeError, UnicodeError):
        return None


class ProgressRenderer:
    """Sole terminal owner; stage bars measure units, jobs bar measures jobs.

    Redirected terminals receive phase milestones, not carriage-return bars.
    Reading/displaying a bad snapshot cannot alter the run's scientific result.
    """

    def __init__(self, *, seeds: list[int], logs_dir: Path, action: str,
                 model: str, attack: str, enabled: bool, stream: TextIO | None = None,
                 tty: bool | None = None, plaintext_interval: float = 30.0) -> None:
        self.seeds = seeds
        self.logs_dir = Path(logs_dir)
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        self.tty = (bool(self.stream.isatty()) if tty is None else tty) and enabled
        self.plaintext_interval = plaintext_interval
        self.action = action
        self.jobs_done = 0
        self._phase: dict[int, str] = {}
        self._status: dict[int, str] = {}
        self._stage_started: dict[int, float] = {}
        self._count: dict[int, int] = {}
        self._last_text: dict[int, float] = {}
        self._bars: dict[int, Any] = {}
        self._job_bar: Any = None
        if enabled:
            self._print(f'Run | action={action} model={model} attack={attack} seeds={",".join(map(str, seeds))}')
        if self.tty:
            try:
                for position, seed in enumerate(seeds):
                    self._bars[seed] = tqdm(total=None, desc=f'Seed {seed} waiting', unit='units',
                                            position=position, leave=False, dynamic_ncols=True,
                                            file=self.stream)
                self._job_bar = tqdm(total=len(seeds), desc='Completed seed jobs',
                                     unit='jobs', position=len(seeds), leave=True,
                                     dynamic_ncols=True, file=self.stream)
            except (OSError, ValueError):
                self.close()
                self.tty = False

    def _print(self, line: str) -> None:
        try:
            print(line, file=self.stream, flush=True)
        except (OSError, ValueError):
            self.enabled = False

    def refresh(self, active: list[int]) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        for seed in active:
            data = read_snapshot(self.logs_dir / f'progress_seed_{seed}.json', seed=seed)
            if data is None:
                continue
            phase = str(data['phase'])
            phase_changed = self._phase.get(seed) != phase
            status_changed = self._status.get(seed) != data.get('status')
            if phase_changed:
                self._stage_started[seed] = now
            stage_elapsed = int(now - self._stage_started.get(seed, now))
            completed = data['completed']
            total = data['total']
            if self.tty:
                try:
                    bar = self._bars[seed]
                    if phase_changed:
                        bar.set_description_str(f'Seed {seed} {phase}', refresh=False)
                        # tqdm.reset(total=None) retains the previous total. Explicitly
                        # clear it so opaque stages never inherit an invented percent.
                        bar.total = total
                        bar.reset(total=total)
                        self._count[seed] = 0
                    if completed is not None:
                        bar.update(max(0, completed - self._count.get(seed, 0)))
                        self._count[seed] = completed
                    else:
                        bar.refresh()
                except (OSError, ValueError):
                    self.close()
                    self.tty = False
            else:
                if phase_changed or status_changed or (
                    now - self._last_text.get(seed, float('-inf')) >= self.plaintext_interval
                ):
                    if total is not None:
                        pct = f' ({100 * completed / total:.0f}%)' if total else ''
                        work = f' processed={completed}/{total} {data.get("unit", "units")}{pct}' 
                    else:
                        work = ' active (no measured total)'
                    self._print(f'seed={seed} phase={phase}{work} elapsed={stage_elapsed}s status={data.get("status", "running")}')
                    self._last_text[seed] = now
            self._phase[seed] = phase
            self._status[seed] = str(data.get('status', 'running'))

    def finish_seed(self, seed: int, *, ok: bool, log_path: Path, error: BaseException | None = None) -> None:
        self.jobs_done += 1
        if self._job_bar is not None:
            try:
                self._job_bar.update(1)
            except (OSError, ValueError):
                pass
        if ok:
            self._print(f'seed={seed} completed jobs={self.jobs_done}/{len(self.seeds)} log={log_path}')
        else:
            self._print(f'seed={seed} failed action={self.action} log={log_path} error={error}')

    def close(self) -> None:
        for bar in self._bars.values():
            try:
                bar.close()
            except (OSError, ValueError):
                pass
        self._bars.clear()
        if self._job_bar is not None:
            try:
                self._job_bar.close()
            except (OSError, ValueError):
                pass
            self._job_bar = None
