"""Stdlib-only ``Infra`` helper for run tracking and outputs management."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import __main__

logger = logging.getLogger(__name__)

DEFAULT_OUTPUTS_ROOT = Path("/tmp")


class Infra:
    """Lower-level run tracking and outputs management.
    An OS environment variable ``OUTPUTS_ROOT`` can specify output dir for run records.

    - ``capture_env(experiment_id)`` to open a run (snapshot provenance and allocate a fresh run dir).
    - ``run_with_timer(env)`` to time the run.
    - ``record_success(env, params, result, timer, extra_meta)`` to write the record on success.
    """

    @dataclass(frozen=True)
    class Env:
        experiment_id: str
        run_dir: Path
        git_rev: str
        git_dirty: bool
        python: str
        platform: str
        hostname: str

    @dataclass
    class Timer:
        started: float
        finished: float | None = None

        @property
        def duration_s(self) -> float | None:
            if self.finished is None:
                return None
            return round(self.finished - self.started, 3)

    # ---- public API ----

    @staticmethod
    def capture_env(experiment_id: str, *, run_dir: Path | None = None) -> Infra.Env:
        """Open a run: snapshot provenance and allocate a fresh ``run_dir``.

        Pass ``run_dir`` to pin an existing directory; otherwise a new
        ``run_NNNN`` is allocated under the outputs root.
        """
        if run_dir is None:
            run_dir = Infra.allocate_run_dir(Infra.outputs_root(), experiment_id)
        env = Infra.Env(
            experiment_id=experiment_id,
            run_dir=run_dir,
            git_rev=Infra._git(["rev-parse", "HEAD"]),
            git_dirty=bool(Infra._git(["status", "--porcelain"])),
            python=platform.python_version(),
            platform=platform.platform(),
            hostname=platform.node(),
        )
        logger.info(f"Allocated run `{experiment_id}/{run_dir.name}`.")
        logger.debug(
            "(Output dir=`%s`; Git rev=%s%s; Python=%s; platform=%s; hostname=%s)",
            run_dir,
            env.git_rev[:9],
            "(dirty)" if env.git_dirty else "",
            env.python,
            env.platform,
            env.hostname,
        )
        return env

    @staticmethod
    @contextmanager
    def run_with_timer(env: Infra.Env) -> Iterator[Infra.Timer]:
        """Time the run; on any failure, mark ``run_dir`` failed and re-raise."""
        timer = Infra.Timer(started=time.time())
        logger.debug("Starting run at %s", Infra._iso(timer.started))
        try:
            yield timer
        except BaseException as error:
            timer.finished = time.time()
            try:
                Infra.record_fail(env=env, timer=timer, error=error)
            except Exception:
                logger.exception("could not write failure record")
            raise  # re-raise
        else:
            timer.finished = time.time()
        finally:
            finished_at = Infra._iso(timer.finished) if timer.finished is not None else None
            logger.debug(
                "Finished run at %s (duration: %.3fs)", finished_at, timer.duration_s or 0.0
            )

    @staticmethod
    def record_fail(*, env: Infra.Env, timer: Infra.Timer, error: BaseException) -> Path:
        """Mark ``env.run_dir`` as a failed run: metadata with status + error.

        No ``result.json`` is written.
        """
        detail = f"{type(error).__name__}: {error}"
        Infra._write_json(
            env.run_dir / "metadata.json",
            Infra._metadata(env, timer, {"status": "failed", "error": detail}),
        )
        logger.error(
            "FAILED %s (%.3fs): %s -> %s",
            env.run_dir.name,
            timer.duration_s or 0.0,
            detail,
            env.run_dir,
        )
        return env.run_dir

    @staticmethod
    def record_success(
        *,
        env: Infra.Env,
        params: dict[str, Any],
        result: dict[str, Any],
        timer: Infra.Timer,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        """Write the three record files into ``env.run_dir`` (status=success).

        ``params.json`` may be pre-written (a runner can write it up front so
        child processes read the resolved params); it is written only when
        absent. ``result.json`` may already hold a mid-run partial flush, so it
        is overwritten with the final result.
        """
        run_dir = env.run_dir
        if not (run_dir / "params.json").exists():
            Infra._write_json(run_dir / "params.json", params)
        Infra._write_json(run_dir / "result.json", result, overwrite=True)
        Infra._write_json(
            run_dir / "metadata.json",
            Infra._metadata(env, timer, {"status": "success", **(extra_meta or {})}),
        )
        logger.info(
            "Done `%s/%s` (SUCCESS) in %.3fs -> %s",
            env.experiment_id,
            run_dir.name,
            timer.duration_s or 0.0,
            run_dir,
        )
        return run_dir

    # ---- run-dir / outputs helpers ----

    @staticmethod
    def outputs_root() -> Path:
        """Run-records root; override via ``$OUTPUTS_ROOT`` (else ``/tmp``)."""
        override = os.environ.get("OUTPUTS_ROOT")
        return Path(override) if override else DEFAULT_OUTPUTS_ROOT

    @staticmethod
    def allocate_run_dir(outputs_root: Path, experiment: str) -> Path:
        """Create and return the next ``<outputs_root>/<experiment>/run_NNNN``."""
        exp_dir = outputs_root / experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        indices = [
            int(p.name.removeprefix("run_"))
            for p in exp_dir.glob("run_*")
            if p.is_dir() and p.name.removeprefix("run_").isdigit()
        ]
        run_dir = exp_dir / f"run_{max(indices, default=0) + 1:04d}"
        run_dir.mkdir()
        return run_dir

    @staticmethod
    def write_json(path: Path, obj: dict[str, Any]) -> None:
        """Public JSON writer (overwrites).

        Runners use it to pre-write ``params.json`` (so per-cell subprocesses
        can read the resolved params) and to partial-flush ``result.json``
        mid-run.
        """
        Infra._write_json(path, obj, overwrite=True)

    @staticmethod
    def _write_json(path: Path, obj: dict[str, Any], overwrite: bool = False) -> None:
        if not overwrite and path.exists():
            raise FileExistsError(f"{path} already exists; not overwriting")
        path.write_text(json.dumps(obj))

    @staticmethod
    def _git(args: list[str]) -> str:
        out = subprocess.run(
            ["git", *args], cwd=Infra._main_script_dir(), capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    @staticmethod
    def _main_script_dir() -> Path | None:
        main_file = getattr(__main__, "__file__", None)
        if main_file is None:
            return None
        return Path(main_file).resolve().parent

    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=UTC).isoformat(timespec="seconds")

    @staticmethod
    def _metadata(
        env: Infra.Env, timer: Infra.Timer, extra_meta: dict[str, Any] | None
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "experiment": env.experiment_id,
            "git_rev": env.git_rev,
            "git_dirty": env.git_dirty,
            "started_at": Infra._iso(timer.started),
            "finished_at": Infra._iso(timer.finished) if timer.finished is not None else None,
            "duration_s": timer.duration_s,
            "python": env.python,
            "argv": list(sys.argv),
        }
        if extra_meta:
            meta.update(extra_meta)
        return meta
