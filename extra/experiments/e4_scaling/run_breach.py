"""E4 breach runner: time real Breach's scaling column (MATLAB-only).

A single ``matlab -batch`` runs the WHOLE sweep -- MATLAB startup is expensive
and Breach has no OOM hazard, so per-cell subprocesses (unlike stlcgpp) are the
wrong model. This renders a self-contained ``.m`` from ``run_e4_breach.m.tmpl``
into the run dir, runs it, then reads back the rows the MATLAB side wrote;
Python keeps owning the record via ``Infra``. The MATLAB script rewrites
``breach_rows.json`` after every cell, so a crash still leaves the finished
cells.

    uv run --only-group experiments python run_breach.py --breach-path ../breach
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "extra" / "experiments"))

import tyro  # noqa: E402
from _real import RunResult, make_result, run_real  # noqa: E402

EXPERIMENT = "e4_scaling"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunParams:
    """The condition: real Breach's scaling column. Exposed as CLI flags."""

    spec: str = "G[0,5]((x > 0) and F[0,2](y > 0))"
    """Canonical workload string; recorded into params.json for provenance only."""
    breach_spec: str = "alw_[0,5] ((x[t] > 0) and ev_[0,2] (y[t] > 0))"
    """Breach STL string equivalent of ``spec`` (timed by the MATLAB side)."""
    breach_path: str = "../breach"
    """Breach toolbox path for ``addpath(genpath(...)); InitBreach`` (sibling clone; see HANDOFF.md)."""
    signal_duration: float = 10.0
    seed: int = 0
    batch_grid: list[int] = field(default_factory=lambda: [1, 8])
    """Breach is single-trace; cost is linear in N, so it runs only on this subset."""
    timestep_grid: list[int] = field(default_factory=lambda: [101, 201, 501, 1001, 2001, 5001])
    repeats: int = 5
    """Adaptive: cells with warmup > slow_cell_s run once, else this many times."""
    slow_cell_s: float = 5.0


def _matlab_vec(values: list[int]) -> str:
    return "[" + " ".join(str(int(v)) for v in values) + "]"


def emit_matlab(params: RunParams) -> str:
    """Render the Breach sweep MATLAB script from the checked-in template.

    The full script lives as plain MATLAB in ``run_e4_breach.m.tmpl``; only the
    resolved sweep knobs are substituted here (the dry-run-testable seam,
    analogous to ``breach_handoff.emit()``). Mirrors the Breach conventions in
    ``extra/experiments/e1_divergence/breach_handoff.py`` (BreachTraceSystem +
    STL_Formula + STL_Eval, plus a write_meta block); ``rows_path`` /
    ``meta_path`` arrive as ``matlab -batch`` function arguments.
    """
    subs = {
        "@@SEED@@": str(params.seed),
        "@@DURATION@@": f"{params.signal_duration:g}",
        "@@TS@@": _matlab_vec(params.timestep_grid),
        "@@NS@@": _matlab_vec(params.batch_grid),
        "@@REPEATS@@": str(params.repeats),
        "@@SLOW_CELL_S@@": f"{params.slow_cell_s:g}",
        "@@FORMULA@@": params.breach_spec.replace("'", "''"),
    }
    template = (Path(__file__).resolve().parent / "run_e4_breach.m.tmpl").read_text()
    for key, value in subs.items():
        template = template.replace(key, value)
    return template


def _parse_meta(meta_path: Path) -> dict[str, str]:
    """Parse the ``key: value`` lines the MATLAB write_meta block writes."""
    meta: dict[str, str] = {}
    if not meta_path.exists():
        return meta
    for line in meta_path.read_text().splitlines():
        if ": " in line:
            key, val = line.split(": ", 1)
            meta[key.strip()] = val.strip()
    return meta


def runner(params: RunParams, run_dir: Path, params_path: Path) -> RunResult:
    """Time real Breach over the grid via ONE ``matlab -batch`` invocation.

    Generates the sweep ``.m`` into ``run_dir``, runs it under ``matlab`` (which
    runs the WHOLE sweep), then reads back the rows the MATLAB side wrote. The
    ``params_path`` (run dir's params.json) is unused -- Breach has no per-cell
    children to feed.
    """
    script_path = run_dir / "run_e4_breach.m"
    rows_path = run_dir / "breach_rows.json"
    meta_path = run_dir / "breach_meta.txt"
    script_path.write_text(emit_matlab(params))

    cmd = (
        f"cd('{REPO_ROOT}'); addpath(genpath('{params.breach_path}')); "
        f"InitBreach; addpath('{run_dir}'); "
        f"run_e4_breach('{rows_path}', '{meta_path}')"
    )
    proc = subprocess.run(["matlab", "-batch", cmd], capture_output=True, text=True)
    for line in proc.stdout.strip().splitlines():
        if line:
            logger.info("%s", line)
    if proc.returncode != 0 and proc.stderr.strip():
        logger.info("matlab stderr: %s", proc.stderr.strip()[-500:])

    rows: list[dict[str, Any]] = []
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
    if proc.returncode != 0 and not rows:
        raise RuntimeError(
            f"matlab exited {proc.returncode} with no rows: {proc.stderr.strip()[-300:]}"
        )

    bmeta = _parse_meta(meta_path)
    meta = {
        "version": bmeta.get("breach_commit", "unknown"),
        "matlab": bmeta.get("matlab", "unknown"),
        "breach_path": bmeta.get("breach_path", params.breach_path),
        "spec": params.breach_spec,
        "batching_model": "MATLAB loop over single-trace STL_Eval evaluations",
    }
    return make_result("breach-real", rows, meta, partial=proc.returncode != 0)


def main() -> None:
    params = tyro.cli(RunParams)
    logger.info("E4 scaling benchmark: tool=breach")
    run_real(
        EXPERIMENT,
        params,
        runner,
        lambda r: {"breach": r.info["version"], "matlab": r.info["matlab"]},
    )


if __name__ == "__main__":
    main()
