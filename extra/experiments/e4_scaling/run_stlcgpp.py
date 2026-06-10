"""E4 stlcgpp runner: time real STLCG++'s scaling column.

STLCG++ batches natively via ``torch.vmap``; hard mode (approx_method='true',
the package default), padding='last', float32.

Fairness: the formula object is hoisted out of the timed region (rebuilt per
timestep count, since its intervals are in samples) and the float32 tensor is
marshalled before the clock starts; only ``formula(signal)`` is timed.

Each cell runs in its own SUBPROCESS (the ``--cell`` protocol): the unfold-based
window materialization can exceed host memory and the OOM killer SIGKILLs
uncatchably, so isolation turns a kill into one recorded 'oom-killed' row. Cells
whose estimated peak exceeds ``memory_budget_bytes`` are skipped without
spawning; partial results flush after every cell (runs are not resumed). The
``--cell`` child is sniffed before tyro so it never imports tyro+rich -- it
reads params from the run dir's ``params.json`` and prints one JSON row.

    uv run --only-group experiments python run_stlcgpp.py
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "extra" / "experiments"))

from _lib.infra import Infra  # noqa: E402
from _real import RunResult, make_result, real_row, run_real  # noqa: E402
from _timing import make_values, time_adaptive  # noqa: E402

EXPERIMENT = "e4_scaling"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunParams:
    """The condition: real STLCG++'s scaling column. Exposed as CLI flags."""

    spec: str = "G[0,5]((x > 0) and F[0,2](y > 0))"
    """Canonical workload string; recorded into params.json for provenance only."""
    signal_duration: float = 10.0
    seed: int = 0
    batch_grid: list[int] = field(default_factory=lambda: [1, 8, 64, 256])
    timestep_grid: list[int] = field(default_factory=lambda: [101, 201, 501, 1001, 2001, 5001])
    repeats: int = 5
    """Adaptive: cells with warmup > slow_cell_s run once, else this many times."""
    slow_cell_s: float = 5.0
    memory_budget_bytes: int = 2_147_483_648
    """Skip cells whose raw unfold size N*T*(w_G+w_F)*4 exceeds this."""


def _params_from_json(path: Path) -> RunParams:
    """Rebuild RunParams from a record's params.json (child --cell path)."""
    return RunParams(**json.loads(path.read_text()))


def _estimated_bytes(n: int, t: int, step: float) -> int:
    windows = round(5 / step) + round(2 / step)
    return n * t * windows * 4


def stlcgpp_cell(n: int, t: int, params: RunParams) -> dict[str, Any]:
    """Time ONE STLCG++ cell (runs in a per-cell subprocess)."""
    import torch
    from stlcgpp.formula import Always, And, Eventually, Predicate

    # Hoisted: formula construction (intervals are in samples).
    step = params.signal_duration / (t - 1)
    x = Predicate("x", lambda s: s[:, 0])
    y = Predicate("y", lambda s: s[:, 1])
    phi = Always(
        And(x > 0.0, Eventually(y > 0.0, interval=[0, round(2 / step)])),
        interval=[0, round(5 / step)],
    )

    def single(sig: Any) -> Any:
        return phi(sig, padding="last", approx_method="true")

    _times, xv, yv = make_values(n, t, params.seed, params.signal_duration)
    signal = torch.tensor(np.stack([xv, yv], axis=2), dtype=torch.float32)
    fn: Callable[[], object] = (
        (lambda: single(signal[0])) if n == 1 else (lambda: torch.vmap(single)(signal))
    )
    try:
        mean_ms, std_ms, repeats = time_adaptive(fn, params.repeats, params.slow_cell_s)
        return real_row("stlcgpp-real", n, t, mean_ms, std_ms, repeats)
    except (RuntimeError, MemoryError) as exc:
        return {
            "backend": "stlcgpp-real",
            "batch_size": n,
            "timesteps": t,
            "status": "error",
            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


def _meta() -> dict[str, Any]:
    return {
        "version": importlib.metadata.version("stlcgpp"),
        "torch": importlib.metadata.version("torch"),
        "spec": "Always(And(x>0, Eventually(y>0, [0, 2/step])), [0, 5/step])",
        "approx_method": "true",
        "padding": "last",
        "dtype": "float32",
        "batching_model": "torch.vmap over the time-major formula; "
        "one subprocess per cell (OOM isolation)",
    }


def runner(params: RunParams, run_dir: Path, params_path: Path) -> RunResult:
    """Orchestrate STLCG++ cells, one SUBPROCESS each (OOM isolation).

    ``run_dir`` is a fresh dir; the partial result is flushed to
    ``run_dir/result.json`` after every cell, so an OOM-killed run still
    records the cells that finished (runs are not resumed). Each child reads
    the resolved params from ``params_path`` (the run dir's ``params.json``,
    written up front by ``run_real``).
    """
    meta = _meta()
    rows: list[dict[str, Any]] = []
    partial_path = run_dir / "result.json"

    for t in params.timestep_grid:
        step = params.signal_duration / (t - 1)
        for n in params.batch_grid:
            est = _estimated_bytes(n, t, step)
            budget = params.memory_budget_bytes
            if est > budget:
                rows.append(
                    {
                        "backend": "stlcgpp-real",
                        "batch_size": n,
                        "timesteps": t,
                        "status": "skipped",
                        "reason": f"estimated peak {est / 1024**3:.1f} GiB "
                        f"(unfold-based window materialization) exceeds the "
                        f"{budget / 1024**3:.0f} GiB budget",
                    }
                )
                logger.info("  N=%4d, T=%5d: SKIP (est %.1f GiB)", n, t, est / 1024**3)
            else:
                proc = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--cell",
                        str(n),
                        str(t),
                        "--params-json",
                        str(params_path),
                    ],
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    row = json.loads(proc.stdout.strip().splitlines()[-1])
                    rows.append(row)
                    if row["status"] == "ok":
                        logger.info(
                            "  N=%4d, T=%5d: %9.2f ms (%8.2f ms/trace, x%d)",
                            n,
                            t,
                            row["mean_ms"],
                            row["mean_ms"] / n,
                            row["repeats"],
                        )
                    else:
                        logger.info("  N=%4d, T=%5d: ERROR %s", n, t, row["reason"][:80])
                else:
                    reason = (
                        "OOM-killed (SIGKILL)"
                        if proc.returncode in (137, -9)
                        else f"subprocess exit {proc.returncode}: {proc.stderr[-200:]}"
                    )
                    rows.append(
                        {
                            "backend": "stlcgpp-real",
                            "batch_size": n,
                            "timesteps": t,
                            "status": "oom-killed" if proc.returncode in (137, -9) else "error",
                            "reason": reason,
                        }
                    )
                    logger.info("  N=%4d, T=%5d: %s", n, t, reason)
            Infra.write_json(
                partial_path, asdict(make_result("stlcgpp-real", rows, meta, partial=True))
            )
    return make_result("stlcgpp-real", rows, meta, partial=False)


def _run_cell_child(argv: list[str]) -> None:
    """Internal: time one stlcgpp cell and print its row as JSON, then exit.

    A MACHINE protocol -- sniffed before tyro so the child never imports tyro.
    Params come from the run dir's params.json (``--params-json``). The JSON row
    goes to stdout via ``print`` (the parent reads it); this path configures no
    logging, keeping stdout to the one row.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cell", nargs=2, type=int, required=True, metavar=("N", "T"))
    p.add_argument("--params-json", dest="params_json", type=Path, required=True)
    ns, _ = p.parse_known_args(argv[1:])
    params = _params_from_json(ns.params_json)
    print(json.dumps(stlcgpp_cell(ns.cell[0], ns.cell[1], params)))


def main() -> None:
    # Child --cell path: a machine protocol, handled before tyro so the
    # per-cell subprocesses never import tyro+rich. It prints its JSON row
    # to stdout and configures no logging (stdout stays the one row).
    if "--cell" in sys.argv:
        _run_cell_child(sys.argv)
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import tyro

    params = tyro.cli(RunParams)
    logger.info("E4 scaling benchmark: tool=stlcgpp")
    run_real(
        EXPERIMENT,
        params,
        runner,
        lambda r: {"stlcgpp": r.info["version"], "torch": r.info["torch"]},
    )


if __name__ == "__main__":
    main()
