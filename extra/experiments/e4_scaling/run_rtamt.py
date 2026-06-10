"""E4 rtamt runner: time real RTAMT's scaling column.

RTAMT is single-trace, so a batch of N is a Python loop over N
``spec.evaluate()`` calls (timed as a whole, per-trace time reported).

Fairness: spec CONSTRUCTION is hoisted out of the timed region (the
``StlDiscreteTimeSpecification`` is built + parsed once) and the per-trace
dataset dicts are marshalled before the clock starts; only ``spec.evaluate()``
is timed.

    uv run --only-group experiments python run_rtamt.py
"""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "extra" / "experiments"))

import tyro  # noqa: E402
from _real import RunResult, make_result, real_row, run_real  # noqa: E402
from _timing import make_values, time_adaptive  # noqa: E402

EXPERIMENT = "e4_scaling"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunParams:
    """The condition: real RTAMT's scaling column. Exposed as CLI flags."""

    spec: str = "G[0,5]((x > 0) and F[0,2](y > 0))"
    """Canonical workload string; recorded into params.json for provenance only."""
    rtamt_spec: str = "always[0:5]((x > 0) and (eventually[0:2](y > 0)))"
    signal_duration: float = 10.0
    seed: int = 0
    batch_grid: list[int] = field(default_factory=lambda: [1, 8])
    """RTAMT is single-trace; cost is linear in N, so it runs only on this subset."""
    timestep_grid: list[int] = field(default_factory=lambda: [101, 201, 501, 1001, 2001, 5001])
    repeats: int = 5
    """Adaptive: cells with warmup > slow_cell_s run once, else this many times."""
    slow_cell_s: float = 5.0


def bench(params: RunParams) -> RunResult:
    import rtamt

    # Hoisted: one spec, parsed once.
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var("x", "float")
    spec.declare_var("y", "float")
    spec.spec = params.rtamt_spec
    spec.parse()

    rows: list[dict[str, Any]] = []
    for n in params.batch_grid:
        for t in params.timestep_grid:
            times, x, y = make_values(n, t, params.seed, params.signal_duration)
            # Hoisted: per-trace dataset dicts prepared outside the clock.
            datasets = [
                {"time": times.tolist(), "x": x[i].tolist(), "y": y[i].tolist()} for i in range(n)
            ]

            def run(ds: list[dict[str, Any]] = datasets) -> None:
                for dataset in ds:
                    spec.evaluate(dataset)

            mean_ms, std_ms, repeats = time_adaptive(run, params.repeats, params.slow_cell_s)
            rows.append(real_row("rtamt-real", n, t, mean_ms, std_ms, repeats))
            logger.info(
                "  N=%4d, T=%5d: %9.2f ms (%8.2f ms/trace, x%d)",
                n,
                t,
                mean_ms,
                mean_ms / n,
                repeats,
            )
    meta = {
        "version": importlib.metadata.version("rtamt"),
        "spec": params.rtamt_spec,
        "batching_model": "python loop over single-trace evaluations",
    }
    return make_result("rtamt-real", rows, meta, partial=False)


def runner(params: RunParams, run_dir: Path, params_path: Path) -> RunResult:
    """In-process sweep; the run dir is unused (no per-cell isolation needed)."""
    return bench(params)


def main() -> None:
    params = tyro.cli(RunParams)
    logger.info("E4 scaling benchmark: tool=rtamt")
    run_real(EXPERIMENT, params, runner, lambda r: {"rtamt": r.info["version"]})


if __name__ == "__main__":
    main()
