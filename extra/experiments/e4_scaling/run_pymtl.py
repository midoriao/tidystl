"""E4 pymtl runner: time real py-metric-temporal-logic's scaling column.

py-metric-temporal-logic (``import mtl``) is single-trace, so a batch of N is a
Python loop over N ``phi(trace, ...)`` evaluations (timed as a whole, per-trace
time reported), like ``run_rtamt``. mtl resolves temporal windows on a ``dt``
pivot grid; we set ``dt`` to the trace's sample spacing
(``signal_duration / (T - 1)``) so its resolution matches the workload grid.

Fairness: the formula is PARSED once outside the timed region and the per-trace
``{name: [(time, value), ...]}`` dicts are marshalled before the clock starts;
only ``phi(trace, time=None, quantitative=True, dt=dt)`` is timed. mtl is slow
and its cost grows fast with ``1/dt``, so the adaptive timer runs the heavy
cells once.

    uv run --only-group experiments python run_pymtl.py
"""

from __future__ import annotations

import importlib.metadata
import logging
import signal
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "extra" / "experiments"))

import tyro  # noqa: E402
from _real import RunResult, make_result, real_row, run_real  # noqa: E402
from _timing import make_values  # noqa: E402

EXPERIMENT = "e4_scaling"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class CellTimeoutError(Exception):
    """A single cell evaluation exceeded the wall-clock cap."""


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Raise ``CellTimeoutError`` if the body runs longer than ``seconds``.

    Uses ``SIGALRM`` (main thread, Unix): py-MTL is pure Python, so the alarm
    interrupts it at the next bytecode boundary. ``seconds <= 0`` disables it.
    """
    if seconds <= 0:
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise CellTimeoutError

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def _measure_cell(
    run: Callable[[], object], repeats: int, slow_cell_s: float, timeout_s: float
) -> tuple[float, float, int]:
    """``time_adaptive`` with a per-evaluation wall-clock cap.

    One warmup (excluded), then ``repeats`` timed runs unless the warmup already
    exceeds ``slow_cell_s`` (then run once). Each individual evaluation is bound
    by ``timeout_s``; exceeding it raises ``CellTimeoutError`` so the caller records
    the cell as timed out rather than blocking on a tool that does not scale.
    """
    with _time_limit(timeout_s):
        t0 = time.perf_counter()
        run()  # warmup (excluded)
        warmup_s = time.perf_counter() - t0
    n_repeats = 1 if warmup_s > slow_cell_s else repeats
    timings: list[float] = []
    for _ in range(n_repeats):
        with _time_limit(timeout_s):
            t0 = time.perf_counter()
            run()
            timings.append((time.perf_counter() - t0) * 1e3)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return statistics.mean(timings), std, n_repeats


@dataclass(frozen=True)
class RunParams:
    """The condition: real py-MTL's scaling column. Exposed as CLI flags."""

    spec: str = "G[0,5]((x > 0) and F[0,2](y > 0))"
    """Canonical workload string; recorded into params.json for provenance only."""
    pymtl_spec: str = "G[0,5] (x & F[0,2] y)"
    """mtl string equivalent of ``spec`` (threshold-0 predicates are bare atoms)."""
    signal_duration: float = 10.0
    seed: int = 0
    batch_grid: list[int] = field(default_factory=lambda: [1, 8])
    """py-MTL is single-trace; cost is linear in N, so it runs only on this subset."""
    timestep_grid: list[int] = field(default_factory=lambda: [101, 201, 501, 1001, 2001, 5001])
    repeats: int = 5
    """Adaptive: cells with warmup > slow_cell_s run once, else this many times."""
    slow_cell_s: float = 5.0
    timeout_s: float = 10.0
    """Per-cell wall-clock cap: a cell whose single evaluation exceeds this is
    recorded as 'timeout' and skipped (py-MTL does not scale to large T). <=0 disables."""


def bench(params: RunParams) -> RunResult:
    import mtl

    # Hoisted: one formula, parsed once.
    phi = mtl.parse(params.pymtl_spec)

    rows: list[dict[str, Any]] = []
    for n in params.batch_grid:
        for t in params.timestep_grid:
            times, x, y = make_values(n, t, params.seed, params.signal_duration)
            dt = params.signal_duration / (t - 1)
            # Hoisted: per-trace (time, value) traces prepared outside the clock.
            traces = [
                {
                    "x": list(zip(times.tolist(), x[i].tolist(), strict=True)),
                    "y": list(zip(times.tolist(), y[i].tolist(), strict=True)),
                }
                for i in range(n)
            ]

            def run(trs: list[dict[str, Any]] = traces, d: float = dt) -> None:
                for trace in trs:
                    phi(trace, time=None, quantitative=True, dt=d)

            try:
                mean_ms, std_ms, repeats = _measure_cell(
                    run, params.repeats, params.slow_cell_s, params.timeout_s
                )
            except CellTimeoutError:
                rows.append(
                    {
                        "backend": "pymtl-real",
                        "batch_size": n,
                        "timesteps": t,
                        "status": "timeout",
                        "reason": f"single evaluation exceeded the {params.timeout_s:g}s cap",
                    }
                )
                logger.info("  N=%4d, T=%5d: TIMEOUT (> %gs)", n, t, params.timeout_s)
                continue
            rows.append(real_row("pymtl-real", n, t, mean_ms, std_ms, repeats))
            logger.info(
                "  N=%4d, T=%5d: %9.2f ms (%8.2f ms/trace, x%d)",
                n,
                t,
                mean_ms,
                mean_ms / n,
                repeats,
            )
    meta = {
        "version": importlib.metadata.version("metric-temporal-logic"),
        "spec": params.pymtl_spec,
        "dt_rule": "signal_duration / (T - 1) (sample spacing)",
        "batching_model": "python loop over single-trace evaluations",
    }
    return make_result("pymtl-real", rows, meta, partial=False)


def runner(params: RunParams, run_dir: Path, params_path: Path) -> RunResult:
    """In-process sweep; the run dir is unused (no per-cell isolation needed)."""
    return bench(params)


def main() -> None:
    params = tyro.cli(RunParams)
    logger.info("E4 scaling benchmark: tool=pymtl")
    run_real(EXPERIMENT, params, runner, lambda r: {"pymtl": r.info["version"]})


if __name__ == "__main__":
    main()
