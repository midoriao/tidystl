"""E5a runner: one backend's seeded falsification trials.

Named benchmarks (``specs.BENCHMARKS``) are falsified with either CMA-ES or
simulated annealing (``search.falsify``). Every falsification claim is
validated system-level (exact dense check), so no backend acts as a verdict
oracle. All parameters are frozen as ``RunParams`` defaults.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tyro

_HERE = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent
for _p in (_HERE, _EXPERIMENTS):
    if _p.as_posix() not in sys.path:
        sys.path.insert(0, _p.as_posix())

import search  # noqa: E402
import specs  # noqa: E402
import tidystl_compat  # noqa: E402
from _lib.infra import Infra  # noqa: E402

from tidystl import parse, use  # noqa: E402

use(tidystl_compat)

EXPERIMENT = "e5a_falsification"

logging.basicConfig(
    level=logging.INFO,
    format="\033[2m%(asctime)s\033[0m [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunParams:
    """The condition for ONE run (frozen paper config; smoke overrides
    ``n_repetitions``/``eval_budget`` only).
    """

    backend: str = "breach"
    """Tool-compatible backend to evaluate (breach | rtamt | rtamt_dense | pymtl |
    taliro | stlcgpp). `native` is the oracle, not a condition."""
    benchmark: str = "m2_mass_spring"
    """Named benchmark from specs.BENCHMARKS (m2_mass_spring | m1_speed | m3_coupled)."""
    optimizer: str = "cma"
    """Falsification optimizer (cma | anneal)."""
    threshold: float | None = None
    """Spec difficulty threshold override; None uses the benchmark hero threshold."""
    dt: float | None = None
    """Monitor sampling step override; None uses the benchmark model dt."""
    seed_base: int = 20260605
    """Trial ``i`` uses seed ``seed_base + i``."""
    n_repetitions: int = 50
    """Seeded falsification trials in this run."""
    eval_budget: int = 300
    """Max objective evaluations per trial."""


@dataclass
class RunResult:
    """The facts: falsifying rate, evals-to-falsification, per-trial outcomes."""

    backend: str
    benchmark: str
    optimizer: str
    threshold: float
    dt: float
    n_trials: int
    n_falsified: int
    falsifying_rate: float
    median_evals: float | None
    total_failed_claims: int
    evals_to_falsification: list[int]
    trials: list[search.Trial]


def runner(params: RunParams) -> RunResult:
    bench = specs.BENCHMARKS[params.benchmark]
    dt = params.dt if params.dt is not None else bench.model.dt
    threshold = params.threshold if params.threshold is not None else bench.hero_threshold
    spec = bench.spec(threshold)
    specs.assert_on_grid(spec, dt)  # a dt override must keep the spec bounds aligned
    phi = parse(spec)

    trials = [
        search.falsify(
            optimizer=params.optimizer,
            model=bench.model,
            phi=phi,
            backend=params.backend,
            seed=params.seed_base + i,
            budget=params.eval_budget,
            dt=dt,
        )
        for i in range(params.n_repetitions)
    ]

    evals_dist = sorted(
        t.evals_to_falsification
        for t in trials
        if t.falsified and t.evals_to_falsification is not None
    )
    return RunResult(
        backend=params.backend,
        benchmark=params.benchmark,
        optimizer=params.optimizer,
        threshold=threshold,
        dt=dt,
        n_trials=len(trials),
        n_falsified=len(evals_dist),
        falsifying_rate=len(evals_dist) / len(trials),
        median_evals=float(np.median(evals_dist)) if evals_dist else None,
        total_failed_claims=sum(t.failed_claims for t in trials),
        evals_to_falsification=evals_dist,
        trials=trials,
    )


def main() -> None:
    params = tyro.cli(RunParams)
    env = Infra.capture_env(EXPERIMENT)

    logger.info("Params: %s", params)

    with Infra.run_with_timer(env) as timer:
        result = runner(params)

    Infra.record_success(
        env=env,
        params=asdict(params),
        result=asdict(result),
        timer=timer,
    )


if __name__ == "__main__":
    main()
