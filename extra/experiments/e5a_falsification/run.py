"""E5a runner: one backend's seeded falsification trials.

The system is a closed-form linear speed-control model (no integrator) over a
bounded piecewise-constant throttle space; every falsification claim is
validated system-level (exact dense check), so no backend acts as a verdict
oracle. All parameters are frozen as ``RunParams`` defaults.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import tyro
from numpy.typing import NDArray

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

from _lib.infra import Infra  # noqa: E402

from tidystl import Signal, parse, robustness  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402

EXPERIMENT = "e5a_falsification"

logging.basicConfig(
    level=logging.INFO,
    format="\033[2m%(asctime)s\033[0m [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelParams:
    """Frozen speed-control model ``v' = K*u(t) - c*v`` and its throttle space."""

    t_end: float = 8.0
    dt: float = 0.5
    """Monitor sampling step (17 samples)."""
    k_gain: float = 2.0
    """Throttle gain."""
    drag: float = 0.15
    """Linear drag coefficient."""
    v_init: float = 0.0
    n_segments: int = 4
    """Piecewise-constant throttle -> n-D search space."""
    u_lo: float = 0.0
    u_hi: float = 1.0
    v_limit: float = 8.8
    violation_margin: float = 1e-3
    """System-level robust-violation margin."""


@dataclass(frozen=True)
class AlgorithmParams:
    """Frozen (1+1)-ES search constants (Gaussian mutation, random restarts)."""

    sigma_init: float = 0.25
    sigma_decay: float = 0.97
    """Multiplicative decay applied to sigma on each rejected mutation."""
    stall_limit: int = 20
    """Rejections before a uniform random restart."""


@dataclass(frozen=True)
class RunParams:
    """The condition for ONE run (frozen paper config; smoke overrides
    ``n_repetitions``/``eval_budget`` only).
    """

    backend: str = "native"
    """Robustness backend to evaluate (native | breach | rtamt)."""
    seed_base: int = 20260605
    """Trial ``i`` uses seed ``seed_base + i``."""
    n_repetitions: int = 50
    """Seeded falsification trials in this run."""
    eval_budget: int = 300
    """Max objective evaluations per trial."""
    spec: str = "(v <= 8.8) and (F[0.5,1.5](v <= 25))"
    """STL specification to falsify."""
    model: ModelParams = field(default_factory=ModelParams)
    algorithm: AlgorithmParams = field(default_factory=AlgorithmParams)


@dataclass
class Trial:
    """One seeded falsification run's outcome."""

    seed: int
    falsified: bool
    evals_to_falsification: int | None
    evals_used: int
    failed_claims: int
    best_objective: float


@dataclass
class RunResult:
    """The facts: falsifying rate, evals-to-falsification, per-trial outcomes."""

    backend: str
    n_trials: int
    n_falsified: int
    falsifying_rate: float
    median_evals: float | None
    total_failed_claims: int
    evals_to_falsification: list[int]
    trials: list[Trial]


def monitor_times(model: ModelParams) -> NDArray[np.floating]:
    return np.arange(0.0, model.t_end + model.dt / 2, model.dt)


def _v_segment(
    v0: float, u: float, elapsed: NDArray[np.floating], model: ModelParams
) -> NDArray[np.floating]:
    """Closed-form speed within one constant-throttle segment."""
    target = model.k_gain * u / model.drag
    return target + (v0 - target) * np.exp(-model.drag * elapsed)


def simulate(u: NDArray[np.floating], model: ModelParams) -> NDArray[np.floating]:
    """Exact speed trace sampled on the monitor grid for throttle vector u."""
    times = monitor_times(model)
    segment_len = model.t_end / model.n_segments
    v = np.empty_like(times)
    v0 = model.v_init
    for k in range(model.n_segments):
        t_start = k * segment_len
        mask = (times >= t_start - 1e-12) & (times <= t_start + segment_len + 1e-12)
        v[mask] = _v_segment(v0, float(u[k]), times[mask] - t_start, model)
        v0 = float(_v_segment(v0, float(u[k]), np.asarray([segment_len]), model)[0])
    return v


def true_sup_speed(u: NDArray[np.floating], model: ModelParams) -> float:
    """EXACT system-level sup of v over [0, t_end] (attained at a segment
    boundary, since v is monotone per segment; no backend involved).
    """
    segment_len = model.t_end / model.n_segments
    sup = model.v_init
    v0 = model.v_init
    for k in range(model.n_segments):
        v0 = float(_v_segment(v0, float(u[k]), np.asarray([segment_len]), model)[0])
        sup = max(sup, v0)
    return sup


def is_true_violation(u: NDArray[np.floating], model: ModelParams) -> bool:
    """System-level robust violation of the speed limit (the only
    falsifiable conjunct; v < 10 < 25 by construction)."""
    return true_sup_speed(u, model) > model.v_limit + model.violation_margin


def make_objective(
    phi: Node, backend: str, model: ModelParams
) -> Callable[[NDArray[np.floating]], float]:
    """Monitor-style objective: min over the reported robustness vector."""
    times = monitor_times(model)

    def objective(u: NDArray[np.floating]) -> float:
        signal = Signal.from_dict(times, {"v": simulate(u, model)})
        rho = robustness(phi, signal, backend=backend)
        return float(np.min(np.asarray(rho)[0]))

    return objective


def search_run(seed: int, phi: Node, params: RunParams) -> Trial:
    """One seeded (1+1)-ES falsification run; a claim (objective < 0) is
    validated system-level by ``is_true_violation`` before it stops the run.
    """
    model = params.model
    algo = params.algorithm
    rng = np.random.default_rng(seed)
    objective = make_objective(phi, params.backend, model)
    budget = params.eval_budget

    evals = 0
    failed_claims = 0
    found_at: int | None = None
    best_objective = float("inf")

    def evaluate_candidate(u: NDArray[np.floating]) -> float:
        nonlocal evals, failed_claims, found_at, best_objective
        evals += 1
        value = objective(u)
        best_objective = min(best_objective, value)
        if value < 0 and found_at is None:
            if is_true_violation(u, model):
                found_at = evals
            else:
                failed_claims += 1
        return value

    while evals < budget and found_at is None:
        x = rng.uniform(model.u_lo, model.u_hi, model.n_segments)
        fx = evaluate_candidate(x)
        sigma = algo.sigma_init
        stall = 0
        while evals < budget and found_at is None and stall < algo.stall_limit:
            y = np.clip(rng.normal(x, sigma), model.u_lo, model.u_hi)
            fy = evaluate_candidate(y)
            if fy < fx:
                x, fx = y, fy
                stall = 0
            else:
                stall += 1
                sigma *= algo.sigma_decay

    return Trial(
        seed=seed,
        falsified=found_at is not None,
        evals_to_falsification=found_at,
        evals_used=evals,
        failed_claims=failed_claims,
        best_objective=best_objective,
    )


def runner(params: RunParams) -> RunResult:
    phi = parse(params.spec)
    trials = [search_run(params.seed_base + i, phi, params) for i in range(params.n_repetitions)]
    evals_dist = sorted(
        t.evals_to_falsification
        for t in trials
        if t.falsified and t.evals_to_falsification is not None
    )
    return RunResult(
        backend=params.backend,
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
