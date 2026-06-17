"""CMA-ES and simulated-annealing falsification search producing a Trial.

Objective = formula robustness at t=0 under the chosen backend. A candidate
with objective < 0 is a claim; it is oracle-confirmed before it stops the run,
otherwise counted as a failed claim. Deterministic given seed.

Both optimizers assume specs are on-grid (signal dt == model dt), so no
regridding is needed.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import cma
import models
import numpy as np
import oracle
from numpy.typing import NDArray
from scipy.optimize import dual_annealing

from tidystl import Signal, robustness
from tidystl.core.nodes import Node


@dataclass
class Trial:
    seed: int
    falsified: bool
    evals_to_falsification: int | None
    evals_used: int
    failed_claims: int
    best_objective: float


def _objective(model: models.OdeModel, phi: Node, backend: str, u: NDArray, times: NDArray) -> float:
    y = models.simulate(model, u, times)
    signal = Signal.from_dict(times, {model.output_var: y})
    return float(np.asarray(robustness(phi, signal, backend=backend))[0, 0])


def cma_falsify(
    *,
    model: models.OdeModel,
    phi: Node,
    backend: str,
    seed: int,
    budget: int,
    dt: float,
) -> Trial:
    times = models.monitor_times(model, dt)
    lo, hi = model.u_lo, model.u_hi
    x0 = np.full(model.n_segments, 0.5 * (lo + hi))
    sigma0 = 0.25 * (hi - lo)
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma0,
        {"bounds": [lo, hi], "seed": seed, "maxfevals": budget, "verbose": -9},
    )

    evals = 0
    failed_claims = 0
    found_at: int | None = None
    best = float("inf")

    while not es.stop() and evals < budget and found_at is None:
        candidates = es.ask()
        values = []
        for x in candidates:
            xa = np.asarray(x)
            evals += 1
            f = _objective(model, phi, backend, xa, times)
            values.append(f)
            best = min(best, f)
            if f < 0 and found_at is None and oracle.is_true_violation(model, xa, phi, dt):
                found_at = evals
            elif f < 0 and found_at is None:
                failed_claims += 1
            if evals >= budget:
                break
        if len(values) == len(candidates):
            es.tell(candidates, values)

    return Trial(
        seed=seed,
        falsified=found_at is not None,
        evals_to_falsification=found_at,
        evals_used=evals,
        failed_claims=failed_claims,
        best_objective=best,
    )


class _StopSearchError(Exception):
    """Unwinds the annealing search: budget exhausted, or first validated hit."""


def _make_eval(model: models.OdeModel, phi: Node, backend: str, dt: float, state: dict):
    """Objective closure recording evals/claims/first oracle-confirmed hit.
    Raises _StopSearchError on budget exhaustion (hard cap) or once the first
    oracle-confirmed falsification is found (falsification stops at a counterexample).
    """
    times = models.monitor_times(model, dt)

    def objective(u):
        if state["evals"] >= state["budget"]:
            raise _StopSearchError
        u = np.asarray(u)
        state["evals"] += 1
        val = _objective(model, phi, backend, u, times)
        state["best"] = min(state["best"], val)
        if val < 0 and state["found_at"] is None:
            if oracle.is_true_violation(model, u, phi, dt):
                state["found_at"] = state["evals"]
                raise _StopSearchError  # stop at the first validated counterexample
            state["failed_claims"] += 1
        return val

    return objective


def anneal_falsify(*, model: models.OdeModel, phi: Node, backend: str, seed: int, budget: int, dt: float) -> Trial:
    state = {"evals": 0, "failed_claims": 0, "found_at": None,
             "best": float("inf"), "budget": budget}
    obj = _make_eval(model, phi, backend, dt, state)
    bounds = [(model.u_lo, model.u_hi)] * model.n_segments
    with contextlib.suppress(_StopSearchError):
        dual_annealing(obj, bounds, maxfun=budget, seed=seed, no_local_search=True)
    return Trial(seed=seed, falsified=state["found_at"] is not None,
                 evals_to_falsification=state["found_at"], evals_used=state["evals"],
                 failed_claims=state["failed_claims"], best_objective=state["best"])


def falsify(*, optimizer: str, model: models.OdeModel, phi: Node, backend: str, seed: int, budget: int, dt: float) -> Trial:
    if optimizer == "cma":
        return cma_falsify(model=model, phi=phi, backend=backend, seed=seed, budget=budget, dt=dt)
    if optimizer == "anneal":
        return anneal_falsify(model=model, phi=phi, backend=backend, seed=seed, budget=budget, dt=dt)
    raise ValueError(f"unknown optimizer {optimizer!r}")
