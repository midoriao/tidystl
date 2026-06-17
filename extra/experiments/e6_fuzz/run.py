"""E6 runner: fuzz random (formula, signal) pairs across all registered
backends and write one run record. ``stlcgpp_torch`` is fed a ``TorchSignal``;
all other backends a ``Signal``. A backend that cannot evaluate a formula
abstains (recorded), so the matrix downstream excludes that pair for it.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import tidystl_compat  # noqa: E402

from tidystl import Node, Signal, TorchSignal, list_backends, robustness, use  # noqa: E402

use(tidystl_compat)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generator  # noqa: E402

from extra.experiments._lib.infra import Infra  # noqa: E402

EXPERIMENT = "e6_fuzz"
TORCH_BACKENDS = frozenset({"stlcgpp_torch"})
ABSTAIN_EXC = (NotImplementedError, ValueError, TypeError)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_signal(times: np.ndarray, values: np.ndarray, names: list[str]) -> Signal:
    """A ``Signal`` from ``(times, (n_vars, length) values, names)``."""
    return Signal.from_dict(
        times=times,
        values={name: values[i : i + 1, :] for i, name in enumerate(names)},
    )


def build_torch_signal(times: np.ndarray, values: np.ndarray, names: list[str]) -> TorchSignal:
    """A ``TorchSignal`` mirroring ``build_signal`` (torch float64 tensors)."""
    return TorchSignal.from_dict(
        times=times,
        values={
            name: torch.tensor(values[i : i + 1, :], dtype=torch.float64)
            for i, name in enumerate(names)
        },
    )


def evaluate_cell(formula: Node, sig: Signal, tsig: TorchSignal, backend: str) -> dict[str, Any]:
    """Evaluate one (formula, signal) on one backend.

    Returns one of:
      {"status": "ok", "scalar": float, "signal": [float, ...]}
      {"status": "abstain", "exc": str, "msg": str}   # expected unsupported
      {"status": "error", "exc": str, "msg": str}     # unexpected: a bug signal
    """
    signal = tsig if backend in TORCH_BACKENDS else sig
    try:
        result = robustness(formula, signal, backend=backend)
    except ABSTAIN_EXC as exc:
        return {"status": "abstain", "exc": type(exc).__name__, "msg": str(exc)[:200]}
    except Exception as exc:  # noqa: BLE001 -- unexpected types are recorded, not raised
        return {"status": "error", "exc": type(exc).__name__, "msg": str(exc)[:200]}
    array = result.detach().cpu().numpy() if hasattr(result, "detach") else np.asarray(result)
    array = np.asarray(array, dtype=float)
    return {
        "status": "ok",
        "scalar": float(array[0, 0]),
        "signal": [float(x) for x in array[0, :]],
    }


@dataclass(frozen=True)
class RunParams:
    """Sweep knobs (the tyro CLI surface). ``batch.sh`` varies ``seed``."""

    n_pairs: int = 500
    seed: int = 0
    max_depth: int = 3
    min_vars: int = 1
    max_vars: int = 3
    min_length: int = 8
    max_length: int = 32
    broad: bool = True
    regimes: tuple[str, ...] = generator.REGIMES
    atol: float = 1e-6
    rtol: float = 1e-6


@dataclass
class RunResult:
    """Raw facts: per-pair, per-backend cells, plus the comparison tolerances.

    ``aggregate.py`` is the only consumer; it pools ``pairs`` across runs.
    """

    backends: list[str]
    n_pairs: int
    atol: float
    rtol: float
    pairs: list[dict[str, Any]] = field(default_factory=list)


def runner(params: RunParams) -> RunResult:
    """Generate ``n_pairs`` (formula, signal) pairs and evaluate every backend.

    Each pair gets a fresh signal (regime cycles deterministically over
    ``params.regimes``) and a formula over that signal's variables, so the
    formula's variables always exist in the signal.
    """
    rng = np.random.default_rng(params.seed)
    backends = list_backends()
    pairs: list[dict[str, Any]] = []
    for index in range(params.n_pairs):
        regime = params.regimes[index % len(params.regimes)]
        n_vars = int(rng.integers(params.min_vars, params.max_vars + 1))
        length = int(rng.integers(params.min_length, params.max_length + 1))
        names = generator.var_names(n_vars)
        times, values = generator.gen_signal_arrays(rng, regime, n_vars, length)
        formula = generator.gen_formula(rng, names, params.max_depth, params.broad)
        sig = build_signal(times, values, names)
        tsig = build_torch_signal(times, values, names)
        pairs.append({
            "index": index,
            "regime": regime,
            "n_vars": n_vars,
            "length": length,
            "operators": sorted(generator.operators_in(formula)),
            "has_equality": generator.has_equality_predicate(formula),
            "cells": {b: evaluate_cell(formula, sig, tsig, b) for b in backends},
        })
    return RunResult(
        backends=backends,
        n_pairs=params.n_pairs,
        atol=params.atol,
        rtol=params.rtol,
        pairs=pairs,
    )


def _summarize(result: RunResult) -> str:
    """One-line per-backend ok/abstain/error tally for the console."""
    parts = []
    for backend in result.backends:
        counts = {"ok": 0, "abstain": 0, "error": 0}
        for pair in result.pairs:
            counts[pair["cells"][backend]["status"]] += 1
        parts.append(f"{backend}:{counts['ok']}/{counts['abstain']}a/{counts['error']}e")
    return "  ".join(parts)


def main() -> None:
    params = tyro.cli(RunParams)
    env = Infra.capture_env(EXPERIMENT)
    with Infra.run_with_timer(env) as timer:
        result = runner(params)
    Infra.record_success(
        env=env,
        params=asdict(params),
        result=asdict(result),
        timer=timer,
    )
    logger.info("e6_fuzz n_pairs=%d seed=%d", params.n_pairs, params.seed)
    logger.info(_summarize(result))


if __name__ == "__main__":
    main()
