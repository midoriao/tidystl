"""E5b runner: evaluate one (spec, signal, backend) and record its per-timestep
robustness and Boolean verdict. The signal/backend sweep lives in ``batch.sh``,
comparison in ``aggregate.py``; parameters are frozen as ``RunParams`` defaults.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tyro

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

import tidystl_compat  # noqa: E402
from _lib.infra import Infra  # noqa: E402

from tidystl import Signal, parse, robustness, use  # noqa: E402

use(tidystl_compat)

EXPERIMENT = "e5b_verdict_flip"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalParams:
    """One-parameter signal family: a fixed prefix with a swept final sample."""

    times: list[float] = field(default_factory=lambda: [0.0, 1.0, 2.0])
    names: list[str] = field(default_factory=lambda: ["x"])
    """Signal variables; the same member trace is bound to each."""
    base_values: list[float] = field(default_factory=lambda: [-5.0, -1.0])
    """Fixed prefix samples, before the swept final sample."""
    s: float = 3.0
    """Swept final sample appended to ``base_values``."""


@dataclass(frozen=True)
class RunParams:
    """The condition for ONE run (frozen paper config; ``batch.sh`` sweeps
    ``signal.s`` and the backend pairs across the two families).
    """

    backend: str = "native"
    """Robustness backend to evaluate (native | breach | rtamt)."""
    spec: str = "F[2,4](x >= 0)"
    """STL specification under test."""
    verdict_t_index: int = 1
    """Timestep index at which the Boolean verdict is read."""
    signal: SignalParams = field(default_factory=SignalParams)


@dataclass
class RunResult:
    """The facts: per-timestep robustness and the Boolean verdict."""

    spec: str
    backend: str
    signal: dict[str, Any]
    values: list[float]
    robustness: list[float]
    verdict: bool
    verdict_t_index: int


def runner(params: RunParams) -> RunResult:
    sig = params.signal
    times = np.asarray(sig.times, dtype=float)
    member = np.array([*sig.base_values, float(sig.s)])
    signal = Signal.from_dict(times, dict.fromkeys(sig.names, member))
    rho = robustness(parse(params.spec), signal, backend=params.backend)[0]
    return RunResult(
        spec=params.spec,
        backend=params.backend,
        signal=asdict(sig),
        values=[float(v) for v in member],
        robustness=[float(v) for v in rho],
        verdict=bool(float(rho[params.verdict_t_index]) >= 0.0),
        verdict_t_index=params.verdict_t_index,
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
