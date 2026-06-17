"""E4 backend runner: time tidystl backends over the grid of batch sizes and timesteps.
See README.md in this directory for details and usage.
"""

from __future__ import annotations

import itertools
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "extra" / "experiments"))

import tidystl_compat  # noqa: E402
import tidystl  # noqa: E402
from _lib.infra import Infra  # noqa: E402
from _timing import make_values, time_with_repeat  # noqa: E402

tidystl.use(tidystl_compat)

EXPERIMENT = "e4_scaling"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunParams:
    """The condition parameters for single run."""

    backend: str = "native"
    spec: str = "G[0,5]((x > 0) and F[0,2](y > 0))"
    signal_duration: float = 10.0
    seed: int = 0
    batch_grid: list[int] = field(default_factory=lambda: [1, 8, 64, 256])
    timestep_grid: list[int] = field(default_factory=lambda: [101, 201, 501, 1001, 2001, 5001])
    repeats: int = 10
    """Per-cell timing repeats."""

    def __str__(self) -> str:
        return (
            f"backend={self.backend}, spec='{self.spec}', signal_duration={self.signal_duration}, "
            f"seed={self.seed}, batch_grid={self.batch_grid}, timestep_grid={self.timestep_grid}, "
            f"repeats={self.repeats}"
        )


@dataclass(frozen=True)
class RunResult:
    """The facts recorded from the run."""

    backend: str
    spec: str
    repeats: int
    results: list[CaseResult]


@dataclass(frozen=True)
class CaseResult:
    backend: str
    batch_size: int
    timesteps: int
    mean_ms: float
    std_ms: float


# ---- benchmark ----


def runner(params: RunParams) -> RunResult:
    """Time one tidystl backend over the grid."""
    if params.backend == "tidystl_simd":
        try:
            import tidystl_simd
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise SystemExit(
                "tidystl_simd backend selected but the extension is not available; "
                "run under the experiments group so uv builds it: "
                "`uv run --group experiments python run_backend.py --backend tidystl_simd` "
                "(a Rust toolchain is required to compile it)"
            ) from exc
        tidystl.use(tidystl_simd)

    total_cases = len(params.batch_grid) * len(params.timestep_grid)
    backend = params.backend
    phi = tidystl.parse(params.spec)
    repeats = params.repeats
    rows: list[CaseResult] = []

    logger.info("Backend=%s (%d cases)", backend, total_cases)

    for run_idx, (n, t) in enumerate(
        itertools.product(params.batch_grid, params.timestep_grid),
        start=1,
    ):
        times, x, y = make_values(n, t, params.seed, params.signal_duration)
        signal = tidystl.Signal.from_dict(times=times, values={"x": x, "y": y})

        def run(sig: Any = signal, be: str = backend) -> None:
            tidystl.robustness(phi, sig, backend=be)

        mean_ms, std_ms = time_with_repeat(run, repeats)
        rows.append(
            CaseResult(
                backend=backend,
                batch_size=n,
                timesteps=t,
                mean_ms=round(mean_ms, 4),
                std_ms=round(std_ms, 4),
            )
        )
        logger.info(
            "  [%d/%d] N=%4d, T=%5d: %7.2f ± %.2f ms",
            run_idx,
            total_cases,
            n,
            t,
            mean_ms,
            std_ms,
        )

    return RunResult(backend=backend, spec=params.spec, repeats=repeats, results=rows)


def main() -> None:
    params = tyro.cli(RunParams)
    env = Infra.capture_env(EXPERIMENT)

    logger.info("Params: " + str(params))

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
