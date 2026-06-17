"""E6 runner (exp 1 -- factorization): for one tool, verify that
``generic(config_tool)`` reproduces faithful ``tool`` over the registries.

Records a per-spec agreement table: for each spec with a `tidystl` field, whether
generic and faithful agree, and any error messages when they don't.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import tidystl_compat  # noqa: E402
from tidystl_compat import GenericBackend, GenericConfig  # noqa: E402

from extra.experiments._lib.infra import Infra  # noqa: E402
from tidystl import Signal, evaluate, parse, use  # noqa: E402

use(tidystl_compat)

EXPERIMENT = "e6_fingerprint"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TOOLS = ["native", "breach", "rtamt", "stlcgpp", "taliro", "pymtl"]


@dataclass(frozen=True)
class RunParams:
    """Run condition: one tool checked against the registries."""

    tool: str = "breach"
    specs_baseline: str = "extra/experiments/registry/specs_baseline.json"
    signals_baseline: str = "extra/experiments/registry/signals_baseline.json"
    specs_div: str = "extra/experiments/registry/specs_div.json"
    signals_div: str = "extra/experiments/registry/signals_div.json"
    atol: float = 1e-6


@dataclass
class SpecResult:
    """Factorization check for one (tool, spec) pair."""

    spec_name: str
    registry: str  # "baseline" or "div"
    agree: bool
    status: str  # "OK" | "GAP" | "MISMATCH" | "GAP_INCONSISTENT"
    generic_error: str | None = None
    faithful_error: str | None = None
    max_diff: float | None = None


@dataclass
class RunResult:
    """Factorization facts for one tool over both registries."""

    tool: str
    n_ok: int
    n_gap: int
    n_mismatch: int
    n_gap_inconsistent: int
    specs: list[SpecResult] = field(default_factory=list)  # type: ignore[assignment]


def _make_signal(entry: dict[str, Any]) -> Signal:
    times = np.asarray(entry["times"], dtype=np.float64)
    values: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64) for k, v in entry["values"].items()
    }
    return Signal.from_dict(times, values)


def _load_registry(specs_path: Path, signals_path: Path) -> list[tuple[str, str, dict[str, Any]]]:
    """Return (spec_name, formula_text, signal_entry) for specs with 'tidystl' key."""
    specs: dict[str, Any] = json.loads(specs_path.read_text())
    signals: dict[str, Any] = json.loads(signals_path.read_text())
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for name, spec_entry in specs.items():
        if "tidystl" not in spec_entry:
            continue
        if name not in signals:
            continue
        rows.append((name, spec_entry["tidystl"], signals[name]))
    return rows


def _check_pair(
    tool: str,
    formula_text: str,
    signal: Signal,
    tool_configs: dict[str, Any],
    atol: float,
) -> tuple[str | None, str | None, str, float | None]:
    """Return (generic_error, faithful_error, status, max_diff)."""
    cfg = GenericConfig.from_dict(tool_configs[tool])
    backend = GenericBackend(cfg)
    phi = parse(formula_text)

    got_arr: np.ndarray | None = None
    got_exc: Exception | None = None
    want_arr: np.ndarray | None = None
    want_exc: Exception | None = None

    try:
        got_arr = backend.evaluate(phi, signal).robustness
    except Exception as exc:
        got_exc = exc

    try:
        want_arr = evaluate(phi, signal, backend=tool).robustness
    except Exception as exc:
        want_exc = exc

    generic_error = f"{type(got_exc).__name__}: {got_exc}" if got_exc is not None else None
    faithful_error = f"{type(want_exc).__name__}: {want_exc}" if want_exc is not None else None

    if got_exc is not None and want_exc is not None:
        return generic_error, faithful_error, "GAP", None
    if got_exc is not None:
        return generic_error, faithful_error, "GAP_INCONSISTENT", None
    if want_exc is not None:
        return generic_error, faithful_error, "GAP_INCONSISTENT", None

    assert got_arr is not None and want_arr is not None
    if np.allclose(got_arr, want_arr, atol=atol, equal_nan=True):
        return None, None, "OK", 0.0
    else:
        max_diff = float(np.max(np.abs(got_arr - want_arr)))
        return None, None, "MISMATCH", max_diff


def runner(params: RunParams) -> RunResult:
    space: dict[str, Any] = json.loads(
        (REPO_ROOT / "extra/experiments/registry/semantics_space.json").read_text()
    )
    tool_configs: dict[str, Any] = space["tool_configs"]

    if params.tool not in tool_configs:
        raise ValueError(f"unknown tool {params.tool!r}; known: {', '.join(tool_configs)}")

    entries_baseline = _load_registry(
        REPO_ROOT / params.specs_baseline, REPO_ROOT / params.signals_baseline
    )
    entries_div = _load_registry(REPO_ROOT / params.specs_div, REPO_ROOT / params.signals_div)

    spec_results: list[SpecResult] = []

    for registry_label, entries in [("baseline", entries_baseline), ("div", entries_div)]:
        for spec_name, formula_text, signal_entry in entries:
            signal = _make_signal(signal_entry)
            generic_error, faithful_error, status, max_diff = _check_pair(
                params.tool, formula_text, signal, tool_configs, params.atol
            )
            spec_results.append(
                SpecResult(
                    spec_name=spec_name,
                    registry=registry_label,
                    agree=(status == "OK"),
                    status=status,
                    generic_error=generic_error,
                    faithful_error=faithful_error,
                    max_diff=max_diff,
                )
            )

    n_ok = sum(1 for r in spec_results if r.status == "OK")
    n_gap = sum(1 for r in spec_results if r.status == "GAP")
    n_mismatch = sum(1 for r in spec_results if r.status == "MISMATCH")
    n_gap_inconsistent = sum(1 for r in spec_results if r.status == "GAP_INCONSISTENT")

    return RunResult(
        tool=params.tool,
        n_ok=n_ok,
        n_gap=n_gap,
        n_mismatch=n_mismatch,
        n_gap_inconsistent=n_gap_inconsistent,
        specs=spec_results,
    )


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

    logger.info(
        "tool=%s  OK=%d  GAP=%d  MISMATCH=%d  GAP_INCONSISTENT=%d",
        result.tool,
        result.n_ok,
        result.n_gap,
        result.n_mismatch,
        result.n_gap_inconsistent,
    )
    if result.n_mismatch > 0 or result.n_gap_inconsistent > 0:
        for r in result.specs:
            if r.status in ("MISMATCH", "GAP_INCONSISTENT"):
                logger.warning(
                    "  %s %s::%s  generic=%s  faithful=%s  max_diff=%s",
                    r.status,
                    r.registry,
                    r.spec_name,
                    r.generic_error or "OK",
                    r.faithful_error or "OK",
                    r.max_diff,
                )


if __name__ == "__main__":
    main()
