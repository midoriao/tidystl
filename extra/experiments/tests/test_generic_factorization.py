"""Factorization invariant: generic(config_X) reproduces faithful backend X.

For each tool in the Core 6 and each spec-with-`tidystl` key in the baseline
and div registries, we assert that:

  GenericBackend(GenericConfig.from_dict(tool_configs[tool])).evaluate(phi, signal).robustness
    == evaluate(phi, signal, backend=tool).robustness    (to atol=1e-6)

Exceptions:
  - GAP: both generic and faithful raise -- consistent coverage gap, skipped.
  - DOCUMENTED_EXCEPTIONS: known out-of-axis divergences that are intentional.
    Each entry must have an inline comment explaining the reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tidystl_compat import GenericBackend, GenericConfig

from tidystl import Signal, evaluate, parse

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / "extra/experiments/registry"

# ---- documented out-of-axis exceptions -----------------------------------------------
#
# Format: set of (tool, registry_prefix::spec_name) tuples.
# Any (tool, spec) pair listed here is expected to MISMATCH and is not asserted equal.
# Remove an entry the moment the underlying quirk is fixed in the generic backend.

DOCUMENTED_EXCEPTIONS: set[tuple[str, str]] = {
    # breach / div_interp_sparse:
    #   Spec G[0.5,1.5] on signal with times=[0,2,4] -- the window [0.5,1.5] falls
    #   entirely between consecutive sample points.  The faithful BreachBackend uses PL
    #   interpolation at the WINDOW ENDPOINTS (samples strategy for predicates, interp
    #   strategy for window boundaries), which is a mixed behavior that the Core 6 axes
    #   do not separately parameterize.  The generic breach config
    #   (signal_model="pl_samples") falls back to midpoint interpolation for an
    #   otherwise-empty window, giving a different value.  This is the div_interp_sparse
    #   divergence case documented in specs_div.json and is an intentional coverage gap
    #   in the taxonomy (no "window-endpoint strategy" axis).
    ("breach", "registry::div_interp_sparse"),
    # breach / pc_window:
    #   Same out-of-axis quirk as div_interp_sparse, on the e1 implicit-choice
    #   profile's interpolation probe (spec G[0.5,1.5] on times=[0,2,4]): faithful
    #   breach reads the window-start PWL endpoint (x(0.5)=1.75) while the generic
    #   breach config falls back to window-midpoint interpolation (x(1.0)=4.5).
    #   Identical coverage gap (no "window-endpoint strategy" axis); this case
    #   enters the registry via the e1 profile cases (specs_div.json::pc_window).
    ("breach", "registry::pc_window"),
    # breach / div_window_sampling:
    #   Same out-of-axis quirk as div_interp_sparse/pc_window, on the E2 tool-vs-tool
    #   hero case: a buried G[1,2] window lies entirely between samples on a
    #   non-uniform grid, so faithful breach reads the interpolated window-start
    #   value (+2.0) while the generic breach config falls back to window-midpoint
    #   (+1.5). Identical coverage gap (no "window-endpoint strategy" axis); this
    #   case enters the registry via the E2 localization PR (specs_div.json).
    ("breach", "registry::div_window_sampling"),
}

# ---- helpers -----------------------------------------------------------------------

_TOOLS = ["native", "breach", "rtamt", "stlcgpp", "taliro", "pymtl"]
_ATOL = 1e-6


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _make_signal(entry: dict[str, Any]) -> Signal:
    times = np.asarray(entry["times"], dtype=np.float64)
    values: dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=np.float64) for k, v in entry["values"].items()
    }
    return Signal.from_dict(times, values)


def _registry_pairs(
    specs_path: Path, signals_path: Path, prefix: str
) -> list[tuple[str, str, str, Signal]]:
    """Return (tool, key, formula_text, signal) for each spec with 'tidystl' field."""
    specs: dict[str, Any] = _load_json(specs_path)
    signals: dict[str, Any] = _load_json(signals_path)
    tool_configs_raw: dict[str, Any] = _load_json(REGISTRY / "semantics_space.json")["tool_configs"]
    rows: list[tuple[str, str, str, Signal]] = []
    for spec_name, spec_entry in specs.items():
        if "tidystl" not in spec_entry:
            continue
        if spec_name not in signals:
            continue
        formula_text: str = spec_entry["tidystl"]
        signal = _make_signal(signals[spec_name])
        key = f"{prefix}::{spec_name}"
        for tool in _TOOLS:
            if tool not in tool_configs_raw:
                continue
            rows.append((tool, key, formula_text, signal))
    return rows


def _all_pairs() -> list[tuple[str, str, str, Signal]]:
    return _registry_pairs(
        REGISTRY / "specs_baseline.json",
        REGISTRY / "signals_baseline.json",
        "baseline",
    ) + _registry_pairs(
        REGISTRY / "specs_div.json",
        REGISTRY / "signals_div.json",
        "registry",
    )


# ---- parametrized test -------------------------------------------------------------

_PAIRS = _all_pairs()


@pytest.mark.parametrize(
    "tool,key,formula_text,signal",
    _PAIRS,
    ids=[f"{tool}/{key}" for tool, key, _, _ in _PAIRS],
)
def test_generic_reproduces_faithful(
    tool: str, key: str, formula_text: str, signal: Signal
) -> None:
    """generic(config_tool) and evaluate(..., backend=tool) agree on (formula, signal)."""
    tool_configs: dict[str, Any] = _load_json(REGISTRY / "semantics_space.json")["tool_configs"]

    phi = parse(formula_text)
    cfg = GenericConfig.from_dict(tool_configs[tool])
    backend = GenericBackend(cfg)

    # --- evaluate both sides, catching exceptions ---
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

    # --- classify ---
    pair = (tool, key)

    if got_exc is not None and want_exc is not None:
        # Consistent coverage gap: both raise -- skip (do not fail).
        pytest.skip(
            f"GAP: both generic and faithful raise for ({tool}, {key}): "
            f"generic={type(got_exc).__name__}, faithful={type(want_exc).__name__}"
        )

    if got_exc is not None and want_exc is None:
        pytest.fail(
            f"GAP_INCONSISTENT: generic raises {type(got_exc).__name__} "
            f"but faithful succeeded for ({tool}, {key}): {got_exc!r}"
        )

    if got_exc is None and want_exc is not None:
        pytest.fail(
            f"GAP_INCONSISTENT: faithful raises {type(want_exc).__name__} "
            f"but generic succeeded for ({tool}, {key}): {want_exc!r}"
        )

    # Both succeeded.
    assert got_arr is not None
    assert want_arr is not None

    if pair in DOCUMENTED_EXCEPTIONS:
        # Known out-of-axis quirk: verify it still mismatches (if it ever agrees,
        # the entry should be removed from DOCUMENTED_EXCEPTIONS).
        if np.allclose(got_arr, want_arr, atol=_ATOL, equal_nan=True):
            pytest.fail(
                f"DOCUMENTED_EXCEPTION no longer mismatches -- "
                f"remove ({tool!r}, {key!r}) from DOCUMENTED_EXCEPTIONS: "
                f"got={got_arr.flatten().tolist()}, want={want_arr.flatten().tolist()}"
            )
        # Expected mismatch -- pass silently (xfail would suppress the report; skip
        # gives a clear "expected divergence" entry in the run log).
        pytest.skip(
            f"DOCUMENTED_EXCEPTION: ({tool}, {key}) is a known out-of-axis quirk "
            f"(see DOCUMENTED_EXCEPTIONS comment for reason)"
        )

    assert np.allclose(got_arr, want_arr, atol=_ATOL, equal_nan=True), (
        f"generic(config_{tool}) != faithful {tool} on {key}:\n"
        f"  got  = {got_arr.flatten().tolist()}\n"
        f"  want = {want_arr.flatten().tolist()}\n"
        f"  max_diff = {np.max(np.abs(got_arr - want_arr)):.4g}"
    )
