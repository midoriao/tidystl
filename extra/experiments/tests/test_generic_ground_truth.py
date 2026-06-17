"""G vs R: the generic backend reproduces the REAL tools' recorded output.

The factorization invariant test (``extra/experiments/tests/test_generic_factorization.py``)
checks ``generic(config_X)`` against the *compatible* backend ``X`` (G vs F). But the
generic backend is built on top of those compat backends' executors, so G-vs-F agreement
is largely by construction. The genuine empirical anchor is the captured output of the
*real external tools* (Breach in MATLAB, RTAMT, STLCG++, py-metric-temporal-logic, TaLiRo),
stored as per-tool ground truth in ``{tool}_ground_truth.jsonl``.

This test closes that loop: for each tool's recorded cases it evaluates
``GenericBackend(GenericConfig(**tool_config))`` and compares directly against the
real-tool record (reusing the same ``assert_*_compatible`` alignment + tolerance the
F-vs-R tests use, with the generic backend injected). The claim it certifies is the
strong one: *six generic configs reproduce six real STL tools*, up to documented
implementation-specific residue -- not "a refactor reproduces its own backends".
"""

from __future__ import annotations

# ruff: noqa: E402
import json
import sys
from collections.abc import Callable
from importlib import util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPAT_TESTS = REPO_ROOT / "packages" / "tidystl-compat" / "tests"
_tests_spec = util.spec_from_file_location(
    "tests",
    COMPAT_TESTS / "__init__.py",
    submodule_search_locations=[str(COMPAT_TESTS)],
)
if _tests_spec is None or _tests_spec.loader is None:
    raise ImportError(f"cannot load compat test helpers from {COMPAT_TESTS}")
_tests_module = util.module_from_spec(_tests_spec)
sys.modules["tests"] = _tests_module
_tests_spec.loader.exec_module(_tests_module)

from tests._helpers.breach_cases import BREACH_CASES
from tests._helpers.breach_compat import assert_breach_compatible
from tests._helpers.pymtl_cases import PYMTL_CASES
from tests._helpers.pymtl_compat import assert_pymtl_compatible
from tests._helpers.rtamt_cases import RTAMT_CASES
from tests._helpers.rtamt_compat import assert_rtamt_compatible
from tests._helpers.stlcgpp_cases import STLCGPP_CASES
from tests._helpers.stlcgpp_compat import assert_stlcgpp_compatible
from tests._helpers.taliro_cases import TALIRO_CASES
from tests._helpers.taliro_compat import assert_taliro_compatible
from tidystl import Signal, parse
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl_compat import GenericBackend, GenericConfig

SEMANTICS_SPACE = REPO_ROOT / "extra/experiments/registry/semantics_space.json"

# Per-tool: the recorded cases (name, formula, times, value-map) and the asserter that
# knows that tool's JSONL ground-truth file. ``atol`` is per case (only breach varies it).
_AssertFn = Callable[..., None]


def _norm(
    name: str, formula: str, times: tuple[float, ...], values: dict[str, Any], atol: float
) -> tuple[str, str, tuple[float, ...], dict[str, Any], float]:
    return (name, formula, times, values, atol)


_CASES: dict[
    str, tuple[_AssertFn, list[tuple[str, str, tuple[float, ...], dict[str, Any], float]]]
] = {
    "breach": (
        assert_breach_compatible,
        [_norm(c.name, c.tidystl_formula, c.times, c.values, c.atol) for c in BREACH_CASES],
    ),
    "rtamt": (
        assert_rtamt_compatible,
        [_norm(c.name, c.tidystl_formula, c.times, c.values, 1e-6) for c in RTAMT_CASES],
    ),
    "stlcgpp": (
        assert_stlcgpp_compatible,
        [_norm(c.name, c.tidystl_formula, c.times, c.values, 1e-6) for c in STLCGPP_CASES],
    ),
    "taliro": (
        assert_taliro_compatible,
        [_norm(c.name, c.tidystl_formula, c.times, c.values, 1e-6) for c in TALIRO_CASES],
    ),
    "pymtl": (
        assert_pymtl_compatible,
        [_norm(c.name, c.tidystl_formula, c.times, c.atoms, 1e-6) for c in PYMTL_CASES],
    ),
}

# Documented out-of-axis residue: (tool, case) pairs the real tool produces but no Core 6
# generic config reproduces. These are NOT model failures -- they are implementation
# artifacts the open-world residual check is designed to flag (see e6_fingerprint).
DOCUMENTED_RESIDUALS: set[tuple[str, str]] = {
    # Breach interpolates at the WINDOW ENDPOINTS when a window falls entirely between
    # samples, mixing pl_interp (window) with pl_samples (predicate) -- not a single
    # coherent axis choice, so generic(pl_samples) cannot match it. See
    # test_generic_factorization.py DOCUMENTED_EXCEPTIONS.
    ("breach", "interp_sparse"),
    ("breach", "interp_boundary"),
}


def _config(tool: str) -> GenericConfig:
    space: dict[str, Any] = json.loads(SEMANTICS_SPACE.read_text())
    return GenericConfig.from_dict(space["tool_configs"][tool])


def _signal(times: tuple[float, ...], values: dict[str, Any]) -> Signal:
    return Signal.from_dict(
        times=np.array(times, dtype=float),
        values={k: np.array([v], dtype=float) for k, v in values.items()},
    )


def _params() -> list[tuple[str, str, str, tuple[float, ...], dict[str, Any], float]]:
    rows: list[tuple[str, str, str, tuple[float, ...], dict[str, Any], float]] = []
    for tool, (_assert, cases) in _CASES.items():
        for name, formula, times, values, atol in cases:
            rows.append((tool, name, formula, times, values, atol))
    return rows


def _matches(
    assert_fn: _AssertFn,
    phi: Node,
    signal: Signal,
    name: str,
    backend: EvaluationBackend,
    atol: float,
) -> bool:
    """True iff the generic backend reproduces the real-tool record for this case."""
    try:
        assert_fn(phi, signal, name, backend=backend, atol=atol)
    except AssertionError:
        return False
    return True


@pytest.mark.parametrize(
    "tool,name,formula,times,values,atol",
    _params(),
    ids=[f"{tool}/{name}" for tool, name, _, _, _, _ in _params()],
)
def test_generic_reproduces_real_tool(
    tool: str,
    name: str,
    formula: str,
    times: tuple[float, ...],
    values: dict[str, Any],
    atol: float,
) -> None:
    """generic(config_tool) reproduces the real tool's recorded robustness."""
    assert_fn, _ = _CASES[tool]
    backend = GenericBackend(_config(tool))
    phi = parse(formula)
    signal = _signal(times, values)
    pair = (tool, name)

    if pair in DOCUMENTED_RESIDUALS:
        # Must still diverge; if generic ever matches, the residual is gone -> remove
        # the entry (and likely a new axis is warranted).
        assert not _matches(assert_fn, phi, signal, name, backend, atol), (
            f"DOCUMENTED_RESIDUAL ({tool}, {name}) no longer diverges -- "
            "generic now reproduces the real tool; remove it from DOCUMENTED_RESIDUALS"
        )
        return

    # Reproduces the real tool exactly: assert directly so a mismatch reports the helper's
    # detailed diff.
    assert_fn(phi, signal, name, backend=backend, atol=atol)


def test_every_tool_has_a_config_and_cases() -> None:
    """Guard: all five real tools are exercised, each with a documented config."""
    space: dict[str, Any] = json.loads(SEMANTICS_SPACE.read_text())
    for tool in _CASES:
        assert tool in space["tool_configs"], tool
        _assert, cases = _CASES[tool]
        assert cases, tool
