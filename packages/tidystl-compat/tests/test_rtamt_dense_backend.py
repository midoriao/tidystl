"""RtamtDenseBackend reproduces RTAMT's dense-time (PWC) robustness.

Each shared case is evaluated by the backend and compared, on RTAMT's emitted
domain, against the dense ground truth via PWC sampling. A few focused tests pin
behaviors that distinguish the dense backend from the discrete one: acceptance
of non-uniform grids and the dense ``until`` semantics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.rtamt_cases import RTAMT_CASES, RtamtCase
from tests._helpers.rtamt_dense_compat import assert_rtamt_dense_compatible
from tidystl import Signal, parse
from tidystl.core.nodes import Node
from tidystl_compat import RtamtDenseBackend

GROUND_TRUTH = Path(__file__).parent / "rtamt_dense_ground_truth.jsonl"


def _signal(case: RtamtCase) -> Signal:
    return Signal.from_dict(
        times=np.array(case.times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in case.values.items()},
    )


def _const(v: float) -> Node:
    return Node(kind="const", attrs={"value": v})


def _var(name: str) -> Node:
    return Node(kind="var", attrs={"name": name})


def _pred(name: str, op: str, left: Node, right: Node) -> Node:
    return Node(kind="predicate", attrs={"name": name, "op": op, "left": left, "right": right})


@pytest.mark.parametrize("case", RTAMT_CASES, ids=lambda case: case.name)
def test_rtamt_dense_ground_truth(case: RtamtCase) -> None:
    if not GROUND_TRUTH.exists():
        pytest.skip(
            "Run extra/other_tools/rtamt/generate_ground_truth.py first "
            "(rtamt_dense_ground_truth.jsonl missing)"
        )
    formula = parse(case.tidystl_formula)
    signal = _signal(case)
    assert_rtamt_dense_compatible(formula, signal, case.name)


def test_rtamt_dense_accepts_non_uniform_grid() -> None:
    # The discrete RtamtBackend rejects non-uniform grids; the dense backend,
    # working over real-time PWC windows, accepts them.
    sig = Signal.from_dict(
        times=np.array([0.0, 1.0, 3.0]),
        values={"x": np.array([[5.0, -2.0, 4.0]])},
    )
    pred = _pred("p", ">=", _var("x"), _const(0.0))
    phi = Node(kind="always", children=(pred,), attrs={"interval": (0.0, 1.0)})

    rho = robustness_dense(phi, sig)
    # G[0,1] at t=0 reduces the PWC predicate over [0,1]: min(5, -2) = -2.
    np.testing.assert_allclose(rho[0, 0], -2.0)


def robustness_dense(formula: Node, signal: Signal) -> np.ndarray:
    from tidystl import robustness

    return robustness(formula, signal, backend=RtamtDenseBackend())


def test_rtamt_dense_until_differs_from_discrete() -> None:
    # until_tight: dense `until` reports -1 at t=2 where discrete reports 1.
    sig = Signal.from_dict(
        times=np.arange(5, dtype=float),
        values={
            "x": np.array([[1.0, 1.0, 1.0, -1.0, -1.0]]),
            "y": np.array([[-1.0, -1.0, 2.0, 2.0, 2.0]]),
        },
    )
    p = _pred("p", ">=", _var("x"), _const(0.0))
    q = _pred("q", ">=", _var("y"), _const(0.0))
    phi = Node(kind="until", children=(p, q), attrs={"interval": (1.0, 2.0)})

    rho = robustness_dense(phi, sig)
    np.testing.assert_allclose(rho[0, 2], -1.0)
