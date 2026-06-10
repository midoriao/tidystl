from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from tidystl import evaluate
from tidystl.core.signal import Signal
from tidystl.frontend.parser import parse


def _signal(t: list[float], **vars: list[float]) -> Signal:
    times = np.array(t, dtype=float)
    values = cast(
        dict[str, NDArray[np.floating]],
        {k: np.array(v, dtype=float) for k, v in vars.items()},
    )
    return Signal.from_dict(times, values)


def test_evaluate_returns_evaluation_result():
    f = parse("""
    [predicates]
    p : x > 0
    [stl]
    p
    """)
    sig = _signal([0, 1, 2], x=[1.0, -1.0, 1.0])
    result = evaluate(f, sig)
    assert hasattr(result, "robustness")
    assert hasattr(result, "trace_for")


def test_eval_result_robustness_matches_robustness_fn():
    from tidystl import robustness

    f = parse("""
    [predicates]
    p : x > 0
    [stl]
    G[0,1](p)
    """)
    sig = _signal([0.0, 1.0, 2.0], x=[2.0, 3.0, 1.0])
    result = evaluate(f, sig)
    expected = robustness(f, sig)
    np.testing.assert_allclose(result.robustness, expected)


def test_trace_predicate():
    f = parse("""
    [predicates]
    p : x > 0
    [stl]
    G[0,1](p)
    """)
    sig = _signal([0.0, 1.0, 2.0], x=[2.0, 3.0, 1.0])
    result = evaluate(f, sig)
    (pred,) = f.children
    trace = result.trace_for(pred)
    assert trace.shape == (1, 3)
    np.testing.assert_allclose(trace[0], [2.0, 3.0, 1.0])


def test_trace_and_children():
    f = parse("""
    [predicates]
    px : x > 0
    py : y > 1
    [stl]
    px and py
    """)
    sig = _signal([0.0, 1.0], x=[2.0, -1.0], y=[3.0, 0.5])
    result = evaluate(f, sig)
    lhs = result.trace_for(f.children[0])
    rhs = result.trace_for(f.children[1])
    np.testing.assert_allclose(lhs[0], [2.0, -1.0])
    np.testing.assert_allclose(rhs[0], [2.0, -0.5])


def test_trace_unknown_node_raises():
    f = parse("""
    [predicates]
    p : x > 0
    [stl]
    p
    """)
    sig = _signal([0.0, 1.0], x=[1.0, 2.0])
    result = evaluate(f, sig)
    other = parse("""
    [predicates]
    q : y > 0
    [stl]
    q
    """)
    with pytest.raises(KeyError):
        result.trace_for(other)


def test_trace_is_exposed():
    f = parse("""
    [predicates]
    p : x > 0
    [stl]
    p
    """)
    sig = _signal([0.0, 1.0], x=[1.0, 2.0])
    result = evaluate(f, sig)
    assert result.has_trace


def test_traced_nodes_lists_observable_nodes():
    f = parse("""
    [predicates]
    p : x > 0
    [stl]
    G[0,1](p)
    """)
    sig = _signal([0.0, 1.0, 2.0], x=[1.0, 2.0, 3.0])
    result = evaluate(f, sig)
    nodes = list(result.traced_nodes())
    assert f in nodes
    assert f.children[0] in nodes


def test_public_api_exports():
    import tidystl

    assert hasattr(tidystl, "evaluate")
    assert hasattr(tidystl, "EvaluationResult")
    assert hasattr(tidystl, "horizon")
    assert hasattr(tidystl, "required_max_gap")
