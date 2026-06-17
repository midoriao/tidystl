import numpy as np
import pytest

from tidystl import NativeBackend, Signal, TorchSignal, evaluate, parse, robustness
from tidystl.backends.helper import BaseResult
from tidystl.core.nodes import Node
from tidystl.frontend.evaluator import resolve_backend
from tidystl_compat import (
    BreachBackend,
    RtamtBackend,
    StlcgppBackend,
    StlcgppTorchBackend,
)


def _signal_xy(x: list[float], y: list[float]) -> Signal:
    return Signal.from_dict(
        times=np.arange(len(x), dtype=float),
        values={
            "x": np.array([x], dtype=float),
            "y": np.array([y], dtype=float),
        },
    )


def _pred_xy_gt(name: str, left: str, right: str) -> Node:
    return Node(
        kind="predicate",
        attrs={
            "name": name,
            "op": ">",
            "left": Node(kind="var", attrs={"name": left}),
            "right": Node(kind="var", attrs={"name": right}),
        },
    )


def test_breach_backend_can_be_resolved_by_name() -> None:
    assert isinstance(resolve_backend("breach"), BreachBackend)


def test_rtamt_backend_can_be_resolved_by_name() -> None:
    assert isinstance(resolve_backend("rtamt"), RtamtBackend)


def test_stlcgpp_backend_can_be_resolved_by_name() -> None:
    assert isinstance(resolve_backend("stlcgpp"), StlcgppBackend)


def test_stlcgpp_torch_backend_can_be_resolved_by_name() -> None:
    assert isinstance(resolve_backend("stlcgpp_torch"), StlcgppTorchBackend)


def test_backend_instance_can_be_passed_directly() -> None:
    sig = _signal_xy([2.0, -1.0], [3.0, 4.0])
    formula = Node(
        kind="and",
        children=(_pred_xy_gt("px", "x", "y"), _pred_xy_gt("py", "y", "x")),
    )
    rho = robustness(formula, sig, backend=BreachBackend())
    np.testing.assert_allclose(rho[0], [-1.0, -1.0])


def test_evaluate_returns_trace_capable_result_for_breach_backend() -> None:
    sig = _signal_xy([2.0, -1.0], [3.0, 4.0])
    formula = _pred_xy_gt("px", "x", "y")
    result = evaluate(formula, sig, backend="breach")
    assert result.has_trace
    np.testing.assert_allclose(result.trace_for(formula)[0], [-1.0, -5.0])


def test_evaluate_returns_trace_capable_result_for_rtamt_backend() -> None:
    sig = _signal_xy([2.0, -1.0], [3.0, 4.0])
    formula = _pred_xy_gt("px", "x", "y")
    result = evaluate(formula, sig, backend="rtamt")
    assert result.has_trace
    np.testing.assert_allclose(result.trace_for(formula)[0], [-1.0, -5.0])


def test_breach_extends_penultimate_for_top_level_and() -> None:
    formula = parse("[predicates]\np: x >= 0\nq: y >= 0\n[stl]\np and q")
    sig = Signal.from_dict(
        times=np.array([0.0, 1.0, 2.0]),
        values={"x": np.array([[5.0, 5.0, -3.0]]), "y": np.array([[5.0, 5.0, -3.0]])},
    )
    rho = robustness(formula, sig, backend=BreachBackend())
    np.testing.assert_allclose(rho[0, -1], 5.0)


def test_native_and_breach_agree_on_temporal_operators() -> None:
    sig = Signal.from_dict(
        times=np.arange(5, dtype=float),
        values={"x": np.array([[3.0, 1.0, -1.0, 2.0, 4.0]])},
    )
    formula = Node(
        kind="always",
        children=(
            Node(
                kind="predicate",
                attrs={
                    "name": "p",
                    "op": ">=",
                    "left": Node(kind="var", attrs={"name": "x"}),
                    "right": Node(kind="const", attrs={"value": 0.0}),
                },
            ),
        ),
        attrs={"interval": (0.0, 2.0)},
    )
    np.testing.assert_allclose(
        robustness(formula, sig, backend="native"),
        robustness(formula, sig, backend="breach"),
    )


def test_native_does_not_extend_last_for_top_level_and() -> None:
    sig = _signal_xy([2.0, 3.0, -1.0], [1.0, -1.0, 4.0])
    formula = Node(
        kind="and",
        children=(_pred_xy_gt("px", "x", "y"), _pred_xy_gt("py", "y", "x")),
    )
    rho_native = robustness(formula, sig, backend="native")
    rho_breach = robustness(formula, sig, backend="breach")
    assert not np.allclose(rho_native[0, -1], rho_breach[0, -1])
    np.testing.assert_allclose(rho_breach[0, -1], rho_breach[0, -2])


def test_all_backend_results_are_base_result() -> None:
    sig = Signal.from_dict(
        times=np.array([0.0, 1.0, 2.0]),
        values={"x": np.array([[1.0, -1.0, 1.0]])},
    )
    formula = Node(
        kind="predicate",
        attrs={
            "name": "p",
            "op": ">=",
            "left": Node(kind="var", attrs={"name": "x"}),
            "right": Node(kind="const", attrs={"value": 0.0}),
        },
    )
    for backend in ("native", "breach", "rtamt", "stlcgpp"):
        result = evaluate(formula, sig, backend=backend)
        assert isinstance(result, BaseResult), f"{backend} result is not a BaseResult"


def test_numpy_backends_reject_torch_signal() -> None:
    formula = _pred_xy_gt("p", "x", "y")
    fake = TorchSignal(
        values=np.zeros((1, 2, 3)),
        times=np.arange(3, dtype=float),
        labels={"x": 0, "y": 1},
    )
    for backend in (NativeBackend(), BreachBackend(), RtamtBackend(), StlcgppBackend()):
        with pytest.raises(TypeError, match=backend.name):
            backend.evaluate(formula, fake)


def test_torch_backend_rejects_numpy_signal() -> None:
    formula = _pred_xy_gt("p", "x", "y")
    sig = _signal_xy([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="TorchSignal"):
        StlcgppTorchBackend().evaluate(formula, sig)


def test_and_or_combination_breach_extends_penultimate() -> None:
    """Lifted from test_integration.py: Breach extends the final output at the
    outermost binary op."""
    t = np.arange(5, dtype=float)
    sig = Signal.from_dict(
        times=t,
        values={
            "x": np.array([[1.0, -1.0, 2.0, -2.0, 3.0]]),
            "y": np.array([[2.0, 3.0, -1.0, 1.0, -1.0]]),
            "z": np.array([[-1.0, -1.0, -1.0, 5.0, -1.0]]),
        },
    )
    phi = parse("((x >= 0) and (y >= 0)) or (z >= 0)")
    rho = robustness(phi, sig, backend="breach")
    np.testing.assert_allclose(rho[0], [1.0, -1.0, -1.0, 5.0, 5.0])


def test_list_backends_includes_registered_compat_names() -> None:
    from tidystl import list_backends

    available = list_backends()
    # native (core) plus the compat backends are all discoverable.
    assert "native" in available
    assert "breach" in available
    # Every backend class the compat package exposes is reflected in the
    # listing -- without hardcoding the name strings here.
    for backend_cls in (BreachBackend, RtamtBackend, StlcgppBackend, StlcgppTorchBackend):
        assert backend_cls().name in available
    # The listing is sorted.
    assert available == sorted(available)
