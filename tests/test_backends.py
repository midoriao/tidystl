import numpy as np
import pytest

from tidystl import (
    BreachBackend,
    NativeBackend,
    RtamtBackend,
    Signal,
    StlcgppBackend,
    StlcgppTorchBackend,
    TorchSignal,
    evaluate,
    robustness,
)
from tidystl.backends.helper import BaseResult
from tidystl.core.nodes import Node
from tidystl.frontend.evaluator import resolve_backend


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


def test_default_backend_is_native() -> None:
    backend = resolve_backend(None)
    assert isinstance(backend, NativeBackend)


def test_native_backend_can_be_resolved_by_name() -> None:
    backend = resolve_backend("native")
    assert isinstance(backend, NativeBackend)


def test_breach_backend_can_be_resolved_by_name() -> None:
    backend = resolve_backend("breach")
    assert isinstance(backend, BreachBackend)


def test_rtamt_backend_can_be_resolved_by_name() -> None:
    backend = resolve_backend("rtamt")
    assert isinstance(backend, RtamtBackend)


def test_stlcgpp_backend_can_be_resolved_by_name() -> None:
    backend = resolve_backend("stlcgpp")
    assert isinstance(backend, StlcgppBackend)


def test_stlcgpp_torch_backend_can_be_resolved_by_name() -> None:
    backend = resolve_backend("stlcgpp_torch")
    assert isinstance(backend, StlcgppTorchBackend)


def test_backend_instance_can_be_passed_directly() -> None:
    sig = _signal_xy([2.0, -1.0], [3.0, 4.0])
    formula = Node(
        kind="and",
        children=(
            _pred_xy_gt("px", "x", "y"),
            _pred_xy_gt("py", "y", "x"),
        ),
    )

    rho = robustness(formula, sig, backend=BreachBackend())
    np.testing.assert_allclose(rho[0], [-1.0, -1.0])


def test_evaluate_returns_trace_capable_result_for_native_backend() -> None:
    sig = _signal_xy([2.0, -1.0], [3.0, 4.0])
    formula = _pred_xy_gt("px", "x", "y")

    result = evaluate(formula, sig, backend="native")
    assert result.has_trace
    np.testing.assert_allclose(result.trace_for(formula)[0], [-1.0, -5.0])


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
    """Breach copies the penultimate robustness to the final timestep for
    top-level AND/OR formulas, masking a dip at the final sample."""
    from tidystl.frontend.parser import parse

    formula = parse("[predicates]\np: x >= 0\nq: y >= 0\n[stl]\np and q")
    # x=y dip sharply at the final point; penultimate value is strongly positive.
    sig = Signal.from_dict(
        times=np.array([0.0, 1.0, 2.0]),
        values={
            "x": np.array([[5.0, 5.0, -3.0]]),
            "y": np.array([[5.0, 5.0, -3.0]]),
        },
    )

    rho = robustness(formula, sig, backend=BreachBackend())
    # Final timestep is overwritten with penultimate value: min(5,5)=5.
    np.testing.assert_allclose(rho[0, -1], 5.0)


def test_native_and_breach_agree_on_temporal_operators() -> None:
    """G and F operators give the same result in both backends."""
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
    rho_native = robustness(formula, sig, backend="native")
    rho_breach = robustness(formula, sig, backend="breach")
    np.testing.assert_allclose(rho_native, rho_breach)


def test_native_does_not_extend_last_for_top_level_and() -> None:
    """NativeBackend does not apply Breach's penultimate-extension for And/Or."""
    sig = _signal_xy([2.0, 3.0, -1.0], [1.0, -1.0, 4.0])
    formula = Node(
        kind="and",
        children=(
            _pred_xy_gt("px", "x", "y"),
            _pred_xy_gt("py", "y", "x"),
        ),
    )
    rho_native = robustness(formula, sig, backend="native")
    rho_breach = robustness(formula, sig, backend="breach")
    # Native: actual robustness at the last timestep
    # Breach: copies the penultimate value to the last position
    assert not np.allclose(rho_native[0, -1], rho_breach[0, -1])
    # Breach last value equals the penultimate value
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


def _fake_torch_signal() -> TorchSignal:
    # numpy arrays stand in for tensors; the guard must reject before touching values
    return TorchSignal(
        values=np.zeros((1, 2, 3)),
        times=np.arange(3, dtype=float),
        labels={"x": 0, "y": 1},
    )


def test_numpy_backends_reject_torch_signal() -> None:
    formula = _pred_xy_gt("p", "x", "y")
    for backend in (NativeBackend(), BreachBackend(), RtamtBackend(), StlcgppBackend()):
        with pytest.raises(TypeError, match=backend.name):
            backend.evaluate(formula, _fake_torch_signal())


def test_torch_backend_rejects_numpy_signal() -> None:
    formula = _pred_xy_gt("p", "x", "y")
    sig = _signal_xy([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    with pytest.raises(TypeError, match="TorchSignal"):
        StlcgppTorchBackend().evaluate(formula, sig)
