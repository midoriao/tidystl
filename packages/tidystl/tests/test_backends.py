import numpy as np
import pytest

from tidystl import NativeBackend, Signal, TorchSignal, evaluate
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
    assert isinstance(resolve_backend(None), NativeBackend)


def test_native_backend_can_be_resolved_by_name() -> None:
    assert isinstance(resolve_backend("native"), NativeBackend)


def test_evaluate_returns_trace_capable_result_for_native_backend() -> None:
    sig = _signal_xy([2.0, -1.0], [3.0, 4.0])
    formula = _pred_xy_gt("px", "x", "y")
    result = evaluate(formula, sig, backend="native")
    assert result.has_trace
    np.testing.assert_allclose(result.trace_for(formula)[0], [-1.0, -5.0])


def test_native_backend_rejects_torch_signal() -> None:
    formula = _pred_xy_gt("p", "x", "y")
    fake = TorchSignal(
        values=np.zeros((1, 2, 3)),
        times=np.arange(3, dtype=float),
        labels={"x": 0, "y": 1},
    )
    with pytest.raises(TypeError, match="native"):
        NativeBackend().evaluate(formula, fake)
