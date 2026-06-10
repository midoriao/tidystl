import numpy as np
import pytest

from tidystl import EvaluationResult, evaluate, robustness
from tidystl.core.evaluator import BackendRegistry
from tidystl.core.nodes import ArithBinaryKind, ComparisonOp, Node
from tidystl.core.signal import Signal


def _const(v: float) -> Node:
    return Node(kind="const", attrs={"value": v})


def _var(n: str) -> Node:
    return Node(kind="var", attrs={"name": n})


def _binop(kind: ArithBinaryKind, left: Node, right: Node) -> Node:
    return Node(kind=kind, children=(left, right))


def _pred(name: str, op: ComparisonOp, left: Node, right: Node) -> Node:
    return Node(
        kind="predicate",
        attrs={"name": name, "op": op, "left": left, "right": right},
    )


def _make_signal(x: list[float]) -> Signal:
    arr = np.array(x)[np.newaxis, :]
    return Signal.from_dict(
        times=np.arange(len(x), dtype=float),
        values={"x": arr},
    )


class TestBackendRegistry:
    def test_unknown_backend_name_lists_available(self) -> None:
        sig = _make_signal([1.0, 2.0])
        phi = _pred("p", ">=", _var("x"), _const(0.0))
        with pytest.raises(LookupError) as excinfo:
            robustness(phi, sig, backend="does-not-exist")
        message = str(excinfo.value)
        assert "does-not-exist" in message
        assert "available backends" in message

    def test_get_message_includes_registered_names(self) -> None:
        class _Dummy:
            name = "dummy"

            def evaluate(
                self, formula: Node, signal: Signal
            ) -> EvaluationResult:  # pragma: no cover
                raise NotImplementedError

        registry = BackendRegistry()
        registry.register(_Dummy())  # type: ignore[arg-type]
        with pytest.raises(LookupError, match="dummy"):
            registry.get("missing")


class TestTraceArithmetic:
    def test_const(self) -> None:
        sig = _make_signal([1.0, 2.0, 3.0])
        const = _const(5.0)
        formula = _pred("p", ">=", const, _const(0.0))
        result = evaluate(formula, sig)
        trace = result.trace_for(const)
        assert trace.shape == (1, 3)
        np.testing.assert_allclose(trace, 5.0)

    def test_var(self) -> None:
        sig = _make_signal([1.0, 2.0, 3.0])
        var = _var("x")
        formula = _pred("p", ">=", var, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(var), [[1.0, 2.0, 3.0]])

    def test_add(self) -> None:
        sig = _make_signal([1.0, 2.0])
        expr = _binop("+", _var("x"), _const(10.0))
        formula = _pred("p", ">=", expr, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(expr), [[11.0, 12.0]])

    def test_sub(self) -> None:
        sig = _make_signal([5.0, 3.0])
        expr = _binop("-", _var("x"), _const(2.0))
        formula = _pred("p", ">=", expr, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(expr), [[3.0, 1.0]])

    def test_mul(self) -> None:
        sig = _make_signal([2.0, 3.0])
        expr = _binop("*", _var("x"), _const(4.0))
        formula = _pred("p", ">=", expr, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(expr), [[8.0, 12.0]])

    def test_div(self) -> None:
        sig = _make_signal([10.0, 6.0])
        expr = _binop("/", _var("x"), _const(2.0))
        formula = _pred("p", ">=", expr, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(expr), [[5.0, 3.0]])

    def test_pow(self) -> None:
        sig = _make_signal([2.0, 3.0])
        expr = _binop("^", _var("x"), _const(2.0))
        formula = _pred("p", ">=", expr, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(expr), [[4.0, 9.0]])

    def test_abs(self) -> None:
        sig = _make_signal([-3.0, 2.0])
        expr = Node(kind="abs", children=(_var("x"),))
        formula = _pred("p", ">=", expr, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(expr), [[3.0, 2.0]])

    def test_sqrt(self) -> None:
        sig = _make_signal([4.0, 9.0])
        expr = Node(kind="sqrt", children=(_var("x"),))
        formula = _pred("p", ">=", expr, _const(0.0))
        result = evaluate(formula, sig)
        np.testing.assert_allclose(result.trace_for(expr), [[2.0, 3.0]])


class TestEvalCompare:
    def test_gte(self) -> None:
        sig = _make_signal([3.0, 5.0, 1.0])
        phi = _pred("p", ">=", _var("x"), _const(2.0))
        np.testing.assert_allclose(robustness(phi, sig), [[1.0, 3.0, -1.0]])

    def test_lte(self) -> None:
        sig = _make_signal([3.0, 5.0, 1.0])
        phi = _pred("p", "<=", _var("x"), _const(4.0))
        np.testing.assert_allclose(robustness(phi, sig), [[1.0, -1.0, 3.0]])

    def test_gt(self) -> None:
        sig = _make_signal([3.0, 5.0, 1.0])
        phi = _pred("p", ">", _var("x"), _const(2.0))
        np.testing.assert_allclose(robustness(phi, sig), [[1.0, 3.0, -1.0]])

    def test_lt(self) -> None:
        sig = _make_signal([3.0, 5.0, 1.0])
        phi = _pred("p", "<", _var("x"), _const(4.0))
        np.testing.assert_allclose(robustness(phi, sig), [[1.0, -1.0, 3.0]])

    def test_eq(self) -> None:
        sig = _make_signal([2.0, 3.0])
        phi = _pred("p", "==", _var("x"), _const(3.0))
        np.testing.assert_allclose(robustness(phi, sig), [[-1.0, 0.0]])


class TestEvalBoolean:
    def test_not(self) -> None:
        sig = _make_signal([3.0, -1.0])
        pred = _pred("p", ">=", _var("x"), _const(0.0))
        phi: Node = Node(kind="not", children=(pred,))
        np.testing.assert_allclose(robustness(phi, sig), [[-3.0, 1.0]])

    def test_and(self) -> None:
        times = np.arange(3, dtype=float)
        sig = Signal.from_dict(
            times=times,
            values={
                "x": np.array([[5.0, 1.0, 3.0]]),
                "y": np.array([[2.0, 4.0, 1.0]]),
            },
        )
        p = _pred("p", ">=", _var("x"), _const(0.0))
        q = _pred("q", ">=", _var("y"), _const(0.0))
        phi: Node = Node(kind="and", children=(p, q))
        np.testing.assert_allclose(robustness(phi, sig), [[2.0, 1.0, 1.0]])

    def test_or(self) -> None:
        times = np.arange(3, dtype=float)
        sig = Signal.from_dict(
            times=times,
            values={
                "x": np.array([[5.0, 1.0, 3.0]]),
                "y": np.array([[2.0, 4.0, 1.0]]),
            },
        )
        p = _pred("p", ">=", _var("x"), _const(0.0))
        q = _pred("q", ">=", _var("y"), _const(0.0))
        phi: Node = Node(kind="or", children=(p, q))
        # Native: max(x,y) = [5,4,3]; BreachBackend extends penultimate → [5,4,4]
        np.testing.assert_allclose(robustness(phi, sig), [[5.0, 4.0, 3.0]])


class TestEvalTemporal:
    def test_predicate_returns_robustness(self) -> None:
        sig = _make_signal([3.0, 5.0, 1.0])
        pred = _pred("x_pos", ">=", _var("x"), _const(2.0))
        result = robustness(pred, sig)
        np.testing.assert_allclose(result, [[1.0, 3.0, -1.0]])
        assert result.shape == (1, 3)

    def test_always_full_window(self) -> None:
        sig = _make_signal([3.0, 1.0, 4.0, 1.0, 5.0])
        pred = _pred("p", ">=", _var("x"), _const(0.0))
        phi: Node = Node(kind="always", children=(pred,), attrs={"interval": (0.0, 4.0)})
        result = robustness(phi, sig)
        assert result[0, 0] == 1.0

    def test_always_sliding(self) -> None:
        sig = _make_signal([5.0, 2.0, 8.0, 1.0, 6.0])
        pred = _pred("p", ">=", _var("x"), _const(0.0))
        phi: Node = Node(kind="always", children=(pred,), attrs={"interval": (0.0, 2.0)})
        result = robustness(phi, sig)
        np.testing.assert_allclose(result[0, :3], [2.0, 1.0, 1.0])

    def test_always_boundary_clamp(self) -> None:
        sig = _make_signal([3.0, 1.0, 4.0])
        pred = _pred("p", ">=", _var("x"), _const(0.0))
        phi: Node = Node(kind="always", children=(pred,), attrs={"interval": (0.0, 2.0)})
        result = robustness(phi, sig)
        assert result[0, 2] == 4.0

    def test_always_batch(self) -> None:
        times = np.arange(5, dtype=float)
        sig = Signal.from_dict(
            times=times,
            values={
                "x": np.array(
                    [
                        [3.0, 1.0, 4.0, 1.0, 5.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                    ]
                ),
            },
        )
        pred = _pred("p", ">=", _var("x"), _const(0.0))
        phi: Node = Node(kind="always", children=(pred,), attrs={"interval": (0.0, 2.0)})
        result = robustness(phi, sig)
        np.testing.assert_allclose(result[0, :3], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(result[1, :3], [1.0, 2.0, 3.0])

    def test_eventually_sliding(self) -> None:
        sig = _make_signal([1.0, 5.0, 2.0, 8.0, 3.0])
        pred = _pred("p", ">=", _var("x"), _const(0.0))
        phi: Node = Node(
            kind="eventually",
            children=(pred,),
            attrs={"interval": (0.0, 2.0)},
        )
        result = robustness(phi, sig)
        np.testing.assert_allclose(result[0, :3], [5.0, 8.0, 8.0])

    def test_until_basic(self) -> None:
        times = np.arange(5, dtype=float)
        sig = Signal.from_dict(
            times=times,
            values={
                "x": np.array([[1.0, 1.0, 1.0, -1.0, -1.0]]),
                "y": np.array([[-1.0, -1.0, 2.0, 2.0, 2.0]]),
            },
        )
        p = _pred("p", ">=", _var("x"), _const(0.0))
        q = _pred("q", ">=", _var("y"), _const(0.0))
        phi: Node = Node(kind="until", children=(p, q), attrs={"interval": (0.0, 3.0)})
        result = robustness(phi, sig)
        assert result[0, 0] == 1.0

    def test_until_never_satisfied(self) -> None:
        times = np.arange(5, dtype=float)
        sig = Signal.from_dict(
            times=times,
            values={
                "x": np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]),
                "y": np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]]),
            },
        )
        p = _pred("p", ">=", _var("x"), _const(0.0))
        q = _pred("q", ">=", _var("y"), _const(0.0))
        phi: Node = Node(kind="until", children=(p, q), attrs={"interval": (0.0, 4.0)})
        result = robustness(phi, sig)
        assert result[0, 0] < 0
