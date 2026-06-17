"""PL (piecewise-linear dense-time) DAG infrastructure.

Shared by NativeBackend and BreachBackend: op types, DAG builder, executor,
and result class. The result class exposes trace data and supports
`with_top_output` for backends that need to post-process the top-level output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tidystl.backends import algorithms
from tidystl.backends.helper import (
    ArithmeticDagBuilder,
    ArithmeticOpEvaluator,
    BaseResult,
    ComputationDag,
    DagEmitter,
    DagRef,
)
from tidystl.core.helper import Folding, fold_tree
from tidystl.core.nodes import ArithNode, Node
from tidystl.core.signal import Signal


@dataclass(frozen=True)
class Ineq:
    direction: Literal["ge", "le"]


@dataclass(frozen=True)
class Eq:
    pass


@dataclass(frozen=True)
class Negate:
    pass


@dataclass(frozen=True)
class PointwiseMin:
    pass


@dataclass(frozen=True)
class PointwiseMax:
    pass


@dataclass(frozen=True)
class WindowMin:
    start: float
    end: float


@dataclass(frozen=True)
class WindowMax:
    start: float
    end: float


@dataclass(frozen=True)
class BoundedUntilKernel:
    start: float
    end: float


_COMPARE_OPS: dict[str, object] = {
    ">=": Ineq("ge"),
    ">": Ineq("ge"),
    "<=": Ineq("le"),
    "<": Ineq("le"),
    "==": Eq(),
}


class PLResult(BaseResult):
    """Evaluation result for PL backends; supports sub-formula trace queries."""

    def with_top_output(self, output_id: str, new_value: NDArray[np.floating]) -> PLResult:
        """Return a new result with the top-level robustness replaced by `new_value`."""
        new_outputs = dict(self._outputs)
        new_outputs[output_id] = new_value
        return PLResult(robustness=new_value, trace=self._trace, outputs=new_outputs)


class PLDagBuilder(Folding[DagRef]):
    def __init__(self) -> None:
        self._emitter = DagEmitter()
        self._arith = ArithmeticDagBuilder(self._emitter.emit)

    def build(
        self, formula: Node
    ) -> tuple[ComputationDag, dict[int, tuple[Node | ArithNode, str]]]:
        fragment = fold_tree(formula, self)
        return self._emitter.build_result(fragment)

    def atom(self, node: Node) -> DagRef:
        if node.kind != "predicate":
            raise NotImplementedError(f"unknown STL atom kind {node.kind!r}")

        op_name = node.attrs["op"]
        left_node = node.attrs["left"]
        right_node = node.attrs["right"]
        if not isinstance(op_name, str):
            raise TypeError("predicate node requires str op attr")
        if not isinstance(left_node, Node) or not isinstance(right_node, Node):
            raise TypeError("predicate node requires Node left/right attrs")

        left = fold_tree(left_node, self._arith)
        right = fold_tree(right_node, self._arith)
        return self._emitter.emit(node, _COMPARE_OPS[op_name], (left.output_id, right.output_id))

    def unary(self, node: Node, child: DagRef) -> DagRef:
        match node.kind:
            case "not":
                op = Negate()
            case "always":
                start, end = self._require_interval(node)
                op = WindowMin(start, end)
            case "eventually":
                start, end = self._require_interval(node)
                op = WindowMax(start, end)
            case _:
                raise NotImplementedError(f"unknown STL unary operator {node.kind}")
        return self._emitter.emit(node, op, (child.output_id,))

    def binary(self, node: Node, left: DagRef, right: DagRef) -> DagRef:
        match node.kind:
            case "and":
                op = PointwiseMin()
            case "or":
                op = PointwiseMax()
            case "until":
                start, end = self._require_interval(node)
                op = BoundedUntilKernel(start, end)
            case _:
                raise NotImplementedError(f"unknown STL binary operator {node.kind}")
        return self._emitter.emit(node, op, (left.output_id, right.output_id))

    def nary(self, node: Node, children: tuple[DagRef, ...]) -> DagRef:
        raise NotImplementedError(f"n-ary STL nodes are not supported: {node.kind!r}")

    @staticmethod
    def _require_interval(node: Node) -> tuple[float, float]:
        match node.attrs.get("interval"):
            case (float(start), float(end)):
                return start, end
            case _:
                raise ValueError(f"temporal operator {node.kind} missing interval")


class PLExecutor:
    def __init__(self, signal: Signal) -> None:
        self.signal = signal
        self._arith = ArithmeticOpEvaluator(signal)

    def execute(
        self,
        dag: ComputationDag,
        trace: dict[int, tuple[Node | ArithNode, str]],
    ) -> PLResult:
        outputs: dict[str, NDArray[np.floating]] = {}
        for node in dag.nodes:
            inputs = tuple(outputs[input_id] for input_id in node.inputs)
            outputs[node.id] = self._evaluate_op(node.op, inputs)
        return PLResult(
            robustness=outputs[dag.output],
            trace=trace,
            outputs=outputs,
        )

    def _evaluate_op(
        self,
        op: object,
        inputs: tuple[NDArray[np.floating], ...],
    ) -> NDArray[np.floating]:
        arith_result = self._arith.evaluate(op, inputs)
        if arith_result is not None:
            return arith_result
        match op:
            case Ineq(direction="ge"):
                left, right = inputs
                return left - right
            case Ineq(direction="le"):
                left, right = inputs
                return right - left
            case Eq():
                left, right = inputs
                return -np.abs(left - right)
            case Negate():
                (child,) = inputs
                return -child
            case PointwiseMin():
                left, right = inputs
                return np.minimum(left, right)
            case PointwiseMax():
                left, right = inputs
                return np.maximum(left, right)
            case WindowMin(start=start, end=end):
                (child,) = inputs
                return algorithms.sliding_reduce(self.signal.times, child, start, end, np.min)
            case WindowMax(start=start, end=end):
                (child,) = inputs
                return algorithms.sliding_reduce(self.signal.times, child, start, end, np.max)
            case BoundedUntilKernel(start=start, end=end):
                left, right = inputs
                return algorithms.eval_until(self.signal.times, left, right, start, end)
            case _:
                raise NotImplementedError(f"unknown DAG op {type(op).__name__}")
