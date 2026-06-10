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
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.helper import Folding, fold_tree
from tidystl.core.nodes import ArithNode, Node
from tidystl.core.signal import Signal, TorchSignal


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
class DiscreteWindowMin:
    start: int
    end: int


@dataclass(frozen=True)
class DiscreteWindowMax:
    start: int
    end: int


@dataclass(frozen=True)
class DiscreteBoundedUntil:
    start: int
    end: int


class _RtamtResult(BaseResult):
    pass


_COMPARE_OPS: dict[str, object] = {
    ">=": Ineq("ge"),
    ">": Ineq("ge"),
    "<=": Ineq("le"),
    "<": Ineq("le"),
    "==": Eq(),
}


def _require_interval(node: Node) -> tuple[float, float]:
    match node.attrs.get("interval"):
        case (float(start), float(end)):
            return start, end
        case _:
            raise ValueError(f"temporal operator {node.kind} missing interval")


def _sampling_period(times: NDArray[np.floating]) -> float:
    if times.shape[0] < 2:
        return 1.0
    diffs = np.diff(times)
    step = float(diffs[0])
    if step <= 0.0 or not np.allclose(diffs, step, rtol=1e-9, atol=1e-12):
        raise ValueError("rtamt backend requires a strictly increasing uniform time grid")
    return step


def _discrete_offsets(node: Node, step: float) -> tuple[int, int]:
    start, end = _require_interval(node)
    start_idx = round(start / step)
    end_idx = round(end / step)
    if not np.isclose(start, start_idx * step, rtol=1e-9, atol=1e-12):
        raise ValueError(f"interval start {start} is not aligned to sampling period {step}")
    if not np.isclose(end, end_idx * step, rtol=1e-9, atol=1e-12):
        raise ValueError(f"interval end {end} is not aligned to sampling period {step}")
    if start_idx < 0 or end_idx < start_idx:
        raise ValueError(f"invalid interval {start, end}")
    return start_idx, end_idx


class _DagBuilder(Folding[DagRef]):
    def __init__(self, signal: Signal) -> None:
        self._emitter = DagEmitter()
        self._arith = ArithmeticDagBuilder(self._emitter.emit)
        self._step = _sampling_period(signal.times)

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
                start, end = _discrete_offsets(node, self._step)
                op = DiscreteWindowMin(start, end)
            case "eventually":
                start, end = _discrete_offsets(node, self._step)
                op = DiscreteWindowMax(start, end)
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
                start, end = _discrete_offsets(node, self._step)
                op = DiscreteBoundedUntil(start, end)
            case _:
                raise NotImplementedError(f"unknown STL binary operator {node.kind}")
        return self._emitter.emit(node, op, (left.output_id, right.output_id))

    def nary(self, node: Node, children: tuple[DagRef, ...]) -> DagRef:
        raise NotImplementedError(f"n-ary STL nodes are not supported: {node.kind!r}")


class _Executor:
    def __init__(self, signal: Signal) -> None:
        self._arith = ArithmeticOpEvaluator(signal)

    def execute(
        self,
        dag: ComputationDag,
        trace: dict[int, tuple[Node | ArithNode, str]],
    ) -> _RtamtResult:
        outputs: dict[str, NDArray[np.floating]] = {}
        for node in dag.nodes:
            inputs = tuple(outputs[input_id] for input_id in node.inputs)
            outputs[node.id] = self._evaluate_op(node.op, inputs)
        return _RtamtResult(
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
            case DiscreteWindowMin(start=start, end=end):
                (child,) = inputs
                return algorithms.discrete_sliding_reduce(child, start, end, np.min, np.inf)
            case DiscreteWindowMax(start=start, end=end):
                (child,) = inputs
                return algorithms.discrete_sliding_reduce(child, start, end, np.max, -np.inf)
            case DiscreteBoundedUntil(start=start, end=end):
                left, right = inputs
                return algorithms.discrete_eval_until(left, right, start, end)
            case _:
                raise NotImplementedError(f"unknown DAG op {type(op).__name__}")


class RtamtBackend(EvaluationBackend):
    """RTAMT-compatible backend over uniform discrete-time traces."""

    name = "rtamt"

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> _RtamtResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        dag, trace = _DagBuilder(signal).build(formula)
        return _Executor(signal).execute(dag, trace)
