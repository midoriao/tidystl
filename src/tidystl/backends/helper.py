"""Backend-private helpers shared by concrete backend implementations.

`BaseResult` is exported publicly (no leading underscore) to satisfy pyright's
reportPrivateUsage rule, but it is an internal extension point for backend
authors, not a stable public API.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import count
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tidystl.core.helper import Folding
from tidystl.core.nodes import ArithNode, Node
from tidystl.core.signal import Signal


@dataclass(frozen=True)
class DagNode:
    id: str
    inputs: tuple[str, ...]
    op: object


@dataclass(frozen=True)
class ComputationDag:
    nodes: tuple[DagNode, ...]
    output: str


@dataclass(frozen=True)
class DagRef:
    output_id: str


@dataclass(frozen=True)
class LoadConst:
    value: float


@dataclass(frozen=True)
class LoadVar:
    name: str


@dataclass(frozen=True)
class Add:
    pass


@dataclass(frozen=True)
class Sub:
    pass


@dataclass(frozen=True)
class Mul:
    pass


@dataclass(frozen=True)
class Div:
    pass


@dataclass(frozen=True)
class Pow:
    pass


@dataclass(frozen=True)
class AbsoluteValue:
    pass


@dataclass(frozen=True)
class SquareRoot:
    pass


_ARITH_BINARY_OPS: dict[str, object] = {
    "+": Add(),
    "-": Sub(),
    "*": Mul(),
    "/": Div(),
    "^": Pow(),
}


class DagEmitter:
    def __init__(self) -> None:
        self._counter = count()
        self._nodes: list[DagNode] = []
        self._trace: dict[int, tuple[Node, str]] = {}

    def emit(self, ast_node: Node, op: object, inputs: tuple[str, ...]) -> DagRef:
        node_id = f"dag-{next(self._counter)}"
        self._nodes.append(DagNode(id=node_id, inputs=inputs, op=op))
        self._trace[id(ast_node)] = (ast_node, node_id)
        return DagRef(output_id=node_id)

    def build_result(self, fragment: DagRef) -> tuple[ComputationDag, dict[int, tuple[Node, str]]]:
        dag = ComputationDag(nodes=tuple(self._nodes), output=fragment.output_id)
        return dag, dict(self._trace)

    def traced_nodes(self) -> Iterable[Node]:
        return [node for node, _ in self._trace.values()]


class ArithmeticDagBuilder(Folding[DagRef]):
    def __init__(self, emit: Callable[[Node, object, tuple[str, ...]], DagRef]) -> None:
        self._emit = emit

    def atom(self, node: Node) -> DagRef:
        match node.kind:
            case "const":
                value = node.attrs["value"]
                if not isinstance(value, float):
                    raise TypeError("const node requires float value attr")
                return self._emit(node, LoadConst(value), ())
            case "var":
                name = node.attrs["name"]
                if not isinstance(name, str):
                    raise TypeError("var node requires str name attr")
                return self._emit(node, LoadVar(name), ())
            case _:
                raise NotImplementedError(f"unknown arithmetic atom kind {node.kind!r}")

    def unary(self, node: Node, child: DagRef) -> DagRef:
        match node.kind:
            case "abs":
                return self._emit(node, AbsoluteValue(), (child.output_id,))
            case "sqrt":
                return self._emit(node, SquareRoot(), (child.output_id,))
            case _:
                raise NotImplementedError(f"unknown arithmetic unary kind {node.kind!r}")

    def binary(self, node: Node, left: DagRef, right: DagRef) -> DagRef:
        op = _ARITH_BINARY_OPS.get(node.kind)
        if op is None:
            raise NotImplementedError(f"unknown arithmetic binary kind {node.kind!r}")
        return self._emit(node, op, (left.output_id, right.output_id))

    def nary(self, node: Node, children: tuple[DagRef, ...]) -> DagRef:
        raise NotImplementedError(f"n-ary arithmetic nodes are not supported: {node.kind!r}")


class ArithmeticOpEvaluator:
    def __init__(self, signal: Signal) -> None:
        self._signal = signal

    def evaluate(
        self,
        op: object,
        inputs: tuple[NDArray[np.floating], ...],
    ) -> NDArray[np.floating] | None:
        match op:
            case LoadConst(value=value):
                shape = (
                    self._signal.values.shape[0],
                    self._signal.values.shape[2],
                )
                return np.full(shape, value)
            case LoadVar(name=name):
                return self._signal[name]
            case Add():
                left, right = inputs
                return left + right
            case Sub():
                left, right = inputs
                return left - right
            case Mul():
                left, right = inputs
                return left * right
            case Div():
                left, right = inputs
                return left / right
            case Pow():
                left, right = inputs
                return np.power(left, right)
            case AbsoluteValue():
                (child,) = inputs
                return np.abs(child)
            case SquareRoot():
                (child,) = inputs
                return np.sqrt(child)
            case _:
                return None


class BaseResult:
    """Shared result base for all tracing backends."""

    has_trace: bool = True

    def __init__(
        self,
        robustness: Any,
        trace: dict[int, tuple[Node | ArithNode, str]],
        outputs: dict[str, Any],
    ) -> None:
        self.robustness = robustness
        self._trace = trace
        self._outputs = outputs

    def trace_for(self, node: Node | ArithNode) -> Any:
        entry = self._trace.get(id(node))
        if entry is None:
            raise KeyError(
                f"{type(node).__name__} node has no representative trace output in this backend"
            )
        _, dag_id = entry
        return self._outputs[dag_id]

    def traced_nodes(self) -> Iterable[Node | ArithNode]:
        return [node for node, _ in self._trace.values()]


class TorchArithmeticOpEvaluator:
    def __init__(self, signal: Any) -> None:
        self._signal = signal

    def evaluate(
        self,
        op: object,
        inputs: tuple[Any, ...],
    ) -> Any | None:
        import torch  # type: ignore[import-untyped]

        _t: Any = torch  # shadow as Any so downstream calls are not Unknown
        match op:
            case LoadConst(value=value):
                shape = (
                    self._signal.values.shape[0],
                    self._signal.values.shape[2],
                )
                return _t.full(
                    shape,
                    value,
                    dtype=self._signal.values.dtype,
                    device=self._signal.values.device,
                )
            case LoadVar(name=name):
                return self._signal[name]
            case Add():
                left, right = inputs
                return left + right
            case Sub():
                left, right = inputs
                return left - right
            case Mul():
                left, right = inputs
                return left * right
            case Div():
                left, right = inputs
                return left / right
            case Pow():
                left, right = inputs
                return _t.pow(left, right)
            case AbsoluteValue():
                (child,) = inputs
                return _t.abs(child)
            case SquareRoot():
                (child,) = inputs
                return _t.sqrt(child)
            case _:
                return None
