from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
    TorchArithmeticOpEvaluator,
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
class DiscreteWindowMinLast:
    start: int
    end: int


@dataclass(frozen=True)
class DiscreteWindowMaxLast:
    start: int
    end: int


@dataclass(frozen=True)
class DiscreteBoundedUntilInclusiveLast:
    start: int
    end: int


class _StlcgppResult(BaseResult):
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
        raise ValueError("stlcgpp backend requires a strictly increasing uniform time grid")
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
    def __init__(self, signal: Signal | TorchSignal) -> None:
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
                op = DiscreteWindowMinLast(start, end)
            case "eventually":
                start, end = _discrete_offsets(node, self._step)
                op = DiscreteWindowMaxLast(start, end)
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
                op = DiscreteBoundedUntilInclusiveLast(start, end)
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
    ) -> _StlcgppResult:
        outputs: dict[str, NDArray[np.floating]] = {}
        for node in dag.nodes:
            inputs = tuple(outputs[input_id] for input_id in node.inputs)
            outputs[node.id] = self._evaluate_op(node.op, inputs)
        return _StlcgppResult(
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
            case DiscreteWindowMinLast(start=start, end=end):
                (child,) = inputs
                return algorithms.discrete_sliding_reduce_last(child, start, end, np.min)
            case DiscreteWindowMaxLast(start=start, end=end):
                (child,) = inputs
                return algorithms.discrete_sliding_reduce_last(child, start, end, np.max)
            case DiscreteBoundedUntilInclusiveLast(start=start, end=end):
                left, right = inputs
                return algorithms.discrete_eval_until_inclusive_last(left, right, start, end)
            case _:
                raise NotImplementedError(f"unknown DAG op {type(op).__name__}")


def _torch_maxish(
    signal: Any,
    *,
    approx_method: str,
    temperature: float,
):
    import torch

    if approx_method == "true":
        return torch.max(signal, dim=-1).values
    if approx_method == "softmax":
        weights = torch.nn.functional.softmax(temperature * signal, dim=-1)
        return (weights * signal).sum(dim=-1)
    if approx_method == "logsumexp":
        return torch.logsumexp(temperature * signal, dim=-1) / temperature
    raise ValueError(f"unknown approx_method {approx_method!r}")


def _torch_minish(
    signal: Any,
    *,
    approx_method: str,
    temperature: float,
):
    return -_torch_maxish(
        -signal,
        approx_method=approx_method,
        temperature=temperature,
    )


def _torch_sliding_reduce_last(
    rho: Any,
    start: int,
    end: int,
    *,
    mode: Literal["min", "max"],
    approx_method: str,
    temperature: float,
):
    import torch

    windows: list[Any] = []
    t_len = rho.shape[-1]
    reduce_fn = _torch_minish if mode == "min" else _torch_maxish

    for i in range(t_len):
        j_start = i + start
        if j_start >= t_len:
            window = rho[:, -1:]
        else:
            j_end = min(i + end + 1, t_len)
            window = rho[:, j_start:j_end]
        windows.append(reduce_fn(window, approx_method=approx_method, temperature=temperature))

    return torch.stack(windows, dim=-1)


def _torch_eval_until_inclusive_last(
    rho_p: Any,
    rho_q: Any,
    start: int,
    end: int,
    *,
    approx_method: str,
    temperature: float,
):
    import torch

    outputs: list[Any] = []
    t_len = rho_p.shape[-1]

    for i in range(t_len):
        candidates: list[Any] = []
        for offset in range(start, end + 1):
            j_clamped = min(i + offset, t_len - 1)
            q_val = rho_q[:, j_clamped]
            p_min = _torch_minish(
                rho_p[:, i : j_clamped + 1],
                approx_method=approx_method,
                temperature=temperature,
            )
            candidate = _torch_minish(
                torch.stack([q_val, p_min], dim=-1),
                approx_method=approx_method,
                temperature=temperature,
            )
            candidates.append(candidate)
        outputs.append(
            _torch_maxish(
                torch.stack(candidates, dim=-1),
                approx_method=approx_method,
                temperature=temperature,
            )
        )

    return torch.stack(outputs, dim=-1)


class _StlcgppTorchResult(BaseResult):
    """Like _StlcgppResult but holds torch tensors; robustness supports .backward()."""


class _TorchExecutor:
    def __init__(
        self,
        signal: TorchSignal,
        *,
        approx_method: str,
        temperature: float,
    ) -> None:
        self._arith = TorchArithmeticOpEvaluator(signal)
        self._approx_method = approx_method
        self._temperature = temperature

    def execute(
        self,
        dag: ComputationDag,
        trace: dict[int, tuple[Node | ArithNode, str]],
    ) -> _StlcgppTorchResult:
        outputs: dict[str, Any] = {}
        for node in dag.nodes:
            inputs = tuple(outputs[input_id] for input_id in node.inputs)
            outputs[node.id] = self._evaluate_op(node.op, inputs)
        return _StlcgppTorchResult(
            robustness=outputs[dag.output],
            trace=trace,
            outputs=outputs,
        )

    def _evaluate_op(
        self,
        op: object,
        inputs: tuple[Any, ...],
    ) -> Any:
        import torch

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
                return -torch.abs(left - right)
            case Negate():
                (child,) = inputs
                return -child
            case PointwiseMin():
                left, right = inputs
                return _torch_minish(
                    torch.stack([left, right], dim=-1),
                    approx_method=self._approx_method,
                    temperature=self._temperature,
                )
            case PointwiseMax():
                left, right = inputs
                return _torch_maxish(
                    torch.stack([left, right], dim=-1),
                    approx_method=self._approx_method,
                    temperature=self._temperature,
                )
            case DiscreteWindowMinLast(start=start, end=end):
                (child,) = inputs
                return _torch_sliding_reduce_last(
                    child,
                    start,
                    end,
                    mode="min",
                    approx_method=self._approx_method,
                    temperature=self._temperature,
                )
            case DiscreteWindowMaxLast(start=start, end=end):
                (child,) = inputs
                return _torch_sliding_reduce_last(
                    child,
                    start,
                    end,
                    mode="max",
                    approx_method=self._approx_method,
                    temperature=self._temperature,
                )
            case DiscreteBoundedUntilInclusiveLast(start=start, end=end):
                left, right = inputs
                return _torch_eval_until_inclusive_last(
                    left,
                    right,
                    start,
                    end,
                    approx_method=self._approx_method,
                    temperature=self._temperature,
                )
            case _:
                raise NotImplementedError(f"unknown DAG op {type(op).__name__}")


class StlcgppBackend(EvaluationBackend):
    """STLCG++-compatible backend over uniform discrete-time traces."""

    name = "stlcgpp"

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> _StlcgppResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        dag, trace = _DagBuilder(signal).build(formula)
        return _Executor(signal).execute(dag, trace)


class StlcgppTorchBackend(EvaluationBackend):
    """Torch-backed STLCG++ backend with autograd support."""

    name = "stlcgpp_torch"

    def __init__(self, *, approx_method: str = "true", temperature: float = 1.0) -> None:
        self._approx_method = approx_method
        self._temperature = temperature

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> _StlcgppTorchResult:
        if not isinstance(signal, TorchSignal):
            raise TypeError(
                f"{self.name} backend requires a TorchSignal, got {type(signal).__name__}"
            )
        dag, trace = _DagBuilder(signal).build(formula)
        return _TorchExecutor(
            signal,
            approx_method=self._approx_method,
            temperature=self._temperature,
        ).execute(dag, trace)
