"""Parametrized backend realizing arbitrary points in the Core 6 config space.

Unlike the faithful per-tool backends, `generic` exposes the implicit-semantics
choices as independent knobs. With its default config it reproduces `native`.
See packages/tidystl/docs/design.md for the backend registration model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

import tidystl.backends.algorithms as algorithms
from tidystl.backends._pl_dag import (
    BoundedUntilKernel,
    Eq,
    PLDagBuilder,
    PLExecutor,
    WindowMax,
    WindowMin,
)
from tidystl.backends.helper import ArithmeticOpEvaluator, BaseResult, ComputationDag, DagRef
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.helper import fold_tree
from tidystl.core.nodes import ArithNode, Node
from tidystl.core.signal import Signal, TorchSignal
from tidystl_compat.generic_strategies import equality_score, pl_window_reduce
from tidystl_compat.pymtl import PymtlDagBuilder as _PymtlDagBuilder
from tidystl_compat.pymtl import PymtlExecutor as _PymtlExecutor
from tidystl_compat.rtamt import (
    DiscreteBoundedUntil,
    DiscreteWindowMax,
    DiscreteWindowMin,
    Ineq,
    Negate,
    PointwiseMax,
    PointwiseMin,
)
from tidystl_compat.rtamt import (
    DiscreteDagBuilder as _DiscreteDagBuilder,
)
from tidystl_compat.rtamt import (
    DiscreteResult as _DiscreteResult,
)
from tidystl_compat.rtamt import (
    Eq as DiscreteEq,
)
from tidystl_compat.taliro import predicate_norm, taliro_eval_until

SignalModel = Literal["pl_interp", "pl_samples", "zoh", "discrete"]
Boundary = Literal["clamp", "pessimistic"]
Terminal = Literal["none", "extend_penultimate"]
Predicate = Literal["signed", "euclidean"]
Equality = Literal["signed", "bigm", "epsilon"]
UntilPrefix = Literal["inclusive", "exclusive"]

_ALLOWED: dict[str, tuple[str, ...]] = {
    "signal_model": ("pl_interp", "pl_samples", "zoh", "discrete"),
    "boundary": ("clamp", "pessimistic"),
    "terminal": ("none", "extend_penultimate"),
    "predicate": ("signed", "euclidean"),
    "equality": ("signed", "bigm", "epsilon"),
    "until_prefix": ("inclusive", "exclusive"),
}


@dataclass(frozen=True)
class GenericConfig:
    """A point in the Core 6 implicit-semantics configuration space.

    Defaults reproduce `native` (principled PL dense-time semantics).
    """

    signal_model: SignalModel = "pl_interp"
    boundary: Boundary = "clamp"
    terminal: Terminal = "none"
    predicate: Predicate = "signed"
    equality: Equality = "signed"
    until_prefix: UntilPrefix = "inclusive"

    def __post_init__(self) -> None:
        for field, allowed in _ALLOWED.items():
            value = getattr(self, field)
            if value not in allowed:
                raise ValueError(
                    f"GenericConfig: invalid {field!r} value {value!r}; allowed: {allowed}"
                )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenericConfig:
        """Construct a ``GenericConfig`` from a plain ``dict[str, str]`` (e.g. from JSON).

        This is the typed entry-point for building configs from the registry JSON so
        callers do not need an unsafe ``**d`` spread into a ``Literal``-typed constructor.
        """
        return cls(
            signal_model=cast("SignalModel", d["signal_model"]),
            boundary=cast("Boundary", d["boundary"]),
            terminal=cast("Terminal", d["terminal"]),
            predicate=cast("Predicate", d["predicate"]),
            equality=cast("Equality", d["equality"]),
            until_prefix=cast("UntilPrefix", d["until_prefix"]),
        )


@dataclass(frozen=True)
class EuclideanIneq:
    """Predicate op carrying the precomputed ``||A||`` Euclidean normalization factor."""

    direction: Literal["ge", "le"]
    norm: float


class _GenericPLDagBuilder(PLDagBuilder):
    """PL DAG builder that emits ``EuclideanIneq`` when ``predicate='euclidean'``."""

    def __init__(self, config: GenericConfig) -> None:
        super().__init__()
        self._config = config

    def atom(self, node: Node) -> DagRef:
        if self._config.predicate != "euclidean":
            return super().atom(node)
        # euclidean path: validate and compute ||A||
        if node.kind != "predicate":
            raise NotImplementedError(f"unknown STL atom kind {node.kind!r}")
        op_name = node.attrs["op"]
        left_node = node.attrs["left"]
        right_node = node.attrs["right"]
        if not isinstance(op_name, str):
            raise TypeError("predicate node requires str op attr")
        if not isinstance(left_node, Node) or not isinstance(right_node, Node):
            raise TypeError("predicate node requires Node left/right attrs")
        if op_name == "==":
            raise NotImplementedError("TaLiRo half-spaces cannot represent equality predicates")
        left = fold_tree(left_node, self._arith)
        right = fold_tree(right_node, self._arith)
        direction: Literal["ge", "le"] = "ge" if op_name in (">", ">=") else "le"
        norm = predicate_norm(left_node, right_node)  # raises NotImplementedError if nonlinear
        return self._emitter.emit(
            node, EuclideanIneq(direction, norm), (left.output_id, right.output_id)
        )


class _GenericPLExecutor(PLExecutor):
    def __init__(self, signal: Signal, config: GenericConfig) -> None:
        super().__init__(signal)
        self._config = config

    def _evaluate_op(
        self, op: object, inputs: tuple[NDArray[np.floating], ...]
    ) -> NDArray[np.floating]:
        endpoints = "interp" if self._config.signal_model == "pl_interp" else "samples"
        match op:
            case EuclideanIneq(direction="ge", norm=norm):
                left, right = inputs
                return (left - right) / norm
            case EuclideanIneq(direction="le", norm=norm):
                left, right = inputs
                return (right - left) / norm
            case Eq():
                left, right = inputs
                return equality_score(left, right, self._config.equality)
            case WindowMin(start=start, end=end):
                (child,) = inputs
                return pl_window_reduce(
                    self.signal.times,
                    child,
                    start,
                    end,
                    np.min,
                    endpoints=endpoints,
                    boundary=self._config.boundary,
                    identity=np.inf,
                )
            case WindowMax(start=start, end=end):
                (child,) = inputs
                return pl_window_reduce(
                    self.signal.times,
                    child,
                    start,
                    end,
                    np.max,
                    endpoints=endpoints,
                    boundary=self._config.boundary,
                    identity=-np.inf,
                )
            case BoundedUntilKernel(start=start, end=end):
                left, right = inputs
                if self._config.boundary == "pessimistic" and endpoints == "samples":
                    # Pessimistic: no past-end extension; empty witness window yields -inf.
                    # This matches TaLiRo's taliro_eval_until semantics.
                    return taliro_eval_until(self.signal.times, left, right, start, end)
                # clamp (and interp): fall through to eval_until which end-extends.
                return super()._evaluate_op(op, inputs)
            case _:
                return super()._evaluate_op(op, inputs)


class _GenericDiscreteExecutor:
    """Execute a discrete DAG with boundary/until-prefix knobs from GenericConfig."""

    def __init__(self, signal: Signal, config: GenericConfig) -> None:
        self._arith = ArithmeticOpEvaluator(signal)
        self._config = config

    def execute(
        self,
        dag: ComputationDag,
        trace: dict[int, tuple[Node | ArithNode, str]],
    ) -> _DiscreteResult:
        outputs: dict[str, NDArray[np.floating]] = {}
        for node in dag.nodes:
            inputs = tuple(outputs[input_id] for input_id in node.inputs)
            outputs[node.id] = self._evaluate_op(node.op, inputs)
        return _DiscreteResult(
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
            case DiscreteEq():
                left, right = inputs
                return equality_score(left, right, self._config.equality)
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
                if self._config.boundary == "pessimistic":
                    return algorithms.discrete_sliding_reduce(child, start, end, np.min, np.inf)
                else:  # clamp
                    return algorithms.discrete_sliding_reduce_last(child, start, end, np.min)
            case DiscreteWindowMax(start=start, end=end):
                (child,) = inputs
                if self._config.boundary == "pessimistic":
                    return algorithms.discrete_sliding_reduce(child, start, end, np.max, -np.inf)
                else:  # clamp
                    return algorithms.discrete_sliding_reduce_last(child, start, end, np.max)
            case DiscreteBoundedUntil(start=start, end=end):
                left, right = inputs
                if self._config.until_prefix == "exclusive":
                    return algorithms.discrete_eval_until(left, right, start, end)
                else:  # inclusive
                    return algorithms.discrete_eval_until_inclusive_last(left, right, start, end)
            case _:
                raise NotImplementedError(f"unknown DAG op {type(op).__name__}")


class GenericBackend(EvaluationBackend):
    """Parametrized evaluation backend. Pass an instance as `backend=`."""

    name = "generic"

    def __init__(self, config: GenericConfig | None = None) -> None:
        self.config = config or GenericConfig()

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> BaseResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        if self.config.signal_model in ("pl_interp", "pl_samples"):
            dag, trace = _GenericPLDagBuilder(self.config).build(formula)
            result = _GenericPLExecutor(signal, self.config).execute(dag, trace)
            if (
                self.config.terminal == "extend_penultimate"
                and formula.kind in ("and", "or")
                and len(formula.children) == 2
            ):
                extended = result.robustness.copy()
                if extended.shape[-1] >= 2:
                    extended[:, -1] = extended[:, -2]
                return result.with_top_output(dag.output, extended)
            return result
        if self.config.signal_model == "discrete":
            dag, trace = _DiscreteDagBuilder(signal).build(formula)
            return _GenericDiscreteExecutor(signal, self.config).execute(dag, trace)
        if self.config.signal_model == "zoh":
            dag, trace = _PymtlDagBuilder().build(formula)
            return _PymtlExecutor(signal, dt=0.1).execute(dag, trace)
        raise NotImplementedError(f"signal_model {self.config.signal_model!r} not yet supported")
