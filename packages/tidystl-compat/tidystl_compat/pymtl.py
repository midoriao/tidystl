"""py-metric-temporal-logic compatible backend.

Reproduces the robustness semantics of the ``metric-temporal-logic`` package
(``import mtl``): piecewise-constant (ZOH) interpolation over an arbitrary,
possibly non-uniform time grid, with right-half-open temporal windows evaluated
on a ``dt`` pivot grid. See ``packages/tidystl/docs/design.md`` for the
semantics summary and ``tests/test_pymtl_compat.py`` for cross-validation
against the real library.
"""

from __future__ import annotations

from collections.abc import Callable
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

# --- Piecewise-constant grid kernels (py-mtl ZOH semantics) ---
#
# py-mtl evaluates temporal operators over a ``dt`` pivot grid with zero-order-hold
# (piecewise-constant) interpolation and right-half-open windows ``[t + lo, t + hi)``.
# These helpers resample a batched ``(N, T)`` trace onto a shared ``dt`` grid, run the
# reduction, and sample back to the original times.


def build_grid(times: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
    """Uniform ``dt`` grid covering ``[times[0], times[-1]]`` (py-mtl pivots)."""
    t0 = float(times[0])
    t1 = float(times[-1])
    n_steps = int(round((t1 - t0) / dt))
    return t0 + dt * np.arange(n_steps + 1)


def zoh_to_grid(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Zero-order-hold resample ``(N, T)`` onto ``grid`` -> ``(N, len(grid))``."""
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, times.shape[0] - 1)
    return rho[:, idx]


def _orig_indices(
    times: NDArray[np.floating], grid: NDArray[np.floating], dt: float
) -> NDArray[np.intp]:
    offsets: NDArray[np.floating] = (times - grid[0]) / dt
    idx = np.asarray(np.rint(offsets), dtype=np.intp)
    # Guard against any rounding divergence from build_grid's step count.
    return np.clip(idx, 0, grid.shape[0] - 1)


def pc_window_reduce(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    *,
    lo: float,
    hi: float,
    dt: float,
    reducer: Callable[..., NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Right-half-open windowed reduction ``[t + lo, t + hi)`` on the dt grid.

    Returns ``NaN`` where the window starts beyond the trace end (py-mtl leaves
    those times undefined and the ground-truth generator omits them).
    """
    grid = build_grid(times, dt)
    g = zoh_to_grid(times, rho, grid)
    lo_off = int(round(lo / dt))
    hi_off = int(round(hi / dt)) - 1  # half-open right endpoint
    reduced = algorithms.discrete_sliding_reduce(g, lo_off, hi_off, reducer, np.nan)
    return reduced[:, _orig_indices(times, grid, dt)]


def pc_weak_until_grid(
    rho_l: NDArray[np.floating], rho_r: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Untimed weak-until robustness over a shared grid (py-mtl recurrence).

    Mirrors ``mtl.evaluator.apply_weak_until`` for every batch row:
    ``prev = max(right, min(left, prev), -max_right)`` scanned backward.
    """
    n, m = rho_l.shape
    out = np.empty_like(rho_l)
    prev = np.full(n, np.inf, dtype=rho_l.dtype)
    max_right = np.full(n, -np.inf, dtype=rho_l.dtype)
    for k in range(m - 1, -1, -1):
        left = rho_l[:, k]
        right = rho_r[:, k]
        max_right = np.maximum(max_right, right)
        prev = np.maximum(np.maximum(right, np.minimum(left, prev)), -max_right)
        out[:, k] = prev
    return out


def _suffix_max(rho: NDArray[np.floating]) -> NDArray[np.floating]:
    """Untimed eventually: running max over the suffix ``[k, end]``."""
    # reverse, cumulative-max, reverse back
    return np.maximum.accumulate(rho[:, ::-1], axis=1)[:, ::-1]


def _clamp_fill(g: NDArray[np.floating]) -> NDArray[np.floating]:
    """Forward-fill trailing NaNs with each row's last defined value (ZOH).

    py-mtl's ``&`` (``dense_compose``) extends a window operator past its own
    domain end by holding its last breakpoint value, so when a window term is
    combined with longer-lived terms it clamps rather than truncates.
    """
    out = g.copy()
    n, m = out.shape
    for i in range(n):
        last = np.nan
        for k in range(m):
            if np.isnan(out[i, k]):
                out[i, k] = last
            else:
                last = out[i, k]
    return out


def _window_on_grid(
    g: NDArray[np.floating],
    *,
    lo: float,
    hi: float,
    dt: float,
    reducer: Callable[..., NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Clamping window variant used by the until desugaring conjuncts.

    Unlike the standalone ``pc_window_reduce`` (always/eventually), which leaves
    trailing NaN past the operator's domain, this clamps the tail so a window
    term combines correctly under py-mtl's ZOH-extending ``&``.
    """
    lo_off = int(round(lo / dt))
    hi_off = int(round(hi / dt)) - 1
    reduced = algorithms.discrete_sliding_reduce(g, lo_off, hi_off, reducer, np.nan)
    return _clamp_fill(reduced)


def pc_timed_until(
    times: NDArray[np.floating],
    rho_l: NDArray[np.floating],
    rho_r: NDArray[np.floating],
    *,
    lo: float,
    hi: float,
    dt: float,
) -> NDArray[np.floating]:
    """py-mtl timed-until robustness sampled at the original times.

    Reproduces ``sugar.timed_until``:
        F[lo,hi] r  &  G[0,lo] l  &  G[lo,lo] until(l, r)
    where py-mtl's ``until(l, r)`` is itself ``WeakUntil(l, r) & F r`` (``F r``
    is the untimed ``env(r)`` suffix max), and ``G[lo,lo]`` (zero width) is
    identity in py-mtl. Computed on the shared dt grid, then sampled back to
    ``times``.
    """
    grid = build_grid(times, dt)
    gl = zoh_to_grid(times, rho_l, grid)
    gr = zoh_to_grid(times, rho_r, grid)

    f_r = _window_on_grid(gr, lo=lo, hi=hi, dt=dt, reducer=np.max)
    # ``G[0,0]`` is identity in py-mtl, so skip the window when lo == 0.
    g_l = _window_on_grid(gl, lo=0.0, hi=lo, dt=dt, reducer=np.min) if lo > 0 else gl
    strong_until = np.minimum(pc_weak_until_grid(gl, gr), _suffix_max(gr))

    combined = np.minimum(np.minimum(f_r, g_l), strong_until)
    return combined[:, _orig_indices(times, grid, dt)]


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
class PcWindowMin:
    lo: float
    hi: float


@dataclass(frozen=True)
class PcWindowMax:
    lo: float
    hi: float


@dataclass(frozen=True)
class PcTimedUntil:
    lo: float
    hi: float


class _PymtlResult(BaseResult):
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


class _DagBuilder(Folding[DagRef]):
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
                op: object = Negate()
            case "always":
                lo, hi = _require_interval(node)
                op = PcWindowMin(lo, hi)
            case "eventually":
                lo, hi = _require_interval(node)
                op = PcWindowMax(lo, hi)
            case _:
                raise NotImplementedError(f"unknown STL unary operator {node.kind}")
        return self._emitter.emit(node, op, (child.output_id,))

    def binary(self, node: Node, left: DagRef, right: DagRef) -> DagRef:
        match node.kind:
            case "and":
                op: object = PointwiseMin()
            case "or":
                op = PointwiseMax()
            case "until":
                lo, hi = _require_interval(node)
                op = PcTimedUntil(lo, hi)
            case _:
                raise NotImplementedError(f"unknown STL binary operator {node.kind}")
        return self._emitter.emit(node, op, (left.output_id, right.output_id))

    def nary(self, node: Node, children: tuple[DagRef, ...]) -> DagRef:
        raise NotImplementedError(f"n-ary STL nodes are not supported: {node.kind!r}")


class _Executor:
    def __init__(self, signal: Signal, dt: float) -> None:
        self._signal = signal
        self._dt = dt
        self._arith = ArithmeticOpEvaluator(signal)

    def execute(
        self,
        dag: ComputationDag,
        trace: dict[int, tuple[Node | ArithNode, str]],
    ) -> _PymtlResult:
        outputs: dict[str, NDArray[np.floating]] = {}
        for node in dag.nodes:
            inputs = tuple(outputs[input_id] for input_id in node.inputs)
            outputs[node.id] = self._evaluate_op(node.op, inputs)
        return _PymtlResult(robustness=outputs[dag.output], trace=trace, outputs=outputs)

    def _evaluate_op(
        self, op: object, inputs: tuple[NDArray[np.floating], ...]
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
                # py-mtl has no equality predicate (its atoms are bare signal
                # names), so this has no cross-validated analogue; we follow
                # tidystl's own robustness convention, matching the rtamt backend.
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
            case PcWindowMin(lo=lo, hi=hi):
                (child,) = inputs
                return pc_window_reduce(
                    self._signal.times, child, lo=lo, hi=hi, dt=self._dt, reducer=np.min
                )
            case PcWindowMax(lo=lo, hi=hi):
                (child,) = inputs
                return pc_window_reduce(
                    self._signal.times, child, lo=lo, hi=hi, dt=self._dt, reducer=np.max
                )
            case PcTimedUntil(lo=lo, hi=hi):
                left, right = inputs
                return pc_timed_until(self._signal.times, left, right, lo=lo, hi=hi, dt=self._dt)
            case _:
                raise NotImplementedError(f"unknown DAG op {type(op).__name__}")


# Public aliases so other modules can import without reaching into private names.
PymtlDagBuilder = _DagBuilder
PymtlExecutor = _Executor
PymtlResult = _PymtlResult


class PymtlBackend(EvaluationBackend):
    """py-mtl compatible backend over piecewise-constant non-uniform traces."""

    name = "pymtl"

    def __init__(self, dt: float = 0.1) -> None:
        self.dt = dt

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> _PymtlResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        dag, trace = _DagBuilder().build(formula)
        return _Executor(signal, self.dt).execute(dag, trace)
