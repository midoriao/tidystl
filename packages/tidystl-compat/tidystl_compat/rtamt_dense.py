"""RTAMT dense-time-compatible backend.

Reproduces RTAMT's dense-time STL robustness semantics in pure Python, as a
counterpart to the discrete-time ``RtamtBackend``. RTAMT models dense-time
signals as piecewise-constant (PWC), right-continuous and held from the left:
for all ``t in [t_i, t_{i+1})`` the value is ``w(t_i)`` (arXiv:2501.18608).

Differences from the discrete backend:

- Bounded ``always``/``eventually`` reduce over the PWC segments that OVERLAP the
  real-time window ``[t+a, t+b]`` -- i.e. the samples held over that window,
  including the segment active at ``t+a`` (held from an earlier sample) and the
  final sample held to ``+inf``. This is sample-and-hold, not interpolation.
- The sampling grid need not be uniform; windows are resolved in real time.
- End of trace truncates the domain rather than padding ``inf``/``-inf``. On the
  fixed output grid this surfaces as the last in-window value being held, since
  once the window starts past the final sample only that held value is visible.
- Bounded ``until`` follows RTAMT's dense composition, which differs from the
  discrete witness-exclusive rule and can disagree with it on the shared domain:
  ``l U[a,b] r`` is ``F[a,b] r  AND  G[0,a] (l U r)`` (with the ``G[0,a]`` factor
  dropped when ``a = 0``), where ``l U r`` is the unbounded until.

Equality predicates use the metric ``-|lhs-rhs|`` (matching RTAMT), unlike
Breach's BigM sign semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tidystl.backends._pl_dag import (
    BoundedUntilKernel,
    PLDagBuilder,
    PLExecutor,
    PLResult,
    WindowMax,
    WindowMin,
)
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal

#: Tolerance for matching window endpoints against sample times.
_EPS = 1e-9


def _dense_window_reduce(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    start: float,
    end: float,
    reducer: Callable[..., Any],
) -> NDArray[np.floating]:
    """Sample-and-hold sliding reduce over the PWC window ``[t+start, t+end]``.

    For each output time ``t``, reduce over the PWC segments overlapping the
    real-time window ``[t+start, t+end]``: the segment active at the window start
    (held from the left), every sample inside the window, and the final sample
    held to ``+inf``. With a sorted ``times`` grid this is the contiguous block
    ``rho[:, k_start : k_end + 1]`` where ``k_start``/``k_end`` are the last
    sample indices at or before the window's start/end.
    """
    t_len = len(times)
    out = np.empty_like(rho)
    last = float(times[-1])

    for i in range(t_len):
        lo = float(times[i]) + start
        hi = float(times[i]) + end

        # Index of the segment held over the window start (the sample active at
        # ``lo``), and the last sample at or before the window end.
        k_start = int(np.searchsorted(times, lo + _EPS, side="right")) - 1
        k_end = int(np.searchsorted(times, min(hi, last) + _EPS, side="right")) - 1
        k_start = max(k_start, 0)
        k_end = max(k_end, k_start)

        out[:, i] = reducer(rho[:, k_start : k_end + 1], axis=-1)

    return out


def _unbounded_until(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> NDArray[np.floating]:
    """RTAMT dense unbounded until via the backward max-min recurrence.

    ``U[T-1] = min(l, r)`` and
    ``U[i] = max(min(l[i], r[i]), min(l[i], U[i+1]))``.
    """
    out = np.empty_like(left)
    out[:, -1] = np.minimum(left[:, -1], right[:, -1])
    for i in range(left.shape[-1] - 2, -1, -1):
        hold = np.minimum(left[:, i], out[:, i + 1])
        out[:, i] = np.maximum(np.minimum(left[:, i], right[:, i]), hold)
    return out


def _dense_bounded_until(
    times: NDArray[np.floating],
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    start: float,
    end: float,
) -> NDArray[np.floating]:
    """RTAMT dense bounded until: ``F[a,b] r AND G[0,a] (l U r)`` (drop G if a=0)."""
    eventually_r = _dense_window_reduce(times, right, start, end, np.max)
    unbounded = _unbounded_until(left, right)
    if start > _EPS:
        guarded = _dense_window_reduce(times, unbounded, 0.0, start, np.min)
        return np.minimum(eventually_r, guarded)
    return np.minimum(eventually_r, unbounded)


class _RtamtDenseExecutor(PLExecutor):
    """PL executor with RTAMT dense-time (PWC) window and until kernels."""

    def _evaluate_op(
        self,
        op: object,
        inputs: tuple[NDArray[np.floating], ...],
    ) -> NDArray[np.floating]:
        match op:
            case WindowMin(start=start, end=end):
                (child,) = inputs
                return _dense_window_reduce(self.signal.times, child, start, end, np.min)
            case WindowMax(start=start, end=end):
                (child,) = inputs
                return _dense_window_reduce(self.signal.times, child, start, end, np.max)
            case BoundedUntilKernel(start=start, end=end):
                left, right = inputs
                return _dense_bounded_until(self.signal.times, left, right, start, end)
            case _:
                return super()._evaluate_op(op, inputs)


class RtamtDenseBackend(EvaluationBackend):
    """Evaluation backend that targets RTAMT dense-time (PWC) compatibility."""

    name = "rtamt_dense"

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> PLResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        dag, trace = PLDagBuilder().build(formula)
        return _RtamtDenseExecutor(signal).execute(dag, trace)
