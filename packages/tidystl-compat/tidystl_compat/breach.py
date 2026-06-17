"""Breach-compatible backend.

Targets Breach (MATLAB) runtime semantics, including implementation-specific
behaviors not present in tidystl's principled PL semantics:

- Top-level binary And/Or: extends the penultimate robustness value to the
  final timestep, matching Breach's reported output convention.
- Windowed G[a,b]/F[a,b]: Breach's RobustEv kernel reduces over the sample
  points inside [t+a, t+b] ONLY -- it does not interpolate the signal at
  window endpoints (visible on non-uniform grids; matlab-run ground truth
  div_nonuniform_always, 2026-06-05). When the window contains no sample
  points, Breach uses the interpolated predicate value at t+a for the first
  sample and the current predicate value rho(t) for subsequent samples
  (div_interp_sparse); windows entirely past the signal end clamp to the
  last value.
- Equality predicates: Breach parses ``lhs == rhs`` into
  ``fun__zero(abs(rhs-lhs), zero_threshold__, true_value__, alpha__)`` and
  sets ``true_value__ = alpha__ = 10000`` on every formula
  (@STL_Formula/STL_Formula.m, Breach 1.11.4), yielding BigM sign semantics
  (+10000 when |lhs-rhs| <= 1e-13, -10000 otherwise) rather than the metric
  ``-|lhs-rhs|`` used by the other backends.

Use NativeBackend for tidystl's principled dense-time PL semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tidystl.backends import algorithms
from tidystl.backends._pl_dag import (
    Eq,
    PLDagBuilder,
    PLExecutor,
    PLResult,
    WindowMax,
    WindowMin,
)
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal

#: Breach equality-predicate constants. Breach parses ``lhs == rhs`` into
#: ``fun__zero(abs(rhs-lhs), zero_threshold__, true_value__, alpha__)``
#: (@STL_Formula/STL_Formula.m, @STL_Formula/private/fun__zero.m) and
#: post-parse sets ``true_value__ = alpha__ = 10000`` on every formula.
_EQ_ZERO_THRESHOLD = 1e-13
_EQ_TRUE_VALUE = 10000.0
_EQ_ALPHA = 10000.0


def _sliding_reduce_breach(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    start: float,
    end: float,
    reducer: Callable[..., Any],
) -> NDArray[np.floating]:
    """Breach-compatible sliding reduce over the window [t+start, t+end].

    Mimics Breach's RobustEv kernel (``alw`` is ``-RobustEv(-rho)``, so min
    and max share it), as pinned by the recorded and matlab-run ground truth:

    - Windows containing sample points reduce over those SAMPLES ONLY;
      Breach does not interpolate the signal at window endpoints. Visible
      on non-uniform grids (matlab-run div_nonuniform_always: G[0,1] at
      t=1.5 returns the in-window sample value, not the PL minimum that
      includes the interpolated right endpoint).
    - Empty interior windows: at the first sample (i=0), interpolate rho at
      t+start; at subsequent samples, return rho[:, i] (matlab-run
      div_interp_sparse).
    - Windows entirely past the signal end clamp to the last value.
    """
    t_len = len(times)
    out = np.empty_like(rho)

    for i in range(t_len):
        t_lo = float(times[i]) + start
        t_hi = float(times[i]) + end

        if t_lo > float(times[-1]):
            # Window entirely past signal end: clamp to last value.
            out[:, i] = rho[:, -1]
            continue

        t_lo_c = max(t_lo, float(times[0]))
        t_hi_c = min(t_hi, float(times[-1]))

        j_start = min(int(np.searchsorted(times, t_lo_c, side="left")), t_len)
        j_end = min(int(np.searchsorted(times, t_hi_c, side="right")), t_len)

        if j_start >= j_end:
            # No sample points in the window (window between consecutive samples).
            if i == 0:
                out[:, i] = algorithms.interp_at(times, rho, t_lo)
            else:
                out[:, i] = rho[:, i]
            continue

        # Samples-only reduction: no window-endpoint interpolation.
        out[:, i] = reducer(rho[:, j_start:j_end], axis=-1)

    return out


class _BreachExecutor(PLExecutor):
    """PL executor with Breach-specific overrides (windows, equality)."""

    def _evaluate_op(
        self,
        op: object,
        inputs: tuple[NDArray[np.floating], ...],
    ) -> NDArray[np.floating]:
        match op:
            case Eq():
                left, right = inputs
                return np.where(
                    np.abs(left - right) <= _EQ_ZERO_THRESHOLD,
                    _EQ_TRUE_VALUE,
                    -_EQ_ALPHA,
                )
            case WindowMin(start=start, end=end):
                (child,) = inputs
                return _sliding_reduce_breach(self.signal.times, child, start, end, np.min)
            case WindowMax(start=start, end=end):
                (child,) = inputs
                return _sliding_reduce_breach(self.signal.times, child, start, end, np.max)
            case _:
                return super()._evaluate_op(op, inputs)


class BreachBackend(EvaluationBackend):
    """Evaluation backend that targets Breach runtime compatibility."""

    name = "breach"

    @staticmethod
    def _extend_last(arr: NDArray[np.floating]) -> NDArray[np.floating]:
        result = arr.copy()
        if result.shape[-1] >= 2:
            result[:, -1] = result[:, -2]
        return result

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> PLResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        dag, trace = PLDagBuilder().build(formula)
        result = _BreachExecutor(signal).execute(dag, trace)
        if not (formula.kind in ("and", "or") and len(formula.children) == 2):
            return result

        # Breach extends the penultimate value at the final sample for top-level
        # binary boolean operators to match Breach's reported semantics.
        return result.with_top_output(dag.output, self._extend_last(result.robustness))
