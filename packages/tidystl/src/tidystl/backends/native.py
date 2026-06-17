"""Native tidystl backend.

Implements tidystl's principled piecewise-linear (PL) dense-time semantics:

- Predicates are evaluated at every sample point via linear interpolation.
- G[a,b] and F[a,b] compute the exact PL window min/max over the interval
  [t+a, t+b], including interpolated boundary values at t+a and t+b when
  those do not coincide with sample points.
- No Breach implementation-specific adjustments.

Use BreachBackend when you need bit-for-bit Breach compatibility.
"""

from __future__ import annotations

from tidystl.backends._pl_dag import PLDagBuilder, PLExecutor, PLResult
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal


class NativeBackend(EvaluationBackend):
    """tidystl's default PL dense-time evaluation backend."""

    name = "native"

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> PLResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        dag, trace = PLDagBuilder().build(formula)
        return PLExecutor(signal).execute(dag, trace)
