"""TaLiRo (dp_taliro)-compatible backend.

Reproduces TaLiRo robustness semantics in pure Python (no MATLAB):

- Predicates are linear half-spaces ``A x <= b``; robustness is the Euclidean
  signed distance ``(b - A x)/||A||``, i.e. it normalizes by ``||A||``.
  Single-variable predicates (``||A||=1``) coincide with the raw signed value;
  multi-variable predicates (``x+y>=0``) divide by ``||A||``.
- Timed operators reduce over sample points inside the closed real-time window
  ``[t+a, t+b]`` only: no endpoint interpolation, no past-end extension; an
  empty window yields the reduction identity (+inf for always, -inf for
  eventually / until witnesses).
- Equality and non-linear predicates are not TaLiRo half-spaces and raise.

dp_taliro reports a scalar (robustness at t=0); this backend produces the full
(N, T) trace and ``rho[:, 0]`` is the validated quantity.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

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
from tidystl.backends.helper import DagRef
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.helper import fold_tree
from tidystl.core.nodes import ArithNode, Node
from tidystl.core.signal import Signal, TorchSignal


def affine_coeffs(node: ArithNode) -> tuple[float, dict[str, float]]:
    """Fold an arithmetic AST into ``(constant, {variable: coefficient})``.

    Raises ``NotImplementedError`` for any non-affine sub-expression, since
    those cannot be TaLiRo half-spaces.
    """
    kind = node.kind
    if kind == "const":
        value = node.attrs["value"]
        if not isinstance(value, float):
            raise TypeError("const node requires float value attr")
        return value, {}
    if kind == "var":
        name = node.attrs["name"]
        if not isinstance(name, str):
            raise TypeError("var node requires str name attr")
        return 0.0, {name: 1.0}
    if kind in ("+", "-"):
        lc, lco = affine_coeffs(node.children[0])
        rc, rco = affine_coeffs(node.children[1])
        sign = 1.0 if kind == "+" else -1.0
        coeffs = dict(lco)
        for var, coeff in rco.items():
            coeffs[var] = coeffs.get(var, 0.0) + sign * coeff
        return lc + sign * rc, coeffs
    if kind == "*":
        lc, lco = affine_coeffs(node.children[0])
        rc, rco = affine_coeffs(node.children[1])
        if not lco:
            return lc * rc, {var: lc * coeff for var, coeff in rco.items()}
        if not rco:
            return lc * rc, {var: coeff * rc for var, coeff in lco.items()}
        raise NotImplementedError(
            "TaLiRo predicates must be linear half-spaces; "
            "product of two variable terms is not affine"
        )
    if kind == "/":
        lc, lco = affine_coeffs(node.children[0])
        rc, rco = affine_coeffs(node.children[1])
        if rco:
            raise NotImplementedError(
                "TaLiRo predicates must be linear; division by a variable term is not affine"
            )
        return lc / rc, {var: coeff / rc for var, coeff in lco.items()}
    raise NotImplementedError(
        f"TaLiRo predicate sub-expression {kind!r} is not an affine half-space form"
    )


def predicate_norm(left: ArithNode, right: ArithNode) -> float:
    """Euclidean norm ``||A||`` of the predicate ``left <op> right``.

    ``A`` is the coefficient vector of ``left - right`` over the signal
    variables. Raises if the predicate has no variables (``||A|| = 0``).
    """
    _, lco = affine_coeffs(left)
    _, rco = affine_coeffs(right)
    coeffs = dict(lco)
    for var, coeff in rco.items():
        coeffs[var] = coeffs.get(var, 0.0) - coeff
    norm = math.sqrt(sum(coeff * coeff for coeff in coeffs.values()))
    if norm == 0.0:
        raise NotImplementedError("TaLiRo predicate has no signal variables; not a half-space")
    return norm


def taliro_sliding_reduce(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    start: float,
    end: float,
    reducer: Callable[..., NDArray[np.floating]],
    empty_value: float,
) -> NDArray[np.floating]:
    """TaLiRo (dp_taliro) timed reduction over the closed window [t+start, t+end].

    Reduces over sample points inside the real-time window ONLY: no endpoint
    interpolation and no past-end extension. A window containing no samples
    yields ``empty_value`` (+inf for always/min, -inf for eventually/max),
    the identity of the reduction.

    Window membership uses a 1e-9 absolute tolerance, which assumes time stamps
    on the order of unity (the regime of tidystl signals and the dp_taliro fixtures).
    """

    t_len = len(times)
    out = np.empty_like(rho)
    eps = 1e-9
    for i in range(t_len):
        t_lo = float(times[i]) + start
        t_hi = float(times[i]) + end
        j_start = int(np.searchsorted(times, t_lo - eps, side="left"))
        j_end = int(np.searchsorted(times, t_hi + eps, side="right"))
        if j_start < j_end:
            out[:, i] = reducer(rho[:, j_start:j_end], axis=-1)
        else:
            out[:, i] = empty_value
    return out


def taliro_eval_until(
    times: NDArray[np.floating],
    rho_p: NDArray[np.floating],
    rho_q: NDArray[np.floating],
    start: float,
    end: float,
) -> NDArray[np.floating]:
    """TaLiRo bounded-until over the closed real-time window [t+start, t+end].

    ``rho(i) = max over witnesses j in the window of
    min( rho_q(j), min over i<=k<=j of rho_p(k) )``. Empty witness set yields
    ``-inf``; no past-end extension.

    Window membership uses a 1e-9 absolute tolerance, which assumes time stamps
    on the order of unity (the regime of tidystl signals and the dp_taliro fixtures).
    """

    n, t_len = rho_p.shape
    out = np.full((n, t_len), -np.inf, dtype=rho_p.dtype)
    eps = 1e-9
    for i in range(t_len):
        t_i = float(times[i])
        j_start = int(np.searchsorted(times, t_i + start - eps, side="left"))
        j_end = int(np.searchsorted(times, t_i + end + eps, side="right"))
        for j in range(max(j_start, i), j_end):
            q_val = rho_q[:, j]
            p_min = np.min(rho_p[:, i : j + 1], axis=-1)
            out[:, i] = np.maximum(out[:, i], np.minimum(q_val, p_min))
    return out


@dataclass(frozen=True)
class TaliroIneq:
    """Predicate op carrying the precomputed ``||A||`` normalization factor."""

    direction: Literal["ge", "le"]
    norm: float


class _TaliroDagBuilder(PLDagBuilder):
    """PL DAG builder that attaches each predicate's ``||A||`` and rejects ``==``."""

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
        if op_name == "==":
            raise NotImplementedError("TaLiRo half-spaces cannot represent equality predicates")
        left = fold_tree(left_node, self._arith)
        right = fold_tree(right_node, self._arith)
        direction: Literal["ge", "le"] = "ge" if op_name in (">", ">=") else "le"
        norm = predicate_norm(left_node, right_node)
        return self._emitter.emit(
            node, TaliroIneq(direction, norm), (left.output_id, right.output_id)
        )


class _TaliroExecutor(PLExecutor):
    """PL executor with TaLiRo predicate normalization and samples-only kernels."""

    def _evaluate_op(
        self,
        op: object,
        inputs: tuple[NDArray[np.floating], ...],
    ) -> NDArray[np.floating]:
        match op:
            case TaliroIneq(direction="ge", norm=norm):
                left, right = inputs
                return (left - right) / norm
            case TaliroIneq(direction="le", norm=norm):
                left, right = inputs
                return (right - left) / norm
            case WindowMin(start=start, end=end):
                (child,) = inputs
                return taliro_sliding_reduce(self.signal.times, child, start, end, np.min, np.inf)
            case WindowMax(start=start, end=end):
                (child,) = inputs
                return taliro_sliding_reduce(self.signal.times, child, start, end, np.max, -np.inf)
            case BoundedUntilKernel(start=start, end=end):
                left, right = inputs
                return taliro_eval_until(self.signal.times, left, right, start, end)
            case _:
                return super()._evaluate_op(op, inputs)


class TaliroBackend(EvaluationBackend):
    """Evaluation backend targeting TaLiRo (dp_taliro) runtime compatibility."""

    name = "taliro"

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> PLResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal, got {type(signal).__name__}")
        dag, trace = _TaliroDagBuilder().build(formula)
        return _TaliroExecutor(signal).execute(dag, trace)
