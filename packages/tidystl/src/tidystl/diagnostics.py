"""Divergence localization between two backend evaluations.

When two backends evaluate the *same* formula on the *same* signal but report
different robustness, :func:`localize` pins the disagreement to its origin in
the formula structure. Mirroring that same-formula/same-trace setup,
:func:`localize` takes one formula, one signal, and the two backends to compare,
evaluates both, and returns the *minimal divergent nodes*: the lowest nodes
whose own per-node robustness trace disagrees while every descendant's subtree
still agrees, so the divergence is introduced there rather than inherited from
below.

This is the mechanism the architecture's shared formula representation exists to
support: because every backend exposes per-node traces over one common syntax
tree (:meth:`EvaluationResult.trace_for`), two backends' intermediate outputs
align node by node and a single post-order walk localizes where they first part.

Predicates are localization leaves: the parser folds a predicate's internal
arithmetic into the predicate node's attributes, so the walk never descends into
it and none of the implicit-choice differences this surfaces concerns
predicate-internal arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from tidystl.core.backend_interface import EvaluationBackend, EvaluationResult
from tidystl.core.evaluator import BackendRegistry, evaluate
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal

#: A per-node trace source: either an evaluated result or a bare
#: ``node -> robustness-trace`` lookup (used by record-based aggregation that
#: does not keep a live :class:`EvaluationResult`).
TraceSource = EvaluationResult | Callable[[Node], Any]

# Sentinel for "this source has no trace for this node". Two sources that both
# lack a node agree there; one having it and the other not is a divergence.
_MISSING: Any = object()


@dataclass(frozen=True)
class DivergentNode:
    """One minimal divergent node found by :func:`localize`.

    ``node`` is the formula AST node where the divergence originates; ``index``
    is its 0-based position in the post-order walk; ``is_root`` flags the whole
    formula's root; ``first_divergent_index`` is the flat index into the
    (batch-aligned) trace where the two sources first differ (the time step, for
    a single-trace signal).
    """

    node: Node
    index: int
    is_root: bool
    first_divergent_index: int


def _as_lookup(source: TraceSource) -> Callable[[Node], Any]:
    """Normalize a trace source to a ``node -> trace`` callable that returns
    :data:`_MISSING` for nodes the source does not expose."""
    if hasattr(source, "trace_for"):
        if not getattr(source, "has_trace", False):
            raise ValueError(
                "EvaluationResult exposes no per-node traces (has_trace is False); "
                "localization needs a trace-capable backend on both sides"
            )
        trace_for = source.trace_for  # type: ignore[union-attr]
    elif callable(source):
        trace_for = source
    else:
        raise TypeError(f"expected an EvaluationResult or a node->trace callable, got {source!r}")

    def get(node: Node) -> Any:
        try:
            return trace_for(node)
        except KeyError:
            return _MISSING

    return get


def _close_mask(a: Any, b: Any, *, atol: float) -> Any:
    aa, bb = np.asarray(a), np.asarray(b)
    if aa.shape != bb.shape:
        return None  # shape mismatch: treated as fully divergent
    exact = aa == bb  # also covers equal infinities
    with np.errstate(invalid="ignore"):
        return exact | (np.abs(aa - bb) <= atol)


def _agree(a: Any, b: Any, *, atol: float) -> bool:
    if a is _MISSING or b is _MISSING:
        return a is _MISSING and b is _MISSING
    mask = _close_mask(a, b, atol=atol)
    return mask is not None and bool(np.all(mask))


def _first_divergent_index(a: Any, b: Any, *, atol: float) -> int:
    if a is _MISSING or b is _MISSING:
        return 0
    mask = _close_mask(a, b, atol=atol)
    if mask is None:
        return 0
    diverging = np.flatnonzero(~mask)
    return int(diverging[0]) if diverging.size else 0


def localize_results(
    formula: Node,
    source_a: TraceSource,
    source_b: TraceSource,
    *,
    atol: float = 1e-9,
) -> list[DivergentNode]:
    """Minimal divergent nodes between two precomputed evaluations of ``formula``.

    Lower-level form of :func:`localize`. ``source_a`` and ``source_b`` are the
    two backends' results for the same ``formula`` (the same parsed AST, so
    per-node traces align by node identity). Each may be an
    :class:`EvaluationResult` or a bare ``node -> trace`` callable. Use this when
    you already evaluated, or when you only hold per-node traces (e.g.
    record-based aggregation that does not re-evaluate); otherwise prefer
    :func:`localize`, which takes one signal and the two backends and so cannot
    be handed mismatched inputs.

    Two traces agree when they are equal within absolute tolerance ``atol``
    (equal infinities count as agreement). The walk is post-order, so returned
    nodes are lowest-first; an empty list means the two agree on every node.
    Several independent divergence origins yield several entries.
    """
    lookup_a = _as_lookup(source_a)
    lookup_b = _as_lookup(source_b)

    # Post-order walk of the formula, recording each node's child positions.
    order: list[Node] = []
    children_of: list[list[int]] = []

    def walk(node: Node) -> int:
        kids = [walk(child) for child in node.children]
        idx = len(order)
        order.append(node)
        children_of.append(kids)
        return idx

    walk(formula)
    last = len(order) - 1

    subtree_agrees: list[bool] = [False] * len(order)
    minimal: list[DivergentNode] = []
    for idx, node in enumerate(order):
        trace_a = lookup_a(node)
        trace_b = lookup_b(node)
        node_agrees = _agree(trace_a, trace_b, atol=atol)
        children_agree = all(subtree_agrees[c] for c in children_of[idx])
        subtree_agrees[idx] = node_agrees and children_agree
        if not node_agrees and children_agree:
            minimal.append(
                DivergentNode(
                    node=node,
                    index=idx,
                    is_root=idx == last,
                    first_divergent_index=_first_divergent_index(trace_a, trace_b, atol=atol),
                )
            )
    return minimal


def localize(
    formula: Node,
    signal: Signal | TorchSignal,
    backend_a: str | EvaluationBackend,
    backend_b: str | EvaluationBackend,
    *,
    atol: float = 1e-9,
    registry: BackendRegistry | None = None,
) -> list[DivergentNode]:
    """Localize where two backends diverge on the same formula and signal.

    Evaluates ``formula`` over ``signal`` under ``backend_a`` and ``backend_b``
    (each a registered backend name or a backend instance, as for
    :func:`~tidystl.robustness`) and returns the minimal divergent nodes: the
    lowest sub-formulas whose own robustness trace disagrees while every
    descendant still agrees. Returned nodes are post-order (lowest first); an
    empty list means the two backends agree on every node.

    Passing one formula and one signal is deliberate: it is the
    same-formula/same-trace comparison the tool is built around, so the two
    sides cannot be accidentally evaluated on different inputs. To localize
    results you already have (or raw per-node traces), use
    :func:`localize_results`.
    """
    result_a = evaluate(formula, signal, backend=backend_a, registry=registry)
    result_b = evaluate(formula, signal, backend=backend_b, registry=registry)
    return localize_results(formula, result_a, result_b, atol=atol)
