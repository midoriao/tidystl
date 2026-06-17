from __future__ import annotations

import math

from tidystl.core.nodes import Node


def _interval(node: Node) -> tuple[float, float]:
    match node.attrs.get("interval"):
        case (float(start), float(end)):
            return start, end
        case _:
            raise ValueError(f"temporal operator {node.kind} missing interval")


def horizon(node: Node) -> float:
    """Future reach of the formula.

    Returns the maximum look-ahead across all temporal operators.
    """
    match node.kind, node.children:
        case "predicate", _:
            return 0.0
        case "not", (child,):
            return horizon(child)
        case "always" | "eventually", (child,):
            return horizon(child) + _interval(node)[1]
        case "and" | "or", (left, right):
            return max(horizon(left), horizon(right))
        case "until", (left, right):
            return max(horizon(left), horizon(right)) + _interval(node)[1]
        case _:
            raise NotImplementedError(f"horizon not implemented for kind {node.kind!r}")


def required_max_gap(node: Node) -> float:
    """Smallest temporal-window upper bound `b` across the formula.

    For each temporal operator with interval `[a, b]`, this returns the
    minimum `b` (not the minimum width `b - a`). The intent is a
    conservative upper bound on the sampling gap that still lets every
    temporal window be observed from the origin sample: a window
    `[t + a, t + b]` always reaches at least up to `t + b`, so guaranteeing
    a sample by `b` is sufficient regardless of `a`. Predicates contribute
    no constraint and return `math.inf`.
    """
    match node.kind, node.children:
        case "predicate", _:
            return math.inf
        case "not", (child,):
            return required_max_gap(child)
        case "always" | "eventually", (child,):
            return min(required_max_gap(child), _interval(node)[1])
        case "and" | "or", (left, right):
            return min(required_max_gap(left), required_max_gap(right))
        case "until", (left, right):
            return min(
                required_max_gap(left),
                required_max_gap(right),
                _interval(node)[1],
            )
        case _:
            raise NotImplementedError(f"required_max_gap not implemented for kind {node.kind!r}")
