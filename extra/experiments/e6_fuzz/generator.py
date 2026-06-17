"""E6 generators: seeded random STL formulas and signals for the fuzz sweep.

All randomness flows through a single ``numpy.random.Generator`` so a run is
reproducible from its integer seed. Signals live on a uniform integer time
grid (step 1); formulas are well-typed ``tidystl`` ASTs over the signal's
variables.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
from numpy.random import Generator

from tidystl import Node

REGIMES = ("random_walk", "iid_uniform", "iid_gaussian")


def var_names(n_vars: int) -> list[str]:
    """Canonical variable names ``x0..x{n-1}`` for a signal with ``n_vars``."""
    return [f"x{i}" for i in range(n_vars)]


def gen_signal_arrays(
    rng: Generator, regime: str, n_vars: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(times, values)`` for one signal on the integer grid ``0..length-1``.

    ``values`` has shape ``(n_vars, length)``. Regimes: ``random_walk``
    (cumulative gaussian steps), ``iid_uniform``, ``iid_gaussian``.
    """
    times = np.arange(length, dtype=float)
    if regime == "random_walk":
        values = np.cumsum(rng.normal(0.0, 1.0, size=(n_vars, length)), axis=1)
    elif regime == "iid_uniform":
        values = rng.uniform(-5.0, 5.0, size=(n_vars, length))
    elif regime == "iid_gaussian":
        values = rng.normal(0.0, 2.0, size=(n_vars, length))
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return times, values


STL_OPS = frozenset({"not", "and", "or", "always", "eventually", "until"})
COMPARE_OPS = (">", ">=", "<", "<=")


def operators_in(node: Node) -> set[str]:
    """Set of STL operator kinds appearing in ``node`` (predicates excluded)."""
    found: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.kind in STL_OPS:
            found.add(current.kind)
        stack.extend(current.children)
    return found


def has_equality_predicate(node: Node) -> bool:
    """True if any predicate in ``node`` uses ``==``.

    Equality predicates are a semantic singularity that backends resolve
    incompatibly (Breach BigM, TaLiRo abstains, others use ``-|lhs-rhs|``),
    so aggregation stratifies agreement by this flag.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if current.kind == "predicate" and current.attrs.get("op") == "==":
            return True
        stack.extend(current.children)
    return False


def _affine_expr(terms: list[tuple[float, str]], const: float) -> Node:
    """Build the arithmetic AST ``sum_i coeff_i * x_i + const``."""
    nodes: list[Node] = [
        Node(kind="*", children=(
            Node(kind="const", attrs={"value": float(coeff)}),
            Node(kind="var", attrs={"name": name}),
        ))
        for coeff, name in terms
    ]
    nodes.append(Node(kind="const", attrs={"value": float(const)}))
    expr = nodes[0]
    for node in nodes[1:]:
        expr = Node(kind="+", children=(expr, node))
    return expr


def _gen_predicate(
    rng: Generator, variables: list[str], broad: bool, counter: Iterator[int]
) -> Node:
    """A half-space predicate ``<affine expr> <op> 0`` over a random subset.

    In ``broad`` mode it occasionally emits an equality op or a non-affine
    ``var*var`` term, both of which some backends (e.g. TaLiRo) reject -> the
    abstention diversity we want.
    """
    name = f"p{next(counter)}"
    k = int(rng.integers(1, len(variables) + 1))
    chosen = list(rng.choice(variables, size=k, replace=False))
    terms = [(float(rng.uniform(-2.0, 2.0)), v) for v in chosen]
    const = float(rng.uniform(-2.0, 2.0))

    if broad and len(chosen) >= 2 and rng.random() < 0.10:
        # Non-affine term: a product of two variables.
        left = Node(kind="*", children=(
            Node(kind="var", attrs={"name": chosen[0]}),
            Node(kind="var", attrs={"name": chosen[1]}),
        ))
    else:
        left = _affine_expr(terms, const)

    op = "==" if broad and rng.random() < 0.15 else str(rng.choice(COMPARE_OPS))

    return Node(kind="predicate", attrs={
        "name": name,
        "op": op,
        "left": left,
        "right": Node(kind="const", attrs={"value": 0.0}),
    })


def _gen_interval(rng: Generator, broad: bool) -> dict[str, object]:
    """Interval attrs for a temporal op; ``broad`` may omit it (unbounded)."""
    if broad and rng.random() < 0.10:
        return {}  # unbounded: backends requiring an interval will abstain
    start = int(rng.integers(0, 4))
    end = start + int(rng.integers(0, 5))
    return {"interval": (float(start), float(end))}


def gen_formula(rng: Generator, variables: list[str], max_depth: int, broad: bool) -> Node:
    """A random well-typed STL formula over ``variables``, bounded by depth.

    Operators: not/and/or/always/eventually/until plus half-space predicates.
    Determinism: identical ``(seed, variables, max_depth, broad)`` -> identical
    AST, because the counter and rng advance in fixed recursion order.
    """
    counter = itertools.count()
    kinds = ("predicate", "not", "and", "or", "always", "eventually", "until")

    def rec(depth: int) -> Node:
        if depth <= 0:
            return _gen_predicate(rng, variables, broad, counter)
        kind = str(rng.choice(kinds))
        if kind == "predicate":
            return _gen_predicate(rng, variables, broad, counter)
        if kind == "not":
            return Node(kind="not", children=(rec(depth - 1),))
        if kind in ("and", "or"):
            return Node(kind=kind, children=(rec(depth - 1), rec(depth - 1)))
        if kind in ("always", "eventually"):
            return Node(kind=kind, attrs=_gen_interval(rng, broad), children=(rec(depth - 1),))
        # until
        return Node(kind="until", attrs=_gen_interval(rng, broad), children=(rec(depth - 1), rec(depth - 1)))

    return rec(max_depth)
