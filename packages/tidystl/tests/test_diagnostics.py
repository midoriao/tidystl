"""Tests for divergence localization (tidystl.diagnostics.localize)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tidystl import DivergentNode, Signal, evaluate, localize, localize_results, parse
from tidystl.core.nodes import Node


def _const_signal(names: list[str], times: np.ndarray, value: float) -> Signal:
    return Signal.from_dict(
        times=times,
        values={n: np.full((1, len(times)), value) for n in names},
    )


def _traces_by_id(res: Any) -> dict[int, np.ndarray]:
    """node-identity -> trace array, for the nodes a result exposes (Node is
    unhashable, so identity is keyed explicitly)."""
    return {id(n): np.asarray(res.trace_for(n)) for n in res.traced_nodes()}


def _lookup(by_id: dict[int, np.ndarray]) -> Callable[[Node], Any]:
    def get(node: Node) -> Any:
        try:
            return by_id[id(node)]
        except KeyError as exc:  # surfaced as "missing trace" by localize
            raise KeyError(node) from exc

    return get


def test_identical_backends_have_no_divergence() -> None:
    # High-level API: one formula, one signal, two backends (same here).
    phi = parse("G[0,3]((x >= 0) and F[0,1](y >= 0))")
    sig = _const_signal(["x", "y"], np.linspace(0, 5, 11), 1.0)
    assert localize(phi, sig, "native", "native") == []


def test_localizes_to_the_node_a_lookup_perturbs() -> None:
    phi = parse("G[0,3]((x >= 0) and F[0,1](y >= 0))")
    sig = _const_signal(["x", "y"], np.linspace(0, 3, 7), 2.0)
    res = evaluate(phi, sig)

    and_node = phi.children[0]
    eventually = and_node.children[1]
    assert eventually.kind == "eventually"

    base = _traces_by_id(res)
    perturbed = dict(base)
    perturbed[id(eventually)] = base[id(eventually)] + 5.0  # diverge only at F[0,1]

    found = localize_results(phi, _lookup(base), _lookup(perturbed), atol=1e-9)
    assert [d.node for d in found] == [eventually]
    assert isinstance(found[0], DivergentNode)
    assert found[0].is_root is False


def test_predicate_is_a_leaf() -> None:
    phi = parse("(x >= 0) and (y >= 0)")
    sig = _const_signal(["x", "y"], np.linspace(0, 2, 5), 1.0)
    res = evaluate(phi, sig)
    pred = phi.children[0]
    assert pred.kind == "predicate"
    base = _traces_by_id(res)
    perturbed = {**base, id(pred): base[id(pred)] - 3.0}
    found = localize_results(phi, _lookup(base), _lookup(perturbed))
    assert [d.node for d in found] == [pred]


def test_root_divergence_is_flagged() -> None:
    phi = parse("(x >= 0) or (y >= 0)")
    sig = _const_signal(["x", "y"], np.linspace(0, 2, 5), 1.0)
    res = evaluate(phi, sig)
    base = _traces_by_id(res)
    perturbed = {**base, id(phi): base[id(phi)] + 1.0}  # only the root 'or' differs
    found = localize_results(phi, _lookup(base), _lookup(perturbed))
    assert [d.node for d in found] == [phi]
    assert found[0].is_root is True


def test_multiple_independent_origins() -> None:
    phi = parse("(x >= 0) and (y >= 0)")
    sig = _const_signal(["x", "y"], np.linspace(0, 2, 5), 1.0)
    res = evaluate(phi, sig)
    left, right = phi.children
    base = _traces_by_id(res)
    # Diverge at BOTH predicate leaves: the root inherits, so it is not minimal.
    perturbed = {**base, id(left): base[id(left)] + 1.0, id(right): base[id(right)] + 1.0}
    found = localize_results(phi, _lookup(base), _lookup(perturbed))
    assert {id(d.node) for d in found} == {id(left), id(right)}


def test_first_divergent_index_is_reported() -> None:
    phi = parse("x >= 0")
    sig = _const_signal(["x"], np.linspace(0, 4, 5), 1.0)
    res = evaluate(phi, sig)
    base = _traces_by_id(res)
    bumped = base[id(phi)].copy()
    bumped[:, 2] += 9.0  # first disagreement at time index 2
    found = localize_results(phi, _lookup(base), _lookup({**base, id(phi): bumped}))
    assert len(found) == 1
    assert found[0].first_divergent_index == 2


def test_within_tolerance_counts_as_agreement() -> None:
    phi = parse("x >= 0")
    sig = _const_signal(["x"], np.linspace(0, 2, 3), 1.0)
    res = evaluate(phi, sig)
    base = _traces_by_id(res)
    nudged = {**base, id(phi): base[id(phi)] + 1e-12}
    assert localize_results(phi, _lookup(base), _lookup(nudged), atol=1e-9) == []


def test_equal_infinities_agree() -> None:
    phi = parse("x >= 0")
    sig = _const_signal(["x"], np.linspace(0, 2, 3), 1.0)
    res = evaluate(phi, sig)
    base = {k: np.full_like(v, np.inf) for k, v in _traces_by_id(res).items()}
    assert localize_results(phi, _lookup(base), _lookup(dict(base))) == []


def test_missing_on_one_side_is_a_divergence() -> None:
    # If only one source exposes a node's trace, that node diverges.
    phi = parse("x >= 0")
    sig = _const_signal(["x"], np.linspace(0, 2, 3), 1.0)
    res = evaluate(phi, sig)
    base = _traces_by_id(res)
    empty: dict[int, np.ndarray] = {}  # exposes nothing
    found = localize_results(phi, _lookup(base), _lookup(empty))
    assert [d.node for d in found] == [phi]


def test_high_level_localize_accepts_backend_names() -> None:
    # localize() evaluates internally; identical backends never diverge, and
    # localize_results on the same live result agrees too.
    phi = parse("G[0,2](x >= 0)")
    sig = _const_signal(["x"], np.linspace(0, 2, 5), 1.0)
    assert localize(phi, sig, "native", "native") == []
    res = evaluate(phi, sig)
    assert localize_results(phi, res, res) == []
