"""Named (model, spec) benchmarks and the semantic guardrails that keep backend
divergence attributable to principled semantics (not tool quirks).

Guardrails (design section 5):
  - top-level node is a temporal operator (avoids Breach extend-penultimate);
  - no equality predicate (avoids Breach BigM);
  - every predicate is a single-variable half-space (keeps taliro valid).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import models

from tidystl import parse
from tidystl.core.nodes import Node

_TEMPORAL_TOP = {"always", "eventually", "until"}


def iter_nodes(node: Node) -> Iterator[Node]:
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def _is_single_var_halfspace(pred: Node) -> bool:
    op = pred.attrs.get("op")
    if op == "==":
        return False
    left = pred.attrs.get("left")
    right = pred.attrs.get("right")
    if not isinstance(left, Node) or not isinstance(right, Node):
        return False
    # exactly one var node and one const node, var is a bare variable
    kinds = {left.kind, right.kind}
    return kinds == {"var", "const"}


def assert_principled(spec: str) -> None:
    """Raise ValueError if ``spec`` violates a guardrail."""
    phi = parse(spec)
    if phi.kind not in _TEMPORAL_TOP:
        raise ValueError(
            f"spec top-level must be a temporal operator {_TEMPORAL_TOP}, got {phi.kind!r}: {spec}"
        )
    for n in iter_nodes(phi):
        if n.kind == "predicate":
            if n.attrs.get("op") == "==":
                raise ValueError(f"equality predicate is not allowed (Breach BigM): {spec}")
            if not _is_single_var_halfspace(n):
                raise ValueError(f"predicate must be a single-variable half-space: {spec}")


def assert_on_grid(spec: str, dt: float, tol: float = 1e-9) -> None:
    """Raise ValueError if any temporal interval bound is not a multiple of dt."""
    phi = parse(spec)
    for n in iter_nodes(phi):
        interval = n.attrs.get("interval")
        if interval is None:
            continue
        for bound in interval:
            ratio = bound / dt
            if abs(ratio - round(ratio)) > tol:
                raise ValueError(f"interval bound {bound} is not aligned to dt={dt}: {spec}")


@dataclass(frozen=True)
class Benchmark:
    name: str
    model: models.OdeModel
    spec_template: str          # e.g. "G[0.0,5.0](x <= {thr})"
    thresholds: tuple[float, ...]
    hero_threshold: float

    def spec(self, threshold: float | None = None) -> str:
        """The STL spec string at ``threshold`` (hero threshold if None)."""
        thr = self.hero_threshold if threshold is None else threshold
        return self.spec_template.format(thr=thr)


def _make(name: str, template: str, thresholds: tuple[float, ...], hero: float) -> Benchmark:
    b = Benchmark(name=name, model=models.MODELS[name], spec_template=template,
                  thresholds=thresholds, hero_threshold=hero)
    for thr in thresholds:
        assert_principled(b.spec(thr))
        assert_on_grid(b.spec(thr), b.model.dt)
    return b


# Hard on-grid bounded-safety specs near each model's reachable extremum.
# Reachable maxima (full throttle): m1 v~4.94, m2 |x|~0.58, m3 h2~2.14.
BENCHMARKS: dict[str, Benchmark] = {
    "m2_mass_spring": _make("m2_mass_spring", "G[0.0,5.0](x <= {thr})",
                            (0.40, 0.45, 0.50, 0.54), 0.45),
    "m1_speed": _make("m1_speed", "G[0.0,8.0](v <= {thr})",
                      (4.3, 4.5, 4.7, 4.85), 4.5),
    "m3_coupled": _make("m3_coupled", "G[0.0,8.0](h2 <= {thr})",
                        (1.6, 1.8, 2.0), 1.8),
}

HERO = "m2_mass_spring"
