"""E6 exp 6 -- systematic divergence localization.

The hand-written localizer in ``e2_localization`` finds the minimal divergent
node systematically, but *names the responsible choice* with a hand-authored
per-backend op vocabulary (``primitive_op`` in ``e2_localization/aggregate.py``,
including special cases like ``"+ terminal-step extension"``). This module
replaces that ad-hoc attribution with a principled one built on the generic
backend:

* WHERE: the minimal divergent node (same tree diff, kept).
* WHICH CHOICE: by *ablation* over the Core 6 config axes -- the minimal set of
  axes whose flip (from one config toward the other) reproduces the divergence
  at that node. The axis names come from the configuration, not a hand vocabulary.
  Crucially this isolates the responsible axis even when two tools differ on
  several axes at once (a ceteris-paribus comparison real tools cannot give).
* UNEXPLAINED: when no config reproduces the real tool (open-world residual), the
  node is flagged as implementation-specific rather than mislabeled as an axis.

Public API (all pure, used by the tests and the ``main`` demonstrations):
  ``attribute(cfg_a, cfg_b, phi, signal)`` -> per minimal divergent node, the
      minimal sufficient axis set (or ``None`` if the full flip cannot reproduce).
  ``real_divergence_nodes(cfg, tool, phi, signal)`` -> nodes where the generic
      config diverges from the real tool.
  ``is_open_world(tool, probe)`` -> True iff no Core 6 config reproduces the real
      tool's output on this probe.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    AXES,
    Probe,
    documented_configs,
    load_battery,
    observed_signature,
)
from extra.experiments.e6_fingerprint.openworld import best_match, total_residual  # noqa: E402
import tidystl_compat  # noqa: E402
from tidystl import Signal, evaluate, parse, use  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402
from tidystl.diagnostics import localize_results  # noqa: E402
from tidystl_compat import GenericBackend, GenericConfig  # noqa: E402

use(tidystl_compat)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def node_label(node: Node) -> str:
    """Compact node label, e.g. ``F[2,4]``, ``and``, ``predicate >=``."""
    match node.kind:
        case "always" | "eventually" | "until":
            iv = node.attrs.get("interval")
            sym = {"always": "G", "eventually": "F", "until": "U"}[node.kind]
            if isinstance(iv, tuple) and len(iv) == 2:
                return f"{sym}[{iv[0]:g},{iv[1]:g}]"
            return sym
        case "predicate":
            return f"predicate {node.attrs.get('op', '')}".strip()
        case _:
            return node.kind


def _trace(result: object, node: Node) -> NDArray[np.floating] | None:
    """Per-node robustness row, or None if the backend cannot expose it."""
    try:
        arr = np.asarray(result.trace_for(node), dtype=np.float64)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 -- backend may not expose this node's trace
        return None
    return arr[0] if arr.ndim == 2 else arr


def _agree(a: NDArray[np.floating] | None, b: NDArray[np.floating] | None, atol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and bool(np.allclose(a, b, atol=atol, equal_nan=True))


def minimal_divergent_nodes(
    phi: Node, res_a: object, res_b: object, atol: float = 1e-6
) -> list[Node]:
    """Nodes whose own trace differs while every STL descendant agrees (post-order).

    Thin wrapper over the shared core localizer
    (:func:`tidystl.localize_results`): the tree diff lives in the tool now, and
    this module keeps only the experiment's axis-ablation attribution on top of
    it. These backend results are already evaluated, so this uses the
    result-based form rather than ``localize(formula, signal, ...)``.
    """
    return [dv.node for dv in localize_results(phi, res_a, res_b, atol=atol)]


def responsible_axes(
    cfg_a: GenericConfig,
    cfg_b: GenericConfig,
    phi: Node,
    signal: object,
    node: Node,
    atol: float = 1e-6,
) -> list[str] | None:
    """Minimal set of differing axes whose flip (cfg_a -> cfg_b) reproduces cfg_b's
    value at ``node``. ``None`` if even the full flip does not (open-world residual).

    Inert axes (those that do not affect this node on this case) are excluded
    automatically: only a subset that actually reproduces the value is returned.
    """
    diff = [ax for ax in AXES if getattr(cfg_a, ax) != getattr(cfg_b, ax)]
    target = _trace(GenericBackend(cfg_b).evaluate(phi, signal), node)
    for k in range(1, len(diff) + 1):
        for subset in combinations(diff, k):
            flipped = replace(cfg_a, **{ax: getattr(cfg_b, ax) for ax in subset})
            t = _trace(GenericBackend(flipped).evaluate(phi, signal), node)
            if _agree(t, target, atol):
                return list(subset)
    return None


def attribute(
    cfg_a: GenericConfig, cfg_b: GenericConfig, phi: Node, signal: object, atol: float = 1e-6
) -> list[tuple[Node, list[str] | None]]:
    """Per minimal divergent node between two configs, the minimal responsible axis set."""
    res_a = GenericBackend(cfg_a).evaluate(phi, signal)
    res_b = GenericBackend(cfg_b).evaluate(phi, signal)
    nodes = minimal_divergent_nodes(phi, res_a, res_b, atol)
    return [(n, responsible_axes(cfg_a, cfg_b, phi, signal, n, atol)) for n in nodes]


def real_divergence_nodes(
    cfg: GenericConfig, tool: str, phi: Node, signal: object, atol: float = 1e-6
) -> list[Node]:
    """Minimal divergent nodes between ``generic(cfg)`` and the real tool."""
    gen = GenericBackend(cfg).evaluate(phi, signal)
    real = evaluate(phi, signal, backend=tool)  # validated against the real tool
    return minimal_divergent_nodes(phi, gen, real, atol)


def is_open_world(tool: str, probe: Probe) -> bool:
    """True iff no Core 6 config reproduces the real tool's root output on this probe."""
    observed = observed_signature(tool, [probe])
    _, residual = best_match(observed, [probe])
    return total_residual(residual) > 0.0


# ---------------------------------------------------------------------------
# Demonstrations


def _fmt(axes: list[str] | None) -> str:
    return "OPEN-WORLD residual (no config)" if axes is None else str(axes)


def demo_ceteris_paribus(probe: Probe, base: GenericConfig, axis: str, alt: str) -> None:
    cfg_b = replace(base, **{axis: alt})
    phi = parse(probe.formula_text)
    logger.info("  case=%s  flip %s: %s -> %s", probe.name, axis, getattr(base, axis), alt)
    rows = attribute(base, cfg_b, phi, probe.signal)
    if not rows:
        logger.info("    (no divergence on this case)")
    for node, axes in rows:
        logger.info(
            "    minimal divergent node `%s` -> responsible axis: %s", node_label(node), _fmt(axes)
        )


def demo_cross_tool(probe: Probe, tool_a: str, tool_b: str, docs: dict[str, GenericConfig]) -> None:
    cfg_a, cfg_b = docs[tool_a], docs[tool_b]
    phi = parse(probe.formula_text)
    diff = [ax for ax in AXES if getattr(cfg_a, ax) != getattr(cfg_b, ax)]
    logger.info("  case=%s  %s vs %s  (configs differ on: %s)", probe.name, tool_a, tool_b, diff)
    for node, axes in attribute(cfg_a, cfg_b, phi, probe.signal):
        logger.info(
            "    minimal divergent node `%s` -> responsible axis set: %s",
            node_label(node),
            _fmt(axes),
        )


def demo_open_world(probe: Probe, tool: str, docs: dict[str, GenericConfig]) -> None:
    cfg = docs[tool]
    phi = parse(probe.formula_text)
    nodes = real_divergence_nodes(cfg, tool, phi, probe.signal)
    logger.info("  case=%s  generic(config_%s) vs real %s", probe.name, tool, tool)
    if not nodes:
        logger.info("    (reproduces the real tool exactly)")
        return
    verdict = (
        "OPEN-WORLD residual (no config)"
        if is_open_world(tool, probe)
        else "explained by some config"
    )
    for node in nodes:
        logger.info("    minimal divergent node `%s` -> %s", node_label(node), verdict)


def main() -> None:
    battery = {p.name: p for p in load_battery()}
    docs = documented_configs()
    native = docs["native"]

    logger.info("E6 exp 6 -- systematic divergence localization\n")

    logger.info("(A) ceteris-paribus axis attribution (flip one axis, all else fixed):")
    demo_ceteris_paribus(battery["div_boundary_F24"], native, "boundary", "pessimistic")
    demo_ceteris_paribus(battery["div_terminal_and"], native, "terminal", "extend_penultimate")
    demo_ceteris_paribus(battery["multivar_affine"], native, "predicate", "euclidean")

    logger.info("\n(B) cross-tool decomposition (real tool pair -> axis set per node):")
    demo_cross_tool(battery["div_boundary_F24"], "native", "rtamt", docs)
    demo_cross_tool(battery["div_terminal_and"], "native", "breach", docs)

    logger.info("\n(C) open-world residual (Breach window-endpoint quirk):")
    quirk = Probe(
        name="interp_sparse",
        formula_text="G[0.5,1.5](x > 0)",
        signal=Signal.from_dict(np.array([0.0, 2.0, 4.0]), {"x": np.array([-1.0, 10.0, -1.0])}),
    )
    demo_open_world(quirk, "breach", docs)


if __name__ == "__main__":
    main()
