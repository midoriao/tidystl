"""E6 exp 5 -- open-world residual and planted-mutant detection.

A tool whose behavior is explained by some Core 6 config has zero residual against
its best-matching config. A tool with a behavior outside the modeled axes leaves a
nonzero residual on the probes that exercise the unmodeled rule; we report and
localize it. A planted mutant (deliberately off-model) demonstrates the check fires.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tidystl_compat import GenericBackend, GenericConfig  # noqa: E402

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    TOOLS,
    Probe,
    ProbeValue,
    Signature,
    enumerate_configs,
    load_battery,
    observed_signature,
    signature,
)
from tidystl import parse  # noqa: E402
from tidystl.backends._pl_dag import Ineq, PLDagBuilder, PLExecutor, PLResult  # noqa: E402
from tidystl.core.backend_interface import EvaluationBackend  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402
from tidystl.core.signal import Signal, TorchSignal  # noqa: E402


def probe_residual(observed: ProbeValue, modeled: ProbeValue) -> float:
    """Residual between one observed and one modeled probe value.

    - both arrays: max elementwise abs difference (inf if shapes differ).
    - raise vs raise (same type): 0. Any other raise/array mismatch: inf.
    """
    o_str, m_str = isinstance(observed, str), isinstance(modeled, str)
    if o_str or m_str:
        return 0.0 if (o_str and m_str and observed == modeled) else float("inf")
    o = np.asarray(observed, dtype=np.float64)
    m = np.asarray(modeled, dtype=np.float64)
    if o.shape != m.shape:
        return float("inf")
    with np.errstate(invalid="ignore"):
        diff = np.abs(o - m)
    # treat matching +/-inf and matching NaN as zero residual
    same = (o == m) | (np.isnan(o) & np.isnan(m))
    diff = np.where(same, 0.0, diff)
    result = float(np.max(diff)) if diff.size else 0.0
    # A leftover NaN means one side is NaN where the other is finite (mismatched
    # NaN-ness): the modeled config cannot reproduce the value, so it is an
    # incompatibility (infinite residual), not a small numeric difference. Keeping
    # it NaN would also poison best_match's min-search (`t < NaN` is always False).
    return float("inf") if np.isnan(result) else result


def signature_residual(observed: Signature, modeled: Signature) -> list[float]:
    """Per-probe residual vector between two signatures of equal length."""
    return [probe_residual(o, m) for o, m in zip(observed, modeled, strict=True)]


def total_residual(residual: list[float]) -> float:
    """Sum of finite residuals; inf if any probe is incompatible (raise mismatch)."""
    if any(np.isinf(r) for r in residual):
        return float("inf")
    return float(sum(residual))


def best_match(
    observed: Signature, battery: list[Probe]
) -> tuple[GenericConfig | None, list[float]]:
    """The enumerated config whose battery signature has the smallest total residual
    to ``observed``. Returns (config, per-probe residual).

    ``config`` is ``None`` only if the config space is empty, which never happens
    (``enumerate_configs`` always yields 192), so in practice a config is always
    returned. On an *empty battery* every config has zero residual, so the first
    enumerated config is returned and any observation is vacuously "explained" --
    callers that care should pass a non-empty battery.
    """
    best_cfg: GenericConfig | None = None
    best_res: list[float] = []
    best_total = float("inf")
    for cfg in enumerate_configs():
        res = signature_residual(observed, signature(cfg, battery))
        t = total_residual(res)
        if best_cfg is None or t < best_total:
            best_cfg, best_res, best_total = cfg, res, t
    return best_cfg, best_res


_MUTANT_BIAS = 7.0


class _MutantExecutor(PLExecutor):
    """PL executor with an off-model additive bias on selected DAG op types."""

    def __init__(self, signal: Signal, bias_types: tuple[type, ...]) -> None:
        super().__init__(signal)
        self._bias_types = bias_types

    def _evaluate_op(
        self, op: object, inputs: tuple[NDArray[np.floating], ...]
    ) -> NDArray[np.floating]:
        base = super()._evaluate_op(op, inputs)
        if isinstance(op, self._bias_types):
            return base + _MUTANT_BIAS
        return base


class MutantBackend(EvaluationBackend):
    """A deliberately off-model evaluator: native PL semantics plus an additive bias
    on selected DAG ops (default: inequality predicates, ``Ineq``).

    No Core 6 configuration can reproduce it; it exists only to demonstrate the
    open-world residual check fires and localizes correctly. ``bias_types`` chooses
    which ops carry the bias -- e.g. ``(Ineq,)`` perturbs the predicate, ``(Add,)``
    perturbs an arithmetic ``+`` inside a predicate, which the localizer then pins.
    """

    name = "mutant"

    def __init__(self, bias_types: tuple[type, ...] = (Ineq,)) -> None:
        self._bias_types = bias_types

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> PLResult:
        if not isinstance(signal, Signal):
            raise TypeError(f"{self.name} backend requires a Signal")
        dag, trace = PLDagBuilder().build(formula)
        return _MutantExecutor(signal, self._bias_types).execute(dag, trace)

    def observed_signature(self, battery: list[Probe]) -> Signature:
        out: list[ProbeValue] = []
        for probe in battery:
            phi = parse(probe.formula_text)
            try:
                res = self.evaluate(phi, probe.signal)
                arr = np.asarray(res.robustness, dtype=np.float64)
                out.append(arr[0] if arr.ndim == 2 else arr)
            except Exception as exc:  # noqa: BLE001
                out.append(f"RAISES:{type(exc).__name__}")
        return tuple(out)


class OpenWorldReport(TypedDict):
    """Result of an open-world check: the closest config, whether it fully
    explains the observation, and the residual it leaves."""

    best_config: GenericConfig | None
    explained: bool
    total_residual: float
    per_probe_residual: list[float]
    offending_probes: list[str]


def detect_open_world(observed: Signature, battery: list[Probe]) -> OpenWorldReport:
    """Best-match a tool and report whether its behavior is fully explained."""
    config, residual = best_match(observed, battery)
    total = total_residual(residual)
    return {
        "best_config": config,
        "explained": total == 0.0,
        "total_residual": total,
        "per_probe_residual": residual,
        "offending_probes": [
            battery[i].name for i, r in enumerate(residual) if r > 0.0 or np.isinf(r)
        ],
    }


def localize_residual(mutant: MutantBackend, battery: list[Probe]) -> list[str]:
    """For each probe the mutant diverges on, the kind of the minimal divergent AST
    node vs the best-matching config (a compact analogue of the E2 localizer).

    Compares per-node traces of the mutant against ``GenericBackend(best_config)``;
    a node is minimal-divergent when its trace differs but all descendants agree.
    """
    observed = mutant.observed_signature(battery)
    best_cfg, _ = best_match(observed, battery)
    assert best_cfg is not None
    kinds: list[str] = []
    for probe in battery:
        phi = parse(probe.formula_text)
        try:
            mut = mutant.evaluate(phi, probe.signal)
            ref = GenericBackend(best_cfg).evaluate(phi, probe.signal)
        except Exception:  # noqa: BLE001 -- skip probes either side cannot evaluate
            continue
        kinds.extend(_minimal_divergent_kinds(phi, mut, ref))
    return kinds


def _subnodes(node: Node) -> tuple[Node, ...]:
    """Operand sub-nodes to descend into. A predicate keeps its two arithmetic
    operands in ``attrs["left"]/["right"]`` (its ``children`` are empty); every other
    node's operands are its ``children``. Returning the predicate operands lets the
    localizer descend into arithmetic and pin a divergent ``+``/``*``/``var`` rather
    than stopping at the predicate.
    """
    if node.kind == "predicate":
        left, right = node.attrs["left"], node.attrs["right"]
        assert isinstance(left, Node) and isinstance(right, Node)
        return (left, right)
    return node.children


def _minimal_divergent_kinds(phi: Node, a: PLResult, b: object) -> list[str]:
    """Kinds of nodes whose own trace differs while all descendants agree."""
    out: list[str] = []

    def agree(node: Node) -> bool:
        children_ok = all(agree(c) for c in _subnodes(node))
        try:
            ta = np.asarray(a.trace_for(node), dtype=np.float64)
            tb = np.asarray(b.trace_for(node), dtype=np.float64)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 -- trace unavailable for this node on one side
            # Cannot compare this node, so cannot claim it diverges (treating a
            # missing trace as divergence would emit spurious localizations). Defer
            # to the children's verdict instead.
            return children_ok
        node_ok = ta.shape == tb.shape and bool(np.allclose(ta, tb, atol=1e-6, equal_nan=True))
        if not node_ok and children_ok:
            out.append(node.kind)
        return node_ok and children_ok

    agree(phi)
    return out


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    battery = load_battery()
    logger.info("E6 exp 5 -- open-world residual")
    logger.info("in-model tools (each should be explained, residual 0):")
    for tool in TOOLS:
        report = detect_open_world(observed_signature(tool, battery), battery)
        logger.info(
            "  %-8s explained=%-5s residual=%.3g",
            tool,
            report["explained"],
            report["total_residual"],
        )
    mutant = MutantBackend()
    report = detect_open_world(mutant.observed_signature(battery), battery)
    logger.info("planted mutant (off-model predicate bias):")
    logger.info(
        "  explained=%s residual=%.3g offending=%d/%d probes",
        report["explained"],
        report["total_residual"],
        len(report["offending_probes"]),
        len(battery),
    )
    kinds = localize_residual(mutant, battery)
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    logger.info("  localized minimal-divergent nodes: %s", counts or "(none)")


if __name__ == "__main__":
    main()
