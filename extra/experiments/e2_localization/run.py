"""E2 runner: record one backend's per-node robustness traces on one
(spec, signal). Comparison lives in ``aggregate.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import tidystl_compat  # noqa: F401,E402  -- registers the compat backends as a side effect

from extra.experiments._lib.infra import Infra  # noqa: E402
from tidystl import Signal, evaluate, parse  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402

EXPERIMENT = "e2_localization"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunParams:
    """The run condition; ``batch.sh`` sweeps ``backend`` over fixed (spec, signal) pairs."""

    backend: str = "native"
    spec: str = "div_terminal_and"
    signal: str = "div_terminal_and"
    specs_json: str = "extra/experiments/registry/specs_div.json"
    signals_json: str = "extra/experiments/registry/signals_div.json"


@dataclass
class NodeTrace:
    """One AST node's recorded robustness time series (a fact)."""

    index: int
    kind: str
    subformula: str
    trace: list[float]


@dataclass
class RunInput:
    """The condition: one backend on one (spec, signal) from the registries."""

    backend: str
    spec_name: str
    spec: str
    signal_name: str
    signal: dict[str, Any]


@dataclass
class RunResult:
    """The facts: per-node robustness traces of the parsed spec."""

    backend: str
    spec_name: str
    signal_name: str
    spec: str
    times: list[float]
    nodes: list[NodeTrace]


def _fmt_arith(node: Node) -> str:
    match node.kind:
        case "var":
            return str(node.attrs["name"])
        case "const":
            return f"{node.attrs['value']:g}"
        case "abs" | "sqrt":
            return f"{node.kind}({_fmt_arith(node.children[0])})"
        case _:
            left, right = node.children
            return f"{_fmt_arith(left)} {node.kind} {_fmt_arith(right)}"


def _interval(node: Node) -> tuple[float, float]:
    interval = node.attrs.get("interval")
    assert isinstance(interval, tuple) and len(interval) == 2
    start, end = interval
    assert isinstance(start, float) and isinstance(end, float)
    return start, end


def subformula_label(node: Node) -> str:
    """Compact human-readable label for an AST node (a recorded fact)."""
    match node.kind:
        case "predicate":
            left, right = node.attrs["left"], node.attrs["right"]
            assert isinstance(left, Node) and isinstance(right, Node)
            return f"{_fmt_arith(left)} {node.attrs['op']} {_fmt_arith(right)}"
        case "always" | "eventually" | "until":
            start, end = _interval(node)
            symbol = {"always": "G", "eventually": "F", "until": "U"}[node.kind]
            return f"{symbol}[{start:g},{end:g}]"
        case _:
            return node.kind


def _load_entry(path: Path, name: str) -> dict[str, Any]:
    """One named entry from a registry JSON; SystemExit lists available names."""
    registry: dict[str, Any] = json.loads(path.read_text())
    if name not in registry:
        raise SystemExit(f"unknown name {name!r} in {path} (available: {', '.join(registry)})")
    return registry[name]


def runner(input: RunInput) -> RunResult:
    signal_entry = input.signal
    phi = parse(input.spec)
    times = np.asarray(signal_entry["times"], dtype=np.float64)
    values = {
        name: np.asarray(vals, dtype=np.float64) for name, vals in signal_entry["values"].items()
    }
    signal = Signal.from_dict(times, values)

    res = evaluate(phi, signal, backend=input.backend)

    nodes: list[NodeTrace] = []

    def walk(node: Node) -> None:
        for child in node.children:
            walk(child)
        # len(nodes) is evaluated before append, so this is the 0-based post-order index.
        nodes.append(
            NodeTrace(
                index=len(nodes),
                kind=node.kind,
                subformula=subformula_label(node),
                trace=[float(v) for v in np.asarray(res.trace_for(node))[0]],
            )
        )

    walk(phi)

    return RunResult(
        backend=input.backend,
        spec_name=input.spec_name,
        signal_name=input.signal_name,
        spec=input.spec,
        times=list(signal_entry["times"]),
        nodes=nodes,
    )


def main() -> None:
    params = tyro.cli(RunParams)

    specs_json = REPO_ROOT / params.specs_json
    signals_json = REPO_ROOT / params.signals_json
    spec_entry = _load_entry(specs_json, params.spec)
    spec_text = spec_entry.get("tidystl")
    if spec_text is None:
        raise SystemExit(
            f"spec {params.spec!r} in {specs_json} has no 'tidystl' key "
            f"(available keys: {', '.join(spec_entry)})"
        )
    signal_entry = _load_entry(signals_json, params.signal)
    inp = RunInput(
        backend=params.backend,
        spec_name=params.spec,
        spec=spec_text,
        signal_name=params.signal,
        signal=signal_entry,
    )

    env = Infra.capture_env(EXPERIMENT)
    with Infra.run_with_timer(env) as timer:
        result = runner(inp)

    Infra.record_success(
        env=env,
        params=asdict(params),
        result=asdict(result),
        timer=timer,
    )

    logger.info(
        "spec=%s signal=%s backend=%s nodes=%d",
        result.spec_name,
        result.signal_name,
        result.backend,
        len(result.nodes),
    )


if __name__ == "__main__":
    main()
