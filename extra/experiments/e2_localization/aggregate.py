"""E2 aggregate: minimal divergent nodes between two backends.

No re-evaluation -- the two records' traces are realigned by post-order node
index (stable, since the spec re-parses deterministically), and a node is the
divergence point when its traces disagree (within ``atol``) but every
descendant agrees. With no ``--spec``/``--signal``/``--backends`` it compares
every recorded pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tidystl import parse  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402

EXPERIMENT = "e2_localization"


@dataclass
class RunRecord:
    """One run's saved facts: ``params.json`` (config) plus ``result.json``."""

    run_dir: Path
    config: dict[str, Any]
    result: dict[str, Any]


def gather_facts(result_dir: Path) -> list[RunRecord]:
    """Complete run records under ``result_dir``, ascending by run number
    (input file ``params.json``, or legacy ``config.json``; partial runs skipped).
    """
    records: list[RunRecord] = []
    run_dirs = sorted(
        (p for p in result_dir.glob("run_*") if p.is_dir() and p.name.removeprefix("run_").isdigit()),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        params_path = run_dir / "params.json"
        input_path = params_path if params_path.exists() else run_dir / "config.json"
        if not (result_path.exists() and input_path.exists()):
            continue
        try:
            records.append(
                RunRecord(
                    run_dir=run_dir,
                    config=json.loads(input_path.read_text()),
                    result=json.loads(result_path.read_text()),
                )
            )
        except json.JSONDecodeError:
            continue
    return records


def _interval(node: Node) -> tuple[float, float]:
    interval = node.attrs.get("interval")
    assert isinstance(interval, tuple) and len(interval) == 2
    start, end = interval
    assert isinstance(start, float) and isinstance(end, float)
    return start, end


def primitive_op(node: Node, backend: str, *, step: float, is_root: bool) -> str:
    """The backend's primitive operation at ``node`` (labels the comparison).

    Mirrors the op vocabulary documented in the backend modules:
    ``_pl_dag.py`` (native, breach), ``rtamt.py``; the breach labels also
    name the backend's documented executor deviations (``breach.py``).
    """
    if backend in ("native", "breach"):
        label = _pl_op(node)
        if backend == "breach":
            if node.kind == "always" and _interval(node)[0] > 0:
                label += " overridden by _sliding_min_breach"
            if is_root and node.kind in ("and", "or"):
                label += " + terminal-step extension"
        return label
    if backend == "rtamt":
        return _rtamt_op(node, step=step)
    raise ValueError(f"no op vocabulary for backend {backend!r}")


def _pl_op(node: Node) -> str:
    match node.kind:
        case "predicate":
            op = node.attrs["op"]
            return "Eq" if op == "==" else f"Ineq({op})"
        case "not":
            return "Negate"
        case "and":
            return "PointwiseMin"
        case "or":
            return "PointwiseMax"
        case "always":
            start, end = _interval(node)
            return f"WindowMin[{start:g},{end:g}]"
        case "eventually":
            start, end = _interval(node)
            return f"WindowMax[{start:g},{end:g}]"
        case "until":
            start, end = _interval(node)
            return f"BoundedUntilKernel[{start:g},{end:g}]"
        case _:
            raise ValueError(f"unknown STL node kind {node.kind!r}")


def _rtamt_op(node: Node, *, step: float) -> str:
    match node.kind:
        case "predicate":
            op = node.attrs["op"]
            return "Eq" if op == "==" else f"Ineq({op})"
        case "not":
            return "Negate"
        case "and":
            return "PointwiseMin"
        case "or":
            return "PointwiseMax"
        case "always" | "eventually" | "until":
            start, end = _interval(node)
            lo, hi = round(start / step), round(end / step)
            name = {
                "always": "DiscreteWindowMin",
                "eventually": "DiscreteWindowMax",
                "until": "DiscreteBoundedUntil",
            }[node.kind]
            return f"{name}[{lo},{hi}]"
        case _:
            raise ValueError(f"unknown STL node kind {node.kind!r}")


def _columns_agree(a: Any, b: Any, *, atol: float) -> bool:
    arr_a, arr_b = np.asarray(a), np.asarray(b)
    exact = arr_a == arr_b  # covers equal infinities
    with np.errstate(invalid="ignore"):
        close = exact | (np.abs(arr_a - arr_b) <= atol)
    return bool(np.all(close))


def _first_divergent_index(a: Any, b: Any, *, atol: float) -> int:
    arr_a, arr_b = np.asarray(a), np.asarray(b)
    exact = arr_a == arr_b
    with np.errstate(invalid="ignore"):
        close = exact | (np.abs(arr_a - arr_b) <= atol)
    return int(np.flatnonzero(~close)[0])


def _ast_walk(phi: Node) -> tuple[list[Node], list[list[int]]]:
    """Post-order ``Node`` list plus each node's child indices (index-aligned)."""
    ast_nodes: list[Node] = []
    child_indices: list[list[int]] = []

    def walk(node: Node) -> int:
        children = [walk(child) for child in node.children]
        idx = len(ast_nodes)
        ast_nodes.append(node)
        child_indices.append(children)
        return idx

    walk(phi)
    return ast_nodes, child_indices


def localize(record_a: RunRecord, record_b: RunRecord, *, atol: float) -> list[dict[str, Any]]:
    """All minimal divergent nodes between two backends on one (spec, signal).

    A node is minimal-divergent when its recorded traces disagree while
    every descendant's traces agree. Returns post-order (lowest first);
    empty list means the backends agree everywhere on this input.
    """
    backend_a = record_a.result["backend"]
    backend_b = record_b.result["backend"]
    nodes_a = record_a.result["nodes"]
    nodes_b = record_b.result["nodes"]
    cond = f"(spec {record_a.result['spec_name']!r}, signal {record_a.result['signal_name']!r})"

    if len(nodes_a) != len(nodes_b):
        raise SystemExit(
            f"node-count mismatch for {cond}: "
            f"{backend_a} has {len(nodes_a)}, {backend_b} has {len(nodes_b)}"
        )
    for idx, (na, nb) in enumerate(zip(nodes_a, nodes_b, strict=True)):
        if na["kind"] != nb["kind"] or na["subformula"] != nb["subformula"]:
            raise SystemExit(
                f"AST mismatch at index {idx} for {cond}: "
                f"{backend_a} {na['kind']}/{na['subformula']!r} vs "
                f"{backend_b} {nb['kind']}/{nb['subformula']!r}"
            )

    spec = record_a.result["spec"]
    phi = parse(spec)
    ast_nodes, child_indices = _ast_walk(phi)
    if len(ast_nodes) != len(nodes_a):
        raise SystemExit(
            f"re-parsed AST size {len(ast_nodes)} != recorded {len(nodes_a)} for {cond}"
        )

    if record_a.result["times"] != record_b.result["times"]:
        raise SystemExit(f"time-grid mismatch for {cond} between {backend_a} and {backend_b}")
    times = record_a.result["times"]
    step = float(times[1] - times[0]) if len(times) > 1 else 1.0
    n = len(ast_nodes)
    last = n - 1

    agree: list[bool] = [False] * n  # whole-subtree agreement, post-order
    minimal: list[dict[str, Any]] = []
    for idx in range(n):
        trace_a = nodes_a[idx]["trace"]
        trace_b = nodes_b[idx]["trace"]
        node_ok = _columns_agree(trace_a, trace_b, atol=atol)
        children_ok = all(agree[c] for c in child_indices[idx])
        agree[idx] = node_ok and children_ok
        if not node_ok and children_ok:
            node = ast_nodes[idx]
            is_root = idx == last
            minimal.append(
                {
                    "index": idx,
                    "subformula": nodes_a[idx]["subformula"],
                    "kind": node.kind,
                    "op_a": primitive_op(node, backend_a, step=step, is_root=is_root),
                    "op_b": primitive_op(node, backend_b, step=step, is_root=is_root),
                    "first_divergent_timestep": _first_divergent_index(trace_a, trace_b, atol=atol),
                    "is_root": is_root,
                }
            )
    return minimal


def console_listing(
    spec_name: str,
    signal_name: str,
    backend_a: str,
    backend_b: str,
    minimal: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"$ localize --spec {spec_name} --signal {signal_name} --backends {backend_a},{backend_b}"
    ]
    if not minimal:
        lines.append("  backends agree on every traced node")
        return lines
    for node in minimal:
        lines.append(
            f"  first divergent op at `{node['subformula']}`: "
            f"{node['op_a']} vs {node['op_b']} "
            f"(t-index {node['first_divergent_timestep']})"
        )
    return lines


def _latest_by_cond_backend(
    records: list[RunRecord],
) -> dict[tuple[str, str, str], RunRecord]:
    latest: dict[tuple[str, str, str], RunRecord] = {}
    for record in records:  # gather_facts is ascending: last wins
        key = (
            record.result["spec_name"],
            record.result["signal_name"],
            record.result["backend"],
        )
        latest[key] = record
    return latest


def _comparisons(
    latest: dict[tuple[str, str, str], RunRecord],
    spec_filter: str | None,
    signal_filter: str | None,
    backends_filter: tuple[str, str] | None,
) -> list[tuple[str, str, str, str]]:
    """Deterministic (spec_name, signal_name, backend_a, backend_b) list."""
    conds = sorted({(spec, signal) for spec, signal, _ in latest})
    if spec_filter is not None:
        conds = [(s, sig) for s, sig in conds if s == spec_filter]
    if signal_filter is not None:
        conds = [(s, sig) for s, sig in conds if sig == signal_filter]
    out: list[tuple[str, str, str, str]] = []
    for spec, signal in conds:
        present = sorted(b for s, sig, b in latest if s == spec and sig == signal)
        if backends_filter is not None:
            a, b = backends_filter
            missing = [x for x in (a, b) if x not in present]
            if missing:
                raise SystemExit(
                    f"(spec {spec!r}, signal {signal!r}): "
                    f"no record for backend(s): {', '.join(missing)}"
                )
            out.append((spec, signal, a, b))
        else:
            for a, b in combinations(present, 2):
                out.append((spec, signal, a, b))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", default=None, help="restrict to one spec name")
    parser.add_argument("--signal", default=None, help="restrict to one signal name")
    parser.add_argument(
        "--backends", default=None, help="comma-separated backend pair, e.g. native,breach"
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    backends_filter: tuple[str, str] | None = None
    if args.backends is not None:
        parts = [b.strip() for b in args.backends.split(",")]
        if len(parts) != 2:
            raise SystemExit("--backends takes exactly two comma-separated names")
        backends_filter = (parts[0], parts[1])

    records = gather_facts(args.result_dir)
    if not records:
        raise SystemExit(f"no run records under {args.result_dir}")
    latest = _latest_by_cond_backend(records)

    comparisons = _comparisons(latest, args.spec, args.signal, backends_filter)
    if not comparisons:
        raise SystemExit("no comparisons to run (check --spec / --signal / --backends)")

    for spec_name, signal_name, backend_a, backend_b in comparisons:
        record_a = latest[(spec_name, signal_name, backend_a)]
        record_b = latest[(spec_name, signal_name, backend_b)]
        minimal = localize(record_a, record_b, atol=args.atol)
        for line in console_listing(spec_name, signal_name, backend_a, backend_b, minimal):
            print(line)
        print()


if __name__ == "__main__":
    main()
