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
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tidystl import parse  # noqa: E402
from tidystl.core.nodes import Node  # noqa: E402
from tidystl.diagnostics import localize_results  # noqa: E402

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


def primitive_op(node: Node, backend: str, *, step: float) -> str:
    """The backend's genuine primitive operation at ``node`` (labels the comparison).

    Mirrors the per-node op vocabulary documented in the backend modules:
    ``_pl_dag.py`` (native, breach), ``rtamt.py``, ``pymtl.py``, ``taliro.py``,
    and ``stlcgpp.py``. This names the operator at the node only; it does not
    annotate executor- or root-output conventions (e.g. Breach's terminal-step
    extension or its sliding-window kernel) -- those are not properties of the
    op, and reporting them here would assert an implicit-choice attribution the
    tool cannot mechanically justify. When two backends share the same op at a
    divergent node, ``console_listing`` flags the divergence as a convention
    difference instead (see backend docs).
    """
    if backend in ("native", "breach"):
        return _pl_op(node)
    if backend == "rtamt":
        return _rtamt_op(node, step=step)
    if backend == "pymtl":
        return _pymtl_op(node)
    if backend == "taliro":
        return _taliro_op(node)
    if backend == "stlcgpp":
        return _stlcgpp_op(node, step=step)
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


def _pymtl_op(node: Node) -> str:
    """The pymtl (py-metric-temporal-logic) primitive at ``node``.

    pymtl uses zero-order-hold interpolation and right-half-open temporal
    windows ``[t + a, t + b)`` on a ``dt`` pivot grid (see
    ``tidystl_compat/pymtl.py``); the half-open bracket names the excluded
    right endpoint that drives the divergence from native's closed window.
    """
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
            name = {
                "always": "ZOHWindowMin",
                "eventually": "ZOHWindowMax",
                "until": "ZOHBoundedUntil",
            }[node.kind]
            return f"{name}[{start:g},{end:g})"
        case _:
            raise ValueError(f"unknown STL node kind {node.kind!r}")


def _taliro_op(node: Node) -> str:
    """The TaLiRo (dp_taliro) primitive at ``node``.

    TaLiRo reduces temporal operators over the sample points that fall inside
    the CLOSED real-time window ``[t + a, t + b]`` only -- no interpolation and
    no past-end extension; a window containing no samples yields the reduction
    identity (``+inf`` for always, ``-inf`` for eventually/until). Predicates
    are Euclidean signed distances normalized by ``||A||`` (see
    ``tidystl_compat/taliro.py``).
    """
    match node.kind:
        case "predicate":
            op = node.attrs["op"]
            return "Eq (unsupported)" if op == "==" else f"Ineq({op})/||A||"
        case "not":
            return "Negate"
        case "and":
            return "PointwiseMin"
        case "or":
            return "PointwiseMax"
        case "always" | "eventually" | "until":
            start, end = _interval(node)
            name = {
                "always": "SampleMin",
                "eventually": "SampleMax",
                "until": "SampleBoundedUntil",
            }[node.kind]
            return f"{name}[{start:g},{end:g}]"
        case _:
            raise ValueError(f"unknown STL node kind {node.kind!r}")


def _stlcgpp_op(node: Node, *, step: float) -> str:
    """The STLCG++ primitive at ``node``.

    STLCG++ evaluates temporal operators with discrete index windows on a
    uniform grid, via inclusive "last"-anchored recurrences (see
    ``tidystl_compat/stlcgpp.py``: ``DiscreteWindowMinLast`` etc.). Index
    bounds mirror ``_rtamt_op``: ``round(a / step), round(b / step)``.
    """
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
                "always": "DiscreteWindowMinLast",
                "eventually": "DiscreteWindowMaxLast",
                "until": "DiscreteBoundedUntilInclusiveLast",
            }[node.kind]
            return f"{name}[{lo},{hi}]"
        case _:
            raise ValueError(f"unknown STL node kind {node.kind!r}")


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


def _temporal_depth(node: Node) -> int:
    """Max nesting of temporal operators (always/eventually/until) on any
    root-to-leaf path; 0 for a purely Boolean/predicate formula."""
    here = 1 if node.kind in ("always", "eventually", "until") else 0
    return here + max((_temporal_depth(child) for child in node.children), default=0)


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
    ast_nodes, _child_indices = _ast_walk(phi)
    if len(ast_nodes) != len(nodes_a):
        raise SystemExit(
            f"re-parsed AST size {len(ast_nodes)} != recorded {len(nodes_a)} for {cond}"
        )

    if record_a.result["times"] != record_b.result["times"]:
        raise SystemExit(f"time-grid mismatch for {cond} between {backend_a} and {backend_b}")
    times = record_a.result["times"]
    step = float(times[1] - times[0]) if len(times) > 1 else 1.0

    # The recorded per-node traces align with the re-parsed AST by post-order
    # index; expose them to the shared core localizer keyed by node identity.
    trace_a_by_id = {id(node): nodes_a[i]["trace"] for i, node in enumerate(ast_nodes)}
    trace_b_by_id = {id(node): nodes_b[i]["trace"] for i, node in enumerate(ast_nodes)}

    minimal: list[dict[str, Any]] = []
    for dv in localize_results(
        phi,
        lambda node: trace_a_by_id[id(node)],
        lambda node: trace_b_by_id[id(node)],
        atol=atol,
    ):
        minimal.append(
            {
                "index": dv.index,
                "subformula": nodes_a[dv.index]["subformula"],
                "kind": dv.node.kind,
                "op_a": primitive_op(dv.node, backend_a, step=step),
                "op_b": primitive_op(dv.node, backend_b, step=step),
                "first_divergent_timestep": dv.first_divergent_index,
                "is_root": dv.is_root,
            }
        )
    return minimal


def console_listing(
    spec_name: str,
    signal_name: str,
    backend_a: str,
    backend_b: str,
    n_nodes: int,
    depth: int,
    root_a: float,
    root_b: float,
    minimal: list[dict[str, Any]],
) -> list[str]:
    lines = [
        f"$ localize --spec {spec_name} --signal {signal_name} --backends {backend_a},{backend_b}",
        f"  {n_nodes} nodes, temporal depth {depth}  |  "
        f"root robustness: {backend_a}={root_a:+.4g}, {backend_b}={root_b:+.4g}",
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
        if node["op_a"] == node["op_b"]:
            # Same genuine op on both sides: the divergence is not an op
            # mismatch but a convention difference. We do not name the implicit
            # choice here -- the tool reports node + timestep mechanically and
            # leaves attribution to the controlled single-choice case design.
            # The only mechanical fact we can add is whether a root-output
            # post-process is even possible: it can occur only at the formula
            # root, so a non-root same-op divergence is necessarily an executor
            # convention, while a root one may be either (see backend docs).
            scope = "root-output or executor" if node["is_root"] else "executor"
            lines.append(f"    same op; divergence is a {scope} convention -- see backend docs")
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


# --- paper summary table (tabular only; numbers computed from the records) ---

# Which tool-vs-tool comparisons appear as one row each in the paper table. Only
# this selection is curated; every number in the row is computed from records.
_FEATURED = [
    ("div_until_boundary", "div_until_boundary", "breach", "rtamt"),
    ("div_window_sampling", "div_window_sampling", "breach", "taliro"),
]
# Per-tool verdict breakdown for one spec (highlights the outlier tool).
_VERDICT_BREAKDOWN = ("div_until_boundary", "div_until_boundary", ["breach", "rtamt", "taliro", "stlcgpp"])
_TOOL_TEX = {
    "breach": "Breach", "rtamt": "RTAMT", "taliro": "TaLiRo", "stlcgpp": "STLCG++",
    "native": "native", "pymtl": "pymtl",
}


def _root_robustness(record: RunRecord) -> float:
    return float(record.result["nodes"][-1]["trace"][0])


def _load_div_signals() -> dict[str, Any]:
    """Signal registry (records keep only ``times``, not per-variable values)."""
    path = Path(__file__).resolve().parents[1] / "registry" / "signals_div.json"
    try:
        return json.loads(path.read_text())
    except OSError:
        return {}


def _fmt_values(values: list[float]) -> str:
    """Compact value list; collapse a constant trace to ``<v> (const)``."""
    if len(set(values)) == 1:
        return f"{values[0]:g} (const)"
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


def _case_tex(spec_name: str) -> str:
    return "\\texttt{" + spec_name.removeprefix("div_").replace("_", "\\_") + "}"


def _origin_tex(subformula: str) -> str:
    """``G[1,3]`` -> ``$G_{[1,3]}$``; other labels -> ``\\texttt{...}``."""
    m = re.fullmatch(r"([GFU])\[([^\]]*)\]", subformula)
    if m:
        return f"$ {m.group(1)}_{{[{m.group(2)}]}} $".replace(" ", "")
    return "\\texttt{" + subformula.replace("_", "\\_") + "}"


def _effect_tex(root_a: float, root_b: float, atol: float) -> str:
    if (root_a > 0) != (root_b > 0):
        return "verdict flip"
    if abs(root_a - root_b) > atol:
        return "value gap"
    return "trace-only divergence"


def _num_tex(value: float) -> str:
    return f"${value:+.4g}$"


def _signal_comments(feat: list[dict[str, Any]]) -> list[str]:
    """One ``%`` line per featured case describing its signal (times + per-var
    values, constants collapsed). Values come from the registry, not the records."""
    signals = _load_div_signals()
    lines: list[str] = []
    for f in feat:
        sg = signals.get(f["signal"])
        if not sg:
            continue
        parts = [f"t={_fmt_values(sg['times'])}"]
        parts += [f"{name}={_fmt_values(vals)}" for name, vals in sg["values"].items()]
        lines.append(f"%   {f['spec'].removeprefix('div_')}: " + "; ".join(parts))
    return lines


def emit_summary_tex(latest: dict[tuple[str, str, str], RunRecord], *, atol: float) -> str:
    """Render the tool-vs-tool summary as a single ``tabular`` (one row per
    featured comparison): tools, formula size/depth, both roots, the effect, and
    the localized origin together with the kernel each tool runs there. Secondary
    facts (the 4-tool verdict breakdown, first-divergence steps, breach's
    sliding-min note) are emitted as comments for the caption, not as extra rows.
    """
    feat = []
    for spec, signal, a, b in _FEATURED:
        ra, rb = latest[(spec, signal, a)], latest[(spec, signal, b)]
        feat.append({
            "spec": spec, "signal": signal, "a": a, "b": b,
            "spec_text": ra.result["spec"],
            "n": len(ra.result["nodes"]),
            "depth": _temporal_depth(parse(ra.result["spec"])),
            "root_a": _root_robustness(ra), "root_b": _root_robustness(rb),
            "minimal": localize(ra, rb, atol=atol),
        })

    out = [
        "% AUTO-GENERATED from extra/outputs/e2_localization records by",
        "% extra/experiments/e2_localization/aggregate.py --emit-tex; do not hand-edit.",
        "% tabular only -- wrap in your own table float / caption.",
        "% Preamble: \\usepackage{booktabs}.",
        "% 'l/r' columns follow the 'tools compared' order (left tool / right tool).",
        "%",
        "% formulas (tidystl syntax):",
        *[f"%   {f['spec'].removeprefix('div_')}: {f['spec_text']}" for f in feat],
        "%",
        "% signals:",
        *_signal_comments(feat),
        "\\begin{tabular}{@{}llccl@{}}",
        "\\toprule",
        "Case & tools compared & $|\\phi|$/depth & $\\rho$ (l/r) & divergence \\\\",
        "\\midrule",
    ]
    for f in feat:
        origins = ", ".join(_origin_tex(m["subformula"]) for m in f["minimal"]) or "--"
        out.append(
            f"{_case_tex(f['spec'])} & {_TOOL_TEX[f['a']]} vs.\\ {_TOOL_TEX[f['b']]} & "
            f"{f['n']} / {f['depth']} & {_num_tex(f['root_a'])} / {_num_tex(f['root_b'])} & "
            f"{_effect_tex(f['root_a'], f['root_b'], atol)} on {origins} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", ""]

    # Kernel-level detail (the named primitive each tool runs at the origin),
    # kept as a comment so the table body stays short.
    for f in feat:
        kdet = "; ".join(
            f"{_origin_tex(m['subformula'])} {_TOOL_TEX[f['a']]} {m['op_a']} vs.\\ "
            f"{_TOOL_TEX[f['b']]} {m['op_b']}"
            for m in f["minimal"]
        )
        out.append(f"% kernels at origin -- {f['spec'].removeprefix('div_')}: {kdet}")

    # Secondary facts as comments (for the caption), so the body stays one table.
    spec, signal, tools = _VERDICT_BREAKDOWN
    roots = {t: _root_robustness(latest[(spec, signal, t)]) for t in tools}
    counts = Counter(round(v, 6) for v in roots.values())
    verdict = ", ".join(
        f"{_TOOL_TEX[t]} {roots[t]:+.4g}" + (" (outlier)" if counts[round(roots[t], 6)] == 1 else "")
        for t in tools
    )
    out.append(f"% {spec} verdict at t=0 across tools: {verdict}.")
    steps = "; ".join(
        f"{f['spec']} {', '.join('t-index ' + str(m['first_divergent_timestep']) for m in f['minimal'])}"
        for f in feat
    )
    out.append(f"% first-divergence step: {steps}.")
    return "\n".join(out) + "\n"


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
    parser.add_argument(
        "--emit-tex", type=Path, default=None, metavar="PATH",
        help="also write the paper summary table (tabular only) to PATH",
    )
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
        nodes_a = record_a.result["nodes"]
        n_nodes = len(nodes_a)
        depth = _temporal_depth(parse(record_a.result["spec"]))
        root_a = float(nodes_a[-1]["trace"][0])
        root_b = float(record_b.result["nodes"][-1]["trace"][0])
        for line in console_listing(
            spec_name, signal_name, backend_a, backend_b, n_nodes, depth, root_a, root_b, minimal
        ):
            print(line)
        print()

    if args.emit_tex is not None:
        args.emit_tex.write_text(emit_summary_tex(latest, atol=args.atol))
        print(f"wrote {args.emit_tex}")


if __name__ == "__main__":
    main()
