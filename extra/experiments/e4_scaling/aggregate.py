"""E4 aggregate: merge the latest record per column into ``scaling_merged.json``
and regenerate ``tab:scaling`` (``--emit-tex PATH``), deriving ``per_trace_ms``
/ ``throughput_msts`` from the recorded means. Cell conventions (extrapolation
dagger, OOM) are in the table footnotes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT = "e4_scaling"

#: Canonical column order of the merged results: every backend/tool the batch
#: collects, in display order. ``merge`` gathers ALL collected columns (any not
#: listed here are appended); which of them reach the tex table is a separate,
#: configurable selection (see ``COLUMN_LABELS`` / ``--tex-columns``).
BACKEND_ORDER = [
    "native",
    "breach",
    "rtamt",
    "rtamt_dense",
    "pymtl",
    "stlcgpp",
    "taliro",
    "tidystl_simd",
    "rtamt-real",
    "pymtl-real",
    "stlcgpp-real",
]

#: (N, T) rows shown in the paper table (layout free to change per the
#: 2026-06-05 decision; full grid lives in the merged JSON).
TEX_ROWS = [(1, 101), (1, 1001), (64, 501), (64, 5001), (256, 501), (256, 5001)]

#: Tex column key -> header label, for every column that CAN be rendered.
#: Data is merged for all collected backends; this maps the ones a table may
#: display to their short headers. ``-c.`` marks a tidystl compatibility
#: backend; bare names are the real external tools.
COLUMN_LABELS = {
    "native": "Native",
    "breach": "Breach-c.",
    "rtamt": "RTAMT-c.",
    "rtamt_dense": "RTAMT-dense-c.",
    "pymtl": "py-MTL-c.",
    "stlcgpp": "STLCG++-c.",
    "stlcgpp_torch": "STLCG++-torch",
    "taliro": "TaLiRo-c.",
    "tidystl_simd": "SIMD",
    "breach-real": "Breach",
    "rtamt-real": "RTAMT",
    "pymtl-real": "py-MTL",
    "stlcgpp-real": "STLCG++",
}

#: Columns rendered into the paper tex table by default, in table order.
#: Override on the command line with ``--tex-columns native breach rtamt ...``.
DEFAULT_TEX_COLUMNS = [
    "breach",
    "taliro",
    "rtamt",
    "rtamt-real",
    "pymtl",
    "pymtl-real",
    "tidystl_simd",
]

#: Real tools timed single-trace (batch is a loop); larger-N cells extrapolate
#: linearly from the measured per-trace cost and are marked with a dagger.
SINGLE_TRACE_REAL = ("rtamt-real", "pymtl-real", "breach-real")

FAIRNESS_NOTES = {
    "construction": "real-tool spec construction and input "
    "marshalling hoisted out of the timed region",
    "rtamt_batching": "real RTAMT is single-trace; batches are a "
    "Python loop; measured at N in {1,8} (per-trace cost flat in "
    "N); larger batches extrapolate linearly and are marked",
    "stlcgpp_batching": "real STLCG++ batches via torch.vmap; cells "
    "whose unfold-based window materialization exceeds a 2 GiB raw "
    "budget are OOM on the 31 GiB host and reported as such",
    "tidystl_backends": "native/breach/rtamt are tidystl backends "
    "with per-node tracing on; tidystl_simd is the no-trace Rust "
    "executor (ceiling)",
}


def gather_facts(result_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """``(run_dir, result)`` per run under ``result_dir``, ascending; runs
    without a readable ``result.json`` are skipped.
    """
    out: list[tuple[Path, dict[str, Any]]] = []
    run_dirs = sorted(
        (p for p in result_dir.glob("run_*") if p.is_dir() and p.name.removeprefix("run_").isdigit()),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue
        out.append((run_dir, result))
    return out


def select_columns(
    result_dir: Path,
) -> tuple[dict[str, tuple[str, dict[str, Any]]], dict[str, tuple[str, dict[str, Any]]]]:
    """Latest backend record per backend, latest tool record per tool.

    Returns ``(backend_results, tool_results)`` each mapping name ->
    ``(source, result)``. ``source`` is the provenance string used on
    the merged rows.
    """
    backend_results: dict[str, tuple[str, dict[str, Any]]] = {}
    tool_results: dict[str, tuple[str, dict[str, Any]]] = {}
    for run_dir, result in gather_facts(result_dir):
        if "backend" in result and "tool" in result:
            raise SystemExit(
                f"ambiguous record: contains both 'backend' and 'tool' keys (run dir: {run_dir})"
            )
        if result.get("partial"):
            print(
                f"warning: {run_dir.name} is a partial (interrupted) record",
                file=sys.stderr,
            )
        if "backend" in result and "results" in result:  # run_backend.py
            source = f"{run_dir.name} (backend={result['backend']})"
            backend_results[result["backend"]] = (source, result)  # ascending: last wins
        elif "tool" in result and "results" in result:  # run_{rtamt,stlcgpp,breach}.py
            source = f"{run_dir.name} (tool={result['tool']})"
            tool_results[result["tool"]] = (source, result)
    return backend_results, tool_results


def merge(result_dir: Path) -> dict[str, Any]:
    backend_results, tool_results = select_columns(result_dir)

    by_backend: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    spec: str | None = None
    tool_versions: dict[str, str] = {}

    # tidystl backend records: each carries one backend's rows.
    for backend, (source, result) in backend_results.items():
        by_backend[backend] = (source, result["results"])
        if spec is None and "spec" in result:
            spec = result["spec"]

    # real-tool records: the per-cell rows already carry their merged-row
    # backend names (rtamt-real / stlcgpp-real); group by row backend. The
    # tool's provenance (version/torch/spec) lives under ``info``.
    for tool, (source, result) in tool_results.items():
        if "info" not in result:
            raise KeyError(f"tool record {source} lacks the 'info' block")
        info = result["info"]
        rows = result["results"]
        for row_backend in {r["backend"] for r in rows}:
            by_backend[row_backend] = (source, [r for r in rows if r["backend"] == row_backend])
        if tool == "rtamt-real":
            tool_versions["rtamt"] = info["version"]
        if tool == "stlcgpp-real":
            tool_versions["stlcgpp"] = info["version"]
            tool_versions["torch"] = info["torch"]

    missing = [b for b in BACKEND_ORDER if b not in by_backend]
    if missing:
        print(f"warning: no rows for {missing}; cells will print '--'", file=sys.stderr)

    merged_rows: list[dict[str, Any]] = []
    order = BACKEND_ORDER + sorted(set(by_backend) - set(BACKEND_ORDER))
    for backend in order:
        if backend not in by_backend:
            continue
        source, rows = by_backend[backend]
        for row in rows:
            merged = {**row, "status": row.get("status", "ok"), "source": source}
            # Derived metrics are computed HERE, uniformly, from the
            # recorded mean: runners record raw facts only. Stored copies
            # on older records are overridden.
            if merged["status"] == "ok":
                n, t = merged["batch_size"], merged["timesteps"]
                mean_ms = merged["mean_ms"]
                merged["per_trace_ms"] = round(mean_ms / n, 4)
                merged["throughput_msts"] = round((n * t) / (mean_ms / 1e3) / 1e6, 3)
            merged_rows.append(merged)

    return {
        "experiment": "E4 merged scaling (tab:scaling source)",
        "spec": spec,
        "fairness_notes": FAIRNESS_NOTES,
        "tool_versions": tool_versions,
        "results": merged_rows,
    }


def _index(merged: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {(r["backend"], r["batch_size"], r["timesteps"]): r for r in merged["results"]}


def _tex_cell(
    index: dict[tuple[str, int, int], dict[str, Any]],
    backend: str,
    n: int,
    t: int,
) -> str:
    row = index.get((backend, n, t))
    if row is None and backend in SINGLE_TRACE_REAL:
        # Per-trace cost is N-independent; extrapolate from an OK base, preferring
        # the larger measured batch (N=8 over N=1) but falling back if it was
        # capped (timeout) / missing.
        base = index.get((backend, 8, t))
        if base is None or base["status"] != "ok":
            base = index.get((backend, 1, t))
        if base is not None and base["status"] == "ok":
            return f"${n * base['per_trace_ms']:.0f}^\\dagger$"
        return "--"
    if row is None:
        return "--"
    if row["status"] != "ok":
        if row["status"] in ("skipped", "oom-killed"):
            return "OOM"
        if row["status"] == "timeout":
            return "t/o"
        return "--"
    mean, std = row["mean_ms"], row["std_ms"]
    if mean >= 100:
        return f"${mean:.0f} \\pm {std:.0f}$"
    return f"${mean:.2f} \\pm {std:.2f}$"


def emit_tex(merged: dict[str, Any], tex_columns: list[str]) -> str:
    """Render ``tab:scaling`` for the selected columns, in the given order.

    ``tex_columns`` is the display subset (keys into ``COLUMN_LABELS``); the
    merged data still holds every collected backend. A selected column with no
    merged rows simply prints ``--`` in every cell.
    """
    unknown = [c for c in tex_columns if c not in COLUMN_LABELS]
    if unknown:
        available = ", ".join(COLUMN_LABELS)
        raise SystemExit(f"unknown tex column(s): {', '.join(unknown)}; available: {available}")

    columns = [(c, COLUMN_LABELS[c]) for c in tex_columns]
    index = _index(merged)
    # Bare tabular fragment only: no table/figure float, no caption, no comment
    # header -- the paper wraps it in its own table environment and supplies the
    # caption/legend. Cell conventions for that legend: dagger = N x per-trace
    # extrapolation for single-trace real tools; OOM = window materialization
    # over the host memory budget; t/o = py-MTL hit the per-cell wall-clock cap.
    lines = [
        "\\begin{tabular}{rr " + " ".join(["r"] * len(columns)) + "}",
        "\\hline",
        "$N$ & $T$ & " + " & ".join(label for _, label in columns) + " \\\\",
        "\\hline",
    ]
    for n, t in TEX_ROWS:
        cells = [_tex_cell(index, backend, n, t) for backend, _ in columns]
        lines.append(f"{n:>4} & {t:>5} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}%", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--emit-tex", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--tex-columns",
        nargs="+",
        default=list(DEFAULT_TEX_COLUMNS),
        metavar="COLUMN",
        help=(
            "columns to render into the tex table, in table order "
            f"(default: {' '.join(DEFAULT_TEX_COLUMNS)}); choices: {', '.join(COLUMN_LABELS)}. "
            "Data is merged for all collected backends regardless"
        ),
    )
    args = parser.parse_args()

    merged = merge(args.result_dir)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.result_dir / "scaling_merged.json"
    out_path.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"wrote {out_path} ({len(merged['results'])} rows)")

    body = emit_tex(merged, list(args.tex_columns))
    if args.emit_tex is not None:
        args.emit_tex.write_text(body)
        print(f"wrote {args.emit_tex}")
    else:
        print("\n" + body)


if __name__ == "__main__":
    main()
