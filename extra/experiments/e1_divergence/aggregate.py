"""E1 aggregate -> ``matrix.json`` + ``reproduction.json``; emits
``tab:divergence-matrix``. The Breach column is read from
``cache/breach_column/`` rather than the normal column dir.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "cache"
REPO_ROOT = Path(__file__).resolve().parents[3]


# =========================================================================
# Configuration (inlined; this experiment is fully described in-module)
# =========================================================================
# No runners: ``batch.sh`` drives the shared eval tools
# (``tools/eval_tidystl.py``, ``tools/eval_external_tools.py``) to produce
# one column file per backend/tool under ``columns_dir``; the Breach column
# is imported from ``cache/breach_column/`` (the MATLAB-machine run) by
# this module, which owns all comparison over the diagonal (spec name ==
# signal name). ``breach_handoff.py`` reads ``SPECS``/``SIGNALS``/
# ``ANNOTATIONS`` from here so the pair set has a single source of truth.

#: Spec/signal registries by block. The order of these lists fixes the
#: name (row) order: baseline first, then divergence.
SPECS = (
    "extra/experiments/registry/specs_baseline.json",
    "extra/experiments/registry/specs_div.json",
)
SIGNALS = (
    "extra/experiments/registry/signals_baseline.json",
    "extra/experiments/registry/signals_div.json",
)
COLUMNS_DIR = "extra/outputs/e1_divergence/columns"

#: Agreement tolerance: exact equality (covering equal infinities) or
#: ``|diff| <= ATOL`` on mutually defined timesteps counts as agreement.
ATOL = 1e-6

#: tidystl backends emitted by ``eval_tidystl.py`` (one column file each).
BACKENDS = ("native", "breach", "rtamt", "stlcgpp")
#: The real tools whose columns populate the matrix.
REAL_TOOLS = ("breach", "rtamt", "stlcgpp")
#: V(b): the tidystl backend expected to reproduce each real tool.
EXPECTED_BACKEND = {"breach": "breach", "rtamt": "rtamt", "stlcgpp": "stlcgpp"}

#: Human-readable provenance recorded in the product files' metadata.
NORMALIZATION = (
    "per-timestep robustness over the pair's original times; missing tool "
    "timesteps NaN-padded by time-value alignment; agreement on mutually "
    "defined timesteps; equal infinities agree"
)
BREACH_STATUS = (
    "live matlab-run column (Breach 1.11.4, MATLAB R2022b, dockerized, "
    "2026-06-05) under cache/breach_column/; recorded test CSVs as fallback "
    "where spec+data match"
)

#: name -> (decision point, table) for divergent cells. Migrated from the
#: pairs module. Baseline rows whose real tools diverge on documented
#: corner cases are annotated honestly rather than hidden. Verified
#: 2026-06-05: STLCG++'s until uses an inclusive prefix minimum where
#: RTAMT's is exclusive -> the "open vs. closed interval endpoints" row of
#: tab:divergence; the ``*_boundary`` pairs exercise windows past the trace
#: end -> the "boundary rule" row of tab:siginterp. The top-level binary
#: and/or rows: recorded real Breach extends the penultimate value over the
#: final timestep (the mechanism ``div_terminal_and`` isolates). The
#: divergence block lists one targeted pair per decision point.
ANNOTATIONS: dict[str, tuple[str, str]] = {
    "until_basic": ("open vs. closed interval endpoints", "tab:divergence"),
    "until_tight": ("open vs. closed interval endpoints", "tab:divergence"),
    "until_never_sat": ("open vs. closed interval endpoints", "tab:divergence"),
    "until_boundary": ("boundary rule", "tab:siginterp"),
    "always_boundary": ("boundary rule", "tab:siginterp"),
    "eventually_boundary": ("boundary rule", "tab:siginterp"),
    # F[1,3] on 7 samples: at t=6 the window [7,9] is entirely past the
    # trace end; RTAMT -inf (pessimistic) vs STLCG++ 7.0 (padding=last).
    "eventually_offset": ("boundary rule", "tab:siginterp"),
    "and_two": ("terminal-step rule", "tab:siginterp"),
    "or_two": ("terminal-step rule", "tab:siginterp"),
    "combined_and_or": ("terminal-step rule", "tab:siginterp"),
    "div_boundary_F24": ("boundary rule", "tab:siginterp"),
    "div_terminal_and": ("terminal-step rule", "tab:siginterp"),
    "div_interp_sparse": ("interpolation", "tab:siginterp"),
    "div_nonuniform_always": ("signal model", "tab:siginterp"),
    "div_equality": ("equality predicates", "tab:divergence"),
}

#: Fallback recorded CSVs under ``tests/breach_ground_truth/`` (used only
#: when ``cache/breach_column/<name>.csv`` is absent); the curated usable
#: subset, migrated from the pairs module. These baseline CSVs match the
#: pair's trace data and a robustness-equivalent spec (verified against
#: generate_ground_truth.m, 2026-06-05). NOT usable despite a shared name:
#: always_boundary, arith_sum, arith_difference (different specs/traces);
#: absent from the Breach set: eventually_boundary, until_boundary.
BREACH_RECORDED_CSV = {
    "predicate_gte": "predicate_gte.csv",
    "not_simple": "not_simple.csv",
    "and_two": "and_two.csv",
    "or_two": "or_two.csv",
    "combined_and_or": "combined_and_or.csv",
    "always_sliding": "always_sliding.csv",
    "eventually_sliding": "eventually_sliding.csv",
    "eventually_offset": "eventually_offset.csv",
    "until_basic": "until_basic.csv",
    "until_tight": "until_tight.csv",
    "until_never_sat": "until_never_sat.csv",
    # div block: reuses the recorded Breach case interp_sparse (identical data).
    "div_interp_sparse": "interp_sparse.csv",
}

#: Registry file stem -> matrix block label. The order of ``SPECS``/
#: ``SIGNALS`` fixes the name (row) order, preserving the old pair order.
_BLOCK_FROM_STEM = {"baseline": "baseline", "div": "divergence"}


# --- name metadata (from the registries) ---------------------------------


def block_for(path: Path) -> str:
    """Matrix block label for a registry file (baseline | divergence)."""
    stem = path.stem.split("_")[-1]
    if stem not in _BLOCK_FROM_STEM:
        raise SystemExit(f"cannot infer block from registry file {path} (stem {stem!r})")
    return _BLOCK_FROM_STEM[stem]


def load_names(spec_paths: list[Path], signal_paths: list[Path]) -> list[dict[str, Any]]:
    """Ordered name metadata: registry order, baseline then div.

    Each entry carries ``name``, ``block``, ``spec`` (the spec registry
    entry), and ``times`` (the signal's time grid).
    """
    if len(spec_paths) != len(signal_paths):
        raise SystemExit("config specs/signals must list the same number of files")
    names: list[dict[str, Any]] = []
    for spec_path, signal_path in zip(spec_paths, signal_paths, strict=True):
        block = block_for(spec_path)
        specs = json.loads(spec_path.read_text())
        signals = json.loads(signal_path.read_text())
        for name, spec in specs.items():
            signal = signals[name]
            names.append(
                {
                    "name": name,
                    "block": block,
                    "spec": spec,
                    "times": tuple(float(t) for t in signal["times"]),
                }
            )
    return names


# --- column-file loading -------------------------------------------------


def _diagonal(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select diagonal entries (spec == signal) keyed by name.

    Off-diagonal entries in the column file are harmless extra data.
    """
    return {
        r["spec"]: {k: v for k, v in r.items() if k not in ("spec", "signal")}
        for r in results
        if r["spec"] == r["signal"]
    }


def _read_column(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing column file: {path} (run batch.sh first)")
    return json.loads(path.read_text())


def load_backend_columns(
    columns_dir: Path, backends: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per backend, diagonal column merged over the baseline + div blocks."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for backend in backends:
        merged: dict[str, dict[str, Any]] = {}
        for blk in ("baseline", "div"):
            payload = _read_column(columns_dir / f"backend_{backend}_{blk}.json")
            merged.update(_diagonal(payload["results"]))
        out[backend] = merged
    return out


def load_tool_columns(
    columns_dir: Path, tools: list[str]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Per venv tool, diagonal column + the file's version metadata.

    Returns ``(columns, versions)``. ``versions[tool]`` is the file's
    top-level keys minus ``results`` (e.g. ``tool``/``version``/
    ``python``/...); ``eval_external_tools`` records only those.
    """
    columns: dict[str, dict[str, dict[str, Any]]] = {}
    versions: dict[str, dict[str, Any]] = {}
    for tool in tools:
        merged: dict[str, dict[str, Any]] = {}
        meta: dict[str, Any] = {}
        for blk in ("baseline", "div"):
            payload = _read_column(columns_dir / f"tool_{tool}_{blk}.json")
            merged.update(_diagonal(payload["results"]))
            meta = {k: v for k, v in payload.items() if k != "results"}
        columns[tool] = merged
        versions[tool] = meta
    return columns, versions


# --- Breach column (imported facts) --------------------------------------


def _csv_rows(path: Path) -> list[list[float]]:
    """Parse a (time, robustness) CSV (skip header) into raw rows."""
    lines = path.read_text().strip().splitlines()[1:]
    return [[float(c) for c in line.split(",")] for line in lines]


def breach_case(name: str, spec: dict[str, Any], recorded_csv: dict[str, str]) -> dict[str, Any]:
    """Real-Breach facts for one name, by provenance precedence.

    Precedence: ``cache/breach_column/<name>.csv`` (the live MATLAB run)
    -> the recorded regression CSV under ``tests/breach_ground_truth/``
    (``BREACH_RECORDED_CSV``) -> ``unsupported`` if the spec
    entry has no ``breach`` key (a-priori inexpressible) -> ``pending``
    (awaiting the MATLAB run). RAW (time, robustness) rows are recorded;
    alignment to the signal grid happens below.
    """
    live = CACHE_DIR / "breach_column" / f"{name}.csv"
    if live.exists():
        return {"status": "ok", "rows": _csv_rows(live), "source": f"matlab-run:{live.name}"}
    recorded = recorded_csv.get(name)
    if recorded is not None:
        path = REPO_ROOT / "tests" / "breach_ground_truth" / recorded
        if path.exists():
            return {"status": "ok", "rows": _csv_rows(path), "source": f"recorded:{recorded}"}
        return {"status": "error", "reason": f"recorded CSV missing: {path.name}"}
    if "breach" not in spec:
        return {
            "status": "unsupported",
            "reason": "not expressible in Breach STL (a-priori; the handoff "
            "run includes a verification attempt)",
        }
    return {"status": "pending", "reason": "awaiting the MATLAB-machine run (HANDOFF.md)"}


# --- normalization + comparison ------------------------------------------


def align_to_times(
    times: tuple[float, ...],
    tool_times: list[float],
    tool_values: list[float],
    *,
    atol: float,
) -> list[float]:
    """Align a tool's (time, robustness) rows to the signal's time grid.

    Timesteps the tool does not emit become NaN ("no value", excluded
    from comparisons). A tool timestep matching no signal timestep within
    ``atol`` is an error: that would silently misalign the matrix.
    """
    out = [float("nan")] * len(times)
    for t, rho in zip(tool_times, tool_values, strict=True):
        deltas = [abs(t - ref) for ref in times]
        idx = int(np.argmin(deltas))
        if deltas[idx] > atol:
            raise ValueError(f"tool timestep t={t} matches no signal timestep (atol={atol})")
        out[idx] = float(rho)
    return out


def compare_columns(a: list[float], b: list[float], *, atol: float) -> dict[str, Any]:
    """Agreement between two normalized columns on mutually defined timesteps.

    Exact equality (covering equal infinities) or |diff| <= atol counts
    as agreement per timestep.
    """
    arr_a, arr_b = np.asarray(a), np.asarray(b)
    both = ~(np.isnan(arr_a) | np.isnan(arr_b))
    if not both.any():
        return {"status": "no-overlap", "n_compared": 0}
    a_d, b_d = arr_a[both], arr_b[both]
    exact = a_d == b_d
    with np.errstate(invalid="ignore"):
        close = exact | (np.abs(a_d - b_d) <= atol)
    agree = bool(np.all(close))
    inexact_diff = np.abs(a_d[~exact] - b_d[~exact])
    return {
        "status": "agree" if agree else "diverge",
        "n_compared": int(both.sum()),
        "max_abs_diff": float(inexact_diff.max()) if inexact_diff.size else 0.0,
        "first_divergent_timestep": (
            None if agree else int(np.flatnonzero(both)[np.flatnonzero(~close)[0]])
        ),
    }


def real_tool_column(
    times: tuple[float, ...], case: dict[str, Any], *, atol: float
) -> dict[str, Any]:
    """Normalize one real-tool case (raw facts) to the signal's time grid.

    RTAMT and Breach record raw (time, robustness) ``rows`` (possibly
    truncated); STLCG++ records one value per input timestep, already
    aligned.
    """
    column: dict[str, Any] = {"status": case["status"]}
    if case["status"] != "ok":
        column["reason"] = case.get("reason", "")
        return column
    if "rows" in case:  # RTAMT / Breach: (time, robustness) rows
        tool_times = [r[0] for r in case["rows"]]
        tool_values = [r[1] for r in case["rows"]]
        column["robustness"] = align_to_times(times, tool_times, tool_values, atol=atol)
        column["raw_rows"] = case["rows"]
    else:  # STLCG++: one value per input timestep, already aligned
        column["robustness"] = [float(v) for v in case["robustness"]]
    return column


# --- annotation + matrix/reproduction cells ------------------------------


def annotation_for(name: str, annotations: dict[str, Any]) -> dict[str, str] | None:
    """Decision-point annotation for ``name`` from ``ANNOTATIONS``."""
    entry = annotations.get(name)
    if entry is None:
        return None
    point, table = entry
    return {"decision_point": point, "table": table}


def matrix_cell(
    name: str,
    col_a: dict[str, Any],
    col_b: dict[str, Any],
    annotations: dict[str, Any],
    *,
    atol: float,
) -> dict[str, Any]:
    """Pairwise agreement cell between two real-tool columns.

    Special statuses take precedence in the order error > pending >
    unsupported('--'); when both columns are special, the first in that
    order wins (legacy-inherited; do not reorder).
    """
    for special in ("error", "pending", "unsupported"):
        if special in (col_a["status"], col_b["status"]):
            status = "--" if special == "unsupported" else special
            return {"status": status}
    cell = compare_columns(col_a["robustness"], col_b["robustness"], atol=atol)
    if cell["status"] == "diverge":
        annotation = annotation_for(name, annotations)
        if annotation is not None:
            cell["annotation"] = annotation
        else:
            cell["annotation"] = {"decision_point": "UNANNOTATED", "table": "??"}
    return cell


def reproduction_entry(
    tool: str,
    tool_col: dict[str, Any],
    backend_cols: dict[str, dict[str, Any]],
    *,
    expected_backend: dict[str, str],
    atol: float,
) -> dict[str, Any]:
    """V(b): which tidystl backends reproduce this real-tool column."""
    expected = expected_backend[tool]
    expected_col = backend_cols[expected]
    entry: dict[str, Any] = {"expected_backend": expected}
    if tool_col["status"] != "ok":
        if tool_col["status"] in ("pending", "error"):
            entry["status"] = tool_col["status"]
        else:  # tool rejects the pair: does the backend reject it too?
            entry["status"] = (
                "coverage-match" if expected_col["status"] != "ok" else "coverage-mismatch"
            )
            entry["backend_status"] = expected_col["status"]
        return entry
    if expected_col["status"] != "ok":
        entry["status"] = "coverage-mismatch"
        entry["backend_status"] = expected_col["status"]
        entry["backend_reason"] = expected_col.get("reason", "")
        return entry
    matches = [
        backend
        for backend, col in backend_cols.items()
        if col["status"] == "ok"
        and compare_columns(col["robustness"], tool_col["robustness"], atol=atol)["status"]
        == "agree"
    ]
    comparison = compare_columns(expected_col["robustness"], tool_col["robustness"], atol=atol)
    entry["status"] = "reproduced" if expected in matches else "NOT-REPRODUCED"
    entry["matching_backends"] = matches
    entry["n_compared"] = comparison["n_compared"]
    entry["max_abs_diff"] = comparison.get("max_abs_diff")
    return entry


# --- paper table (tab:divergence-matrix) ----------------------------------


TOOLS = ("breach", "rtamt", "stlcgpp")

MARK = {"common": "$=$", "differs": "$\\ast$", "inexpressible": "--", "special": "$\\dagger$"}

# Cases where exactly two tools express the spec and they diverge: no
# majority exists, so the deviating tool must be named explicitly. The
# mark is ``dagger`` (documented special handling), and the table caption
# in the paper explains the specific behavior.
SPECIAL_HANDLING = {
    # Observed 2026-06-05 (case notes): real RTAMT 0.3.5 silently treats
    # a non-uniform grid as uniformly indexed samples instead of raising.
    "div_nonuniform_always": "rtamt",
}

DECISION_POINT_SHORT = {
    "boundary rule": "boundary rule",
    "terminal-step rule": "terminal-step rule",
    "open vs. closed interval endpoints": "open vs.\\ closed endpoints",
    "interpolation": "window alignment",  # renamed in the paper (rv2026)
    "signal model": "signal model",
    "equality predicates": "equality predicates",
}

TEX_BLOCKS = (
    ("baseline", "Baseline block (operator coverage)"),
    ("divergence", "Divergence block (one case per decision point)"),
)


def spec_tex(spec: str) -> str:
    """Render a tidystl surface spec as LaTeX math (paper notation)."""
    s = spec
    s = re.sub(r"G\[([^\]]+)\]", r"\\Box_{[\1]}", s)
    s = re.sub(r"F\[([^\]]+)\]", r"\\Diamond_{[\1]}", s)
    s = re.sub(r"\s+U\[([^\]]+)\]\s+", r" \\mathbin{U_{[\1]}} ", s)
    s = s.replace(" and ", " \\wedge ").replace(" or ", " \\vee ")
    s = s.replace("not ", "\\neg ")
    s = s.replace(">=", "\\ge").replace("<=", "\\le").replace("==", "=")
    return f"${s}$"


def expressible(spec: dict[str, Any]) -> dict[str, bool]:
    return {t: t in spec for t in TOOLS}


def tool_marks(name: str, spec: dict[str, Any], pairs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Derive per-tool marks from pairwise statuses and expressibility."""
    expr = expressible(spec)
    # Consistency: a cell is "--" iff at least one member is inexpressible.
    for key, cell in pairs.items():
        a, b = key.split("|")
        if (cell["status"] == "--") != (not expr[a] or not expr[b]):
            raise SystemExit(f"{name}: pair {key} status contradicts expressibility")

    marks = {t: MARK["inexpressible"] for t in TOOLS if not expr[t]}
    alive = [t for t in TOOLS if expr[t]]
    diverging = [k for k, p in pairs.items() if p["status"] == "diverge"]
    if len(alive) <= 1 or not diverging:
        # No comparison possible, or full agreement among expressers.
        marks.update(dict.fromkeys(alive, MARK["common"]))
        return marks
    if len(alive) == 2:
        special = SPECIAL_HANDLING.get(name)
        if special not in alive:
            raise SystemExit(
                f"{name}: two expressers diverge with no majority; "
                f"add a SPECIAL_HANDLING entry naming the deviating tool"
            )
        marks.update({t: MARK["special"] if t == special else MARK["common"] for t in alive})
        return marks
    # All three express: the deviating tool is the one in every diverging pair.
    deviating = set.intersection(*(set(k.split("|")) for k in diverging))
    if len(diverging) != 2 or len(deviating) != 1:
        raise SystemExit(f"{name}: diverging pairs {diverging} have no single deviator")
    dev = deviating.pop()
    marks.update({t: MARK["differs"] if t == dev else MARK["common"] for t in alive})
    return marks


def row_decision_point(
    name: str, pairs: dict[str, dict[str, Any]], annotations: dict[str, Any]
) -> str:
    points = {
        (p.get("annotation") or {}).get("decision_point")
        for p in pairs.values()
        if p.get("status") == "diverge"
    } - {None}
    if len(points) > 1:
        raise SystemExit(f"{name}: conflicting annotations {points}")
    if points:
        return DECISION_POINT_SHORT[points.pop()]
    if all(p.get("status") == "--" for p in pairs.values()):
        # Pure coverage row: the decision point comes from the config
        # annotations.
        point = annotations[name][0]
        return DECISION_POINT_SHORT[point] + " (coverage)"
    return ""


def emit_tex(
    names: list[dict[str, Any]],
    matrix_cases: list[dict[str, Any]],
    annotations: dict[str, Any],
) -> str:
    """Render the ``tab:divergence-matrix`` tabular body from the matrix."""
    meta_by_name = {m["name"]: m for m in names}
    lines = [
        "% AUTO-GENERATED from extra/outputs/e1_divergence/matrix.json",
        "% by extra/experiments/e1_divergence/aggregate.py (tidystl repo); do not hand-edit.",
        "% B = Breach 1.11.4, R = RTAMT 0.3.5, S = STLCG++ 0.0.2 (hard min/max).",
        "% Cells: = matches the common behavior (atol 1e-6, mutually defined",
        "% timesteps), * differs from the other two, -- cannot express the case",
        "% (coverage finding), dagger documented special handling.",
        "\\begin{tabular}{l c ccc l}",
        "\\hline",
        "Case & $|\\sigma|$ & B & R & S & Decision point \\\\",
        "\\hline",
    ]
    for blk, label in TEX_BLOCKS:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for entry in matrix_cases:
            if entry["block"] != blk:
                continue
            meta = meta_by_name[entry["name"]]
            spec = meta["spec"]
            marks = tool_marks(entry["name"], spec, entry["pairs"])
            cells = [marks[t] for t in TOOLS]
            lines.append(
                f"\\quad {spec_tex(spec['tidystl'])} & {len(meta['times'])} & "
                + " & ".join(cells)
                + f" & {row_decision_point(entry['name'], entry['pairs'], annotations)} \\\\"
            )
        lines.append("\\hline")
    lines.append("\\end{tabular}%")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--emit-tex", type=Path, default=None, metavar="PATH")
    args = parser.parse_args()

    atol: float = ATOL
    backends: list[str] = list(BACKENDS)
    real_tools: list[str] = list(REAL_TOOLS)
    expected_backend: dict[str, str] = EXPECTED_BACKEND
    annotations: dict[str, Any] = ANNOTATIONS
    recorded_csv: dict[str, str] = BREACH_RECORDED_CSV

    spec_paths = [REPO_ROOT / p for p in SPECS]
    signal_paths = [REPO_ROOT / p for p in SIGNALS]
    names = load_names(spec_paths, signal_paths)

    columns_dir = REPO_ROOT / COLUMNS_DIR
    backend_columns = load_backend_columns(columns_dir, backends)
    venv_columns, tool_meta = load_tool_columns(
        columns_dir, [t for t in real_tools if t != "breach"]
    )

    matrix_cases: list[dict[str, Any]] = []
    repro_cases: list[dict[str, Any]] = []
    for meta in names:
        name = meta["name"]
        times = meta["times"]
        backend_cols = {backend: backend_columns[backend][name] for backend in backends}
        real_cols: dict[str, dict[str, Any]] = {}
        for tool in real_tools:
            if tool == "breach":
                case = breach_case(name, meta["spec"], recorded_csv)
            else:
                case = venv_columns[tool][name]
            real_cols[tool] = real_tool_column(times, case, atol=atol)
        matrix_cases.append(
            {
                "name": name,
                "block": meta["block"],
                "pairs": {
                    f"{a}|{b}": matrix_cell(
                        name, real_cols[a], real_cols[b], annotations, atol=atol
                    )
                    for i, a in enumerate(real_tools)
                    for b in real_tools[i + 1 :]
                },
            }
        )
        repro_cases.append(
            {
                "name": name,
                "block": meta["block"],
                "tools": {
                    tool: reproduction_entry(
                        tool,
                        real_cols[tool],
                        backend_cols,
                        expected_backend=expected_backend,
                        atol=atol,
                    )
                    for tool in real_tools
                },
            }
        )

    tool_versions: dict[str, Any] = {}
    for tool in real_tools:
        if tool == "breach":
            tool_versions[tool] = {"source": "csv-import"}
        else:
            tool_versions[tool] = tool_meta[tool]
    metadata = {
        "atol": atol,
        "normalization": NORMALIZATION,
        "tool_versions": tool_versions,
        "breach_status": BREACH_STATUS,
    }

    out_dir = args.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix.json").write_text(
        json.dumps(
            {"experiment": "E1 pairwise matrix", "meta": metadata, "cases": matrix_cases}, indent=2
        )
        + "\n"
    )
    (out_dir / "reproduction.json").write_text(
        json.dumps(
            {"experiment": "E1 V(b) reproduction", "meta": metadata, "cases": repro_cases}, indent=2
        )
        + "\n"
    )

    print(f"wrote {out_dir / 'matrix.json'}")
    print(f"wrote {out_dir / 'reproduction.json'}")
    print("\n=== matrix ===")
    for m in matrix_cases:
        cells = "  ".join(f"{pair}:{cell['status']}" for pair, cell in m["pairs"].items())
        print(f"  {m['name']:<24} {cells}")
    print("\n=== reproduction (V(b)) ===")
    for r in repro_cases:
        cells = "  ".join(f"{tool}:{e['status']}" for tool, e in r["tools"].items())
        print(f"  {r['name']:<24} {cells}")

    body = emit_tex(names, matrix_cases, annotations)
    if args.emit_tex is None:
        print("\n" + body, end="")
    else:
        args.emit_tex.write_text(body)
        print(f"wrote {args.emit_tex}")


if __name__ == "__main__":
    main()
