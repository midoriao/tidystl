"""E6 aggregate: pool e6_fuzz run records into agreement matrices, per-operator
divergence, and an abstention table; write the product JSONs + a tex matrix;
print a markdown summary. Consumes records; never reruns the sweep.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

OPERATORS = ("not", "and", "or", "always", "eventually", "until")


def values_match(a: float, b: float, atol: float, rtol: float) -> bool:
    """Numeric agreement: equal infinities agree; NaN never agrees."""
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def gather(result_dir: Path) -> tuple[list[str], list[dict], float, float]:
    """Pool ``pairs`` across every run record under ``result_dir``.

    Returns ``(backends, pairs, atol, rtol)``. All records must share the same
    backend list and tolerances (they come from the same code); the first
    record fixes them and later records are validated against it.
    """
    run_dirs = sorted(
        (p for p in result_dir.glob("run_*") if p.is_dir() and p.name.removeprefix("run_").isdigit()),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    backends: list[str] | None = None
    atol = rtol = None
    pairs: list[dict] = []
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            continue
        if backends is None:
            backends, atol, rtol = data["backends"], data["atol"], data["rtol"]
        elif data["backends"] != backends:
            raise SystemExit(f"{run_dir}: backend list differs from earlier records")
        elif (data["atol"], data["rtol"]) != (atol, rtol):
            raise SystemExit(f"{run_dir}: tolerances differ from earlier records")
        pairs.extend(data["pairs"])
    if backends is None:
        raise SystemExit(f"no usable run records under {result_dir} (run batch.sh first)")
    return backends, pairs, float(atol), float(rtol)


def _both_ok(pair: dict, bi: str, bj: str) -> tuple[dict, dict] | None:
    ci, cj = pair["cells"][bi], pair["cells"][bj]
    if ci["status"] == "ok" and cj["status"] == "ok":
        return ci, cj
    return None


def _symmetric(backends: list[str], rate: dict[tuple[str, str], float | None]) -> dict[str, dict[str, float | None]]:
    """Expand pair-keyed rates into a full symmetric matrix with 1.0 diagonal."""
    out: dict[str, dict[str, float | None]] = {b: {} for b in backends}
    for bi in backends:
        out[bi][bi] = 1.0
    for (bi, bj), value in rate.items():
        out[bi][bj] = value
        out[bj][bi] = value
    return out


def scalar_matrix(pairs: list[dict], backends: list[str], atol: float, rtol: float) -> dict[str, dict[str, float | None]]:
    """Pairwise t=0 agreement rate; ``None`` when no pair is comparable."""
    rate: dict[tuple[str, str], float | None] = {}
    for bi, bj in itertools.combinations(backends, 2):
        comparable = matched = 0
        for pair in pairs:
            cells = _both_ok(pair, bi, bj)
            if cells is None:
                continue
            comparable += 1
            if values_match(cells[0]["scalar"], cells[1]["scalar"], atol, rtol):
                matched += 1
        rate[(bi, bj)] = (matched / comparable) if comparable else None
    return _symmetric(backends, rate)


def _signal_matches(si: list[float], sj: list[float], atol: float, rtol: float) -> bool:
    if len(si) != len(sj):
        return False
    return all(values_match(a, b, atol, rtol) for a, b in zip(si, sj, strict=False))


def full_signal_matrix(pairs: list[dict], backends: list[str], atol: float, rtol: float) -> dict[str, dict[str, float | None]]:
    """Pairwise whole-trace agreement on same-length pairs; ``None`` if none."""
    rate: dict[tuple[str, str], float | None] = {}
    for bi, bj in itertools.combinations(backends, 2):
        comparable = matched = 0
        for pair in pairs:
            cells = _both_ok(pair, bi, bj)
            if cells is None:
                continue
            si, sj = cells[0]["signal"], cells[1]["signal"]
            if len(si) != len(sj):
                continue  # not comparable: excluded from the denominator
            comparable += 1
            if _signal_matches(si, sj, atol, rtol):
                matched += 1
        rate[(bi, bj)] = (matched / comparable) if comparable else None
    return _symmetric(backends, rate)


def per_operator(pairs: list[dict], backends: list[str], atol: float, rtol: float) -> dict[str, dict[str, Any]]:
    """Per operator: how often backends that both evaluate the formula disagree.

    For each operator, restrict to formulas containing it, then average the
    scalar-disagreement rate over all comparable backend pairs and formulas.
    ``mean_disagreement`` is ``None`` when the operator never appears (or no
    pair is comparable).
    """
    out: dict[str, dict[str, Any]] = {}
    for op in OPERATORS:
        subset = [p for p in pairs if op in p["operators"]]
        comparable = disagree = 0
        for pair in subset:
            for bi, bj in itertools.combinations(backends, 2):
                cells = _both_ok(pair, bi, bj)
                if cells is None:
                    continue
                comparable += 1
                if not values_match(cells[0]["scalar"], cells[1]["scalar"], atol, rtol):
                    disagree += 1
        out[op] = {
            "n_formulas": len(subset),
            "mean_disagreement": (disagree / comparable) if comparable else None,
        }
    return out


def abstention_table(pairs: list[dict], backends: list[str]) -> dict[str, dict[str, Any]]:
    """Per backend: ok/abstain/error counts, abstain rate, and exc breakdown."""
    out: dict[str, dict[str, Any]] = {}
    total = len(pairs)
    for backend in backends:
        counts = {"ok": 0, "abstain": 0, "error": 0}
        excs: dict[str, int] = {}
        for pair in pairs:
            cell = pair["cells"][backend]
            counts[cell["status"]] += 1
            if cell["status"] in ("abstain", "error"):
                key = cell["exc"]
                excs[key] = excs.get(key, 0) + 1
        out[backend] = {
            **counts,
            "rate": (counts["abstain"] / total) if total else 0.0,
            "exc_breakdown": dict(sorted(excs.items(), key=lambda kv: -kv[1])),
        }
    return out


def assemble_product(backends: list[str], pairs: list[dict], atol: float, rtol: float) -> dict[str, Any]:
    """The full product: matrices + per-operator + abstention + regime slices."""
    regimes = sorted({p["regime"] for p in pairs})
    by_regime = {
        regime: scalar_matrix([p for p in pairs if p["regime"] == regime], backends, atol, rtol)
        for regime in regimes
    }
    # Equality predicates are resolved incompatibly across backends (Breach
    # BigM, TaLiRo abstains, others metric), so stratify scalar agreement by
    # whether the formula contains one; the "without_equality" matrix is the
    # cleaner cross-backend picture.
    by_equality = {
        "with_equality": scalar_matrix(
            [p for p in pairs if p.get("has_equality")], backends, atol, rtol
        ),
        "without_equality": scalar_matrix(
            [p for p in pairs if not p.get("has_equality")], backends, atol, rtol
        ),
    }
    return {
        "experiment": "E6 randomized cross-backend divergence",
        "backends": backends,
        "n_pairs": len(pairs),
        "atol": atol,
        "rtol": rtol,
        "scalar_matrix": scalar_matrix(pairs, backends, atol, rtol),
        "full_signal_matrix": full_signal_matrix(pairs, backends, atol, rtol),
        "per_operator": per_operator(pairs, backends, atol, rtol),
        "abstention": abstention_table(pairs, backends),
        "by_regime": by_regime,
        "by_equality": by_equality,
    }


def _fmt(value: float | None) -> str:
    return " n/a " if value is None else f"{value:5.2f}"


def _print_matrix(backends: list[str], matrix: dict[str, dict[str, float | None]], title: str) -> None:
    print(title)
    print("             " + "".join(f"{b[:6]:>7}" for b in backends))
    for bi in backends:
        print(f"{bi:>12} " + "".join(f"{_fmt(matrix[bi][bj])}  " for bj in backends))


def print_console(product: dict[str, Any]) -> None:
    backends = product["backends"]
    print(f"\nE6 randomized divergence  (n_pairs={product['n_pairs']}, "
          f"atol={product['atol']}, rtol={product['rtol']})\n")
    _print_matrix(backends, product["scalar_matrix"], "Scalar (t=0) pairwise agreement:")
    _print_matrix(
        backends,
        product["by_equality"]["without_equality"],
        "\nScalar agreement EXCLUDING formulas with equality predicates:",
    )
    print("\nAbstention (abstain rate, ok/abstain/error):")
    for backend in backends:
        a = product["abstention"][backend]
        print(f"  {backend:>14}  rate={a['rate']:.2f}  {a['ok']}/{a['abstain']}/{a['error']}  {a['exc_breakdown']}")
    print("\nPer-operator mean scalar disagreement:")
    for op in OPERATORS:
        entry = product["per_operator"][op]
        print(f"  {op:>12}  n={entry['n_formulas']:>4}  disagree={_fmt(entry['mean_disagreement'])}")


def emit_tex(product: dict[str, Any]) -> str:
    """Render the scalar agreement matrix as a tabular (tab:e6-divergence)."""
    backends = product["backends"]
    cols = "l" + "r" * len(backends)
    lines = [
        "% AUTO-GENERATED from extra/outputs/e6_fuzz/matrix.json",
        "% by extra/experiments/e6_fuzz/aggregate.py (tidystl repo); do not hand-edit.",
        f"% n_pairs={product['n_pairs']}, atol={product['atol']}, rtol={product['rtol']}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        " & " + " & ".join(b.replace("_", "\\_") for b in backends) + " \\\\",
        "\\hline",
    ]
    for bi in backends:
        cells = []
        for bj in backends:
            value = product["scalar_matrix"][bi][bj]
            cells.append("n/a" if value is None else f"{value:.2f}")
        lines.append(bi.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}%"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--emit-tex", type=Path, default=None, metavar="PATH")
    args = parser.parse_args()

    backends, pairs, atol, rtol = gather(args.result_dir)
    product = assemble_product(backends, pairs, atol, rtol)
    print_console(product)

    (args.result_dir / "matrix.json").write_text(
        json.dumps({k: product[k] for k in ("experiment", "backends", "n_pairs", "atol", "rtol", "scalar_matrix", "full_signal_matrix", "by_regime", "by_equality")}, indent=2) + "\n"
    )
    (args.result_dir / "per_operator.json").write_text(json.dumps(product["per_operator"], indent=2) + "\n")
    (args.result_dir / "abstention.json").write_text(json.dumps(product["abstention"], indent=2) + "\n")
    print(f"\nwrote matrix.json / per_operator.json / abstention.json to {args.result_dir}")

    body = emit_tex(product)
    if args.emit_tex is None:
        print("\n" + body, end="")
    else:
        args.emit_tex.write_text(body)
        print(f"wrote {args.emit_tex}")


if __name__ == "__main__":
    main()
