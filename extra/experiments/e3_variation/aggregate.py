"""E3 aggregate: assemble ``variation.json`` / ``tab:variation`` from the
per-variant records.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT = "e3_variation"


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

# Row catalog: (variant, display name, layer summary), in table order.
# The full-suite check is reported separately, not as a row. Must stay
# in sync with the VARIANT_CATALOG order in run.py and the batch.sh
# sweep. Layer cells are deliberately terse; the prose paragraph carries
# the detail.
VARIANTS = (
    ("breach", "Breach-compatible", "executor op overrides"),
    ("rtamt", "RTAMT-compatible", "own lowering + executor"),
    ("stlcgpp", "STLCG++-compatible", "own lowering + executor"),
    ("stlcgpp_torch", "STLCG++ Torch executor", "executor only"),
    ("tidystl_simd", "Rust SIMD executor", "executor only (native ext.)"),
)
SUITE_VARIANT = "full_suite"

# Display groups of the tex table, keyed by the records' ``kind`` field.
GROUPS = (
    ("semantic", "Semantic variants (what is computed)"),
    ("computational", "Computational variants (how it is computed)"),
)


def select_records(result_dir: Path) -> dict[str, RunRecord]:
    """Latest record per variant (gather_facts is ascending: last wins)."""
    latest: dict[str, RunRecord] = {}
    for record in gather_facts(result_dir):
        latest[record.result["variant"]] = record
    return latest


def assemble_product(records: dict[str, RunRecord]) -> dict[str, Any]:
    """Validate the record set and build the ``variation.json`` product."""
    missing_rows = [v for v, _, _ in VARIANTS if v not in records]
    if missing_rows:
        raise SystemExit(f"missing variant record(s): {', '.join(missing_rows)}; run batch.sh")
    if SUITE_VARIANT not in records:
        raise SystemExit(
            f"missing the {SUITE_VARIANT!r} record (whole-library check); run batch.sh"
        )

    rows = [records[v].result for v, _, _ in VARIANTS]
    return {
        "experiment": "E3 variation table",
        "loc_rule": rows[0]["loc_rule"],
        "full_suite": records[SUITE_VARIANT].result["tests"],
        "rows": rows,
    }


def print_console_table(product: dict[str, Any]) -> None:
    print(f"E3 variation table  (loc_rule: {product['loc_rule'][:60]}...)\n")
    print(f"{'variant':<16}{'kind':<16}{'pyLOC':>6}{'rsLOC':>7}  tests")
    for row in product["rows"]:
        tests = row["tests"]
        status = f"{tests['passed']} passed" + ("" if tests["ok"] else " [FAIL]")
        rust = str(row["rust_loc"]) if row["rust_loc"] is not None else "--"
        print(f"{row['variant']:<16}{row['kind']:<16}{row['python_loc']:>6}{rust:>7}  {status}")
    print(f"\nfull suite: {product['full_suite']['summary']}")


def loc_cell(row: dict[str, Any]) -> str:
    if row["rust_loc"]:
        return f"{row['python_loc']} + {row['rust_loc']} Rust"
    return str(row["python_loc"])


def emit_tex(product: dict[str, Any]) -> str:
    """Render the ``tab:variation`` tabular body from the product."""
    rows = {r["variant"]: r for r in product["rows"]}
    lines = [
        "% AUTO-GENERATED from extra/outputs/e3_variation/variation.json",
        "% by extra/experiments/e3_variation/aggregate.py (tidystl repo); do not hand-edit.",
        f"% LOC rule: {product['loc_rule']}",
        f"% Full suite: {product['full_suite']['passed']} passed, "
        f"{product['full_suite']['failed']} failed.",
        "\\setlength{\\tabcolsep}{4pt}%",
        "\\begin{tabular}{l l r r}",
        "\\hline",
        "Variant & Layer changed & LOC & Tests \\\\",
        "\\hline",
    ]
    for kind, label in GROUPS:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\")
        for variant, name, layer in VARIANTS:
            row = rows[variant]
            if row["kind"] != kind:
                continue
            if not row["tests"]["ok"]:
                raise SystemExit(f"{variant}: variant tests not green; refusing to emit")
            lines.append(
                f"\\quad {name} & {layer} & {loc_cell(row)} & {row['tests']['passed']} \\\\"
            )
        lines.append("\\hline")
    lines.append("\\end{tabular}%")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--emit-tex", type=Path, default=None, metavar="PATH")
    args = parser.parse_args()

    product = assemble_product(select_records(args.result_dir))
    print_console_table(product)

    product_path = args.result_dir / "variation.json"
    product_path.write_text(json.dumps(product, indent=2) + "\n")
    print(f"\nwrote {product_path}")

    body = emit_tex(product)
    if args.emit_tex is None:
        print("\n" + body, end="")
    else:
        args.emit_tex.write_text(body)
        print(f"wrote {args.emit_tex}")


if __name__ == "__main__":
    main()
