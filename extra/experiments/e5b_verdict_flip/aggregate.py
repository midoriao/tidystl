from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FamilyKey = tuple[str, tuple[str, ...], tuple[float, ...], int]


def _family_key(result: dict[str, Any]) -> FamilyKey:
    sig = result["signal"]
    return (
        result["spec"],
        tuple(sig["names"]),
        tuple(float(v) for v in sig["base_values"]),
        int(result["verdict_t_index"]),
    )


def gather_facts(result_dir: Path) -> list[dict[str, Any]]:
    """Each run's ``result.json`` under ``result_dir``, ascending; partial runs skipped."""
    facts: list[dict[str, Any]] = []
    run_dirs = sorted(
        (p for p in result_dir.glob("run_*") if p.is_dir() and p.name.removeprefix("run_").isdigit()),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            facts.append(json.loads(result_path.read_text()))
        except json.JSONDecodeError:
            continue
    return facts


def index_records(facts: list[dict[str, Any]]) -> dict[tuple[FamilyKey, float, str], dict[str, Any]]:
    """Latest fact per ``(family_key, s, backend)``; ascending order so last wins."""
    index: dict[tuple[FamilyKey, float, str], dict[str, Any]] = {}
    for result in facts:
        s = float(result["signal"]["s"])
        index[(_family_key(result), s, result["backend"])] = result
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-values",
        default=None,
        help="comma-separated signal values of the ranking reference, e.g. '5.0,-1.0,5.0'",
    )
    args = parser.parse_args()

    facts = gather_facts(args.result_dir)
    if not facts:
        sys.exit(f"no run records under {args.result_dir}")
    ref_values: list[float] | None = None
    if args.reference_values is not None:
        ref_values = [float(v) for v in args.reference_values.split(",")]

    # Reference facts (matching ``values``) seed the ranking and are NOT
    # swept families themselves; partition them out before grouping.
    # Reference selection is float-exact: reference values are exact
    # decimals (e.g. 5.0,-1.0), stable across JSON round-trips.
    def _is_reference(fact: dict[str, Any]) -> bool:
        return ref_values is not None and fact["values"] == ref_values

    ref_facts = [f for f in facts if _is_reference(f)]
    swept = [f for f in facts if not _is_reference(f)]
    index = index_records(swept)
    # Latest reference robustness per (spec, backend).
    ref_rho_by: dict[tuple[str, str], list[float]] = {}
    for f in ref_facts:
        ref_rho_by[(f["spec"], f["backend"])] = f["robustness"]

    # Families and their backends, in deterministic order.
    families: dict[FamilyKey, set[str]] = {}
    s_count: dict[FamilyKey, set[float]] = {}
    for fam_key, s, backend in index:
        families.setdefault(fam_key, set()).add(backend)
        s_count.setdefault(fam_key, set()).add(s)

    # Drop a single-point family (the ranking reference) when a richer
    # swept family shares its spec; the reference is auxiliary, not a sweep.
    # Safety net: drops any residual reference-only family (a single s value)
    # whose spec already has a swept family; harmless when --reference-values
    # was given because those records were already pulled out above.
    rich_specs = {fk[0] for fk, ss in s_count.items() if len(ss) > 1}
    families = {
        fk: bs for fk, bs in families.items() if not (len(s_count[fk]) == 1 and fk[0] in rich_specs)
    }

    failures: list[str] = []
    for fam_key in sorted(families):
        backends = sorted(families[fam_key])
        spec, names, base_values, t_index = fam_key
        label = f"{spec} | names={list(names)} base={list(base_values)} t={t_index}"
        if len(backends) != 2:
            failures.append(f"family {label!r} has {len(backends)} backends ({backends}), need 2")
            continue
        backend_a, backend_b = backends

        # Paired s values present for both backends, sorted.
        s_a = {s for (fk, s, b) in index if fk == fam_key and b == backend_a}
        s_b = {s for (fk, s, b) in index if fk == fam_key and b == backend_b}
        paired = sorted(s_a & s_b)
        if not paired:
            failures.append(f"family {label!r}: no s value present for both backends")
            continue

        flips = 0
        for s in paired:
            rec_a = index[(fam_key, s, backend_a)]
            rec_b = index[(fam_key, s, backend_b)]
            if rec_a["verdict"] != rec_b["verdict"]:
                flips += 1
        n = len(paired)
        print(f"{spec}  {backend_a} vs {backend_b}  n={n} flips={flips} rate={flips / n:.4f}")

        # Ranking inversions, only if a reference record exists for this
        # spec under both backends (the boundary family has none).
        ref_min: dict[str, float] = {}
        for backend in backends:
            if (spec, backend) in ref_rho_by:
                ref_min[backend] = min(ref_rho_by[(spec, backend)])
        if len(ref_min) != 2:
            continue

        inversions = 0
        for s in paired:
            rec_a = index[(fam_key, s, backend_a)]
            rec_b = index[(fam_key, s, backend_b)]
            order_a = min(rec_a["robustness"]) < ref_min[backend_a]
            order_b = min(rec_b["robustness"]) < ref_min[backend_b]
            if order_a != order_b:
                inversions += 1
        print(
            f"{spec}  {backend_a} vs {backend_b}  n={n} inversions={inversions} "
            f"rate={inversions / n:.4f}"
        )

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
