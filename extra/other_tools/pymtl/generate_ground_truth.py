"""Generate py-metric-temporal-logic ground-truth JSONL from the shared cases.

``metric-temporal-logic`` ships in the ``experiments`` dependency group, so run:

    uv run --only-group experiments python extra/other_tools/pymtl/generate_ground_truth.py

It evaluates every case in ``packages/tidystl-compat/tests/_helpers/pymtl_cases.py`` with the real ``mtl``
engine and writes a single ``pymtl_ground_truth.jsonl`` file into
``packages/tidystl-compat/tests/``.

Each entry's robustness is py-mtl's value sampled at original case times. Times
beyond py-mtl's defined output domain (its last breakpoint) are omitted so the
compat test only checks the region py-mtl actually defines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mtl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "tidystl-compat"))

from tests._helpers.jsonl_io import write_jsonl  # noqa: E402

OUT_FILE = REPO_ROOT / "packages" / "tidystl-compat" / "tests" / "pymtl_ground_truth.jsonl"
DT = 0.1


def _pc_lookup(breakpoints: list[tuple[float, float]], t: float) -> float:
    """Piecewise-constant value at ``t`` (value of the last breakpoint <= t)."""
    value = breakpoints[0][1]
    for bt, bv in breakpoints:
        if bt <= t + 1e-9:
            value = bv
        else:
            break
    return float(value)


def _rows_for_case(case) -> list[tuple[float, float]]:
    phi = mtl.parse(case.pymtl_formula)
    trace = {name: list(zip(case.times, vals, strict=True)) for name, vals in case.atoms.items()}
    signal = phi(trace, time=None, quantitative=True, dt=DT)
    breakpoints = [(float(t), float(v)) for t, v in signal]
    domain_end = breakpoints[-1][0]
    rows: list[tuple[float, float]] = []
    for t in case.times:
        if t > domain_end + 1e-9:
            continue
        rows.append((float(t), _pc_lookup(breakpoints, t)))
    return rows


def main() -> None:
    from tests._helpers.pymtl_cases import PYMTL_CASES

    records = []
    for case in PYMTL_CASES:
        rows = _rows_for_case(case)
        records.append(
            {
                "name": case.name,
                "time": [float(t) for t, _ in rows],
                "robustness": [float(r) for _, r in rows],
            }
        )
    write_jsonl(OUT_FILE, records)
    print(f"wrote {OUT_FILE} ({len(records)} cases)")


if __name__ == "__main__":
    main()
