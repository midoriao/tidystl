"""Generate RTAMT ground-truth CSV files from the shared test cases.

``rtamt`` ships in the ``experiments`` dependency group, so run this with:

    uv run --only-group experiments python extra/other_tools/rtamt/generate_ground_truth.py

It evaluates every case in ``tests/_helpers/rtamt_cases.py`` with the real
RTAMT engine and writes one ``<case>.csv`` per case into
``tests/rtamt_ground_truth/``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import rtamt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "tests" / "rtamt_ground_truth"


def _evaluate_case(formula: str, times: tuple[float, ...], values: dict[str, tuple[float, ...]]):
    spec = rtamt.StlDiscreteTimeSpecification()
    for name in values:
        spec.declare_var(name, "float")
    spec.spec = formula
    spec.parse()

    dataset = {"time": list(times)}
    dataset.update({name: list(trace) for name, trace in values.items()})
    return spec.evaluate(dataset)


def main() -> None:
    from tests._helpers.rtamt_cases import RTAMT_CASES

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for case in RTAMT_CASES:
        rows = _evaluate_case(case.rtamt_formula, case.times, case.values)
        out_path = OUT_DIR / f"{case.name}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("time", "robustness"))
            writer.writerows(rows)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
