"""Generate RTAMT ground-truth JSONL files from the shared test cases.

``rtamt`` ships in the ``experiments`` dependency group. Its pinned
``antlr4-python3-runtime==4.7`` imports ``typing.io``, which was removed in
Python 3.13, so this must run under Python 3.12 (or earlier):

    uv run --python 3.12 --only-group experiments \
        python extra/other_tools/rtamt/generate_ground_truth.py

It evaluates every case in ``packages/tidystl-compat/tests/_helpers/rtamt_cases.py``
with the real RTAMT engine under BOTH time models and writes two files into
``packages/tidystl-compat/tests/``:

- ``rtamt_ground_truth.jsonl``       -- discrete-time (samples on the input grid)
- ``rtamt_dense_ground_truth.jsonl`` -- dense-time (piecewise-linear breakpoints)

The two share the ``{name, time[], robustness[]}`` schema, but the meaning of
``time``/``robustness`` differs: discrete rows are samples aligned to the input
grid, whereas dense rows are piecewise-constant (PWC) breakpoints. RTAMT models
dense-time signals as right-continuous and held from the left -- for all
``t in [t_i, t_{i+1})`` the value is ``w(t_i)`` (arXiv:2501.18608) -- so a
breakpoint is emitted only where the constant value changes and the value
BETWEEN two breakpoints is held constant, NOT interpolated. Dense output also
truncates the time domain for bounded operators (no ``inf``/``-inf``
end-of-trace sentinels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import rtamt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "tidystl-compat"))

from tests._helpers.jsonl_io import write_jsonl  # noqa: E402

TESTS_DIR = REPO_ROOT / "packages" / "tidystl-compat" / "tests"
DISCRETE_OUT = TESTS_DIR / "rtamt_ground_truth.jsonl"
DENSE_OUT = TESTS_DIR / "rtamt_dense_ground_truth.jsonl"


def _evaluate_discrete(
    formula: str, times: tuple[float, ...], values: dict[str, tuple[float, ...]]
):
    spec = rtamt.StlDiscreteTimeSpecification()
    for name in values:
        spec.declare_var(name, "float")
    spec.spec = formula
    spec.parse()

    dataset = {"time": list(times)}
    dataset.update({name: list(trace) for name, trace in values.items()})
    return spec.evaluate(dataset)


def _evaluate_dense(formula: str, times: tuple[float, ...], values: dict[str, tuple[float, ...]]):
    spec = rtamt.StlDenseTimeSpecification()
    for name in values:
        spec.declare_var(name, "float")
    spec.spec = formula
    spec.parse()

    # Dense-time takes one ``[name, [[t, v], ...]]`` segment list per variable.
    args = [
        [name, [[float(t), float(v)] for t, v in zip(times, trace, strict=True)]]
        for name, trace in values.items()
    ]
    return spec.evaluate(*args)


def main() -> None:
    from tests._helpers.rtamt_cases import RTAMT_CASES

    discrete_records = []
    dense_records = []
    for case in RTAMT_CASES:
        discrete_rows = _evaluate_discrete(case.rtamt_formula, case.times, case.values)
        discrete_records.append(
            {
                "name": case.name,
                "time": [float(t) for t, _ in discrete_rows],
                "robustness": [float(r) for _, r in discrete_rows],
            }
        )
        dense_rows = _evaluate_dense(case.rtamt_formula, case.times, case.values)
        dense_records.append(
            {
                "name": case.name,
                "time": [float(t) for t, _ in dense_rows],
                "robustness": [float(r) for _, r in dense_rows],
            }
        )

    write_jsonl(DISCRETE_OUT, discrete_records)
    print(f"wrote {DISCRETE_OUT} ({len(discrete_records)} cases)")
    write_jsonl(DENSE_OUT, dense_records)
    print(f"wrote {DENSE_OUT} ({len(dense_records)} cases)")


if __name__ == "__main__":
    main()
