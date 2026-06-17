"""Breach compatibility tests.

Each case in ``BREACH_CASES`` defines the same signal and formula used in
``extra/other_tools/breach/generate_ground_truth.m``, then compares tidystl
output against the Breach ground truth in ``breach_ground_truth.jsonl`` (one
JSON object per case).

Run ``extra/other_tools/breach/generate_ground_truth.m`` in MATLAB first to
generate that file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.breach_cases import BREACH_CASES, BreachCase
from tests._helpers.breach_compat import assert_breach_compatible
from tidystl import Signal, parse

GROUND_TRUTH = Path(__file__).parent / "breach_ground_truth.jsonl"


def _signal(case: BreachCase) -> Signal:
    return Signal.from_dict(
        times=np.array(case.times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in case.values.items()},
    )


@pytest.mark.parametrize("case", BREACH_CASES, ids=lambda case: case.name)
def test_breach_ground_truth(case: BreachCase) -> None:
    if not GROUND_TRUTH.exists():
        pytest.skip(
            "Run extra/other_tools/breach/generate_ground_truth.m first "
            "(breach_ground_truth.jsonl missing)"
        )
    formula = parse(case.tidystl_formula)
    signal = _signal(case)
    assert_breach_compatible(formula, signal, case.name, atol=case.atol)


def test_case_names_match_fixtures() -> None:
    """Every recorded Breach case has a fixture and vice versa (drift guard)."""
    import json

    case_names = {case.name for case in BREACH_CASES}
    record_names = {
        json.loads(line)["name"] for line in GROUND_TRUTH.read_text().splitlines() if line.strip()
    }
    assert case_names == record_names, (
        f"drift: only-in-cases={case_names - record_names}, "
        f"only-in-records={record_names - case_names}"
    )
