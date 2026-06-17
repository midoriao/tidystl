from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.rtamt_cases import RTAMT_CASES, RtamtCase
from tests._helpers.rtamt_compat import assert_rtamt_compatible
from tidystl import Signal, parse

GROUND_TRUTH = Path(__file__).parent / "rtamt_ground_truth.jsonl"


def _signal(case: RtamtCase) -> Signal:
    return Signal.from_dict(
        times=np.array(case.times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in case.values.items()},
    )


@pytest.mark.parametrize("case", RTAMT_CASES, ids=lambda case: case.name)
def test_rtamt_ground_truth(case: RtamtCase) -> None:
    if not GROUND_TRUTH.exists():
        pytest.skip(
            "Run extra/other_tools/rtamt/generate_ground_truth.py first "
            "(rtamt_ground_truth.jsonl missing)"
        )
    formula = parse(case.tidystl_formula)
    signal = _signal(case)
    assert_rtamt_compatible(formula, signal, case.name)
