from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.rtamt_cases import RTAMT_CASES, RtamtCase
from tests._helpers.rtamt_compat import assert_rtamt_compatible
from tidystl import Signal, parse

GROUND_TRUTH = Path(__file__).parent / "rtamt_ground_truth"


def _needs(name: str):
    return pytest.mark.skipif(
        not (GROUND_TRUTH / f"{name}.csv").exists(),
        reason=f"Run extra/other_tools/rtamt/generate_ground_truth.py first ({name}.csv missing)",
    )


def _signal(case: RtamtCase) -> Signal:
    return Signal.from_dict(
        times=np.array(case.times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in case.values.items()},
    )


@pytest.mark.parametrize("case", RTAMT_CASES, ids=lambda case: case.name)
def test_rtamt_ground_truth(case: RtamtCase) -> None:
    marker = _needs(case.name)
    if marker.args[0]:
        pytest.skip(marker.kwargs["reason"])

    formula = parse(case.tidystl_formula)
    signal = _signal(case)
    assert_rtamt_compatible(formula, signal, GROUND_TRUTH / f"{case.name}.csv")
