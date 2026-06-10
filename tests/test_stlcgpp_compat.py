from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.stlcgpp_cases import STLCGPP_CASES, StlcgppCase
from tests._helpers.stlcgpp_compat import assert_stlcgpp_compatible
from tidystl import Signal, parse

GROUND_TRUTH = Path(__file__).parent / "stlcgpp_ground_truth"


def _needs(name: str):
    return pytest.mark.skipif(
        not (GROUND_TRUTH / f"{name}.csv").exists(),
        reason=f"Run extra/other_tools/stlcgpp/generate_ground_truth.py first ({name}.csv missing)",
    )


def _signal(case: StlcgppCase) -> Signal:
    return Signal.from_dict(
        times=np.array(case.times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in case.values.items()},
    )


@pytest.mark.parametrize("case", STLCGPP_CASES, ids=lambda case: case.name)
def test_stlcgpp_ground_truth(case: StlcgppCase) -> None:
    marker = _needs(case.name)
    if marker.args[0]:
        pytest.skip(marker.kwargs["reason"])

    formula = parse(case.tidystl_formula)
    signal = _signal(case)
    assert_stlcgpp_compatible(formula, signal, GROUND_TRUTH / f"{case.name}.csv")
