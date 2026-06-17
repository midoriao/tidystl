from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.pymtl_cases import PYMTL_CASES, PymtlCase
from tests._helpers.pymtl_compat import assert_pymtl_compatible
from tidystl import Signal, parse

GROUND_TRUTH = Path(__file__).parent / "pymtl_ground_truth.jsonl"


def _signal(case: PymtlCase) -> Signal:
    return Signal.from_dict(
        times=np.array(case.times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in case.atoms.items()},
    )


@pytest.mark.parametrize("case", PYMTL_CASES, ids=lambda case: case.name)
def test_pymtl_ground_truth(case: PymtlCase) -> None:
    if not GROUND_TRUTH.exists():
        pytest.skip(
            "Run extra/other_tools/pymtl/generate_ground_truth.py first "
            "(pymtl_ground_truth.jsonl missing)"
        )
    formula = parse(case.tidystl_formula)
    signal = _signal(case)
    assert_pymtl_compatible(formula, signal, case.name)
