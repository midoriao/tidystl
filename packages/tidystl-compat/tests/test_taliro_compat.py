from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.jsonl_io import read_jsonl
from tests._helpers.taliro_cases import TALIRO_CASES, TaliroCase
from tests._helpers.taliro_compat import assert_taliro_compatible
from tidystl import Signal, parse

GROUND_TRUTH = Path(__file__).parent / "taliro_ground_truth.jsonl"


def _signal(case: TaliroCase) -> Signal:
    return Signal.from_dict(
        times=np.array(case.times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in case.values.items()},
    )


@pytest.mark.parametrize("case", TALIRO_CASES, ids=lambda case: case.name)
def test_taliro_ground_truth(case: TaliroCase) -> None:
    if not GROUND_TRUTH.exists():
        pytest.skip(
            "Run extra/other_tools/taliro generator first (taliro_ground_truth.jsonl missing)"
        )
    assert_taliro_compatible(parse(case.tidystl_formula), _signal(case), case.name)


def test_case_names_match_fixtures() -> None:
    case_names = {case.name for case in TALIRO_CASES}
    fixture_names = {rec["name"] for rec in read_jsonl(GROUND_TRUTH)}
    assert case_names == fixture_names, (
        f"drift: only-in-cases={case_names - fixture_names}, "
        f"only-in-fixtures={fixture_names - case_names}"
    )
