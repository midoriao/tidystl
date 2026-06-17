"""Tests for the PWC-aware comparison primitive and the dense RTAMT fixtures.

RTAMT dense-time signals are piecewise-constant (PWC), right-continuous and held
from the left: for all ``t in [t_i, t_{i+1})`` the value is ``w(t_i)``
(arXiv:2501.18608). ``pwc_sample_at`` evaluates that step function.

The fixture-consistency test pins the documented relationship between the two
ground-truth files: for every operator RTAMT implements identically across its
two time models, PWC-sampling the dense fixture on a case's input grid must
reproduce the discrete fixture wherever the discrete value is finite. The two
diverge only at end-of-trace, where the discrete path pads ``inf``/``-inf`` for
bounded operators whose window runs past the trace, while the dense path simply
truncates the time domain.

The bounded ``until`` operator is the exception: RTAMT evaluates it with
different algorithms in the two time models (the discrete backend uses a
witness-exclusive left prefix; the dense engine uses ``until_timed_operation``),
so dense and discrete disagree on the shared finite domain. That divergence is
pinned separately by ``test_dense_until_diverges_from_discrete``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.jsonl_io import read_jsonl
from tests._helpers.pwc import pwc_sample_at
from tests._helpers.rtamt_cases import RTAMT_CASES, RtamtCase

TESTS_DIR = Path(__file__).parent
DISCRETE_GT = TESTS_DIR / "rtamt_ground_truth.jsonl"
DENSE_GT = TESTS_DIR / "rtamt_dense_ground_truth.jsonl"

CASES_BY_NAME = {case.name: case for case in RTAMT_CASES}

# `until` is implemented with different algorithms across RTAMT's two time
# models, so the dense/discrete agreement invariant does not extend to it.
_NON_UNTIL = tuple(c for c in RTAMT_CASES if "until" not in c.rtamt_formula)

# `until` cases that actually disagree on the shared finite domain.
# (`until_never_sat` coincidentally agrees because its value is constant -1.)
_UNTIL_DIVERGENT = ("until_basic", "until_tight", "until_boundary")


def _load(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        rec["name"]: (
            np.asarray(rec["time"], dtype=float),
            np.asarray(rec["robustness"], dtype=float),
        )
        for rec in read_jsonl(path)
    }


# --------------------------------------------------------------------------- #
# pwc_sample_at: the PWC step-evaluation primitive
# --------------------------------------------------------------------------- #


def test_pwc_holds_value_from_the_left_between_breakpoints() -> None:
    times = np.array([0.0, 1.0, 3.0])
    values = np.array([2.0, -1.0, 4.0])
    # [1, 3) holds -1; the value is NOT interpolated toward 4.
    got = pwc_sample_at(times, values, np.array([1.0, 1.5, 2.0, 2.999]))
    np.testing.assert_array_equal(got, np.array([-1.0, -1.0, -1.0, -1.0]))


def test_pwc_is_exact_at_breakpoints() -> None:
    times = np.array([0.0, 1.0, 3.0])
    values = np.array([2.0, -1.0, 4.0])
    got = pwc_sample_at(times, values, times)
    np.testing.assert_array_equal(got, values)


def test_pwc_holds_last_value_past_final_breakpoint() -> None:
    times = np.array([0.0, 3.0])
    values = np.array([2.0, 4.0])
    got = pwc_sample_at(times, values, np.array([3.0, 5.0, 100.0]))
    np.testing.assert_array_equal(got, np.array([4.0, 4.0, 4.0]))


def test_pwc_accepts_scalar_query() -> None:
    times = np.array([0.0, 1.0])
    values = np.array([2.0, -1.0])
    assert pwc_sample_at(times, values, 0.5) == 2.0


def test_pwc_rejects_query_before_first_breakpoint() -> None:
    times = np.array([0.0, 1.0])
    values = np.array([2.0, -1.0])
    with pytest.raises(ValueError, match="before first breakpoint"):
        pwc_sample_at(times, values, -0.5)


def test_pwc_rejects_empty_signal() -> None:
    with pytest.raises(ValueError, match="empty"):
        pwc_sample_at(np.array([]), np.array([]), 0.0)


def test_pwc_reconstructs_and_two_fixture() -> None:
    # and_two: x=(3,-1,2,4,-2), y=(1,2,-1,3,5); min is constant -1 on [1,3),
    # so the dense fixture drops t=2. Sampling on the full grid restores it.
    dense = _load(DENSE_GT)["and_two"]
    got = pwc_sample_at(*dense, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(got, np.array([1.0, -1.0, -1.0, 3.0, -2.0]))


# --------------------------------------------------------------------------- #
# Fixture consistency: dense PWC == discrete on the finite (non-truncated) domain
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _NON_UNTIL, ids=lambda c: c.name)
def test_dense_pwc_matches_discrete_on_finite_domain(case: RtamtCase) -> None:
    discrete = _load(DISCRETE_GT)
    dense = _load(DENSE_GT)
    assert case.name in dense, f"{case.name} missing from dense ground truth"

    grid = np.asarray(case.times, dtype=float)
    disc_t, disc_r = discrete[case.name]
    np.testing.assert_array_equal(disc_t, grid)  # discrete is grid-aligned

    sampled = pwc_sample_at(*dense[case.name], grid)

    finite = np.isfinite(disc_r)
    assert finite.any(), f"{case.name} has no finite discrete values to compare"
    np.testing.assert_allclose(
        sampled[finite],
        disc_r[finite],
        atol=1e-9,
        err_msg=f"{case.name}: dense PWC disagrees with discrete on the finite domain",
    )


@pytest.mark.parametrize("name", _UNTIL_DIVERGENT)
def test_dense_until_diverges_from_discrete(name: str) -> None:
    # RTAMT's dense and discrete `until` use different algorithms; they disagree
    # on the shared finite domain. Pin the divergence so a future change to
    # either model (or our fixtures) surfaces here rather than silently.
    grid = np.asarray(CASES_BY_NAME[name].times, dtype=float)
    _disc_t, disc_r = _load(DISCRETE_GT)[name]
    sampled = pwc_sample_at(*_load(DENSE_GT)[name], grid)

    finite = np.isfinite(disc_r)
    assert not np.allclose(sampled[finite], disc_r[finite], atol=1e-9), (
        f"{name}: dense and discrete until unexpectedly agree on the finite domain"
    )
