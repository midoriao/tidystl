"""Piecewise-constant (PWC) signal sampling for RTAMT dense-time ground truth.

RTAMT models dense-time signals as piecewise-constant, right-continuous and held
from the left: for all ``t in [t_i, t_{i+1})`` the value is ``w(t_i)``
(arXiv:2501.18608). A dense ground-truth record stores only the breakpoints
where the constant value changes, so reading the robustness at an arbitrary time
means holding the value of the nearest breakpoint at or before that time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def pwc_sample_at(
    breakpoint_times: ArrayLike,
    breakpoint_values: ArrayLike,
    query_times: ArrayLike,
) -> NDArray[np.floating]:
    """Evaluate a PWC (right-continuous, held-from-left) step signal.

    ``breakpoint_times`` must be strictly increasing. The value returned for a
    query time ``t`` is ``breakpoint_values[k]`` for the largest ``k`` with
    ``breakpoint_times[k] <= t``; values at and after the final breakpoint hold
    the last value. A scalar query yields a 0-d array (usable as a float).

    Raises ``ValueError`` for an empty signal, mismatched array lengths, or a
    query time strictly before the first breakpoint (where the signal is
    undefined).
    """
    times = np.asarray(breakpoint_times, dtype=float)
    values = np.asarray(breakpoint_values, dtype=float)
    query = np.asarray(query_times, dtype=float)

    if times.ndim != 1 or values.ndim != 1:
        raise ValueError("breakpoint_times and breakpoint_values must be 1-D")
    if times.shape != values.shape:
        raise ValueError(
            f"breakpoint length mismatch: {times.shape} times vs {values.shape} values"
        )
    if times.size == 0:
        raise ValueError("cannot sample an empty PWC signal")

    # Index of the last breakpoint at or before each query time.
    idx = np.searchsorted(times, query, side="right") - 1
    if np.any(idx < 0):
        raise ValueError("query time before first breakpoint: PWC signal is undefined there")
    return values[idx]
