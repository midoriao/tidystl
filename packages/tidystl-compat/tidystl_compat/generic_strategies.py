"""Pure policy strategies for the generic backend (window reduction, scoring)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from tidystl.backends import algorithms

Reducer = Callable[..., NDArray[np.floating]]

#: Equality-predicate BigM constants (mirror Breach's STL_Formula.m values).
_EQ_ZERO_THRESHOLD = 1e-13
_EQ_BIGM = 10000.0
_EQ_EPSILON = 1e-9


def equality_score(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    policy: str,
) -> NDArray[np.floating]:
    """Score an equality predicate according to the given policy.

    ``"signed"``: ``-|left - right|``  (dense-time metric, matches native/rtamt/stlcgpp)
    ``"bigm"``:   ``+10000`` when ``|left - right| <= 1e-13``, else ``-10000``
    ``"epsilon"``: same BigM values but with looser threshold ``1e-9``
    """
    diff = np.abs(left - right)
    if policy == "signed":
        return -diff
    if policy == "bigm":
        return np.where(diff <= _EQ_ZERO_THRESHOLD, _EQ_BIGM, -_EQ_BIGM)
    if policy == "epsilon":
        return np.where(diff <= _EQ_EPSILON, _EQ_BIGM, -_EQ_BIGM)
    raise ValueError(f"unknown equality policy {policy!r}")


def pl_window_reduce(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    start: float,
    end: float,
    reducer: Reducer,
    *,
    endpoints: str,  # "interp" | "samples"
    boundary: str,  # "clamp" | "pessimistic"
    identity: float,
) -> NDArray[np.floating]:
    """PL window reduction parametrized by endpoint handling and boundary rule.

    A window contributes its interior samples (always) plus, under ``endpoints
    == "interp"``, the interpolated values at its clamped endpoints. When the
    window collects no candidates at all it is empty: ``pessimistic`` returns
    the reducer identity (matching taliro's empty-window rule), while ``clamp``
    falls back to the interpolated midpoint. A window starting past the trace
    end is empty too: ``clamp`` holds the last value, ``pessimistic`` returns
    the identity.
    """
    t_len = len(times)
    out = np.empty_like(rho)
    for i in range(t_len):
        t_lo = float(times[i]) + start
        t_hi = float(times[i]) + end
        if t_lo > float(times[-1]):
            out[:, i] = rho[:, -1] if boundary == "clamp" else identity
            continue
        t_lo_c = max(t_lo, float(times[0]))
        t_hi_c = min(t_hi, float(times[-1]))
        j_start = min(int(np.searchsorted(times, t_lo_c, side="left")), t_len)
        j_end = min(int(np.searchsorted(times, t_hi_c, side="right")), t_len)
        candidates: list[NDArray[np.floating]] = []
        if j_start < j_end:
            candidates.append(rho[:, j_start:j_end])
        if endpoints == "interp":
            eps = 1e-12
            if (
                j_start < t_len
                and abs(float(times[min(j_start, t_len - 1)]) - t_lo_c) > eps
                and t_lo_c > float(times[0]) + eps
            ):
                candidates.append(algorithms.interp_at(times, rho, t_lo_c)[:, np.newaxis])
            if (
                j_end > 0
                and abs(float(times[min(j_end - 1, t_len - 1)]) - t_hi_c) > eps
                and t_hi_c < float(times[-1]) - eps
            ):
                candidates.append(algorithms.interp_at(times, rho, t_hi_c)[:, np.newaxis])
        if candidates:
            out[:, i] = reducer(np.concatenate(candidates, axis=-1), axis=-1)
        elif boundary == "pessimistic":
            out[:, i] = identity
        else:
            midpoint = (t_lo_c + t_hi_c) / 2
            out[:, i] = reducer(algorithms.interp_at(times, rho, midpoint)[:, np.newaxis], axis=-1)
    return out
