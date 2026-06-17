"""Pure backend algorithms shared across concrete implementations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def interp_at(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    t: float,
) -> NDArray[np.floating]:
    """Linearly interpolate a batched trace at time ``t`` with endpoint clamping."""

    if t <= times[0]:
        return rho[:, 0].copy()
    if t >= times[-1]:
        return rho[:, -1].copy()
    idx = int(np.searchsorted(times, t, side="right")) - 1
    t0, t1 = float(times[idx]), float(times[idx + 1])
    if t1 == t0:
        return rho[:, idx].copy()
    alpha = (t - t0) / (t1 - t0)
    return rho[:, idx] + alpha * (rho[:, idx + 1] - rho[:, idx])


def sliding_reduce(
    times: NDArray[np.floating],
    rho: NDArray[np.floating],
    start: float,
    end: float,
    reducer: Callable[..., NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Apply a pointwise reduction over each signal window ``[t+start, t+end]``."""

    t_len = len(times)
    out = np.empty_like(rho)

    for i in range(t_len):
        t_lo = float(times[i]) + start
        t_hi = float(times[i]) + end
        t_lo_c = max(t_lo, float(times[0]))
        t_hi_c = min(t_hi, float(times[-1]))

        j_start = min(int(np.searchsorted(times, t_lo_c, side="left")), t_len)
        j_end = min(int(np.searchsorted(times, t_hi_c, side="right")), t_len)

        candidates: list[NDArray[np.floating]] = []
        if j_start < j_end:
            candidates.append(rho[:, j_start:j_end])

        eps = 1e-12
        if (
            j_start < t_len
            and abs(float(times[min(j_start, t_len - 1)]) - t_lo_c) > eps
            and t_lo_c > float(times[0]) + eps
        ):
            candidates.append(interp_at(times, rho, t_lo_c)[:, np.newaxis])
        if (
            j_end > 0
            and abs(float(times[min(j_end - 1, t_len - 1)]) - t_hi_c) > eps
            and t_hi_c < float(times[-1]) - eps
        ):
            candidates.append(interp_at(times, rho, t_hi_c)[:, np.newaxis])

        if candidates:
            all_vals = np.concatenate(candidates, axis=-1)
        else:
            midpoint = (t_lo_c + t_hi_c) / 2
            all_vals = interp_at(times, rho, midpoint)[:, np.newaxis]

        out[:, i] = reducer(all_vals, axis=-1)

    return out


def eval_until(
    times: NDArray[np.floating],
    rho_p: NDArray[np.floating],
    rho_q: NDArray[np.floating],
    start: float,
    end: float,
) -> NDArray[np.floating]:
    """Evaluate bounded-until robustness over batched traces on a fixed time grid."""

    n, t_len = rho_p.shape
    out = np.full((n, t_len), -np.inf)

    for i in range(t_len):
        t_i = float(times[i])
        j_start = min(int(np.searchsorted(times, t_i + start, side="left")), t_len)
        j_end = min(int(np.searchsorted(times, t_i + end, side="right")), t_len)

        for j in range(j_start, j_end):
            q_val = rho_q[:, j]
            p_min = np.min(rho_p[:, i : j + 1], axis=-1)
            out[:, i] = np.maximum(out[:, i], np.minimum(q_val, p_min))

        if t_i + end > float(times[-1]):
            q_ext = rho_q[:, -1]
            p_min = np.min(rho_p[:, i:], axis=-1)
            out[:, i] = np.maximum(out[:, i], np.minimum(q_ext, p_min))

    return out


def discrete_sliding_reduce(
    rho: NDArray[np.floating],
    start: int,
    end: int,
    reducer: Callable[..., NDArray[np.floating]],
    empty_value: float,
) -> NDArray[np.floating]:
    """Apply a reduction over discrete windows ``[i + start, i + end]``.

    If the window starts beyond the available trace, the result is filled with
    ``empty_value``. This matches RTAMT's discrete-time bounded-future
    semantics for ``always`` and ``eventually``.
    """

    n, t_len = rho.shape
    out = np.full((n, t_len), empty_value, dtype=rho.dtype)

    for i in range(t_len):
        j_start = i + start
        if j_start >= t_len:
            continue
        j_end = min(i + end + 1, t_len)
        out[:, i] = reducer(rho[:, j_start:j_end], axis=-1)

    return out


def discrete_sliding_reduce_last(
    rho: NDArray[np.floating],
    start: int,
    end: int,
    reducer: Callable[..., NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Apply a reduction over discrete windows with last-value extension.

    This matches the practical compatibility target used for STLCG++ fixture
    generation with ``padding="last"``: truncated windows are extended by
    repeating the final sample rather than returning an empty sentinel.
    """

    n, t_len = rho.shape
    out = np.empty((n, t_len), dtype=rho.dtype)

    for i in range(t_len):
        j_start = i + start
        if j_start >= t_len:
            out[:, i] = rho[:, -1]
            continue
        j_end = min(i + end + 1, t_len)
        out[:, i] = reducer(rho[:, j_start:j_end], axis=-1)

    return out


def discrete_eval_until(
    rho_p: NDArray[np.floating],
    rho_q: NDArray[np.floating],
    start: int,
    end: int,
) -> NDArray[np.floating]:
    """Evaluate bounded-until over a discrete-time trace.

    This follows RTAMT's documented discrete-time future semantics:
    ``rho(phi U[a,b] psi, w, t)`` is ``-inf`` when ``t + a`` is beyond the
    trace, otherwise it takes the max over witness indices ``j`` in
    ``[t + a, t + b]`` of ``min(rho(psi, j), min_{k in [t, j)} rho(phi, k))``.
    """

    n, t_len = rho_p.shape
    out = np.full((n, t_len), -np.inf, dtype=rho_p.dtype)

    for i in range(t_len):
        j_start = i + start
        if j_start >= t_len:
            continue

        j_stop = min(i + end, t_len - 1)
        p_prefix = np.full(n, np.inf, dtype=rho_p.dtype)

        for j in range(i, j_start):
            p_prefix = np.minimum(p_prefix, rho_p[:, j])

        for j in range(j_start, j_stop + 1):
            out[:, i] = np.maximum(out[:, i], np.minimum(rho_q[:, j], p_prefix))
            p_prefix = np.minimum(p_prefix, rho_p[:, j])

    return out


def discrete_eval_until_inclusive_last(
    rho_p: NDArray[np.floating],
    rho_q: NDArray[np.floating],
    start: int,
    end: int,
) -> NDArray[np.floating]:
    """Evaluate bounded-until with an inclusive left prefix and last extension.

    This matches the current STLCG++ fixture semantics when generated with
    ``padding="last"``:

    - the witness timestep contributes to the left-prefix minimum
    - out-of-bounds future indices reuse the final sample
    """

    n, t_len = rho_p.shape
    out = np.empty((n, t_len), dtype=rho_p.dtype)

    for i in range(t_len):
        best = np.full(n, -np.inf, dtype=rho_p.dtype)

        for offset in range(start, end + 1):
            j = i + offset
            j_clamped = min(j, t_len - 1)
            q_val = rho_q[:, j_clamped]
            p_min = np.min(rho_p[:, i : j_clamped + 1], axis=-1)
            best = np.maximum(best, np.minimum(q_val, p_min))

        out[:, i] = best

    return out
