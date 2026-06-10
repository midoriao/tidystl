"""Shared timing and signal-generation helpers for the E4 runners.

stdlib + numpy only, so importing it does NOT pull in tidystl -- safe in the
stlcgpp ``--cell`` child, which must stay light.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np


def make_values(
    batch: int, timesteps: int, seed: int, signal_duration: float
) -> tuple[Any, Any, Any]:
    """Deterministic (times, x, y) for one batch x timestep cell."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, signal_duration, timesteps)
    x = rng.standard_normal((batch, timesteps))
    y = rng.standard_normal((batch, timesteps))
    return times, x, y


def time_with_repeat(
    fn: Callable[[], object], repeats: int, n_warmups: int = 1
) -> tuple[float, float]:
    """Run ``fn`` for the given number of repeats, returning mean and stdev of the timings in milliseconds."""
    timings: list[float] = []
    for _ in range(n_warmups):
        fn()  # warmup (excluded)
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(timings), statistics.stdev(timings) if len(timings) > 1 else 0.0


def time_adaptive(
    fn: Callable[[], object], repeats: int, slow_cell_s: float
) -> tuple[float, float, int]:
    """Like ``time_with_repeat`` but adaptive: one warmup (excluded), then
    ``repeats`` timed runs unless the warmup already exceeds ``slow_cell_s``
    (then run once). Returns (mean_ms, std_ms, n_repeats)."""
    t0 = time.perf_counter()
    fn()  # warmup (excluded)
    warmup_s = time.perf_counter() - t0
    n_repeats = 1 if warmup_s > slow_cell_s else repeats
    timings: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1e3)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return statistics.mean(timings), std, n_repeats
