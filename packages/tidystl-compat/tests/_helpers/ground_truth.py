from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from tests._helpers.jsonl_io import read_jsonl
from tests._helpers.pwc import pwc_sample_at
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal
from tidystl.frontend.evaluator import robustness


@functools.cache
def _load_records(jsonl_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load a JSONL ground-truth file into ``{name: (times, robustness)}`` (cached per path)."""
    records: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rec in read_jsonl(jsonl_path):
        records[rec["name"]] = (
            np.atleast_1d(np.asarray(rec["time"], dtype=float)),
            np.atleast_1d(np.asarray(rec["robustness"], dtype=float)),
        )
    return records


def assert_jsonl_compatible(
    formula: Node,
    signal: Signal,
    jsonl_path: str | Path,
    name: str,
    *,
    backend: EvaluationBackend | str | None = None,
    atol: float = 1e-6,
) -> None:
    """Assert tidystl robustness matches a named case in a JSONL ground-truth file."""

    rho = robustness(formula, signal, backend)

    records = _load_records(Path(jsonl_path))
    if name not in records:
        raise KeyError(f"{name!r} not in {jsonl_path}")
    expected_times, expected_rho = records[name]

    for i, t in enumerate(expected_times):
        idx = int(np.argmin(np.abs(signal.times - t)))
        actual = rho[0, idx]
        np.testing.assert_allclose(
            actual,
            expected_rho[i],
            atol=atol,
            err_msg=f"Mismatch at t={t}: tidystl={actual}, expected={expected_rho[i]}",
        )


def assert_jsonl_pwc_compatible(
    formula: Node,
    signal: Signal,
    jsonl_path: str | Path,
    name: str,
    *,
    backend: EvaluationBackend | str | None = None,
    atol: float = 1e-6,
) -> None:
    """Assert tidystl robustness matches a piecewise-constant (PWC) ground truth.

    The ground-truth ``time``/``robustness`` arrays are PWC breakpoints (RTAMT
    dense-time: right-continuous, held from the left). The expected value at each
    of the signal's sample times is obtained by holding the value of the nearest
    breakpoint at or before that time (``pwc_sample_at``).

    Comparison is restricted to RTAMT's emitted domain ``[time[0], time[-1]]``.
    Bounded operators truncate that domain instead of padding past the trace
    end; beyond the last breakpoint RTAMT reports nothing, while the backend may
    legitimately compute STL-complete values there, so those samples are skipped.
    """

    rho = robustness(formula, signal, backend)

    records = _load_records(Path(jsonl_path))
    if name not in records:
        raise KeyError(f"{name!r} not in {jsonl_path}")
    gt_times, gt_rho = records[name]

    domain_lo = float(gt_times[0])
    domain_hi = float(gt_times[-1])
    eps = 1e-9
    compared = 0
    for i, t in enumerate(np.asarray(signal.times, dtype=float)):
        if t < domain_lo - eps or t > domain_hi + eps:
            continue
        expected = float(pwc_sample_at(gt_times, gt_rho, t))
        np.testing.assert_allclose(
            rho[0, i],
            expected,
            atol=atol,
            err_msg=f"{name}: mismatch at t={t}: tidystl={rho[0, i]}, expected={expected}",
        )
        compared += 1
    assert compared > 0, f"{name}: no signal samples fall within the ground-truth domain"
