from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal
from tidystl.frontend.evaluator import robustness


def assert_csv_compatible(
    formula: Node,
    signal: Signal,
    ground_truth_csv: str | Path,
    *,
    backend: EvaluationBackend | str | None = None,
    atol: float = 1e-6,
) -> None:
    """Assert that tidystl robustness matches a CSV ground-truth trace."""

    rho = robustness(formula, signal, backend)

    path = Path(ground_truth_csv)
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    expected_times = np.array([float(r["time"]) for r in rows])
    expected_rho = np.array([float(r["robustness"]) for r in rows])

    for i, t in enumerate(expected_times):
        idx = int(np.argmin(np.abs(signal.times - t)))
        actual = rho[0, idx]
        np.testing.assert_allclose(
            actual,
            expected_rho[i],
            atol=atol,
            err_msg=f"Mismatch at t={t}: tidystl={actual}, expected={expected_rho[i]}",
        )
