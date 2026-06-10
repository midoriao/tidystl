from __future__ import annotations

from pathlib import Path

from tests._helpers.ground_truth import assert_csv_compatible
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal


def assert_rtamt_compatible(
    formula: Node,
    signal: Signal,
    rtamt_csv: str | Path,
    *,
    backend: EvaluationBackend | str | None = "rtamt",
    atol: float = 1e-6,
) -> None:
    """Assert that tidystl robustness matches RTAMT ground truth."""

    assert_csv_compatible(
        formula,
        signal,
        rtamt_csv,
        backend=backend,
        atol=atol,
    )
