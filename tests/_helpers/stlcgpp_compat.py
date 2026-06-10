from __future__ import annotations

from pathlib import Path

from tests._helpers.ground_truth import assert_csv_compatible
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal


def assert_stlcgpp_compatible(
    formula: Node,
    signal: Signal,
    stlcgpp_csv: str | Path,
    *,
    backend: EvaluationBackend | str | None = "stlcgpp",
    atol: float = 1e-6,
) -> None:
    """Assert that tidystl robustness matches STLCG++ ground truth."""

    assert_csv_compatible(
        formula,
        signal,
        stlcgpp_csv,
        backend=backend,
        atol=atol,
    )
