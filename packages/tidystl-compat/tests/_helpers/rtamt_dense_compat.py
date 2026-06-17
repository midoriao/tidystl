from __future__ import annotations

from pathlib import Path

from tests._helpers.ground_truth import assert_jsonl_pwc_compatible
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal

GROUND_TRUTH = Path(__file__).parent.parent / "rtamt_dense_ground_truth.jsonl"


def assert_rtamt_dense_compatible(
    formula: Node,
    signal: Signal,
    name: str,
    *,
    backend: EvaluationBackend | str | None = "rtamt_dense",
    atol: float = 1e-6,
) -> None:
    """Assert tidystl robustness matches RTAMT dense-time (PWC) ground truth."""

    assert_jsonl_pwc_compatible(formula, signal, GROUND_TRUTH, name, backend=backend, atol=atol)
