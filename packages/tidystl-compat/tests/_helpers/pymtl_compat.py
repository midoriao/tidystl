from __future__ import annotations

from pathlib import Path

from tests._helpers.ground_truth import assert_jsonl_compatible
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal

GROUND_TRUTH = Path(__file__).parent.parent / "pymtl_ground_truth.jsonl"


def assert_pymtl_compatible(
    formula: Node,
    signal: Signal,
    name: str,
    *,
    backend: EvaluationBackend | str | None = "pymtl",
    atol: float = 1e-6,
) -> None:
    """Assert that tidystl robustness matches py-mtl ground truth (``name`` keys the JSONL)."""

    assert_jsonl_compatible(formula, signal, GROUND_TRUTH, name, backend=backend, atol=atol)
