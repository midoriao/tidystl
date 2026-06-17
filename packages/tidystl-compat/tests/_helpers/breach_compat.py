from __future__ import annotations

from pathlib import Path

from tests._helpers.ground_truth import assert_jsonl_compatible
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal

GROUND_TRUTH = Path(__file__).parent.parent / "breach_ground_truth.jsonl"


def assert_breach_compatible(
    formula: Node,
    signal: Signal,
    name: str,
    *,
    backend: EvaluationBackend | str = "breach",
    atol: float = 1e-6,
) -> None:
    """Assert that tidystl robustness matches Breach ground truth.

    ``name`` keys into ``breach_ground_truth.jsonl``. Defaults to
    ``backend="breach"`` so these assertions test the Breach-compatible backend
    regardless of the global default.
    """
    assert_jsonl_compatible(formula, signal, GROUND_TRUTH, name, backend=backend, atol=atol)
