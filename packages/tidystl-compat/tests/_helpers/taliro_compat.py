from __future__ import annotations

from pathlib import Path

from tests._helpers.ground_truth import assert_jsonl_compatible
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal

GROUND_TRUTH = Path(__file__).parent.parent / "taliro_ground_truth.jsonl"


def assert_taliro_compatible(
    formula: Node,
    signal: Signal,
    name: str,
    *,
    backend: EvaluationBackend | str | None = "taliro",
    atol: float = 1e-6,
) -> None:
    """Assert that tidystl robustness matches TaLiRo (dp_taliro) ground truth.

    ``name`` keys into ``taliro_ground_truth.jsonl``. Each TaLiRo case holds a
    single point at t=0 (dp_taliro reports a scalar); the check compares
    ``rho[0, 0]``.
    """
    assert_jsonl_compatible(formula, signal, GROUND_TRUTH, name, backend=backend, atol=atol)
