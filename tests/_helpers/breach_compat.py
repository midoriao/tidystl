from pathlib import Path

from tests._helpers.ground_truth import assert_csv_compatible
from tidystl.core.backend_interface import EvaluationBackend
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal


def assert_breach_compatible(
    formula: Node,
    signal: Signal,
    breach_csv: str | Path,
    *,
    backend: EvaluationBackend | str = "breach",
    atol: float = 1e-6,
) -> None:
    """Assert that tidystl robustness matches Breach ground truth.

    The CSV file should have columns: time, robustness
    (one row per timestep, single trace).

    Defaults to `backend="breach"` so these assertions always test the
    Breach-compatible backend regardless of what the global default is.
    """
    assert_csv_compatible(
        formula,
        signal,
        breach_csv,
        backend=backend,
        atol=atol,
    )
