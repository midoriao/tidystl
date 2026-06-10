from pathlib import Path

import numpy as np
import pytest

from tidystl.core.nodes import Node
from tidystl.core.signal import Signal


def _x_pos_predicate() -> Node:
    return Node(
        kind="predicate",
        attrs={
            "name": "p",
            "op": ">=",
            "left": Node(kind="var", attrs={"name": "x"}),
            "right": Node(kind="const", attrs={"value": 0.0}),
        },
    )


class TestAssertBreachCompatible:
    def test_matching_output_passes(self, tmp_path: Path) -> None:
        from tests._helpers.breach_compat import assert_breach_compatible

        csv_file = tmp_path / "breach.csv"
        csv_file.write_text("time,robustness\n0.0,3.0\n1.0,3.0\n2.0,3.0\n")

        t = np.array([0.0, 1.0, 2.0])
        sig = Signal.from_dict(times=t, values={"x": np.array([[3.0, 3.0, 3.0]])})
        phi = _x_pos_predicate()

        assert_breach_compatible(phi, sig, csv_file)

    def test_mismatched_output_raises(self, tmp_path: Path) -> None:
        from tests._helpers.breach_compat import assert_breach_compatible

        csv_file = tmp_path / "breach.csv"
        csv_file.write_text("time,robustness\n0.0,999.0\n")

        t = np.array([0.0, 1.0, 2.0])
        sig = Signal.from_dict(times=t, values={"x": np.array([[3.0, 3.0, 3.0]])})
        phi = _x_pos_predicate()

        with pytest.raises(AssertionError):
            assert_breach_compatible(phi, sig, csv_file)
