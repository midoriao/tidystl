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


class TestAssertJsonlCompatible:
    def _write(self, path: Path, name: str, rho: float) -> None:
        import json

        path.write_text(
            json.dumps({"name": name, "time": [0.0, 1.0, 2.0], "robustness": [rho, rho, rho]})
            + "\n"
        )

    def test_matching_output_passes(self, tmp_path: Path) -> None:
        from tests._helpers.ground_truth import assert_jsonl_compatible

        jsonl = tmp_path / "gt.jsonl"
        self._write(jsonl, "c", 3.0)
        sig = Signal.from_dict(
            times=np.array([0.0, 1.0, 2.0]), values={"x": np.array([[3.0, 3.0, 3.0]])}
        )
        assert_jsonl_compatible(_x_pos_predicate(), sig, jsonl, "c")

    def test_mismatched_output_raises(self, tmp_path: Path) -> None:
        from tests._helpers.ground_truth import assert_jsonl_compatible

        jsonl = tmp_path / "gt.jsonl"
        self._write(jsonl, "c", 999.0)
        sig = Signal.from_dict(
            times=np.array([0.0, 1.0, 2.0]), values={"x": np.array([[3.0, 3.0, 3.0]])}
        )
        with pytest.raises(AssertionError):
            assert_jsonl_compatible(_x_pos_predicate(), sig, jsonl, "c")

    def test_missing_name_raises_keyerror(self, tmp_path: Path) -> None:
        from tests._helpers.ground_truth import assert_jsonl_compatible

        jsonl = tmp_path / "gt.jsonl"
        self._write(jsonl, "c", 3.0)
        sig = Signal.from_dict(
            times=np.array([0.0, 1.0, 2.0]), values={"x": np.array([[3.0, 3.0, 3.0]])}
        )
        with pytest.raises(KeyError):
            assert_jsonl_compatible(_x_pos_predicate(), sig, jsonl, "absent")
