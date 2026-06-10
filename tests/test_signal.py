import numpy as np
import pytest

from tidystl.core.signal import Signal


class TestSignalInit:
    def test_shape_and_fields(self) -> None:
        values = np.zeros((2, 3, 10))
        times = np.linspace(0, 1, 10)
        labels = {"x": 0, "y": 1, "z": 2}
        sig = Signal(values=values, times=times, labels=labels)
        assert sig.values.shape == (2, 3, 10)
        assert sig.times.shape == (10,)
        assert sig.labels == {"x": 0, "y": 1, "z": 2}


class TestSignalFromDict:
    def test_from_dict_stacks_variables(self) -> None:
        times = np.linspace(0, 1, 5)
        x = np.ones((3, 5))
        y = np.zeros((3, 5))
        sig = Signal.from_dict(times=times, values={"x": x, "y": y})
        assert sig.values.shape == (3, 2, 5)
        assert sig.labels == {"x": 0, "y": 1}
        np.testing.assert_array_equal(sig["x"], x)
        np.testing.assert_array_equal(sig["y"], y)

    def test_from_dict_length_mismatch_raises(self) -> None:
        times = np.linspace(0, 1, 5)
        x = np.ones((3, 7))  # 7 != 5
        with pytest.raises(ValueError, match="values\\['x'\\].*7.*5"):
            Signal.from_dict(times=times, values={"x": x})

    def test_from_dict_1d_length_mismatch_raises(self) -> None:
        times = np.linspace(0, 1, 5)
        x = np.ones(3)  # 3 != 5
        with pytest.raises(ValueError, match="values\\['x'\\].*3.*5"):
            Signal.from_dict(times=times, values={"x": x})

    def test_from_dict_single_trace_unsqueezed(self) -> None:
        """A 1D array (T,) should be treated as (1, T)."""
        times = np.linspace(0, 1, 5)
        x = np.ones(5)
        sig = Signal.from_dict(times=times, values={"x": x})
        assert sig.values.shape == (1, 1, 5)


class TestSignalGetitem:
    def test_getitem_variable(self) -> None:
        values = np.arange(24).reshape(2, 3, 4).astype(float)
        times = np.linspace(0, 1, 4)
        sig = Signal(values=values, times=times, labels={"a": 0, "b": 1, "c": 2})
        result = sig["b"]
        np.testing.assert_array_equal(result, values[:, 1, :])
        assert result.shape == (2, 4)

    def test_getitem_variable_and_timestep(self) -> None:
        values = np.arange(24).reshape(2, 3, 4).astype(float)
        times = np.linspace(0, 1, 4)
        sig = Signal(values=values, times=times, labels={"a": 0, "b": 1, "c": 2})
        result = sig["b", 2]
        np.testing.assert_array_equal(result, values[:, 1, 2])
        assert result.shape == (2,)

    def test_getitem_unknown_variable_raises(self) -> None:
        sig = Signal(
            values=np.zeros((1, 1, 5)),
            times=np.linspace(0, 1, 5),
            labels={"x": 0},
        )
        with pytest.raises(KeyError):
            sig["unknown"]
