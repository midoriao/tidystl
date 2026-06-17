from __future__ import annotations

import math

import numpy as np
import pytest

from tidystl import Signal, robustness
from tidystl.core.nodes import Node
from tidystl.frontend.parser import parse
from tidystl_compat.taliro import (
    affine_coeffs,
    predicate_norm,
    taliro_eval_until,
    taliro_sliding_reduce,
)


def _t(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=float)


def test_sliding_min_samples_only_no_interpolation() -> None:
    # G[0,1] on x=[10,0] at t=[0,2]: window [0,1] holds only t=0 -> 10 (not interp 5)
    times = _t(0.0, 2.0)
    rho = np.array([[10.0, 0.0]])
    out = taliro_sliding_reduce(times, rho, 0.0, 1.0, np.min, np.inf)
    assert out[0, 0] == 10.0


def test_sliding_max_samples_only_no_interpolation() -> None:
    # F[0,1] on x=[0,10] at t=[0,2]: window [0,1] holds only t=0 -> 0 (not interp 5)
    times = _t(0.0, 2.0)
    rho = np.array([[0.0, 10.0]])
    out = taliro_sliding_reduce(times, rho, 0.0, 1.0, np.max, -np.inf)
    assert out[0, 0] == 0.0


def test_sliding_empty_window_returns_identity() -> None:
    # G[0.2,0.8] on t=[0,1,2]: no sample in window at t=0 -> +inf
    times = _t(0.0, 1.0, 2.0)
    rho = np.array([[5.0, -3.0, 5.0]])
    out_min = taliro_sliding_reduce(times, rho, 0.2, 0.8, np.min, np.inf)
    out_max = taliro_sliding_reduce(times, rho, 0.2, 0.8, np.max, -np.inf)
    assert out_min[0, 0] == np.inf
    assert out_max[0, 0] == -np.inf


def test_sliding_past_end_no_extension() -> None:
    # G[2,4] on x=[5,-1,3] t=[0,1,2]: only in-window sample is t=2 (x=3) -> 3
    times = _t(0.0, 1.0, 2.0)
    rho = np.array([[5.0, -1.0, 3.0]])
    out = taliro_sliding_reduce(times, rho, 2.0, 4.0, np.min, np.inf)
    assert out[0, 0] == 3.0


def test_sliding_offset_window_closed_endpoints() -> None:
    # G[1,3] on x=[1,5,2,8,3] t=0..4: window [1,3] -> samples t=1,2,3 -> min 2
    times = _t(0, 1, 2, 3, 4)
    rho = np.array([[1.0, 5.0, 2.0, 8.0, 3.0]])
    out = taliro_sliding_reduce(times, rho, 1.0, 3.0, np.min, np.inf)
    assert out[0, 0] == 2.0


def test_until_basic() -> None:
    # (x>0) U[0,3] (y>0): x=[1,1,1,-1,-1], y=[-1,-1,2,2,2] -> 1
    times = _t(0, 1, 2, 3, 4)
    p = np.array([[1.0, 1.0, 1.0, -1.0, -1.0]])
    q = np.array([[-1.0, -1.0, 2.0, 2.0, 2.0]])
    out = taliro_eval_until(times, p, q, 0.0, 3.0)
    assert out[0, 0] == 1.0


def test_until_empty_window_returns_neg_inf() -> None:
    # U[3,5] on a 3-sample trace t=[0,1,2]: no witness in [3,5] at t=0 -> -inf
    times = _t(0.0, 1.0, 2.0)
    p = np.array([[1.0, 2.0, -1.0]])
    q = np.array([[-2.0, 3.0, 1.0]])
    out = taliro_eval_until(times, p, q, 3.0, 5.0)
    assert out[0, 0] == -np.inf


def test_until_never_satisfied() -> None:
    # q always negative -> robustness is the negative q value (-1)
    times = _t(0, 1, 2, 3, 4)
    p = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    q = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0]])
    out = taliro_eval_until(times, p, q, 0.0, 4.0)
    assert out[0, 0] == -1.0


def test_until_offset_start_excludes_current_step() -> None:
    # (x>0) U[1,2] (y>0): x=[1,1,1,-1,-1], y=[-1,-1,2,2,2] -> witnesses j in [1,2] -> 1
    times = _t(0, 1, 2, 3, 4)
    p = np.array([[1.0, 1.0, 1.0, -1.0, -1.0]])
    q = np.array([[-1.0, -1.0, 2.0, 2.0, 2.0]])
    out = taliro_eval_until(times, p, q, 1.0, 2.0)
    assert out[0, 0] == 1.0


def test_sliding_reduce_batched_rows() -> None:
    # two independent traces reduced together over G[0,2]
    times = _t(0, 1, 2, 3, 4)
    rho = np.array([[5.0, 2.0, 8.0, 1.0, 6.0], [3.0, 9.0, 4.0, 7.0, 0.0]])
    out = taliro_sliding_reduce(times, rho, 0.0, 2.0, np.min, np.inf)
    assert out[0, 0] == 2.0  # min(5,2,8)
    assert out[1, 0] == 3.0  # min(3,9,4)


def _pred_parts(formula: str) -> tuple[Node, Node]:
    node = parse(formula)
    assert node.kind == "predicate"
    left, right = node.attrs["left"], node.attrs["right"]
    assert isinstance(left, Node) and isinstance(right, Node)
    return left, right


def test_affine_single_var() -> None:
    left, _ = _pred_parts("x >= 3")
    const, coeffs = affine_coeffs(left)
    assert coeffs == {"x": 1.0}
    assert const == 0.0


def test_affine_scaled_var() -> None:
    left, _ = _pred_parts("2 * x >= 6")
    _, coeffs = affine_coeffs(left)
    assert coeffs == {"x": 2.0}


def test_affine_unary_minus() -> None:
    left, _ = _pred_parts("-x >= 0")
    _, coeffs = affine_coeffs(left)
    assert coeffs == {"x": -1.0}


def test_predicate_norm_single_var_is_one() -> None:
    left, right = _pred_parts("x >= 3")
    assert predicate_norm(left, right) == 1.0


def test_predicate_norm_sum_is_sqrt2() -> None:
    left, right = _pred_parts("x + y >= 0")
    assert math.isclose(predicate_norm(left, right), math.sqrt(2.0))


def test_predicate_norm_difference_is_sqrt2() -> None:
    left, right = _pred_parts("x - y >= 1")
    assert math.isclose(predicate_norm(left, right), math.sqrt(2.0))


def test_predicate_norm_scaled_is_two() -> None:
    left, right = _pred_parts("2 * x >= 6")
    assert math.isclose(predicate_norm(left, right), 2.0)


def test_nonlinear_predicate_raises() -> None:
    left, right = _pred_parts("x * y >= 0")
    with pytest.raises(NotImplementedError):
        predicate_norm(left, right)


def test_var_free_predicate_raises() -> None:
    left, right = _pred_parts("3 >= 1")
    with pytest.raises(NotImplementedError):
        predicate_norm(left, right)


def _sig(values: dict[str, list[float]], times: list[float]) -> Signal:
    return Signal.from_dict(
        times=np.array(times, dtype=float),
        values={k: np.array([v], dtype=float) for k, v in values.items()},
    )


def test_backend_single_var_predicate() -> None:
    sig = _sig({"x": [5, 2, 4, 1, 6]}, [0, 1, 2, 3, 4])
    rho = robustness(parse("x >= 3"), sig, "taliro")
    assert rho[0, 0] == 2.0


def test_backend_normalized_multivar_predicate() -> None:
    sig = _sig({"x": [3, 3], "y": [0, 0]}, [0, 1])
    rho = robustness(parse("x + y >= 0"), sig, "taliro")
    assert math.isclose(rho[0, 0], 3.0 / math.sqrt(2.0))  # (x+y-0)/||A|| = 3/sqrt(2)


def test_backend_timed_always_samples_only() -> None:
    sig = _sig({"x": [5, 2, 8, 1, 6]}, [0, 1, 2, 3, 4])
    rho = robustness(parse("G[0,2](x >= 0)"), sig, "taliro")
    assert rho[0, 0] == 2.0


def test_backend_le_predicate() -> None:
    # le direction: (rhs - lhs)/||A|| = (3 - 5) = -2 at t=0
    sig = _sig({"x": [5, 2]}, [0, 1])
    rho = robustness(parse("x <= 3"), sig, "taliro")
    assert rho[0, 0] == -2.0


def test_backend_equality_predicate_raises() -> None:
    sig = _sig({"x": [1, 1]}, [0, 1])
    with pytest.raises(NotImplementedError):
        robustness(parse("x == 3"), sig, "taliro")
