from __future__ import annotations

import numpy as np
import pytest
import tidystl_simd  # noqa: F401  — registers backend as side-effect

from tidystl import Signal, parse, robustness


def _sig(times: list[float], **kwargs: list[float]) -> Signal:
    return Signal.from_dict(
        times=np.array(times, dtype=float),
        values={k: np.array([v], dtype=float) for k, v in kwargs.items()},
    )


def _compare(phi_str: str, sig: Signal, atol: float = 1e-10) -> None:
    phi = parse(phi_str)
    native = robustness(phi, sig, backend="native")
    simd = robustness(phi, sig, backend="tidystl_simd")
    np.testing.assert_allclose(simd, native, atol=atol)


class TestPredicates:
    def test_ineq_ge(self) -> None:
        _compare("x >= 0", _sig([0.0, 1.0, 2.0], x=[1.0, -1.0, 2.0]))

    def test_ineq_le(self) -> None:
        _compare("x <= 1", _sig([0.0, 1.0, 2.0], x=[0.0, 1.0, 3.0]))

    def test_ineq_gt(self) -> None:
        _compare("x > 0", _sig([0.0, 1.0, 2.0], x=[1.0, -1.0, 2.0]))

    def test_ineq_lt(self) -> None:
        _compare("x < 2", _sig([0.0, 1.0, 2.0], x=[1.0, 3.0, 2.0]))

    def test_eq(self) -> None:
        _compare("x == 1", _sig([0.0, 1.0, 2.0], x=[1.0, 0.0, 2.0]))

    def test_arith_expr(self) -> None:
        _compare("x + y >= 0", _sig([0.0, 1.0], x=[1.0, -2.0], y=[-0.5, 3.0]))

    def test_arith_const(self) -> None:
        _compare("x >= 1.5", _sig([0.0, 1.0, 2.0], x=[2.0, 1.0, 3.0]))


class TestBooleanOps:
    def test_not(self) -> None:
        _compare("not (x >= 0)", _sig([0.0, 1.0, 2.0], x=[1.0, -1.0, 2.0]))

    def test_and(self) -> None:
        _compare("(x >= 0) and (y >= 0)",
                 _sig([0.0, 1.0, 2.0], x=[1.0, -1.0, 2.0], y=[2.0, 1.0, -1.0]))

    def test_or(self) -> None:
        _compare("(x >= 0) or (y >= 0)",
                 _sig([0.0, 1.0, 2.0], x=[-1.0, -1.0, 2.0], y=[2.0, 1.0, -3.0]))


class TestTemporalOperators:
    def test_G_aligned_window(self) -> None:
        _compare("G[0,1](x >= 0)", _sig([0.0, 1.0, 2.0, 3.0, 4.0], x=[2.0, -1.0, 3.0, -2.0, 1.0]))

    def test_G_interp_sparse(self) -> None:
        # Diverges from Breach — uses true PL min with interpolated boundaries
        _compare("G[0.5,1.5](x >= 0)", _sig([0.0, 2.0, 4.0], x=[-1.0, 10.0, -1.0]))

    def test_G_interp_boundary(self) -> None:
        _compare("G[0.2,0.8](x >= 0)", _sig([0.0, 1.0, 2.0], x=[5.0, -3.0, 5.0]))

    def test_G_offset_zero(self) -> None:
        _compare("G[0,2](x >= 0)", _sig([0.0, 1.0, 2.0, 3.0], x=[1.0, 2.0, -1.0, 3.0]))

    def test_F_simple(self) -> None:
        _compare("F[0,1](x >= 0)", _sig([0.0, 1.0, 2.0], x=[2.0, -1.0, 3.0]))

    def test_F_interp(self) -> None:
        _compare("F[0.3,0.7](x >= 0)", _sig([0.0, 1.0, 2.0], x=[-1.0, 5.0, -1.0]))

    def test_U_simple(self) -> None:
        _compare(
            "(x >= 0) U[0,2] (y >= 0)",
            _sig([0.0, 1.0, 2.0, 3.0, 4.0], x=[1.0, 1.0, 1.0, -1.0, 1.0], y=[-1.0, -1.0, 2.0, -1.0, -1.0]),
        )

    def test_U_offset_start(self) -> None:
        # U[1,3]: witness window starts at t+1, not t
        _compare(
            "(x >= 0) U[1,3] (y >= 0)",
            _sig([0.0, 1.0, 2.0, 3.0, 4.0], x=[1.0, 1.0, 1.0, 1.0, 1.0], y=[-1.0, -1.0, 3.0, -1.0, -1.0]),
        )

    def test_U_positive_witness(self) -> None:
        # U[0,4]: y becomes positive at t=2, p holds before that; out[0] should be > 0
        _compare(
            "(x >= 0) U[0,4] (y >= 0)",
            _sig([0.0, 1.0, 2.0, 3.0, 4.0], x=[2.0, 2.0, 2.0, 2.0, 2.0], y=[-1.0, -1.0, 3.0, -1.0, -1.0]),
        )

    def test_nested_G_F(self) -> None:
        _compare("G[0,2](F[0,1](x >= 0))", _sig(list(range(6)), x=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]))

    def test_batched_signal(self) -> None:
        times = np.arange(5, dtype=float)
        x_vals = np.array([[2.0, -1.0, 3.0, -2.0, 1.0],
                           [0.5,  0.5, 0.5,  0.5, 0.5],
                           [-1.0, 2.0, -2.0, 3.0, -3.0]])
        sig = Signal.from_dict(times=times, values={"x": x_vals})
        phi = parse("G[0,1](x >= 0)")
        native = robustness(phi, sig, backend="native")
        simd = robustness(phi, sig, backend="tidystl_simd")
        np.testing.assert_allclose(simd, native, atol=1e-10)


class TestEdgeCases:
    def test_G_instantaneous_window(self) -> None:
        # G[0,0] is the identity: robustness equals the predicate robustness pointwise
        _compare("G[0,0](x >= 0)", _sig([0.0, 1.0, 2.0], x=[1.0, -1.0, 2.0]))

    def test_G_window_extends_past_last_sample(self) -> None:
        # At t[-2]=3.0, G[0,2] window is [3,5] but signal ends at 4.0 — right-boundary clamp
        _compare("G[0,2](x >= 0)", _sig([0.0, 1.0, 2.0, 3.0, 4.0], x=[2.0, -1.0, 3.0, 1.0, 0.5]))

    def test_F_window_extends_past_last_sample(self) -> None:
        _compare("F[0,2](x >= 0)", _sig([0.0, 1.0, 2.0, 3.0, 4.0], x=[2.0, -1.0, 3.0, 1.0, 0.5]))

    def test_single_timestep_signal(self) -> None:
        # Single sample: every temporal operator reduces to that sample's robustness
        _compare("G[0,1](x >= 0)", _sig([0.0], x=[1.5]))
        _compare("F[0,1](x >= 0)", _sig([0.0], x=[-0.5]))


def test_simd_backend_rejects_non_signal() -> None:
    from tidystl_simd import SIMDBackend

    from tidystl.core.signal import TorchSignal

    phi = parse("x >= 0")
    fake = TorchSignal(
        values=np.zeros((1, 1, 3)),
        times=np.arange(3, dtype=float),
        labels={"x": 0},
    )
    with pytest.raises(TypeError, match="tidystl_simd"):
        SIMDBackend().evaluate(phi, fake)
