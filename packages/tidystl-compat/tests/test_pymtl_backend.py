from __future__ import annotations

import numpy as np
import pytest

from tidystl import Signal, parse, robustness
from tidystl.core.evaluator import resolve_backend
from tidystl.frontend.evaluator import evaluate
from tidystl_compat.pymtl import PymtlBackend, pc_weak_until_grid, pc_window_reduce


def _signal(values: dict[str, list[float]], times: list[float]) -> Signal:
    return Signal.from_dict(
        times=np.array(times, dtype=float),
        values={name: np.array([trace], dtype=float) for name, trace in values.items()},
    )


def test_pymtl_backend_resolves_by_name() -> None:
    backend = resolve_backend("pymtl")
    assert backend.name == "pymtl"
    assert isinstance(backend, PymtlBackend)


def test_pymtl_backend_default_dt() -> None:
    assert PymtlBackend().dt == 0.1
    assert PymtlBackend(dt=0.05).dt == 0.05


def test_pymtl_backend_rejects_torch_signal() -> None:
    class _FakeTorchSignal:
        pass

    with pytest.raises(TypeError, match="pymtl"):
        PymtlBackend().evaluate(parse("x >= 0"), _FakeTorchSignal())  # type: ignore[arg-type]


def test_pymtl_predicate_robustness() -> None:
    sig = _signal({"x": [5.0, 2.0, 4.0, 1.0, 6.0]}, [0, 1, 2, 3, 4])
    rho = robustness(parse("x >= 3"), sig, backend="pymtl")
    np.testing.assert_allclose(rho[0], [2.0, -1.0, 1.0, -2.0, 3.0])


def test_pymtl_not_and_or() -> None:
    sig = _signal(
        {"x": [3.0, -1.0, 2.0, 4.0, -2.0], "y": [1.0, 2.0, -1.0, 3.0, 5.0]}, [0, 1, 2, 3, 4]
    )
    np.testing.assert_allclose(
        robustness(parse("not (x >= 0)"), sig, backend="pymtl")[0],
        [-3.0, 1.0, -2.0, -4.0, 2.0],
    )
    np.testing.assert_allclose(
        robustness(parse("(x >= 0) and (y >= 0)"), sig, backend="pymtl")[0],
        [1.0, -1.0, -1.0, 3.0, -2.0],
    )
    np.testing.assert_allclose(
        robustness(parse("(x >= 0) or (y >= 0)"), sig, backend="pymtl")[0],
        [3.0, 2.0, 2.0, 4.0, 5.0],
    )


def test_pymtl_result_has_trace() -> None:
    sig = _signal({"x": [5.0, 2.0]}, [0, 1])
    formula = parse("x >= 3")
    result = evaluate(formula, sig, backend="pymtl")
    assert result.has_trace
    np.testing.assert_allclose(result.trace_for(formula)[0], [2.0, -1.0])


def test_pc_window_reduce_eventually_half_open() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rho = np.array([[1.0, -2.0, 3.0, -1.0, 0.5]])
    # F[0,2]: right-half-open [t, t+2) on the dt grid, last-grid clamp.
    out = pc_window_reduce(times, rho, lo=0.0, hi=2.0, dt=0.1, reducer=np.max)
    np.testing.assert_allclose(out[0], [1.0, 3.0, 3.0, 0.5, 0.5], atol=1e-9)


def test_pc_window_reduce_eventually_offset_truncates() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    rho = np.array([[1.0, -2.0, 3.0, -1.0, 0.5]])
    out = pc_window_reduce(times, rho, lo=1.0, hi=3.0, dt=0.1, reducer=np.max)
    # Domain truncates at t = end - lo = 3; t=4 is NaN (undefined).
    np.testing.assert_allclose(out[0, :4], [3.0, 3.0, 0.5, 0.5], atol=1e-9)
    assert np.isnan(out[0, 4])


def test_pc_weak_until_grid_matches_recurrence() -> None:
    # Reproduces apply_weak_until on a single grid (N=1).
    rho_l = np.array([[1.0, 1.0, -1.0, -1.0]])
    rho_r = np.array([[-1.0, 2.0, 2.0, -3.0]])
    out = pc_weak_until_grid(rho_l, rho_r)
    # Backward: prev,maxr = inf,-inf
    # k=3: maxr=-3; prev=max(-3,min(-1,inf),3)=3
    # k=2: maxr=2;  prev=max(2,min(-1,3),-2)=2
    # k=1: maxr=2;  prev=max(2,min(1,2),-2)=2
    # k=0: maxr=2;  prev=max(-1,min(1,2),-2)=1
    np.testing.assert_allclose(out[0], [1.0, 2.0, 2.0, 3.0], atol=1e-9)


def test_pymtl_non_uniform_grid_eventually() -> None:
    # Non-uniform times; ZOH semantics. F[0,2](x>=0).
    # Validated against real py-mtl (mtl.parse("F[0,2] x"), dt=0.1):
    # returns 1.0 at t=0 and -2.0 at t=0.5.
    times = [0.0, 0.5, 2.5, 3.0]
    sig = _signal({"x": [1.0, -2.0, 3.0, -1.0]}, times)
    rho = robustness(parse("F[0,2](x >= 0)"), sig, backend="pymtl")
    # At t=0: window [0,2) covers x=1 on [0,0.5), x=-2 on [0.5,2) -> max 1.
    # At t=0.5: window [0.5,2.5) -> x=-2 on [0.5,2.5) (3 enters only at 2.5) -> -2.
    np.testing.assert_allclose(rho[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(rho[0, 1], -2.0, atol=1e-6)


def test_pymtl_batch_independent_rows() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = {"x": np.array([[1.0, -2.0, 3.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0, 0.0]])}
    sig = Signal.from_dict(times=times, values=values)
    rho = robustness(parse("F[0,2](x >= 0)"), sig, backend="pymtl")
    assert rho.shape == (2, 5)
    np.testing.assert_allclose(rho[0], [1.0, 3.0, 3.0, 0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(rho[1], [0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_pymtl_custom_dt_does_not_break() -> None:
    # Smoke test: a non-default dt evaluates without error and, for this
    # grid-aligned case, reproduces the same result as the default dt.
    sig = _signal({"x": [1.0, -2.0, 3.0]}, [0.0, 1.0, 2.0])
    rho = robustness(parse("F[0,1](x >= 0)"), sig, backend=PymtlBackend(dt=0.05))
    np.testing.assert_allclose(rho[0, 0], 1.0, atol=1e-6)
