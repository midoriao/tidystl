import models
import numpy as np
import pytest


@pytest.mark.parametrize("name", ["m1_speed", "m2_mass_spring", "m3_coupled"])
def test_registry_has_model(name):
    assert name in models.MODELS
    m = models.MODELS[name]
    assert m.n_segments >= 1
    assert m.u_lo < m.u_hi
    assert m.t_end > 0
    assert m.dt > 0


def test_monitor_times_span_and_step():
    m = models.MODELS["m2_mass_spring"]
    t = models.monitor_times(m, m.dt)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(m.t_end)
    assert np.allclose(np.diff(t), m.dt)


def test_simulate_shape_matches_grid():
    m = models.MODELS["m1_speed"]
    u = np.full(m.n_segments, 0.5)
    t = models.monitor_times(m, m.dt)
    y = models.simulate(m, u, t)
    assert y.shape == t.shape
    assert np.all(np.isfinite(y))


def test_dense_agrees_with_coarse_at_shared_times():
    # The dense grid is a refinement of the monitor grid, so values at the
    # shared monitor sample times must agree closely.
    m = models.MODELS["m2_mass_spring"]
    u = np.linspace(m.u_lo, m.u_hi, m.n_segments)
    coarse_t = models.monitor_times(m, m.dt)
    coarse_y = models.simulate(m, u, coarse_t)
    dense_t, dense_y = models.simulate_dense(m, u, m.dt)
    # locate monitor times inside the dense grid
    idx = np.searchsorted(dense_t, coarse_t)
    assert np.allclose(dense_t[idx], coarse_t, atol=1e-9)
    assert np.allclose(dense_y[idx], coarse_y, atol=1e-3)


def test_zero_input_mass_spring_stays_at_rest():
    m = models.MODELS["m2_mass_spring"]
    u = np.zeros(m.n_segments)
    t = models.monitor_times(m, m.dt)
    y = models.simulate(m, u, t)
    assert np.allclose(y, 0.0, atol=1e-9)
