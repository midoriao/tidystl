import models
import numpy as np
import oracle
import specs

from tidystl import parse


def test_violation_detected_when_signal_breaches_safety():
    # hero spec G[0.0,8.0](v <= 4.5): full throttle pushes v ~= 4.94 above 4.5 -> violation.
    m = models.MODELS["m1_speed"]
    phi = parse(specs.BENCHMARKS["m1_speed"].spec())
    u = np.full(m.n_segments, m.u_hi)
    assert oracle.is_true_violation(m, u, phi, m.dt) is True


def test_no_violation_when_signal_stays_safe():
    m = models.MODELS["m1_speed"]
    phi = parse(specs.BENCHMARKS["m1_speed"].spec())
    u = np.full(m.n_segments, m.u_lo)  # zero throttle -> v stays at 0
    assert oracle.is_true_violation(m, u, phi, m.dt) is False
