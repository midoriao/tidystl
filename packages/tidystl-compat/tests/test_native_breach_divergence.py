"""Native vs Breach agreement/divergence cases.

Moved out of the core native-semantics suite so the core tests stay
independent of the tidystl_compat plugin. These compare NativeBackend's exact
piecewise-linear window reductions against BreachBackend's samples-only kernel.
"""

from __future__ import annotations

import numpy as np

import tidystl
import tidystl_compat
from tidystl import Signal, parse, robustness

tidystl.use(tidystl_compat)


def test_native_and_breach_agree_when_samples_in_window() -> None:
    """G[1,3]: both backends agree when every window contains samples.

    With sample spacing 1 and window width 2, every window [t+1, t+3]
    contains at least one sample point.
    """
    sig = Signal.from_dict(
        times=np.arange(7, dtype=float),
        values={"x": np.array([[3.0, 1.0, 4.0, 1.0, 5.0, 2.0, 3.0]])},
    )
    phi = parse("G[1,3](x >= 0)")
    rho_native = robustness(phi, sig, backend="native")
    rho_breach = robustness(phi, sig, backend="breach")
    np.testing.assert_allclose(rho_native, rho_breach, atol=1e-10)


def test_fractional_endpoints_native_vs_breach() -> None:
    """G[0.5,1.5] with fractional window endpoints: documented divergence.

    Native computes the exact PL min including interpolated window endpoints;
    Breach applies a samples-only reduction (pinned by the matlab-run
    div_nonuniform_always ground truth, Breach 1.11.4, 2026-06-05).
    """
    sig = Signal.from_dict(
        times=np.arange(5, dtype=float),
        values={"x": np.array([[2.0, -1.0, 3.0, -2.0, 1.0]])},
    )
    phi = parse("G[0.5,1.5](x >= 0)")
    rho_native = robustness(phi, sig, backend="native")
    rho_breach = robustness(phi, sig, backend="breach")
    np.testing.assert_allclose(rho_native[0], [-1.0, 0.5, -2.0, -0.5, 1.0], atol=1e-10)
    np.testing.assert_allclose(rho_breach[0], [-1.0, 3.0, -2.0, 1.0, 1.0], atol=1e-10)
