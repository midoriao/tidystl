"""Native (principled PL) semantics tests.

These tests document the exact behavior of NativeBackend for cases where
it diverges from BreachBackend. NativeBackend always computes the true
piecewise-linear window min/max; BreachBackend may return different values
due to Breach implementation quirks (see test_breach_compat.py).
"""

from __future__ import annotations

import numpy as np

from tidystl import Signal, parse, robustness


class TestNativeGOffsetWindow:
    """G[a,b] windows that do not align with sample points.

    NativeBackend computes the exact PL window minimum, including the
    interpolated boundary values at t+a and t+b. BreachBackend mimics
    Breach's RobustEv kernel: samples-only reduction when the window
    contains samples; rho(t+a) at the first sample and rho(t) at
    subsequent samples when it does not.
    """

    def test_interp_sparse_native(self) -> None:
        """G[0.5,1.5](x >= 0): native gives true PL min for each window.

        Signal: t=[0,2,4], x=[-1,10,-1]. The window [t+0.5, t+1.5] contains
        no sample points at any timestep.

        Native PL min:
          t=0: min over [0.5, 1.5] = rho(0.5) = 1.75  (increasing segment)
          t=2: min over [2.5, 3.5] = rho(3.5) = 1.75  (decreasing segment)
          t=4: min over [4.5, 5.5] = rho(4)   = -1    (clamped)

        BreachBackend gives [1.75, 10, -1] — diverges at t=2.
        """
        sig = Signal.from_dict(
            times=np.array([0, 2, 4], dtype=float),
            values={"x": np.array([[-1, 10, -1]], dtype=float)},
        )
        phi = parse("G[0.5,1.5](x >= 0)")
        rho = robustness(phi, sig, backend="native")
        np.testing.assert_allclose(rho[0], [1.75, 1.75, -1.0], atol=1e-10)

    def test_interp_boundary_native(self) -> None:
        """G[0.2,0.8](x >= 0): native gives true PL min for each window.

        Signal: t=[0,1,2], x=[5,-3,5]. The window [t+0.2, t+0.8] contains
        no sample points at any timestep.

        Native PL min:
          t=0: min over [0.2, 0.8] = rho(0.8) = -1.4  (decreasing segment)
          t=1: min over [1.2, 1.8] = rho(1.2) = -1.4  (increasing segment)
          t=2: min over [2.2, 2.8] = rho(2)   =  5    (clamped)

        BreachBackend gives [3.4, -3, 5] — diverges at t=0 and t=1.
        """
        sig = Signal.from_dict(
            times=np.array([0, 1, 2], dtype=float),
            values={"x": np.array([[5, -3, 5]], dtype=float)},
        )
        phi = parse("G[0.2,0.8](x >= 0)")
        rho = robustness(phi, sig, backend="native")
        np.testing.assert_allclose(rho[0], [-1.4, -1.4, 5.0], atol=1e-10)

    def test_native_and_breach_agree_when_samples_in_window(self) -> None:
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

    def test_fractional_endpoints_native_vs_breach(self) -> None:
        """G[0.5,1.5] with fractional window endpoints: documented divergence.

        Signal: t=[0..4], x=[2,-1,3,-2,1]. Every interior window [t+0.5,
        t+1.5] contains exactly one sample, with both window endpoints
        falling between samples.

        Native (exact PL min, including interpolated endpoints):
          t=0: min(rho(0.5)=0.5, -1, rho(1.5)=1)   = -1
          t=1: min(rho(1.5)=1, 3, rho(2.5)=0.5)    = 0.5
          t=2: min(rho(2.5)=0.5, -2, rho(3.5)=-0.5) = -2
          t=3: min(rho(3.5)=-0.5, 1, clamped)      = -0.5
          t=4: window [4.5,5.5] past end, clamped  = 1

        Breach (samples-only reduction; window-endpoint interpolation is
        NOT applied -- pinned by the matlab-run div_nonuniform_always
        ground truth, Breach 1.11.4, 2026-06-05):
          t=0: min over samples {t=1} = -1
          t=1: min over samples {t=2} = 3
          t=2: min over samples {t=3} = -2
          t=3: min over samples {t=4} = 1
          t=4: window past end, clamped = 1
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
