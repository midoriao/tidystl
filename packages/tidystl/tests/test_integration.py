import numpy as np

from tidystl import Signal, parse, robustness


class TestEndToEnd:
    def test_always_positive(self) -> None:
        """G[0,5](x >= 0) on a signal that is always positive."""
        t = np.linspace(0, 10, 100)
        x = np.ones((1, 100)) * 3.0
        sig = Signal.from_dict(times=t, values={"x": x})

        phi = parse("""
        G[0,5](x >= 0)
        """)

        rho = robustness(phi, sig)
        assert rho.shape == (1, 100)
        np.testing.assert_allclose(rho[0, 0], 3.0)

    def test_eventually_found(self) -> None:
        """F[0,5](x >= 10) on a ramp signal."""
        t = np.linspace(0, 10, 101)
        x = t[np.newaxis, :]
        sig = Signal.from_dict(times=t, values={"x": x})

        phi = parse("""
        F[0,5](x >= 10)
        """)

        rho = robustness(phi, sig)
        assert rho.shape == (1, 101)
        assert rho[0, 50] >= 0  # t=5.0

    def test_nested_always_eventually(self) -> None:
        """G[0,5](F[0,2](x >= -1.5)) on sin(t): always satisfied since sin >= -1."""
        t = np.linspace(0, 10, 200)
        x = np.sin(t)[np.newaxis, :]
        sig = Signal.from_dict(times=t, values={"x": x})

        phi = parse("""
        G[0,5](F[0,2](x >= -1.5))
        """)

        rho = robustness(phi, sig)
        assert rho.shape == (1, 200)
        assert rho[0, 0] > 0

    def test_batch_consistency(self) -> None:
        """Batch of N traces gives same results as N individual evaluations."""
        t = np.linspace(0, 5, 50)
        x_batch = np.random.default_rng(42).standard_normal((10, 50))
        sig_batch = Signal.from_dict(times=t, values={"x": x_batch})

        phi = parse("""
        G[0,2](x >= 0)
        """)

        rho_batch = robustness(phi, sig_batch)

        for i in range(10):
            sig_single = Signal.from_dict(times=t, values={"x": x_batch[i : i + 1, :]})
            rho_single = robustness(phi, sig_single)
            np.testing.assert_allclose(rho_batch[i : i + 1, :], rho_single, atol=1e-10)
