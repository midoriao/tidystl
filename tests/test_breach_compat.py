"""Breach compatibility tests.

Each test defines the same signal and formula used in
`extra/other_tools/breach/generate_ground_truth.m`, then compares tidystl output against the
Breach CSV ground truth.

Run `extra/other_tools/breach/generate_ground_truth.m` in MATLAB first to generate the CSV files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._helpers.breach_compat import assert_breach_compatible
from tidystl import Signal, parse

GROUND_TRUTH = Path(__file__).parent / "breach_ground_truth"


def _needs(name: str):
    """Skip if Breach ground-truth CSV hasn't been generated yet."""
    return pytest.mark.skipif(
        not (GROUND_TRUTH / f"{name}.csv").exists(),
        reason=f"Run extra/other_tools/breach/generate_ground_truth.m first ({name}.csv missing)",
    )


# ---------------------------------------------------------------------------
# Basic operators
# ---------------------------------------------------------------------------


class TestBreachBasicOps:
    @_needs("predicate_gte")
    def test_predicate_gte(self) -> None:
        """x >= 3  (robustness = x - 3)."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={"x": np.array([[5, 2, 4, 1, 6]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 3
            [stl]
            p
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "predicate_gte.csv")

    @_needs("not_simple")
    def test_not_simple(self) -> None:
        """not (x >= 0)  (robustness = -x)."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={"x": np.array([[3, -1, 2, -4, 5]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            not p
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "not_simple.csv")

    @_needs("and_two")
    def test_and_two(self) -> None:
        """(x >= 0) and (y >= 0)  (robustness = min(x, y))."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[3, -1, 2, 4, -2]], dtype=float),
                "y": np.array([[1, 2, -1, 3, 5]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p and q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "and_two.csv")

    @_needs("or_two")
    def test_or_two(self) -> None:
        """(x >= 0) or (y >= 0)  (robustness = max(x, y))."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[3, -1, 2, 4, -2]], dtype=float),
                "y": np.array([[1, 2, -1, 3, 5]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p or q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "or_two.csv")

    @_needs("combined_and_or")
    def test_combined_and_or(self) -> None:
        """((x >= 0) and (y >= 0)) or (z >= 0)."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[1, -1, 2, -2, 3]], dtype=float),
                "y": np.array([[2, 3, -1, 1, -1]], dtype=float),
                "z": np.array([[-1, -1, -1, 5, -1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            r : z >= 0
            [stl]
            (p and q) or r
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "combined_and_or.csv")


# ---------------------------------------------------------------------------
# Always (G)
# ---------------------------------------------------------------------------


class TestBreachAlways:
    @_needs("always_sliding")
    def test_always_sliding(self) -> None:
        """G[0,2](x >= 0) sliding min."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={"x": np.array([[5, 2, 8, 1, 6]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0,2](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "always_sliding.csv")

    @_needs("always_boundary")
    def test_always_boundary(self) -> None:
        """G[0,5](x >= 0) on short signal — tests out-of-bounds clamping."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2], dtype=float),
            values={"x": np.array([[3, 1, 4]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0,5](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "always_boundary.csv")


# ---------------------------------------------------------------------------
# Eventually (F)
# ---------------------------------------------------------------------------


class TestBreachEventually:
    @_needs("eventually_sliding")
    def test_eventually_sliding(self) -> None:
        """F[0,2](x >= 0) sliding max."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={"x": np.array([[1, 5, 2, 8, 3]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            F[0,2](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "eventually_sliding.csv")

    @_needs("eventually_offset")
    def test_eventually_offset(self) -> None:
        """F[1,3](x >= 0) — offset window."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4, 5, 6], dtype=float),
            values={"x": np.array([[1, 5, 2, 8, 3, 1, 7]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            F[1,3](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "eventually_offset.csv")


# ---------------------------------------------------------------------------
# Until
# ---------------------------------------------------------------------------


class TestBreachUntil:
    @_needs("until_basic")
    def test_until_basic(self) -> None:
        """(x >= 0) U[0,3] (y >= 0)."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[1, 1, 1, -1, -1]], dtype=float),
                "y": np.array([[-1, -1, 2, 2, 2]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p U[0,3] q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "until_basic.csv")

    @_needs("until_tight")
    def test_until_tight(self) -> None:
        """(x >= 0) U[1,2] (y >= 0) — tight interval."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[1, 1, 1, -1, -1]], dtype=float),
                "y": np.array([[-1, -1, 2, 2, 2]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p U[1,2] q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "until_tight.csv")

    @_needs("until_never_sat")
    def test_until_never_sat(self) -> None:
        """(x >= 0) U[0,4] (y >= 0) where y is always negative."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[1, 1, 1, 1, 1]], dtype=float),
                "y": np.array([[-1, -1, -1, -1, -1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p U[0,4] q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "until_never_sat.csv")


# ---------------------------------------------------------------------------
# Nested temporal
# ---------------------------------------------------------------------------


class TestBreachNested:
    _times = np.arange(10, dtype=float)
    _x = np.array([[2, -1, 3, -2, 1, -3, 4, 0.5, -1, 2]], dtype=float)

    @_needs("nested_GF")
    def test_nested_gf(self) -> None:
        """G[0,3](F[0,2](x >= 0))."""
        sig = Signal.from_dict(times=self._times, values={"x": self._x})
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0,3](F[0,2](p))
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "nested_GF.csv")

    @_needs("nested_FG")
    def test_nested_fg(self) -> None:
        """F[0,2](G[0,1](x >= 0))."""
        sig = Signal.from_dict(times=self._times, values={"x": self._x})
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            F[0,2](G[0,1](p))
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "nested_FG.csv")


# ---------------------------------------------------------------------------
# Non-uniform time sampling
# ---------------------------------------------------------------------------


class TestBreachNonUniform:
    _times = np.array([0, 0.5, 1.5, 3, 4], dtype=float)

    @_needs("nonuniform_always")
    def test_nonuniform_always(self) -> None:
        """G[0,1](x >= 0) on non-uniform times."""
        sig = Signal.from_dict(
            times=self._times,
            values={"x": np.array([[0, 3, 1, 4, 2]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0,1](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "nonuniform_always.csv")

    @_needs("nonuniform_until")
    def test_nonuniform_until(self) -> None:
        """(x >= 0) U[0,2] (y >= 0) on non-uniform times."""
        sig = Signal.from_dict(
            times=self._times,
            values={
                "x": np.array([[2, 1, -1, 1, 3]], dtype=float),
                "y": np.array([[-1, -1, 2, -1, 1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p U[0,2] q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "nonuniform_until.csv")


# ---------------------------------------------------------------------------
# Interpolation edge cases
# ---------------------------------------------------------------------------


class TestBreachInterpolation:
    @_needs("interp_sparse")
    def test_interp_sparse(self) -> None:
        """G[0.5,1.5](x >= 0) on sparse samples.

        Samples at t = [0, 2, 4]. The window [t+0.5, t+1.5] falls entirely
        between sample points for every timestep. Breach uses rho(t+a) at the
        first sample and rho(t) at subsequent samples when no samples fall in
        the window (NativeBackend computes the true PL window minimum instead).
        """
        sig = Signal.from_dict(
            times=np.array([0, 2, 4], dtype=float),
            values={"x": np.array([[-1, 10, -1]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0.5,1.5](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "interp_sparse.csv")

    @_needs("interp_boundary")
    def test_interp_boundary(self) -> None:
        """G[0.2,0.8](x >= 0) — window boundaries between adjacent samples.

        Samples at t = [0, 1, 2], x = [5, -3, 5]. The window [t+0.2, t+0.8]
        contains no sample points for any timestep. Breach uses rho(t+a) at
        the first sample and rho(t) at subsequent samples (NativeBackend uses
        the true PL window minimum).
        """
        sig = Signal.from_dict(
            times=np.array([0, 1, 2], dtype=float),
            values={"x": np.array([[5, -3, 5]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0.2,0.8](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "interp_boundary.csv")


# ---------------------------------------------------------------------------
# Dense signal
# ---------------------------------------------------------------------------


class TestBreachDense:
    @_needs("dense_sine")
    def test_dense_sine(self) -> None:
        """G[0,2](x >= -0.5) on sin(t) with 101 samples.

        Uses a larger tolerance because MATLAB's linspace and numpy's
        linspace produce subtly different floating-point time grids,
        causing window boundaries to include/exclude boundary samples
        differently.
        """
        t = np.linspace(0, 10, 101)
        sig = Signal.from_dict(
            times=t,
            values={"x": np.sin(t)[np.newaxis, :]},
        )
        phi = parse("""
            [predicates]
            p : x >= -0.5
            [stl]
            G[0,2](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "dense_sine.csv", atol=0.1)


# ---------------------------------------------------------------------------
# Until: larger signals / complex transitions
# ---------------------------------------------------------------------------


class TestBreachUntilExtended:
    @_needs("until_long_transitions")
    def test_until_long_transitions(self) -> None:
        """(x >= 0) U[0,4] (y >= 0) on 10-point signal with many sign changes."""
        sig = Signal.from_dict(
            times=np.arange(10, dtype=float),
            values={
                "x": np.array([[2, 1, 3, 1, -1, 2, 1, -2, 1, 3]], dtype=float),
                "y": np.array([[-3, -2, -1, 1, 2, -1, 3, 1, -1, 2]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p U[0,4] q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "until_long_transitions.csv")

    @_needs("until_offset_window")
    def test_until_offset_window(self) -> None:
        """(x >= 0) U[1,3] (y >= 0) on 8-point signal."""
        sig = Signal.from_dict(
            times=np.arange(8, dtype=float),
            values={
                "x": np.array([[3, 2, 1, 2, 3, 1, -1, 2]], dtype=float),
                "y": np.array([[-1, -2, -1, 4, -1, 2, 1, -1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p U[1,3] q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "until_offset_window.csv")


# ---------------------------------------------------------------------------
# Nested temporal + boolean
# ---------------------------------------------------------------------------


class TestBreachNestedBoolean:
    @_needs("nested_G_and_F")
    def test_nested_g_and_f(self) -> None:
        """G[0,2](p and F[0,1](q)) — temporal wrapping boolean + temporal."""
        sig = Signal.from_dict(
            times=np.arange(7, dtype=float),
            values={
                "x": np.array([[2, 1, 3, 0.5, 2, 1, 4]], dtype=float),
                "y": np.array([[-1, 2, -1, 3, -2, 1, 2]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            G[0,2](p and F[0,1](q))
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "nested_G_and_F.csv")

    @_needs("nested_F_or_G")
    def test_nested_f_or_g(self) -> None:
        """F[0,2](p or G[0,1](q)) — eventually wrapping or + always."""
        sig = Signal.from_dict(
            times=np.arange(8, dtype=float),
            values={
                "x": np.array([[-1, 2, -1, 3, -2, 1, -1, 4]], dtype=float),
                "y": np.array([[1, -1, 2, -2, 3, -1, 1, -1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            F[0,2](p or G[0,1](q))
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "nested_F_or_G.csv")


# ---------------------------------------------------------------------------
# Arithmetic predicates
# ---------------------------------------------------------------------------


class TestBreachArithmetic:
    @_needs("arith_difference")
    def test_arith_difference(self) -> None:
        """x - y >= 1  (includes exact-zero robustness at t=2)."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[5, 3, 1, 4, 2]], dtype=float),
                "y": np.array([[1, 4, 0, 2, 3]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x - y >= 1
            [stl]
            p
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "arith_difference.csv")

    @_needs("arith_sum")
    def test_arith_sum(self) -> None:
        """G[0,1](x + y >= 0) — arithmetic addition inside temporal."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[-3, 1, -2, 4, -1]], dtype=float),
                "y": np.array([[2, -3, 1, -5, 2]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x + y >= 0
            [stl]
            G[0,1](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "arith_sum.csv")


# ---------------------------------------------------------------------------
# Multi-variable in temporal
# ---------------------------------------------------------------------------


class TestBreachMultiVar:
    @_needs("multivar_until")
    def test_multivar_until(self) -> None:
        """(p and q) U[0,2] r — three signals, boolean+until interaction."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[3, 2, -1, 1, 4]], dtype=float),
                "y": np.array([[1, -1, 2, 3, -2]], dtype=float),
                "z": np.array([[-2, -1, 1, 2, 3]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            r : z >= 0
            [stl]
            (p and q) U[0,2] r
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "multivar_until.csv")


# ---------------------------------------------------------------------------
# Zero crossings / all-negative / exact satisfaction
# ---------------------------------------------------------------------------


class TestBreachZeroCrossing:
    @_needs("zero_crossings")
    def test_zero_crossings(self) -> None:
        """F[0,2](x >= 0) on signal with multiple zero crossings."""
        sig = Signal.from_dict(
            times=np.arange(7, dtype=float),
            values={"x": np.array([[-2, 1, -3, 0, 2, -1, 0.5]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            F[0,2](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "zero_crossings.csv")

    @_needs("all_negative")
    def test_all_negative(self) -> None:
        """G[0,2](x >= 0) on entirely negative signal."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={"x": np.array([[-5, -3, -1, -4, -2]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0,2](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "all_negative.csv")

    @_needs("exact_satisfaction")
    def test_exact_satisfaction(self) -> None:
        """(x >= 0) or (y >= 0) with many zero-robustness values."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={
                "x": np.array([[0, 1, 0, -1, 0]], dtype=float),
                "y": np.array([[0, 0, 1, 0, -1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p or q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "exact_satisfaction.csv")


# ---------------------------------------------------------------------------
# Eventually with large offset
# ---------------------------------------------------------------------------


class TestBreachEventuallyExtended:
    @_needs("eventually_large_offset")
    def test_eventually_large_offset(self) -> None:
        """F[2,5](x >= 0) on 10-point signal."""
        sig = Signal.from_dict(
            times=np.arange(10, dtype=float),
            values={
                "x": np.array([[-2, -1, 3, -1, 5, 2, -3, 1, 4, -1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            F[2,5](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "eventually_large_offset.csv")


# ---------------------------------------------------------------------------
# Deeply nested (3 levels)
# ---------------------------------------------------------------------------


class TestBreachDeeplyNested:
    @_needs("deeply_nested_GFG")
    def test_deeply_nested_gfg(self) -> None:
        """G[0,1](F[0,1](G[0,1](x >= 0))) — three-level temporal nesting."""
        sig = Signal.from_dict(
            times=np.arange(8, dtype=float),
            values={
                "x": np.array([[1, -0.5, 2, -1, 0.5, 3, -2, 1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0,1](F[0,1](G[0,1](p)))
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "deeply_nested_GFG.csv")


# ---------------------------------------------------------------------------
# Single-point signal
# ---------------------------------------------------------------------------


class TestBreachEdgeCases:
    @_needs("single_point")
    def test_single_point(self) -> None:
        """Single-point signal — degenerate trace."""
        sig = Signal.from_dict(
            times=np.array([0.0]),
            values={"x": np.array([[3.0]])},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            p
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "single_point.csv")


# ---------------------------------------------------------------------------
# out_of_bounds="clamp" — windows extending past signal boundaries
# ---------------------------------------------------------------------------


class TestBreachClamp:
    @_needs("clamp_always_past")
    def test_clamp_always_past(self) -> None:
        """G[2,4](x >= 0) — window entirely past signal end.

        Signal ends at t=2 with x=3. The window [t+2, t+4] is past the end
        for all timesteps; clamped extension should give 3.
        """
        sig = Signal.from_dict(
            times=np.array([0, 1, 2], dtype=float),
            values={"x": np.array([[5, -1, 3]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[2,4](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "clamp_always_past.csv")

    @_needs("clamp_eventually_past")
    def test_clamp_eventually_past(self) -> None:
        """F[2,4](x >= 0) — window entirely past signal end, last value negative.

        Signal ends at t=2 with x=-2. Clamped extension yields -2
        everywhere in the extended region.
        """
        sig = Signal.from_dict(
            times=np.array([0, 1, 2], dtype=float),
            values={"x": np.array([[-5, -1, -2]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            F[2,4](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "clamp_eventually_past.csv")

    @_needs("clamp_always_neg_tail")
    def test_clamp_always_neg_tail(self) -> None:
        """G[0,5](x >= 0) — negative last value clamped across extended window.

        Signal [5, 3, -2] at t=[0,1,2]. The clamped -2 is the global minimum,
        dominating the result at every timestep.
        """
        sig = Signal.from_dict(
            times=np.array([0, 1, 2], dtype=float),
            values={"x": np.array([[5, 3, -2]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            G[0,5](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "clamp_always_neg_tail.csv")

    @_needs("clamp_eventually_big_window")
    def test_clamp_eventually_big_window(self) -> None:
        """F[0,8](x >= 0) — big window far past signal end."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2, 3, 4], dtype=float),
            values={"x": np.array([[-2, 1, -1, 3, -4]], dtype=float)},
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            [stl]
            F[0,8](p)
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "clamp_eventually_big_window.csv")

    @_needs("clamp_until_past")
    def test_clamp_until_past(self) -> None:
        """(x >= 0) U[3,5] (y >= 0) — until window entirely past signal end."""
        sig = Signal.from_dict(
            times=np.array([0, 1, 2], dtype=float),
            values={
                "x": np.array([[1, 2, -1]], dtype=float),
                "y": np.array([[-2, 3, 1]], dtype=float),
            },
        )
        phi = parse("""
            [predicates]
            p : x >= 0
            q : y >= 0
            [stl]
            p U[3,5] q
        """)
        assert_breach_compatible(phi, sig, GROUND_TRUTH / "clamp_until_past.csv")
