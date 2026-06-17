from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tidystl import Signal, evaluate, parse
from tidystl_compat import GenericBackend, GenericConfig

FORMULAS = [
    "x >= 0",
    "G[0,2](x >= 0)",
    "F[1,3](x >= 0)",
    "(x >= 0) and (y >= 0)",
    "(x >= 0) or (y >= 0)",
    "not (x >= 0)",
    "(x >= 0) U[0,2] (y >= 0)",
]


def _signal() -> Signal:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return Signal.from_dict(
        times,
        {"x": np.array([-1.0, 2.0, -3.0, 4.0, -5.0]), "y": np.array([1.0, -1.0, 1.0, -1.0, 1.0])},
    )


def test_default_generic_matches_native() -> None:
    signal = _signal()
    backend = GenericBackend(GenericConfig())
    for text in FORMULAS:
        phi = parse(text)
        got = backend.evaluate(phi, signal).robustness
        want = evaluate(phi, signal, backend="native").robustness
        np.testing.assert_allclose(got, want, atol=1e-9, err_msg=text)


def test_pessimistic_empty_window_collapses_pl_samples() -> None:
    # pl_samples: a window with no interior samples is empty. pessimistic ->
    # reducer identity (-inf for eventually); clamp -> finite midpoint fallback.
    times = np.array([0.0, 2.0, 4.0])
    signal = Signal.from_dict(times, {"x": np.array([1.0, -3.0, 5.0])})
    phi = parse("F[0.5,1.5](x >= 0)")  # at t=0 the window (0.5,1.5) holds no sample
    pess = (
        GenericBackend(GenericConfig(signal_model="pl_samples", boundary="pessimistic"))
        .evaluate(phi, signal)
        .robustness
    )
    clamp = (
        GenericBackend(GenericConfig(signal_model="pl_samples", boundary="clamp"))
        .evaluate(phi, signal)
        .robustness
    )
    assert pess[0, 0] == -np.inf
    assert np.isfinite(clamp[0, 0])


def test_pl_samples_matches_breach_on_windows_with_samples() -> None:
    # On windows that DO contain interior samples, pl_samples (samples-only,
    # no endpoint interpolation) reproduces breach exactly. (Breach's
    # interior-empty-window quirk is a documented out-of-axis exception and is
    # deliberately not exercised here.)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal.from_dict(times, {"x": np.array([-1.0, 2.0, -3.0, 4.0, -5.0])})
    for text in ["G[0.5,2.5](x >= 0)", "F[0.5,2.5](x >= 0)"]:
        phi = parse(text)
        got = (
            GenericBackend(GenericConfig(signal_model="pl_samples"))
            .evaluate(phi, signal)
            .robustness
        )
        want = evaluate(phi, signal, backend="breach").robustness
        np.testing.assert_allclose(got, want, atol=1e-9, err_msg=text)


def test_pl_samples_differs_from_pl_interp_on_subsample_endpoints() -> None:
    # Sanity: the endpoint-handling knob actually changes the result. The left
    # endpoint at t+0.5 interpolates toward the extreme out-of-window sample
    # x(0) = -100, so pl_interp (includes the endpoint) drops far below the
    # interior-only minimum that pl_samples reports.
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal.from_dict(times, {"x": np.array([-100.0, 2.0, 3.0, 4.0, 5.0])})
    phi = parse("G[0.5,2.5](x >= 0)")
    interp = (
        GenericBackend(GenericConfig(signal_model="pl_interp")).evaluate(phi, signal).robustness
    )
    samples = (
        GenericBackend(GenericConfig(signal_model="pl_samples")).evaluate(phi, signal).robustness
    )
    assert not np.allclose(interp, samples)


def test_equality_bigm_matches_breach() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal.from_dict(times, {"x": np.array([3.0, 2.0, 4.0, 3.0, 1.0])})
    phi = parse("x == 3")
    got = GenericBackend(GenericConfig(equality="bigm")).evaluate(phi, signal).robustness
    want = evaluate(phi, signal, backend="breach").robustness
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_equality_signed_is_negative_abs() -> None:
    times = np.array([0.0, 1.0])
    signal = Signal.from_dict(times, {"x": np.array([3.0, 1.0])})
    phi = parse("x == 3")
    got = GenericBackend(GenericConfig(equality="signed")).evaluate(phi, signal).robustness
    np.testing.assert_allclose(got, np.array([[0.0, -2.0]]), atol=1e-9)


BREACH_CONFIG = GenericConfig(
    signal_model="pl_samples", boundary="clamp", terminal="extend_penultimate", equality="bigm"
)


def test_terminal_extend_matches_breach() -> None:
    times = np.array([0.0, 1.0, 2.0])
    signal = Signal.from_dict(
        times, {"x": np.array([5.0, 5.0, -3.0]), "y": np.array([5.0, 5.0, -3.0])}
    )
    phi = parse("(x >= 0) and (y >= 0)")
    got = GenericBackend(BREACH_CONFIG).evaluate(phi, signal).robustness
    want = evaluate(phi, signal, backend="breach").robustness
    np.testing.assert_allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# Task 5: Discrete signal model (rtamt / stlcgpp reproduction)
# ---------------------------------------------------------------------------

RTAMT_CONFIG = GenericConfig(
    signal_model="discrete", boundary="pessimistic", until_prefix="exclusive"
)
STLCGPP_CONFIG = GenericConfig(signal_model="discrete", boundary="clamp", until_prefix="inclusive")

DISCRETE_FORMULAS = ["G[0,2](x >= 0)", "F[1,3](x >= 0)", "(x >= 0) U[0,2] (y >= 0)"]


def _uniform_signal() -> Signal:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return Signal.from_dict(
        times,
        {"x": np.array([-1.0, 2.0, -3.0, 4.0, -5.0]), "y": np.array([1.0, -1.0, 1.0, -1.0, 1.0])},
    )


@pytest.mark.parametrize("text", DISCRETE_FORMULAS)
def test_generic_discrete_matches_rtamt(text: str) -> None:
    signal = _uniform_signal()
    phi = parse(text)
    got = GenericBackend(RTAMT_CONFIG).evaluate(phi, signal).robustness
    want = evaluate(phi, signal, backend="rtamt").robustness
    np.testing.assert_allclose(got, want, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("text", DISCRETE_FORMULAS)
def test_generic_discrete_matches_stlcgpp(text: str) -> None:
    signal = _uniform_signal()
    phi = parse(text)
    got = GenericBackend(STLCGPP_CONFIG).evaluate(phi, signal).robustness
    want = evaluate(phi, signal, backend="stlcgpp").robustness
    np.testing.assert_allclose(got, want, atol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# Task 6: Euclidean predicate scoring (taliro reproduction)
# ---------------------------------------------------------------------------

TALIRO_CONFIG = GenericConfig(
    signal_model="pl_samples", boundary="pessimistic", predicate="euclidean"
)


def test_generic_euclidean_matches_taliro_unit_norm() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal.from_dict(times, {"x": np.array([-1.0, 2.0, -3.0, 4.0, -5.0])})
    phi = parse("G[0,2](x >= 0)")  # ||A|| = 1, so euclidean == signed here
    got = GenericBackend(TALIRO_CONFIG).evaluate(phi, signal).robustness
    want = evaluate(phi, signal, backend="taliro").robustness
    np.testing.assert_allclose(got, want, atol=1e-9, equal_nan=True)


def test_generic_euclidean_matches_taliro_non_unit_norm() -> None:
    # x + y >= 0: ||A|| = sqrt(2); normalization changes numbers vs signed
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal.from_dict(
        times,
        {
            "x": np.array([1.0, -1.0, 2.0, 0.0, 3.0]),
            "y": np.array([2.0, 1.0, -1.0, 4.0, -2.0]),
        },
    )
    phi = parse("G[0,2](x + y >= 0)")  # ||A|| = sqrt(2)
    got = GenericBackend(TALIRO_CONFIG).evaluate(phi, signal).robustness
    want = evaluate(phi, signal, backend="taliro").robustness
    np.testing.assert_allclose(got, want, atol=1e-9, equal_nan=True)
    # Sanity: result differs from signed (un-normalized)
    signed_got = (
        GenericBackend(
            GenericConfig(signal_model="pl_samples", boundary="pessimistic", predicate="signed")
        )
        .evaluate(phi, signal)
        .robustness
    )
    assert not np.allclose(got, signed_got)


def test_generic_euclidean_equality_raises() -> None:
    # taliro rejects == predicates; euclidean should do the same
    times = np.array([0.0, 1.0])
    signal = Signal.from_dict(times, {"x": np.array([1.0, 2.0])})
    phi = parse("x == 1")
    with pytest.raises(NotImplementedError):
        GenericBackend(TALIRO_CONFIG).evaluate(phi, signal)


def test_generic_euclidean_nonlinear_predicate_raises() -> None:
    # non-affine predicates (x*y) should raise NotImplementedError
    times = np.array([0.0, 1.0])
    signal = Signal.from_dict(times, {"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])})
    phi = parse("x * y >= 0")
    with pytest.raises(NotImplementedError):
        GenericBackend(TALIRO_CONFIG).evaluate(phi, signal)


# ---------------------------------------------------------------------------
# Task 7: ZOH signal model (pymtl reproduction)
# ---------------------------------------------------------------------------

PYMTL_CONFIG = GenericConfig(signal_model="zoh")


def test_generic_zoh_matches_pymtl() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    signal = Signal.from_dict(
        times,
        {"x": np.array([-1.0, 2.0, -3.0, 4.0, -5.0]), "y": np.array([1.0, -1.0, 1.0, -1.0, 1.0])},
    )
    for text in ["G[0,2](x >= 0)", "F[1,3](x >= 0)", "(x >= 0) and (y >= 0)"]:
        phi = parse(text)
        got = GenericBackend(PYMTL_CONFIG).evaluate(phi, signal).robustness
        want = evaluate(phi, signal, backend="pymtl").robustness
        np.testing.assert_allclose(got, want, atol=1e-9, equal_nan=True, err_msg=text)


def test_discrete_boundary_headline_rtamt_vs_stlcgpp() -> None:
    # div_boundary_F24: F[2,4] window runs past trace end. rtamt(pessimistic)
    # reports -inf (violated); stlcgpp(clamp/last-value) reports +3 (satisfied).
    times = np.array([0.0, 1.0, 2.0])
    signal = Signal.from_dict(times, {"x": np.array([-5.0, -1.0, 3.0])})
    phi = parse("F[2,4](x >= 0)")
    rt = GenericBackend(RTAMT_CONFIG).evaluate(phi, signal).robustness
    st = GenericBackend(STLCGPP_CONFIG).evaluate(phi, signal).robustness
    assert rt[0, 1] == -np.inf  # window [3,5] past end -> empty -> -inf
    assert st[0, 1] == pytest.approx(3.0)  # type: ignore[misc]  # last-value extension -> 3
    # and each matches its faithful backend
    np.testing.assert_allclose(
        rt, evaluate(phi, signal, backend="rtamt").robustness, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        st, evaluate(phi, signal, backend="stlcgpp").robustness, atol=1e-9, equal_nan=True
    )


# ---------------------------------------------------------------------------
# Task 8: Register generic backend by name; Core 6 taxonomy
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SPACE = REPO_ROOT / "extra/experiments/registry/semantics_space.json"


def test_generic_registered_by_name() -> None:
    times = np.array([0.0, 1.0, 2.0])
    signal = Signal.from_dict(times, {"x": np.array([-1.0, 1.0, -1.0])})
    phi = parse("x >= 0")
    got = evaluate(phi, signal, backend="generic").robustness
    np.testing.assert_allclose(got, np.array([[-1.0, 1.0, -1.0]]), atol=1e-9)


def test_taxonomy_configs_load() -> None:
    space = json.loads(SPACE.read_text())
    for _tool, cfg in space["tool_configs"].items():
        GenericConfig(**cfg)  # raises TypeError on unknown/missing field


def test_generic_config_rejects_invalid_option() -> None:
    with pytest.raises(ValueError):
        GenericConfig(boundary="closed")  # type: ignore[arg-type]  # not a valid option
    with pytest.raises(ValueError):
        GenericConfig(signal_model="banana")  # type: ignore[arg-type]


def test_generic_config_accepts_all_taxonomy_configs() -> None:
    space = json.loads(
        (
            Path(__file__).resolve().parents[3] / "extra/experiments/registry/semantics_space.json"
        ).read_text()
    )
    for cfg in space["tool_configs"].values():
        GenericConfig(**cfg)  # must not raise
