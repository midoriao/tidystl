"""Breach ground-truth cases as data (name, formula, signal).

Each case mirrors one block of ``extra/other_tools/breach/generate_ground_truth.m``
and keys into ``breach_ground_truth.jsonl``. Holding the cases as data (rather than
one test method each) lets both the Breach-compat test and the generic-backend
ground-truth test (``test_generic_ground_truth.py``) iterate the same fixtures.

``tidystl_formula`` uses the ``[predicates] ... [stl] ...`` block syntax the
original per-method tests used, so the parsed AST is byte-for-byte the one that
produced the recorded ``breach`` robustness. ``atol`` defaults to ``1e-6``;
``dense_sine`` relaxes it because MATLAB's and numpy's ``linspace`` grids differ
slightly at window boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BreachCase:
    name: str
    tidystl_formula: str
    times: tuple[float, ...]
    values: dict[str, tuple[float, ...]]
    atol: float = 1e-6


def _p(*lines: str) -> str:
    return "\n".join(lines)


_PRED_P = _p("[predicates]", "p : x >= 0", "[stl]")
_PRED_PQ = _p("[predicates]", "p : x >= 0", "q : y >= 0", "[stl]")

# dense_sine: 101-point sine; computed here so the fixture stays a single source.
_DENSE_T = tuple(float(t) for t in np.linspace(0, 10, 101))
_DENSE_X = tuple(float(v) for v in np.sin(np.linspace(0, 10, 101)))


BREACH_CASES: tuple[BreachCase, ...] = (
    # --- basic operators ---
    BreachCase(
        "predicate_gte",
        _p("[predicates]", "p : x >= 3", "[stl]", "p"),
        (0, 1, 2, 3, 4),
        {"x": (5, 2, 4, 1, 6)},
    ),
    BreachCase("not_simple", _p(_PRED_P, "not p"), (0, 1, 2, 3, 4), {"x": (3, -1, 2, -4, 5)}),
    BreachCase(
        "and_two",
        _p(_PRED_PQ, "p and q"),
        (0, 1, 2, 3, 4),
        {"x": (3, -1, 2, 4, -2), "y": (1, 2, -1, 3, 5)},
    ),
    BreachCase(
        "or_two",
        _p(_PRED_PQ, "p or q"),
        (0, 1, 2, 3, 4),
        {"x": (3, -1, 2, 4, -2), "y": (1, 2, -1, 3, 5)},
    ),
    BreachCase(
        "combined_and_or",
        _p("[predicates]", "p : x >= 0", "q : y >= 0", "r : z >= 0", "[stl]", "(p and q) or r"),
        (0, 1, 2, 3, 4),
        {"x": (1, -1, 2, -2, 3), "y": (2, 3, -1, 1, -1), "z": (-1, -1, -1, 5, -1)},
    ),
    # --- always ---
    BreachCase("always_sliding", _p(_PRED_P, "G[0,2](p)"), (0, 1, 2, 3, 4), {"x": (5, 2, 8, 1, 6)}),
    BreachCase("always_boundary", _p(_PRED_P, "G[0,5](p)"), (0, 1, 2), {"x": (3, 1, 4)}),
    # --- eventually ---
    BreachCase(
        "eventually_sliding", _p(_PRED_P, "F[0,2](p)"), (0, 1, 2, 3, 4), {"x": (1, 5, 2, 8, 3)}
    ),
    BreachCase(
        "eventually_offset",
        _p(_PRED_P, "F[1,3](p)"),
        (0, 1, 2, 3, 4, 5, 6),
        {"x": (1, 5, 2, 8, 3, 1, 7)},
    ),
    # --- until ---
    BreachCase(
        "until_basic",
        _p(_PRED_PQ, "p U[0,3] q"),
        (0, 1, 2, 3, 4),
        {"x": (1, 1, 1, -1, -1), "y": (-1, -1, 2, 2, 2)},
    ),
    BreachCase(
        "until_tight",
        _p(_PRED_PQ, "p U[1,2] q"),
        (0, 1, 2, 3, 4),
        {"x": (1, 1, 1, -1, -1), "y": (-1, -1, 2, 2, 2)},
    ),
    BreachCase(
        "until_never_sat",
        _p(_PRED_PQ, "p U[0,4] q"),
        (0, 1, 2, 3, 4),
        {"x": (1, 1, 1, 1, 1), "y": (-1, -1, -1, -1, -1)},
    ),
    # --- nested temporal ---
    BreachCase(
        "nested_GF",
        _p(_PRED_P, "G[0,3](F[0,2](p))"),
        tuple(range(10)),
        {"x": (2, -1, 3, -2, 1, -3, 4, 0.5, -1, 2)},
    ),
    BreachCase(
        "nested_FG",
        _p(_PRED_P, "F[0,2](G[0,1](p))"),
        tuple(range(10)),
        {"x": (2, -1, 3, -2, 1, -3, 4, 0.5, -1, 2)},
    ),
    # --- non-uniform sampling ---
    BreachCase(
        "nonuniform_always", _p(_PRED_P, "G[0,1](p)"), (0, 0.5, 1.5, 3, 4), {"x": (0, 3, 1, 4, 2)}
    ),
    BreachCase(
        "nonuniform_until",
        _p(_PRED_PQ, "p U[0,2] q"),
        (0, 0.5, 1.5, 3, 4),
        {"x": (2, 1, -1, 1, 3), "y": (-1, -1, 2, -1, 1)},
    ),
    # --- interpolation edge cases (the documented window-endpoint quirk) ---
    BreachCase("interp_sparse", _p(_PRED_P, "G[0.5,1.5](p)"), (0, 2, 4), {"x": (-1, 10, -1)}),
    BreachCase("interp_boundary", _p(_PRED_P, "G[0.2,0.8](p)"), (0, 1, 2), {"x": (5, -3, 5)}),
    # --- dense signal ---
    BreachCase(
        "dense_sine",
        _p("[predicates]", "p : x >= -0.5", "[stl]", "G[0,2](p)"),
        _DENSE_T,
        {"x": _DENSE_X},
        atol=0.1,
    ),
    # --- until: larger / complex ---
    BreachCase(
        "until_long_transitions",
        _p(_PRED_PQ, "p U[0,4] q"),
        tuple(range(10)),
        {"x": (2, 1, 3, 1, -1, 2, 1, -2, 1, 3), "y": (-3, -2, -1, 1, 2, -1, 3, 1, -1, 2)},
    ),
    BreachCase(
        "until_offset_window",
        _p(_PRED_PQ, "p U[1,3] q"),
        tuple(range(8)),
        {"x": (3, 2, 1, 2, 3, 1, -1, 2), "y": (-1, -2, -1, 4, -1, 2, 1, -1)},
    ),
    # --- nested temporal + boolean ---
    BreachCase(
        "nested_G_and_F",
        _p(_PRED_PQ, "G[0,2](p and F[0,1](q))"),
        tuple(range(7)),
        {"x": (2, 1, 3, 0.5, 2, 1, 4), "y": (-1, 2, -1, 3, -2, 1, 2)},
    ),
    BreachCase(
        "nested_F_or_G",
        _p(_PRED_PQ, "F[0,2](p or G[0,1](q))"),
        tuple(range(8)),
        {"x": (-1, 2, -1, 3, -2, 1, -1, 4), "y": (1, -1, 2, -2, 3, -1, 1, -1)},
    ),
    # --- arithmetic predicates ---
    BreachCase(
        "arith_difference",
        _p("[predicates]", "p : x - y >= 1", "[stl]", "p"),
        (0, 1, 2, 3, 4),
        {"x": (5, 3, 1, 4, 2), "y": (1, 4, 0, 2, 3)},
    ),
    BreachCase(
        "arith_sum",
        _p("[predicates]", "p : x + y >= 0", "[stl]", "G[0,1](p)"),
        (0, 1, 2, 3, 4),
        {"x": (-3, 1, -2, 4, -1), "y": (2, -3, 1, -5, 2)},
    ),
    # --- multi-variable in temporal ---
    BreachCase(
        "multivar_until",
        _p("[predicates]", "p : x >= 0", "q : y >= 0", "r : z >= 0", "[stl]", "(p and q) U[0,2] r"),
        (0, 1, 2, 3, 4),
        {"x": (3, 2, -1, 1, 4), "y": (1, -1, 2, 3, -2), "z": (-2, -1, 1, 2, 3)},
    ),
    # --- zero crossings / all-negative / exact ---
    BreachCase(
        "zero_crossings",
        _p(_PRED_P, "F[0,2](p)"),
        tuple(range(7)),
        {"x": (-2, 1, -3, 0, 2, -1, 0.5)},
    ),
    BreachCase(
        "all_negative", _p(_PRED_P, "G[0,2](p)"), (0, 1, 2, 3, 4), {"x": (-5, -3, -1, -4, -2)}
    ),
    BreachCase(
        "exact_satisfaction",
        _p(_PRED_PQ, "p or q"),
        (0, 1, 2, 3, 4),
        {"x": (0, 1, 0, -1, 0), "y": (0, 0, 1, 0, -1)},
    ),
    # --- eventually large offset ---
    BreachCase(
        "eventually_large_offset",
        _p(_PRED_P, "F[2,5](p)"),
        tuple(range(10)),
        {"x": (-2, -1, 3, -1, 5, 2, -3, 1, 4, -1)},
    ),
    # --- deeply nested (3 levels) ---
    BreachCase(
        "deeply_nested_GFG",
        _p(_PRED_P, "G[0,1](F[0,1](G[0,1](p)))"),
        tuple(range(8)),
        {"x": (1, -0.5, 2, -1, 0.5, 3, -2, 1)},
    ),
    # --- single-point signal ---
    BreachCase("single_point", _p(_PRED_P, "p"), (0.0,), {"x": (3.0,)}),
    # --- clamp: windows past signal end ---
    BreachCase("clamp_always_past", _p(_PRED_P, "G[2,4](p)"), (0, 1, 2), {"x": (5, -1, 3)}),
    BreachCase("clamp_eventually_past", _p(_PRED_P, "F[2,4](p)"), (0, 1, 2), {"x": (-5, -1, -2)}),
    BreachCase("clamp_always_neg_tail", _p(_PRED_P, "G[0,5](p)"), (0, 1, 2), {"x": (5, 3, -2)}),
    BreachCase(
        "clamp_eventually_big_window",
        _p(_PRED_P, "F[0,8](p)"),
        (0, 1, 2, 3, 4),
        {"x": (-2, 1, -1, 3, -4)},
    ),
    BreachCase(
        "clamp_until_past",
        _p(_PRED_PQ, "p U[3,5] q"),
        (0, 1, 2),
        {"x": (1, 2, -1), "y": (-2, 3, 1)},
    ),
)
