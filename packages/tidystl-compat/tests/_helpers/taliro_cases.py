from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaliroCase:
    name: str
    tidystl_formula: str
    times: tuple[float, ...]
    values: dict[str, tuple[float, ...]]


_T5 = (0.0, 1.0, 2.0, 3.0, 4.0)
_T3 = (0.0, 1.0, 2.0)

TALIRO_CASES: tuple[TaliroCase, ...] = (
    TaliroCase("predicate_gte", "x >= 3", _T5, {"x": (5.0, 2.0, 4.0, 1.0, 6.0)}),
    TaliroCase("not_simple", "not (x >= 0)", _T5, {"x": (3.0, -1.0, 2.0, -4.0, 5.0)}),
    TaliroCase(
        "and_two",
        "(x >= 0) and (y >= 0)",
        _T5,
        {"x": (3.0, -1.0, 2.0, 4.0, -2.0), "y": (1.0, 2.0, -1.0, 3.0, 5.0)},
    ),
    TaliroCase(
        "or_two",
        "(x >= 0) or (y >= 0)",
        _T5,
        {"x": (3.0, -1.0, 2.0, 4.0, -2.0), "y": (1.0, 2.0, -1.0, 3.0, 5.0)},
    ),
    TaliroCase("always_untimed", "G[0,4](x >= 0)", _T5, {"x": (5.0, 2.0, 8.0, 1.0, 6.0)}),
    TaliroCase("always_sliding", "G[0,2](x >= 0)", _T5, {"x": (5.0, 2.0, 8.0, 1.0, 6.0)}),
    TaliroCase("always_boundary", "G[0,5](x >= 0)", _T3, {"x": (3.0, 1.0, 4.0)}),
    TaliroCase("eventually_untimed", "F[0,4](x >= 0)", _T5, {"x": (1.0, 5.0, 2.0, 8.0, 3.0)}),
    TaliroCase("eventually_sliding", "F[0,2](x >= 0)", _T5, {"x": (1.0, 5.0, 2.0, 8.0, 3.0)}),
    TaliroCase(
        "until_basic",
        "(x >= 0) U[0,3] (y >= 0)",
        _T5,
        {"x": (1.0, 1.0, 1.0, -1.0, -1.0), "y": (-1.0, -1.0, 2.0, 2.0, 2.0)},
    ),
    TaliroCase(
        "until_untimed",
        "(x >= 0) U[0,4] (y >= 0)",
        _T5,
        {"x": (1.0, 1.0, 1.0, -1.0, -1.0), "y": (-1.0, -1.0, 2.0, 2.0, 2.0)},
    ),
    TaliroCase(
        "nested_GF",
        "G[0,3](F[0,2](x >= 0))",
        tuple(float(i) for i in range(10)),
        {"x": (2.0, -1.0, 3.0, -2.0, 1.0, -3.0, 4.0, 0.5, -1.0, 2.0)},
    ),
    TaliroCase(
        "nested_FG",
        "F[0,2](G[0,1](x >= 0))",
        tuple(float(i) for i in range(10)),
        {"x": (2.0, -1.0, 3.0, -2.0, 1.0, -3.0, 4.0, 0.5, -1.0, 2.0)},
    ),
    TaliroCase(
        "nonuniform_always",
        "G[0,1](x >= 0)",
        (0.0, 0.5, 1.5, 3.0, 4.0),
        {"x": (0.0, 3.0, 1.0, 4.0, 2.0)},
    ),
    TaliroCase("norm_sum", "x + y >= 0", (0.0, 1.0), {"x": (3.0, 3.0), "y": (0.0, 0.0)}),
    TaliroCase("norm_diff", "x - y >= 1", (0.0, 1.0), {"x": (5.0, 5.0), "y": (1.0, 1.0)}),
    # ---- edge cases ----
    TaliroCase("always_empty_window", "G[0.2,0.8](x >= 0)", _T3, {"x": (5.0, -3.0, 5.0)}),
    TaliroCase("eventually_empty_window", "F[0.2,0.8](x >= 0)", _T3, {"x": (5.0, -3.0, 5.0)}),
    TaliroCase("always_past_end", "G[2,4](x >= 0)", _T3, {"x": (5.0, -1.0, 3.0)}),
    TaliroCase("eventually_past_end", "F[2,4](x >= 0)", _T3, {"x": (-5.0, -1.0, -2.0)}),
    TaliroCase("always_big_window", "G[0,5](x >= 0)", _T3, {"x": (5.0, 3.0, -2.0)}),
    TaliroCase(
        "eventually_big_window",
        "F[0,8](x >= 0)",
        _T5,
        {"x": (-2.0, 1.0, -1.0, 3.0, -4.0)},
    ),
    TaliroCase("always_offset", "G[1,3](x >= 0)", _T5, {"x": (1.0, 5.0, 2.0, 8.0, 3.0)}),
    TaliroCase("eventually_offset", "F[1,3](x >= 0)", _T5, {"x": (1.0, 5.0, 2.0, 8.0, 3.0)}),
    TaliroCase("interp_probe_ev", "F[0,1](x >= 0)", (0.0, 2.0), {"x": (0.0, 10.0)}),
    TaliroCase("interp_probe_alw", "G[0,1](x >= 0)", (0.0, 2.0), {"x": (10.0, 0.0)}),
    TaliroCase(
        "until_past_end",
        "(x >= 0) U[3,5] (y >= 0)",
        _T3,
        {"x": (1.0, 2.0, -1.0), "y": (-2.0, 3.0, 1.0)},
    ),
    TaliroCase(
        "until_never_sat",
        "(x >= 0) U[0,4] (y >= 0)",
        _T5,
        {"x": (1.0, 1.0, 1.0, 1.0, 1.0), "y": (-1.0, -1.0, -1.0, -1.0, -1.0)},
    ),
    TaliroCase(
        "until_tight",
        "(x >= 0) U[1,2] (y >= 0)",
        _T5,
        {"x": (1.0, 1.0, 1.0, -1.0, -1.0), "y": (-1.0, -1.0, 2.0, 2.0, 2.0)},
    ),
)
