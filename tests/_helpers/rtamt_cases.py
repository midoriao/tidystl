from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RtamtCase:
    name: str
    tidystl_formula: str
    rtamt_formula: str
    times: tuple[float, ...]
    values: dict[str, tuple[float, ...]]


RTAMT_CASES: tuple[RtamtCase, ...] = (
    RtamtCase(
        name="predicate_gte",
        tidystl_formula="x >= 3",
        rtamt_formula="x >= 3",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={"x": (5.0, 2.0, 4.0, 1.0, 6.0)},
    ),
    RtamtCase(
        name="not_simple",
        tidystl_formula="not (x >= 0)",
        rtamt_formula="not (x >= 0)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={"x": (3.0, -1.0, 2.0, -4.0, 5.0)},
    ),
    RtamtCase(
        name="and_two",
        tidystl_formula="(x >= 0) and (y >= 0)",
        rtamt_formula="((x >= 0) and (y >= 0))",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={
            "x": (3.0, -1.0, 2.0, 4.0, -2.0),
            "y": (1.0, 2.0, -1.0, 3.0, 5.0),
        },
    ),
    RtamtCase(
        name="or_two",
        tidystl_formula="(x >= 0) or (y >= 0)",
        rtamt_formula="((x >= 0) or (y >= 0))",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={
            "x": (3.0, -1.0, 2.0, 4.0, -2.0),
            "y": (1.0, 2.0, -1.0, 3.0, 5.0),
        },
    ),
    RtamtCase(
        name="combined_and_or",
        tidystl_formula="((x >= 0) and (y >= 0)) or (z >= 0)",
        rtamt_formula="(((x >= 0) and (y >= 0)) or (z >= 0))",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={
            "x": (1.0, -1.0, 2.0, -2.0, 3.0),
            "y": (2.0, 3.0, -1.0, 1.0, -1.0),
            "z": (-1.0, -1.0, -1.0, 5.0, -1.0),
        },
    ),
    RtamtCase(
        name="always_sliding",
        tidystl_formula="G[0,2](x >= 0)",
        rtamt_formula="always[0:2](x >= 0)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={"x": (5.0, 2.0, 8.0, 1.0, 6.0)},
    ),
    RtamtCase(
        name="always_boundary",
        tidystl_formula="G[1,2](x >= 0)",
        rtamt_formula="always[1:2](x >= 0)",
        times=(0.0, 1.0, 2.0),
        values={"x": (3.0, 1.0, 4.0)},
    ),
    RtamtCase(
        name="eventually_sliding",
        tidystl_formula="F[0,2](x >= 0)",
        rtamt_formula="eventually[0:2](x >= 0)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={"x": (1.0, 5.0, 2.0, 8.0, 3.0)},
    ),
    RtamtCase(
        name="eventually_offset",
        tidystl_formula="F[1,3](x >= 0)",
        rtamt_formula="eventually[1:3](x >= 0)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        values={"x": (1.0, 5.0, 2.0, 8.0, 3.0, 1.0, 7.0)},
    ),
    RtamtCase(
        name="eventually_boundary",
        tidystl_formula="F[1,2](x >= 0)",
        rtamt_formula="eventually[1:2](x >= 0)",
        times=(0.0, 1.0, 2.0),
        values={"x": (3.0, 1.0, 4.0)},
    ),
    RtamtCase(
        name="until_basic",
        tidystl_formula="(x >= 0) U[0,3] (y >= 0)",
        rtamt_formula="((x >= 0) until[0:3] (y >= 0))",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={
            "x": (1.0, 1.0, 1.0, -1.0, -1.0),
            "y": (-1.0, -1.0, 2.0, 2.0, 2.0),
        },
    ),
    RtamtCase(
        name="until_tight",
        tidystl_formula="(x >= 0) U[1,2] (y >= 0)",
        rtamt_formula="((x >= 0) until[1:2] (y >= 0))",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={
            "x": (1.0, 1.0, 1.0, -1.0, -1.0),
            "y": (-1.0, -1.0, 2.0, 2.0, 2.0),
        },
    ),
    RtamtCase(
        name="until_never_sat",
        tidystl_formula="(x >= 0) U[0,4] (y >= 0)",
        rtamt_formula="((x >= 0) until[0:4] (y >= 0))",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        values={
            "x": (1.0, 1.0, 1.0, 1.0, 1.0),
            "y": (-1.0, -1.0, -1.0, -1.0, -1.0),
        },
    ),
    RtamtCase(
        name="until_boundary",
        tidystl_formula="(x >= 0) U[1,2] (y >= 0)",
        rtamt_formula="((x >= 0) until[1:2] (y >= 0))",
        times=(0.0, 1.0),
        values={
            "x": (5.0, -5.0),
            "y": (-1.0, 2.0),
        },
    ),
    RtamtCase(
        name="arith_sum",
        tidystl_formula="x + y >= 0",
        rtamt_formula="x + y >= 0",
        times=(0.0, 1.0, 2.0, 3.0),
        values={
            "x": (1.0, -2.0, 3.0, -4.0),
            "y": (2.0, 3.0, -1.0, 5.0),
        },
    ),
    RtamtCase(
        name="arith_difference",
        tidystl_formula="x - y >= 0",
        rtamt_formula="x - y >= 0",
        times=(0.0, 1.0, 2.0, 3.0),
        values={
            "x": (5.0, 2.0, 1.0, 4.0),
            "y": (3.0, 3.0, 2.0, 0.0),
        },
    ),
)
