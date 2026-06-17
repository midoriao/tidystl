from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PymtlCase:
    name: str
    tidystl_formula: str
    pymtl_formula: str
    times: tuple[float, ...]
    atoms: dict[str, tuple[float, ...]]


# Pointwise cases (Task 3). Temporal cases are appended in Tasks 4 and 5.
PYMTL_CASES: tuple[PymtlCase, ...] = (
    PymtlCase(
        name="predicate",
        tidystl_formula="x >= 0",
        pymtl_formula="x",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (5.0, -2.0, 4.0, -1.0, 6.0)},
    ),
    PymtlCase(
        name="not_simple",
        tidystl_formula="not (x >= 0)",
        pymtl_formula="~x",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (3.0, -1.0, 2.0, -4.0, 5.0)},
    ),
    PymtlCase(
        name="and_two",
        tidystl_formula="(x >= 0) and (y >= 0)",
        pymtl_formula="(x & y)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (3.0, -1.0, 2.0, 4.0, -2.0), "y": (1.0, 2.0, -1.0, 3.0, 5.0)},
    ),
    PymtlCase(
        name="or_two",
        tidystl_formula="(x >= 0) or (y >= 0)",
        pymtl_formula="(x | y)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (3.0, -1.0, 2.0, 4.0, -2.0), "y": (1.0, 2.0, -1.0, 3.0, 5.0)},
    ),
    PymtlCase(
        name="eventually_sliding",
        tidystl_formula="F[0,2](x >= 0)",
        pymtl_formula="F[0,2] x",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (1.0, -2.0, 3.0, -1.0, 0.5)},
    ),
    PymtlCase(
        name="eventually_offset",
        tidystl_formula="F[1,3](x >= 0)",
        pymtl_formula="F[1,3] x",
        times=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        atoms={"x": (1.0, 5.0, 2.0, 8.0, 3.0, 1.0, 7.0)},
    ),
    PymtlCase(
        name="always_sliding",
        tidystl_formula="G[0,2](x >= 0)",
        pymtl_formula="G[0,2] x",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (5.0, 2.0, 8.0, 1.0, 6.0)},
    ),
    PymtlCase(
        name="always_offset",
        tidystl_formula="G[1,3](x >= 0)",
        pymtl_formula="G[1,3] x",
        times=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        atoms={"x": (5.0, 2.0, 8.0, 1.0, 6.0, 4.0, 9.0)},
    ),
    PymtlCase(
        name="until_basic",
        tidystl_formula="(x >= 0) U[0,3] (y >= 0)",
        pymtl_formula="(x U[0,3] y)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (1.0, 1.0, 1.0, -1.0, -1.0), "y": (-1.0, -1.0, 2.0, 2.0, 2.0)},
    ),
    PymtlCase(
        name="until_tight",
        tidystl_formula="(x >= 0) U[1,2] (y >= 0)",
        pymtl_formula="(x U[1,2] y)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (1.0, 1.0, 1.0, -1.0, -1.0), "y": (-1.0, -1.0, 2.0, 2.0, 2.0)},
    ),
    PymtlCase(
        name="until_never_sat",
        tidystl_formula="(x >= 0) U[0,4] (y >= 0)",
        pymtl_formula="(x U[0,4] y)",
        times=(0.0, 1.0, 2.0, 3.0, 4.0),
        atoms={"x": (1.0, 1.0, 1.0, 1.0, 1.0), "y": (-1.0, -1.0, -1.0, -1.0, -1.0)},
    ),
    PymtlCase(
        name="eventually_non_uniform",
        tidystl_formula="F[0,2](x >= 0)",
        pymtl_formula="F[0,2] x",
        times=(0.0, 0.5, 2.5, 3.0),
        atoms={"x": (1.0, -2.0, 3.0, -1.0)},
    ),
    PymtlCase(
        name="always_non_uniform",
        tidystl_formula="G[0,2](x >= 0)",
        pymtl_formula="G[0,2] x",
        times=(0.0, 0.5, 2.5, 3.0),
        atoms={"x": (1.0, -2.0, 3.0, -1.0)},
    ),
    PymtlCase(
        name="until_non_uniform",
        tidystl_formula="(x >= 0) U[0,2] (y >= 0)",
        pymtl_formula="(x U[0,2] y)",
        times=(0.0, 0.5, 1.5, 2.5, 3.0),
        atoms={"x": (1.0, 1.0, 1.0, -1.0, -1.0), "y": (-1.0, -1.0, 2.0, 2.0, 2.0)},
    ),
)
