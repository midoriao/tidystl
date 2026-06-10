from __future__ import annotations

from dataclasses import dataclass

from tests._helpers.rtamt_cases import RTAMT_CASES


@dataclass(frozen=True)
class StlcgppCase:
    name: str
    tidystl_formula: str
    times: tuple[float, ...]
    values: dict[str, tuple[float, ...]]


STLCGPP_CASES: tuple[StlcgppCase, ...] = tuple(
    StlcgppCase(
        name=case.name,
        tidystl_formula=case.tidystl_formula,
        times=case.times,
        values=case.values,
    )
    for case in RTAMT_CASES
)
