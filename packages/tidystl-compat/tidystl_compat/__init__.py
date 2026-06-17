"""tidystl_compat: compatibility backends that reproduce the robustness
semantics of external STL tools (Breach, RTAMT, py-metric-temporal-logic,
STLCG++, TaLiRo), plus the parametrized `generic` backend that spans the
Core 6 semantics space."""

from typing import Any

from tidystl.core.evaluator import BackendRegistry
from tidystl_compat.breach import BreachBackend
from tidystl_compat.generic import GenericBackend, GenericConfig
from tidystl_compat.pymtl import PymtlBackend
from tidystl_compat.rtamt import RtamtBackend
from tidystl_compat.rtamt_dense import RtamtDenseBackend
from tidystl_compat.stlcgpp import StlcgppBackend, StlcgppTorchBackend
from tidystl_compat.taliro import TaliroBackend


def backends() -> list[Any]:
    """Instantiate all compatibility backends."""

    return [
        BreachBackend(),
        RtamtBackend(),
        RtamtDenseBackend(),
        PymtlBackend(),
        StlcgppBackend(),
        StlcgppTorchBackend(),
        TaliroBackend(),
        GenericBackend(),
    ]


def register(registry: BackendRegistry) -> None:
    """Register all compatibility backends with ``registry``."""

    for backend in backends():
        registry.register(backend)


__all__ = [
    "BreachBackend",
    "RtamtBackend",
    "RtamtDenseBackend",
    "PymtlBackend",
    "StlcgppBackend",
    "StlcgppTorchBackend",
    "TaliroBackend",
    "GenericBackend",
    "GenericConfig",
    "backends",
    "register",
]
