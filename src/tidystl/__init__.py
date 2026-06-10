"""tidystl: STL robustness computation -- numpy-native, batch-first, dense-time PL semantics."""

from tidystl.backends.breach import BreachBackend
from tidystl.backends.native import NativeBackend
from tidystl.backends.rtamt import RtamtBackend
from tidystl.backends.stlcgpp import StlcgppBackend, StlcgppTorchBackend
from tidystl.core.backend_interface import (
    EvaluationResult,
)
from tidystl.core.evaluator import (
    BackendRegistry,
    evaluate,
    get_default_backend_registry,
    robustness,
)
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal
from tidystl.frontend.analysis import horizon, required_max_gap
from tidystl.frontend.parser import parse

get_default_backend_registry().register(NativeBackend(), default=True)
get_default_backend_registry().register(BreachBackend())
get_default_backend_registry().register(RtamtBackend())
get_default_backend_registry().register(StlcgppBackend())
get_default_backend_registry().register(StlcgppTorchBackend())

__all__ = [
    "Node",
    "Signal",
    "TorchSignal",
    "NativeBackend",
    "BreachBackend",
    "RtamtBackend",
    "StlcgppBackend",
    "StlcgppTorchBackend",
    "robustness",
    "evaluate",
    "EvaluationResult",
    "BackendRegistry",
    "parse",
    "horizon",
    "required_max_gap",
]
