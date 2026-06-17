"""tidystl: STL robustness computation -- numpy-native, batch-first, dense-time PL semantics."""

from tidystl.backends.native import NativeBackend
from tidystl.core.backend_interface import (
    EvaluationResult,
)
from tidystl.core.evaluator import (
    BackendRegistry,
    evaluate,
    get_default_backend_registry,
    list_backends,
    robustness,
    use,
)
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal
from tidystl.diagnostics import DivergentNode, localize, localize_results
from tidystl.frontend.analysis import horizon, required_max_gap
from tidystl.frontend.parser import parse

get_default_backend_registry().register(NativeBackend(), default=True)

__all__ = [
    "Node",
    "Signal",
    "TorchSignal",
    "NativeBackend",
    "list_backends",
    "use",
    "robustness",
    "evaluate",
    "EvaluationResult",
    "BackendRegistry",
    "get_default_backend_registry",
    "parse",
    "horizon",
    "required_max_gap",
    "localize",
    "localize_results",
    "DivergentNode",
]
