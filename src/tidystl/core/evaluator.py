from __future__ import annotations

from typing import Any

from tidystl.core.backend_interface import (
    EvaluationBackend,
    EvaluationResult,
)
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal

BackendRef = str | EvaluationBackend | None


class BackendRegistry:
    """Name-to-backend mapping used to resolve `backend=` arguments.

    The first registered backend (or the one registered with
    `default=True`) becomes the default used when `backend` is None.
    Importing `tidystl` populates a module-level default registry with
    all built-in backends; pass a custom registry to `evaluate()` /
    `robustness()` to override it.
    """

    def __init__(self) -> None:
        self._backends: dict[str, EvaluationBackend] = {}
        self._default_name: str | None = None

    def register(self, backend: EvaluationBackend, *, default: bool = False) -> None:
        self._backends[backend.name] = backend
        if default or self._default_name is None:
            self._default_name = backend.name

    def get(self, name: str) -> EvaluationBackend:
        try:
            return self._backends[name]
        except KeyError:
            available = ", ".join(sorted(self._backends)) or "(none registered)"
            raise LookupError(
                f"unknown backend {name!r}; available backends: {available}"
            ) from None

    def default(self) -> EvaluationBackend:
        if self._default_name is None:
            raise LookupError("no default backend is registered")
        return self._backends[self._default_name]


_DEFAULT_BACKEND_REGISTRY = BackendRegistry()


def get_default_backend_registry() -> BackendRegistry:
    return _DEFAULT_BACKEND_REGISTRY


def resolve_backend(
    backend: BackendRef,
    *,
    registry: BackendRegistry | None = None,
) -> EvaluationBackend:
    resolved_registry = registry or get_default_backend_registry()
    if backend is None:
        return resolved_registry.default()
    if isinstance(backend, str):
        return resolved_registry.get(backend)
    return backend


def evaluate(
    formula: Node,
    signal: Signal | TorchSignal,
    backend: BackendRef = None,
    *,
    registry: BackendRegistry | None = None,
) -> EvaluationResult:
    """Evaluate `formula` over `signal` and return the full backend result.

    `backend` is a registered name (e.g. "native", "breach"), a backend
    instance, or None for the registry default. The result exposes
    `robustness` plus the sub-formula trace API (`trace_for`,
    `traced_nodes`) for backends with trace support.
    """
    selected_backend = resolve_backend(backend, registry=registry)
    return selected_backend.evaluate(formula, signal)


def robustness(
    formula: Node,
    signal: Signal | TorchSignal,
    backend: BackendRef = None,
    *,
    registry: BackendRegistry | None = None,
) -> Any:
    """Robustness of `formula` over `signal`, shape (N, T).

    Shorthand for `evaluate(...).robustness`. Returns a numpy array for
    `Signal` inputs and a torch tensor for `TorchSignal` inputs (hence
    the `Any` return type).
    """
    return evaluate(formula, signal, backend=backend, registry=registry).robustness
