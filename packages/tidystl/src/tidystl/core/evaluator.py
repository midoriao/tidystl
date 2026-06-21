from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

from tidystl.core.backend_interface import (
    EvaluationBackend,
    EvaluationResult,
)
from tidystl.core.nodes import Node
from tidystl.core.signal import Signal, TorchSignal

BackendRef = str | EvaluationBackend | None
UseTarget = Any


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
        """Register `backend` under `backend.name`.

        The first registered backend, or any registered with `default=True`,
        becomes the default used when `backend=None`.
        """
        self._backends[backend.name] = backend
        if default or self._default_name is None:
            self._default_name = backend.name

    def get(self, name: str) -> EvaluationBackend:
        """Return the backend registered under `name`.

        Raises `LookupError` (listing the available names) if `name` is not
        registered.
        """
        try:
            return self._backends[name]
        except KeyError:
            available = ", ".join(sorted(self._backends)) or "(none registered)"
            raise LookupError(
                f"unknown backend {name!r}; available backends: {available}"
            ) from None

    def names(self) -> list[str]:
        """Registered backend names, sorted."""
        return sorted(self._backends)

    def default(self) -> EvaluationBackend:
        if self._default_name is None:
            raise LookupError("no default backend is registered")
        return self._backends[self._default_name]


RegisterFunc = Callable[[BackendRegistry], None]
BackendsFunc = Callable[[], Iterable[Any]]

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


def list_backends(registry: BackendRegistry | None = None) -> list[str]:
    """Names of all currently registered backends, sorted.

    Uses the module-level default registry unless `registry` is given.
    After `tidystl.use(tidystl_compat)`, the returned list includes the
    compat backends (e.g. `"breach"`); before that it is just `["native"]`.
    """
    return (registry or get_default_backend_registry()).names()


def use(plugin: UseTarget, *, registry: BackendRegistry | None = None) -> None:
    """Register a backend plugin, backend instance, or iterable of backends.

    A plugin module should expose ``register(registry)``. For direct use, pass
    any backend instance or iterable of backend instances.
    """

    resolved_registry = registry or get_default_backend_registry()
    register = getattr(plugin, "register", None)
    if callable(register):
        cast(RegisterFunc, register)(resolved_registry)
        return

    backends = getattr(plugin, "backends", None)
    if callable(backends):
        _register_many(cast(BackendsFunc, backends)(), resolved_registry)
        return

    if _looks_like_backend(plugin):
        resolved_registry.register(cast(EvaluationBackend, plugin))
        return

    if isinstance(plugin, str) or not isinstance(plugin, Iterable):
        raise TypeError("tidystl.use() expects a plugin module, backend, or iterable of backends")

    _register_many(cast(Iterable[Any], plugin), resolved_registry)


def _register_many(backends: Iterable[Any], registry: BackendRegistry) -> None:
    for backend in backends:
        if not _looks_like_backend(backend):
            raise TypeError("tidystl.use() iterable items must be backend instances")
        registry.register(cast(EvaluationBackend, backend))


def _looks_like_backend(candidate: Any) -> bool:
    return isinstance(getattr(candidate, "name", None), str) and callable(
        getattr(candidate, "evaluate", None)
    )


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
