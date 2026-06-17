from tidystl.core.evaluator import BackendRegistry

from .backend import SIMDBackend


def backends() -> list[SIMDBackend]:
    return [SIMDBackend()]


def register(registry: BackendRegistry) -> None:
    for backend in backends():
        registry.register(backend)


__all__ = ["SIMDBackend", "backends", "register"]
