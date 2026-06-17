"""Compatibility re-exports for evaluator APIs moved to ``tidystl.core``."""

from __future__ import annotations

from tidystl.core.evaluator import (
    BackendRegistry,
    evaluate,
    get_default_backend_registry,
    resolve_backend,
    robustness,
    use,
)

__all__ = [
    "BackendRegistry",
    "evaluate",
    "get_default_backend_registry",
    "resolve_backend",
    "robustness",
    "use",
]
