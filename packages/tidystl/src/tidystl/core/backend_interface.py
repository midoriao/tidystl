"""Interfaces at the backend boundary.

`EvaluationResult` is the single Protocol that every backend's evaluate()
must satisfy. It exposes `robustness` plus the trace API as methods;
backends that do not expose traces set `has_trace = False` and raise
`NotImplementedError` from `trace_for`/`traced_nodes`.

Backend-internal helpers (DAG node types, etc.) live in
`tidystl.backends.helper`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from tidystl.core.nodes import ArithNode, Node
from tidystl.core.signal import Signal, TorchSignal


class EvaluationResult(Protocol):
    robustness: Any
    has_trace: bool

    def trace_for(self, node: Node | ArithNode) -> Any:
        """Sub-formula robustness trace for `node`.

        Backends without trace support raise `NotImplementedError`. Backends
        with trace support raise `KeyError` for nodes that are not part of
        the evaluated formula.
        """
        ...

    def traced_nodes(self) -> Iterable[Node | ArithNode]:
        """All nodes whose trace is observable via `trace_for`.

        Backends without trace support raise `NotImplementedError`.
        """
        ...


class EvaluationBackend(Protocol):
    name: str

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> EvaluationResult:
        """Evaluate `formula` over `signal`.

        Backends accept the signal flavor they are built for and raise
        `TypeError` for the other (numpy backends require `Signal`;
        torch backends require `TorchSignal`).
        """
        ...
