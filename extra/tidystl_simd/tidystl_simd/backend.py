from __future__ import annotations

from collections.abc import Iterable
from typing import NoReturn

import numpy as np
from numpy.typing import NDArray

from tidystl import Signal
from tidystl.core.nodes import Node
from tidystl.core.signal import TorchSignal

from . import _ext


class _SIMDResult:
    has_trace = False

    def __init__(self, robustness: NDArray[np.floating]) -> None:
        self.robustness = robustness

    def trace_for(self, node: object) -> NoReturn:
        raise NotImplementedError("SIMDBackend does not support traces; check has_trace first")

    def traced_nodes(self) -> Iterable[object]:
        return []


class _SIMDEvaluator:
    def __init__(self, signal: Signal) -> None:
        self._signal = signal

    def evaluate(self, formula: Node) -> _SIMDResult:
        rho = self._eval(formula)
        return _SIMDResult(robustness=rho)

    def _eval(self, node: Node) -> NDArray[np.floating]:
        match node.kind:
            case "predicate":
                left_node = node.attrs["left"]
                right_node = node.attrs["right"]
                if not isinstance(left_node, Node) or not isinstance(right_node, Node):
                    raise TypeError("predicate node requires Node left/right attrs")
                left = self._eval_arith(left_node)
                right = self._eval_arith(right_node)
                op = node.attrs["op"]
                if not isinstance(op, str):
                    raise TypeError("predicate node requires str op attr")
                match op:
                    case ">=" | ">":
                        return left - right
                    case "<=" | "<":
                        return right - left
                    case "==":
                        return -np.abs(left - right)
                    case _:
                        raise NotImplementedError(f"unknown predicate op {op!r}")
            case "not":
                return -self._eval(node.children[0])
            case "and":
                return np.minimum(self._eval(node.children[0]), self._eval(node.children[1]))
            case "or":
                return np.maximum(self._eval(node.children[0]), self._eval(node.children[1]))
            case "always":
                start, end = self._interval(node)
                child = self._eval(node.children[0])
                return _ext.eval_globally(self._signal.times, child, start, end)
            case "eventually":
                start, end = self._interval(node)
                child = self._eval(node.children[0])
                return _ext.eval_finally(self._signal.times, child, start, end)
            case "until":
                start, end = self._interval(node)
                p = self._eval(node.children[0])
                q = self._eval(node.children[1])
                return _ext.eval_until(self._signal.times, p, q, start, end)
            case _:
                raise NotImplementedError(f"unknown STL node kind {node.kind!r}")

    def _eval_arith(self, node: Node) -> NDArray[np.floating]:
        match node.kind:
            case "const":
                n = self._signal.values.shape[0]
                t = len(self._signal.times)
                return np.full((n, t), float(node.attrs["value"]))
            case "var":
                name = node.attrs["name"]
                if not isinstance(name, str):
                    raise TypeError("var node requires str name attr")
                return self._signal[name]  # (N, T)
            case "+" | "-" | "*" | "/" | "^":
                left = self._eval_arith(node.children[0])
                right = self._eval_arith(node.children[1])
                match node.kind:
                    case "+": return left + right
                    case "-": return left - right
                    case "*": return left * right
                    case "/": return left / right
                    case "^": return np.power(left, right)
                    case _: raise AssertionError("unreachable")
            case "abs":
                return np.abs(self._eval_arith(node.children[0]))
            case "sqrt":
                return np.sqrt(self._eval_arith(node.children[0]))
            case _:
                raise NotImplementedError(f"unknown arith node kind {node.kind!r}")

    @staticmethod
    def _interval(node: Node) -> tuple[float, float]:
        iv = node.attrs.get("interval")
        if not isinstance(iv, tuple) or len(iv) != 2:
            raise ValueError(f"temporal node {node.kind!r} missing interval attr")
        return float(iv[0]), float(iv[1])


class SIMDBackend:
    """SIMD-accelerated STL backend (Rust/PyO3).

    Note: SIMD min/max reductions follow Intel MINPD/MAXPD semantics and do not propagate NaN.
    Use the native backend if NaN propagation in robustness traces is required.
    """

    name = "tidystl_simd"

    def evaluate(self, formula: Node, signal: Signal | TorchSignal) -> _SIMDResult:
        if not isinstance(signal, Signal):
            raise TypeError(
                f"{self.name} backend requires a Signal, got {type(signal).__name__}"
            )
        return _SIMDEvaluator(signal).evaluate(formula)
