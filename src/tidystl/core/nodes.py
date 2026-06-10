from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

ComparisonOp: TypeAlias = Literal[">=", ">", "<=", "<", "=="]

ArithAtomKind: TypeAlias = Literal["const", "var"]
ArithUnaryKind: TypeAlias = Literal["abs", "sqrt"]
ArithBinaryKind: TypeAlias = Literal["+", "-", "*", "/", "^"]

STLAtomKind: TypeAlias = Literal["predicate"]
STLUnaryKind: TypeAlias = Literal["not", "always", "eventually"]
STLBinaryKind: TypeAlias = Literal["and", "or", "until"]

ArithKind: TypeAlias = ArithAtomKind | ArithUnaryKind | ArithBinaryKind
STLKind: TypeAlias = STLAtomKind | STLUnaryKind | STLBinaryKind
NodeKind: TypeAlias = ArithKind | STLKind


def _empty_children() -> tuple[Node, ...]:
    return ()


def _empty_attrs() -> dict[str, object]:
    return {}


@dataclass(kw_only=True)
class Node:
    """Formula AST node, as produced by `parse()`.

    `kind` selects the operator (see the Kind type aliases above);
    `children` holds the operand subtrees; operator-specific data
    (predicate name/op, interval bounds, ...) lives in `attrs`.
    Arithmetic and STL nodes share this one type: `ArithNode` is an
    alias kept for call-site clarity.
    """

    kind: NodeKind
    children: tuple[Node, ...] = field(default_factory=_empty_children)
    attrs: dict[str, object] = field(default_factory=_empty_attrs)


ArithNode: TypeAlias = Node
