from __future__ import annotations

from typing import Protocol, TypeVar

from tidystl.core.nodes import Node

T = TypeVar("T")


class Folding(Protocol[T]):
    def atom(self, node: Node) -> T: ...
    def unary(self, node: Node, child: T) -> T: ...
    def binary(self, node: Node, left: T, right: T) -> T: ...
    def nary(self, node: Node, children: tuple[T, ...]) -> T: ...


def fold_tree(node: Node, algebra: Folding[T]) -> T:
    arity = len(node.children)
    if arity == 0:
        return algebra.atom(node)
    if arity == 1:
        (child,) = node.children
        return algebra.unary(node, fold_tree(child, algebra))
    if arity == 2:
        left, right = node.children
        return algebra.binary(
            node,
            fold_tree(left, algebra),
            fold_tree(right, algebra),
        )
    return algebra.nary(node, tuple(fold_tree(child, algebra) for child in node.children))
