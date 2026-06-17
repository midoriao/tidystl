"""Lark-based parser for tidystl specifications."""

from __future__ import annotations

from lark import Lark, Token, Transformer
from lark import v_args as _v_args  # pyright: ignore[reportUnknownVariableType]
from lark.exceptions import VisitError

from tidystl.core.nodes import (
    ArithNode,
    ComparisonOp,
    Node,
)

SECTIONED_GRAMMAR = r"""
    start: pred_section stl_section

    pred_section: _PRED_HEADER pred_def*
    pred_def: NAME ":" arith_expr COMPARE_OP arith_expr

    stl_section: _STL_HEADER stl_expr

    _PRED_HEADER: "[predicates]"
    _STL_HEADER: "[stl]"

    ?stl_expr: stl_or

    ?stl_or: stl_and
        | stl_or _OR stl_and        -> stl_or

    ?stl_and: stl_until
        | stl_and _AND stl_until    -> stl_and

    ?stl_until: stl_unary
        | stl_unary _UNTIL interval stl_unary  -> stl_until

    ?stl_unary: stl_atom
        | _NOT stl_unary            -> stl_not
        | _ALWAYS interval "(" stl_expr ")"   -> stl_always
        | _EVENTUALLY interval "(" stl_expr ")"  -> stl_eventually

    ?stl_atom: NAME                 -> stl_pred_ref
        | "(" stl_expr ")"

    interval: "[" NUMBER "," NUMBER "]"

    ?arith_expr: arith_sum

    ?arith_sum: arith_prod
        | arith_sum "+" arith_prod  -> arith_add
        | arith_sum "-" arith_prod  -> arith_sub

    ?arith_prod: arith_pow
        | arith_prod "*" arith_pow  -> arith_mul
        | arith_prod "/" arith_pow  -> arith_div

    ?arith_pow: arith_atom
        | arith_atom "^" arith_pow  -> arith_pow_op

    ?arith_atom: NUMBER             -> arith_number
        | "-" arith_atom            -> arith_neg
        | NAME                      -> arith_var
        | _ABS "(" arith_expr ")"   -> arith_abs
        | _SQRT "(" arith_expr ")"  -> arith_sqrt
        | "(" arith_expr ")"

    // Keywords as explicit terminals (underscore prefix = filtered out)
    _AND: "and"
    _OR: "or"
    _NOT: "not"
    _ALWAYS: "G"
    _EVENTUALLY: "F"
    _UNTIL: "U"
    _ABS: "abs"
    _SQRT: "sqrt"

    COMPARE_OP: ">=" | "<=" | ">" | "<" | "=="

    NAME: /(?!(?:and|or|not|abs|sqrt)\b)[a-zA-Z_][a-zA-Z0-9_]*/

    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

INLINE_GRAMMAR = r"""
    start: stl_expr

    ?stl_expr: stl_or

    ?stl_or: stl_and
        | stl_or _OR stl_and        -> stl_or

    ?stl_and: stl_until
        | stl_and _AND stl_until    -> stl_and

    ?stl_until: stl_unary
        | stl_unary _UNTIL interval stl_unary  -> stl_until

    ?stl_unary: stl_atom
        | _NOT stl_unary            -> stl_not
        | _ALWAYS interval "(" stl_expr ")"   -> stl_always
        | _EVENTUALLY interval "(" stl_expr ")"  -> stl_eventually

    ?stl_atom: arith_expr COMPARE_OP arith_expr -> inline_predicate
        | "(" stl_expr ")"

    interval: "[" NUMBER "," NUMBER "]"

    ?arith_expr: arith_sum

    ?arith_sum: arith_prod
        | arith_sum "+" arith_prod  -> arith_add
        | arith_sum "-" arith_prod  -> arith_sub

    ?arith_prod: arith_pow
        | arith_prod "*" arith_pow  -> arith_mul
        | arith_prod "/" arith_pow  -> arith_div

    ?arith_pow: arith_atom
        | arith_atom "^" arith_pow  -> arith_pow_op

    ?arith_atom: NUMBER             -> arith_number
        | "-" arith_atom            -> arith_neg
        | NAME                      -> arith_var
        | _ABS "(" arith_expr ")"   -> arith_abs
        | _SQRT "(" arith_expr ")"  -> arith_sqrt
        | "(" arith_expr ")"

    _AND: "and"
    _OR: "or"
    _NOT: "not"
    _ALWAYS: "G"
    _EVENTUALLY: "F"
    _UNTIL: "U"
    _ABS: "abs"
    _SQRT: "sqrt"

    COMPARE_OP: ">=" | "<=" | ">" | "<" | "=="

    NAME: /(?!(?:and|or|not|abs|sqrt)\b)[a-zA-Z_][a-zA-Z0-9_]*/

    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

_sectioned_parser = Lark(SECTIONED_GRAMMAR, parser="earley", ambiguity="resolve")
_inline_parser = Lark(INLINE_GRAMMAR, parser="earley", ambiguity="resolve")


_COMPARISON_OP: dict[str, ComparisonOp] = {
    ">=": ">=",
    ">": ">",
    "<=": "<=",
    "<": "<",
    "==": "==",
}


@_v_args(inline=True)  # pyright: ignore[reportUntypedClassDecorator]
class _SpecTransformer(Transformer[Token, Node]):
    def __init__(self) -> None:
        super().__init__()
        self._predicates: dict[str, tuple[ComparisonOp, ArithNode, ArithNode]] = {}
        self._inline_predicate_count = 0

    def arith_number(self, token: Token) -> Node:
        return Node(kind="const", attrs={"value": float(token)})

    def arith_var(self, token: Token) -> Node:
        return Node(kind="var", attrs={"name": str(token)})

    def arith_add(self, left: ArithNode, right: ArithNode) -> Node:
        return Node(kind="+", children=(left, right))

    def arith_sub(self, left: ArithNode, right: ArithNode) -> Node:
        return Node(kind="-", children=(left, right))

    def arith_mul(self, left: ArithNode, right: ArithNode) -> Node:
        return Node(kind="*", children=(left, right))

    def arith_div(self, left: ArithNode, right: ArithNode) -> Node:
        return Node(kind="/", children=(left, right))

    def arith_pow_op(self, base: ArithNode, exp: ArithNode) -> Node:
        return Node(kind="^", children=(base, exp))

    def arith_neg(self, child: ArithNode) -> Node:
        return Node(
            kind="*",
            children=(Node(kind="const", attrs={"value": -1.0}), child),
        )

    def arith_abs(self, child: ArithNode) -> Node:
        return Node(kind="abs", children=(child,))

    def arith_sqrt(self, child: ArithNode) -> Node:
        return Node(kind="sqrt", children=(child,))

    def inline_predicate(
        self,
        left: ArithNode,
        op: Token,
        right: ArithNode,
    ) -> Node:
        name = f"__inline_predicate_{self._inline_predicate_count}"
        self._inline_predicate_count += 1
        return Node(
            kind="predicate",
            attrs={"name": name, "op": _COMPARISON_OP[str(op)], "left": left, "right": right},
        )

    def pred_def(self, name: Token, left: ArithNode, op: Token, right: ArithNode) -> None:
        self._predicates[str(name)] = (_COMPARISON_OP[str(op)], left, right)

    def pred_section(self, *_args: object) -> None:
        pass

    def stl_pred_ref(self, name: Token) -> Node:
        name_str = str(name)
        if name_str not in self._predicates:
            msg = f"undefined predicate: {name_str}"
            raise ValueError(msg)
        op, left, right = self._predicates[name_str]
        return Node(
            kind="predicate",
            attrs={"name": name_str, "op": op, "left": left, "right": right},
        )

    def stl_section(self, expr: Node) -> Node:
        return expr

    def stl_not(self, child: Node) -> Node:
        return Node(kind="not", children=(child,))

    def stl_and(self, left: Node, right: Node) -> Node:
        return Node(kind="and", children=(left, right))

    def stl_or(self, left: Node, right: Node) -> Node:
        return Node(kind="or", children=(left, right))

    def interval(self, a: Token, b: Token) -> tuple[float, float]:
        return (float(a), float(b))

    def stl_always(self, iv: tuple[float, float], child: Node) -> Node:
        return Node(kind="always", children=(child,), attrs={"interval": iv})

    def stl_eventually(self, iv: tuple[float, float], child: Node) -> Node:
        return Node(kind="eventually", children=(child,), attrs={"interval": iv})

    def stl_until(self, left: Node, iv: tuple[float, float], right: Node) -> Node:
        return Node(kind="until", children=(left, right), attrs={"interval": iv})

    def start(self, *args: object) -> Node:
        node = args[-1]
        if not isinstance(node, Node):
            msg = "parser did not produce a Node"
            raise TypeError(msg)
        return node


def parse(spec: str) -> Node:
    """Parse a tidystl specification string into a Node tree."""
    stripped = spec.strip()
    parser = _sectioned_parser if _is_sectioned(stripped) else _inline_parser

    tree = parser.parse(spec)  # pyright: ignore[reportUnknownMemberType]
    try:
        result: object = _SpecTransformer().transform(tree)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except VisitError as exc:
        if exc.orig_exc is not None:  # pyright: ignore[reportUnnecessaryComparison]
            raise exc.orig_exc from None
        raise  # pragma: no cover
    if not isinstance(result, Node):
        msg = "Parser did not produce a Node"
        raise TypeError(msg)
    return result


def _is_sectioned(stripped: str) -> bool:
    return "[predicates]" in stripped or "[stl]" in stripped
