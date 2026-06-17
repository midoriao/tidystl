from tidystl.core.nodes import Node


def _const(value: float) -> Node:
    return Node(kind="const", attrs={"value": value})


def _var(name: str) -> Node:
    return Node(kind="var", attrs={"name": name})


def _pred(name: str, op: str = ">=", left_var: str = "x", right_val: float = 0.0) -> Node:
    return Node(
        kind="predicate",
        attrs={"name": name, "op": op, "left": _var(left_var), "right": _const(right_val)},
    )


class TestNodeConstruction:
    def test_predicate_atom(self) -> None:
        node = _pred("x_pos")
        assert node.attrs["name"] == "x_pos"
        assert node.attrs["op"] == ">="

    def test_always_unary(self) -> None:
        node = Node(kind="always", children=(_pred("p"),), attrs={"interval": (0.0, 5.0)})
        assert node.kind == "always"
        assert node.attrs["interval"] == (0.0, 5.0)

    def test_until_binary(self) -> None:
        node = Node(
            kind="until",
            children=(_pred("p"), _pred("q", right_val=1.0)),
            attrs={"interval": (0.0, 3.0)},
        )
        assert node.kind == "until"
        assert node.attrs["interval"] == (0.0, 3.0)

    def test_nested_arith(self) -> None:
        expr = Node(
            kind="abs",
            children=(
                Node(
                    kind="+",
                    children=(_var("x"), _const(1.0)),
                ),
            ),
        )
        assert expr.kind == "abs"

    def test_node_is_base_class(self) -> None:
        p = _pred("p")
        assert isinstance(p, Node)
        assert isinstance(Node(kind="always", children=(p,), attrs={"interval": (0.0, 1.0)}), Node)
        assert isinstance(Node(kind="not", children=(p,)), Node)
        assert isinstance(Node(kind="and", children=(p, p)), Node)
        assert isinstance(Node(kind="or", children=(p, p)), Node)
        assert isinstance(
            Node(kind="eventually", children=(p,), attrs={"interval": (0.0, 1.0)}),
            Node,
        )
