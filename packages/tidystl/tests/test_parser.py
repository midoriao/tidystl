import pytest
from lark.exceptions import UnexpectedEOF

from tidystl.frontend.parser import parse


class TestParsePredicates:
    def test_simple_predicate(self) -> None:
        node = parse("""
        [predicates]
        x_pos : x >= 0

        [stl]
        x_pos
        """)
        assert node.kind == "predicate"
        assert node.attrs["name"] == "x_pos"
        assert node.attrs["op"] == ">="

    def test_predicate_with_arithmetic(self) -> None:
        node = parse("""
        [predicates]
        fast : velocity - 0.5 * ref >= 0

        [stl]
        fast
        """)
        assert node.kind == "predicate"

    def test_predicate_with_abs(self) -> None:
        node = parse("""
        [predicates]
        in_range : abs(error) <= 0.1

        [stl]
        in_range
        """)
        assert node.kind == "predicate"
        assert node.attrs["op"] == "<="

    def test_predicate_with_sqrt(self) -> None:
        node = parse("""
        [predicates]
        p : sqrt(x) >= 1

        [stl]
        p
        """)
        assert node.kind == "predicate"

    def test_predicate_with_pow(self) -> None:
        node = parse("""
        [predicates]
        p : x ^ 2 >= 4

        [stl]
        p
        """)
        assert node.kind == "predicate"


class TestParseSTL:
    def test_inline_predicate(self) -> None:
        node = parse("x >= 0")
        assert node.kind == "predicate"
        assert node.attrs["op"] == ">="

    def test_always(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0

        [stl]
        G[0,5](p)
        """)
        assert node.kind == "always"
        assert node.attrs["interval"] == (0.0, 5.0)
        assert node.children[0].kind == "predicate"

    def test_eventually(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0

        [stl]
        F[0,3](p)
        """)
        assert node.kind == "eventually"
        assert node.attrs["interval"] == (0.0, 3.0)

    def test_and(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0
        q : y >= 0

        [stl]
        p and q
        """)
        assert node.kind == "and"

    def test_or(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0
        q : y >= 0

        [stl]
        p or q
        """)
        assert node.kind == "or"

    def test_not(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0

        [stl]
        not p
        """)
        assert node.kind == "not"

    def test_until(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0
        q : y >= 1

        [stl]
        p U[0,5] q
        """)
        assert node.kind == "until"
        assert node.attrs["interval"] == (0.0, 5.0)

    def test_nested(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0

        [stl]
        G[0,10](F[0,3](p))
        """)
        assert node.kind == "always"
        assert node.children[0].kind == "eventually"
        assert node.children[0].children[0].kind == "predicate"

    def test_inline_temporal_formula(self) -> None:
        node = parse("G[0,10](F[0,3](x >= 0))")
        assert node.kind == "always"
        assert node.children[0].kind == "eventually"
        assert node.children[0].children[0].kind == "predicate"

    def test_inline_boolean_formula(self) -> None:
        node = parse("(x >= 0) and (y >= 1)")
        assert node.kind == "and"
        assert node.children[0].kind == "predicate"
        assert node.children[1].kind == "predicate"


class TestParsePrecedence:
    def test_and_binds_tighter_than_or(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0
        q : y >= 0
        r : z >= 0

        [stl]
        p or q and r
        """)
        assert node.kind == "or"
        assert node.children[1].kind == "and"

    def test_parentheses_override_precedence(self) -> None:
        node = parse("""
        [predicates]
        p : x >= 0
        q : y >= 0
        r : z >= 0

        [stl]
        (p or q) and r
        """)
        assert node.kind == "and"
        assert node.children[0].kind == "or"


class TestParseErrors:
    def test_undefined_predicate(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            parse("""
            [predicates]
            p : x >= 0

            [stl]
            q
            """)

    def test_inline_bare_name_is_rejected(self) -> None:
        with pytest.raises(UnexpectedEOF):
            parse("p")
