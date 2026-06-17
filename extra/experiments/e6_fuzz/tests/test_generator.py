import generator
import numpy as np

from tidystl import Node


def test_var_names():
    assert generator.var_names(3) == ["x0", "x1", "x2"]


def test_signal_shape_and_grid():
    rng = np.random.default_rng(0)
    times, values = generator.gen_signal_arrays(rng, "random_walk", n_vars=2, length=5)
    assert times.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert values.shape == (2, 5)


def test_signal_determinism():
    a = generator.gen_signal_arrays(np.random.default_rng(7), "iid_gaussian", 3, 8)
    b = generator.gen_signal_arrays(np.random.default_rng(7), "iid_gaussian", 3, 8)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_signal_unknown_regime():
    import pytest

    with pytest.raises(ValueError):
        generator.gen_signal_arrays(np.random.default_rng(0), "nope", 1, 3)


def _all_nodes(node):
    out = [node]
    for child in node.children:
        out.extend(_all_nodes(child))
    return out


def test_formula_determinism():
    f1 = generator.gen_formula(np.random.default_rng(3), ["x0", "x1"], max_depth=3, broad=True)
    f2 = generator.gen_formula(np.random.default_rng(3), ["x0", "x1"], max_depth=3, broad=True)
    assert f1 == f2


def test_formula_depth_zero_is_predicate():
    f = generator.gen_formula(np.random.default_rng(0), ["x0"], max_depth=0, broad=False)
    assert f.kind == "predicate"
    assert isinstance(f.attrs["left"], Node) and isinstance(f.attrs["right"], Node)
    assert f.attrs["op"] in (">", ">=", "<", "<=", "==")


def test_temporal_intervals_are_float_pairs_when_present():
    # A deep tree very likely contains a temporal op; assert any present one is well-formed.
    f = generator.gen_formula(np.random.default_rng(1), ["x0", "x1"], max_depth=4, broad=False)
    for node in _all_nodes(f):
        if node.kind in ("always", "eventually", "until"):
            iv = node.attrs["interval"]
            assert isinstance(iv, tuple) and len(iv) == 2
            assert all(isinstance(x, float) for x in iv)
            assert iv[0] <= iv[1]


def test_operators_in():
    pred = Node(kind="predicate", attrs={"name": "p", "op": ">", "left": Node(kind="var", attrs={"name": "x0"}), "right": Node(kind="const", attrs={"value": 0.0})})
    phi = Node(kind="always", attrs={"interval": (0.0, 2.0)}, children=(Node(kind="not", children=(pred,)),))
    assert generator.operators_in(phi) == {"always", "not"}


def _eqpred(op):
    return Node(kind="predicate", attrs={"name": "p", "op": op, "left": Node(kind="var", attrs={"name": "x0"}), "right": Node(kind="const", attrs={"value": 0.0})})


def test_has_equality_predicate():
    assert generator.has_equality_predicate(_eqpred("=="))
    assert not generator.has_equality_predicate(_eqpred(">"))
    # Nested: an equality buried under temporal/boolean ops is still found.
    nested = Node(kind="eventually", attrs={"interval": (0.0, 1.0)},
                  children=(Node(kind="and", children=(_eqpred(">"), _eqpred("=="))),))
    assert generator.has_equality_predicate(nested)
    no_eq = Node(kind="and", children=(_eqpred(">"), _eqpred("<=")))
    assert not generator.has_equality_predicate(no_eq)
