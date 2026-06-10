import numpy as np

from tidystl import Signal, StlcgppBackend, robustness
from tidystl.core.nodes import Node


def _const(v: float) -> Node:
    return Node(kind="const", attrs={"value": v})


def _var(name: str) -> Node:
    return Node(kind="var", attrs={"name": name})


def _pred(name: str, op: str, left: Node, right: Node) -> Node:
    return Node(
        kind="predicate",
        attrs={"name": name, "op": op, "left": left, "right": right},
    )


def test_stlcgpp_always_extends_last_sample_at_trace_end() -> None:
    sig = Signal.from_dict(
        times=np.arange(3, dtype=float),
        values={"x": np.array([[3.0, 1.0, 4.0]])},
    )
    pred = _pred("p", ">=", _var("x"), _const(0.0))
    phi = Node(kind="always", children=(pred,), attrs={"interval": (1.0, 2.0)})

    rho = robustness(phi, sig, backend=StlcgppBackend())
    np.testing.assert_allclose(rho[0], [1.0, 4.0, 4.0])


def test_stlcgpp_eventually_extends_last_sample_at_trace_end() -> None:
    sig = Signal.from_dict(
        times=np.arange(3, dtype=float),
        values={"x": np.array([[3.0, 1.0, 4.0]])},
    )
    pred = _pred("p", ">=", _var("x"), _const(0.0))
    phi = Node(kind="eventually", children=(pred,), attrs={"interval": (1.0, 2.0)})

    rho = robustness(phi, sig, backend=StlcgppBackend())
    np.testing.assert_allclose(rho[0], [4.0, 4.0, 4.0])


def test_stlcgpp_until_includes_witness_sample_in_left_prefix() -> None:
    sig = Signal.from_dict(
        times=np.arange(2, dtype=float),
        values={
            "x": np.array([[5.0, -5.0]]),
            "y": np.array([[-1.0, 2.0]]),
        },
    )
    p = _pred("p", ">=", _var("x"), _const(0.0))
    q = _pred("q", ">=", _var("y"), _const(0.0))
    phi = Node(kind="until", children=(p, q), attrs={"interval": (1.0, 1.0)})

    rho = robustness(phi, sig, backend=StlcgppBackend())
    assert rho[0, 0] == -5.0


def test_stlcgpp_requires_uniform_sampling_grid() -> None:
    sig = Signal.from_dict(
        times=np.array([0.0, 1.0, 3.0]),
        values={"x": np.array([[1.0, 2.0, 3.0]])},
    )
    pred = _pred("p", ">=", _var("x"), _const(0.0))
    phi = Node(kind="always", children=(pred,), attrs={"interval": (1.0, 1.0)})

    try:
        robustness(phi, sig, backend=StlcgppBackend())
    except ValueError as exc:
        assert "uniform time grid" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-uniform sampling grid")
