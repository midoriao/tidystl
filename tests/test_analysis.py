import math

from tidystl.frontend.analysis import horizon, required_max_gap
from tidystl.frontend.parser import parse


def test_horizon_predicate():
    f = parse("""
    [predicates]
    p : x >= 0

    [stl]
    p
    """)
    assert horizon(f) == 0.0


def test_horizon_always():
    f = parse("""
    [predicates]
    p : x >= 0

    [stl]
    G[2,7](p)
    """)
    assert horizon(f) == 7.0


def test_horizon_eventually():
    f = parse("""
    [predicates]
    p : x >= 0

    [stl]
    F[0,10](p)
    """)
    assert horizon(f) == 10.0


def test_horizon_nested():
    # G[0,5](F[0,3](p)) -> 5 + 3 = 8
    f = parse("""
    [predicates]
    p : x >= 0

    [stl]
    G[0,5](F[0,3](p))
    """)
    assert horizon(f) == 8.0


def test_horizon_and():
    # max(3, 7) = 7
    f = parse("""
    [predicates]
    p : x >= 0
    q : y >= 0

    [stl]
    G[0,3](p) and G[0,7](q)
    """)
    assert horizon(f) == 7.0


def test_required_max_gap_predicate():
    f = parse("""
    [predicates]
    p : x >= 0

    [stl]
    p
    """)
    assert math.isinf(required_max_gap(f))


def test_required_max_gap_always():
    f = parse("""
    [predicates]
    p : x >= 0

    [stl]
    G[0,5](p)
    """)
    assert required_max_gap(f) == 5.0


def test_required_max_gap_nested():
    # min(5, 3) = 3
    f = parse("""
    [predicates]
    p : x >= 0

    [stl]
    G[0,5](F[0,3](p))
    """)
    assert required_max_gap(f) == 3.0


def test_required_max_gap_and():
    # min(5, 2) = 2
    f = parse("""
    [predicates]
    p : x >= 0
    q : y >= 0

    [stl]
    G[0,5](p) and F[0,2](q)
    """)
    assert required_max_gap(f) == 2.0
