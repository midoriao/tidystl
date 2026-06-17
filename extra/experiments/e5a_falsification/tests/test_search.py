import pytest
import search
import specs

from tidystl import parse


def _trial(optimizer, benchmark, seed, budget=200):
    b = specs.BENCHMARKS[benchmark]
    return search.falsify(optimizer=optimizer, model=b.model, phi=parse(b.spec()),
                          backend="rtamt", seed=seed, budget=budget, dt=b.model.dt)


@pytest.mark.parametrize("opt", ["cma", "anneal"])
def test_respects_budget(opt):
    assert _trial(opt, "m1_speed", seed=1, budget=60).evals_used <= 60


@pytest.mark.parametrize("opt", ["cma", "anneal"])
def test_finds_and_validates(opt):
    t = _trial(opt, "m1_speed", seed=7, budget=300)
    assert t.falsified is True
    assert t.evals_to_falsification is not None
    assert 1 <= t.evals_to_falsification <= t.evals_used


@pytest.mark.parametrize("opt", ["cma", "anneal"])
def test_deterministic(opt):
    a = _trial(opt, "m1_speed", seed=3, budget=120)
    b = _trial(opt, "m1_speed", seed=3, budget=120)
    assert a.evals_to_falsification == b.evals_to_falsification
    assert a.evals_used == b.evals_used
