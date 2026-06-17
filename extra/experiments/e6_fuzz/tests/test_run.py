import generator
import numpy as np
import run

from tidystl import Node


def _pred(var, op, const):
    return Node(kind="predicate", attrs={
        "name": "p0", "op": op,
        "left": Node(kind="var", attrs={"name": var}),
        "right": Node(kind="const", attrs={"value": float(const)}),
    })


def _signals():
    times = np.arange(6, dtype=float)
    values = np.array([[0.0, 1.0, -1.0, 2.0, 0.0, 3.0]])
    names = ["x0"]
    return run.build_signal(times, values, names), run.build_torch_signal(times, values, names)


def test_evaluate_cell_ok_native():
    sig, tsig = _signals()
    phi = Node(kind="always", attrs={"interval": (0.0, 2.0)}, children=(_pred("x0", ">", 0.0),))
    cell = run.evaluate_cell(phi, sig, tsig, "native")
    assert cell["status"] == "ok"
    assert cell["scalar"] == -1.0
    assert len(cell["signal"]) == 6


def test_evaluate_cell_torch_backend_uses_torch_signal():
    sig, tsig = _signals()
    phi = Node(kind="always", attrs={"interval": (0.0, 2.0)}, children=(_pred("x0", ">", 0.0),))
    cell = run.evaluate_cell(phi, sig, tsig, "stlcgpp_torch")
    assert cell["status"] == "ok"
    assert cell["scalar"] == -1.0


def test_evaluate_cell_abstains_on_equality_for_taliro():
    sig, tsig = _signals()
    phi = _pred("x0", "==", 0.0)
    cell = run.evaluate_cell(phi, sig, tsig, "taliro")
    assert cell["status"] == "abstain"
    assert cell["exc"] in ("NotImplementedError", "ValueError", "TypeError")


def test_runner_small_is_deterministic_and_well_shaped():
    params = run.RunParams(n_pairs=6, seed=0, max_depth=2)
    r1 = run.runner(params)
    r2 = run.runner(params)
    assert r1.backends == [
        "breach", "native", "pymtl", "rtamt", "rtamt_dense",
        "stlcgpp", "stlcgpp_torch", "taliro",
    ]
    assert len(r1.pairs) == 6
    # Determinism: same seed -> identical recorded facts. Compare via JSON
    # because robustness values may be NaN (nan != nan defeats dict ==), and
    # this also confirms the facts are JSON-serializable for the run record.
    import json
    assert json.dumps(r1.pairs) == json.dumps(r2.pairs)
    # Every pair has a cell for every backend, each with a status.
    for pair in r1.pairs:
        assert set(pair["cells"]) == set(r1.backends)
        for cell in pair["cells"].values():
            assert cell["status"] in ("ok", "abstain", "error")
        assert isinstance(pair["operators"], list)
        assert isinstance(pair["has_equality"], bool)
        assert pair["regime"] in generator.REGIMES
