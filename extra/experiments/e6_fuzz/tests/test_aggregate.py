import math

import aggregate


def _ok(scalar, signal):
    return {"status": "ok", "scalar": scalar, "signal": signal}


def _abstain():
    return {"status": "abstain", "exc": "ValueError", "msg": "x"}


def test_values_match_rules():
    assert aggregate.values_match(1.0, 1.0 + 1e-9, 1e-6, 1e-6)
    assert not aggregate.values_match(1.0, 2.0, 1e-6, 1e-6)
    assert aggregate.values_match(math.inf, math.inf, 1e-6, 1e-6)
    assert not aggregate.values_match(math.inf, 1.0, 1e-6, 1e-6)
    assert not aggregate.values_match(math.nan, math.nan, 1e-6, 1e-6)


def test_scalar_matrix_agreement_and_abstention():
    backends = ["a", "b", "c"]
    pairs = [
        # a,b agree; c abstains
        {"operators": ["always"], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0]), "b": _ok(1.0, [1.0]), "c": _abstain()}},
        # a,b disagree; c ok
        {"operators": ["until"], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0]), "b": _ok(5.0, [5.0]), "c": _ok(1.0, [1.0])}},
    ]
    matrix = aggregate.scalar_matrix(pairs, backends, atol=1e-6, rtol=1e-6)
    # a vs b: 1 agree of 2 comparable -> 0.5
    assert matrix["a"]["b"] == 0.5
    assert matrix["b"]["a"] == 0.5
    assert matrix["a"]["a"] == 1.0
    # a vs c: only pair 2 comparable (pair 1 c abstains), they agree -> 1.0
    assert matrix["a"]["c"] == 1.0


def test_scalar_matrix_no_comparable_is_none():
    backends = ["a", "b"]
    pairs = [{"operators": [], "regime": "iid_uniform",
              "cells": {"a": _ok(1.0, [1.0]), "b": _abstain()}}]
    matrix = aggregate.scalar_matrix(pairs, backends, atol=1e-6, rtol=1e-6)
    assert matrix["a"]["b"] is None


def test_full_signal_matrix_requires_same_length_and_all_match():
    backends = ["a", "b"]
    pairs = [
        {"operators": [], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0, 2.0]), "b": _ok(1.0, [1.0, 2.0])}},   # match
        {"operators": [], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0, 2.0]), "b": _ok(1.0, [1.0, 9.0])}},   # one ts differs
        {"operators": [], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0]), "b": _ok(1.0, [1.0, 2.0])}},        # length mismatch -> N/A
    ]
    matrix = aggregate.full_signal_matrix(pairs, backends, atol=1e-6, rtol=1e-6)
    # comparable = 2 (the length-mismatch pair excluded); 1 fully matches -> 0.5
    assert matrix["a"]["b"] == 0.5


def test_per_operator_divergence():
    backends = ["a", "b"]
    pairs = [
        {"operators": ["until"], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0]), "b": _ok(5.0, [5.0])}},   # until: disagree
        {"operators": ["always"], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0]), "b": _ok(1.0, [1.0])}},   # always: agree
    ]
    per_op = aggregate.per_operator(pairs, backends, atol=1e-6, rtol=1e-6)
    assert per_op["until"]["n_formulas"] == 1
    assert per_op["until"]["mean_disagreement"] == 1.0
    assert per_op["always"]["mean_disagreement"] == 0.0
    # An operator that never appears reports zero formulas and None disagreement.
    assert per_op["not"]["n_formulas"] == 0
    assert per_op["not"]["mean_disagreement"] is None


def test_abstention_table():
    backends = ["a", "b"]
    pairs = [
        {"operators": [], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0]), "b": _abstain()}},
        {"operators": [], "regime": "iid_uniform",
         "cells": {"a": _ok(1.0, [1.0]), "b": _abstain()}},
    ]
    table = aggregate.abstention_table(pairs, backends)
    assert table["b"]["abstain"] == 2
    assert table["b"]["ok"] == 0
    assert table["a"]["ok"] == 2
    assert table["b"]["rate"] == 1.0


def test_assemble_product_has_all_sections():
    backends = ["a", "b"]
    pairs = [{"operators": ["always"], "regime": "iid_uniform", "has_equality": False,
              "cells": {"a": _ok(1.0, [1.0]), "b": _ok(1.0, [1.0])}}]
    product = aggregate.assemble_product(backends, pairs, atol=1e-6, rtol=1e-6)
    assert set(product) >= {"scalar_matrix", "full_signal_matrix", "per_operator", "abstention", "by_regime", "by_equality", "n_pairs", "backends"}
    assert product["n_pairs"] == 1
    assert "iid_uniform" in product["by_regime"]
    assert set(product["by_equality"]) == {"with_equality", "without_equality"}


def test_by_equality_splits_agreement():
    backends = ["a", "b"]
    pairs = [
        # equality formula: a, b disagree
        {"operators": [], "regime": "iid_uniform", "has_equality": True,
         "cells": {"a": _ok(1.0, [1.0]), "b": _ok(9.0, [9.0])}},
        # non-equality formula: a, b agree
        {"operators": [], "regime": "iid_uniform", "has_equality": False,
         "cells": {"a": _ok(1.0, [1.0]), "b": _ok(1.0, [1.0])}},
    ]
    product = aggregate.assemble_product(backends, pairs, atol=1e-6, rtol=1e-6)
    assert product["by_equality"]["with_equality"]["a"]["b"] == 0.0
    assert product["by_equality"]["without_equality"]["a"]["b"] == 1.0
