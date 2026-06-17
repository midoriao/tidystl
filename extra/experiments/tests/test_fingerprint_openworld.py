"""E6 exp 5 -- open-world residual + planted-mutant tests.

Part of the e6_fingerprint experiment, run via `make test` in extra/experiments
(not CI). Residual matching enumerates generic configs and decodes tool
observations, all needing the backends registered by `import tidystl_compat`
(done by the package conftest). Repo root is on the path so the
`extra.experiments...` import of the experiment source resolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    TOOLS,
    Probe,
    load_battery,
    observed_signature,
)
from extra.experiments.e6_fingerprint.openworld import best_match, total_residual  # noqa: E402
from tidystl.backends.helper import Add  # noqa: E402


@pytest.fixture(scope="module")
def battery() -> list[Probe]:
    return load_battery()


@pytest.mark.parametrize("tool", TOOLS)
def test_in_model_tools_have_zero_residual(tool: str, battery: list[Probe]) -> None:
    observed = observed_signature(tool, battery)
    config, residual = best_match(observed, battery)
    assert config is not None
    assert total_residual(residual) <= 1e-12


from extra.experiments.e6_fingerprint.openworld import (  # noqa: E402
    MutantBackend,
    detect_open_world,
    localize_residual,
)


def test_planted_mutant_is_flagged_open_world(battery: list[Probe]) -> None:
    # The mutant adds a fixed off-model bias to predicate robustness -- no Core 6
    # config can reproduce it, so its best match leaves a nonzero residual.
    mutant = MutantBackend()
    observed = mutant.observed_signature(battery)
    report = detect_open_world(observed, battery)
    assert report["explained"] is False
    assert report["total_residual"] > 0.0
    # at least one probe carries the residual
    assert any(r > 0.0 for r in report["per_probe_residual"])


def test_in_model_tool_is_explained(battery: list[Probe]) -> None:
    observed = observed_signature("native", battery)
    report = detect_open_world(observed, battery)
    assert report["explained"] is True
    assert report["total_residual"] == 0.0


def test_localize_points_at_the_predicate_node(battery: list[Probe]) -> None:
    # The bias lives on the predicate op, so every minimal divergent node the
    # localizer reports must be a predicate -- not just at least one. This also
    # guards against the localizer over-reporting unrelated/untraceable nodes.
    mutant = MutantBackend()
    nodes = localize_residual(mutant, battery)
    assert nodes, "localizer found no divergent node for the planted mutant"
    assert all(kind == "predicate" for kind in nodes), nodes


def test_localize_descends_into_predicate_arithmetic(battery: list[Probe]) -> None:
    # A mutant whose off-model bias lives on the arithmetic `+` (an operand inside a
    # predicate, not a child of it) must localize to the `+` node, not the enclosing
    # predicate -- proving the localizer descends into predicate arithmetic. The
    # battery's `multivar_affine` probe is `G[0,2](x + y >= 0)`.
    mutant = MutantBackend(bias_types=(Add,))
    nodes = localize_residual(mutant, battery)
    assert nodes, "localizer found no divergent node for the arithmetic mutant"
    assert all(kind == "+" for kind in nodes), nodes
