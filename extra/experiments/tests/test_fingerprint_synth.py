"""E6 exp 3 -- probe synthesis tests.

Part of the e6_fingerprint experiment, run via `make test` in extra/experiments
(not CI). They evaluate against the tool backends (via decode.observed_signature)
and the generic backend, all of which register on `import tidystl_compat` (done
by the package conftest). The repo root is put on the path to resolve the
`extra.experiments...` import of the experiment source.

The 192-config signature matrix is expensive to build, so it is a module-scoped
fixture shared by every test here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extra.experiments.e6_fingerprint.decode import Probe  # noqa: E402
from extra.experiments.e6_fingerprint.synth import (  # noqa: E402
    SignatureMatrix,
    candidate_probes,
    discriminates,
    find_axis_discriminators,
    quotient_classes,
    synthesize_battery,
)


@pytest.fixture(scope="module")
def probes() -> list[Probe]:
    return candidate_probes()


@pytest.fixture(scope="module")
def matrix(probes: list[Probe]) -> SignatureMatrix:
    return SignatureMatrix.build(probes)


def test_candidate_probes_are_well_formed_and_plentiful(probes: list[Probe]) -> None:
    assert len(probes) >= 20  # a real search pool, not just the hand battery
    assert all(isinstance(p, Probe) for p in probes)
    # names are unique
    assert len({p.name for p in probes}) == len(probes)


def test_matrix_covers_all_configs_and_candidates(
    probes: list[Probe], matrix: SignatureMatrix
) -> None:
    assert matrix.n_configs == 192
    assert set(matrix.probe_names) == {p.name for p in probes}


def test_boundary_is_discriminated_by_a_past_end_window_probe(matrix: SignatureMatrix) -> None:
    # F[2,4](x>=0)@past_end runs past the trace end -> separates clamp vs pessimistic.
    name = "F_2_4__x____0@past_end"
    assert discriminates(matrix, name, "boundary")


def test_every_axis_has_at_least_one_discriminator(matrix: SignatureMatrix) -> None:
    discriminators = find_axis_discriminators(matrix)
    for axis in ("signal_model", "boundary", "terminal", "predicate", "equality", "until_prefix"):
        assert discriminators[axis] is not None, axis


def test_synthesized_battery_refines_to_pool_maximum(
    matrix: SignatureMatrix, synth_names: list[str]
) -> None:
    # The synthesized battery achieves the finest quotient the whole pool can:
    full = len(quotient_classes(matrix, list(matrix.probe_names)))
    chosen = len(quotient_classes(matrix, synth_names))
    assert chosen == full
    # ...and it is minimal: dropping any probe coarsens the quotient.
    for name in synth_names:
        reduced = [n for n in synth_names if n != name]
        assert len(quotient_classes(matrix, reduced)) < chosen


from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    decode_config,
    documented_configs,
    equivalence_classes,
    load_battery,
    observed_signature,
)


@pytest.fixture(scope="module")
def synth_names(matrix: SignatureMatrix) -> list[str]:
    return synthesize_battery(matrix)


@pytest.fixture(scope="module")
def synth_probes(probes: list[Probe], synth_names: list[str]) -> list[Probe]:
    names = set(synth_names)
    return [p for p in probes if p.name in names]


def test_synthesis_is_at_least_as_discriminating_as_hand_battery(
    matrix: SignatureMatrix, synth_names: list[str]
) -> None:
    # The hand-made Tier A battery induces some quotient; the synthesized battery
    # must be at least as fine.
    hand_classes = len(equivalence_classes(load_battery()))
    synth_classes = len(quotient_classes(matrix, synth_names))
    assert synth_classes >= hand_classes


def test_synthesis_resolves_every_axis_the_hand_battery_resolves(
    synth_probes: list[Probe],
) -> None:
    # For each tool, every axis the hand battery resolves must also be resolved
    # (to the same value) by the synthesized battery.
    hand = load_battery()
    hand_classes = equivalence_classes(hand)
    synth_classes = equivalence_classes(synth_probes)
    for tool in documented_configs():
        hand_res = decode_config(observed_signature(tool, hand), hand, hand_classes).per_axis
        synth_res = decode_config(
            observed_signature(tool, synth_probes), synth_probes, synth_classes
        ).per_axis
        for axis, value in hand_res.items():
            if value is not None:
                assert synth_res[axis] == value, (tool, axis)
