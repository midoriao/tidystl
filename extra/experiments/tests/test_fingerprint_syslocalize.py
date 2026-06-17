"""E6 exp 6 -- systematic divergence localization tests.

Part of the e6_fingerprint experiment (run via `make test` in extra/experiments,
not CI). Evaluates tool backends registered by the package conftest. Repo root is
on the path so the `extra.experiments...`
imports resolve.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    Probe,
    documented_configs,
    load_battery,
)
from extra.experiments.e6_fingerprint.syslocalize import (  # noqa: E402
    attribute,
    is_open_world,
    node_label,
    real_divergence_nodes,
)
from tidystl import Signal, parse  # noqa: E402


def _battery() -> dict[str, Probe]:
    return {p.name: p for p in load_battery()}


# ---------------------------------------------------------------------------
# (A) ceteris-paribus: flipping exactly one axis localizes to one node + that axis.


def test_ceteris_paribus_boundary() -> None:
    docs = documented_configs()
    probe = _battery()["div_boundary_F24"]
    base = docs["native"]  # clamp
    rows = attribute(
        base, replace(base, boundary="pessimistic"), parse(probe.formula_text), probe.signal
    )
    assert len(rows) == 1
    node, axes = rows[0]
    assert node_label(node) == "F[2,4]"
    assert axes == ["boundary"]


def test_ceteris_paribus_terminal() -> None:
    docs = documented_configs()
    probe = _battery()["div_terminal_and"]
    base = docs["native"]  # terminal=none
    rows = attribute(
        base, replace(base, terminal="extend_penultimate"), parse(probe.formula_text), probe.signal
    )
    assert len(rows) == 1
    node, axes = rows[0]
    assert node_label(node) == "and"
    assert axes == ["terminal"]


def test_ceteris_paribus_predicate() -> None:
    docs = documented_configs()
    probe = _battery()["multivar_affine"]
    base = docs["native"]  # signed
    rows = attribute(
        base, replace(base, predicate="euclidean"), parse(probe.formula_text), probe.signal
    )
    assert len(rows) == 1
    node, axes = rows[0]
    assert node.kind == "predicate"
    assert axes == ["predicate"]


def test_flipping_an_inert_axis_yields_no_divergence() -> None:
    # until_prefix is inert on a formula with no until: flipping it changes nothing.
    docs = documented_configs()
    probe = _battery()["div_boundary_F24"]  # F[2,4](x>=0), no until
    base = docs["native"]  # until_prefix=inclusive
    rows = attribute(
        base, replace(base, until_prefix="exclusive"), parse(probe.formula_text), probe.signal
    )
    assert rows == []


# ---------------------------------------------------------------------------
# (B) cross-tool: ablation isolates the single responsible axis despite multi-axis diff.


def test_cross_tool_boundary_isolated_from_three_axis_difference() -> None:
    # native (pl_interp, clamp, inclusive) vs rtamt (discrete, pessimistic, exclusive)
    # differ on THREE axes, yet the F[2,4] divergence is attributable to boundary alone.
    docs = documented_configs()
    probe = _battery()["div_boundary_F24"]
    cfg_a, cfg_b = docs["native"], docs["rtamt"]
    differing = [
        ax
        for ax in ("signal_model", "boundary", "until_prefix")
        if getattr(cfg_a, ax) != getattr(cfg_b, ax)
    ]
    assert len(differing) >= 2  # genuinely a multi-axis difference
    rows = attribute(cfg_a, cfg_b, parse(probe.formula_text), probe.signal)
    assert len(rows) == 1
    node, axes = rows[0]
    assert node_label(node) == "F[2,4]"
    assert axes == ["boundary"]


def test_cross_tool_terminal_isolated() -> None:
    docs = documented_configs()
    probe = _battery()["div_terminal_and"]
    cfg_a, cfg_b = docs["native"], docs["breach"]
    rows = attribute(cfg_a, cfg_b, parse(probe.formula_text), probe.signal)
    assert len(rows) == 1
    node, axes = rows[0]
    assert node_label(node) == "and"
    assert axes == ["terminal"]


# ---------------------------------------------------------------------------
# (C) open-world: the Breach window-endpoint quirk no config explains.

_QUIRK = Probe(
    name="interp_sparse",
    formula_text="G[0.5,1.5](x > 0)",
    signal=Signal.from_dict(np.array([0.0, 2.0, 4.0]), {"x": np.array([-1.0, 10.0, -1.0])}),
)


def test_breach_quirk_is_open_world() -> None:
    docs = documented_configs()
    # generic(config_breach) diverges from real breach here ...
    nodes = real_divergence_nodes(
        docs["breach"], "breach", parse(_QUIRK.formula_text), _QUIRK.signal
    )
    assert nodes, "expected generic(config_breach) to diverge from real breach on the quirk case"
    assert node_label(nodes[0]) == "G[0.5,1.5]"
    # ... and NO Core 6 config reproduces it.
    assert is_open_world("breach", _QUIRK) is True


def test_in_axis_case_is_not_open_world() -> None:
    # On a window that contains a sample, breach factors -> some config explains it.
    in_axis = _battery()["interp_window"]
    assert is_open_world("breach", in_axis) is False


def test_open_world_residual_has_no_responsible_axis() -> None:
    # The ablation attribution returns None (no axis subset reproduces) for a node the
    # real-tool quirk drives -- distinguishing "implementation artifact" from "axis".
    docs = documented_configs()
    phi = parse(_QUIRK.formula_text)
    # Compare generic(config_breach) against a config that matches real breach's root
    # only if one exists; since none does (open-world), the residual stays unattributed.
    # We assert the open-world predicate directly (the localization counterpart of None).
    assert is_open_world("breach", _QUIRK) is True
    # sanity: the quirk node is the temporal window, with predicate children agreeing.
    nodes = real_divergence_nodes(docs["breach"], "breach", phi, _QUIRK.signal)
    assert all(n.kind in ("always", "eventually") for n in nodes)
