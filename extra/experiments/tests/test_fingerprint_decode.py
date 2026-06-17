"""E6 exp 2 -- black-box semantics identification (decoder recovery test).

For each faithful tool, the decoder must recover the tool's DOCUMENTED Core 6
config from its black-box per-timestep root robustness on the battery, up to the
identifiability quotient:

- the documented config is IN the matched equivalence class (core recovery claim);
- every RESOLVED per-axis value equals the documented config's value.

A separate test asserts the battery is informative (the six tools do not all
collapse into one equivalence class).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The E6 decoder lives in the experiments tree, outside this package. Put the
# repo root on the path so the cross-package import resolves when the compat
# suite is run from packages/tidystl-compat.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    AXES,
    TOOLS,
    EquivalenceClass,
    GenericConfig,
    Probe,
    Signature,
    decode_config,
    documented_configs,
    equivalence_classes,
    load_battery,
    observed_signature,
    signatures_equal,
)


@pytest.fixture(scope="module")
def battery() -> list[Probe]:
    return load_battery()


@pytest.fixture(scope="module")
def classes(battery: list[Probe]) -> list[EquivalenceClass]:
    return equivalence_classes(battery)


@pytest.fixture(scope="module")
def docs() -> dict[str, GenericConfig]:
    return documented_configs()


@pytest.mark.parametrize("tool", TOOLS)
def test_decoder_recovers_documented_config(
    tool: str,
    battery: list[Probe],
    classes: list[EquivalenceClass],
    docs: dict[str, GenericConfig],
) -> None:
    """The tool's documented config is in its matched class; resolved axes match it."""
    observed = observed_signature(tool, battery)
    result = decode_config(observed, battery, classes)
    doc = docs[tool]

    # Core recovery claim: the true config is among the configs indistinguishable
    # from the tool on this battery.
    assert doc in result, (
        f"{tool}: documented config not in matched equivalence class (class size {len(result)})"
    )

    # Every resolved (non-None) axis must agree with the documented config; None
    # axes are the genuine quotient and are not asserted.
    for axis in AXES:
        inferred = result.per_axis[axis]
        if inferred is not None:
            assert inferred == getattr(doc, axis), (
                f"{tool}: axis {axis} resolved to {inferred!r} "
                f"but documented value is {getattr(doc, axis)!r}"
            )


def test_battery_discriminates(
    battery: list[Probe],
    classes: list[EquivalenceClass],
) -> None:
    """The battery is informative: the six tools span several equivalence classes."""
    observed_sigs = [observed_signature(tool, battery) for tool in TOOLS]

    distinct: list[Signature] = []
    for sig in observed_sigs:
        if not any(signatures_equal(sig, seen) for seen in distinct):
            distinct.append(sig)

    assert len(distinct) >= 4, (
        f"battery is not informative: only {len(distinct)} distinct signatures among the six tools"
    )
