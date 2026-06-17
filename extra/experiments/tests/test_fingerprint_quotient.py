"""E6 exp 4 -- identifiability-quotient report tests.

Part of the e6_fingerprint experiment, run via `make test` in extra/experiments
(not CI). The report decodes tool observations, which need the tool backends
registered by the package conftest. Repo root is on the path so the
`extra.experiments...` import of the experiment source
resolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    documented_configs,
    equivalence_classes,
    load_battery,
)
from extra.experiments.e6_fingerprint.quotient import quotient_report  # noqa: E402


def test_quotient_report_structure_and_counts() -> None:
    battery = load_battery()
    report = quotient_report(battery)
    # top-level counts agree with the Tier A engine
    assert report["n_configs"] == 192
    assert report["n_classes"] == len(equivalence_classes(battery))
    # one entry per tool, each with class_size and per_axis resolved/unresolved
    tools = set(documented_configs())
    assert set(report["tools"]) == tools
    for entry in report["tools"].values():
        assert entry["class_size"] >= 1
        assert set(entry["resolved_axes"]) | set(entry["unresolved_axes"]) == set(report["axes"])
        # the documented config must be recoverable (in its class)
        assert entry["recovered"] is True


def test_per_axis_identifiability_present() -> None:
    report = quotient_report(load_battery())
    # for each axis, the report says whether it is ever resolved for any tool
    for axis in report["axes"]:
        assert axis in report["axis_identifiability"]
        assert isinstance(report["axis_identifiability"][axis]["resolved_for"], list)
