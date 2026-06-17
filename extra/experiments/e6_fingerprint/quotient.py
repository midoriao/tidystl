"""E6 exp 4 -- the identifiability-quotient report.

Turns the Tier A observational-equivalence quotient into a structured artifact:
the class count, each tool's class size and resolved/unresolved axes, and a
per-axis summary of which tools each axis is identifiable for. This is the
precise statement of what finite traces on the battery fragment can resolve.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extra.experiments.e6_fingerprint.decode import (  # noqa: E402
    AXES,
    Probe,
    decode_config,
    documented_configs,
    equivalence_classes,
    load_battery,
    observed_signature,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def quotient_report(battery: list[Probe]) -> dict[str, Any]:
    classes = equivalence_classes(battery)
    docs = documented_configs()
    axes = list(AXES)

    tools: dict[str, Any] = {}
    resolved_for: dict[str, list[str]] = {axis: [] for axis in axes}
    for tool, doc in docs.items():
        result = decode_config(observed_signature(tool, battery), battery, classes)
        resolved = [a for a in axes if result.per_axis[a] is not None]
        unresolved = [a for a in axes if result.per_axis[a] is None]
        for a in resolved:
            resolved_for[a].append(tool)
        tools[tool] = {
            "class_size": len(result),
            "recovered": doc in result,
            "resolved_axes": resolved,
            "unresolved_axes": unresolved,
            "per_axis": dict(result.per_axis),
        }

    return {
        "n_configs": sum(len(c.members) for c in classes),
        "n_classes": len(classes),
        "axes": axes,
        "tools": tools,
        "axis_identifiability": {axis: {"resolved_for": resolved_for[axis]} for axis in axes},
    }


def main() -> None:
    report = quotient_report(load_battery())
    logger.info("E6 exp 4 -- identifiability quotient")
    logger.info("%d configs -> %d classes", report["n_configs"], report["n_classes"])
    for tool, entry in report["tools"].items():
        logger.info(
            "  %-8s class=%-3d resolved=%s",
            tool,
            entry["class_size"],
            ",".join(entry["resolved_axes"]) or "(none)",
        )
    out = Path(__file__).resolve().parent / "quotient_report.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
