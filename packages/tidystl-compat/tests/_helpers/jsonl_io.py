"""Read/write JSON Lines ground-truth files (stdlib only, no numpy/tidystl).

Each line is one case object: ``{"name": str, "time": [float], "robustness": [float]}``.
Lines are written sorted by ``name`` for stable, reviewable diffs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def read_jsonl(path: str | Path) -> list[Record]:
    """Parse a JSONL file into a list of dicts, skipping blank lines."""
    records: list[Record] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[Record]) -> None:
    """Write records as JSONL, one object per line, sorted by ``name``."""
    ordered = sorted(records, key=lambda r: r["name"])
    with Path(path).open("w") as f:
        for rec in ordered:
            f.write(json.dumps(rec) + "\n")
