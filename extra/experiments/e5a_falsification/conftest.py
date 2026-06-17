"""Make the e5a modules importable by tests and register compat backends."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXPERIMENTS = _HERE.parent

for p in (_HERE, _EXPERIMENTS):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())

import tidystl_compat  # noqa: F401,E402  -- registers compat backends as a side effect
