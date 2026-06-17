"""Pytest setup for the experiment test suite.

Importing tidystl_compat registers the generic backend and every faithful
compat backend (breach, rtamt, pymtl, stlcgpp, taliro) with the default
tidystl registry. Doing it here, in the package conftest, guarantees the
backends are resolvable by name in every test module -- including when a
single test file is run in isolation -- rather than relying on some other test
module having imported the package first.

These tests validate the e6_fingerprint experiment and are intentionally kept
out of CI; run them with `make test` in extra/experiments.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPAT_ROOT = REPO_ROOT / "packages" / "tidystl-compat"
if str(COMPAT_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPAT_ROOT))
sys.modules.pop("tests", None)

import tidystl_compat  # noqa: E402,F401 # pyright: ignore[reportUnusedImport] -- registers backends
