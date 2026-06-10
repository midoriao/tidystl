"""Execute every fenced ``python`` code block in the user-facing docs.

Covers ``docs/usage.md`` (user manual) and ``docs/reference.md`` (API
reference). Each block must be self-contained (its own imports and data) so
that readers can copy-paste it verbatim. Blocks that import torch are skipped
when torch is not installed, but are still compiled so syntax errors surface
everywhere. Illustrative, non-runnable fragments must use a non-``python``
fence (``text``, ``bash``, ...) to stay out of this suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
DOCS = [_DOCS_DIR / "usage.md", _DOCS_DIR / "reference.md"]

_FENCE = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def _extract_blocks(doc: Path) -> list[tuple[str, str]]:
    """Return (test_id, source) for each fenced python block.

    The test id is the doc stem plus the 1-based line number of the opening
    fence, so a failing parametrized test points straight at the offending
    block.
    """
    text = doc.read_text(encoding="utf-8")
    blocks: list[tuple[str, str]] = []
    for match in _FENCE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        blocks.append((f"{doc.stem}:L{line}", match.group(1)))
    return blocks


_BLOCKS = [block for doc in DOCS for block in _extract_blocks(doc)]


@pytest.mark.parametrize("doc", DOCS, ids=[d.stem for d in DOCS])
def test_doc_has_python_blocks(doc: Path) -> None:
    assert _extract_blocks(doc), f"no fenced python blocks found in {doc}"


@pytest.mark.parametrize(("block_id", "source"), _BLOCKS, ids=[b[0] for b in _BLOCKS])
def test_usage_doc_block_executes(block_id: str, source: str) -> None:
    code = compile(source, block_id, "exec")
    if "import torch" in source:
        pytest.importorskip("torch")
    namespace: dict[str, object] = {"__name__": "__usage_doc__"}
    exec(code, namespace)  # noqa: S102
