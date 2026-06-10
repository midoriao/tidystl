"""E3 runner: measure one variant's LOC, layer, and tests.

LOC is attributed by top-level definition via ``ast`` per ``LOC_RULE``;
``VARIANT_CATALOG`` names which files/defs each variant owns (the
``full_suite`` entry is the whole-library check, no LOC).
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from extra.experiments._lib.infra import Infra  # noqa: E402

EXPERIMENT = "e3_variation"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The LOC counting rule, recorded with every run (stated in the paper).
LOC_RULE = (
    "variant-specific code only; blank/comment-only/docstring lines "
    "excluded; shared infrastructure excluded; attribution by top-level "
    "definition name (see script); Rust counted separately"
)

#: Variant catalog (inlined; was config.toml ``[[variants]]``). One entry
#: per catalog row: the layer description, the LOC attribution spec
#: (``python`` = list of ``{file, include_only?, exclude?}`` counted by the
#: AST-based rule), and the variant's test targets. The ``full_suite``
#: entry is the whole-library test check (no LOC). ``aggregate.py`` keeps
#: its own display catalog (table order, terse layer cells).
VARIANT_CATALOG: list[dict[str, Any]] = [
    {
        "variant": "breach",
        "kind": "semantic",
        "layer": "op-override on the shared PL executor (+ documented terminal-step post-step)",
        "trace_support": True,
        "test_targets": ["tests/test_breach_compat.py", "tests/test_backends.py"],
        "python": [{"file": "src/tidystl/backends/breach.py"}],
    },
    {
        "variant": "rtamt",
        "kind": "semantic",
        "layer": "own lowering + executor (discrete-time ops)",
        "trace_support": True,
        "test_targets": ["tests/test_rtamt_compat.py", "tests/test_rtamt_backend.py"],
        "python": [{"file": "src/tidystl/backends/rtamt.py"}],
    },
    {
        "variant": "stlcgpp",
        "kind": "semantic",
        "layer": "own lowering + executor (discrete ops, hard min/max)",
        "trace_support": True,
        "test_targets": ["tests/test_stlcgpp_compat.py", "tests/test_stlcgpp_backend.py"],
        "python": [
            {
                "file": "src/tidystl/backends/stlcgpp.py",
                "exclude": [
                    "_torch_maxish",
                    "_torch_minish",
                    "_torch_sliding_reduce_last",
                    "_torch_eval_until_inclusive_last",
                    "_StlcgppTorchResult",
                    "_TorchExecutor",
                    "StlcgppTorchBackend",
                ],
            }
        ],
    },
    {
        "variant": "stlcgpp_torch",
        "kind": "computational",
        "layer": (
            "executor swap over the same lowering (torch ops; optional smooth "
            "approx); semantics unchanged"
        ),
        "trace_support": True,
        "test_targets": ["tests/test_stlcgpp_torch_backend.py"],
        "python": [
            {
                "file": "src/tidystl/backends/stlcgpp.py",
                "include_only": [
                    "_torch_maxish",
                    "_torch_minish",
                    "_torch_sliding_reduce_last",
                    "_torch_eval_until_inclusive_last",
                    "_StlcgppTorchResult",
                    "_TorchExecutor",
                    "StlcgppTorchBackend",
                ],
            },
            {
                "file": "src/tidystl/backends/helper.py",
                "include_only": ["TorchArithmeticOpEvaluator"],
            },
        ],
    },
    {
        "variant": "tidystl_simd",
        "kind": "computational",
        "layer": "native-extension executor (Rust SIMD); semantics of native; no trace support",
        "trace_support": False,
        "test_targets": ["extra/tidystl_simd/tests"],
        "rust_globs": ["extra/tidystl_simd/src/*.rs"],
        "python": [
            {"file": "extra/tidystl_simd/tidystl_simd/backend.py"},
            {"file": "extra/tidystl_simd/tidystl_simd/__init__.py"},
        ],
    },
    {
        "variant": "full_suite",
        "kind": "suite",
        "layer": "(whole library)",
        "trace_support": False,
        "test_targets": ["tests/"],
        "python": [],
    },
]


@dataclass(frozen=True)
class RunParams:
    """The run knob (inlined; was config.toml): which variant to measure.

    ``batch.sh`` sweeps the catalog (every ``VARIANT_CATALOG`` entry).
    """

    variant: str = "breach"


@dataclass
class RunInput:
    """The condition: one variant catalog entry under one LOC rule."""

    variant: dict[str, Any]
    loc_rule: str


@dataclass
class RunResult:
    """The facts: code cost, locality, and regression evidence."""

    variant: str
    kind: str
    layer: str
    loc_rule: str
    python_loc: int
    rust_loc: int | None
    tests: dict[str, Any]
    trace_support: bool


def _docstring_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def _toplevel_ranges(tree: ast.Module) -> dict[str, tuple[int, int]]:
    """Name -> (first line incl. decorators, last line) for top-level defs."""
    ranges: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            assert node.end_lineno is not None
            ranges[node.name] = (start, node.end_lineno)
    return ranges


def count_python_loc(
    path: Path,
    *,
    include_only: frozenset[str] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> int:
    """Non-blank, non-comment, non-docstring lines, attributed by name.

    ``include_only``: count only the named top-level definitions.
    ``exclude``: count the whole file minus the named definitions.
    """
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    docstrings = _docstring_lines(tree)
    ranges = _toplevel_ranges(tree)

    selected: set[int]
    if include_only is not None:
        unknown = include_only - ranges.keys()
        if unknown:
            raise KeyError(f"{path.name}: unknown definitions {sorted(unknown)}")
        selected = set()
        for name in include_only:
            start, end = ranges[name]
            selected.update(range(start, end + 1))
    else:
        selected = set(range(1, len(lines) + 1))
        for name in exclude:
            start, end = ranges[name]
            selected.difference_update(range(start, end + 1))

    count = 0
    for i in sorted(selected):
        text = lines[i - 1].strip()
        if text and not text.startswith("#") and i not in docstrings:
            count += 1
    return count


def count_rust_loc(paths: list[Path]) -> int:
    """Non-blank lines not starting with ``//`` (incl. ``///`` docs)."""
    count = 0
    for path in paths:
        for line in path.read_text().splitlines():
            text = line.strip()
            if text and not text.startswith("//"):
                count += 1
    return count


def run_pytest(targets: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *targets],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = re.search(r"(\d+) passed", summary)
    failed = re.search(r"(\d+) failed", summary)
    return {
        "targets": targets,
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "ok": proc.returncode == 0,
        "summary": summary,
    }


def python_loc(specs: list[dict[str, Any]]) -> int:
    """Total LOC over a variant's ``python`` attribution specs."""
    total = 0
    for spec in specs:
        include_only = spec.get("include_only")
        total += count_python_loc(
            REPO_ROOT / spec["file"],
            include_only=frozenset(include_only) if include_only is not None else None,
            exclude=frozenset(spec.get("exclude", ())),
        )
    return total


def runner(input: RunInput) -> RunResult:
    """Measured facts for one variant (LOC + tests + trace support).

    Identity fields (variant/kind/layer) come straight from the catalog
    entry; the rest is measured here.
    """
    variant = input.variant
    rust_globs: list[str] = variant.get("rust_globs", [])
    rust_paths = sorted(p for g in rust_globs for p in REPO_ROOT.glob(g))
    return RunResult(
        variant=variant["variant"],
        kind=variant["kind"],
        layer=variant["layer"],
        loc_rule=input.loc_rule,
        python_loc=python_loc(variant["python"]),
        rust_loc=count_rust_loc(rust_paths) if rust_globs else None,
        tests=run_pytest(variant["test_targets"]),
        trace_support=variant["trace_support"],
    )


def select_variant(variants: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """The single catalog entry named ``name``; error listing names if absent."""
    for entry in variants:
        if entry["variant"] == name:
            return entry
    available = ", ".join(v["variant"] for v in variants)
    raise SystemExit(f"unknown variant {name!r}; available: {available}")


def main() -> None:
    params = tyro.cli(RunParams)
    entry = select_variant(VARIANT_CATALOG, params.variant)
    inp = RunInput(variant=entry, loc_rule=LOC_RULE)

    env = Infra.capture_env(EXPERIMENT)
    with Infra.run_with_timer(env) as timer:
        result = runner(inp)

    Infra.record_success(
        env=env,
        params=asdict(params),
        result=asdict(result),
        timer=timer,
    )

    tests = result.tests
    rust = str(result.rust_loc) if result.rust_loc is not None else "--"
    status = f"{tests['passed']} passed" + ("" if tests["ok"] else " [FAIL]")
    logger.info(
        "variant=%s pyLOC=%d rsLOC=%s tests=%s ok=%s",
        result.variant,
        result.python_loc,
        rust,
        status,
        tests["ok"],
    )


if __name__ == "__main__":
    main()
