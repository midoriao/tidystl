"""Shared scaffolding for the E4 real-tool runners (``run_rtamt`` /
``run_stlcgpp`` / ``run_breach``): the per-cell row and result-record shapes
and the ``Infra`` wrapper that opens a run dir, times the sweep, and writes the
record.

stdlib-only (no tidystl, no numpy), like ``_lib.infra`` -- safe to import in the
shared experiments venv and in the stlcgpp ``--cell`` child.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from _lib.infra import Infra


@dataclass
class RunResult:
    """The facts: per-cell timing rows plus tool-specific provenance.

    ``info`` holds machine provenance (python, version, torch/matlab, spec,
    batching_model, ...); ``tool`` is the record self-identifier
    (``rtamt-real`` / ``stlcgpp-real`` / ``breach-real``).
    """

    tool: str
    partial: bool
    info: dict[str, Any]
    results: list[dict[str, Any]]


def real_row(
    backend: str, n: int, t: int, mean_ms: float, std_ms: float, repeats: int
) -> dict[str, Any]:
    """One ``ok`` timing row. Raw facts only; per-trace/throughput are derived
    in aggregate.py. ``repeats``/``status`` ARE facts: repeats is adaptive per
    cell, and status records skip/OOM outcomes for the non-``ok`` rows."""
    return {
        "backend": backend,
        "batch_size": n,
        "timesteps": t,
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(std_ms, 4),
        "repeats": repeats,
        "status": "ok",
    }


def make_result(
    tool: str, rows: list[dict[str, Any]], meta: dict[str, Any], *, partial: bool
) -> RunResult:
    """Wrap rows + provenance into a ``RunResult`` (stamps the python version)."""
    info = {"python": sys.version.split()[0], **meta}
    return RunResult(tool=tool, partial=partial, info=info, results=rows)


def run_real(
    experiment: str,
    params: Any,
    runner: Callable[[Any, Path, Path], RunResult],
    tool_versions: Callable[[RunResult], dict[str, str]],
) -> None:
    """Open a run dir, time ``runner``, and write the record via ``Infra``.

    ``runner(params, run_dir, params_path) -> RunResult``: rtamt ignores the dir
    args; stlcgpp/breach write into the fresh ``run_dir`` (stlcgpp children also
    read the up-front ``params.json`` at ``params_path``). ``tool_versions``
    maps the finished result to the record's ``tool_versions`` extra-meta.
    """
    env = Infra.capture_env(experiment)
    # params.json is written up front so the stlcgpp children can read the
    # resolved params during the run; record_success then skips it (write-if-absent).
    params_path = env.run_dir / "params.json"
    Infra.write_json(params_path, asdict(params))
    with Infra.run_with_timer(env) as timer:
        result = runner(params, env.run_dir, params_path)
    Infra.record_success(
        env=env,
        params=asdict(params),
        result=asdict(result),
        timer=timer,
        extra_meta={"tool_versions": tool_versions(result)},
    )
