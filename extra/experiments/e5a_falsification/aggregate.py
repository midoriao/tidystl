"""Aggregate E5a records into a per-optimizer ECDF and a difficulty-sweep figure.

Reads committed run records; never reruns the experiment. The companion
``agreement.py`` produces the random-trace agreement table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

import specs  # noqa: E402

MIN_REPETITIONS = 50
# Tool-compatible backends only (native is the oracle/reference, never a condition).
BACKENDS = ("breach", "rtamt", "rtamt_dense", "pymtl", "taliro", "stlcgpp")
OPTIMIZERS = ("cma", "anneal")
HERO = "m2_mass_spring"

# Six distinguishable styles; pymtl highlighted (the expected outlier).
STYLE = {
    "breach": {"color": "#1a6faf", "linestyle": "-"},
    "rtamt": {"color": "#4c9f70", "linestyle": "-"},
    "rtamt_dense": {"color": "#8a6fbf", "linestyle": "--"},
    "pymtl": {"color": "#c44e52", "linestyle": "-"},
    "taliro": {"color": "#dd8452", "linestyle": "--"},
    "stlcgpp": {"color": "#937860", "linestyle": "-."},
}


def gather_facts(result_dir: Path) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    run_dirs = sorted(
        (p for p in result_dir.glob("run_*") if p.is_dir() and p.name.removeprefix("run_").isdigit()),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    for run_dir in run_dirs:
        rp, pp = run_dir / "result.json", run_dir / "params.json"
        if not (rp.exists() and pp.exists()):
            continue
        try:
            result = json.loads(rp.read_text())
            params = json.loads(pp.read_text())
        except json.JSONDecodeError:
            continue
        facts.append({**result, "eval_budget": params["eval_budget"]})
    return facts


def select(result_dir: Path) -> dict[tuple, dict[str, Any]]:
    """Latest full (n_trials >= 50) record per (backend, benchmark, optimizer, threshold, dt)."""
    latest: dict[tuple, dict[str, Any]] = {}
    for f in gather_facts(result_dir):
        if f["n_trials"] >= MIN_REPETITIONS:
            key = (f["backend"], f["benchmark"], f["optimizer"], round(float(f["threshold"]), 6), float(f["dt"]))
            latest[key] = f
    return latest


def ecdf_steps(evals: list[int], n_runs: int, budget: int) -> tuple[Any, Any]:
    xs = np.concatenate([[0], np.sort(np.asarray(evals, dtype=float)), [budget]])
    ys = np.concatenate([[0.0], np.arange(1, len(evals) + 1) / n_runs, [len(evals) / n_runs]])
    return xs, ys


def ecdf_figure(records: dict[tuple, dict[str, Any]], out: Path) -> None:
    bench = specs.BENCHMARKS[HERO]
    thr = round(bench.hero_threshold, 6)
    dt = float(bench.model.dt)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, opt in zip(axes, OPTIMIZERS, strict=True):
        for b in BACKENDS:
            rec = records.get((b, HERO, opt, thr, dt))
            if rec is None:
                continue
            xs, ys = ecdf_steps(rec["evals_to_falsification"], rec["n_trials"], rec["eval_budget"])
            ax.step(xs, ys, where="post", linewidth=1.5, label=b, **STYLE[b])
        ax.set_title(f"{opt}", fontsize=9)
        ax.set_xlabel("simulations until validated falsification")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("falsifying rate")
    axes[1].legend(fontsize=6.0, loc="lower right", framealpha=0.9)
    fig.suptitle(f"hero={HERO}, threshold={bench.hero_threshold}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def difficulty_figure(records: dict[tuple, dict[str, Any]], out: Path) -> None:
    bench = specs.BENCHMARKS[HERO]
    dt = float(bench.model.dt)
    thresholds = sorted(bench.thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    for ax, opt in zip(axes, OPTIMIZERS, strict=True):
        for b in BACKENDS:
            xs, ys = [], []
            for t in thresholds:
                rec = records.get((b, HERO, opt, round(t, 6), dt))
                if rec is not None:
                    xs.append(t)
                    ys.append(rec["falsifying_rate"])
            if xs:
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=b, **STYLE[b])
        ax.set_title(f"{opt}", fontsize=9)
        ax.set_xlabel(f"spec threshold ({HERO})")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    axes[0].set_ylabel("falsifying rate")
    axes[1].legend(fontsize=6.0, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--ecdf", type=Path, default=None)
    parser.add_argument("--difficulty", type=Path, default=None)
    args = parser.parse_args()

    records = select(args.result_dir)
    if args.ecdf is not None:
        ecdf_figure(records, args.ecdf)
    if args.difficulty is not None:
        difficulty_figure(records, args.difficulty)

    for key, rec in sorted(records.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][3])):
        backend, benchmark, optimizer, threshold, dt = key
        print(f"  {optimizer:<7} {backend:<11} {benchmark:<14} thr={threshold:<5} "
              f"rate={rec['falsifying_rate']:.2f} median={rec['median_evals']} "
              f"n={rec['n_falsified']}/{rec['n_trials']}")


if __name__ == "__main__":
    main()
