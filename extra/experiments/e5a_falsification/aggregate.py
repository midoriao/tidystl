from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MIN_REPETITIONS = 50  # paper-grade evidence threshold; smoke runs are below
BACKENDS = ("native", "breach", "rtamt")

STYLES = {
    "native": {"color": "#1a6faf", "linestyle": "-"},
    "breach": {"color": "#c44e52", "linestyle": "--"},
    "rtamt": {"color": "#55a868", "linestyle": "-."},
}

# Paper wording rule (eval-diagnosis.md P2): a backend is always "the
# X-compatible backend"; bare tool names refer only to the real tools.
# The legend names the decision-point combination, since that is what
# the experiment varies; the backend is the realization.
LABELS = {
    "native": "clamp + as-is (native backend)",
    "breach": "clamp + extend-penultimate (Breach-compat.)",
    "rtamt": "pessimistic + as-is (RTAMT-compat.)",
}

# The named decision-point combination each backend realizes (moved out
# of config.toml; this is cross-condition framing, not a run knob).
DECISION_POINTS = {
    "native": "clamping boundary + as-is terminal",
    "breach": "clamping boundary + extend-penultimate terminal",
    "rtamt": "pessimistic boundary + as-is terminal",
}


def ecdf_steps(evals: list[int], n_runs: int, budget: int) -> tuple[Any, Any]:
    """Right-continuous ECDF over the evaluation budget (plateau = rate)."""
    xs = np.concatenate([[0], np.sort(np.asarray(evals, dtype=float)), [budget]])
    ys = np.concatenate([[0.0], np.arange(1, len(evals) + 1) / n_runs, [len(evals) / n_runs]])
    return xs, ys


def gather_facts(result_dir: Path) -> list[dict[str, Any]]:
    """Each run's ``result.json`` (plus ``eval_budget`` from ``params.json``,
    needed for the ECDF x-axis) under ``result_dir``, ascending; partial runs skipped.
    """
    facts: list[dict[str, Any]] = []
    run_dirs = sorted(
        (p for p in result_dir.glob("run_*") if p.is_dir() and p.name.removeprefix("run_").isdigit()),
        key=lambda p: int(p.name.removeprefix("run_")),
    )
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        params_path = run_dir / "params.json"
        if not (result_path.exists() and params_path.exists()):
            continue
        try:
            result = json.loads(result_path.read_text())
            params = json.loads(params_path.read_text())
        except json.JSONDecodeError:
            continue
        facts.append({**result, "eval_budget": params["eval_budget"]})
    return facts


def select_records(result_dir: Path) -> dict[str, dict[str, Any]]:
    """Latest full record per backend; error naming any missing backend."""
    latest: dict[str, dict[str, Any]] = {}
    for fact in gather_facts(result_dir):
        if fact["n_trials"] >= MIN_REPETITIONS:
            latest[fact["backend"]] = fact  # gather_facts is ascending: last wins
    missing = [b for b in BACKENDS if b not in latest]
    if missing:
        raise SystemExit(
            f"missing full records (n_repetitions >= {MIN_REPETITIONS}) for "
            f"backend(s): {', '.join(missing)}; run the full benchmark"
        )
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="extra copy (paper repo path)")
    args = parser.parse_args()

    records = select_records(args.result_dir)

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for backend in BACKENDS:
        record = records[backend]
        eval_budget: int = record["eval_budget"]
        n_runs: int = record["n_trials"]
        evals = record["evals_to_falsification"]
        xs, ys = ecdf_steps(evals, n_runs, eval_budget)
        rate = record["falsifying_rate"]
        ax.step(
            xs,
            ys,
            where="post",
            label=f"{LABELS[backend]}, rate {rate:.2f}",
            linewidth=1.6,
            **STYLES[backend],
        )

    budget = max(records[b]["eval_budget"] for b in BACKENDS)
    ax.set_xlabel("simulations until first validated falsification")
    ax.set_ylabel("falsifying rate")
    ax.set_xlim(0, budget)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=6.5, loc="center right", framealpha=0.9)
    fig.tight_layout()

    if args.output is not None:
        fig.savefig(args.output)
        print(f"wrote {args.output}")

    for backend in BACKENDS:
        record = records[backend]
        print(
            f"  {backend:<8} rate={record['falsifying_rate']:.2f} "
            f"median={record['median_evals']} "
            f"n={record['n_falsified']}/{record['n_trials']}  "
            f"[{DECISION_POINTS[backend]}]"
        )


if __name__ == "__main__":
    main()
