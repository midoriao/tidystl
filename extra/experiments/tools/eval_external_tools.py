"""Evaluate spec x signal under one real external tool (exploration; no run
records). Invokes the tool by its registry key: ``rtamt`` uses the ``rtamt``
key, ``stlcgpp`` translates the ``stlcgpp`` key through the AST adapter; a
missing key means the spec is inexpressible (status ``unsupported``). Breach
has no local executor. Outputs are tool-native: RTAMT raw (time, robustness)
``rows``, STLCG++ one value per input timestep.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def load_registry(path: Path, filter_csv: str | None) -> dict[str, Any]:
    """Named entries from a registry JSON, optionally narrowed by filter."""
    registry: dict[str, Any] = json.loads(path.read_text())
    if filter_csv is None:
        return registry
    names = [n.strip() for n in filter_csv.split(",") if n.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(
            f"unknown name(s) {', '.join(unknown)} in {path} (available: {', '.join(registry)})"
        )
    return {name: registry[name] for name in names}


def eval_rtamt(spec: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    import rtamt

    text = spec.get("rtamt")
    if text is None:
        return {"status": "unsupported", "reason": "no rtamt spec"}
    try:
        rt_spec = rtamt.StlDiscreteTimeSpecification()
        for name in signal["values"]:
            rt_spec.declare_var(name, "float")
        rt_spec.spec = text
        rt_spec.parse()
        dataset: dict[str, list[float]] = {"time": list(signal["times"])}
        dataset.update({name: list(vals) for name, vals in signal["values"].items()})
        rows = rt_spec.evaluate(dataset)
        return {"status": "ok", "rows": [[float(t), float(rho)] for t, rho in rows]}
    except Exception as exc:  # noqa: BLE001 - record, never abort the sweep
        kind = type(exc).__name__
        is_rejection = "Exception" in kind or isinstance(exc, ValueError)
        return {"status": "unsupported" if is_rejection else "error", "reason": f"{kind}: {exc}"}


def eval_rtamt_dense(spec: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    """RTAMT dense-time (PWC) semantics. Reuses the ``rtamt`` spec text unless a
    dedicated ``rtamt_dense`` key is given (e.g. an off-grid window the discrete
    grammar omits). Output is RTAMT's PWC breakpoints (``rows`` = (time, robustness))."""
    import rtamt

    text = spec.get("rtamt_dense") or spec.get("rtamt")
    if text is None:
        return {"status": "unsupported", "reason": "no rtamt/rtamt_dense spec"}
    try:
        rt_spec = rtamt.StlDenseTimeSpecification()
        for name in signal["values"]:
            rt_spec.declare_var(name, "float")
        rt_spec.spec = text
        rt_spec.parse()
        args = [
            [name, [[float(t), float(v)] for t, v in zip(signal["times"], vals, strict=True)]]
            for name, vals in signal["values"].items()
        ]
        rows = rt_spec.evaluate(*args)
        return {"status": "ok", "rows": [[float(t), float(rho)] for t, rho in rows]}
    except Exception as exc:  # noqa: BLE001 - record, never abort the sweep
        kind = type(exc).__name__
        is_rejection = "Exception" in kind or isinstance(exc, ValueError)
        return {"status": "unsupported" if is_rejection else "error", "reason": f"{kind}: {exc}"}


def eval_pymtl(spec: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    """py-metric-temporal-logic (``mtl``). Atoms ARE the signal value (no arithmetic
    or equality), so only ``>= 0``-threshold predicates are expressible; cases without
    a ``pymtl`` key are inexpressible. Output is mtl's PWC breakpoints (held from the
    left, right-continuous); a bounded operator whose window runs past the trace end
    yields an empty signal (mtl truncates its output domain rather than padding)."""
    import mtl

    text = spec.get("pymtl")
    if text is None:
        return {"status": "unsupported", "reason": "no pymtl spec (a-priori inexpressible)"}
    try:
        phi = mtl.parse(text)
        trace = {
            name: list(zip(signal["times"], vals, strict=True))
            for name, vals in signal["values"].items()
        }
        out = [[float(t), float(v)] for t, v in phi(trace, time=None, quantitative=True, dt=0.1)]
        # mtl emits a breakpoint only where the PWC value changes, so the leading segment
        # (held from the trace start) can be missing -- e.g. a windowed formula whose first
        # breakpoint sits past t0. Anchor the trace start so the t=0 readout is the real value.
        t0 = float(signal["times"][0])
        if out and out[0][0] > t0 + 1e-9:
            v0 = float(phi(trace, time=t0, quantitative=True, dt=0.1))
            out.insert(0, [t0, v0])
        return {"status": "ok", "rows": out}
    except Exception as exc:  # noqa: BLE001 - record, never abort the sweep
        kind = type(exc).__name__
        is_rejection = isinstance(exc, (ValueError, NotImplementedError, KeyError))
        return {"status": "unsupported" if is_rejection else "error", "reason": f"{kind}: {exc}"}


def eval_stlcgpp(spec: dict[str, Any], signal: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    from extra.other_tools.stlcgpp.generate_ground_truth import _formula_to_stlcgpp
    from tidystl import parse

    text = spec.get("stlcgpp")
    if text is None:
        return {"status": "unsupported", "reason": "no stlcgpp spec (a-priori inexpressible)"}
    try:
        var_indices: dict[str, int] = {}
        formula = _formula_to_stlcgpp(parse(text), var_indices)
        var_names = sorted(var_indices, key=lambda n: var_indices[n])
        tensor = torch.tensor(
            np.stack([signal["values"][name] for name in var_names], axis=1),
            dtype=torch.float32,
        )
        rho = formula(tensor, padding="last", approx_method="true")
        return {"status": "ok", "robustness": [float(v) for v in rho.detach().cpu().numpy()]}
    except Exception as exc:  # noqa: BLE001 - record, never abort the sweep
        kind = type(exc).__name__
        is_rejection = isinstance(exc, (ValueError, NotImplementedError, KeyError))
        return {"status": "unsupported" if is_rejection else "error", "reason": f"{kind}: {exc}"}


EVALUATORS = {
    "rtamt": eval_rtamt,
    "rtamt_dense": eval_rtamt_dense,
    "pymtl": eval_pymtl,
    "stlcgpp": eval_stlcgpp,
}
# Tool key -> installed distribution name, where they differ (for version provenance).
DISTRIBUTION = {"rtamt_dense": "rtamt", "pymtl": "metric-temporal-logic"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tool", required=True, choices=sorted(EVALUATORS))
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-specs", default=None, metavar="NAMES")
    parser.add_argument("--filter-signals", default=None, metavar="NAMES")
    args = parser.parse_args()

    # stlcgpp's adapter imports tidystl from the workspace source tree.
    if args.tool == "stlcgpp":
        sys.path.insert(0, str(REPO_ROOT / "packages" / "tidystl" / "src"))

    distribution = DISTRIBUTION.get(args.tool, args.tool)

    specs = load_registry(args.specs, args.filter_specs)
    signals = load_registry(args.signals, args.filter_signals)
    evaluate = EVALUATORS[args.tool]
    results = [
        {"spec": spec_name, "signal": signal_name, **evaluate(spec, signal)}
        for spec_name, spec in specs.items()
        for signal_name, signal in signals.items()
    ]
    payload = {
        "tool": args.tool,
        "version": importlib.metadata.version(distribution),
        "python": sys.version.split()[0],
        "specs": str(args.specs),
        "signals": str(args.signals),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    statuses = [r["status"] for r in results]
    print(
        f"wrote {args.output}  tool={args.tool} "
        f"({len(specs)} specs x {len(signals)} signals): "
        + ", ".join(f"{s}={statuses.count(s)}" for s in sorted(set(statuses)))
    )


if __name__ == "__main__":
    main()
