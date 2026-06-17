"""Evaluate spec x signal under one tidystl backend (exploration; no run
records). Reads named specs/signals from ``registry/`` (using each spec's
``tidystl`` syntax key) and writes one flat JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import tidystl_compat  # noqa: E402

from tidystl import Signal, parse, robustness, use  # noqa: E402

use(tidystl_compat)


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


def evaluate(spec: dict[str, Any], signal: dict[str, Any], backend: str) -> dict[str, Any]:
    """One combination; never raises (status: ok / unsupported / error)."""
    if backend == "tidystl_simd":
        try:
            import tidystl_simd  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        use(tidystl_simd)

    text = spec.get("tidystl")
    if text is None:
        return {"status": "unsupported", "reason": "no tidystl spec"}
    try:
        sig = Signal.from_dict(
            np.asarray(signal["times"], dtype=np.float64),
            {name: np.asarray(vals, dtype=np.float64) for name, vals in signal["values"].items()},
        )
        rho = robustness(parse(text), sig, backend=backend)
        return {"status": "ok", "robustness": [float(v) for v in np.asarray(rho)[0]]}
    except (NotImplementedError, ValueError) as exc:
        # The backend rejects the combination (a coverage finding), as
        # opposed to an unexpected crash.
        return {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - record, never abort the sweep
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter-specs", default=None, metavar="NAMES")
    parser.add_argument("--filter-signals", default=None, metavar="NAMES")
    args = parser.parse_args()

    specs = load_registry(args.specs, args.filter_specs)
    signals = load_registry(args.signals, args.filter_signals)
    results = [
        {"spec": spec_name, "signal": signal_name, **evaluate(spec, signal, args.backend)}
        for spec_name, spec in specs.items()
        for signal_name, signal in signals.items()
    ]
    payload = {
        "backend": args.backend,
        "specs": str(args.specs),
        "signals": str(args.signals),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    statuses = [r["status"] for r in results]
    print(
        f"wrote {args.output}  backend={args.backend} "
        f"({len(specs)} specs x {len(signals)} signals): "
        + ", ".join(f"{s}={statuses.count(s)}" for s in sorted(set(statuses)))
    )


if __name__ == "__main__":
    main()
