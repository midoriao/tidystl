"""Random-trace agreement baseline for E5a.

Draws random inputs to the hero model, evaluates every tool-compatible backend's
robustness at t=0 on the hero spec, and reports -- as a TeX table -- the nominal
falsifying fraction and how often / how much each backend diverges from a
reference (breach). The point: under random sampling the backends agree on the
verdict (sign), so the search-time divergence seen elsewhere is boundary-localized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

import models  # noqa: E402
import specs  # noqa: E402
import tidystl_compat  # noqa: F401,E402  -- registers compat backends

from tidystl import Signal, parse, robustness  # noqa: E402

BACKENDS = ("breach", "rtamt", "rtamt_dense", "pymtl", "taliro", "stlcgpp")
REFERENCE = "breach"
HERO = "m2_mass_spring"


def scan(n_samples: int, seed: int) -> dict[str, np.ndarray]:
    bench = specs.BENCHMARKS[HERO]
    m = bench.model
    phi = parse(bench.spec())
    times = models.monitor_times(m, m.dt)
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {b: [] for b in BACKENDS}
    for _ in range(n_samples):
        u = rng.uniform(m.u_lo, m.u_hi, m.n_segments)
        y = models.simulate(m, u, times)
        s = Signal.from_dict(times, {m.output_var: y})
        for b in BACKENDS:
            vals[b].append(float(np.asarray(robustness(phi, s, backend=b))[0, 0]))
    return {b: np.asarray(v) for b, v in vals.items()}


def to_tex(arr: dict[str, np.ndarray]) -> str:
    base = arr[REFERENCE]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"backend & $P(\rho<0)$ & sign-agree & max $|\Delta\rho|$ \\",
        r"\midrule",
    ]
    for b in BACKENDS:
        d = arr[b]
        negfrac = float(np.mean(d < 0))
        agree = float(np.mean(np.sign(d) == np.sign(base)))
        maxdiff = float(np.max(np.abs(d - base)))
        lines.append(f"{b} & {negfrac:.2f} & {agree:.3f} & {maxdiff:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="write the TeX table here")
    args = parser.parse_args()
    arr = scan(args.n_samples, args.seed)
    tex = to_tex(arr)
    if args.output is not None:
        args.output.write_text(tex)
        print(f"wrote {args.output}")
    print(tex)


if __name__ == "__main__":
    main()
