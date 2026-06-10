"""Falsification demo: semantics determines what the optimizer can see.

System:         Vehicle maintaining speed >= 5 m/s.
                Cruises at 7 m/s, then brakes at the final sample.
Model:          v(t) = 7  for t in {0, 2.5, 5, 7.5}
                v(10) = 7 + a * dt,  dt = 2.5 s,  a <= 0  (braking force)
Specification:  (speed >= 5) and (heading >= -10)
Search:         Coordinate descent over braking force a in [-2, 0].

NativeBackend evaluates every timestep; the objective decreases as |a| grows,
giving the optimizer a clear descent direction to a falsifying trace.

BreachBackend copies the penultimate robustness to the final timestep for
top-level AND/OR formulas.  Since the penultimate speed is
fixed at 7, the Breach objective is *constant* regardless of braking force.
The optimizer using Breach cannot perceive any effect of braking -- it is
completely blind to what happens at the final sample.

Usage:
    uv run python examples/falsification_demo.py
"""

from __future__ import annotations

import numpy as np

from tidystl import Signal, parse, robustness

PHI = parse("(speed >= 5) and (heading >= -10)")
T = 5
DT = 2.5  # sample interval (s)
times = np.arange(T, dtype=float) * DT  # [0, 2.5, 5, 7.5, 10]
CRUISE = 7.0  # cruising speed (m/s)


def simulate(a: float) -> np.ndarray:
    """Apply braking force a at the final sample; cruise otherwise.

    v(t) = CRUISE       for t in {0, 2.5, 5, 7.5}
    v(10) = CRUISE + a * DT   (a <= 0 means braking)
    """
    speed = np.full(T, CRUISE)
    speed[-1] = CRUISE + a * DT
    return speed[np.newaxis, :]


def objective(a: float, backend: str) -> float:
    """Worst-case robustness over the trajectory produced by braking force a."""
    sig = Signal.from_dict(
        times=times,
        values={"speed": simulate(a), "heading": np.zeros((1, T))},
    )
    return float(robustness(PHI, sig, backend=backend).min())


# --- 1. Landscape scan ---
print("=== Falsification demo: v(t) >= 5, braking model ===")
print(f"Cruise at {CRUISE} m/s; brake only at t=10. Spec: (speed >= 5) and (heading >= -10)\n")

scan = np.linspace(-2.0, 0.0, 9)
print(f"  {'a (brake)':>10}  {'v(10)':>6}  {'rho (native)':>13}  {'rho (breach)':>13}")
print("  " + "-" * 50)
for a in scan:
    v_final = CRUISE + a * DT
    rn = objective(a, "native")
    rb = objective(a, "breach")
    mark = "  <- diverge" if abs(rn - rb) > 1e-9 else ""
    print(f"  {a:>10.2f}  {v_final:>6.2f}  {rn:>13.3f}  {rb:>13.3f}{mark}")

print(f"\nBreach objective: constant {objective(0.0, 'breach'):.1f} for all a (final step masked).")
print("Native objective: smooth descent; crosses 0 at a = -0.8 (v(10) = 5.0).\n")

# --- 2. Coordinate-descent falsification ---
print("--- Coordinate-descent (start: a = 0, no braking; step = 0.1) ---")

for backend in ("native", "breach"):
    x, step = 0.0, 0.1
    for _ in range(60):
        candidate = x - step  # try increasing braking (more negative a)
        if candidate < -2.0:  # hard lower bound
            step *= 0.5
            continue
        if objective(candidate, backend) < objective(x, backend) - 1e-9:
            x = candidate
        else:
            step *= 0.5
            if step < 1e-5:
                break

    rho_b = objective(x, backend)
    rho_n = objective(x, "native")
    verdict = "FALSIFIED" if rho_n < 0 else "not falsified"
    extra = (
        "\n  !! Breach is blind to braking: objective is flat, no descent direction."
        if rho_n >= 0
        else ""
    )
    print(f"\n{backend}:")
    print(f"  converged at a = {x:.4f}  (v(10) = {CRUISE + x * DT:.3f} m/s)")
    print(f"  rho ({backend:<6}) = {rho_b:+.3f}")
    print(f"  rho (native)  = {rho_n:+.3f}  [{verdict}]{extra}")
