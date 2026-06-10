from __future__ import annotations

import numpy as np

from tidystl import Signal, parse, robustness


def main() -> None:
    spec = """
    G[0,10](F[0,2](velocity >= 12))
    """

    times = np.linspace(0.0, 10.0, 200, dtype=float)
    velocity = (10.0 + 4.0 * np.sin(times))[np.newaxis, :]
    signal = Signal.from_dict(times=times, values={"velocity": velocity})
    formula = parse(spec)
    rho = robustness(formula, signal, backend="breach")

    print("rho shape:", rho.shape)
    print("rho(t=0):", float(rho[0, 0]))
    print("min rho:", float(np.min(rho)))


if __name__ == "__main__":
    main()
