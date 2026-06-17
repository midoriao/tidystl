"""Backend-neutral falsification oracle.

A claim is confirmed iff the formula's principled (native PL) robustness on a
dense refinement of the monitor grid is below -margin at t=0. native is the
dense reference, deliberately excluded from the compared conditions.
"""

from __future__ import annotations

import models
import numpy as np
from numpy.typing import NDArray

from tidystl import Signal, robustness
from tidystl.core.nodes import Node

ORACLE_BACKEND = "native"
VIOLATION_MARGIN = 1e-6


def is_true_violation(
    model: models.OdeModel,
    u: NDArray[np.floating],
    phi: Node,
    dt: float,
) -> bool:
    dense_t, dense_y = models.simulate_dense(model, u, dt)
    signal = Signal.from_dict(dense_t, {model.output_var: dense_y})
    rho0 = float(np.asarray(robustness(phi, signal, backend=ORACLE_BACKEND))[0, 0])
    return rho0 < -VIOLATION_MARGIN
