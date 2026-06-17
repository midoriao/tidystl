"""ODE benchmark models for E5a falsification.

Each model is a frozen dataclass with scalar dynamics ``rhs`` and an output
map ``output``. A piecewise-constant throttle vector ``u`` (length
``n_segments``) over ``[0, t_end]`` is the search space. ``simulate`` returns
the output variable on a monitor grid; ``simulate_dense`` returns it on a
refinement of that grid for the backend-neutral oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class OdeModel:
    """A throttle-driven ODE benchmark on a fixed monitor grid."""

    name: str
    t_end: float
    dt: float
    n_segments: int
    u_lo: float
    u_hi: float
    output_var: str
    dense_factor: int = 32

    def state0(self) -> NDArray[np.floating]:
        raise NotImplementedError

    def rhs(self, t: float, state: NDArray[np.floating], u: float) -> NDArray[np.floating]:
        raise NotImplementedError

    def output(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        """Map a (S, T) state trajectory to the (T,) output variable."""
        raise NotImplementedError


@dataclass(frozen=True)
class SpeedModel(OdeModel):
    """v' = k*a - c*v - d*v^2 ; a' = (u - a)/tau  (quadratic drag, actuator lag)."""

    k: float = 2.0
    c: float = 0.15
    d: float = 0.05
    tau: float = 0.4

    def state0(self) -> NDArray[np.floating]:
        return np.array([0.0, 0.0])  # [v, a]

    def rhs(self, t, state, u):
        v, a = state
        return np.array([self.k * a - self.c * v - self.d * v * v, (u - a) / self.tau])

    def output(self, state):
        return state[0]  # v


@dataclass(frozen=True)
class MassSpringModel(OdeModel):
    """x'' + 2*zeta*wn*x' + wn^2*x = u(t)."""

    wn: float = 2.0
    zeta: float = 0.15

    def state0(self) -> NDArray[np.floating]:
        return np.array([0.0, 0.0])  # [x, x']

    def rhs(self, t, state, u):
        x, xd = state
        return np.array([xd, u - 2.0 * self.zeta * self.wn * xd - self.wn * self.wn * x])

    def output(self, state):
        return state[0]  # x


@dataclass(frozen=True)
class CoupledModel(OdeModel):
    """Two-tank-like coupled nonlinear flow; output is the second state."""

    k1: float = 0.5
    k2: float = 0.4

    def state0(self) -> NDArray[np.floating]:
        return np.array([0.0, 0.0])  # [h1, h2]

    def rhs(self, t, state, u):
        h1, h2 = state
        f1 = self.k1 * np.sqrt(max(h1, 0.0))
        f2 = self.k2 * np.sqrt(max(h2, 0.0))
        return np.array([u - f1, f1 - f2])

    def output(self, state):
        return state[1]  # h2


def monitor_times(model: OdeModel, dt: float) -> NDArray[np.floating]:
    """Uniform grid 0, dt, ..., t_end (t_end included)."""
    n = int(round(model.t_end / dt))
    return np.linspace(0.0, model.t_end, n + 1)


def _integrate(model: OdeModel, u: NDArray[np.floating], times: NDArray[np.floating]) -> NDArray[np.floating]:
    """Segment-wise integration (constant throttle per segment) sampled at ``times``."""
    seg_len = model.t_end / model.n_segments
    out = np.empty((model.state0().shape[0], times.shape[0]))
    state = model.state0()
    for k in range(model.n_segments):
        t0, t1 = k * seg_len, (k + 1) * seg_len
        in_seg = (times >= t0 - 1e-12) & (times <= t1 + 1e-12)
        # Always include the right boundary so the next segment starts correctly.
        eval_pts = times[in_seg]
        sol = solve_ivp(
            lambda t, s, uk=float(u[k]): model.rhs(t, s, uk),
            (t0, t1),
            state,
            t_eval=np.unique(np.concatenate([[t0], eval_pts, [t1]])),
            rtol=1e-9,
            atol=1e-12,
            max_step=seg_len / 4.0,
        )
        # record the sampled points that belong to this segment
        for j, tt in enumerate(times):
            if t0 - 1e-12 <= tt <= t1 + 1e-12:
                col = int(np.argmin(np.abs(sol.t - tt)))
                out[:, j] = sol.y[:, col]
        state = sol.y[:, -1]  # state at t1 carries to next segment
    return out


def simulate(model: OdeModel, u: NDArray[np.floating], times: NDArray[np.floating]) -> NDArray[np.floating]:
    """Output variable sampled at ``times`` for throttle ``u``."""
    return model.output(_integrate(model, u, times))


def simulate_dense(
    model: OdeModel, u: NDArray[np.floating], dt: float
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Output on a ``dense_factor``-refinement of the monitor grid (for the oracle)."""
    n = int(round(model.t_end / dt)) * model.dense_factor
    dense_t = np.linspace(0.0, model.t_end, n + 1)
    return dense_t, model.output(_integrate(model, u, dense_t))


MODELS: dict[str, OdeModel] = {
    "m1_speed": SpeedModel(
        name="m1_speed", t_end=8.0, dt=0.5, n_segments=4, u_lo=0.0, u_hi=1.0, output_var="v"
    ),
    "m2_mass_spring": MassSpringModel(
        name="m2_mass_spring", t_end=6.0, dt=0.5, n_segments=3, u_lo=-1.0, u_hi=1.0, output_var="x"
    ),
    "m3_coupled": CoupledModel(
        name="m3_coupled", t_end=8.0, dt=0.5, n_segments=4, u_lo=0.0, u_hi=1.0, output_var="h2"
    ),
}
