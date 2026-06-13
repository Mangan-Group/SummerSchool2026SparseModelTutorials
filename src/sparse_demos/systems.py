"""Reference dynamical systems and data-generation helpers.

These provide ground-truth trajectories (optionally with measurement noise) for
benchmarking the sparse-identification methods in the tutorials.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def lotka_volterra(t, state, alpha=1.0, beta=0.1, delta=0.075, gamma=1.5):
    """Predator-prey dynamics. State = [prey, predator]."""
    x, y = state
    return [alpha * x - beta * x * y, delta * x * y - gamma * y]


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz attractor. State = [x, y, z]."""
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def michaelis_menten_reduced(t, state, vmax=1.0, km=0.5):
    """Reduced (QSSA) Michaelis-Menten kinetics. State = [substrate S, product P].

    The substrate is depleted at the rational rate ``vmax * S / (km + S)`` -- a
    term that no finite polynomial library can represent exactly, which is why
    vanilla SINDy fails here and SINDy-PI / SODAs are needed.
    """
    s, _p = state
    rate = vmax * s / (km + s)
    return [-rate, rate]


SYSTEMS = {
    "lotka_volterra": (lotka_volterra, [10.0, 5.0]),
    "lorenz": (lorenz, [-8.0, 8.0, 27.0]),
}


def simulate_mm(t_span=(0.0, 8.0), dt=0.05, s0=1.0, vmax=1.0, km=0.5,
                noise_std=0.0, seed=0):
    """Integrate reduced Michaelis-Menten kinetics; optionally add Gaussian noise.

    Returns ``(t, X)`` with ``X`` columns ``[S, P]``. Ground-truth implicit form:
    ``(km + S) * dS/dt = -vmax * S``.
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(michaelis_menten_reduced, t_span, [s0, 0.0], t_eval=t_eval,
                    args=(vmax, km), rtol=1e-9, atol=1e-9)
    X = sol.y.T
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        X = X + rng.normal(scale=noise_std, size=X.shape)
    return sol.t, X


def simulate(name, t_span=(0.0, 20.0), dt=0.01, x0=None, noise_std=0.0, seed=0, **kwargs):
    """Integrate a named reference system and optionally add Gaussian noise.

    Returns
    -------
    t : (n,) array of time points.
    X : (n, d) array of (noisy) states.
    """
    if name not in SYSTEMS:
        raise KeyError(f"Unknown system {name!r}; choose from {list(SYSTEMS)}")
    rhs, default_x0 = SYSTEMS[name]
    x0 = default_x0 if x0 is None else x0

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, args=tuple(kwargs.values()),
                    rtol=1e-9, atol=1e-9)
    X = sol.y.T

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        X = X + rng.normal(scale=noise_std, size=X.shape)

    return sol.t, X
