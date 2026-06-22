"""From-scratch Weak (integral / Galerkin) SINDy.

A faithful, dependency-light reproduction of the method in

    D. A. Messenger & D. M. Bortz, "Weak SINDy: Galerkin-Based Data-Driven
    Model Selection," Multiscale Model. Simul. 19(3), 1474-1497, 2021.
    (arXiv:2005.04339)

The original reference implementation is in MATLAB; this is a compact NumPy
re-implementation meant to be read top-to-bottom in a teaching notebook.

The idea
--------
Vanilla SINDy must estimate the derivative ``x'`` pointwise from noisy data --
finite differences amplify high-frequency noise by ~ ``1/dt``.  The weak form
removes that step.  Multiply the dynamics ``x_d' = sum_j w_{jd} f_j(x)`` by a
smooth, compactly supported test function ``phi_k`` and integrate over its
support; integrating the left side by parts moves the derivative onto the
*known, analytic* test function:

    integral( phi_k * x_d' ) dt  =  -integral( phi_k' * x_d ) dt        (phi_k = 0 at the ends)

so the weak-form regression problem is

    G w_d = b_d ,   G_{kj} = integral( phi_k * f_j(x) ) dt ,
                    b_{kd} = -integral( phi_k' * x_d ) dt .

Every integral is a plain quadrature against the *measured* states -- no
derivative of the data is ever taken.  Zero-mean measurement noise is averaged
down by the integral (variance reduction), and ``phi_k`` acts as a smooth
low-pass filter whose derivative ``phi_k'`` is computed analytically.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.signal import savgol_filter

from .libraries import polynomial_library
from .solvers import stlsq


# --------------------------------------------------------------------------- #
# Test functions (the "weight" / "Galerkin" functions)
# --------------------------------------------------------------------------- #
def bump_test_function(t, center, radius, p):
    r"""Piecewise-polynomial bump and its analytic derivative.

    .. math:: \phi(t) = (1 - \tau^2)^p \ \text{for}\ |\tau| < 1,\quad
              \tau = (t - c)/r,\quad \phi = 0 \ \text{otherwise.}

    This is the symmetric (``p = q``) case of the paper's
    ``C (t-a)^p (b-t)^q`` family, normalised so ``max phi = 1``.  It is
    ``C^{p-1}`` and vanishes at the support boundary, so the integration-by-parts
    boundary term ``[phi * x]`` is zero for any ``p >= 1``.

    Returns
    -------
    phi, phidot : arrays the same shape as ``t``.
    """
    tau = (t - center) / radius
    inside = np.abs(tau) < 1.0
    phi = np.zeros_like(t, dtype=float)
    phidot = np.zeros_like(t, dtype=float)
    s = tau[inside]
    phi[inside] = (1.0 - s**2) ** p
    # d/dt (1 - tau^2)^p = p (1 - tau^2)^(p-1) * (-2 tau) * (1/r)
    phidot[inside] = p * (1.0 - s**2) ** (p - 1) * (-2.0 * s) / radius
    return phi, phidot


def test_function_matrices(t, K=100, p=8, support_radius=None, support_fraction=0.1):
    r"""Build the test-function and derivative matrices with quadrature folded in.

    ``K`` translated copies of the bump are centred uniformly across the interior
    of ``t`` so each support lies inside the time window.  Trapezoidal-rule
    weights are folded into the rows, so that for any sampled signal ``g``::

        (Phi @ g)[k]    ~= integral( phi_k * g ) dt
        (Phidot @ g)[k] ~= integral( phi_k' * g ) dt

    Parameters
    ----------
    t : (M,) uniform time grid.
    K : number of test functions.
    p : bump exponent (smoothness / spectral concentration).
    support_radius : half-width ``r`` of each bump, in time units.  If ``None``,
        uses ``support_fraction * (t[-1] - t[0])``.

    Returns
    -------
    Phi, Phidot : (K, M) arrays.   centers : (K,) array of bump centres.
    """
    t = np.asarray(t, dtype=float)
    M = t.size
    t0, t1 = t[0], t[-1]
    dt = (t1 - t0) / (M - 1)

    if support_radius is None:
        support_radius = support_fraction * (t1 - t0)
    r = float(support_radius)

    centers = np.linspace(t0 + r, t1 - r, K)

    # trapezoidal quadrature weights
    quad = np.full(M, dt)
    quad[0] *= 0.5
    quad[-1] *= 0.5

    Phi = np.zeros((K, M))
    Phidot = np.zeros((K, M))
    for k, c in enumerate(centers):
        phi, phidot = bump_test_function(t, c, r, p)
        Phi[k] = phi * quad
        Phidot[k] = phidot * quad
    return Phi, Phidot, centers


# --------------------------------------------------------------------------- #
# Weak-form linear system and fit
# --------------------------------------------------------------------------- #
def weak_linear_system(t, Y, degree=2, include_bias=True,
                       K=100, p=8, support_radius=None, support_fraction=0.1):
    """Assemble the weak-form regression system ``G w = b``.

    ``G = Phi @ Theta(Y)`` (library evaluated at the data, NO derivatives) and
    ``b = -Phidot @ Y``.  Returns ``(G, b, names, info)`` so a notebook can show
    the matrices explicitly.
    """
    Y = np.asarray(Y, dtype=float)
    Theta, names = polynomial_library(Y, degree=degree, include_bias=include_bias)
    Phi, Phidot, centers = test_function_matrices(
        t, K=K, p=p, support_radius=support_radius, support_fraction=support_fraction)
    G = Phi @ Theta            # (K, n_terms)
    b = -Phidot @ Y            # (K, n_states)
    info = {"centers": centers, "Phi": Phi, "Phidot": Phidot, "Theta": Theta}
    return G, b, names, info


def gls_whiten(G, b, Phidot, ridge=1e-8):
    r"""Generalized-least-squares whitening (the variance-reduction step in
    Messenger & Bortz).

    To first order the noise in ``b = -Phidot @ Y`` has covariance
    ``sigma^2 * Phidot @ Phidot.T`` -- correlated across overlapping test
    functions.  Whitening both sides by ``Sigma^{-1/2}`` (``Sigma = L L^T``)
    turns the correlated/heteroscedastic problem into one with ~white residuals,
    which sharpens the sparse regression.  Returns the whitened ``(G, b)``.
    """
    K = Phidot.shape[0]
    Sigma = Phidot @ Phidot.T
    Sigma = Sigma + (ridge * np.trace(Sigma) / K) * np.eye(K)
    L = cholesky(Sigma, lower=True)
    return solve_triangular(L, G, lower=True), solve_triangular(L, b, lower=True)


def weak_sindy(t, Y, degree=2, include_bias=True, threshold=0.5, max_iter=10,
               K=100, p=8, support_radius=None, support_fraction=0.1, gls=False):
    """Fit weak-form SINDy.  Returns ``(Xi, names)`` with ``Xi`` of shape
    ``(n_terms, n_states)``.  ``gls=True`` applies the covariance whitening above
    (the full Messenger-Bortz weak SINDy); ``gls=False`` is the unweighted core."""
    G, b, names, info = weak_linear_system(
        t, Y, degree=degree, include_bias=include_bias,
        K=K, p=p, support_radius=support_radius, support_fraction=support_fraction)
    if gls:
        G, b = gls_whiten(G, b, info["Phidot"])
    Xi = stlsq(G, b, threshold=threshold, max_iter=max_iter)
    return Xi, names


# --------------------------------------------------------------------------- #
# Conventional ("vanilla") SINDy baseline -- given its best shot
# --------------------------------------------------------------------------- #
def smoothed_derivative(Y, dt, window=None, polyorder=3):
    """Savitzky-Golay smoothed first derivative (a strong, fair derivative
    estimate for vanilla SINDy -- much better than naive finite differences)."""
    Y = np.asarray(Y, dtype=float)
    M = Y.shape[0]
    if window is None:
        window = max(polyorder + 2, (M // 50) | 1)   # odd, ~2% of the series
    window = min(window, M if M % 2 == 1 else M - 1)
    if window % 2 == 0:
        window += 1
    return savgol_filter(Y, window_length=window, polyorder=polyorder,
                         deriv=1, delta=dt, axis=0)


def finite_difference_derivative(Y, dt):
    """Second-order centred finite differences (the naive baseline)."""
    return np.gradient(np.asarray(Y, dtype=float), dt, axis=0)


def savgol_smooth(Y, window=None, polyorder=3):
    """Savitzky-Golay smoothing of the states themselves (deriv=0)."""
    Y = np.asarray(Y, dtype=float)
    M = Y.shape[0]
    if window is None:
        window = max(polyorder + 2, (M // 50) | 1)
    window = min(window, M if M % 2 == 1 else M - 1)
    if window % 2 == 0:
        window += 1
    return savgol_filter(Y, window_length=window, polyorder=polyorder, deriv=0, axis=0)


def vanilla_sindy(t, Y, degree=2, include_bias=True, threshold=0.5, max_iter=10,
                  derivative="savgol", window=None, polyorder=3, smooth_states=False):
    """Conventional SINDy: estimate ``Y'`` then sparse-regress on the library.

    ``derivative`` is ``"savgol"`` (smoothed, the fair baseline) or ``"fd"``
    (naive finite differences).  ``smooth_states=True`` also evaluates the
    library on Savitzky-Golay-denoised states (mitigates errors-in-variables) --
    use it to give the conventional baseline its strongest shot.  Returns
    ``(Xi, names)``.
    """
    Y = np.asarray(Y, dtype=float)
    dt = (t[-1] - t[0]) / (len(t) - 1)
    if derivative == "savgol":
        dY = smoothed_derivative(Y, dt, window=window, polyorder=polyorder)
    elif derivative == "fd":
        dY = finite_difference_derivative(Y, dt)
    else:
        raise ValueError(f"unknown derivative method {derivative!r}")
    Y_lib = savgol_smooth(Y, window=window, polyorder=polyorder) if smooth_states else Y
    Theta, names = polynomial_library(Y_lib, degree=degree, include_bias=include_bias)
    Xi = stlsq(Theta, dY, threshold=threshold, max_iter=max_iter)
    return Xi, names
