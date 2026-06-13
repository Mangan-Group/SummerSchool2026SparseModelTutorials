"""Sparse regression solvers shared across the tutorials."""

from __future__ import annotations

import numpy as np


def stlsq(Theta, dX, threshold=0.025, max_iter=10):
    """Sequentially Thresholded Least Squares (the core SINDy solver).

    Solves ``dX = Theta @ Xi`` for a sparse ``Xi`` by repeatedly running least
    squares and zeroing out coefficients below ``threshold``.

    Parameters
    ----------
    Theta : (n_samples, n_terms) feature/library matrix.
    dX : (n_samples, n_targets) target derivatives.
    threshold : coefficient magnitude below which a term is pruned.
    max_iter : number of thresholding sweeps.

    Returns
    -------
    Xi : (n_terms, n_targets) sparse coefficient matrix.
    """
    Theta = np.atleast_2d(Theta)
    dX = np.atleast_2d(dX)
    if dX.shape[0] != Theta.shape[0]:
        dX = dX.T

    Xi, *_ = np.linalg.lstsq(Theta, dX, rcond=None)

    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for j in range(dX.shape[1]):
            big = ~small[:, j]
            if big.any():
                Xi[big, j], *_ = np.linalg.lstsq(Theta[:, big], dX[:, j], rcond=None)

    return Xi


def print_model(Xi, names, target_names=None):
    """Pretty-print a discovered model from a coefficient matrix."""
    n_targets = Xi.shape[1]
    target_names = target_names or [f"d/dt x{j}" for j in range(n_targets)]
    for j in range(n_targets):
        terms = [f"{Xi[i, j]:+.3f} {names[i]}" for i in range(len(names)) if Xi[i, j] != 0]
        rhs = " ".join(terms) if terms else "0"
        print(f"{target_names[j]} = {rhs}")
