"""Candidate-function libraries for sparse regression.

A library maps a state matrix ``X`` of shape (n_samples, n_features) to a
feature matrix ``Theta`` of shape (n_samples, n_terms), along with human-readable
names for each term. Sparse-identification methods then solve ``dX = Theta @ Xi``
for a sparse coefficient matrix ``Xi``.
"""

from __future__ import annotations

import itertools

import numpy as np


def polynomial_library(X, degree=2, include_bias=True):
    """Build a polynomial feature library up to ``degree``.

    Returns
    -------
    Theta : (n_samples, n_terms) feature matrix.
    names : list[str] of term names, using x0, x1, ... for the state variables.
    """
    n_samples, n_features = X.shape
    features, names = [], []

    if include_bias:
        features.append(np.ones(n_samples))
        names.append("1")

    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_features), d):
            features.append(np.prod([X[:, i] for i in combo], axis=0))
            names.append(" ".join(f"x{i}" for i in combo))

    return np.column_stack(features), names
