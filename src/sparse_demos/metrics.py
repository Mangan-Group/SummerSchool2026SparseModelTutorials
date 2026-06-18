"""Metrics for scoring discovered models against ground truth.

Used mainly by the benchmark / noise-sweep notebook to quantify when a method
has recovered the correct equations.
"""

from __future__ import annotations

import numpy as np


def coef_l2_error(true_xi, est_xi):
    """Relative L2 error between true and estimated coefficient matrices."""
    true_xi = np.asarray(true_xi, dtype=float)
    est_xi = np.asarray(est_xi, dtype=float)
    denom = np.linalg.norm(true_xi)
    return float(np.linalg.norm(est_xi - true_xi) / denom) if denom else float("nan")


def support_match(true_xi, est_xi, threshold=1e-6):
    """Fraction of terms whose active/inactive status matches ground truth.

    A perfect structural recovery scores 1.0 regardless of coefficient values.
    """
    true_active = np.abs(np.asarray(true_xi)) > threshold
    est_active = np.abs(np.asarray(est_xi)) > threshold
    return float(np.mean(true_active == est_active))


def is_correct_model(true_xi, est_xi, support_threshold=1e-6, rel_tol=0.20):
    """True iff the estimated model has the right support AND coefficients within
    ``rel_tol`` relative L2 error of the truth."""
    return (support_match(true_xi, est_xi, support_threshold) == 1.0
            and coef_l2_error(true_xi, est_xi) <= rel_tol)
