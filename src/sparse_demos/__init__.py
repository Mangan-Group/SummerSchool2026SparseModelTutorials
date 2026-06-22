"""Shared utilities for the sparse model discovery tutorials.

Submodules
----------
systems    Reference dynamical systems and data generation.
libraries  Candidate-function libraries for sparse regression.
solvers    Sparse regression solvers (e.g. STLSQ).
metrics    Scoring discovered models against ground truth.
plotting   Shared visualization helpers.
weak       From-scratch weak (integral) SINDy + conventional-SINDy baselines.
"""

__all__ = ["systems", "libraries", "solvers", "metrics", "plotting", "weak"]
__version__ = "0.1.0"
