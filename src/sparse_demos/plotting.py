"""Shared visualization helpers for the tutorials."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_trajectories(t, X, labels=None, title=None, ax=None):
    """Plot each state variable against time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    labels = labels or [f"x{i}" for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        ax.plot(t, X[:, i], label=labels[i])
    ax.set_xlabel("t")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_noise_sweep(noise_levels, errors_by_method, ylabel="relative coefficient error",
                     title="Noise robustness", ax=None, logy=True,
                     xlabel="noise level (% of signal std)"):
    """Plot a metric vs. noise level, one line per method.

    Parameters
    ----------
    noise_levels : sequence of noise magnitudes (x-axis), as a percentage of the
        signal standard deviation.
    errors_by_method : dict {method_name: sequence of metric values}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, errs in errors_by_method.items():
        ax.plot(noise_levels, errs, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_phase(X, title=None, ax=None):
    """2D phase portrait of the first two state variables."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X[:, 0], X[:, 1], lw=0.8)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    if title:
        ax.set_title(title)
    return ax
