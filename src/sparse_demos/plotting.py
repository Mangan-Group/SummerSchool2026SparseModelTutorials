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
                     title="Noise robustness", ax=None, logy=True):
    """Plot a metric vs. noise level, one line per method.

    Parameters
    ----------
    noise_levels : sequence of noise magnitudes (x-axis).
    errors_by_method : dict {method_name: sequence of metric values}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, errs in errors_by_method.items():
        ax.plot(noise_levels, errs, marker="o", label=name)
    ax.set_xlabel("noise level")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_success_vs_noise(noise_levels, success_by_method, ci_by_method=None,
                          title="Exact-recovery rate vs. noise", ax=None,
                          xlabel=r"noise"):
    """Plot exact-support recovery rate vs. noise, one line per method, with
    optional shaded Wilson confidence bands.

    Parameters
    ----------
    noise_levels : x-axis noise ratios.
    success_by_method : {name: sequence of recovery rates in [0, 1]}.
    ci_by_method : optional {name: sequence of (lo, hi) Wilson intervals}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, succ in success_by_method.items():
        line, = ax.plot(noise_levels, succ, marker="o", label=name)
        if ci_by_method and name in ci_by_method:
            lo = [c[0] for c in ci_by_method[name]]
            hi = [c[1] for c in ci_by_method[name]]
            ax.fill_between(noise_levels, lo, hi, color=line.get_color(), alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("exact-support recovery rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

def plot_success_vs_noise_2(noise_levels, success_by_method, ci_by_method=None,
                          title="Exact-recovery rate vs. noise", ax=None,
                          xlabel=r"noise"):
    """Plot exact-support recovery rate vs. noise, one line per method, with
    optional Wilson confidence error bars.

    Parameters
    ----------
    noise_levels : x-axis noise levels.
    success_by_method : {name: sequence of recovery rates in [0, 1]}.
    ci_by_method : optional {name: sequence of (lo, hi) Wilson intervals}.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, succ in success_by_method.items():
        if ci_by_method and name in ci_by_method:
            lo = [c[0] for c in ci_by_method[name]]
            hi = [c[1] for c in ci_by_method[name]]
            yerr = [[s - l for s, l in zip(succ, lo)],   # distance down to lo
                    [h - s for s, h in zip(succ, hi)]]   # distance up to hi
            ax.errorbar(noise_levels, succ, yerr=yerr, marker="o",
                        capsize=3, label=name)
        else:
            ax.plot(noise_levels, succ, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("exact-support recovery rate")
    ax.set_ylim(-0.03, 1.03)
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
