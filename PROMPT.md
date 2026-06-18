# Build Prompt — Sparse Optimization Tutorial for DE Discovery

> This is the refined specification used to build the tutorial notebooks in this repo.
> It is written as a self-contained prompt: given this document and the `src/sparse_demos`
> helpers, an agent (or a person) should be able to reproduce the full tutorial.
>
> *(Note: this prompt was refined by a manual critique pass — the `gpt-critique-loop` skill
> was unavailable in the build environment.)*

## Goal

Produce a hands-on, ~30-minute teaching tutorial on **data-driven discovery of differential
equations via sparse optimization**: take noisy time-series from a dynamical system and recover
its governing equations as a small, interpretable set of active terms. The tutorial must be a
set of runnable Jupyter notebooks that an attendee can step through live.

## Audience & framing

Researchers/students who know ODEs and basic regression but not necessarily SINDy. Each method
is motivated by a concrete failure of the previous one:

- **Vanilla SINDy** works on clean polynomial dynamics but is fragile to noise (it needs
  pointwise derivative estimates) and cannot represent rational/implicit terms.
- **Weak SINDy** removes the derivative-estimation step via an integral/weak formulation →
  orders-of-magnitude better noise robustness.
- **SINDy-PI** handles *implicit* and *rational* dynamics that no polynomial library can express.
- **SODAs** discovers **differential–algebraic equations (DAEs)**: it first finds the algebraic
  constraints, then the dynamics, as a sequence of convex problems — preserving physical structure.

## Methods (exactly these four)

1. Vanilla SINDy — `pysindy` (`PolynomialLibrary` + `STLSQ`).
2. Weak SINDy — `pysindy` (`WeakPDELibrary`, ODE configuration).
3. SINDy-PI — `pysindy` (`SINDyPI` optimizer; requires `cvxpy`).
4. SODAs — `DaeFinder` (`PolyFeatureMatrix` → `AlgModelFinder` → `get_refined_lib` → `sequentialThLin`).

## Benchmarks (both)

- **Benchmark A — polynomial ODE** (Lotka–Volterra; Lorenz as an aside). Clean recovery, then a
  **noise sweep** for the vanilla-vs-weak head-to-head.
- **Benchmark B — Michaelis–Menten enzyme kinetics** (a DAE with rational/QSSA structure). The
  case where vanilla polynomial SINDy *cannot* succeed and SINDy-PI / SODAs can.

## Notebooks (in `notebooks/`)

| # | File | Content | ~min |
|---|------|---------|------|
| 0 | `00_overview.ipynb` | Sparse-regression idea `Ẋ = Θ(X)Ξ`, the demo arc, method/benchmark map, import sanity check. | 5 |
| 1 | `01_sindy_intro.ipynb` | Vanilla SINDy on clean Lotka–Volterra: library → STLSQ → recovered vs true model. | 7 |
| 2 | `02_weak_sindy.ipynb` | Weak SINDy + a **noise sweep** showing vanilla SINDy failing while the weak form survives. | 6 |
| 3 | `03_sindy_pi.ipynb` | SINDy-PI on Michaelis–Menten rational kinetics; show polynomial SINDy cannot represent the terms. | 6 |
| 4 | `04_sodas_dae.ipynb` | Full SODAs/DaeFinder flow on MM as a DAE: algebraic constraint then dynamics. | 6 |
| 5 | `05_benchmark_comparison.ipynb` | Benchmark A noise-robustness plot; Benchmark B equation/correctness table; summary matrix. | — |

## Hard constraints

- **Installs stay inactive.** Every notebook's first code cell is a *commented* install
  (`# !pip install -r ../requirements.txt`). Nothing auto-installs on run.
- Reusable logic lives in `src/sparse_demos/` (systems, metrics, plotting), imported by the
  notebooks — keep notebook cells about *teaching*, not plumbing.
- SODAs code must use the **real DaeFinder API**, not a reimplementation.
- Each notebook: markdown narrative → commented install → runnable code with saved outputs.

## Success criteria

- Notebook 1 recovers Lotka–Volterra coefficients within tolerance on clean data.
- Notebook 2 exhibits a noise level where vanilla SINDy's recovered model is wrong but weak
  SINDy's is still correct.
- Notebooks 3–4 recover the MM rational dynamics / algebraic constraint.
- Notebook 5's comparison plot and table populate and tell a coherent story.

## Testing requirement

Build a virtual environment (`.venv`), install `requirements.txt`, and execute every notebook
headless (`jupyter nbconvert --execute`) so outputs are populated and errors surface before
delivery. The venv is for testing only; in-notebook installs remain commented.
