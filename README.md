# Sparse Model Tutorials

A ~30-minute hands-on demonstration of **sparse optimization methods for data-driven
discovery of differential equations**. The goal: take noisy time-series data and recover the
governing equations as a small, interpretable set of active terms — finding a sparse $\Xi$ in

$$\dot{X} \;\approx\; \Theta(X)\,\Xi .$$

> Built with [`pysindy`](https://pysindy.readthedocs.io) (SINDy variants) and
> [`DaeFinder`](https://pypi.org/project/DaeFinder/) (the reference implementation of SODAs).

## Methods covered

| Method | Idea | Best for |
| --- | --- | --- |
| **Vanilla SINDy** | Sparse regression on a polynomial library using numerically estimated derivatives. | Clean polynomial ODEs. |
| **Weak SINDy** | Integral / weak formulation against test functions — no pointwise derivatives. | High-noise data. |
| **SINDy-PI** | Implicit formulation that admits **rational** dynamics a polynomial library cannot express. | Rational / implicit systems. |
| **SODAs** | Discover **algebraic constraints first**, then the dynamics (sequential convex problems). | Differential–algebraic equations (DAEs). |

## The 30-minute demo arc

| Notebook | Topic | ~min |
| --- | --- | --- |
| `00_overview.ipynb` | The sparse-regression idea, the method/benchmark map, import sanity check. | 5 |
| `01_sindy_intro.ipynb` | Vanilla SINDy on clean Lotka–Volterra; the noise problem appears. | 7 |
| `02_weak_sindy.ipynb` | Weak SINDy + a noise sweep where vanilla fails and the weak form survives. | 6 |
| `02a_weak_form_and_test_functions.ipynb` | **Theory:** strong vs. weak form, the families of test functions (shapes + spectra), and the weak form built from scratch for an **ODE** *and* a **PDE** (heat equation). | 8 |
| `02b_weak_sindy_vs_sindy.ipynb` | Deep dive: a fair, **pre-registered** Lorenz stress-test — from-scratch weak-form math, then weak SINDy vs finite-difference / smoothed / *oracle* SINDy with Wilson CIs, a paired McNemar test, and test-function sensitivity. | 8 |
| `03b_weak_sindy.ipynb` | **Condensed, talk-ready** version of `02b` (one idea, one example, one plot) — presentable live in **under 10 minutes**. | <10 |
| `03_sindy_pi.ipynb` | SINDy-PI on rational Michaelis–Menten kinetics. | 6 |
| `04_sodas_dae.ipynb` | Full SODAs/DaeFinder flow on Michaelis–Menten as a DAE. | 6 |
| `05_benchmark_comparison.ipynb` | Head-to-head: noise robustness + the DAE/rational case + summary. | — |

## Repository layout

```
SparseModelTutorials/
├── README.md
├── PROMPT.md             # the refined build-spec the notebooks were generated from
├── requirements.txt
├── LICENSE
├── data/                 # generated / cached datasets (gitignored)
├── notebooks/            # 00 … 05, executed with outputs populated
└── src/
    └── sparse_demos/     # shared, reusable code
        ├── systems.py    # reference systems + data generation (LV, Lorenz, Michaelis–Menten)
        ├── libraries.py  # polynomial candidate library
        ├── solvers.py    # from-scratch STLSQ + model printing
        ├── metrics.py    # scoring recovered models vs. ground truth
        └── plotting.py   # trajectory / phase / noise-sweep plots
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab        # then open notebooks/00_overview.ipynb
```

The first code cell of every notebook is a **commented-out** install
(`# !pip install -r ../requirements.txt`) so nothing reinstalls while you teach.

## References

- Brunton, Proctor & Kutz (2016), *Discovering governing equations from data: SINDy*. PNAS.
- Messenger & Bortz (2021), *Weak SINDy: Galerkin-Based Data-Driven Model Selection*. Multiscale
  Model. Simul. 19(3) ([arXiv:2005.04339](https://arxiv.org/abs/2005.04339)) — basis for
  `02b_weak_sindy_vs_sindy.ipynb`.
- Messenger & Bortz (2021), *Weak SINDy for PDEs*. J. Comput. Phys.
- Kaheman, Kutz & Brunton (2020), *SINDy-PI: robust algorithm for parallel implicit sparse
  identification of nonlinear dynamics*. Proc. R. Soc. A.
- Jayadharan, Catlett, Montanari & Mangan (2026), *SODAs: Sparse Optimization for the
  Discovery of Differential and Algebraic equations*. Proc. R. Soc. A
  ([arXiv:2503.05993](https://arxiv.org/abs/2503.05993)).

## License

MIT — see [LICENSE](LICENSE).
