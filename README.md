# Sparse Model Tutorials

A 30-minute hands-on demonstration of **sparse optimization methods for data-driven
discovery of dynamical systems**. The goal is to take noisy time-series (or
spatiotemporal) data and recover the governing equations using a small, interpretable
set of active terms.

## Methods covered

| Method | Idea | Best for |
| --- | --- | --- |
| **SINDy** | Sparse regression on a library of candidate functions evaluated at sampled states, using numerically estimated derivatives. | Clean-ish ODE data, fast first pass. |
| **Weak SINDy (WSINDy)** | Replaces pointwise derivatives with a weak/integral formulation against test functions, dramatically improving noise robustness. | Noisy data, no reliable derivative estimates. |
| **SODAs** | Sparse Optimization for Differential-equation Approximation — group/structured sparsity for systems where terms are shared across equations. | Coupled systems, structured sparsity. |
| **IDENT** | Identifying Differential Equations with Numerical Time evolution — recovers PDEs from spatiotemporal data with error-aware term selection. | PDE discovery from grid data. |

> SODAs and IDENT descriptions will be refined as the implementations land.

## The 30-minute demo arc

1. **(0–5 min) Setup & data** — generate a known system (e.g. Lotka–Volterra, Lorenz), add noise.
2. **(5–12 min) SINDy** — build the feature library, run STLSQ, recover coefficients.
3. **(12–18 min) Weak SINDy** — show how the weak form survives noise that breaks vanilla SINDy.
4. **(18–24 min) SODAs** — structured/group sparsity on a coupled system.
5. **(24–30 min) IDENT** — PDE recovery from spatiotemporal data, plus a wrap-up comparison.

## Repository layout

```
SparseModelTutorials/
├── README.md
├── requirements.txt
├── LICENSE
├── data/                 # generated / cached datasets (gitignored)
├── notebooks/            # one notebook per method + a comparison notebook
│   ├── 00_overview.ipynb
│   ├── 01_sindy.ipynb
│   ├── 02_weak_sindy.ipynb
│   ├── 03_sodas.ipynb
│   └── 04_ident.ipynb
└── src/
    └── sparse_demos/     # shared, reusable code
        ├── __init__.py
        ├── systems.py    # reference dynamical systems & data generation
        ├── libraries.py  # candidate-function libraries
        ├── solvers.py    # sparse regression solvers (STLSQ, etc.)
        └── plotting.py   # shared visualization helpers
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## References

- Brunton, Proctor, Kutz (2016), *Discovering governing equations from data: SINDy*. PNAS.
- Messenger & Bortz (2021), *Weak SINDy for PDEs*. J. Comput. Phys.
- Schaeffer & McCalla (2017), *Sparse model selection via integral terms*.
- Kang, Liao, Liu (2021), *IDENT: Identifying Differential Equations with Numerical Time evolution*. J. Sci. Comput.

## License

MIT — see [LICENSE](LICENSE).
