"""Programmatically build notebooks/02b_weak_sindy_vs_sindy.ipynb.

Kept as a script so the notebook is reproducible and easy to regenerate. Run:
    ../.venv/bin/python build_02b.py
then execute headless with jupyter nbconvert --execute.
"""
import os
import nbformat as nbf

cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------------------- #
md(r"""# 02b · Weak SINDy vs. conventional SINDy — a fair noise stress-test

Notebook `02` introduced weak SINDy. This one makes the case **rigorously**: *when*,
*why*, and *by how much* the weak (integral) form beats conventional SINDy as
measurement noise grows — and, just as importantly, where it does **not**, so the
comparison is honest rather than a strawman.

**What we will show (claim stated narrowly).**
In this controlled Lorenz benchmark — correct-degree library, fixed sampling, and
hyperparameters fixed *before* the experiment — weak SINDy is **substantially more robust
than the canonical finite-difference SINDy** (which collapses by ~2% noise) and
**competitive-to-better than a carefully smoothed SINDy**, while needing *no* derivative or
smoothing-window choice. It tracks an *oracle* baseline that is allowed to tune itself using
the ground-truth answer (an upper bound, not something you can deploy). This is a
tutorial-scale benchmark — not a claim that weak SINDy beats *every* conventional pipeline
(total-variation / spline / Gaussian-process derivatives, ensembling, SR3, …).

> **Method, faithfully and from scratch.** We reproduce the weak formulation of
> Messenger & Bortz, *Weak SINDy: Galerkin-Based Data-Driven Model Selection*,
> Multiscale Model. Simul. **19**(3), 1474–1497 (2021), [arXiv:2005.04339](https://arxiv.org/abs/2005.04339),
> in plain NumPy (`src/sparse_demos/weak.py`) — the reference code is MATLAB.

**Example.** The chaotic **Lorenz** system (the paper's headline). Its broadband
trajectory makes pointwise derivative estimation genuinely hard, its dynamics are smooth
enough that a smoothed baseline recovers it on clean data (so the comparison is fair), and
the ground-truth equations are sparse and recognizable.""")

# --- install (inactive) ---
code("""# --- Environment setup (kept INACTIVE on purpose) -------------------------
# Uncomment to install dependencies the first time (numpy, scipy, scikit-learn,
# matplotlib, pysindy ...). Left commented so the notebook never reinstalls while
# you teach from it.
# !pip install -r ../requirements.txt""")

# --- imports / path ---
code("""import os, sys, warnings, inspect
warnings.filterwarnings("ignore")

def _add_src():
    here = os.getcwd()
    for base in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]:
        cand = os.path.join(base, "src")
        if os.path.isdir(os.path.join(cand, "sparse_demos")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    raise RuntimeError("Could not locate src/sparse_demos")
_add_src()

import numpy as np
import matplotlib.pyplot as plt
from sparse_demos import systems, libraries, solvers, metrics, plotting, weak
%matplotlib inline
np.set_printoptions(suppress=True, precision=3)""")

# ----------------------------------------------------------------------------- #
md(r"""## 1 · The problem with pointwise derivatives

SINDy looks for a sparse coefficient matrix $\Xi$ with $\dot X = \Theta(X)\,\Xi$, where
$\Theta(X)$ is a library of candidate terms. Conventional SINDy must first estimate
$\dot X$ from the samples. With noisy data $y(t_m)=x(t_m)+\varepsilon_m$,
$\varepsilon_m\sim\mathcal N(0,\sigma^2)$, a finite difference gives

$$\frac{y_{m+1}-y_{m-1}}{2\Delta t}=\dot x(t_m)+\underbrace{\frac{\varepsilon_{m+1}-\varepsilon_{m-1}}{2\Delta t}}_{\text{std}=\;\sigma/(\sqrt2\,\Delta t)} .$$

The noise is **amplified by $1/\Delta t$**: finer sampling — usually a good thing — makes the
derivative estimate *worse*. Smoothing (Savitzky–Golay, splines, total variation) helps, but
it trades variance for **bias**, especially where the dynamics move fast. The weak form
removes the differentiation of data entirely.""")

md(r"""## 2 · The weak (Galerkin) formulation

Choose a smooth, compactly supported **test function** $\phi_k$ on $[a_k,b_k]$. Multiply the
$d$-th equation $\dot x_d=\sum_j \Xi_{jd}f_j(x)$ by $\phi_k$, integrate, and integrate by parts:

$$\int_{a_k}^{b_k}\!\phi_k\,\dot x_d\,dt
=\big[\phi_k x_d\big]_{a_k}^{b_k}-\int_{a_k}^{b_k}\!\dot\phi_k\,x_d\,dt
=-\int_{a_k}^{b_k}\!\dot\phi_k\,x_d\,dt,$$

the boundary term vanishing because $\phi_k(a_k)=\phi_k(b_k)=0$. Substituting the dynamics and
discretizing each integral by quadrature on the sample grid, stacking $K$ test functions gives a
linear system **per state** $d$:

$$G\,\xi_d=b_d,\qquad
G_{kj}=\!\int\!\phi_k f_j(y)\,dt\approx\sum_m w_m\phi_k(t_m)f_j(y_m),\qquad
b_{kd}=-\!\int\!\dot\phi_k\,y_d\,dt\approx-\sum_m w_m\dot\phi_k(t_m)y_{md},$$

with trapezoidal weights $w_m$ (endpoints halved). **The only derivative is $\dot\phi_k$,
computed analytically on the known, smooth test function — never on the noisy data.** We then
solve $G\xi_d=b_d$ with the *same* sequential-thresholded least squares (STLS) used for vanilla
SINDy. So against the **finite-difference** baseline, the *only* difference is exactly this —
weak form vs. differentiating the data. (The *smoothed* baseline we add later changes a bit
more: it also denoises the states $X$ that enter the library, so it isolates the broader
*estimation principle* rather than a single line of code.)""")

md(r"""## 3 · Test functions $\phi_k$

We use the piecewise-polynomial "bump" — the symmetric ($p=q$) case of the paper's
$C(t-a)^p(b-t)^q$, normalized so $\max\phi=1$:

$$\phi_k(t)=\bigl(1-\tau^2\bigr)^p,\quad\tau=\frac{t-c_k}{r},\ |\tau|<1,\qquad
\dot\phi_k(t)=\frac{-2p\,\tau}{r}\bigl(1-\tau^2\bigr)^{p-1}.$$

It is $C^{p-1}$ and vanishes at $\tau=\pm1$, so the integration-by-parts boundary term is exactly
zero for $p\ge1$ (we use $p\ge2$). Three knobs — all a **bias–variance–conditioning tradeoff**,
which we fix *before* the experiment:

* **support radius $r$** — larger $r$ averages more noise (less variance) but washes out fast dynamics (more bias);
* **degree $p$** — higher $p$ concentrates the bump and steepens its spectral roll-off;
* **count $K$** — more test functions add equations (rows of $G$) but increasingly correlated ones.""")

code("""# A few bump test functions and their analytic derivatives.
tt = np.linspace(0, 1, 600)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for p in (2, 4, 8, 16):
    phi, _ = weak.bump_test_function(tt, center=0.5, radius=0.18, p=p)
    ax[0].plot(tt, phi, label=f"p={p}")
ax[0].set_title(r"$\\phi(t)=(1-\\tau^2)^p$  (support radius $r$=0.18)"); ax[0].legend(); ax[0].set_xlabel("t")

phi, phidot = weak.bump_test_function(tt, 0.5, 0.18, 8)
ax[1].plot(tt, phi, label=r"$\\phi$")
ax[1].plot(tt, phidot, label=r"$\\dot\\phi$ (analytic)")
ax[1].set_title("test function and its derivative (p=8)"); ax[1].legend(); ax[1].set_xlabel("t")

for c in np.linspace(0.12, 0.88, 6):
    phi, _ = weak.bump_test_function(tt, c, 0.08, 8)
    ax[2].plot(tt, phi)
ax[2].set_title("K translated, overlapping test functions"); ax[2].set_xlabel("t")
plt.tight_layout(); plt.show()""")

md(r"""## 4 · The from-scratch weak-form builder

Here is the actual implementation we use (from `sparse_demos/weak.py`) — short enough to read.
`G = Φ·Θ(Y)` (library evaluated on the data, **no derivatives**) and `b = −Φ′·Y`, with the
trapezoidal quadrature weights folded into `Φ`/`Φ′`.""")

code("""print(inspect.getsource(weak.test_function_matrices))
print(inspect.getsource(weak.weak_linear_system))""")

# ----------------------------------------------------------------------------- #
md(r"""## 5 · Sanity check on clean data (and conditioning)

Before any noise: the from-scratch weak form must recover Lorenz essentially exactly, and the
conventional baseline must too — otherwise the later comparison would be unfair. (Our
from-scratch `weak_sindy` was also checked to agree with `pysindy`'s `WeakPDELibrary` in this
clean configuration.)""")

code("""# Lorenz ground truth in the library's term order.
DT, T_END, DEGREE = 0.002, 10.0, 2
t, X_clean = systems.simulate("lorenz", t_span=(0, T_END), dt=DT)
_, names = libraries.polynomial_library(X_clean, degree=DEGREE)

def make_true_Xi(names):
    Xi = np.zeros((len(names), 3)); idx = {n: i for i, n in enumerate(names)}
    Xi[idx["x0"], 0], Xi[idx["x1"], 0] = -10.0, 10.0                       # x' = 10(y-x)
    Xi[idx["x0"], 1], Xi[idx["x1"], 1], Xi[idx["x0 x2"], 1] = 28.0, -1.0, -1.0  # y' = 28x - y - xz
    Xi[idx["x0 x1"], 2], Xi[idx["x2"], 2] = 1.0, -8.0/3.0                  # z' = xy - (8/3)z
    return Xi
true_Xi = make_true_Xi(names)
TARGETS = ["x'", "y'", "z'"]

Xi_w, _ = weak.weak_sindy(t, X_clean, degree=DEGREE, threshold=0.5, K=200, p=8, support_fraction=0.04)
Xi_v, _ = weak.vanilla_sindy(t, X_clean, degree=DEGREE, threshold=0.5, derivative="fd")
print("clean relative coefficient error:")
print(f"  weak SINDy            {metrics.coef_l2_error(true_Xi, Xi_w):.2e}")
print(f"  conventional SINDy    {metrics.coef_l2_error(true_Xi, Xi_v):.2e}")

# conditioning: pointwise library Theta vs. weak Gram matrix G
Theta, _ = libraries.polynomial_library(X_clean, degree=DEGREE)
G, b, _, _ = weak.weak_linear_system(t, X_clean, degree=DEGREE, K=200, p=8, support_fraction=0.04)
print(f"\\ncond(Theta) = {np.linalg.cond(Theta):.1f}    cond(G) = {np.linalg.cond(G):.1f}")""")

md(r"""## 6 · Why the weak form is robust — and where it still breaks (honest version)

* **Variance reduction.** $b_{kd}$ is a weighted *sum* of the data against $\dot\phi_k$. Zero-mean
  noise averages down: instead of $\sigma/\Delta t$ per point it enters as $\sigma\|\dot\phi_k\|$
  over the whole support, which the support size controls.
* **Spectral view.** Convolving against $\phi_k$ is a smooth low-pass filter; the analytic
  $\dot\phi_k$ differentiates the *filter*, not the signal.
* **Honest limitation.** The weak form removes differentiation of noisy data, but the library
  $f_j(y)$ is still evaluated on noisy states. For a nonlinear feature the bias is $O(\sigma^2)$:
  e.g. $\mathbb E[(x+\varepsilon)^2]=x^2+\sigma^2$ — a constant offset on a squared term — while
  *linear* terms stay unbiased. This errors-in-variables bias is why weak SINDy *also* fails at
  high enough noise, just at a far higher level than conventional SINDy.

> *Refinements in the paper we deliberately omit:* a generalized-least-squares whitening step
> ($\Sigma=V'(V')^\top$) and adaptive test-function placement. The unweighted core below already
> beats conventional SINDy. (We include `weak.gls_whiten` for the curious — but a *naive*
> covariance estimate can actually hurt, which is exactly why the paper's careful version is a
> contribution in its own right.)""")

# ----------------------------------------------------------------------------- #
md(r"""## 7 · Experimental protocol (pre-registered)

To avoid fooling ourselves, everything below the next cell is fixed **in advance**:

* **Noise model.** Signal-relative Gaussian noise: $\sigma_d=\text{ratio}\cdot\text{RMS}(x_d)$ per
  state dimension (so each coordinate gets noise at its own scale). We report both the per-dim and
  the global RMS so "X% noise" is unambiguous.
* **Four conventional baselines / one weak method**, all sharing the library and STLS threshold:
  1. **SINDy (finite diff)** — the canonical, most common form (the lower bound).
  2. **SINDy (smoothed)** — Savitzky–Golay on *both* states and derivative; window fixed in advance
     and *validated to recover on clean data* (a strong, fair baseline — not a strawman).
  3. **weak SINDy** — fixed $p,r,K$, *no* smoothing.
  4. **SINDy (oracle\*)** — allowed to pick its derivative scheme **and** threshold using the
     ground-truth answer. This is **not deployable** (you never know the answer); it is an
     upper bound included only to be maximally fair to conventional SINDy.
* **Success** = exact active-term support **and** relative L2 coefficient error $\le 0.10$ over the
  *full* matrix (so false positives are penalized). We run **paired** trials (identical noise to
  every method) and report the recovery rate with **Wilson 95% confidence intervals**, plus the
  mean false-positive / false-negative term counts.""")

code("""# ---- Pre-registered configuration (NOT tuned to the result) ----
SEED0       = 100           # trial k uses SEED0 + k
THRESHOLD   = 0.5           # STLS threshold, identical for every method
P_BUMP      = 8             # bump exponent p
SUPPORT_FRAC= 0.04          # support radius r = SUPPORT_FRAC * (t_end - t0)
K_TEST      = 200           # number of test functions
SG_WINDOW   = 71            # Savitzky-Golay window (points) -- recovers clean data (sec. 5)
SG_POLY     = 3
NOISE_GRID  = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
N_TRIALS    = 80            # paired trials per noise level (tighter CIs + McNemar power)
ILLUSTRATIVE_NOISE = 0.04   # declared in advance: a mid-transition level for the head-to-head
TOL         = 0.10

def succeeds(true_Xi, Xi):
    return (metrics.support_match(true_Xi, Xi, 1e-6) == 1.0) and (metrics.coef_l2_error(true_Xi, Xi) <= TOL)

def fit_fd(t, Y):
    return weak.vanilla_sindy(t, Y, degree=DEGREE, threshold=THRESHOLD, derivative="fd")[0]
def fit_sg(t, Y):
    return weak.vanilla_sindy(t, Y, degree=DEGREE, threshold=THRESHOLD, derivative="savgol",
                              smooth_states=True, window=SG_WINDOW, polyorder=SG_POLY)[0]
def fit_weak(t, Y):
    return weak.weak_sindy(t, Y, degree=DEGREE, threshold=THRESHOLD, K=K_TEST, p=P_BUMP,
                           support_fraction=SUPPORT_FRAC)[0]

# oracle conventional SINDy: best over {FD, SG-windows} x threshold grid, judged by the truth
_ORACLE_CFGS = [dict(derivative="fd", smooth_states=False)] + \\
               [dict(derivative="savgol", smooth_states=True, window=w) for w in (11, 31, 71, 151)]
_ORACLE_THRS = [0.1, 0.2, 0.35, 0.5, 0.75]
def oracle_succeeds(t, Y, true_Xi):
    for cfg in _ORACLE_CFGS:
        for th in _ORACLE_THRS:
            if succeeds(true_Xi, weak.vanilla_sindy(t, Y, degree=DEGREE, threshold=th, **cfg)[0]):
                return True
    return False
print("protocol defined.")""")

# ----------------------------------------------------------------------------- #
md(r"""## 8 · Head-to-head at one illustrative noise level

At the **pre-declared** mid-transition level (4% noise) and a fixed seed: conventional
finite-difference SINDy returns the wrong model (spurious and/or missing terms), while weak SINDy
recovers the true equations.""")

code("""ratio = ILLUSTRATIVE_NOISE
Xn, sigma = systems.add_measurement_noise(X_clean, ratio, seed=SEED0)
print(f"noise ratio = {ratio:.0%}   per-dim sigma = {sigma}   global RMS = {np.sqrt(np.mean(X_clean**2)):.2f}")

print("\\n================ TRUE ================")
solvers.print_model(true_Xi, names, TARGETS)
print("\\n========= SINDy (finite diff) =======")
solvers.print_model(fit_fd(t, Xn), names, TARGETS)
print("\\n========= SINDy (smoothed) ==========")
solvers.print_model(fit_sg(t, Xn), names, TARGETS)
print("\\n============ weak SINDy =============")
solvers.print_model(fit_weak(t, Xn), names, TARGETS)

for label, Xi in [("SINDy (finite diff)", fit_fd(t, Xn)),
                  ("SINDy (smoothed)",    fit_sg(t, Xn)),
                  ("weak SINDy",          fit_weak(t, Xn))]:
    fp, fn = metrics.fp_fn_counts(true_Xi, Xi)
    print(f"\\n{label:22s} rel.err={metrics.coef_l2_error(true_Xi, Xi):.3f}  "
          f"FP={fp} FN={fn}  correct={succeeds(true_Xi, Xi)}")""")

code("""# the data the methods actually see at 4% noise
fig = plt.figure(figsize=(12, 3.6))
ax1 = fig.add_subplot(1, 3, 1); ax1.plot(t, X_clean[:, 0], lw=1, label="clean x")
ax1.plot(t, Xn[:, 0], lw=0.5, alpha=0.6, label="noisy x"); ax1.set_xlabel("t"); ax1.legend(); ax1.set_title("x(t)")
ax2 = fig.add_subplot(1, 3, 2); ax2.plot(X_clean[:, 0], X_clean[:, 2], lw=0.6); ax2.set_title("clean attractor"); ax2.set_xlabel("x"); ax2.set_ylabel("z")
ax3 = fig.add_subplot(1, 3, 3); ax3.plot(Xn[:, 0], Xn[:, 2], lw=0.3, alpha=0.7, color="C1"); ax3.set_title("noisy data (4%)"); ax3.set_xlabel("x"); ax3.set_ylabel("z")
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------- #
md(r"""## 9 · The main result — recovery rate vs. noise

Now the statistics: for each noise level we run `N_TRIALS` paired realizations and measure each
method's exact-recovery rate, with Wilson 95% bands.""")

code("""methods = {"SINDy (finite diff)": fit_fd, "SINDy (smoothed)": fit_sg, "weak SINDy": fit_weak}
ORACLE = "SINDy (oracle*, uses truth)"
ALL = list(methods) + [ORACLE]
succ = {m: [] for m in ALL}; ci = {m: [] for m in ALL}
err_store = {m: [] for m in methods}     # per-ratio array of per-trial relative coef errors
succ_arr  = {m: [] for m in ALL}         # per-ratio array of per-trial success booleans (paired)

for ratio in NOISE_GRID:
    counts = {m: 0 for m in ALL}
    errs = {m: [] for m in methods}
    sarr = {m: [] for m in ALL}
    for k in range(N_TRIALS):
        Xn, _ = systems.add_measurement_noise(X_clean, ratio, seed=SEED0 + k)
        for m, fn in methods.items():
            Xi = fn(t, Xn)
            ok = bool(succeeds(true_Xi, Xi))
            counts[m] += ok; sarr[m].append(ok)
            errs[m].append(metrics.coef_l2_error(true_Xi, Xi))
        ok_o = bool(oracle_succeeds(t, Xn, true_Xi))
        counts[ORACLE] += ok_o; sarr[ORACLE].append(ok_o)
    for m in ALL:
        succ[m].append(counts[m] / N_TRIALS); ci[m].append(metrics.wilson_interval(counts[m], N_TRIALS))
        succ_arr[m].append(np.array(sarr[m], dtype=bool))
    for m in methods:
        err_store[m].append(np.array(errs[m]))

ax = plotting.plot_success_vs_noise(NOISE_GRID, succ, ci,
        title=f"Lorenz: exact-recovery rate vs. noise ({N_TRIALS} paired trials, Wilson 95% CI)")
ax.axvline(ILLUSTRATIVE_NOISE, color="k", ls=":", alpha=0.5)
plt.show()

hdr = "ratio  " + "".join(f"{m:>28}" for m in ALL)
print(hdr); print("-" * len(hdr))
for i, r in enumerate(NOISE_GRID):
    print(f"{r:5.3f} " + "".join(f"{succ[m][i]:>28.2f}" for m in ALL))""")

md(r"""**Reading the plot.**

* **weak SINDy vs. finite-difference SINDy — decisive.** By ~2% noise the canonical
  finite-difference baseline already fails most of the time while weak SINDy is essentially
  perfect; by 5% it is hopeless. The Wilson bands are far apart, so this gap is unambiguous.
* **weak SINDy vs. the smoothed baseline — competitive-to-better.** Weak SINDy is consistently
  at or above the smoothed baseline *and never had to choose a smoothing window* — but the
  *marginal* bands overlap, so we test the **paired** difference explicitly below (McNemar).
* **weak SINDy tracks the oracle.** The oracle may pick its derivative scheme and threshold using
  the ground-truth answer — an upper bound you cannot deploy. Weak SINDy nearly matches it with
  **no** access to the truth and **no** derivative/smoothing choice.
* **Everything eventually fails** at large enough noise (the $O(\sigma^2)$ library bias): weak
  SINDy is far more robust, not magic.""")

md(r"""### How wrong, not just how often — and is the weak-vs-smoothed edge significant?

Success rates hide *how badly* a method misses. Below: the coefficient-error distribution
(median + interquartile band) per method, then a **paired McNemar test** of whether weak SINDy's
edge over the smoothed baseline reaches significance at each noise level.""")

code("""from scipy.stats import binomtest

# (a) coefficient-error distribution (median + IQR band)
fig, axe = plt.subplots(figsize=(7.5, 4.5))
for m in methods:
    med = [np.median(e) for e in err_store[m]]
    q1  = [np.percentile(e, 25) for e in err_store[m]]
    q3  = [np.percentile(e, 75) for e in err_store[m]]
    line, = axe.plot(NOISE_GRID, med, marker="o", label=m)
    axe.fill_between(NOISE_GRID, q1, q3, color=line.get_color(), alpha=0.2)
axe.axhline(TOL, color="r", ls=":", alpha=0.7, label=f"success tol = {TOL}")
axe.set_yscale("log"); axe.set_xlabel(r"noise ratio  $\\sigma/$RMS")
axe.set_ylabel("relative coefficient L2 error")
axe.set_title("Coefficient-error distribution (median + IQR over trials)")
axe.legend(); axe.grid(True, alpha=0.3); plt.show()

# (b) paired McNemar test: weak SINDy vs SINDy (smoothed)
print("Paired McNemar test   weak SINDy  vs  SINDy (smoothed):")
print(f"  {'ratio':>6} {'b: weak>sg':>11} {'c: sg>weak':>11} {'p-value':>9}")
for i, r in enumerate(NOISE_GRID):
    w = succ_arr["weak SINDy"][i]; s = succ_arr["SINDy (smoothed)"][i]
    b = int(np.sum(w & ~s)); c = int(np.sum(~w & s))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    print(f"  {r:6.3f} {b:>11} {c:>11} {p:>9.3f}{'  *' if p < 0.05 else ''}")""")

md(r"""**Reading the paired test.** Weak SINDy wins more head-to-head matchups than it loses at
*every* noisy level ($b>c$), but the McNemar $p$ only crosses $0.05$ at the highest noise — and
that is before any multiple-comparison correction. So the fair statement is
**competitive-to-better** than a well-tuned smoothed baseline, *not* "decisively better." The
statistically unambiguous win is over the **finite-difference** baseline; the practical advantage
over smoothing is that weak SINDy needs *no* window choice and tends to pull ahead as data grow
scarcer (next section).""")

md(r"""### Is the fixed threshold $\lambda=0.5$ hand-picked?

No: the conclusion holds across STLS thresholds. We re-run the sweep at $\lambda\in\{0.2,0.5,1.0\}$
(the same $\lambda$ for both methods, never chosen using the truth). Finite-difference SINDy
collapses at every threshold; weak SINDy is robust at every threshold.""")

code("""THRS = [0.2, 0.5, 1.0]
fig, axth = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for th in THRS:
    sf_fd, sf_w = [], []
    for ratio in NOISE_GRID:
        cf = cw = 0
        for k in range(N_TRIALS):
            Xn, _ = systems.add_measurement_noise(X_clean, ratio, seed=SEED0 + k)
            cf += succeeds(true_Xi, weak.vanilla_sindy(t, Xn, degree=DEGREE, threshold=th, derivative="fd")[0])
            cw += succeeds(true_Xi, weak.weak_sindy(t, Xn, degree=DEGREE, threshold=th, K=K_TEST, p=P_BUMP, support_fraction=SUPPORT_FRAC)[0])
        sf_fd.append(cf / N_TRIALS); sf_w.append(cw / N_TRIALS)
    axth[0].plot(NOISE_GRID, sf_fd, "o-", label=f"$\\lambda$={th}")
    axth[1].plot(NOISE_GRID, sf_w, "o-", label=f"$\\lambda$={th}")
axth[0].set_title("finite-difference SINDy"); axth[1].set_title("weak SINDy")
axth[0].set_ylabel("exact-recovery rate")
for a in axth:
    a.set_xlabel(r"noise ratio  $\\sigma/$RMS"); a.set_ylim(-0.03, 1.03); a.grid(True, alpha=0.3); a.legend()
plt.suptitle("Robustness to the STLS threshold (threshold never chosen using the truth)")
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------- #
md(r"""## 10 · Robustness control: less data / coarser sampling

A skeptic might ask whether weak SINDy's edge is just an artifact of very fine sampling (lots of
points to integrate over). We rerun the sweep on a **5× coarser** grid ($\Delta t=0.01$, a quarter
of the data). The conventional baselines degrade faster; weak SINDy's lead over the *smoothed*
baseline actually **widens** — smoothing simply has fewer points to average.""")

code("""DT_C = 0.01
t_c, Xc_clean = systems.simulate("lorenz", t_span=(0, T_END), dt=DT_C)
GRID_C = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075]
methods_c = {
    "SINDy (finite diff)": lambda tt, Y: weak.vanilla_sindy(tt, Y, degree=DEGREE, threshold=THRESHOLD, derivative="fd")[0],
    "SINDy (smoothed)":    lambda tt, Y: weak.vanilla_sindy(tt, Y, degree=DEGREE, threshold=THRESHOLD, derivative="savgol", smooth_states=True, window=31, polyorder=SG_POLY)[0],
    "weak SINDy":          lambda tt, Y: weak.weak_sindy(tt, Y, degree=DEGREE, threshold=THRESHOLD, K=150, p=P_BUMP, support_fraction=SUPPORT_FRAC)[0],
}
succ_c = {m: [] for m in methods_c}; ci_c = {m: [] for m in methods_c}
for ratio in GRID_C:
    cnt = {m: 0 for m in methods_c}
    for k in range(N_TRIALS):
        Yn, _ = systems.add_measurement_noise(Xc_clean, ratio, seed=SEED0 + k)
        for m, fn in methods_c.items():
            if succeeds(true_Xi, fn(t_c, Yn)):
                cnt[m] += 1
    for m in methods_c:
        succ_c[m].append(cnt[m] / N_TRIALS); ci_c[m].append(metrics.wilson_interval(cnt[m], N_TRIALS))
ax = plotting.plot_success_vs_noise(GRID_C, succ_c, ci_c,
        title=f"Lorenz, 5x coarser sampling (dt={DT_C}, N={len(t_c)}): recovery vs. noise")
plt.show()""")

# ----------------------------------------------------------------------------- #
md(r"""## 11 · Choosing the test functions — sensitivity of $p$, $r$, $K$

The "right set of test functions" matters. At a fixed noise level we vary one knob at a time
(the others held at the pre-registered values). The clearest lesson: **too wide a support washes
out the dynamics** (bias), while moderate support, moderate $p$, and enough $K$ are robust.""")

code("""ratio = 0.05
Xn, _ = systems.add_measurement_noise(X_clean, ratio, seed=SEED0)
def weak_err(p=P_BUMP, sf=SUPPORT_FRAC, K=K_TEST):
    Xi = weak.weak_sindy(t, Xn, degree=DEGREE, threshold=THRESHOLD, K=K, p=p, support_fraction=sf)[0]
    return metrics.coef_l2_error(true_Xi, Xi)

ps   = [2, 4, 6, 8, 12, 16]
sfs  = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
Ks   = [25, 50, 100, 200, 400]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(ps, [weak_err(p=p) for p in ps], "o-"); ax[0].set_xlabel("bump degree p"); ax[0].set_ylabel("rel. coef error"); ax[0].set_yscale("log"); ax[0].set_title("vary p")
ax[1].plot(sfs, [weak_err(sf=s) for s in sfs], "o-"); ax[1].set_xlabel("support fraction r/(T)"); ax[1].set_yscale("log"); ax[1].set_title("vary support radius"); ax[1].axhline(TOL, color="r", ls=":", alpha=0.6)
ax[2].plot(Ks, [weak_err(K=K) for K in Ks], "o-"); ax[2].set_xlabel("number of test functions K"); ax[2].set_yscale("log"); ax[2].set_title("vary K")
for a in ax: a.grid(True, alpha=0.3)
plt.suptitle(f"Weak-SINDy sensitivity at {ratio:.0%} noise (one knob varied at a time)"); plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------- #
md(r"""## 12 · Takeaways

* **Out of the box, weak SINDy is dramatically more noise-robust than the canonical
  finite-difference SINDy** — the headline gap is large and statistically unambiguous. It is
  **competitive-to-better than a carefully smoothed SINDy** (see the paired McNemar test; the edge
  widens with less data), and it nearly **matches an oracle** baseline that tunes itself using the
  answer — all *without* choosing a derivative scheme or smoothing window.
* **The mechanism**: the weak form never differentiates noisy data — the only derivative lands on
  the smooth, analytic test function, and the integral averages zero-mean noise away.
* **Honest limits.** With heavy *per-instance* tuning (the oracle), conventional SINDy can be
  competitive — but that tuning needs the very answer you are trying to discover. And weak SINDy is
  not magic: the library is still evaluated on noisy states, so at large enough noise the
  $O(\sigma^2)$ errors-in-variables bias defeats every method. This is a tutorial-scale benchmark
  on one system, not a verdict against every conventional SINDy pipeline.
* **Test functions are a real choice:** moderate support radius and degree $p$, and enough $K$;
  too-wide support biases the fit (Section 11).

Next: `03_sindy_pi.ipynb` — dynamics that are *rational*, which no polynomial library can express.""")

# ----------------------------------------------------------------------------- #
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02b_weak_sindy_vs_sindy.ipynb")
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
