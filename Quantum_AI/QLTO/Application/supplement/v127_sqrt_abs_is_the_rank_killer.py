"""Is sqrt|w| the thing destroying the bond dimension, not the data? And a fix if so.

v126 found that the EXACT weight vectors carry real low-rank structure which shot
noise then destroys - but also that the structure is too weak to pay for an MPS
prep (chi=8 at d=8 costs ~2048 gates against the exact prep's 255). This file
asks WHY the structure is weak, and the candidate answer is not the data.

THE THEORY. qlto_qml's encoder makes the angle vector a LINEAR form in the
register bits, ang(x) = A @ bits(x) with A of rank k = n_sys. So the model output
is f(x) = F(A bits(x)) for a smooth F, and on the realizable arm the labels are
y_x = F(A bits(x); theta*) too - the same A. The residual is therefore

    w_x = g(A bits(x))       for a single smooth g

and there is a clean bound on what that costs. Writing s_a(x) = sum_j A_aj x_j,
every cut splits it additively, s_a(x) = s_a(x_L) + s_a(x_R). If g is approximated
by a polynomial of total degree r in the k latent coordinates, the multinomial
theorem expands each monomial into a sum of products (function of x_L) x
(function of x_R), and the number of multi-indices of degree <= r in k variables is

    chi <= C(k + r, r)

at EVERY contiguous cut, with no 2^d anywhere. Credit where due: this bound came
from a reviewer of v126, and it is correct.

SO WHY DOES v126 NOT SEE IT. Because the amplitude prepared is not w. It is

    a(x) = sqrt(|w_x| / Z)

and z -> sqrt(|z|) is NON-SMOOTH at z = 0. A polynomial approximating it needs
degree r growing like a power of 1/epsilon rather than log(1/epsilon), so
C(k+r, r) blows up - and w_x passes through zero constantly, because that is what
a residual does as a model fits. The bound is not violated; the hypothesis of the
bound is.

WHAT THIS FILE MEASURES. The Schmidt spectra of four vectors built from the SAME
run, so the only thing varying is the function applied:

    w            the raw residual        - smooth in the latent coords
    |w|          absolute value          - kink at zero
    sqrt|w|      what qlto_qml prepares  - kink AND infinite derivative
    sqrt(w + c)  SHIFTED, c > max|w|     - smooth, strictly positive

If the ranks come out ordered w < sqrt(w+c) << |w| < sqrt|w|, the rank loss is the
nonlinearity and not the data, and the fix is to prepare a shifted distribution.

AND THE SHIFT IS NOT A TRICK - IT IS A BETTER ALGORITHM. With p_x = (w_x + c)/Z
for c > max|w|, tracing out the register gives

    sum_x p_x f_x  =  (1/Z) [ sum_x w_x f_x  +  c sum_x f_x ]

and the second term is exactly what a UNIFORM register returns, which v74
established costs one circuit. So

    sum_x w_x f_x  =  Z * <shifted register>  -  c * <uniform register>

TWO circuits instead of three, both with smooth amplitude functions, and THE SIGN
SPLIT DISAPPEARS ENTIRELY. That also retires the branch-flip failure mode v124
spent an entire file ruling out.

TIER (project rule R1). The vectors are built from f_exact - TIER B, exact
amplitudes, no shots - because the question is about the STRUCTURE OF THE TARGET
STATE, which is a property of the vector and not of any run. The Schmidt analysis
is TIER C, NO CIRCUIT. Nothing here reports an accuracy or a cost figure. If the
shifted encoding wins, building it as a circuit and measuring cos and gate count
at tier A is the next file, and this one does not pre-empt that.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.circuit.library import efficient_su2

from qlto_qml import QLTOQML

SHOTS = 32768
EPOCHS = 6
LR = 0.30
SEEDS = (0, 1, 2)
CHIS = (1, 2, 3, 4, 6, 8, 12, 16)
EPS = 0.0214                  # v123's measured shot noise on w at |D|=16


def worst_trunc(amp, d):
    """Worst-case truncation error over all contiguous cuts, per chi."""
    amp = np.asarray(amp, float)
    amp = amp / max(np.linalg.norm(amp), 1e-300)
    worst = {c: 0.0 for c in CHIS}
    for cut in range(1, d):
        t = amp.reshape([2] * d)
        m = t.reshape(2 ** (d - cut), 2 ** cut)
        s = np.linalg.svd(m, compute_uv=False)
        s = s / max(np.linalg.norm(s), 1e-300)
        for c in CHIS:
            worst[c] = max(worst[c], float(np.linalg.norm(s[c:])))
    return worst


def chi_star(worst, mx):
    for c in CHIS:
        if c <= mx and worst[c] < EPS:
            return c
    return None


def trajectory(d, n_sys, seed):
    """Exact residual vectors along a descent. TIER B - no shots anywhere."""
    rng = np.random.default_rng(seed)
    S = 1 << d
    alpha = rng.uniform(-1.0, 1.0, (n_sys, d))
    core = efficient_su2(n_sys, reps=1)
    M = core.num_parameters
    probe = QLTOQML(core, alpha, np.zeros(S), shot_budget=SHOTS, sim_seed=1)
    tstar = rng.uniform(-np.pi, np.pi, M)
    y = np.array([probe.f_exact(x, tstar) for x in range(S)])   # realizable
    q = QLTOQML(core, alpha, y, shot_budget=SHOTS, sim_seed=1)
    theta = rng.uniform(-np.pi, np.pi, M)
    out = []
    for _ep in range(EPOCHS):
        f = np.array([q.f_exact(x, theta) for x in range(S)])
        w = f - y
        out.append(w.copy())
        gs = np.zeros(M)
        for i in range(M):
            for sh, sg in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
                t = np.array(theta, float); t[i] += sh
                gs[i] += sg * 0.5 * float(np.mean(
                    w * np.array([q.f_exact(x, t) for x in range(S)])))
        theta = theta - LR * gs / max(np.max(np.abs(gs)), 1e-12)
    return out


def variants(w):
    """The four amplitude vectors, same w, different function applied."""
    c = 1.1 * float(np.max(np.abs(w))) + 1e-9
    pos = np.abs(w) * (w > 0)
    return {
        'w (raw)': w,
        '|w|': np.abs(w),
        'sqrt|w|  (current)': np.sqrt(pos / max(pos.sum(), 1e-300))
        if pos.sum() > 1e-12 else np.abs(w),
        'sqrt(w+c)  (shifted)': np.sqrt((w + c) / (w + c).sum()),
    }


print("=" * 100)
print("v127  IS sqrt|w| THE RANK KILLER?  and the shifted encoding that avoids it")
print("=" * 100)
print("  v126: exact weight vectors have real structure, too weak to pay for MPS.")
print("  Theory (reviewer of v126, correct): if a(x) = p(A bits(x)) with A of rank")
print("  k and p of degree r, then chi <= C(k+r, r) at every cut - no 2^d. But")
print("  qlto_qml prepares sqrt|w|, and sqrt|.| is non-smooth at 0, so r blows up.")
print("  TIER B vectors (exact, no shots) / TIER C analysis (SVD, NO CIRCUIT).")
print()

# --------------------------------------------------------- the four variants
print("-" * 100)
print("PART 1  same run, four amplitude functions, chi* to reach eps=%.4f" % EPS)
print("-" * 100)
print("  n_sys = k = 3 throughout. 'max' is the maximum Schmidt rank 2^(d/2);")
print("  a chi* at or above it is full rank and buys nothing.")
print()
hdr = "      d   |D|   max  " + "".join("%-22s" % v for v in
                                        ('w (raw)', '|w|', 'sqrt|w| (current)',
                                         'sqrt(w+c) (shifted)'))
print(hdr)
print("   " + "-" * 106)
p1 = {}
for d in (4, 6, 8, 10):
    S = 1 << d
    mx = 2 ** (d // 2)
    acc = {}
    for sd in SEEDS:
        for w in trajectory(d, 3, sd):
            for name, v in variants(w).items():
                acc.setdefault(name, []).append(worst_trunc(v, d))
    row = []
    for name in ('w (raw)', '|w|', 'sqrt|w|  (current)', 'sqrt(w+c)  (shifted)'):
        mean = {c: float(np.mean([a[c] for a in acc[name]])) for c in CHIS}
        cs = chi_star(mean, mx)
        p1[(d, name)] = (mean, cs)
        row.append("chi*=%-4s (%.3f@4)" % (cs if cs else "none", mean[4]))
    print("   %4d %5d %5d  %s" % (d, S, mx, "".join("%-22s" % r for r in row)))
print()
print("  Read chi* as the smallest bond dimension whose worst-cut truncation error")
print("  falls under the shot noise the weights already carry. 'none' means no")
print("  tested chi at or below max rank was sufficient.")
print()

# --------------------------------------------------------- the k dependence
print("-" * 100)
print("PART 2  does chi track the LATENT DIMENSION k, as C(k+r,r) predicts?")
print("-" * 100)
print("  The bound depends on k = n_sys (the rank of A), not on d. If it is the")
print("  right mechanism, chi* should grow with k and stay flat in d.")
print()
print("      k   d=6 chi*(raw)   d=8 chi*(raw)   d=6 chi*(shift)  d=8 chi*(shift)")
print("   " + "-" * 82)
for k in (2, 3, 4):
    cells = []
    for name in ('w (raw)', 'sqrt(w+c)  (shifted)'):
        for d in (6, 8):
            acc = []
            for sd in SEEDS:
                for w in trajectory(d, k, sd):
                    acc.append(worst_trunc(variants(w)[name], d))
            mean = {c: float(np.mean([a[c] for a in acc])) for c in CHIS}
            cells.append(chi_star(mean, 2 ** (d // 2)))
    print("   %4d   %11s   %13s   %14s   %14s"
          % (k, cells[0] if cells[0] else "none", cells[1] if cells[1] else "none",
             cells[2] if cells[2] else "none", cells[3] if cells[3] else "none"))
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
cur8 = p1[(8, 'sqrt|w|  (current)')][1]
shf8 = p1[(8, 'sqrt(w+c)  (shifted)')][1]
raw8 = p1[(8, 'w (raw)')][1]
print("  At d=8 (max rank 16):  raw w chi*=%s,  sqrt|w| chi*=%s,  shifted chi*=%s"
      % (raw8 if raw8 else "none", cur8 if cur8 else "none",
         shf8 if shf8 else "none"))
print()
if shf8 and (not cur8 or shf8 < cur8):
    g_cur = 2 ** 8 - 1
    g_shf = 4 * 8 * shf8 * shf8
    print("  THE NONLINEARITY WAS THE PROBLEM, NOT THE DATA. Shifting to a strictly")
    print("  positive amplitude drops chi* from %s to %d at d=8. The bound was never"
          % (cur8 if cur8 else "none (>16)", shf8))
    print("  violated - sqrt|.| simply fails its smoothness hypothesis, and a")
    print("  residual crosses zero constantly, which is precisely what a residual")
    print("  does as a model fits.")
    print()
    print("  WHAT IT WOULD BUY, and this is TIER C arithmetic on chi, NOT a measured")
    print("  gate count: MPS prep at chi=%d is ~4*d*chi^2 = %d gates against the"
          % (shf8, g_shf))
    print("  exact prep's 2^d - 1 = %d. %s" % (g_cur,
          "That is a real saving." if g_shf < g_cur else
          "Still not a saving at this d."))
    dd = next((k2 for k2 in range(2, 30) if 2 ** k2 - 1 > 4 * k2 * shf8 * shf8),
              None)
    if dd:
        print("  Crossover 4*d*chi^2 < 2^d - 1 at chi=%d: d >= %d, |D| >= %d."
              % (shf8, dd, 2 ** dd))
    print()
    print("  AND THE SHIFT IS A BETTER ALGORITHM INDEPENDENTLY OF RANK. With")
    print("  p_x = (w_x + c)/Z the register needs no sign split, because")
    print("      sum_x w_x f_x = Z*<shifted> - c*<uniform>")
    print("  and <uniform> is v74's one-circuit batch mean. TWO circuits instead of")
    print("  three, and the branch-flip failure mode v124 spent a file ruling out")
    print("  cannot occur at all. That is worth building even if the rank were flat.")
else:
    print("  THE SHIFT DOES NOT RESCUE THE RANK. sqrt|.| is not the binding")
    print("  constraint, so the reviewer's smoothness diagnosis - correct as a")
    print("  theorem - is not what limits these vectors. Record that the bound")
    print("  chi <= C(k+r,r) holds and is simply not tight here, and do not build")
    print("  the shifted prep on rank grounds. It may still be worth building on")
    print("  ALGORITHMIC grounds: it removes the sign split and costs two circuits")
    print("  rather than three.")
print()
print("  SCOPE. Realizable labels only (y = f(theta*)), so w IS a function of")
print("  A bits(x) by construction - the most favourable case the theorem covers,")
print("  and an upper bound on what unstructured labels would give. d <= 10,")
print("  reps=1, %d seeds, %d epochs, contiguous cuts, natural qubit order, exact"
      % (len(SEEDS), EPOCHS))
print("  vectors with no shot noise. Gate counts here are arithmetic on chi and")
print("  NOT transpiled circuits; a tier-A build is what would settle the cost.")
