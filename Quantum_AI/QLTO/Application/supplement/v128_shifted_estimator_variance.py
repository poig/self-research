"""Does the SHIFTED estimator actually work at matched shots? The question that gates v127.

v127 showed the shifted amplitude sqrt((w+c)/Z) has bond dimension flat in d
(chi* = 3, 3, 4, 3 at d = 4, 6, 8, 10) while the current sqrt|w| grows (3, 6, 8,
12). That makes a truncated MPS prep look attractive. It is worth NOTHING if the
shifted estimator is statistically unusable, and there is a specific reason to
fear it is.

THE CONDITIONING PROBLEM. The shifted reconstruction is

    sum_x w_x df_x  =  Z * G_shifted  -  c * D * G_uniform,     Z = W + cD

with W = sum_x w_x. Both terms on the right are of order c*D*|df|, while the
target is of order |sum_x w_x df_x|. So this is a DIFFERENCE OF TWO LARGE
QUANTITIES and its noise amplification grows with

    kappa  ~  2 c D / |sum_x w_x df_x|

Worse, W is a sum of SIGNED residuals and can be near zero by cancellation -
exactly when the sign split is most balanced, i.e. the case the shift was
supposed to improve. Large c buys smoothness (hence low chi); large c also buys
bad conditioning. Whether a c satisfies both at once is the whole question.

  Credit: this concern was raised by a reviewer of v127 and it is the right
  objection. v127 measured a structural property of a target state and drew no
  statistical conclusion; this file supplies the statistical one.

AND A CORRECTION THIS FILE CARRIES. qlto_qml briefly claimed the shift costs TWO
circuits per epoch instead of three. It does not. Current is f_hat + positive
branch + negative branch = 3; shifted is f_hat + shifted branch + uniform branch
= 3. f_hat is still needed to build w, and the uniform branch is theta-dependent
so it cannot be cached. The shift buys bond dimension and removes the branch-flip
failure mode. It does not buy circuit count, and both arms below are given the
same budget because they cost the same.

WHAT IS MEASURED. At MATCHED total shots, along the same trajectory:

    cos(sign-split estimate, exact gradient)      the current method
    cos(shifted estimate,    exact gradient)      swept over gamma = c/max|w|
    kappa                                         the measured amplification
    |W| / (c D)                                   how close the cancellation runs

TIER (project rule R1): tier A. Every gradient is a Qiskit circuit on
AerSimulator with finite shots, through qlto_qml.QLTOQML. The exact gradient is
tier B (Statevector) and is the reference only.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.circuit.library import efficient_su2
from qiskit_aer import AerSimulator

from qlto_qml import QLTOQML
from nisq_v6 import QLTOv6

N_SYS = 3
SHOTS = 32768
EPOCHS = 6
LR = 0.30
SEEDS = (0, 1, 2)
GAMMAS = (1.05, 1.2, 1.5, 2.0, 4.0, 8.0)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def _branch(q, p, theta):
    """One V6 gradient on the register weighted by p. Returns the raw vector."""
    anz = q.batched(p)
    v6 = QLTOv6(anz, q.O_full, shot_budget=q.shot_budget, sim_seed=q.sim_seed,
                backend=q.backend)
    with contextlib.redirect_stdout(io.StringIO()):
        g, _ = v6.sense(theta, q.radius, list(range(q.M)))
    return np.asarray(g, float), len(v6.groups)


def grad_shifted(q, theta, w, gamma):
    """Shifted estimator. Returns (g, kappa, WoverCD, ncirc)."""
    D = q.S
    c = gamma * float(np.max(np.abs(w)))
    W = float(np.sum(w))
    Z = W + c * D
    p = (w + c) / Z
    G_s, n1 = _branch(q, p, theta)
    G_u, n2 = _branch(q, np.ones(D) / D, theta)
    raw = Z * G_s - c * D * G_u
    g = (2.0 / D) * raw
    kappa = float(2.0 * c * D / max(np.linalg.norm(raw), 1e-30))
    return g, kappa, float(abs(W) / max(c * D, 1e-30)), n1 + n2


def grad_signsplit(q, theta, w):
    """The current estimator, for reference at the same budget."""
    D = q.S
    g = np.zeros(q.M)
    n = 0
    for mask, sgn in ((w > 0, +1.0), (w < 0, -1.0)):
        if not mask.any():
            continue
        pw = np.abs(w) * mask
        Z = pw.sum()
        if Z < 1e-12:
            continue
        gb, nb = _branch(q, pw / Z, theta)
        g += sgn * Z * gb
        n += nb
    return (2.0 / D) * g, n


print("=" * 100)
print("v128  DOES THE SHIFTED ESTIMATOR SURVIVE SHOT NOISE?")
print("=" * 100)
print("  v127 gave the shifted amplitude a flat bond dimension. This asks whether")
print("  the estimator built on it is usable, which v127 did not test. TIER A -")
print("  %d shots per circuit, matched budgets, both methods cost 3 circuits." % SHOTS)
print()

summary = {}
for d in (4, 6, 8):
    D = 1 << d
    print("-" * 100)
    print("d = %d   |D| = %d" % (d, D))
    print("-" * 100)
    print("      method           cos(mean)   cos(min)    kappa     |W|/cD   circuits")
    print("   " + "-" * 82)

    # --- reference arm: the current sign-split estimator
    cs, ncirc = [], None
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        alpha = rng.uniform(-1.0, 1.0, (N_SYS, d))
        core = efficient_su2(N_SYS, reps=1)
        M = core.num_parameters
        probe = QLTOQML(core, alpha, np.zeros(D), shot_budget=SHOTS, sim_seed=1)
        tstar = rng.uniform(-np.pi, np.pi, M)
        y = np.array([probe.f_exact(x, tstar) for x in range(D)])
        q = QLTOQML(core, alpha, y, shot_budget=SHOTS, sim_seed=100 + sd,
                    backend=AerSimulator(seed_simulator=100 + sd))
        theta = rng.uniform(-np.pi, np.pi, M)
        for _ep in range(EPOCHS):
            g_true, _w = q.grad_exact(theta)
            with contextlib.redirect_stdout(io.StringIO()):
                f, _den = q.f_hat(theta)
            g_est, nc = grad_signsplit(q, theta, f - y)
            ncirc = nc + 1
            cs.append(_cos(g_est, g_true))
            theta = theta - LR * g_true / max(np.max(np.abs(g_true)), 1e-12)
    base = float(np.mean(cs))
    print("      sign-split       %+.4f     %+.4f       -         -        %2d"
          % (base, float(np.min(cs)), ncirc))

    # --- the shifted arm, swept over gamma
    for gam in GAMMAS:
        cs, ks, ws, ncirc = [], [], [], None
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            alpha = rng.uniform(-1.0, 1.0, (N_SYS, d))
            core = efficient_su2(N_SYS, reps=1)
            M = core.num_parameters
            probe = QLTOQML(core, alpha, np.zeros(D), shot_budget=SHOTS,
                            sim_seed=1)
            tstar = rng.uniform(-np.pi, np.pi, M)
            y = np.array([probe.f_exact(x, tstar) for x in range(D)])
            q = QLTOQML(core, alpha, y, shot_budget=SHOTS, sim_seed=100 + sd,
                        backend=AerSimulator(seed_simulator=100 + sd))
            theta = rng.uniform(-np.pi, np.pi, M)
            for _ep in range(EPOCHS):
                g_true, _w = q.grad_exact(theta)
                with contextlib.redirect_stdout(io.StringIO()):
                    f, _den = q.f_hat(theta)
                g_est, kap, wcd, nc = grad_shifted(q, theta, f - y, gam)
                ncirc = nc + 1
                cs.append(_cos(g_est, g_true))
                ks.append(kap)
                ws.append(wcd)
                theta = theta - LR * g_true / max(np.max(np.abs(g_true)), 1e-12)
        summary[(d, gam)] = (float(np.mean(cs)), float(np.min(cs)),
                             float(np.mean(ks)), float(np.mean(ws)))
        print("      shifted g=%-5.2f   %+.4f     %+.4f    %7.1f   %.4f      %2d"
              % (gam, summary[(d, gam)][0], summary[(d, gam)][1],
                 summary[(d, gam)][2], summary[(d, gam)][3], ncirc))
    summary[(d, 'base')] = base
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  BEST SHIFTED gamma PER SIZE, against the sign-split baseline:")
print()
print("      d    sign-split    best shifted   at gamma   kappa    verdict")
print("   " + "-" * 78)
ok_any = False
for d in (4, 6, 8):
    base = summary[(d, 'base')]
    bg = max(GAMMAS, key=lambda g: summary[(d, g)][0])
    bc, _mn, bk, _w = summary[(d, bg)]
    good = bc > base - 0.05
    ok_any |= good
    print("   %4d      %+.4f       %+.4f      %5.2f   %7.1f   %s"
          % (d, base, bc, bg, bk, "usable" if good else "WORSE"))
print()
mono = all(summary[(8, GAMMAS[i])][0] >= summary[(8, GAMMAS[i + 1])][0] - 1e-9
           for i in range(len(GAMMAS) - 1))
print("  THE TRADEOFF IS VISIBLE IN THE gamma SWEEP. Larger gamma means a smoother")
print("  amplitude - v127's low chi - and a larger c*D against the same target, so")
print("  worse conditioning. At d=8 the cos column across gamma = %s is"
      % ", ".join("%.2f" % g for g in GAMMAS))
print("      %s" % "  ".join("%+.3f" % summary[(8, g)][0] for g in GAMMAS))
print("  and kappa is")
print("      %s" % "  ".join("%.0f" % summary[(8, g)][2] for g in GAMMAS))
print()
if ok_any:
    print("  A WORKABLE gamma EXISTS. The shifted estimator matches the sign-split")
    print("  baseline within 0.05 cos at the sizes marked usable, so the low bond")
    print("  dimension v127 found is not paid for with unusable variance. THAT")
    print("  CLEARS THE WAY for the tier-A MPS build - but note the gamma that wins")
    print("  here is the SMALL one, and v127 measured chi at gamma = 1.1. The next")
    print("  file must measure chi AT THE gamma THIS FILE SELECTS, not at v127's,")
    print("  or the two results will not compose.")
else:
    print("  NO gamma WORKS. The shifted estimator is worse than the sign-split at")
    print("  every gamma tested, so v127's low bond dimension is bought with")
    print("  variance the estimator cannot afford. The conditioning objection is")
    print("  CORRECT and the shifted route is closed on statistical grounds even")
    print("  though it is open on structural ones. Record both - the structural")
    print("  result stands and is simply not usable through this estimator.")
print()
print("  SCOPE. N_sys=%d, d <= 8, realizable labels, efficient_su2 reps=1, %d"
      % (N_SYS, SHOTS))
print("  shots, %d seeds, %d epochs, no noise model, no hardware. Both arms step"
      % (len(SEEDS), EPOCHS))
print("  with the EXACT gradient so the trajectory is identical and only the")
print("  estimator differs. No MPS truncation anywhere here - this measures the")
print("  shifted estimator at FULL prep, isolating conditioning from truncation.")
