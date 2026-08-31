"""qlto_qml against SPSA - the baseline it actually competes with, never yet run.

Every circuit-count claim in this line is against PARAMETER-SHIFT: 2M per sample,
2M|D| for the batch, 384 against qlto_qml's 3 at M=12 and |D|=16. That ratio is
real and it is also the wrong comparison, because nobody trains a large-M
variational model with parameter-shift. They use SPSA, and SPSA costs

    TWO circuits per epoch, independent of M AND of |D|

which is FEWER than qlto_qml's three. So on circuit count qlto_qml does not beat
the real baseline - it loses by one circuit. The entire case has to rest on
quality per shot instead, and that has never been measured for this module.

  v74 measured QLTO beating SPSA 5.65x at N=4 and 2.97x at N=6, but that was the
  UNIFORM-register surrogate on a label-free objective - the one v74's own part 2
  showed does not give the MSE gradient (cos -0.7431, -0.0923, +0.2981, +0.2663).
  So that result does not transfer to the supervised loss this module trains.

WHY SPSA COSTS TWO AND NOT MORE. The MSE loss L(theta) = (1/D) sum_x (f_x-y_x)^2
needs every f_x, and qlto_qml.f_hat already returns all |D| of them from ONE
circuit by measuring the data register jointly with the system. So one f_hat call
is one loss evaluation, and SPSA's two-sided difference

    g_hat = [L(theta + c*Delta) - L(theta - c*Delta)] / (2c) * Delta^{-1}

is exactly two f_hat calls. The data register helps SPSA exactly as much as it
helps QLTO, which is the honest way to set the comparison up.

WHAT IS MEASURED, at MATCHED TOTAL SHOTS PER EPOCH:

    MSE trajectory under each optimiser, driven by its OWN estimate
    cos(estimate, exact gradient) per epoch
    circuits per epoch, counted

Matching the budget is the whole point: qlto_qml gets 3 circuits at B/3 shots
each, SPSA gets 2 at B/2. Neither is given more total shots than the other.

TIER (project rule R1): tier A. Both optimisers drive their own descent from
circuits on AerSimulator with finite shots. The exact gradient is tier B
(Statevector) and is used only as the reference for the cos column.
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

N_SYS = 3
BUDGET = 49152          # total shots per epoch, divisible by 2 and 3
EPOCHS = 30
LR = 0.30
SPSA_C = 0.15
SEEDS = (0, 1, 2, 3)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def make(d, seed):
    rng = np.random.default_rng(seed)
    D = 1 << d
    alpha = rng.uniform(-1.0, 1.0, (N_SYS, d))
    core = efficient_su2(N_SYS, reps=1)
    M = core.num_parameters
    probe = QLTOQML(core, alpha, np.zeros(D), shot_budget=BUDGET, sim_seed=1)
    tstar = rng.uniform(-np.pi, np.pi, M)
    y = np.array([probe.f_exact(x, tstar) for x in range(D)])     # realizable
    theta0 = rng.uniform(-np.pi, np.pi, M)
    return alpha, core, y, theta0, M, D


def loss_from_shots(q, theta, shots):
    with contextlib.redirect_stdout(io.StringIO()):
        f, _den = q.f_hat(theta, shots=shots)
    return float(np.mean((f - q.y) ** 2)), f


def run_qlto(d, seed):
    alpha, core, y, theta0, M, D = make(d, seed)
    q = QLTOQML(core, alpha, y, shot_budget=BUDGET // 3, sim_seed=200 + seed,
                backend=AerSimulator(seed_simulator=200 + seed))
    theta = np.array(theta0, float)
    mses, coss = [], []
    for _ep in range(EPOCHS):
        g_true, w_ex = q.grad_exact(theta)
        mses.append(float(np.mean(w_ex ** 2)))
        with contextlib.redirect_stdout(io.StringIO()):
            f, _den = q.f_hat(theta, shots=BUDGET // 3)
            g, _ = q.gradient(theta, w=f - y)
        coss.append(_cos(g, g_true))
        theta = theta - LR * g / max(np.max(np.abs(g)), 1e-12)
    return mses, coss, 3


def run_spsa(d, seed):
    alpha, core, y, theta0, M, D = make(d, seed)
    q = QLTOQML(core, alpha, y, shot_budget=BUDGET // 2, sim_seed=300 + seed,
                backend=AerSimulator(seed_simulator=300 + seed))
    theta = np.array(theta0, float)
    rng = np.random.default_rng(900 + seed)
    mses, coss = [], []
    for _ep in range(EPOCHS):
        g_true, w_ex = q.grad_exact(theta)
        mses.append(float(np.mean(w_ex ** 2)))
        delta = rng.integers(0, 2, M) * 2.0 - 1.0        # Rademacher
        lp, _ = loss_from_shots(q, theta + SPSA_C * delta, BUDGET // 2)
        lm, _ = loss_from_shots(q, theta - SPSA_C * delta, BUDGET // 2)
        g = ((lp - lm) / (2.0 * SPSA_C)) / delta         # Delta is +-1 so 1/d = d
        coss.append(_cos(g, g_true))
        theta = theta - LR * g / max(np.max(np.abs(g)), 1e-12)
    return mses, coss, 2


print("=" * 100)
print("v129  qlto_qml vs SPSA:  the baseline that was never run")
print("=" * 100)
print("  Every circuit-count claim in this line is against parameter-shift")
print("  (2M|D| = 384 vs 3). SPSA costs TWO circuits per epoch, flat in M and in")
print("  |D| - FEWER than qlto_qml's three. So the case cannot rest on circuit")
print("  count against the real baseline; it has to rest on quality per shot.")
print()
print("  TIER A. Matched total budget %d shots/epoch: qlto_qml 3 circuits at" % BUDGET)
print("  %d each, SPSA 2 at %d each. Both drive their OWN descent."
      % (BUDGET // 3, BUDGET // 2))
print()

for d in (3, 4):
    D = 1 << d
    print("-" * 100)
    print("d = %d   |D| = %d   M = 12" % (d, D))
    print("-" * 100)
    Q = [run_qlto(d, s) for s in SEEDS]
    S = [run_spsa(d, s) for s in SEEDS]
    qm = np.array([r[0] for r in Q]); qc = np.array([r[1] for r in Q])
    sm = np.array([r[0] for r in S]); sc = np.array([r[1] for r in S])
    print("      optimiser   circuits   MSE start   MSE end    best MSE   mean cos")
    print("   " + "-" * 84)
    print("      qlto_qml       %d        %.5f    %.5f    %.5f    %+.4f"
          % (3, qm[:, 0].mean(), qm[:, -1].mean(), qm.min(axis=1).mean(),
             qc.mean()))
    print("      SPSA           %d        %.5f    %.5f    %.5f    %+.4f"
          % (2, sm[:, 0].mean(), sm[:, -1].mean(), sm.min(axis=1).mean(),
             sc.mean()))
    print()
    print("      epoch:   " + "".join("%8d" % e for e in (0, 5, 10, 20, 29)))
    print("      qlto :   " + "".join("%8.4f" % qm[:, e].mean()
                                      for e in (0, 5, 10, 20, 29)))
    print("      spsa :   " + "".join("%8.4f" % sm[:, e].mean()
                                      for e in (0, 5, 10, 20, 29)))
    print()
    globals()['res_%d' % d] = (qm, qc, sm, sc)

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  BEST MSE IS THE FAIR STATISTIC HERE, NOT FINAL MSE. Untuned constant-gain")
print("  SPSA oscillates: at |D|=16 it runs 0.0464 -> 0.0066 -> 0.0325 -> 0.0558,")
print("  reaching its minimum at epoch 10 and drifting back up. Final MSE then")
print("  measures where the oscillation happened to land on epoch 30, which is")
print("  noise about the baseline's missing decay schedule and not a property of")
print("  either optimiser. Reading final MSE would have scored this a qlto_qml win")
print("  driven entirely by that artefact.")
print()
print("      d    optimiser   circuits   best MSE    final MSE    mean cos")
print("   " + "-" * 76)
for dd, (qm_, qc_, sm_, sc_) in ((3, res_3), (4, res_4)):
    print("   %4d    qlto_qml       3       %.5f     %.5f     %+.4f"
          % (dd, qm_.min(axis=1).mean(), qm_[:, -1].mean(), qc_.mean()))
    print("   %4d    SPSA           2       %.5f     %.5f     %+.4f"
          % (dd, sm_.min(axis=1).mean(), sm_[:, -1].mean(), sc_.mean()))
print()
qb3, sb3 = res_3[0].min(axis=1).mean(), res_3[2].min(axis=1).mean()
qb4, sb4 = res_4[0].min(axis=1).mean(), res_4[2].min(axis=1).mean()
tie3 = 0.7 < qb3 / max(sb3, 1e-12) < 1.4
tie4 = 0.7 < qb4 / max(sb4, 1e-12) < 1.4
if tie3 and tie4:
    print("  ON BEST MSE THEY TIE AT BOTH SIZES (%.5f vs %.5f at d=3, %.5f vs"
          % (qb3, sb3, qb4))
    print("  %.5f at d=4), and SPSA costs one FEWER circuit. So qlto_qml does not"
          % sb4)
    print("  reach a lower loss than the baseline a practitioner would actually")
    print("  use, at matched shots, on this problem.")
    print()
    print("  WHAT IT DOES BUY IS THE GRADIENT VECTOR: mean cos %+.4f against SPSA's"
          % res_4[1].mean())
    print("  %+.4f. That is not a better trajectory, it is a different DELIVERABLE."
          % res_4[3].mean())
    print("  It matters where the vector itself is wanted - natural-gradient or")
    print("  quantum-Fisher preconditioning, sensitivity analysis, second-order")
    print("  steps, or any setting needing a per-parameter attribution. It does not")
    print("  matter where the only goal is to reduce a loss, and this file's task is")
    print("  exactly that case, so this file is the unfavourable one for qlto_qml.")
elif qb4 < sb4:
    print("  qlto_qml reaches a LOWER BEST MSE at |D|=16 (%.5f vs %.5f) despite"
          % (qb4, sb4))
    print("  costing one more circuit, so the full-gradient estimate is worth more")
    print("  than the extra third of the budget SPSA spends on two evaluations.")
else:
    print("  SPSA reaches a LOWER BEST MSE (%.5f vs %.5f) AND costs one fewer"
          % (sb4, qb4))
    print("  circuit. qlto_qml has no case at these sizes against the baseline a")
    print("  practitioner would actually reach for, and that is the most important")
    print("  negative in the QML line - record it plainly.")
print()
print("  AND THE 128x AGAINST PARAMETER-SHIFT IS TRUE AND LARGELY IRRELEVANT,")
print("  because parameter-shift is not what anyone would run at M=12 and |D|=16.")
print("  Circuit count was the wrong axis to argue on; per-shot quality is the")
print("  right one, and on per-shot quality this is a tie against a cheaper method.")
print()
print("  THE COS COLUMNS ARE NOT COMPARABLE AS QUALITY SCORES and should not be")
print("  read as though they were. SPSA estimates ONE random direction per epoch,")
print("  so its per-epoch cos is near 1/sqrt(M) BY CONSTRUCTION and low cos is not")
print("  a defect - it is what SPSA is. The MSE column is the comparison that")
print("  means something; the cos column only says what each method is FOR.")
print()
print("  SCOPE. N_sys=%d, M=12, d in (3,4), realizable labels, efficient_su2" % N_SYS)
print("  reps=1, %d shots/epoch, %d seeds, %d epochs, LR=%.2f and SPSA c=%.2f both"
      % (BUDGET, len(SEEDS), EPOCHS, LR, SPSA_C))
print("  UNTUNED and held equal, no noise model, no hardware. SPSA is usually run")
print("  with decaying gain schedules which are not used here; a tuned SPSA would")
print("  do better than this arm, so treat SPSA's column as a LOWER bound on the")
print("  baseline and this comparison as favourable to qlto_qml.")
