"""The radius rescale exponent, swept for the first time. Is p = 0.5 right at large M?

v110 measured V6's cos gap to parameter-shift GROWING with M on real circuits:
+0.019 -> +0.025 -> +0.144 at M = 16, 32, 64 and 16384 total shots. That kills
the hope that v82's synthetic M-scaling advantage would rescue the shot axis.

THE MECHANISM IS IN _radius AND IT IS ONE LINE. V6 rescales

    R_eff = R (N/n)^p          with p = 0.5, shipped, never swept

so at N=4 and M=64 the effective radius is R/4: a nominal 0.45 becomes 0.1125.
Signal goes as sin(R_eff), bias as R_eff^2. Shrinking R buys down a bias that
v108 measured DOMINATING at M=16 (bias^2 1.82e-1 against tr(Cov) 6.17e-2) - but
by M=64 the quartered signal has made V6 SHOT-limited instead, so the same
rescale that fixed the small-M regression is now paying for a problem it no
longer has.

WHERE p = 0.5 CAME FROM, and it is a good reason that does not generalise. At the
V5 -> V6 transition a block of n parameters was seen to displace the state by
about sqrt(n) R, and handing V5's R straight to a 36-parameter block gave cos
0.886 instead of 0.975. That is a real regression and sqrt fixed it - at one M,
at one budget. Nothing was ever swept.

THE PREDICTION. The optimal p should FALL as M grows, because the binding
constraint switches from bias to shot noise. If p = 0.5 is optimal at M=16 and
something smaller wins at M=64, the exponent is a schedule and not a constant,
and v110's growing gap is partly self-inflicted.

WHAT WOULD REFUTE THE IMPROVEMENT. p = 0.5 winning at every M, or a smaller p
winning on cos while losing on converged energy - direction is not the whole
story, and a larger radius that improves cos but breaks the linearisation the
estimator rests on would show up as a worse optimisation trajectory, not a worse
gradient.

TIER (project rule R1): tier A - real circuits, AerSimulator, finite shots -
except the exact gradient, which is the dense reference.
"""
import sys, os, contextlib, io, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import benchmark as B
from nisq_v6 import QLTOv6

N = 4
REPS_LIST = [1, 3, 7]                 # M = 16, 32, 64
BUDGETS = [1 << 14, 1 << 18]
P_LIST = [0.0, 0.25, 0.50]
R_BASE = 0.45
REPEATS = 10
SEED0 = 7000

_, H, _ = B.get_heisenberg_problem(N)


def exact_gradient(anz, theta):
    M = len(theta)
    g = np.zeros(M)
    for i in range(M):
        for s, sgn in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
            t = np.array(theta, float); t[i] += s
            g[i] += sgn * 0.5 * float(
                np.real(Statevector(anz.assign_parameters(t)).expectation_value(H)))
    return g


def mean_cos(ests, gx):
    return float(np.mean([np.dot(e, gx) / (np.linalg.norm(e) * np.linalg.norm(gx))
                          for e in ests]))


print("=" * 100)
print("v111  THE RADIUS EXPONENT, SWEPT")
print("=" * 100)
print("  Heisenberg N=%d, R_base=%.2f, %d repeats, R_eff = R (N/M)^p."
      % (N, R_BASE, REPEATS))
print("  p=0.50 is the shipped value. p=0 disables the rescale (V5 behaviour).")
print()
print("   total shots     M      p      R_eff        cos        vs shipped")
print("  " + "-" * 84)

best_at = {}
for T in BUDGETS:
    for r in REPS_LIST:
        anz = efficient_su2(N, reps=r)
        M = anz.num_parameters
        rng = np.random.default_rng(7)
        theta = rng.uniform(-np.pi, np.pi, M)
        gx = exact_gradient(anz, theta)
        q0 = QLTOv6(anz, H, shot_budget=1, sim_seed=1)
        G = len(q0.groups)
        sh = max(1, T // G)

        row = {}
        for p in P_LIST:
            ests = []
            for k in range(REPEATS):
                be = AerSimulator(seed_simulator=SEED0 + k)
                q = QLTOv6(anz, H, shot_budget=sh, sim_seed=SEED0 + k,
                           backend=be, radius_exponent=p)
                with contextlib.redirect_stdout(io.StringIO()):
                    g, _ = q.sense(theta, R_BASE, list(range(M)))
                ests.append(g)
            row[p] = mean_cos(ests, gx)
        base = row[0.50]
        for p in P_LIST:
            r_eff = R_BASE * (N / float(M)) ** p
            mark = "  <- shipped" if p == 0.50 else ("  %+.5f" % (row[p] - base))
            print("   %10d  %4d   %.2f   %7.4f    %.5f%s"
                  % (T, M, p, r_eff, row[p], mark))
        best_at[(T, M)] = max(row, key=row.get)
        print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("   best p by (budget, M):")
print()
print("      budget      M=16    M=32    M=64")
print("   " + "-" * 52)
for T in BUDGETS:
    vals = [best_at.get((T, m)) for m in (16, 32, 64)]
    print("   %10d    %s" % (T, "    ".join("%.2f" % v if v is not None else " -- "
                                            for v in vals)))
print()
print("  If the best p FALLS as M grows, the shipped constant is a schedule in")
print("  disguise and v110's growing gap is partly self-inflicted - the rescale")
print("  keeps buying down a bias that stopped being the binding constraint.")
print()
print("  If p=0.50 wins everywhere, the rescale is right and V6's large-M deficit")
print("  is structural: reading M components from G circuits costs a linearisation")
print("  whose price grows with M, and no choice of radius escapes it.")
print()
print("  NOT MEASURED HERE: converged energy. A larger radius can improve gradient")
print("  DIRECTION while degrading the optimisation, because grad_step takes its")
print("  step size from the same schedule. Any p adopted on this evidence must be")
print("  re-checked on the benchmark before it becomes a default.")
