"""Does V6's shot deficit close as M grows, or is "circuits, not shots" permanent?

v109 measured, on real circuits at M=16, that parameter-shift reaches any given
cos on FEWER TOTAL SHOTS than V6 - about 4x fewer - while spending 32x more
circuits. That looks like a clean confirmation of the standing caveat.

BUT M=16 IS THE WORST CASE FOR V6 AND THE CAVEAT NEVER SAYS SO.

  v108 measured the real Heisenberg / efficient_su2 landscape at
  |grad E| ~ M^0.130 and mean|g_i| ~ M^-0.491, which is v82's TOTAL-NORM-FIXED
  regime (predicting 0.0 and -0.5), not the wide-ansatz regime the caveat quotes.

  In that regime v82 fitted tr(Cov) ~ M^1.006 for V6 against M^2.000 for
  parameter-shift, with the ratio PS/V6 climbing 1.67 -> 3.30 -> 6.60 -> 13.13
  -> 26.34 across M = 8, 16, 32, 64, 128.

So two effects pull opposite ways as M grows:

    V6's BIAS is M-independent          - it is set by R, not by parameter count
    parameter-shift's VARIANCE grows    - T/(2MG) shots per circuit, so error
                                          per component grows with M

At M=16 the bias dominates and parameter-shift wins the shot axis. If v82's
scaling is real on circuits, there is an M where that reverses. This file looks
for it, and it is the measurement that decides whether the caveat is a law or an
artefact of the one size it was checked at.

DESIGN. M is grown by ansatz reps at FIXED N, so the Hamiltonian, G and the
qubit count never change and M is the only moving part. cos is measured against
the exact gradient at matched TOTAL shots, per repeat, with R given its best of
a small sweep at every point - which favours V6 and is stated.

WHAT WOULD SETTLE IT.

  the PS-minus-V6 cos gap SHRINKS with M   -> the caveat is an M=16 artefact and
                                              must be restated with M attached
  the gap is FLAT or GROWS                 -> the caveat is right, v82's synthetic
                                              scaling does not survive the bias,
                                              and it should say so

TIER (project rule R1): tier A - real circuits, AerSimulator, finite shots -
except the exact gradient, the dense reference the circuits are checked against.
"""
import sys, os, contextlib, io, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import benchmark as B
from nisq_v6 import QLTOv6

N = 4
REPS_LIST = [1, 3, 7]              # M = 2N(r+1) = 16, 32, 64
BUDGETS = [1 << 14, 1 << 18]
R_LIST = [0.45, 0.60]
REPEATS = 10
SEED0 = 6000

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


class PS:
    """Parameter-shift with cached transpiled templates, one per group."""

    def __init__(self, anz, groups, backend):
        self.anz, self.groups, self.be = anz, groups, backend
        self.M = anz.num_parameters
        self.t = []
        for grp in groups:
            qc = anz.copy()
            axis = {}
            for lbl in grp.paulis.to_labels():
                for q, ch in enumerate(reversed(lbl)):
                    if ch != 'I':
                        axis[q] = ch
            for q, ch in axis.items():
                if ch == 'X':
                    qc.h(q)
                elif ch == 'Y':
                    qc.sdg(q); qc.h(q)
            qc.measure_all()
            self.t.append((transpile(qc, backend, optimization_level=1),
                           list(anz.parameters), grp.paulis.to_labels(),
                           np.real(grp.coeffs)))

    def energy(self, th, shots):
        tot = 0.0
        for tq, params, labels, coeffs in self.t:
            bound = tq.assign_parameters(
                {params[i]: float(th[i]) for i in range(self.M)}, inplace=False)
            counts = self.be.run(bound, shots=shots).result().get_counts()
            n = sum(counts.values())
            acc = 0.0
            for bit, c in counts.items():
                b = bit.replace(' ', '')[::-1]
                for lbl, co in zip(labels, coeffs):
                    s = 1
                    for q, ch in enumerate(reversed(lbl)):
                        if ch != 'I' and b[q] == '1':
                            s = -s
                    acc += co * s * c
            tot += acc / max(n, 1)
        return tot

    def grad(self, theta, shots):
        g = np.zeros(self.M)
        for i in range(self.M):
            tp = np.array(theta, float); tp[i] += np.pi / 2
            tm = np.array(theta, float); tm[i] -= np.pi / 2
            g[i] = 0.5 * (self.energy(tp, shots) - self.energy(tm, shots))
        return g


def mean_cos(ests, gx):
    return float(np.mean([np.dot(e, gx) / (np.linalg.norm(e) * np.linalg.norm(gx))
                          for e in ests]))


print("=" * 100)
print("v110  DOES THE SHOT DEFICIT CLOSE WITH M?")
print("=" * 100)
print("  Heisenberg N=%d, M grown by ansatz reps at fixed N, %d repeats."
      % (N, REPEATS))
print("  R given its best of %s at every point - favours V6, and is stated." % R_LIST)
print("  TIER A except the exact-gradient reference.")
print()
print("   total shots     M   PS circuits   PS cos     V6 cos (R)     gap (PS-V6)")
print("  " + "-" * 88)

gaps = {}
for T in BUDGETS:
    for r in REPS_LIST:
        t0 = time.time()
        anz = efficient_su2(N, reps=r)
        M = anz.num_parameters
        rng = np.random.default_rng(7)
        theta = rng.uniform(-np.pi, np.pi, M)
        gx = exact_gradient(anz, theta)

        q0 = QLTOv6(anz, H, shot_budget=1, sim_seed=1)
        G = len(q0.groups)
        nc = 2 * M * G
        sh_ps = max(1, T // nc)
        sh_v6 = max(1, T // G)

        ests = []
        for k in range(REPEATS):
            be = AerSimulator(seed_simulator=SEED0 + 700 + k)
            ests.append(PS(anz, q0.groups, be).grad(theta, sh_ps))
        c_ps = mean_cos(ests, gx)

        best = (-2.0, None)
        for R in R_LIST:
            ests = []
            for k in range(REPEATS):
                be = AerSimulator(seed_simulator=SEED0 + k)
                q = QLTOv6(anz, H, shot_budget=sh_v6, sim_seed=SEED0 + k,
                           backend=be)
                with contextlib.redirect_stdout(io.StringIO()):
                    g, _ = q.sense(theta, R, list(range(M)))
                ests.append(g)
            c = mean_cos(ests, gx)
            if c > best[0]:
                best = (c, R)
        gaps[(T, M)] = c_ps - best[0]
        print("   %10d  %4d      %5d      %.5f    %.5f (%.2f)     %+.5f   [%.0fs]"
              % (T, M, nc, c_ps, best[0], best[1], c_ps - best[0], time.time() - t0))
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("   the PS-minus-V6 cos gap, as M grows at fixed budget:")
print()
print("      budget        M=16       M=32       M=64      trend")
print("   " + "-" * 72)
for T in BUDGETS:
    row = [gaps.get((T, m)) for m in (16, 32, 64)]
    if all(v is not None for v in row):
        trend = "SHRINKS" if row[-1] < row[0] else "flat or grows"
        print("   %10d   %+.5f   %+.5f   %+.5f    %s"
              % (T, row[0], row[1], row[2], trend))
print()
print("  A gap that SHRINKS means v82's M-scaling is real on circuits and the")
print("  caveat is an M=16 artefact - 'the advantage is circuits, not shots' would")
print("  then need 'at small M' attached to it, and the crossover located.")
print()
print("  A gap that is FLAT or GROWS means V6's R-bias, which does not fall with")
print("  M, outruns parameter-shift's growing per-component variance over this")
print("  range - and the caveat stands as written, now on tier-A evidence rather")
print("  than v82's synthetic model.")
print()
print("  Either way the bias is the term to attack, since it is what v108 measured")
print("  dominating V6's error (bias^2 1.82e-1 against tr(Cov) 6.17e-2 at M=16).")
print()
print("  Scope: one N, one theta, one problem family, %d repeats, two budgets," % REPEATS)
print("  three M. Depth grows with reps, so the M axis is not perfectly clean -")
print("  a deeper ansatz is also a harder landscape.")
