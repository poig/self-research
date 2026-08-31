"""Convergence per TOTAL SHOT - the axis the benchmark never plotted.

Every comparison in this project matches SHOTS PER CIRCUIT (8192 each) and then
counts circuits. That silently hands parameter-shift 2MG times more total shots,
so "V6 converges competitively at 60 circuits against AdamW's 3840" is a
circuit-axis statement and says nothing about the shot axis.

v109/v110 tried to close that gap by measuring gradient QUALITY per shot, and
found parameter-shift ahead at every budget and pulling further ahead with M.
That measurement answers a question the optimiser never asks.

WHY IT IS THE WRONG QUESTION. grad_step is MAX-NORMALISED: it takes
alpha*R*g/max|g| per layer, discarding the gradient's magnitude entirely and
using only its direction, and only to the accuracy needed not to step backwards.
On top of that R decays over epochs, so the trust region shrinks as the run
proceeds - coarse cheap directions early where precision is wasted, finer ones
late. A per-point cos measurement is blind to both.

AND PARAMETER-SHIFT HAS NO SUCH KNOB. It is exact at +-pi/2 by a trigonometric
identity, so it cannot trade accuracy for cost. Its only lever is shots. V6 can
buy a bad gradient cheaply on purpose; parameter-shift pays full price every
epoch including the early ones where a rough direction would have done.

THE CONTROL THAT KEEPS THIS HONEST. If V6 wins on the shot axis, the obvious
referee question is whether the win is the quantum multiplexing or merely the
annealing schedule - which is a classical idea that parameter-shift could adopt.
So there are three arms and they differ in exactly one thing at a time:

    A  V6 gradient        + V6 step rule + R schedule + flat shots
    B  parameter-shift    + V6 step rule + R schedule + flat shots
    C  parameter-shift    + V6 step rule + R schedule + ANNEALED shots

  A vs B  isolates the ESTIMATOR, everything else identical.
  B vs C  isolates the SHOT SCHEDULE, estimator identical.

If C closes the gap to A, the advantage was the schedule and V6 packages a
classical idea well. If it does not, the multiplexing is doing the work.

TIER (project rule R1): tier A for every gradient and every step - real circuits,
AerSimulator, finite shots. Energies are REPORTED with Statevector, 0 circuits,
which is what benchmark.py does and is tier B reporting of a final number only;
no gradient or step ever sees it.
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
EPOCHS = 20
BUDGETS = [1 << 17, 1 << 20]
SEEDS = [11, 22, 33]
R0, R_DECAY = 0.6, 0.9
SEED0 = 8000

anz = efficient_su2(N, reps=1)
_, H, name = B.get_heisenberg_problem(N)
M = anz.num_parameters
EXACT = float(np.min(np.linalg.eigvalsh(H.to_matrix())))


class PSGrad:
    """Parameter-shift gradient, cached transpiled templates."""

    def __init__(self, groups, backend):
        self.be = backend
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
                {params[i]: float(th[i]) for i in range(M)}, inplace=False)
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
        g = np.zeros(M)
        for i in range(M):
            tp = np.array(theta, float); tp[i] += np.pi / 2
            tm = np.array(theta, float); tm[i] -= np.pi / 2
            g[i] = 0.5 * (self.energy(tp, shots) - self.energy(tm, shots))
        return g


def exact_energy(th):
    return float(np.real(Statevector(
        anz.assign_parameters(np.asarray(th, float))).expectation_value(H)))


def run_arm(arm, T, seed):
    """One optimisation under a fixed TOTAL shot budget T. Same step rule for all."""
    be = AerSimulator(seed_simulator=seed)
    q = QLTOv6(anz, H, shot_budget=1, sim_seed=seed, backend=be)
    G = len(q.groups)
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, M)

    if arm == 'A':
        per_epoch = T // EPOCHS
        sh = [max(1, per_epoch // G)] * EPOCHS
    else:
        nc = 2 * M * G
        if arm == 'B':
            per_epoch = T // EPOCHS
            sh = [max(1, per_epoch // nc)] * EPOCHS
        else:                                  # C: annealed, ramp 1 -> 2*mean
            w = np.linspace(0.15, 1.85, EPOCHS)
            w = w / w.sum()
            sh = [max(1, int(T * wi) // nc) for wi in w]
        ps = PSGrad(q.groups, be)

    spent = 0
    for ep in range(EPOCHS):
        R = max(R0 * (R_DECAY ** ep), 1e-4)
        if arm == 'A':
            q.shot_budget = int(sh[ep])
            with contextlib.redirect_stdout(io.StringIO()):
                g, _ = q.sense(theta, R, list(range(M)))
            spent += G * sh[ep]
        else:
            g = ps.grad(theta, int(sh[ep]))
            spent += 2 * M * G * sh[ep]
        theta = q.grad_step(theta, R, list(range(M)), g)
    return exact_energy(theta), spent


print("=" * 100)
print("v112  CONVERGENCE PER TOTAL SHOT")
print("=" * 100)
print("  %s, M=%d, exact ground state %.6f, %d epochs, %d seeds."
      % (name, M, EXACT, EPOCHS, len(SEEDS)))
print("  Identical step rule and R schedule in every arm; only the marked thing differs.")
print("  TIER A for gradients and steps; energies reported by Statevector.")
print()
print("   budget     arm                                 spent      energy         gap")
print("  " + "-" * 90)

res = {}
for T in BUDGETS:
    for arm, label in (('A', 'V6            (flat shots)'),
                       ('B', 'p-shift       (flat shots)'),
                       ('C', 'p-shift  (ANNEALED shots)')):
        t0 = time.time()
        es, sp = [], []
        for sd in SEEDS:
            e, s = run_arm(arm, T, SEED0 + sd)
            es.append(e); sp.append(s)
        res[(T, arm)] = float(np.mean(es))
        print("   %8.1e   %-30s %9.2e   %8.4f    %+.4f   [%.0fs]"
              % (T, label, np.mean(sp), np.mean(es), np.mean(es) - EXACT,
                 time.time() - t0))
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("   budget        A: V6      B: PS flat   C: PS annealed     A-B        C-B")
print("  " + "-" * 88)
for T in BUDGETS:
    a, b, c = res[(T, 'A')], res[(T, 'B')], res[(T, 'C')]
    print("   %8.1e    %8.4f    %8.4f     %8.4f      %+.4f    %+.4f"
          % (T, a, b, c, a - b, c - b))
print()
print("  A - B is the ESTIMATOR at matched total shots, matched step rule, matched")
print("  schedule. NEGATIVE means V6 reaches a lower energy for the same shots -")
print("  which would contradict v109/v110's per-point reading and show that")
print("  gradient quality was never the binding constraint.")
print()
print("  C - B is what SHOT ANNEALING alone buys parameter-shift. If C reaches A,")
print("  the advantage is the schedule, not the multiplexing, and V6's real")
print("  contribution is packaging rather than physics.")
print()
print("  Both matter. A win on A-B that survives C is the strongest form of the")
print("  claim this project can make on the shot axis, and it has never been")
print("  measured because the benchmark matches shots PER CIRCUIT.")
print()
print("  Scope: one problem, one N, %d seeds, %d epochs, one R schedule which was"
      % (len(SEEDS), EPOCHS))
print("  tuned for V6 and is therefore charged against parameter-shift too.")
