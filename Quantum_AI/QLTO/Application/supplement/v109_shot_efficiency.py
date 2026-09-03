"""Does V6 need its shots? cos versus TOTAL SHOT BUDGET, both estimators, on circuits.

The standing caveat says "at matched TOTAL SHOTS the advantage is circuits, not
shots". v108 measured one matched point and found something the caveat does not
account for:

    M=16, 131072 total shots
      V6        3 circuits   43690 shots/circ   tr(Cov) 6.17e-2   bias^2 1.82e-1
      p-shift  96 circuits    1365 shots/circ   tr(Cov) 3.07e-2   bias^2 1.57e-3

V6 IS ALREADY BIAS-DOMINATED THERE - its bias^2 is 3x its variance. Shots beyond
the point where variance falls under bias buy nothing. So a single matched-total
comparison puts V6 at a budget it cannot use, and then reports that it did not
benefit. The question the caveat should be answering is not "who wins at one
budget" but "what does each estimator's error do as the budget moves", and in
particular how FEW shots V6 needs to reach a given direction quality.

WHY cos IS THE RIGHT METRIC HERE, not MSE. grad_step is MAX-NORMALISED: it uses
the gradient's direction and discards its magnitude entirely. A bias that
rescales the gradient therefore costs the optimiser nothing, while the same bias
dominates MSE. Both are reported, because MSE is the honest answer to "is this
estimator accurate" and cos is the honest answer to "does this optimiser step in
the right direction".

R IS SWEPT, because the bias-variance trade is the whole mechanism. bias ~ R^2
and variance ~ 1/(R^2 S), so the optimal R MOVES with the budget - v92's point.
Holding R fixed while sweeping shots measures a schedule, not an estimator.

PARAMETER-SHIFT BECOMES INFEASIBLE FIRST, and that is a structural fact rather
than a tuning artefact: it needs 2MG distinct circuits, so below 2MG total shots
it cannot allocate even one shot per circuit. V6 needs G. Rows where p-shift is
below one shot per circuit are marked.

TIER (project rule R1): tier A - real circuits, AerSimulator, finite shots -
except g_exact, the dense reference the circuits are checked against.
"""
import sys, os, contextlib, io
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import benchmark as B
from nisq_v6 import QLTOv6

N = 4
BUDGETS = [1 << k for k in (10, 12, 14, 16, 18)]
R_LIST = [0.30, 0.45, 0.60]
REPEATS = 12
SEED0 = 4000

anz = efficient_su2(N, reps=1)
_, H, _ = B.get_heisenberg_problem(N)
M = anz.num_parameters
rng = np.random.default_rng(7)
theta = rng.uniform(-np.pi, np.pi, M)

q0 = QLTOv6(anz, H, shot_budget=1, sim_seed=1)
GROUPS = q0.groups
G = len(GROUPS)


def exact_gradient():
    g = np.zeros(M)
    for i in range(M):
        for s, sgn in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
            t = np.array(theta, float); t[i] += s
            g[i] += sgn * 0.5 * float(
                np.real(Statevector(anz.assign_parameters(t)).expectation_value(H)))
    return g


GX = exact_gradient()


# ---- parameter-shift with CACHED transpiled templates -----------------------
_tmpl = {}


def _template(backend, gi):
    key = (id(backend), gi)
    if key in _tmpl:
        return _tmpl[key]
    grp = GROUPS[gi]
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
    t = transpile(qc, backend, optimization_level=1)
    _tmpl[key] = (t, list(anz.parameters), grp.paulis.to_labels(),
                  np.real(grp.coeffs))
    return _tmpl[key]


def energy(backend, th, shots):
    tot = 0.0
    for gi in range(G):
        t, params, labels, coeffs = _template(backend, gi)
        bound = t.assign_parameters(
            {params[i]: float(th[i]) for i in range(M)}, inplace=False)
        counts = backend.run(bound, shots=shots).result().get_counts()
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


def pshift(backend, shots):
    g = np.zeros(M)
    for i in range(M):
        tp = np.array(theta, float); tp[i] += np.pi / 2
        tm = np.array(theta, float); tm[i] -= np.pi / 2
        g[i] = 0.5 * (energy(backend, tp, shots) - energy(backend, tm, shots))
    return g


def summarise(ests):
    ests = np.array(ests)
    mu = ests.mean(0)
    cos = float(np.mean([np.dot(e, GX) / (np.linalg.norm(e) * np.linalg.norm(GX))
                         for e in ests]))
    mse = float(np.mean([np.sum((e - GX) ** 2) for e in ests]))
    return cos, mse


print("=" * 104)
print("v109  SHOT EFFICIENCY:  cos and MSE versus TOTAL shot budget")
print("=" * 104)
print("  Heisenberg N=%d, M=%d, G=%d, |grad E| = %.4f, %d repeats."
      % (N, M, G, np.linalg.norm(GX), REPEATS))
print("  V6 spends G=%d circuits; parameter-shift spends 2MG=%d." % (G, 2 * M * G))
print("  TIER A. cos is averaged PER REPEAT, not on the mean estimate.")
print()
print("   total shots   method        circuits  shots/circ      cos        MSE")
print("  " + "-" * 88)

table = {}
for T in BUDGETS:
    # --- parameter-shift
    nc = 2 * M * G
    sh = T // nc
    if sh >= 1:
        ests = []
        for k in range(REPEATS):
            be = AerSimulator(seed_simulator=SEED0 + 900 + k)
            ests.append(pshift(be, sh))
        cos, mse = summarise(ests)
        table[('ps', T)] = (cos, mse)
        print("   %10d   p-shift        %5d    %8d    %.5f   %.4e"
              % (T, nc, sh, cos, mse))
    else:
        print("   %10d   p-shift        %5d    %8s    %s"
              % (T, nc, "<1", "INFEASIBLE - fewer shots than circuits"))

    # --- V6 at each R
    sh6 = max(1, T // G)
    best = None
    for R in R_LIST:
        ests = []
        for k in range(REPEATS):
            be = AerSimulator(seed_simulator=SEED0 + k)
            q = QLTOv6(anz, H, shot_budget=sh6, sim_seed=SEED0 + k, backend=be)
            with contextlib.redirect_stdout(io.StringIO()):
                g, _ = q.sense(theta, R, list(range(M)))
            ests.append(g)
        cos, mse = summarise(ests)
        tag = "V6 R=%.2f" % R
        print("   %10d   %-12s   %5d    %8d    %.5f   %.4e"
              % (T, tag, G, sh6, cos, mse))
        if best is None or cos > best[0]:
            best = (cos, mse, R)
    table[('v6', T)] = best
    print()

print("=" * 104)
print("READING IT")
print("=" * 104)
print("   total shots     p-shift cos     V6 best cos (R)      V6 circuits / PS circuits")
print("  " + "-" * 88)
for T in BUDGETS:
    v = table.get(('v6', T))
    p = table.get(('ps', T))
    ps_s = "%.5f" % p[0] if p else "infeasible"
    print("   %10d      %-12s    %.5f (R=%.2f)          %d / %d"
          % (T, ps_s, v[0], v[2], G, 2 * M * G))
print()
print("  THE QUESTION THE CAVEAT ANSWERS BADLY. 'At matched total shots the")
print("  advantage is circuits, not shots' is a statement about ONE budget. Read")
print("  the column instead: find the smallest budget at which V6 reaches the cos")
print("  that parameter-shift needs its full budget to reach. The ratio of those")
print("  two budgets is the shot-side advantage, and it is either there or it is")
print("  not - one matched point cannot tell you.")
print()
print("  V6 is bias-limited well before it is shot-limited (v108: bias^2 3x")
print("  variance at 43690 shots/circuit), so its cos should FLATTEN with budget")
print("  while parameter-shift's keeps climbing. Where it flattens is the shot")
print("  count V6 actually needs; everything above that is spent for nothing.")
print()
print("  Scope: one problem, one theta, one N, %d repeats, R swept over %s and the")
print("  best reported - so V6 is given its best radius at every budget while")
print("  parameter-shift has no equivalent knob. That favours V6 and is stated.")
print("  R swept over:", R_LIST)
