"""Shots to reach a TARGET LOSS - the cost metric that matters, replacing v130's cosine.

v130 matched qlto_qml against parameter-shift at equal per-epoch COSINE and got a
break-even of 796 shot-equivalents per circuit. That matching criterion is wrong.

  Cosine measures directional alignment on a single step. An optimiser does not
  need high per-step alignment - SGD converges at cos ~0.1, and SPSA's per-epoch
  cos is ~1/sqrt(M) BY CONSTRUCTION (v129) while it still descends. What matters
  is whether the iterate goes downhill while the trust radius shrinks, and V6
  already shrinks it through _radius / radius_exponent.

v130's own numbers show how much the criterion costs. qlto_qml reached cos 0.6979
on 3,072 shots; parameter-shift needed 12,544 to reach 0.9556. If 0.70 suffices
to converge, QLTO uses 4x FEWER shots and 16x fewer circuits, and the 4x shot
PENALTY v109 reported becomes a 4x shot SAVING. Everything turns on whether the
low-cosine estimate still converges.

WHAT THIS FILE MEASURES INSTEAD. For each arm and each shot budget, run a real
descent driven by that arm's OWN estimate and record

    shots-to-target   total shots spent before MSE first falls below a target
    circuits-to-target  the same in circuits
    reached           whether the target was reached at all inside the epoch cap

The target is set from what the exact-gradient descent achieves on the same
problem, so it is a level both methods could in principle reach rather than an
arbitrary constant.

Then the SAME cost model as v130, but fed with convergence costs:

    r_circ / r_shot  >  (S_q - S_ps) / (C_ps - C_q)

If QLTO reaches the target on fewer shots as well as fewer circuits, there is no
break-even to compute - it dominates outright, and v130's 796 was an artefact of
demanding a per-step accuracy the optimiser never needed.

TIER (project rule R1): tier A. Every descent is driven by circuits on
AerSimulator with finite shots, counted through qlto_qml's instrumented
q.ncircuits / q.nshots. The exact-gradient arm that sets the target is tier B and
is a reference only.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit_aer import AerSimulator

from qlto_qml import QLTOQML

N_SYS = 3
D_QUBITS = 3
MAX_EPOCHS = 40
LR = 0.30
SEEDS = (0, 1, 2)


def setup(seed):
    rng = np.random.default_rng(seed)
    D = 1 << D_QUBITS
    alpha = rng.uniform(-1.0, 1.0, (N_SYS, D_QUBITS))
    core = efficient_su2(N_SYS, reps=1)
    M = core.num_parameters
    probe = QLTOQML(core, alpha, np.zeros(D), shot_budget=4096, sim_seed=1)
    tstar = rng.uniform(-np.pi, np.pi, M)
    y = np.array([probe.f_exact(x, tstar) for x in range(D)])
    theta = rng.uniform(-np.pi, np.pi, M)
    return alpha, core, y, theta, M, D


def exact_floor(seed):
    """What exact-gradient descent reaches. Sets a target both arms could hit."""
    alpha, core, y, theta, M, D = setup(seed)
    q = QLTOQML(core, alpha, y, shot_budget=1024, sim_seed=1)
    best = np.inf
    for _ep in range(MAX_EPOCHS):
        g, w = q.grad_exact(theta)
        best = min(best, float(np.mean(w ** 2)))
        theta = theta - LR * g / max(np.max(np.abs(g)), 1e-12)
    return best


def descend_qlto(shots, seed, target):
    alpha, core, y, theta, M, D = setup(seed)
    q = QLTOQML(core, alpha, y, shot_budget=shots, sim_seed=600 + seed,
                backend=AerSimulator(seed_simulator=600 + seed))
    for _ep in range(MAX_EPOCHS):
        with contextlib.redirect_stdout(io.StringIO()):
            f, _den = q.f_hat(theta)
        if float(np.mean((f - y) ** 2)) <= target:
            return q.ncircuits, q.nshots, True
        with contextlib.redirect_stdout(io.StringIO()):
            g, _ = q.gradient(theta, w=f - y)
        theta = theta - LR * g / max(np.max(np.abs(g)), 1e-12)
    return q.ncircuits, q.nshots, False


def descend_pshift(shots, seed, target):
    alpha, core, y, theta, M, D = setup(seed)
    q = QLTOQML(core, alpha, y, shot_budget=shots, sim_seed=700 + seed,
                backend=AerSimulator(seed_simulator=700 + seed))
    nc = ns = 0
    for _ep in range(MAX_EPOCHS):
        with contextlib.redirect_stdout(io.StringIO()):
            f, _den = q.f_hat(theta, shots=shots)
        nc += 1; ns += shots
        w = f - y
        if float(np.mean(w ** 2)) <= target:
            return nc, ns, True
        g = np.zeros(M)
        for mask, sgn in ((w > 0, +1.0), (w < 0, -1.0)):
            if not mask.any():
                continue
            pw = np.abs(w) * mask
            Z = pw.sum()
            if Z < 1e-12:
                continue
            anz = q.batched(pw / Z)
            gb = np.zeros(M)
            for i in range(M):
                for sh, sg in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
                    t = np.array(theta, float); t[i] += sh
                    bound = anz.assign_parameters(t)
                    bound.measure_all()
                    tq = transpile(bound, q.backend, optimization_level=1)
                    counts = q.backend.run(tq, shots=shots).result().get_counts()
                    nc += 1; ns += shots
                    tot = sum(counts.values())
                    ev = 0.0
                    for bits, ct in counts.items():
                        b = bits.replace(' ', '')
                        # system qubit 0 is CIRCUIT qubit D_QUBITS; see v130
                        ev += (1.0 if b[-(D_QUBITS + 1)] == '0' else -1.0) * ct
                    gb[i] += sg * 0.5 * (ev / tot)
            g += sgn * Z * gb
        g *= 2.0 / D
        theta = theta - LR * g / max(np.max(np.abs(g)), 1e-12)
    return nc, ns, False


print("=" * 100)
print("v131  SHOTS TO A TARGET LOSS, not shots to a target cosine")
print("=" * 100)
print("  v130 matched the arms at equal per-epoch cos and got a 796-shot")
print("  break-even. Wrong criterion: an optimiser needs to descend, not to")
print("  align. SGD converges at cos ~0.1; SPSA's cos is ~1/sqrt(M) by")
print("  construction and it still works. TIER A, both arms drive their own")
print("  descent, counts from q.ncircuits / q.nshots.")
print()

floors = [exact_floor(s) for s in SEEDS]
target = float(np.mean(floors)) * 3.0
print("  Exact-gradient descent reaches mean MSE %.5f over %d seeds."
      % (float(np.mean(floors)), len(SEEDS)))
print("  TARGET = 3x that floor = %.5f - a level both arms could reach."
      % target)
print()
print("-" * 100)
print("      arm            shots/circ   reached   circuits   total shots")
print("   " + "-" * 76)
qbest = pbest = None
for sh in (256, 1024, 4096, 16384):
    r = [descend_qlto(sh, s, target) for s in SEEDS]
    hit = sum(1 for x in r if x[2])
    if hit == len(SEEDS):
        c = float(np.mean([x[0] for x in r])); n = float(np.mean([x[1] for x in r]))
        if qbest is None or n < qbest[1]:
            qbest = (c, n, sh)
    print("      qlto_qml       %7d      %d/%d      %6.1f     %9.0f"
          % (sh, hit, len(SEEDS),
             float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))))
for sh in (64, 256, 1024):
    r = [descend_pshift(sh, s, target) for s in SEEDS]
    hit = sum(1 for x in r if x[2])
    if hit == len(SEEDS):
        c = float(np.mean([x[0] for x in r])); n = float(np.mean([x[1] for x in r]))
        if pbest is None or n < pbest[1]:
            pbest = (c, n, sh)
    print("      param-shift    %7d      %d/%d      %6.1f     %9.0f"
          % (sh, hit, len(SEEDS),
             float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))))
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
if qbest is None or pbest is None:
    print("  One arm never reached the target at any budget tested (qlto %s,"
          % ("ok" if qbest else "FAILED"))
    print("  p-shift %s), so no cost comparison can be made. Widen the sweep."
          % ("ok" if pbest else "FAILED"))
    sys.exit(0)
qc, qn, qs = qbest
pc, pn, ps = pbest
print("  CHEAPEST ROUTE TO MSE <= %.5f, each arm at its own best budget:" % target)
print()
print("      arm            shots/circ   circuits   total shots")
print("   " + "-" * 62)
print("      qlto_qml       %7d      %6.1f     %9.0f" % (qs, qc, qn))
print("      param-shift    %7d      %6.1f     %9.0f" % (ps, pc, pn))
print()
print("      ratio          circuits %.1fx      shots %.2fx"
      % (pc / max(qc, 1e-9), pn / max(qn, 1e-9)))
print()
if qn <= pn and qc <= pc:
    print("  QLTO DOMINATES ON BOTH AXES - fewer circuits AND fewer shots to the same")
    print("  loss. There is no break-even to compute and no r_circ/r_shot assumption")
    print("  to declare: it is cheaper on any machine.")
    print()
    print("  v130's 796-shot break-even was an ARTEFACT of demanding a per-step")
    print("  cosine the optimiser never needed. Matching on alignment charged QLTO")
    print("  for accuracy it does not have to buy, because descent averages over")
    print("  bad steps and the radius shrinks as it converges. WITHDRAW the 796")
    print("  figure; it answers a question nobody is asking.")
elif qn > pn and qc < pc:
    be = (qn - pn) / (pc - qc)
    print("  QLTO still costs more SHOTS (%.2fx) for fewer CIRCUITS (%.1fx), so the"
          % (qn / pn, pc / qc))
    print("  break-even survives the change of criterion, at %.0f shot-equivalents" % be)
    print("  per circuit rather than v130's 796. The direction of v130's conclusion")
    print("  holds; only the number moves.")
else:
    print("  Parameter-shift reaches the target more cheaply on at least one axis")
    print("  (circuits %.1fx, shots %.2fx in QLTO's favour where >1). Read the signs"
          % (pc / max(qc, 1e-9), pn / max(qn, 1e-9)))
    print("  carefully before concluding - this is the arm-by-arm cost, not a")
    print("  verdict on the method.")
print()
print("  SCOPE. N_sys=%d, |D|=%d, M=12, G=1, %d seeds, %d-epoch cap, LR=%.2f held"
      % (N_SYS, 1 << D_QUBITS, len(SEEDS), MAX_EPOCHS, LR))
print("  equal across arms and UNTUNED, realizable labels, no noise model, no")
print("  hardware. Target is 3x the exact-gradient floor; a tighter target would")
print("  favour the lower-variance arm and a looser one the cheaper-per-epoch arm,")
print("  so the ratio is target-dependent and one target is not a scaling law.")
