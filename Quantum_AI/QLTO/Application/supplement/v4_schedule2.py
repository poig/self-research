"""Schedule decoupling, with the step-size confound removed.

v4_schedule showed the normalisation does exactly what the algebra says: total
angle becomes k-independent and |move| goes flat in k. But it also read WORSE on
energy at fixed dt - because a k-independent total angle is a SMALLER angle than
the current schedule accumulates at k>=5, so it simply steps less far. Comparing
at fixed dt measures step length, not schedule quality. Same trap that made the
raw natural gradient look bad until natural_norm matched its magnitude.

Fair test: sweep dt for BOTH schedules and compare
  (a) the best energy each can reach, dt tuned per schedule
  (b) how much that best moves as k changes - the actual point of the fix

If normalised matches on (a) and is far flatter on (b), the change is a strict
improvement in tunability: k stops being a step-size knob in disguise, so k and
dt can be tuned independently instead of trading off against each other.
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister)
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return efficient_su2(N, reps=1), SparsePauliOp.from_list(ops)

EST = StatevectorEstimator()
def energy_at(ansatz, H, p):
    return float(EST.run([(ansatz, H, np.asarray([p]))]).result()[0].data.evs.ravel()[0])

def walk(q, center, k_steps, delta_t, radius, act, grad, normalise):
    n = len(act)
    anc = AncillaRegister(1, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(1, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, radius, act),
              list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=delta_t * np.pi,
                                 synthesis=LieTrotter(reps=1)).control(1),
              [anc[0]] + list(sysr))
    gl = grad[act]
    dg = 1.0 / np.sqrt(max(radius, 1e-9))
    sc = (2.0 / k_steps) if normalise else 1.0
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma = s * np.pi * delta_t * sc
        beta = (1.0 - s) * np.pi * delta_t * sc
        for i in range(n):
            qc.crz(gl[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
        for i in range(n):
            qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_walk(q._run(qc), center, act, radius)

N, R = 4, 0.6
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
q = Q(ansatz, H, shot_budget=65536, num_ancillas=1)
BLK = [b['params'] for b in q.layers]
G = [q.sense_gradient(c, R, a) for a in BLK]
E0 = energy_at(ansatz, H, c)

KS = (2, 5, 10, 15, 20, 30)
DTS = (0.15, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0)

def step_energy(k, dt, norm):
    p = c.copy()
    for bi, a in enumerate(BLK):
        p[a] = walk(q, p, k, dt, R, a, G[bi], norm)
    return energy_at(ansatz, H, p)

print("=" * 84)
print(f"Schedule A/B with dt tuned per schedule.  start E = {E0:.4f}")
print("=" * 84)
for norm in (False, True):
    tag = "NORMALISED (2/k)" if norm else "CURRENT"
    print(f"\n  --- {tag} ---")
    print("  " + f"{'k':>4}" + "".join(f"{'dt=%g' % d:>9}" for d in DTS)
          + f"{'best':>9}")
    print("  " + "-" * (4 + 9 * len(DTS) + 9))
    bests = []
    for k in KS:
        row, vals = f"  {k:>4}", []
        for dt in DTS:
            e = step_energy(k, dt, norm)
            vals.append(e); row += f"{e:>9.3f}"
        bests.append(min(vals))
        print(row + f"{min(vals):>9.3f}", flush=True)
    print(f"  best over dt, per k : {[f'{b:.3f}' for b in bests]}")
    print(f"  spread across k     : {max(bests) - min(bests):.4f}"
          f"   (smaller = k and dt properly decoupled)")
    print(f"  best overall        : {min(bests):.4f}")
