"""V4 candidate 2: decouple k_steps from step size.

DERIVATION (no simulation needed for the claim, only for the consequence).
_execute_walk uses s = (step+0.5)/k, gamma = s*pi*dt, beta = (1-s)*pi*dt, and

    sum_step s = sum_{j=0}^{k-1} (j+0.5)/k = k/2

so the TOTAL angles accumulated over the walk are

    total drift_i = grad_i * drift_gain * pi^2 * dt * k / 4
    total mixing  = pi * dt * k / 2

Both LINEAR IN k. So k_steps is not a resolution knob, it is a step-size
multiplier wearing a resolution costume. Raising k to "resolve better" also
takes a proportionally bigger step, which is why the docs record energy peaking
then declining with k while normalized_entropy falls monotonically, and why
"walk until concentrated" overshoots.

FIX: multiply both by 2/k so sum(gamma) = sum(beta) = pi*dt regardless of k.
Then the walk is a fixed-total-time annealing sweep that k merely discretises
more finely - it CONVERGES as k grows instead of scaling with it, and dt alone
sets the step.

This test measures the consequence: distance moved from the centre, and the
decoded energy, as a function of k, with and without the normalisation.
Prediction: current grows with k until the +-R box saturates it; normalised is
flat in k and converges.
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
    w = q.build_w_gate(param, sysr, center, radius, act)
    qc.append(w, list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=delta_t * np.pi,
                                 synthesis=LieTrotter(reps=1)).control(1),
              [anc[0]] + list(sysr))
    gl = grad[act]
    dg = 1.0 / np.sqrt(max(radius, 1e-9))
    scale = (2.0 / k_steps) if normalise else 1.0
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma = s * np.pi * delta_t * scale
        beta = (1.0 - s) * np.pi * delta_t * scale
        for i in range(n):
            qc.crz(gl[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
        for i in range(n):
            qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_walk(q._run(qc), center, act, radius)


N, R, DT = 4, 0.6, 0.3
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
q = Q(ansatz, H, shot_budget=65536, num_ancillas=1)
BLK = [b['params'] for b in q.layers]
G = [q.sense_gradient(c, R, a) for a in BLK]

print("=" * 79)
print("Analytic: total accumulated angle over the walk")
print("=" * 79)
print(f"  {'k':>5}{'sum(s)':>10}{'sum gamma/(pi dt)':>20}{'normalised':>14}")
print("  " + "-" * 49)
for k in (1, 2, 5, 10, 15, 20, 30):
    ss = sum((j + 0.5) / k for j in range(k))
    print(f"  {k:>5}{ss:>10.2f}{ss:>20.2f}{ss * 2.0 / k:>14.2f}")
print("  sum(s) = k/2 exactly -> current schedule's total angle is LINEAR in k;")
print("  the 2/k factor makes it exactly 1.0 for every k.")

print()
print("=" * 79)
print("Consequence: distance moved and energy vs k")
print("=" * 79)
print(f"  {'k':>5}{'|move| now':>13}{'E now':>11}{'|move| norm':>14}{'E norm':>11}")
print("  " + "-" * 54)
for k in (1, 2, 5, 10, 15, 20, 30):
    pn, pm = c.copy(), c.copy()
    for bi, a in enumerate(BLK):
        pn[a] = walk(q, pn, k, DT, R, a, G[bi], False)
        pm[a] = walk(q, pm, k, DT, R, a, G[bi], True)
    print(f"  {k:>5}{np.linalg.norm(pn - c):>13.4f}{energy_at(ansatz, H, pn):>11.4f}"
          f"{np.linalg.norm(pm - c):>14.4f}{energy_at(ansatz, H, pm):>11.4f}",
          flush=True)
print("  (flat |move| under 'norm' = k decoupled from step size; then k is a")
print("   convergence knob and dt alone sets how far the walk steps)")
