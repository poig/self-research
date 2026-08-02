"""V4 candidate 3: the WALK's Trotter error, which is far worse than sensing's.

_execute_walk evolves with LieTrotter(reps=1) at time = delta_t*pi. At the tuned
delta_t=0.3 that is t=0.942, against the sensing path's tau=0.106 with reps=2.
First-order Trotter error goes as t^2/r, so

    walk     0.942^2 / 1   = 0.888
    sensing  0.106^2 / 2   = 0.0056

about 158x more error in the walk than in the sensing that was just fixed. And
this evolution is not incidental - it is the step that imprints each vertex's
energy as a relative phase between the ancilla branches, i.e. the entire
mechanism by which the walk knows which vertices are good. If it is
mis-Trotterised the walk drifts toward the wrong vertices.

Unlike the sensing case there is no exact target to compare against, so measure
the thing that matters: the energy reached after one full sweep of blocks, as a
function of the walk's product formula. A better formula that does not improve
the energy means the walk is insensitive to this, which is worth knowing too.
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
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

def walk(q, center, k_steps, delta_t, radius, act, grad, synth):
    n = len(act)
    anc = AncillaRegister(1, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(1, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, radius, act),
              list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=delta_t * np.pi,
                                 synthesis=synth).control(1),
              [anc[0]] + list(sysr))
    gl = grad[act]
    dg = 1.0 / np.sqrt(max(radius, 1e-9))
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma, beta = s * np.pi * delta_t, (1.0 - s) * np.pi * delta_t
        for i in range(n):
            qc.crz(gl[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
        for i in range(n):
            qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return qc, q._decode_walk(q._run(qc), center, act, radius)

N, R, K, DT, REP = 4, 0.6, 15, 0.3, 6
ansatz, H = heis(N)
q = Q(ansatz, H, shot_budget=32768, num_ancillas=1)
BLK = [b['params'] for b in q.layers]

SYNTHS = [
    ("lie   r=1  SHIP", lambda: LieTrotter(reps=1)),
    ("lie   r=2      ", lambda: LieTrotter(reps=2)),
    ("lie   r=4      ", lambda: LieTrotter(reps=4)),
    ("lie   r=8      ", lambda: LieTrotter(reps=8)),
    ("suz2  r=1      ", lambda: SuzukiTrotter(order=2, reps=1)),
    ("suz2  r=2      ", lambda: SuzukiTrotter(order=2, reps=2)),
    ("suz2  r=4      ", lambda: SuzukiTrotter(order=2, reps=4)),
]

print("=" * 82)
print(f"Walk product formula A/B.  k={K}, dt={DT} (t={DT*np.pi:.3f}), R={R}, "
      f"{REP} seeds")
print("=" * 82)
print(f"  {'synthesis':<17}{'E mean':>10}{'std':>9}{'E best':>10}{'depth':>8}")
print("  " + "-" * 54)
rows = []
for label, mk in SYNTHS:
    es = []
    for seed in range(REP):
        c = np.random.RandomState(seed).uniform(-np.pi, np.pi,
                                                ansatz.num_parameters)
        G = [q.sense_gradient(c, R, a) for a in BLK]
        p = c.copy()
        for bi, a in enumerate(BLK):
            _, p[a] = walk(q, p, K, DT, R, a, G[bi], mk())
        es.append(energy_at(ansatz, H, p))
    c0 = np.random.RandomState(0).uniform(-np.pi, np.pi, ansatz.num_parameters)
    g0 = q.sense_gradient(c0, R, BLK[0])
    qc, _ = walk(q, c0, K, DT, R, BLK[0], g0, mk())
    d = transpile(qc, q.backend, optimization_level=1).depth()
    rows.append((label, float(np.mean(es)), float(np.std(es)), min(es), d))
    print(f"  {label:<17}{np.mean(es):>10.4f}{np.std(es):>9.4f}"
          f"{min(es):>10.4f}{d:>8}", flush=True)

base = rows[0]
print()
print(f"  shipping mean {base[1]:.4f} at depth {base[4]}")
for lbl, m, sd, b, d in rows[1:]:
    sig = abs(m - base[1]) / max(np.hypot(sd, base[2]) / np.sqrt(REP), 1e-9)
    print(f"  {lbl}: {m - base[1]:+.4f} ({sig:.1f} sigma), depth x{d/base[4]:.1f}")
print()
print("  If none of these move the energy, the walk is INSENSITIVE to its own")
print("  Trotter error - the drift only needs the right ordering of vertex")
print("  energies, not their exact values, and reps=1 is already correct.")