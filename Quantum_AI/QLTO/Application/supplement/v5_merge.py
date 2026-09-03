"""Idea #3: merge CRZ+CRX into one tilted-axis controlled rotation.

Per step per qubit the walk does CRZ(alpha) then CRX(beta) - two controlled gates,
4 CX after decomposition. Since
    RY(phi) Z RY(phi)^dag = Z cos phi + X sin phi
setting theta = sqrt(alpha^2+beta^2) and phi = atan2(beta, alpha) gives
    RY(phi) RZ(theta) RY(-phi) = exp(-i(alpha Z + beta X)/2)
and because controlled-(V W V^dag) = V * controlled-W * V^dag with V uncontrolled,
the controlled version is
    RY(-phi) ; CRZ(theta) ; RY(phi)
i.e. ONE controlled gate (2 CX) plus two cheap single-qubit rotations. CX count
halves.

THE CATCH, stated up front: exp(-i(alphaZ+betaX)/2) is not RX(beta)RZ(alpha) - they
differ by O(alpha*beta) BCH terms, and the angles here are NOT small. At dt=0.3,
k=15, R=0.6, drift_gain=1.29 the drift angle reaches alpha ~ 3.8 for a gradient
component of 2, against beta <= 0.94. So this is a genuine approximation, not a
small-angle one.
It is still worth testing precisely because the walk was MEASURED insensitive to a
158x larger Trotter error (results/v4_walk_trotter.log, every variant within
0.0-0.2 sigma). Same kind of error, and the walk did not care.

PART 1 checks the matrix identity and reports how far the merge is from the
original at the angles actually used. PART 2 runs the A/B.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit.library import PauliEvolutionGate, RZGate, RXGate, RYGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Statevector, Operator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

X = np.array([[0, 1], [1, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)

def expm2(Mx):
    w, v = np.linalg.eigh(Mx)
    return v @ np.diag(np.exp(-1j * w / 2.0)) @ v.conj().T

print("=" * 80)
print("PART 1. Is the merge the identity it claims, and how big is the BCH gap?")
print("=" * 80)
print(f"  {'alpha':>8}{'beta':>8}{'|merge-target|':>16}{'|orig-target|':>15}"
      f"{'|merge-orig|':>14}")
print("  " + "-" * 61)
for alpha, beta in ((0.1, 0.1), (0.5, 0.3), (1.5, 0.9), (3.8, 0.94)):
    target = expm2(alpha * Z + beta * X)                 # exp(-i(aZ+bX)/2)
    theta = np.hypot(alpha, beta); phi = np.arctan2(beta, alpha)
    merge = (RYGate(phi).to_matrix() @ RZGate(theta).to_matrix()
             @ RYGate(-phi).to_matrix())
    orig = RXGate(beta).to_matrix() @ RZGate(alpha).to_matrix()
    f = lambda A, Bm: float(np.linalg.norm(A - Bm))
    print(f"  {alpha:>8.2f}{beta:>8.2f}{f(merge,target):>16.2e}"
          f"{f(orig,target):>15.2e}{f(merge,orig):>14.2e}")
print("  col1 ~0 confirms the identity. col2/col3 are the real approximation gap:")
print("  the merge equals exp(-i(aZ+bX)/2) exactly, and THAT differs from the")
print("  original CRZ-then-CRX product by O(alpha*beta).")


def walk(q, center, k_steps, dt, R, act, g, merged):
    n = len(act)
    anc = AncillaRegister(1, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(1, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=dt * np.pi,
                                 synthesis=LieTrotter(reps=1)).control(1),
              [anc[0]] + list(sysr))
    gl = g[act]; dg = 1.0 / np.sqrt(max(R, 1e-9))
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma = s * np.pi * dt
        beta = (1.0 - s) * np.pi * dt
        if merged:
            for i in range(n):
                al = gl[i] * gamma * 0.5 * np.pi * dg
                th = float(np.hypot(al, beta)); ph = float(np.arctan2(beta, al))
                qc.ry(-ph, param[i])
                qc.crz(th, anc[0], param[i])
                qc.ry(ph, param[i])
        else:
            for i in range(n):
                qc.crz(gl[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
            for i in range(n):
                qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return qc, q._decode_walk(q._run(qc), center, act, R)


def run(prob, merged, seed, epochs=20, k=15, shots=8192):
    ansatz, H, _ = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    best, cx, dep = float('inf'), 0, 0
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            gg = q.sense_gradient(p, R, act)
            qc, blk = walk(q, p, k, dt, R, act, gg, merged)
            p[act] = blk
            if ep == 0:
                t = transpile(qc, q.backend, optimization_level=1)
                cx = max(cx, t.count_ops().get('cx', 0)); dep = max(dep, t.depth())
        E = float(np.real(Statevector(ansatz.assign_parameters(p))
                          .expectation_value(H)))
        best = min(best, E)
    return best, E, cx, dep


print()
print("=" * 80)
print("PART 2. A/B: does halving the walk's CX cost anything?")
print("=" * 80)
SEEDS = (42, 43, 44, 45)
for pname, prob in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                    ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))):
    a, h, _ = prob()
    exact = float(np.min(np.linalg.eigvalsh(h.to_matrix())))
    print(f"\n  --- {pname} | exact {exact:.4f} ---")
    print(f"  {'walk':<14}{'E_best':>10}{'E_final':>10}{'std':>8}{'SEM':>8}"
          f"{'walk CX':>9}{'depth':>8}{'time':>7}")
    print("  " + "-" * 66)
    store = {}
    for merged in (False, True):
        t0 = time.time(); bs, fs, cx, dep = [], [], 0, 0
        for s in SEEDS:
            b, f, c, d = run(prob, merged, s)
            bs.append(b); fs.append(f); cx = max(cx, c); dep = max(dep, d)
        sd = float(np.std(fs)); sem = sd / np.sqrt(len(SEEDS))
        tag = 'merged' if merged else 'CRZ+CRX (V3)'
        store[merged] = (float(np.mean(fs)), sem)
        print(f"  {tag:<14}{np.mean(bs):>10.4f}{np.mean(fs):>10.4f}{sd:>8.4f}"
              f"{sem:>8.4f}{cx:>9}{dep:>8}{time.time()-t0:>6.0f}s", flush=True)
    m0, s0 = store[False]; m1, s1 = store[True]
    print(f"  merged vs V3: {m1-m0:+.4f} "
          f"({abs(m1-m0)/max(np.hypot(s0,s1),1e-9):.1f} sigma)")
