"""F. Is the per-block bias TROTTER ERROR in the sensing evolution?

E1 killed QPE quantisation: 16x finer bins, bias unmoved. What survives is that
the sensed observable is not H but the effective Hamiltonian of the TROTTERISED
evolution, whose error is O(t^2 [H_i,H_j]) and therefore STATE-dependent - so it
lands differently on each block. Consistent with everything so far:

  Hadamard  tau=0.106 with reps=2  -> Trotter step 0.053, bias 1.14-1.24 on Z
  QPE       tau0=0.166 with reps=1 -> Trotter step 0.166 (3x), bias 1.75-2.14
  QPE k     reps=2^a tracks t=2^a*tau0, so the step is CONSTANT in k -> flat. ok

tau and the sin() curvature both scale as tau^2, so a tau sweep cannot separate
them. Trotter REPS at FIXED tau can: more reps shrinks the Trotter error and
leaves the sin() curvature untouched.

  bias -> 1 as reps grows   =>  Trotter error. Fix by raising reps (or by using
                                a better product formula), no change to physics.
  bias flat in reps         =>  not Trotter. Then it is the sin() nonlinearity
                                (Hadamard) and something else again for QPE.
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister)
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate, QFT
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
def energies(ansatz, H, P):
    r = EST.run([(ansatz, H, np.asarray(P))]).result()[0]
    return np.asarray(r.data.evs, dtype=float).ravel()

def smeared_grad(ansatz, H, c, R, act, n_samp=4000, seed=0):
    rng = np.random.RandomState(seed)
    g = np.zeros(ansatz.num_parameters)
    S = rng.choice([-1.0, 1.0], size=(n_samp, len(act)))
    for i in act:
        Pp, Pm = [], []
        for s in S:
            b = c.copy(); b[act] = c[act] + R * s
            bp = b.copy(); bp[i] = c[i] + R; Pp.append(bp)
            bm = b.copy(); bm[i] = c[i] - R; Pm.append(bm)
        g[i] = (energies(ansatz, H, Pp).mean()
                - energies(ansatz, H, Pm).mean()) / (2.0 * R)
    return g


def sense_hadamard(q, center, R, act, reps, synth='lie'):
    """q.sense_gradient with the Trotter synthesis exposed."""
    n = len(act)
    anc = AncillaRegister(1, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(1, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    S = SuzukiTrotter(order=4, reps=reps) if synth == 'suzuki' else LieTrotter(reps=reps)
    qc.append(PauliEvolutionGate(q.H_sense, time=q.tau, synthesis=S).control(1),
              [anc[0]] + list(sysr))
    qc.sdg(anc); qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_gradient(q._run(qc), center, act, R, q.tau)


def sense_qpe(q, center, R, act, rep_mult):
    n, k = len(act), q.num_ancillas
    anc = AncillaRegister(k, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(k, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    for a in range(k):
        qc.append(PauliEvolutionGate(
            q.H_sense, time=(2 ** a) * q.tau0,
            synthesis=LieTrotter(reps=max(1, rep_mult * 2 ** a))).control(1),
            [anc[a]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_gradient_qpe(q._run(qc), center, act, R)


N, R, REP, SHOTS = 4, 0.6, 4, 262144
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
probe = Q(ansatz, H, shot_budget=8192)
BLK = [b['params'] for b in probe.layers]
AX = [b['axis'] for b in probe.layers]
NM = [np.linalg.norm(smeared_grad(ansatz, H, c, R, a)[a]) for a in BLK]
hdr = "  " + f"{'knob':>14}" + "".join(
    f"{'blk%d %s' % (i, AX[i]):>12}" for i in range(len(BLK)))

def show(tag, fn):
    out = f"  {tag:>14}"
    for bi, act in enumerate(BLK):
        rr = [np.linalg.norm(fn(act)[act]) / NM[bi] for _ in range(REP)]
        out += f"{np.mean(rr):>12.3f}"
    print(out, flush=True)

print("=" * 79)
print("F1. Hadamard: Trotter reps at FIXED tau (tau=%.4f)" % probe.tau)
print("=" * 79)
print(hdr); print("  " + "-" * (14 + 12 * len(BLK)))
qh = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=1)
for reps in (1, 2, 4, 8, 16):
    show(f"lie reps={reps}", lambda a, r=reps: sense_hadamard(qh, c, R, a, r))
show("suzuki4 r=4", lambda a: sense_hadamard(qh, c, R, a, 4, 'suzuki'))
print("   (-> 1 means Trotter error; flat means the sin() nonlinearity)")

print()
print("=" * 79)
print("F2. QPE k=4: Trotter reps multiplier at FIXED tau0 (%.4f)" % probe.tau0)
print("=" * 79)
print(hdr); print("  " + "-" * (14 + 12 * len(BLK)))
qq = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=4)
for rm in (1, 2, 4, 8):
    show(f"reps x{rm}", lambda a, r=rm: sense_qpe(qq, c, R, a, r))
print("   (QPE decodes energy directly, so NO sin() term - if this converges to")
print("    1 the whole bias was Trotter, and V3's sensing is exact in principle)")
