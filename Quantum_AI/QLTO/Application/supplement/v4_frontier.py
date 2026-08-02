"""Round 2: find the efficient frontier of the QPE sensing evolution.

Round 1 said Suzuki-2 at reps=2^a reaches bias 0.056 at depth 991, beating x4
(0.061 at 1969) and x2 (0.116 at 979) outright, and beating Richardson at equal
bias for half the noise and half the circuits.

But Suzuki-2 costs ~2x the gates of Lie-Trotter at equal reps, so reps=2^a under
Suzuki is spending 2x depth on the reps schedule alone. Since its error is
O(t^3/r^2) against Lie's O(t^2/r), HALF the reps under Suzuki-2 should cost the
SAME depth as shipping while still cancelling a higher order. If that lands near
bias 0.06 at depth ~500, it strictly dominates the shipping configuration - same
depth, 3x less bias - and is a free fix rather than a trade.

Testing the reps schedule as a divisor of the QPE-mandated 2^a.
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
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
def obs_mean(ansatz, O, P):
    r = EST.run([(ansatz, O, np.asarray(P))]).result()[0]
    return np.asarray(r.data.evs, dtype=float).ravel()

def smeared_grad(ansatz, O, c, R, act):
    n = len(act)
    signs = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(n)]
                      for v in range(2 ** n)])
    P = []
    for s in signs:
        b = c.copy(); b[act] = c[act] + R * s; P.append(b)
    E = obs_mean(ansatz, O, P)
    g = np.zeros(ansatz.num_parameters)
    for i in range(n):
        hi = signs[:, i] > 0
        g[act[i]] = (E[hi].mean() - E[~hi].mean()) / (2.0 * R)
    return g

def qpe_circ(q, center, R, act, mk):
    n, k = len(act), q.num_ancillas
    anc = AncillaRegister(k, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(k, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    for a in range(k):
        qc.append(PauliEvolutionGate(q.H_sense, time=(2 ** a) * q.tau0,
                                     synthesis=mk(a)).control(1),
                  [anc[a]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return qc

N, R, REP, SHOTS = 4, 0.6, 4, 65536
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
probe = Q(ansatz, H, shot_budget=8192)
BLK = [b['params'] for b in probe.layers]
GSM = np.zeros(ansatz.num_parameters)
for a in BLK:
    GSM += smeared_grad(ansatz, probe.H_sense, c, R, a)
NSM = np.linalg.norm(GSM)
NB = [np.linalg.norm(GSM[a]) for a in BLK]

def mk_lie(div):
    return lambda a: LieTrotter(reps=max(1, (2 ** a) // div))
def mk_suz(order, div):
    return lambda a: SuzukiTrotter(order=order, reps=max(1, (2 ** a) // div))

CAND = [
    ("lie   /1  SHIP", mk_lie(1)),
    ("lie   /2      ", mk_lie(2)),
    ("suz2  /8      ", mk_suz(2, 8)),
    ("suz2  /4      ", mk_suz(2, 4)),
    ("suz2  /2      ", mk_suz(2, 2)),
    ("suz2  /1      ", mk_suz(2, 1)),
    ("suz4  /8      ", mk_suz(4, 8)),
    ("suz4  /4      ", mk_suz(4, 4)),
]

q = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=4)
print("=" * 88)
print("QPE sensing frontier: reps schedule = 2^a / div, k=4")
print("=" * 88)
print(f"  ||g_sm||={NSM:.4f}  blocks {[f'{x:.2f}' for x in NB]}  "
      f"{SHOTS} shots x {REP}")
print()
print(f"  {'candidate':<16}{'bias':>9}{'noise':>9}{'depth':>8}"
      f"{'blk0':>8}{'blk1':>8}{'blk2':>8}{'blk3':>8}")
print("  " + "-" * 74)
rows = []
for label, mk in CAND:
    runs = []
    for _ in range(REP):
        g = np.zeros(ansatz.num_parameters)
        for a in BLK:
            g += q._decode_gradient_qpe(q._run(qpe_circ(q, c, R, a, mk)), c, a, R)
        runs.append(g)
    runs = np.array(runs); mean = runs.mean(axis=0)
    bias = np.linalg.norm(mean - GSM) / NSM
    noise = np.mean([np.linalg.norm(r - mean) for r in runs]) / NSM
    d = max(transpile(qpe_circ(q, c, R, a, mk), q.backend,
                      optimization_level=1).depth() for a in BLK)
    ratios = [np.linalg.norm(mean[a]) / NB[i] for i, a in enumerate(BLK)]
    rows.append((label, bias, noise, d))
    print(f"  {label:<16}{bias:>9.4f}{noise:>9.4f}{d:>8}"
          + "".join(f"{r:>8.3f}" for r in ratios), flush=True)

print()
base = rows[0]
print(f"  shipping: bias {base[1]:.4f} at depth {base[3]}")
print("  strictly better than shipping (lower bias AND depth <= shipping):")
win = [r for r in rows if r[1] < base[1] and r[3] <= base[3]]
for r in win:
    print(f"    {r[0]}  bias {r[1]:.4f} ({base[1]/r[1]:.1f}x better), depth {r[3]}")
if not win:
    print("    none - every bias reduction costs depth, so it is a trade")
