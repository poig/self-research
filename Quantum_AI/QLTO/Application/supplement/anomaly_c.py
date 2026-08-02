"""C. Is the ||g|| scatter a NOISE FLOOR or a BIAS?

Test A showed sens/smeared ~ 1 for the block with a large gradient and scattered
for blocks with small ones. Two readings:

  noise floor  each coordinate carries a fixed ABSOLUTE error from shot noise.
               Then the ratio's SPREAD shrinks as 1/sqrt(shots) and its MEAN
               converges to 1. Nothing is wrong with the estimator.
  bias         the ratio's mean stays away from 1 no matter the budget. Then the
               estimator really is mis-scaled per block and needs a fix.

Sweep the budget, repeat at each, report mean +- std of ||g_sens||/||g_smeared||.

D. When is dropping W-dagger actually worth it? Measured at k=15 it saved ~0% of
   depth because the walk body owns the critical path. Sweep k_steps.
"""
import sys, os
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
import contextlib, io
import nisq_v3

_R = nisq_v3.QLTOv3
def QUIET(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)
nisq_v3.QLTOv3 = QUIET

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

N, R, REP = 4, 0.6, 6
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)

print("=" * 79)
print("C. noise floor or bias?  ratio = ||g_sens|| / ||g_smeared||")
print("=" * 79)
probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=8192, num_ancillas=1)
BLK = [b['params'] for b in probe.layers]
GM = [smeared_grad(ansatz, H, c, R, a) for a in BLK]
NM = [np.linalg.norm(GM[i][BLK[i]]) for i in range(len(BLK))]

for label, k in (("Hadamard", 1), ("QPE k=4", 4)):
    print(f"\n  --- {label} ---")
    hdr = "".join(f"{'blk %d (|g|=%.2f)' % (i, NM[i]):>22}" for i in range(len(BLK)))
    print(f"  {'shots':>8}{hdr}")
    print("  " + "-" * (8 + 22 * len(BLK)))
    for shots in (4096, 16384, 65536, 262144, 1048576):
        q = nisq_v3.QLTOv3(ansatz, H, shot_budget=shots, num_ancillas=k)
        row = ""
        for bi, act in enumerate(BLK):
            rr = [np.linalg.norm(q.sense_gradient(c, R, act)[act]) / NM[bi]
                  for _ in range(REP)]
            row += f"{np.mean(rr):>15.3f}+-{np.std(rr):<5.3f}"
        print(f"  {shots:>8}{row}", flush=True)
    print("   (mean -> 1.0 and std -> 0 as shots grow  =>  noise floor, not bias)")

print()
print("=" * 79)
print("D. W-dagger depth saving vs walk length")
print("=" * 79)

def build(q, center, k_steps, delta_t, radius, act, grad, with_wdag):
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
    gl, dg = grad[act], 1.0 / np.sqrt(max(radius, 1e-9))
    for step in range(k_steps):
        s = (step + 0.5) / k_steps
        gamma, beta = s * np.pi * delta_t, (1.0 - s) * np.pi * delta_t
        for i in range(n):
            qc.crz(gl[i] * gamma * 0.5 * np.pi * dg, anc[0], param[i])
        for i in range(n):
            qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    if with_wdag:
        qc.append(w.inverse(), list(param) + list(sysr))
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    tq = transpile(qc, q.backend, optimization_level=1)
    return tq.depth(), tq.size()

q = nisq_v3.QLTOv3(ansatz, H, shot_budget=8192, num_ancillas=1)
act = BLK[0]
g = q.sense_gradient(c, R, act)
print(f"\n  {'k_steps':>8}{'depth W+Wd':>12}{'depth noWd':>12}{'depth saved':>13}"
      f"{'gates W+Wd':>12}{'gates noWd':>12}{'gates saved':>13}")
print("  " + "-" * 82)
for k in (1, 2, 4, 8, 15, 30):
    dw, sw = build(q, c, k, 0.3, R, act, g, True)
    do, so = build(q, c, k, 0.3, R, act, g, False)
    print(f"  {k:>8}{dw:>12}{do:>12}{100*(dw-do)/dw:>12.0f}%"
          f"{sw:>12}{so:>12}{100*(sw-so)/sw:>12.0f}%")
