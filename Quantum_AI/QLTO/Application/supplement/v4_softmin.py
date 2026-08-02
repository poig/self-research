"""V4: if top-m beats the walk, what is the RIGHT decode - and is it real?

v4_argmin found top4 matching or beating the marginal+walk path at half the
circuits and 5.6x lower variance, while pure argmin lost at both sizes. That
brackets the answer: hard argmin discards too much, the uniform marginal keeps
too much, and something in between wins. Two loose ends decide whether this is a
real V4 change or an artefact.

1. CONFOUND: is top-m winning because of the decode, or because of shots?
   marginal spends 2 circuits per block-epoch (sense + walk) = 16384 shots.
   top-m spends 1 circuit = 8192. So top-m already wins at HALF the shots, but
   'top4 @16k' below gives it the same TOTAL shots on one circuit to separate
   "fewer circuits" from "fewer shots".

2. WHAT IS THE PRINCIPLED DECODE? top-m with fixed m is arbitrary - m=4 is 25%
   of the 16 vertices at N=4 but 6% of the 64 at N=6, and it beat the marginal at
   both, so the right rule is unclear. The natural family is a Boltzmann average
   over ALL vertices,

       w_x = exp(-(E_x - E_min)/T),   theta = sum_x w_x theta_x / sum_x w_x

   which is argmin as T->0 and the plain hypercube centre as T->inf. The marginal
   gradient is essentially its first-order term. T is set relative to the spread
   of vertex energies so it is scale-free across problems.
   'frac25' is the other natural rule: keep the best quartile whatever n is.

Every non-marginal decoder costs 1 circuit per block-epoch; marginal costs 2.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister)
from qiskit.circuit.library import efficient_su2, QFT, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
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

def h2():
    return efficient_su2(2, reps=1), SparsePauliOp.from_list([
        ("II", -1.052373245772859), ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045), ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156)])

EST = StatevectorEstimator()
def energy_at(ansatz, H, p):
    return float(EST.run([(ansatz, H, np.asarray([p]))]).result()[0].data.evs.ravel()[0])


def sense_table(q, center, R, act):
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
            synthesis=SuzukiTrotter(order=2, reps=max(1, (2 ** a) // 2))
        ).control(1), [anc[a]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    counts = q._run(qc)

    num = np.zeros((2, n)); den = np.zeros((2, n))
    vnum, vden = {}, {}
    for bitstr, cnt in counts.items():
        parts = bitstr.split()
        if len(parts) != 2:
            continue
        m = int(parts[0], 2); phi = m / (2 ** k)
        if phi >= 0.5:
            phi -= 1.0
        e = -2.0 * np.pi * phi / (q.tau0 + 1e-12)
        xb = parts[1][::-1]
        key = tuple(1 if (i < len(xb) and xb[i] == '1') else 0 for i in range(n))
        vnum[key] = vnum.get(key, 0.0) + e * cnt
        vden[key] = vden.get(key, 0) + cnt
        for i in range(n):
            num[key[i], i] += e * cnt; den[key[i], i] += cnt
    m1 = np.divide(num[1], den[1], out=np.zeros(n), where=den[1] > 0)
    m0 = np.divide(num[0], den[0], out=np.zeros(n), where=den[0] > 0)
    grad = np.zeros(len(center)); grad[act] = (m1 - m0) / (2.0 * R + 1e-12)
    return grad, {v: (vnum[v] / vden[v], vden[v]) for v in vnum}


def weighted_vertices(center, act, R, verts, w):
    p = np.asarray(center, dtype=float).copy()
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return p
    for i, idx in enumerate(act):
        vals = np.array([center[idx] + (R if v[i] else -R) for v in verts])
        p[idx] = float(np.average(vals, weights=w))
    return p


def update(q, center, R, act, mode, k_steps, dt, min_cnt=4):
    grad, table = sense_table(q, center, R, act)
    if mode == 'marginal':
        return q._execute_walk(center, k_steps, dt, R, act, grad)
    good = {v: e for v, (e, cn) in table.items() if cn >= min_cnt}
    if not good:
        return np.asarray(center, dtype=float).copy()
    order = sorted(good, key=good.get)
    if mode.startswith('top'):
        m = max(1, int(mode[3:]))
        sel = order[:m]
        return weighted_vertices(center, act, R, sel, [1.0] * len(sel))
    if mode == 'frac25':
        m = max(1, int(round(0.25 * len(order))))
        sel = order[:m]
        return weighted_vertices(center, act, R, sel, [1.0] * len(sel))
    if mode.startswith('boltz'):
        frac = float(mode[5:])
        E = np.array([good[v] for v in order])
        spread = float(E.max() - E.min())
        T = max(frac * spread, 1e-9)
        w = np.exp(-(E - E.min()) / T)
        return weighted_vertices(center, act, R, order, w)
    raise ValueError(mode)


def run(prob, mode, seed, epochs=20, k=15, shots=8192):
    ansatz, H = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    best = float('inf')
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for a in BLK:
            p = update(q, p, R, a, mode, k, dt)
        best = min(best, energy_at(ansatz, H, p))
    return best, energy_at(ansatz, H, p)


SEEDS = (42, 43, 44, 45, 46, 47)
MODES = [('marginal', 8192, 2), ('top4', 8192, 1), ('top4', 16384, 1),
         ('top8', 8192, 1), ('frac25', 8192, 1),
         ('boltz0.1', 8192, 1), ('boltz0.3', 8192, 1), ('boltz1.0', 8192, 1)]

for name, prob in (("H2", h2), ("Heisenberg N=4", lambda: heis(4)),
                   ("Heisenberg N=6", lambda: heis(6))):
    ansatz, H = prob()
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    nb = len(Q(ansatz, H, shot_budget=8192).layers[0]['params'])
    print(f"\n===== {name} | exact {exact:.4f} | {nb} params/block "
          f"= {2**nb} vertices | {len(SEEDS)} seeds =====", flush=True)
    print(f"  {'decoder':<14}{'shots':>7}{'E_best':>10}{'E_final':>10}"
          f"{'std':>8}{'SEM':>8}{'circ/blk':>10}")
    print("  " + "-" * 67)
    store = {}
    for mode, shots, ncirc in MODES:
        bs, fs = [], []
        for s in SEEDS:
            b, f = run(prob, mode, s, shots=shots)
            bs.append(b); fs.append(f)
        sd = float(np.std(fs)); sem = sd / np.sqrt(len(SEEDS))
        tag = f"{mode}@{shots//1024}k"
        store[tag] = (float(np.mean(fs)), sem)
        print(f"  {tag:<14}{shots:>7}{np.mean(bs):>10.4f}{np.mean(fs):>10.4f}"
              f"{sd:>8.4f}{sem:>8.4f}{ncirc:>10}", flush=True)
    base, bsem = store['marginal@8k']
    print(f"\n  vs marginal (E_final, negative = decoder better):")
    for tag, (m, sem) in store.items():
        if tag == 'marginal@8k':
            continue
        print(f"    {tag:<14}{m - base:+.4f}  "
              f"({abs(m - base) / max(np.hypot(sem, bsem), 1e-9):.1f} sigma)")
