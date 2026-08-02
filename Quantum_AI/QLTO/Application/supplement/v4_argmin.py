"""V4: is ARGMIN a better update than the MARGINAL gradient - and is Grover
therefore worth building at all?

Grover/Duerr-Hoyer finds the minimum of 2^n values in O(2^(n/2)) oracle calls.
Before building that, two things have to be true:

  (1) the argmin update must actually beat the marginal-gradient update, and
  (2) getting the argmin must be hard.

(2) is FALSE at every size benchmarked. The QPE sensing circuit measures the
param register and the energy register in the SAME shot, so its counts are
already a table of (vertex -> sampled energy). With S shots over 2^n vertices
each vertex is sampled S/2^n times; at n=4 and S=32768 that is ~2000 samples
each. The argmin is free. Grover only earns its depth once 2^n >~ S, i.e.
n >~ log2(S) ~ 15 params per block - beyond N=8, the largest benchmarked.

So the question that decides whether Grover is ever worth building is (1), and
(1) is answerable right now for free. Same circuit, same shots, three decoders:

  marginal   what V3 does: per-coordinate conditional mean -> gradient -> WALK
             circuit -> decoded step. Costs 2 circuits per block per epoch.
  argmin     take the lowest-energy vertex and jump to it. A trust-region step:
             the exact best corner of the +-R box. Needs NO walk circuit at all,
             so it costs 1 circuit per block per epoch - HALF of V3's headline
             cost metric.
  top-m      average the m lowest-energy vertices. Between the two: keeps some
             of the averaging that makes the marginal robust to shot noise
             while still concentrating on good vertices.

If argmin loses, Grover is pointless at any n - it accelerates a quantity not
worth having. If argmin wins, it is both a free 2x cost cut today AND the thing
Grover would scale to large n later.
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

EST = StatevectorEstimator()
def energy_at(ansatz, H, p):
    return float(EST.run([(ansatz, H, np.asarray([p]))]).result()[0].data.evs.ravel()[0])


def sense_table(q, center, R, act):
    """One QPE sensing circuit -> (marginal gradient, per-vertex energy table).

    Both come out of the SAME shots. The gradient is the per-coordinate
    conditional mean; the table is the per-vertex mean. Nothing extra is run.
    """
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
        m = int(parts[0], 2)
        phi = m / (2 ** k)
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
    grad = np.zeros(len(center))
    grad[act] = (m1 - m0) / (2.0 * R + 1e-12)
    table = {v: (vnum[v] / vden[v], vden[v]) for v in vnum}
    return grad, table


def vertex_params(center, act, R, bits):
    p = np.asarray(center, dtype=float).copy()
    for i, idx in enumerate(act):
        p[idx] = center[idx] + (R if bits[i] else -R)
    return p


def update(q, center, R, act, mode, k_steps, dt, min_cnt=8):
    grad, table = sense_table(q, center, R, act)
    if mode == 'marginal':
        return q._execute_walk(center, k_steps, dt, R, act, grad), 2
    good = {v: e for v, (e, c) in table.items() if c >= min_cnt}
    if not good:
        return np.asarray(center, dtype=float).copy(), 1
    if mode == 'argmin':
        best = min(good, key=good.get)
        return vertex_params(center, act, R, best), 1
    if mode.startswith('top'):
        m = int(mode[3:])
        order = sorted(good, key=good.get)[:m]
        p = np.asarray(center, dtype=float).copy()
        w = np.array([1.0] * len(order))
        for i, idx in enumerate(act):
            vals = np.array([center[idx] + (R if v[i] else -R) for v in order])
            p[idx] = float(np.average(vals, weights=w))
        return p, 1
    raise ValueError(mode)


def run(N, mode, seed, epochs=20, k=15, shots=8192):
    ansatz, H = heis(N)
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    best, ncirc = float('inf'), 0
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for a in BLK:
            p, c = update(q, p, R, a, mode, k, dt)
            ncirc += c
        best = min(best, energy_at(ansatz, H, p))
    return best, energy_at(ansatz, H, p), ncirc


SEEDS = (42, 43, 44, 45)
for N in (4, 6):
    ansatz, H = heis(N)
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    nblk = len(Q(ansatz, H, shot_budget=8192).layers[0]['params'])
    print(f"\n===== Heisenberg N={N} | exact {exact:.4f} | "
          f"{nblk} params/block = {2**nblk} vertices | 8192 shots "
          f"= {8192 // 2**nblk}/vertex =====", flush=True)
    print(f"  {'decoder':<12}{'E_best':>10}{'std':>8}{'E_final':>10}"
          f"{'circuits':>10}{'time':>7}")
    print("  " + "-" * 57)
    for mode in ('marginal', 'argmin', 'top2', 'top4'):
        t0 = time.time(); bs, fs, nc = [], [], 0
        for s in SEEDS:
            b, f, nc = run(N, mode, s)
            bs.append(b); fs.append(f)
        print(f"  {mode:<12}{np.mean(bs):>10.4f}{np.std(bs):>8.4f}"
              f"{np.mean(fs):>10.4f}{nc:>10}{time.time()-t0:>6.0f}s", flush=True)
