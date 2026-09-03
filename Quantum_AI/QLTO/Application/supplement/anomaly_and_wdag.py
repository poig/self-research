"""Two open v3 items.

A. The ||g|| anomaly: direction cosine 0.999 but norm ratio 0.55 and 2.08 across
   blocks. HYPOTHESIS: the estimator targets the R-SMEARED gradient, not dE, so
   comparing its norm against the exact gradient is the wrong reference. If
   ||g_sensed|| / ||g_smeared|| ~ 1 while ||g_sensed|| / ||g_exact|| scatters,
   there is no anomaly - only a mislabelled baseline.

B. W-dagger removal: W is block-diagonal in the param computational basis, so it
   cannot change the measured (param, anc) distribution. Predicted: identical
   count distributions up to shot noise, lower depth.
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
import nisq_v3


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
    """<H> for a stack of parameter vectors, one broadcast pub."""
    r = EST.run([(ansatz, H, np.asarray(P))]).result()[0]
    return np.asarray(r.data.evs, dtype=float).ravel()


def exact_grad(ansatz, H, c):
    M = ansatz.num_parameters
    P = []
    for i in range(M):
        pp = c.copy(); pp[i] += np.pi / 2; P.append(pp)
        pm = c.copy(); pm[i] -= np.pi / 2; P.append(pm)
    e = energies(ansatz, H, P)
    return 0.5 * (e[0::2] - e[1::2])


def smeared_grad(ansatz, H, c, R, act, n_samp=2000, seed=0):
    """What the marginal estimator actually targets: central difference in
    coordinate i, averaged over uniform +-R perturbations of every OTHER active
    coordinate. This is grad of the R-box-smoothed energy, not grad E."""
    rng = np.random.RandomState(seed)
    g = np.zeros(ansatz.num_parameters)
    S = rng.choice([-1.0, 1.0], size=(n_samp, len(act)))
    for j, i in enumerate(act):
        Pp, Pm = [], []
        for s in S:
            b = c.copy(); b[act] = c[act] + R * s
            bp = b.copy(); bp[i] = c[i] + R; Pp.append(bp)
            bm = b.copy(); bm[i] = c[i] - R; Pm.append(bm)
        g[i] = (energies(ansatz, H, Pp).mean()
                - energies(ansatz, H, Pm).mean()) / (2.0 * R)
    return g


print("=" * 79)
print("A. ||g|| anomaly - is the R-smeared gradient the right reference?")
print("=" * 79)
N = 4
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
gx = exact_grad(ansatz, H, c)
R = 0.6

for label, k in (("Hadamard", 1), ("QPE k=4", 4)):
    q = nisq_v3.QLTOv3(ansatz, H, shot_budget=131072, num_ancillas=k)
    print(f"\n  --- {label} sensing, R={R} ---")
    print(f"  {'blk':<5}{'axis':<6}{'|g_ex|':>9}{'|g_sm|':>9}{'|g_sens|':>10}"
          f"{'sens/ex':>9}{'sens/sm':>9}{'cos_ex':>8}{'cos_sm':>8}")
    print("  " + "-" * 73)
    re, rs = [], []
    for bi, blk in enumerate(q.layers):
        act = blk['params']
        gs = q.sense_gradient(c, R, act)
        gm = smeared_grad(ansatz, H, c, R, act)
        a, b, m = gx[act], gs[act], gm[act]
        na, nb, nm = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(m)
        re.append(nb / (na + 1e-18)); rs.append(nb / (nm + 1e-18))
        print(f"  {bi:<5}{blk['axis']:<6}{na:>9.4f}{nm:>9.4f}{nb:>10.4f}"
              f"{re[-1]:>9.3f}{rs[-1]:>9.3f}"
              f"{float(a@b/(na*nb+1e-18)):>8.3f}"
              f"{float(m@b/(nm*nb+1e-18)):>8.3f}")
    print(f"  spread of sens/ex : {min(re):.3f} - {max(re):.3f}   "
          f"(ratio {max(re)/max(min(re),1e-9):.2f}x)")
    print(f"  spread of sens/sm : {min(rs):.3f} - {max(rs):.3f}   "
          f"(ratio {max(rs)/max(min(rs),1e-9):.2f}x)")

print()
print("=" * 79)
print("B. W-dagger removal - does it change the measured distribution?")
print("=" * 79)


def walk_counts(q, center, k_steps, delta_t, radius, act, grad, with_wdag, seed):
    n = len(act)
    anc = AncillaRegister(1, 'anc')
    param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c_param'), ClassicalRegister(1, 'c_anc'))
    qc.h(anc); qc.h(param)
    w = q.build_w_gate(param, sysr, center, radius, act)
    qc.append(w, list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=delta_t * np.pi,
                                 synthesis=LieTrotter(reps=1)).control(1),
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
    if with_wdag:
        qc.append(w.inverse(), list(param) + list(sysr))
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    tq = transpile(qc, q.backend, optimization_level=1)
    counts = q.backend.run(tq, shots=q.shot_budget,
                           seed_simulator=seed).result().get_counts()
    return counts, tq.depth(), tq.size()


def tvd(c1, c2):
    t1, t2 = sum(c1.values()), sum(c2.values())
    return 0.5 * sum(abs(c1.get(k, 0) / t1 - c2.get(k, 0) / t2)
                     for k in set(c1) | set(c2))


q2 = nisq_v3.QLTOv3(ansatz, H, shot_budget=65536, num_ancillas=1)
print(f"\n  {'blk':<5}{'depth W+Wd':>12}{'depth noWd':>12}{'-%':>7}"
      f"{'gates W+Wd':>12}{'gates noWd':>12}{'TVD':>9}{'|dparam|':>10}")
print("  " + "-" * 77)
for bi, blk in enumerate(q2.layers):
    act = blk['params']
    g = q2.sense_gradient(c, R, act)
    cw, dw, sw = walk_counts(q2, c, 15, 0.3, R, act, g, True, 1234)
    co, do, so = walk_counts(q2, c, 15, 0.3, R, act, g, False, 1234)
    pw = q2._decode_walk(cw, c, act, R)
    po = q2._decode_walk(co, c, act, R)
    print(f"  {bi:<5}{dw:>12}{do:>12}{100*(dw-do)/dw:>6.0f}%{sw:>12}{so:>12}"
          f"{tvd(cw, co):>9.4f}{np.linalg.norm(pw - po):>10.5f}")

# shot-noise floor: same circuit, two different seeds
cA, _, _ = walk_counts(q2, c, 15, 0.3, R, q2.layers[0]['params'],
                       q2.sense_gradient(c, R, q2.layers[0]['params']), True, 7)
cB, _, _ = walk_counts(q2, c, 15, 0.3, R, q2.layers[0]['params'],
                       q2.sense_gradient(c, R, q2.layers[0]['params']), True, 99)
print(f"\n  shot-noise floor (same circuit, seeds 7 vs 99): TVD = {tvd(cA, cB):.4f}")
print("  => a TVD at or below this floor means W-dagger changes nothing.")
