"""Does the coherent walk earn its circuit at LOW shots and LARGE N?

This is the corner v4_softmin never entered. That run pinned shots=8192 for every
arm and stopped at N=6, and concluded Boltzmann T=0.1 TIES the walk at half the
circuits - which, if it were the whole story, would make the walk removable and
leave QLTO a pure sensing primitive. But a tie at a generous shot budget is the
one place the walk is NOT expected to show an advantage, so that test could not
have found one.

THE HYPOTHESIS, and it has a mechanism from T2 rather than being a guess.

  Gradient-descent arms consume the gradient MULTIPLICATIVELY: theta <- theta -
  eta*ghat, so an error dg propagates straight into the step as eta*dg. Precision
  in ghat is therefore worth real energy.

  The walk consumes it as a PHASE RATE. grad_local only sets how fast each param
  qubit accumulates phase; the step itself is a measured vertex of the +-R
  hypercube and is BOUNDED BY R however wrong ghat is. Gradient error changes
  which vertex is likely, not how far the optimiser moves.

  And the decoders differ in shot-order: the marginal is LINEAR, so by T2 it is
  unbiased at any shots-per-vertex including fewer than one. Boltzmann is
  NONLINEAR - it must resolve each vertex's energy before weighting it - so it
  needs shots >~ 2^n and must degrade as the budget falls or n grows.

So the prediction is sharp and two-sided:
  * boltz collapses first, and it collapses where S/2^n gets small
  * walk holds up better than a classical step on the SAME gradient
  * the gap widens with N, because Var(H) is extensive so a given gradient
    precision costs more shots at larger N

Three arms sharing one sensing call per block-epoch, so the gradient is IDENTICAL
across arms and only its consumption differs:

  walk       q._execute_walk - the shipping coherent step
  boltz0.1   Boltzmann-weighted vertex average, the decoder that tied at 8192
  gradstep   classical step of matched magnitude: 0.9*R*ghat/max|ghat|, bounded
             by the same box the walk moves in, so no step size is tuned

PAIRED across arms - every arm starts each seed from the SAME initial parameters.
These notes record two sub-2-sigma results that reversed on replication, both
from comparing across runs, so pairing is not optional here.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    """One QPE sensing circuit -> (marginal gradient, per-vertex energy table)."""
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


def boltz_step(center, act, R, table, frac=0.1, min_cnt=4):
    good = {v: e for v, (e, cn) in table.items() if cn >= min_cnt}
    p = np.asarray(center, dtype=float).copy()
    if not good:
        return p                      # nothing resolved: the failure mode itself
    order = sorted(good, key=good.get)
    E = np.array([good[v] for v in order])
    T = max(frac * float(E.max() - E.min()), 1e-9)
    w = np.exp(-(E - E.min()) / T)
    if w.sum() <= 0:
        return p
    for i, idx in enumerate(act):
        vals = np.array([center[idx] + (R if v[i] else -R) for v in order])
        p[idx] = float(np.average(vals, weights=w))
    return p


def grad_step(center, act, R, grad, alpha=0.9):
    """Bounded classical step: same box as the walk, so no step size is tuned."""
    p = np.asarray(center, dtype=float).copy()
    g = grad[act]
    mx = float(np.max(np.abs(g)))
    if mx < 1e-12:
        return p
    p[act] = p[act] - alpha * R * g / mx
    return p


ARMS = ('walk', 'boltz0.1', 'gradstep')


def run_seed(prob, seed, shots, epochs=20, k=15):
    """All arms from the SAME initial parameters; returns {arm: E_final}."""
    ansatz, H = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers if b['params']]
    p0 = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)

    out = {}
    for arm in ARMS:
        p = p0.copy()
        for ep in range(epochs):
            R = max(0.6 * (0.9 ** ep), 1e-4)
            dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
            for act in BLK:
                grad, table = sense_table(q, p, R, act)
                if arm == 'walk':
                    p = q._execute_walk(p, k, dt, R, act, grad)
                elif arm.startswith('boltz'):
                    p = boltz_step(p, act, R, table, frac=float(arm[5:]))
                else:
                    p = grad_step(p, act, R, grad)
        out[arm] = energy_at(ansatz, H, p)
    return out


SIZES = (4, 6, 8)
BUDGETS = (256, 1024, 4096, 16384)
SEEDS = (42, 43, 44, 45, 46, 47)

print("=" * 100)
print("WALK vs CLASSICAL DECODE ACROSS SHOT BUDGET AND SIZE")
print("=" * 100)
print(f"  {len(SEEDS)} paired seeds, 20 epochs, k_steps=15, QPE k=4.")
print("  Arms share one sensing call per block-epoch, so the gradient is identical.")
print("  vertices/block = 2^n; boltz needs shots >~ vertices, walk does not (T2).")

for N in SIZES:
    prob = (lambda n: (lambda: heis(n)))(N)
    ansatz, H = prob()
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    nb = len([b['params'] for b in Q(ansatz, H, shot_budget=1024).layers
              if b['params']][0])
    print(f"\n  ===== Heisenberg N={N} | exact {exact:.4f} | M={ansatz.num_parameters}"
          f" | {nb} params/block = {2**nb} vertices =====")
    print(f"  {'shots':>7}{'shots/vertex':>14}" +
          "".join(f"{a:>12}" for a in ARMS) +
          f"{'walk-boltz':>12}{'walk-grad':>11}")
    print("  " + "-" * 80)

    for S in BUDGETS:
        res = {a: [] for a in ARMS}
        for sd in SEEDS:
            r = run_seed(prob, sd, S)
            for a in ARMS:
                res[a].append(r[a])
        mean = {a: float(np.mean(res[a])) for a in ARMS}
        sem = {a: float(np.std(res[a]) / np.sqrt(len(SEEDS))) for a in ARMS}
        # paired differences - the whole point of running arms on shared seeds
        dwb = np.array(res['walk']) - np.array(res['boltz0.1'])
        dwg = np.array(res['walk']) - np.array(res['gradstep'])
        print(f"  {S:>7}{S / (2 ** nb):>14.1f}" +
              "".join(f"{mean[a]:>12.4f}" for a in ARMS) +
              f"{dwb.mean():>+12.4f}{dwg.mean():>+11.4f}", flush=True)
        print(f"  {'':>21}" + "".join(f"{'+-%.3f' % sem[a]:>12}" for a in ARMS) +
              f"{'(%.1fs)' % (abs(dwb.mean()) / max(dwb.std() / np.sqrt(len(SEEDS)), 1e-9)):>12}"
              f"{'(%.1fs)' % (abs(dwg.mean()) / max(dwg.std() / np.sqrt(len(SEEDS)), 1e-9)):>11}",
              flush=True)

print()
print("  NEGATIVE walk-boltz / walk-grad means the WALK reached lower energy.")
print("  The hypothesis predicts both columns go negative as shots fall and as N")
print("  grows. If they stay flat, the walk is a circuit spent for nothing and V4")
print("  should sense only. Sigma is on the PAIRED difference, so it is the right")
print("  test - but read it against these notes' own record: 1-2 sigma on six")
print("  seeds has reversed twice before.")
