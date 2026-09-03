"""V4 candidates, judged on gradient fidelity per unit depth BEFORE any of them
is wired into the optimiser.

The target is the R-SMEARED gradient - the quantity sense_gradient actually
estimates. For each candidate we report

  bias     ||mean(g_sens) - g_sm|| / ||g_sm||   over the FULL parameter vector,
           averaging repeats first so shot noise cancels and what is left is
           systematic. This is the thing to minimise.
  noise    mean over repeats of ||g_sens - mean(g_sens)|| / ||g_sm||
  depth    transpiled depth of the deepest sensing circuit (coherence demand)
  circuits sensing circuits per block per epoch (throughput demand)

Candidates:

  reps           the shipping knob. Kills Trotter error but multiplies depth.
  suzuki2        second-order product formula: error O(t^3/r^2) instead of
                 O(t^2/r), at ~2x the gates of one Lie rep. Better error per
                 unit depth if the constant behaves.
  richardson     Lie-Trotter error is O(t^2/r), so g(r) = g_inf + a/r exactly to
                 leading order. Two runs at r and 2r give

                     g_rich = 2 g(2r) - g(r) = g_inf + O(1/r^2)

                 for the DEPTH of r=2 and the CIRCUIT COUNT of 2. That is the
                 whole point: V3 sells low depth, and reps x8 spends exactly
                 what V3 is trying to save. F1's numbers already hint this works
                 - 2(1.220) - 1.616 = 0.824 against Suzuki-4's 0.803.
                 Extrapolation amplifies variance (2x and -1x on two
                 independent estimates -> sqrt(5) ~ 2.24x the noise), so it is
                 only a win if the bias it removes exceeds that.
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
    """EXACT R-smeared gradient by enumerating the block's hypercube.

    A block has only 2^n vertices (n=4 here, so 16), and the estimator's target
    is the uniform marginal over exactly those. So evaluate <O> once per vertex
    and every coordinate's marginal difference falls out of the same numbers -
    16 evaluations per block instead of 32000, and with no sampling error in the
    reference at all.
    """
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

# ── sensing circuits with the product formula exposed ────────────────────────

def had_circ(q, center, R, act, synth):
    n = len(act)
    anc = AncillaRegister(1, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(1, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=q.tau, synthesis=synth).control(1),
              [anc[0]] + list(sysr))
    qc.sdg(anc); qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return qc

def had_run(q, center, R, act, synth):
    qc = had_circ(q, center, R, act, synth)
    return q._decode_gradient(q._run(qc), center, act, R, q.tau)

def qpe_circ(q, center, R, act, mk_synth):
    n, k = len(act), q.num_ancillas
    anc = AncillaRegister(k, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(k, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    for a in range(k):
        qc.append(PauliEvolutionGate(q.H_sense, time=(2 ** a) * q.tau0,
                                     synthesis=mk_synth(a)).control(1),
                  [anc[a]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return qc

def qpe_run(q, center, R, act, mk_synth):
    qc = qpe_circ(q, center, R, act, mk_synth)
    return q._decode_gradient_qpe(q._run(qc), center, act, R)

def depth_of(q, qc):
    return transpile(qc, q.backend, optimization_level=1).depth()

# ── candidates ───────────────────────────────────────────────────────────────
# each: (label, needs_k, fn(q, act) -> gradient, circuits, depth_probe(q, act))

def make_candidates():
    C = []
    for r in (1, 2, 4, 8):
        C.append((f"had lie r={r}" + ("  SHIP" if r == 2 else ""), 1,
                  lambda q, a, r=r: had_run(q, c, R, a, LieTrotter(reps=r)),
                  1, lambda q, a, r=r: had_circ(q, c, R, a, LieTrotter(reps=r))))
    C.append(("had suzuki2 r=1", 1,
              lambda q, a: had_run(q, c, R, a, SuzukiTrotter(order=2, reps=1)),
              1, lambda q, a: had_circ(q, c, R, a, SuzukiTrotter(order=2, reps=1))))
    C.append(("had richardson 1,2", 1,
              lambda q, a: 2.0 * had_run(q, c, R, a, LieTrotter(reps=2))
                           - had_run(q, c, R, a, LieTrotter(reps=1)),
              2, lambda q, a: had_circ(q, c, R, a, LieTrotter(reps=2))))

    for m in (1, 2, 4, 8):
        C.append((f"qpe x{m}" + ("      SHIP" if m == 1 else ""), 4,
                  lambda q, a, m=m: qpe_run(q, c, R, a,
                      lambda i, m=m: LieTrotter(reps=max(1, m * 2 ** i))),
                  1, lambda q, a, m=m: qpe_circ(q, c, R, a,
                      lambda i, m=m: LieTrotter(reps=max(1, m * 2 ** i)))))
    C.append(("qpe suzuki2 x1", 4,
              lambda q, a: qpe_run(q, c, R, a,
                  lambda i: SuzukiTrotter(order=2, reps=max(1, 2 ** i))),
              1, lambda q, a: qpe_circ(q, c, R, a,
                  lambda i: SuzukiTrotter(order=2, reps=max(1, 2 ** i)))))
    C.append(("qpe richardson 1,2", 4,
              lambda q, a: 2.0 * qpe_run(q, c, R, a,
                                lambda i: LieTrotter(reps=max(1, 2 * 2 ** i)))
                           - qpe_run(q, c, R, a,
                                lambda i: LieTrotter(reps=max(1, 2 ** i))),
              2, lambda q, a: qpe_circ(q, c, R, a,
                  lambda i: LieTrotter(reps=max(1, 2 * 2 ** i)))))
    return C

N, R, REP, SHOTS = 4, 0.6, 3, 65536
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
probe = Q(ansatz, H, shot_budget=8192)
BLK = [b['params'] for b in probe.layers]

GSM = np.zeros(ansatz.num_parameters)
for a in BLK:
    GSM += smeared_grad(ansatz, probe.H_sense, c, R, a)
NSM = np.linalg.norm(GSM)

print("=" * 92)
print("V4 CANDIDATES: gradient fidelity vs cost   (target = R-smeared gradient)")
print("=" * 92)
print(f"  ||g_smeared|| = {NSM:.4f} over {len(BLK)} blocks, "
      f"{SHOTS} shots, {REP} repeats")
print()
print(f"  {'candidate':<22}{'bias':>9}{'noise':>9}{'depth':>8}{'circ':>6}"
      f"{'bias*depth':>12}")
print("  " + "-" * 66)

QH = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=1)
QP = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=4)
rows = []
for label, needs_k, fn, ncirc, probe_fn in make_candidates():
    q = QH if needs_k == 1 else QP
    runs = []
    for _ in range(REP):
        g = np.zeros(ansatz.num_parameters)
        for a in BLK:
            g += fn(q, a)
        runs.append(g)
    runs = np.array(runs)
    mean = runs.mean(axis=0)
    bias = np.linalg.norm(mean - GSM) / NSM
    noise = np.mean([np.linalg.norm(r - mean) for r in runs]) / NSM
    d = max(depth_of(q, probe_fn(q, a)) for a in BLK)
    rows.append((label, bias, noise, d, ncirc))
    print(f"  {label:<22}{bias:>9.4f}{noise:>9.4f}{d:>8}{ncirc:>6}"
          f"{bias * d:>12.1f}", flush=True)

print()
best = min(rows, key=lambda r: r[1])
print(f"  lowest bias      : {best[0]} ({best[1]:.4f})")
cheap = min(rows, key=lambda r: r[1] * r[3])
print(f"  best bias*depth  : {cheap[0]} ({cheap[1] * cheap[3]:.1f})")
print()
print("  Richardson is only worth it if its bias drop beats its ~2.24x noise")
print("  amplification AND it beats simply raising reps at equal depth.")
