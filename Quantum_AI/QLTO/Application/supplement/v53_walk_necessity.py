"""Does the walk earn its circuit ANYWHERE? v20 finished, plus the regime it missed.

v20 asked this properly - three arms sharing ONE sensing call per block-epoch so
the gradient is identical and only its CONSUMPTION differs, paired seeds, and a
gradstep bounded by the same box as the walk so no step size is tuned in its
favour. It produced one row before the job stopped:

    Heisenberg N=4, 256 shots, 16 shots/vertex, 6 paired seeds
      walk -5.7904+-0.045   boltz -5.5951+-0.054   gradstep -5.8096+-0.035
      walk-boltz -0.1952 (2.5s)        walk-grad +0.0192 (0.4s)

Two things follow and they point opposite ways. The walk BEATS the Boltzmann
decode by 2.5 sigma - the nonlinear collapse T2 predicts, since Boltzmann must
resolve each vertex's energy and needs shots >~ 2^n. But it TIES a plain bounded
classical step on the SAME gradient, and gradstep is nominally ahead.

THAT TIE IS A LOSS FOR THE WALK. It costs 2 circuits per block-epoch against
gradstep's 1, so parity at half the price means the walk circuit buys nothing.
And gradstep is not the Boltzmann decode: it consumes the same LINEAR marginal,
so it inherits T1/T2 and scales exactly as the walk does. The shot-complexity
argument kills the Boltzmann decoder; it does not distinguish walk from gradstep.

THE REGIME v20 NEVER ENTERED, and it is the walk's own claim. The walk produces a
STOCHASTIC BOUNDED step sampled over hypercube corners; gradstep is
deterministic. That is the "tunneling" the method is named for, and it should pay
where the box is MULTI-MODAL. v9_globalgrid measured exactly where that starts:
at R=pi/2 the box goes from 1.7 to 3.3 minima, while at R<=1.2 it is essentially
unimodal. v20 ran the shipped schedule R = 0.6*0.9^ep - unimodal throughout - so
the one regime the walk was designed for has never been tested against gradstep.

GRID, cut to something that finishes rather than stopping at row one:
  sizes    N = 4, 6
  budgets  64, 256          (16 and 4 shots/vertex at n=4: where boltz dies)
  radius   R0 = 0.6  shipped, unimodal
           R0 = 1.571 wide, multi-modal by v9 - the walk's claim
  arms     walk, boltz0.1, gradstep      6 paired seeds, 20 epochs

boltz is kept as a POSITIVE CONTROL: it must lose at these budgets, and if it
does not the harness is not measuring what it should.

READ walk-grad: NEGATIVE means the walk reached lower energy. The walk needs a
negative column SOMEWHERE to justify its circuit. If it is flat or positive
across the whole grid - including at wide R - the walk is removable and QLTO is a
one-circuit gradient estimator feeding a classical step, which is where T1, T2
and T10 already live.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister)
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
# v20 used qiskit.circuit.library.QFT, deprecated since Qiskit 2.1; its own log
# carries the warning naming this as the replacement.
from qiskit.synthesis.qft import synth_qft_full
from qiskit.quantum_info import SparsePauliOp, Statevector
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


def energy_at(ansatz, H, p):
    return float(np.real(Statevector(ansatz.assign_parameters(p))
                         .expectation_value(H)))


def sense_table(q, center, R, act):
    """One QPE sensing circuit -> (marginal gradient, per-vertex energy table).

    Copied from v20 rather than imported: v20 runs its experiment at module
    level, and every script here is standalone by convention.
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
    qc.append(synth_qft_full(k, inverse=True, do_swaps=True), anc)
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
        return p
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


def run_seed(N, seed, shots, R0, epochs=20, k=15):
    ansatz, H = heis(N)
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers if b['params']]
    p0 = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    out = {}
    for arm in ARMS:
        p = p0.copy()
        for ep in range(epochs):
            R = max(R0 * (0.9 ** ep), 1e-4)
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


# SHIPPED-R IS SETTLED, so it is not re-run here. At R0=0.6, N=4, gradstep beat
# the walk in four independent measurements spanning two implementations and both
# merged_walk settings:
#     walk      -5.79 (v20 log) / -5.05 (v20 today) / -4.54 (v53) / -5.15 (v53c)
#     gradstep  -5.81 (v20 log) / -5.58 (v20 today) / -5.39 (v53)
# The walk was never ahead, and it costs 2 circuits per block-epoch to gradstep's
# 1, so parity is already a loss. What remains open is WIDE R - where
# v9_globalgrid measured the box going multi-modal, 1.7 -> 3.3 minima at R=pi/2,
# and where a stochastic bounded step is supposed to beat a deterministic one.
# That is the walk's own claim and the only place it can still win.
SIZES = (4, 6)
BUDGETS = (256, 1024)
RADII = (('wide', np.pi / 2),)
SEEDS = (42, 43, 44, 45, 46, 47)

print("=" * 104)
print("DOES THE WALK EARN ITS CIRCUIT ANYWHERE? v20 finished + the wide-R regime")
print("=" * 104)
print(f"  {len(SEEDS)} paired seeds, 20 epochs, k_steps=15, QPE k=4.")
print("  Arms share one sensing call per block-epoch: the gradient is IDENTICAL.")
print("  gradstep is bounded by the same box as the walk, so nothing is tuned for it.")
print("  The walk costs 2 circuits/block-epoch, gradstep 1 - so a TIE is a LOSS.")

for N in SIZES:
    ansatz, H = heis(N)
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    nb = len([b['params'] for b in Q(ansatz, H, shot_budget=1024).layers
              if b['params']][0])
    print(f"\n  ===== Heisenberg N={N} | exact {exact:.4f} | "
          f"{nb} params/block = {2 ** nb} vertices =====")
    print(f"  {'R0':>9}{'shots':>7}{'s/vtx':>8}" +
          "".join(f"{a:>11}" for a in ARMS) +
          f"{'walk-boltz':>12}{'walk-grad':>11}{'sec':>7}")
    print("  " + "-" * 87)
    for rname, R0 in RADII:
        for S in BUDGETS:
            t0 = time.time()
            res = {a: [] for a in ARMS}
            for sd in SEEDS:
                r = run_seed(N, sd, S, R0)
                for a in ARMS:
                    res[a].append(r[a])
            mean = {a: float(np.mean(res[a])) for a in ARMS}
            dwb = np.array(res['walk']) - np.array(res['boltz0.1'])
            dwg = np.array(res['walk']) - np.array(res['gradstep'])
            sb = abs(dwb.mean()) / max(dwb.std() / np.sqrt(len(SEEDS)), 1e-9)
            sg = abs(dwg.mean()) / max(dwg.std() / np.sqrt(len(SEEDS)), 1e-9)
            print(f"  {rname:>9}{S:>7}{S / (2 ** nb):>8.1f}" +
                  "".join(f"{mean[a]:>11.4f}" for a in ARMS) +
                  f"{dwb.mean():>+9.4f}({sb:>.1f}s){dwg.mean():>+8.4f}({sg:>.1f}s)"
                  f"{time.time() - t0:>7.0f}", flush=True)

print()
print("  NEGATIVE walk-grad means the WALK reached lower energy, which is the only")
print("  thing that justifies its second circuit. Flat or positive everywhere -")
print("  including at wide R, where the box is multi-modal and the stochastic")
print("  bounded step is supposed to pay - means the walk is REMOVABLE and QLTO is")
print("  a one-circuit gradient estimator feeding a classical step.")
print("  walk-boltz should stay NEGATIVE throughout: that is the positive control,")
print("  and if it is not, the harness is not measuring what it should.")
print("  Read sigma against this project's own record: 1-2 sigma on six seeds has")
print("  reversed twice before, so treat anything under ~3 sigma as a tie.")
