"""G. The residual after Trotter: is it the sin() nonlinearity?

F1 showed the Z-block bias is Trotter (1.616 -> 1.071 as reps 1 -> 4) but the
Y-block deficit is not (0.86-0.95, flat in reps). What remains is that the
Y-basis readout is -<sin(H tau)>, so dividing by tau estimates

    <sin(H tau)>/tau  =  <H>  -  (tau^2/6) <H^3>  +  O(tau^4)

The estimator therefore returns d(smeared <H>) - (tau^2/6) d(smeared <H^3>). The
<H^3> term is a DIFFERENT function of the vertex, so its directional derivative
can carry either sign - which is why some blocks read high and others low. This
is a bias no shot budget removes, and it is invisible unless you compare against
the smeared gradient.

Sweep tau with Trotter held out of the way at reps=4. The bias should fall as
tau^2 and every block should converge to 1.

Also: predicted vs measured. Compute d<H^3> classically and check the correction
    ratio_pred = 1 - (tau^2/6) * d<H^3>_i / d<H>_i
against the measured ratio, per block. Agreeing to first order nails it.
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
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

def smeared_of(ansatz, O, c, R, act, n_samp=4000, seed=0):
    """Smeared gradient of <O> over the +-R hypercube on `act`."""
    rng = np.random.RandomState(seed)
    g = np.zeros(ansatz.num_parameters)
    S = rng.choice([-1.0, 1.0], size=(n_samp, len(act)))
    for i in act:
        Pp, Pm = [], []
        for s in S:
            b = c.copy(); b[act] = c[act] + R * s
            bp = b.copy(); bp[i] = c[i] + R; Pp.append(bp)
            bm = b.copy(); bm[i] = c[i] - R; Pm.append(bm)
        g[i] = (obs_mean(ansatz, O, Pp).mean()
                - obs_mean(ansatz, O, Pm).mean()) / (2.0 * R)
    return g

def sense_had(q, center, R, act, synth):
    n = len(act)
    anc = AncillaRegister(1, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(1, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, center, R, act), list(param) + list(sysr))
    S = (SuzukiTrotter(order=4, reps=4) if synth == 'suzuki'
         else LieTrotter(reps=16))
    qc.append(PauliEvolutionGate(q.H_sense, time=q.tau, synthesis=S).control(1),
              [anc[0]] + list(sysr))
    qc.sdg(anc); qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return q._decode_gradient(q._run(qc), center, act, R, q.tau)

N, R, REP, SHOTS, REPS = 4, 0.6, 4, 262144, 4
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
probe = Q(ansatz, H, shot_budget=8192)
BLK = [b['params'] for b in probe.layers]
AX = [b['axis'] for b in probe.layers]
H0 = probe.H_sense
H3 = (H0 @ H0 @ H0).simplify()

GM = [smeared_of(ansatz, H0, c, R, a) for a in BLK]
G3 = [smeared_of(ansatz, H3, c, R, a) for a in BLK]
NM = [np.linalg.norm(GM[i][BLK[i]]) for i in range(len(BLK))]

print("=" * 79)
print("G1. tau sweep with Trotter suppressed TWO independent ways")
print("=" * 79)
print("  Lie reps=16 landed at 1.009 on blk1 while Suzuki-4 landed at 0.803.")
print("  If residual Trotter (pushes UP) is cancelling sin() (pushes DOWN),")
print("  the two columns disagree at large tau and AGREE as tau -> 0.")
for sy in ('suzuki', 'lie16'):
    print(f"\n  --- {sy} ---")
    print("  " + f"{'tau_scale':>11}{'tau':>8}" +
          "".join(f"{'blk%d %s' % (i, AX[i]):>12}" for i in range(len(BLK))))
    print("  " + "-" * (19 + 12 * len(BLK)))
    for ts in (2.0, 1.0, 0.5, 0.25, 0.125):
        q = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=1, tau_scale=ts)
        out = f"  {ts:>11}{q.tau:>8.4f}"
        for bi, act in enumerate(BLK):
            rr = [np.linalg.norm(sense_had(q, c, R, act, sy)[act]) / NM[bi]
                  for _ in range(REP)]
            out += f"{np.mean(rr):>12.3f}"
        print(out, flush=True)
print("\n   (-> 1 as tau -> 0, falling as tau^2, confirms the sin() term)")

print()
print("=" * 79)
print("G2. predicted vs measured, per COORDINATE, at the shipping tau")
print("=" * 79)
q = Q(ansatz, H, shot_budget=1048576, num_ancillas=1)
t2 = q.tau ** 2 / 6.0
print(f"  tau={q.tau:.4f}   tau^2/6={t2:.6f}   (Suzuki-4, Trotter suppressed)")
print(f"  {'blk':<4}{'ax':<4}{'i':<4}{'d<H>':>10}{'d<H3>':>12}"
      f"{'pred ratio':>12}{'meas ratio':>12}")
print("  " + "-" * 58)
for bi, act in enumerate(BLK):
    meas = np.mean([sense_had(q, c, R, act, 'suzuki') for _ in range(REP)], axis=0)
    for i in act:
        d1, d3 = GM[bi][i], G3[bi][i]
        if abs(d1) < 1e-3:
            continue
        pred = 1.0 - t2 * d3 / d1
        print(f"  {bi:<4}{AX[bi]:<4}{i:<4}{d1:>10.4f}{d3:>12.2f}"
              f"{pred:>12.3f}{meas[i]/d1:>12.3f}")
print("   (pred tracking meas coordinate-by-coordinate confirms the mechanism;")
print("    note the correction can go either way, which is the axis split)")
