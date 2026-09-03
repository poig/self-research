"""E. What CAUSES the per-block scale bias?

Test C proved it is a converged bias, not shot noise, and that it tracks the
block's rotation axis. Two candidate mechanisms, each with a knob:

  QPE quantisation   the readout is a sampled eigenvalue rounded into one of 2^k
                     bins of width 2*margin*||H0||/2^k. At k=4 that is ~2.4
                     energy units against a signal of ~0.25 - a tenth of a bin.
                     Rounding a distribution is not mean-preserving, so a bias is
                     expected. KNOB: raise k. Bias should fall as bins shrink.

  Hadamard sin()     the Y-basis readout is -<sin(H tau)>, NOT -tau<H>. Dividing
                     by tau gives <sin(H tau)>/tau = <H> - tau^2<H^3>/6 + ...
                     The correction depends on the spread of H at that vertex,
                     which differs per block. KNOB: lower tau_scale. Bias should
                     fall as tau^2.

If neither knob moves it, the bias is in the marginal-decode itself and both
mechanisms are wrong.
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
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

N, R, REP, SHOTS = 4, 0.6, 4, 262144
ansatz, H = heis(N)
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
probe = Q(ansatz, H, shot_budget=8192)
BLK = [b['params'] for b in probe.layers]
AX = [b['axis'] for b in probe.layers]
NM = [np.linalg.norm(smeared_grad(ansatz, H, c, R, a)[a]) for a in BLK]

def row(q, tag):
    out = f"  {tag:>12}"
    for bi, act in enumerate(BLK):
        rr = [np.linalg.norm(q.sense_gradient(c, R, act)[act]) / NM[bi]
              for _ in range(REP)]
        out += f"{np.mean(rr):>12.3f}"
    return out

hdr = "  " + f"{'knob':>12}" + "".join(
    f"{'blk%d %s' % (i, AX[i]):>12}" for i in range(len(BLK)))

print("=" * 79)
print("E1. QPE quantisation: does the bias shrink as the bins shrink?")
print("=" * 79)
print(hdr); print("  " + "-" * (12 + 12 * len(BLK)))
for k in (3, 4, 5, 6, 7, 8):
    q = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=k)
    binw = 2 * q.qpe_margin * q.H0_norm / (2 ** k)
    print(row(q, f"k={k}") + f"   bin={binw:.3f}", flush=True)
print(f"   (signal being resolved is ~{min(NM):.2f} energy units)")

print()
print("=" * 79)
print("E2. Hadamard sin(): does the bias shrink as tau^2?")
print("=" * 79)
print(hdr); print("  " + "-" * (12 + 12 * len(BLK)))
for ts in (2.0, 1.0, 0.5, 0.25, 0.125):
    q = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=1, tau_scale=ts)
    print(row(q, f"tau_s={ts}") + f"   tau={q.tau:.4f}", flush=True)
print("   (tau -> 0 removes the sin() curvature but costs 1/tau^2 in variance)")

print()
print("=" * 79)
print("E3. control: does the bias depend on R? (O(R^3) leakage in the decode)")
print("=" * 79)
print(hdr); print("  " + "-" * (12 + 12 * len(BLK)))
for rad in (0.9, 0.6, 0.3, 0.15):
    NMr = [np.linalg.norm(smeared_grad(ansatz, H, c, rad, a)[a]) for a in BLK]
    q = Q(ansatz, H, shot_budget=SHOTS, num_ancillas=1)
    out = f"  {'R=%.2f' % rad:>12}"
    for bi, act in enumerate(BLK):
        rr = [np.linalg.norm(q.sense_gradient(c, rad, act)[act]) / NMr[bi]
              for _ in range(REP)]
        out += f"{np.mean(rr):>12.3f}"
    print(out, flush=True)
print("   (ratio is vs the smeared gradient AT THAT R, so R-dependence here is")
print("    decode error, not the smearing itself)")
