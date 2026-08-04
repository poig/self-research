"""Measure what the WALK delivers with depth, not what the gradient looks like.

Two previous attempts asked the wrong question.

v35 compared QLTO's final energy against a BFGS "ceiling" computed with exact
statevector energies and 600 iterations from 6 restarts, while QLTO ran at 8192
shots for 20 epochs. That gap mixes shot noise, budget, blindness and error in the
ceiling itself; its "optimiser gap doubles" number is withdrawn.

v35b fixed the confound but kept the wrong figure of merit: cos(g_sensed, grad E)
against the EXACT gradient. The walk does not consume the exact gradient. It uses
grad_local[i] as a PHASE RATE - al = g_i * gamma * 0.5 pi * drift_gain - and the
step it takes is a weighted mean of +-R corners, bounded by R whatever g is. The
overall scale is absorbed by dt and k_steps, which are set by the schedule. Three
measurements in these notes say the same thing from different directions: raising
the walk's Trotter error 158x changes nothing; zeroing the gradient costs 4.32
Hartree while RANDOM drift is worse than none; and the per-block scale error of up
to 2x never stopped it converging. What the walk needs is the SIGN and RELATIVE
structure, not the magnitude.

So measure the output:

    dE          energy change produced by one walk step (negative = descent)
    cos_step    cos(delta_theta, -grad E) - did the step move downhill
    sign_acc    fraction of coordinates whose sensed sign matches the exact one

against depth, with everything else fixed. sign_acc is the quantity the drift
actually needs; cos_step is what the optimiser experiences; dE is the bottom line.

A flat sign_acc with depth means the estimator still tells the walk what it needs,
whatever happens to the gradient's magnitude or its cosine against an exact vector
the walk never sees.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
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
    return SparsePauliOp.from_list(ops)


def E(ansatz, H, p):
    return float(np.real(Statevector(ansatz.assign_parameters(p)).expectation_value(H)))


def exact_grad(ansatz, H, c, act):
    g = np.zeros(len(act))
    for j, i in enumerate(act):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[j] = 0.5 * (E(ansatz, H, pp) - E(ansatz, H, pm))
    return g


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 1e-14 and nv > 1e-14 else 0.0


N, R, DT, KS, SHOTS, REP = 4, 0.6, 0.5, 15, 16384, 8
H = heis(N)

print("=" * 96)
print(f"WALK OUTPUT vs DEPTH — the figure of merit the walk actually has. N={N}, R={R}")
print("=" * 96)
print("  One walk step from a random centre, averaged over 8 centres per depth.")
print("  sign_acc is what the drift consumes; dE is the bottom line.")
print("  'random drift' repeats the step with a sign-randomised gradient, as a control:")
print("  if sign_acc were irrelevant the two dE columns would agree.")
print()
print(f"  {'reps':>5}{'M':>5}{'sign_acc':>11}{'cos_step':>11}{'dE':>11}"
      f"{'dE random':>12}{'|g_sensed|':>12}")
print("  " + "-" * 67)

for reps in (1, 2, 3, 4):
    ansatz = efficient_su2(N, reps=reps)
    M = ansatz.num_parameters
    q = Q(ansatz, H, shot_budget=SHOTS, sim_seed=5)
    act = [b['params'] for b in q.layers if b['params']][0]

    sa, cst, des, der, gn = [], [], [], [], []
    for t in range(REP):
        c = np.random.RandomState(200 + t).uniform(-np.pi, np.pi, M)
        gx = exact_grad(ansatz, H, c, act)
        q.reset_shot_stream()
        g = q.sense_gradient(c, R, act)
        gs = g[act]
        gn.append(float(np.linalg.norm(gs)))
        sa.append(float(np.mean(np.sign(gs) == np.sign(gx))))

        e0 = E(ansatz, H, c)
        p1 = q._execute_walk(c, KS, DT, R, act, g)
        des.append(E(ansatz, H, p1) - e0)
        cst.append(cos(p1[act] - c[act], -gx))

        grand = g.copy()
        rs = np.random.RandomState(900 + t).choice([-1.0, 1.0], size=len(act))
        grand[act] = np.abs(gs) * rs
        p2 = q._execute_walk(c, KS, DT, R, act, grand)
        der.append(E(ansatz, H, p2) - e0)

    print(f"  {reps:>5}{M:>5}{np.mean(sa):>11.4f}{np.mean(cst):>11.4f}"
          f"{np.mean(des):>11.4f}{np.mean(der):>12.4f}{np.mean(gn):>12.4f}",
          flush=True)

print()
print("  If sign_acc and dE stay flat while |g_sensed| falls, then the magnitude")
print("  decline seen in v35b is absorbed by the schedule and is not a failure mode.")
print("  If dE approaches dE-random, the drift has stopped steering and the depth")
print("  effect is real - in which case it is a signal problem, not a coverage one.")
