"""Where does the Boltzmann decode break while the walk holds? The scaling claim.

A first version of this file asked "is the walk necessary?" by racing it against
the Boltzmann decode at N=4, n=4. That is the wrong question and boltzmann_step's
own docstring predicts the failure:

    "It works at n<=6 with 8192 shots and degrades toward n ~ log2(shots) ~ 13,
     which is exactly the wide-block regime where T10's cost advantage lives.
     Shipping it as a default would look free at benchmark sizes and break where
     it matters."

The two decoders compute the SAME functional form - which is why they tie, and
this session explains why: the walk's phase is proportional to energy, so
P ~ sin^2(phi/2) IS a Boltzmann reweighting. But they reach it through estimators
of different complexity class:

    Boltzmann decode   NONLINEAR. Must resolve each vertex's energy before
                       weighting it, so it needs shots >~ 2^n.
    the walk           LINEAR. Degree-1 marginals, unbiased at ANY
                       shots-per-vertex by T2.

So the walk computes a NONLINEAR FUNCTIONAL OF THE LANDSCAPE USING ONLY LINEAR
MEASUREMENTS, and that - not amplification, not Grover - is where the quantum
work is. It is a shot-complexity claim about the decode, invisible at every
benchmark size because every benchmark is n <= 6.

MEASURED HERE, at FIXED shots, as the block widens:

    walk_cos     cos(walk step, best available step) - the walk's step quality
    boltz_cos    the same for the Boltzmann decode
    boltz_ok     whether the guard even permits it at this width
    spv          shots per vertex, 2^n against the budget

The best available step is taken as the direction to the true hypercube argmin,
computed by enumeration, so both decoders are scored against the same target.
If boltz_cos collapses as spv falls below ~8 while walk_cos holds, the scaling
claim is demonstrated rather than asserted, and the walk is necessary for a
reason no N=4 benchmark can show.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def E(a, H, p):
    return float(np.real(Statevector(a.assign_parameters(p)).expectation_value(H)))


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 1e-12 and nv > 1e-12 else 0.0


R, DT, KS, SHOTS, REPS = 0.6, 0.5, 15, 8192, 3

print("=" * 92)
print("DECODE SCALING — the walk is linear, the Boltzmann decode is not")
print("=" * 92)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots FIXED, {REPS} centres per row.")
print(f"  Both decoders scored by cos(step, direction to the true hypercube")
print(f"  argmin), enumerated exactly. spv = shots per vertex = {SHOTS}/2^n.")
print()
print(f"  {'N':>4}{'n':>4}{'2^n':>8}{'spv':>9}{'walk_cos':>10}{'boltz_cos':>11}"
      f"{'boltz_ok':>10}")
print("  " + "-" * 56)

for N in (4, 6, 8, 10, 12):
    H = heis(N)
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            q = nisq_v3.QLTOv3(ansatz, H, shot_budget=SHOTS, sim_seed=17)
    except Exception as e:
        print(f"  {N:>4}   construction failed: {e}")
        continue
    BLK = [b['params'] for b in q.layers if b['params']]
    act = BLK[0]
    n = len(act)
    if n > 14:
        print(f"  {N:>4}{n:>4}   skipped: enumeration too wide")
        continue

    wc, bc = [], []
    ok = True
    for t in range(REPS):
        centre = np.random.RandomState(11 + t).uniform(-np.pi, np.pi, M)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for kk, sv in enumerate(sig):
            p = centre.copy()
            p[act] = p[act] + R * sv
            vals[kk] = E(ansatz, H, p)
        target = R * sig[int(np.argmin(vals))]        # direction to the argmin

        q.reset_shot_stream()
        g = q.sense_gradient(centre, R, act)
        pw = q._execute_walk(centre, KS, DT, R, act, g)
        wc.append(cos(pw[act] - centre[act], target))

        try:
            q.reset_shot_stream()
            pb = q.boltzmann_step(centre, R, act)
            bc.append(cos(pb[act] - centre[act], target))
        except ValueError:
            ok = False
            break

    spv = SHOTS / (2 ** n)
    print(f"  {N:>4}{n:>4}{2 ** n:>8}{spv:>9.2f}{np.mean(wc):>10.4f}"
          f"{(np.mean(bc) if ok and bc else float('nan')):>11.4f}"
          f"{('yes' if ok else 'GUARD'):>10}", flush=True)

print()
print("  walk_cos holding while boltz_cos falls - or the guard fires at all - is")
print("  the scaling claim demonstrated. The walk's marginal is linear and")
print("  unbiased at any shots-per-vertex (T2); the Boltzmann decode must resolve")
print("  each vertex's energy and cannot be. T10 puts the cost-optimal block at")
print("  n* ~ 0.65 M, so the regime where QLTO is CHEAPEST is exactly the regime")
print("  where the classical decode is unavailable - which is why a tie at N=4")
print("  says nothing about whether the walk is needed.")
