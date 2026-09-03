"""Does the degree-1 marginal lose the landscape as depth grows? Measured directly.

v35_reps_scaling.py tried to answer this by comparing QLTO's final energy against a
"ceiling" from BFGS restarts. That comparison is not sound and the number it
produced should not be quoted: the ceiling used EXACT statevector energies with
600 BFGS iterations from 6 starts, while QLTO ran at 8192 shots for 20 epochs. The
resulting "optimiser gap" mixes shot noise, budget, degree-1 blindness, and error
in the ceiling estimate itself - and only the third was the claim. It is the same
defect the benchmark fairness audit already caught once, where baselines received
exact statevector gradients.

The claim under test needs no optimiser. T8 says the Walsh degree of the energy on
the hypercube is bounded by the locality of the EFFECTIVE observable - H conjugated
by everything after the block - so entanglers raise it and a degree-1 estimator
goes blind to the excess. That is a statement about the GRADIENT, so measure the
gradient:

    cos( g_sensed , grad E )      as a function of ansatz depth

with everything else held fixed. No epochs, no schedule, no ceiling, no budget.
Shot noise is separated by also computing the EXACT degree-1 Walsh coefficient,
which is what the estimator targets in the absence of sampling:

    cos_exact   exact deg-1 Walsh coefficient vs exact gradient   -> coverage alone
    cos_sensed  sampled marginal vs exact gradient                -> coverage + shots

If cos_exact degrades with reps while deg3+ grows, coverage is the mechanism and
it is not a sampling artefact. If cos_exact stays flat and only cos_sensed moves,
the effect is shot noise and T8's warning does not bite at these depths.
"""
import sys, os, contextlib, io, itertools
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


def walsh_exact(ansatz, H, c, act, R):
    """Exact degree-1 Walsh coefficients and the degree-3+ residual weight.

    Enumerates the full 2^n hypercube, so this is the estimator's TARGET with no
    sampling whatsoever - the quantity T1 says the marginal converges to.
    """
    n = len(act)
    sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    vals = np.array([E(ansatz, H, _shift(c, act, R, s)) for s in sig])
    deg1 = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)]) / R

    tot = float(np.var(vals))
    if tot < 1e-18:
        return deg1, 0.0
    cols = [np.ones(len(sig))] + [sig[:, i] for i in range(n)]
    cols += [sig[:, i] * sig[:, j] for i in range(n) for j in range(i + 1, n)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return deg1, float(np.var(vals - A @ coef) / tot)


def _shift(c, act, R, s):
    p = np.asarray(c, dtype=float).copy()
    p[act] = p[act] + R * s
    return p


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 1e-14 and nv > 1e-14 else 0.0


N, R, SHOTS, REP = 4, 0.6, 16384, 5
H = heis(N)

print("=" * 94)
print(f"COVERAGE vs DEPTH — gradient quality only, no optimiser. Heisenberg N={N}, R={R}")
print("=" * 94)
print("  cos_exact  : EXACT degree-1 Walsh coefficient vs exact gradient (no shots)")
print("  cos_sensed : sampled marginal vs exact gradient")
print("  deg3+      : landscape variance above degree 2, exact over the full hypercube")
print("  Averaged over 5 random centres; block 0 of each ansatz.")
print()
print(f"  {'reps':>5}{'M':>5}{'n':>4}{'cos_exact':>12}{'cos_sensed':>12}"
      f"{'deg3+':>9}{'|deg1|':>9}")
print("  " + "-" * 56)

for reps in (1, 2, 3, 4):
    ansatz = efficient_su2(N, reps=reps)
    M = ansatz.num_parameters
    q = Q(ansatz, H, shot_budget=SHOTS, sim_seed=5)
    act = [b['params'] for b in q.layers if b['params']][0]

    ce, cs, d3, n1 = [], [], [], []
    for t in range(REP):
        c = np.random.RandomState(100 + t).uniform(-np.pi, np.pi, M)
        gx = exact_grad(ansatz, H, c, act)
        w1, dd = walsh_exact(ansatz, H, c, act, R)
        ce.append(cos(w1, gx)); d3.append(dd); n1.append(float(np.linalg.norm(w1)))
        q.reset_shot_stream()
        gs = q.sense_gradient(c, R, act)[act]
        cs.append(cos(gs, gx))
    print(f"  {reps:>5}{M:>5}{len(act):>4}{np.mean(ce):>12.4f}{np.mean(cs):>12.4f}"
          f"{np.mean(d3):>9.3f}{np.mean(n1):>9.4f}", flush=True)

print()
print("  cos_exact falling with reps => coverage loss, mechanism confirmed, not shots.")
print("  cos_exact flat => T8's warning does not bite at these depths and v35's")
print("  'optimiser gap' was budget and shot noise, not blindness.")
