"""Does more ansatz depth help or hurt? The two effects have never been measured together.

RESEARCH_NOTES lists this under OPEN as the largest available gain: reps=1 caps at
-6.1231 on Heisenberg N=4 while reps=3 reaches the exact -6.4641, so every
optimiser in the suite is fighting over the last 1-2% beneath a ceiling set by the
ansatz rather than by the optimiser. Raising reps lifts the ceiling, and V3's cost
is flat in M, so it should be free.

T8 says the opposite. The Walsh degree of the energy over the hypercube is bounded
by the LOCALITY OF THE EFFECTIVE OBSERVABLE - H conjugated by everything after the
block - so single-qubit gates preserve the bound and ENTANGLERS BREAK IT. More reps
means more entanglers after each block, pushing weight to degree 3 and above, which
the degree-1 marginal cannot see at all. Measured at reps=1: blocks with no
entangler after them carry degree-3 weight 1e-32 (exact zero), blocks before a CX
carry 1e-3.

So raising reps lifts the reachable minimum while degrading the estimator's
coverage of the landscape, and the net is unknown. This measures it.

Also relevant to the barren-plateau question, which raising reps is the standard way
to trigger: getting ALL M components from one circuit does NOT evade the plateau -
the gradient norm and the noise both scale as sqrt(M), so the ratio is unchanged, and
Arrasmith et al. bound cost-function-difference estimators regardless of how many
coordinates share the shots. What this file measures is whether the effect is visible
at reachable sizes, or only asymptotically.

REPORTED PER REPS
  ceiling      best energy any optimiser could reach (exact diagonalisation of the
               ansatz manifold is infeasible, so use a well-converged classical run)
  E_final      what QLTO actually reaches
  gap          E_final - ceiling, i.e. how much is the OPTIMISER's fault
  ||g||        sensed gradient norm, averaged over epochs - the plateau indicator
  deg3+        Walsh weight above degree 2, which the marginal cannot use
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize as scipy_min
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

EST = StatevectorEstimator()


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def energy(ansatz, H, p):
    return float(EST.run([(ansatz, H, np.asarray([p]))]).result()[0].data.evs.ravel()[0])


def ceiling(ansatz, H, tries=6, seed=0):
    """Best energy reachable on this ansatz manifold, by well-converged classical
    optimisation from several starts. An upper bound on what any optimiser can do."""
    best = float('inf')
    for t in range(tries):
        x0 = np.random.RandomState(seed + t).uniform(-np.pi, np.pi,
                                                     ansatz.num_parameters)
        r = scipy_min(lambda x: energy(ansatz, H, x), x0, method='BFGS',
                      options={'maxiter': 600})
        best = min(best, float(r.fun))
    return best


def walsh_deg3plus(q, center, R, act, samples=400, seed=3):
    """Fraction of landscape variance above degree 2 - invisible to the marginal."""
    n = len(act)
    rng = np.random.RandomState(seed)
    sig = rng.choice([-1.0, 1.0], size=(samples, n))
    vals = []
    for s in sig:
        p = np.asarray(center, dtype=float).copy()
        p[act] = p[act] + R * s
        vals.append(energy(q.ansatz, q.hamiltonian, p))
    vals = np.array(vals)
    tot = float(np.var(vals))
    if tot < 1e-18:
        return 0.0
    # project onto degree <=2 Walsh basis
    cols = [np.ones(samples)]
    for i in range(n):
        cols.append(sig[:, i])
    for i in range(n):
        for j in range(i + 1, n):
            cols.append(sig[:, i] * sig[:, j])
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    resid = vals - A @ coef
    return float(np.var(resid) / tot)


N, EPOCHS, SHOTS = 4, 20, 8192
H = heis(N)
exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))

print("=" * 94)
print(f"ANSATZ DEPTH — does raising reps help QLTO? Heisenberg N={N}, exact {exact:.4f}")
print("=" * 94)
print("  ceiling = best reachable on the manifold (6 BFGS restarts).")
print("  gap = E_final - ceiling is the OPTIMISER's shortfall; ceiling - exact is the ANSATZ's.")
print()
print(f"  {'reps':>5}{'M':>5}{'ceiling':>10}{'anz gap':>9}{'E_final':>10}"
      f"{'opt gap':>9}{'|g| mean':>10}{'deg3+':>8}{'circuits':>10}")
print("  " + "-" * 76)

for reps in (1, 2, 3):
    ansatz = efficient_su2(N, reps=reps)
    M = ansatz.num_parameters
    ceil = ceiling(ansatz, H)
    q = Q(ansatz, H, shot_budget=SHOTS, sim_seed=5)
    q.reset_shot_stream()
    BLK = [b['params'] for b in q.layers if b['params']]
    p = np.random.RandomState(42).uniform(-np.pi, np.pi, M)

    gnorms = []
    for ep in range(EPOCHS):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            g = q.sense_gradient(p, R, act)
            gnorms.append(float(np.linalg.norm(g[act])))
            p = q._execute_walk(p, 15, dt, R, act, g)
    ef = energy(ansatz, H, p)
    d3 = walsh_deg3plus(q, p, 0.6, BLK[0])
    print(f"  {reps:>5}{M:>5}{ceil:>10.4f}{ceil - exact:>9.4f}{ef:>10.4f}"
          f"{ef - ceil:>9.4f}{np.mean(gnorms):>10.4f}{d3:>8.3f}"
          f"{q.nefv:>10}", flush=True)

print()
print("  If 'anz gap' shrinks with reps while 'opt gap' stays flat, deeper ansaetze")
print("  are free and the notes' OPEN item is settled in favour of raising reps.")
print("  If 'opt gap' grows or |g| collapses, the estimator is losing the landscape -")
print("  either to barren plateaus or to the degree-3+ weight T8 predicts entanglers")
print("  create, and the deg3+ column separates those two explanations.")
