"""Write the OPTIMAL degree-1 phase, not the energy-proportional one.

v46 measured the walk at 2.035 against a degree-1 ceiling of 5.000 at n=4 - 41%
of what its own phase degree allows. So the nine failed interventions were not
bumping against a bound; there is roughly 2.5x of headroom and none of them found
it.

The reason is a distinction this project has never drawn. The drift writes

    phi(x) = ACC * sum_i g_i x_i,     g_i = E_hat({i}) / R

which is the degree-1 TRUNCATION OF THE ENERGY. But the phase that maximises
concentration is the degree-1 polynomial best approximating the good-set
INDICATOR, and those are different objects. Truncating E is the natural thing to
do and is not the optimal thing to do.

The optimum is computable classically from the same measured coefficients: build
the degree-<=2 model of E from the sensing shots, rank the corners under that
model, then optimise a degree-1 phase to concentrate on the top m. No extra
circuits, and at the sizes QLTO runs the ranking is free - these notes already
record that "the argmin is FREE at every benchmarked size", since S/2^n is 512
samples per vertex at n=4.

ARMS, all with the same sensing, same mixer, same schedule:

    shipped     g_i = E_hat({i})/R              the energy truncation
    opt-exact   g_i = c*_i / ACC, with c* the degree-1 phase optimised against
                the EXACT landscape - establishes whether the ceiling is
                reachable by this circuit at all
    opt-meas    the same, but c* optimised against the degree-<=2 model built
                from the MEASURED coefficients - the deployable version
    opt-weakmix opt-exact with beta scaled down, since the ceiling was derived
                for a PURE phase and the mixer may be what prevents reaching it

If opt-exact reaches the ceiling, the design is validated and opt-meas prices the
deployable version. If opt-exact also falls short, the gap is the mixer or the
accumulated-angle structure rather than the phase, and opt-weakmix separates
those two.
"""
import sys, os, contextlib, io, itertools
import numpy as np
from scipy.optimize import minimize

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12
    from v43_phase_offset import OffsetWalk

R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
ACC = 0.5 * np.pi / np.sqrt(R) * np.pi * DT * KS / 2.0
MS = [1, 2, 4]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]


def sig_and_index(n):
    sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    idx = np.array([int(''.join('1' if s[i] > 0 else '0'
                                for i in range(n))[::-1], 2) for s in sig])
    return sig, idx


def opt_deg1_phase(n, mask, restarts=6, seed=0):
    """Best degree-1 phase (constant + linear) for concentrating on mask."""
    sig, idx = sig_and_index(n)
    A = np.column_stack([np.ones(len(sig))] + [sig[:, i] for i in range(n)])
    Ai = np.empty_like(A)
    Ai[idx] = A
    N, m = 2 ** n, int(mask.sum())
    rng = np.random.RandomState(seed)

    def neg(c):
        p = np.sin((Ai @ c) / 2.0) ** 2
        t = p.sum()
        return -(N / m) * p[mask].sum() / t if t > 1e-12 else 0.0

    best, bc = 0.0, np.zeros(n + 1)
    for r in range(restarts):
        res = minimize(neg, rng.randn(n + 1) * 0.8, method='BFGS',
                       options={'maxiter': 600, 'gtol': 1e-10})
        if -res.fun > best:
            best, bc = -res.fun, res.x
    return best, bc


print("=" * 96)
print("OPTIMAL DEGREE-1 PHASE — the truncation of E is not the optimum")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. All arms share sensing, mixer")
print(f"  and schedule; only the drift coefficients differ. ACC = {ACC:.2f}.")
print()
print(f"  {'problem':>15}{'blk':>4}{'m':>3}{'shipped':>9}{'opt-exact':>11}"
      f"{'opt-meas':>10}{'opt-weak':>10}{'ceiling':>9}")
print("  " + "-" * 71)

agg = {m: {k: [] for k in ('ship', 'oe', 'om', 'ow', 'ceil')} for m in MS}
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=64)
    BLK = [b['params'] for b in probe.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        sig, idx = sig_and_index(n)
        vals = np.empty(len(sig))
        for kk, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[kk] = E(ansatz, H, p)
        e_by = np.empty(2 ** n)
        e_by[idx] = vals
        rank_exact = np.argsort(e_by)

        with contextlib.redirect_stdout(io.StringIO()):
            q = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                           merged_walk=False)
        q.reset_shot_stream()
        g1m, g2m = sense_deg12(q, centre, R, act)

        # degree-<=2 model of E from the MEASURED coefficients
        A1 = np.column_stack([sig[:, i] for i in range(n)])
        mod = A1 @ (g1m * R)
        for i in range(n):
            for j in range(i + 1, n):
                mod = mod + (g2m[i, j] * R) * sig[:, i] * sig[:, j]
        mod_by = np.empty(2 ** n)
        mod_by[idx] = mod
        rank_meas = np.argsort(mod_by)

        def run(gvec, offset=0.0, beta_scale=1.0):
            # The optimiser returns (c_0, c_1..c_n). c_0 is a CONSTANT added to
            # phi(x) and is realised by the ancilla phase gate, so it must be
            # passed through as `offset` - dropping it changes where sin^2 folds.
            # beta_scale scales dt, which scales the drift accumulation ACC too,
            # so g is divided by it to hold the total drift phase fixed and
            # isolate the mixer.
            q.reset_shot_stream()
            dt = DT * beta_scale
            counts = q.walk(centre, KS, dt, R, act,
                            gvec / max(beta_scale, 1e-9),
                            np.zeros((n, n)), False, offset)
            P = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    P[int(parts[1].replace(" ", ""), 2)] += c
            return P / max(P.sum(), 1)

        P_ship = run(g1m)
        for m in MS:
            mask = np.zeros(2 ** n, dtype=bool)
            mask[rank_exact[:m]] = True
            ceil, c_star = opt_deg1_phase(n, mask, seed=bi)

            mask_m = np.zeros(2 ** n, dtype=bool)
            mask_m[rank_meas[:m]] = True
            _, c_meas = opt_deg1_phase(n, mask_m, seed=bi + 7)

            P_oe = run(c_star[1:] / ACC, offset=c_star[0])
            P_om = run(c_meas[1:] / ACC, offset=c_meas[0])
            P_ow = run(c_star[1:] / ACC, offset=c_star[0], beta_scale=0.15)

            f = 2 ** n / m
            row = [f * float(P_ship[mask].sum()), f * float(P_oe[mask].sum()),
                   f * float(P_om[mask].sum()), f * float(P_ow[mask].sum()), ceil]
            for k, v in zip(('ship', 'oe', 'om', 'ow', 'ceil'), row):
                agg[m][k].append(v)
            print(f"  {name if m == MS[0] else '':>15}{bi if m == MS[0] else '':>4}"
                  f"{m:>3}{row[0]:>9.3f}{row[1]:>11.3f}{row[2]:>10.3f}"
                  f"{row[3]:>10.3f}{row[4]:>9.3f}", flush=True)
        print("  " + "." * 71)

print(f"\n  {'m':>4}{'shipped':>10}{'opt-exact':>11}{'opt-meas':>10}"
      f"{'opt-weak':>10}{'ceiling':>9}{'oe/ceil':>9}")
print("  " + "-" * 63)
for m in MS:
    a = {k: np.mean(agg[m][k]) for k in agg[m]}
    print(f"  {m:>4}{a['ship']:>10.3f}{a['oe']:>11.3f}{a['om']:>10.3f}"
          f"{a['ow']:>10.3f}{a['ceil']:>9.3f}"
          f"{a['oe'] / a['ceil'] if a['ceil'] > 1e-9 else 0:>9.2f}")

print()
print("  opt-exact beating shipped is the first derived improvement of the")
print("  session with a bound predicting it in advance. opt-exact reaching the")
print("  ceiling validates the pure-phase model for this circuit; falling short")
print("  means the mixer or the accumulated-angle structure is in the way, and")
print("  opt-weak - the same phase with beta cut to 0.15 - separates them.")
