"""Can we even MEASURE the degree-2 coefficients? Estimator against ground truth.

The degree-2 drift has now failed twice for different stated reasons:

  v42   added the pairwise terms at the shipped scale. Oracle 4x worse
        (corr 0.4535 -> 0.1170). Diagnosed as the unbounded phase channel
        wrapping six extra terms.
  v42b  bounded the TOTAL phase so it cannot wrap. Bounding helped degree-1
        (corr 0.4535 -> 0.5352 at PHI = pi) but degree-2 still lost at every
        PHI, topping out at 0.2537. So the channel was NOT the obstruction and
        the v42 diagnosis was wrong.

One candidate remains, and it is measurable rather than arguable: the degree-2
coefficients may simply not be estimable from this shot record. E_hat({i,j}) =
mean(e * x_i * x_j) uses the same shots as E_hat({i}) but the product of two
random signs does not reduce variance, while the underlying signal is typically
smaller. If the estimate is mostly noise, feeding it into the drift adds a random
phase - which is exactly what the oracle degradation looks like.

This compares the SAMPLED coefficients against the EXACT ones from full 2^n
enumeration, at several shot budgets. No walk, no optimiser.

  cos1, cos2   cosine of sampled against exact, degree 1 and degree 2
  snr1, snr2   ||exact|| / ||sampled - exact||
  w1, w2       exact squared Walsh weight at each degree, so a low cos2 on a
               block with negligible true deg2 weight is not evidence of
               anything

If cos2 is near 1 wherever w2 is appreciable, the coefficients are fine and the
degree-2 failure needs another explanation. If cos2 collapses while cos1 stays
high, the answer is variance, the fix is shots or a better estimator, and T7's
result was a statement about the ESTIMATOR rather than about degree-2 drift.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import Deg2Walk, sense_deg12


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 1e-14 and nv > 1e-14 else 0.0


R = 0.6
SHOTS = [8192, 65536, 262144]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 98)
print("DEGREE-2 ESTIMATOR QUALITY — sampled coefficients against exact enumeration")
print("=" * 98)
print(f"  R={R}. Exact coefficients from the full 2^n hypercube. No walk.")
print(f"  w1,w2 are the exact squared Walsh weights: a low cos2 where w2 ~ 0 is")
print(f"  meaningless, so read the two together.")
print()
print(f"  {'problem':>15}{'blk':>4}{'shots':>8}{'w1':>8}{'w2':>8}"
      f"{'cos1':>8}{'cos2':>8}{'snr1':>8}{'snr2':>8}")
print("  " + "-" * 75)

agg = {s: {'c1': [], 'c2': [], 'w2': []} for s in SHOTS}
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
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for kk, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[kk] = E(ansatz, H, p)
        ex1 = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)]) / R
        ex2 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                ex2[i, j] = float(np.mean(vals * sig[:, i] * sig[:, j])) / R
        iu = np.triu_indices(n, 1)
        w1 = float(np.sum((ex1 * R) ** 2))
        w2 = float(np.sum((ex2[iu] * R) ** 2))

        for S in SHOTS:
            with contextlib.redirect_stdout(io.StringIO()):
                q = Deg2Walk(ansatz, H, shot_budget=S, sim_seed=17)
            q.reset_shot_stream()
            g1, g2 = sense_deg12(q, centre, R, act)
            c1, c2 = cos(g1, ex1), cos(g2[iu], ex2[iu])
            e1 = np.linalg.norm(g1 - ex1)
            e2 = np.linalg.norm(g2[iu] - ex2[iu])
            s1 = np.linalg.norm(ex1) / e1 if e1 > 1e-14 else np.inf
            s2 = np.linalg.norm(ex2[iu]) / e2 if e2 > 1e-14 else np.inf
            agg[S]['c1'].append(c1); agg[S]['c2'].append(c2); agg[S]['w2'].append(w2)
            print(f"  {name if S == SHOTS[0] else '':>15}"
                  f"{bi if S == SHOTS[0] else '':>4}{S:>8}"
                  f"{w1:>8.4f}{w2:>8.4f}{c1:>8.4f}{c2:>8.4f}"
                  f"{s1:>8.2f}{s2:>8.2f}", flush=True)
        print("  " + "." * 75)

print(f"\n  {'shots':>9}{'mean cos1':>12}{'mean cos2':>12}"
      f"{'mean cos2 (w2>0.01)':>22}")
print("  " + "-" * 55)
for S in SHOTS:
    c1 = np.array(agg[S]['c1']); c2 = np.array(agg[S]['c2'])
    w2 = np.array(agg[S]['w2'])
    m = w2 > 0.01
    sub = c2[m].mean() if m.any() else float('nan')
    print(f"  {S:>9}{c1.mean():>12.4f}{c2.mean():>12.4f}{sub:>22.4f}")

print()
print("  cos1 high and cos2 low, on blocks where w2 is appreciable, means the")
print("  degree-2 drift was fed noise and T7 measured an ESTIMATOR failure rather")
print("  than a mechanism failure. cos2 rising with shots would then give the")
print("  budget at which the degree-2 oracle becomes usable - and that number is")
print("  the actual deliverable, because it prices the upgrade.")
