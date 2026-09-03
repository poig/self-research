"""Score the walk against a target the bound says is REACHABLE.

Every measurement this session scored the walk against the argmin: v38's regret,
v39c's mode == x_true, the enhancement P(x_true) 2^n. v45 then showed that target
is unreachable for any low-degree phase on ANY problem, by identity:

    1[x = x*] = prod_i (1 + x*_i x_i)/2 = 2^-n sum_S chi_S(x*) chi_S(x)

so all 2^n Walsh coefficients of a single-point indicator are EQUAL and the
degree profile is C(n,d)/(2^n - 1) regardless of the landscape. Checked exactly
against v45: 4/15 = 0.267, 6/63 = 0.0952, 8/255 = 0.0314.

But marking a SET is problem-dependent, and v45 measured physical landscapes
carrying 4-5x more degree-1 weight than random at m = 4. So the walk has been
graded against an impossible target all session, and may be doing better than it
has been credited for.

TWO NUMBERS PER BLOCK AND SET SIZE:

    achieved     P(G_m) * 2^n / m from the shipped walk, where G_m is the best m
                 corners. 1.0 is uniform.
    ceiling      the same quantity maximised over degree-d phases with
                 P ~ sin^2(phi/2), by direct optimisation - the analogue of
                 phase_degree_bound's MAX ENHANCEMENT = sum_{j<=d} C(n,j), but
                 for a SET rather than a point, so it depends on the landscape.

If achieved tracks ceiling at m > 1, the walk is saturating what its phase degree
allows and the session's low scores were an artefact of the target, not of the
circuit. If achieved sits far below ceiling, there is headroom that none of the
nine interventions found - and it would be the first evidence of any.

This also predicts a result these notes have carried without a cause since the
decoder study: hard argmin decoding LOST (+0.137 at N=4, +0.277 at N=6) while a
Boltzmann decode TIED. A Boltzmann weight is a soft set-marking, so the bound
says it had to win over argmin.
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


def basis(n, deg):
    sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    idx = np.array([int(''.join('1' if s[i] > 0 else '0'
                                for i in range(n))[::-1], 2) for s in sig])
    cols = [np.ones(len(sig))]
    for d in range(1, deg + 1):
        for S in itertools.combinations(range(n), d):
            cols.append(np.prod(sig[:, S], axis=1))
    A = np.stack(cols, axis=1)
    Ai = np.empty_like(A)
    Ai[idx] = A                    # reorder rows into bitmask order
    return Ai


def ceiling_set(n, deg, mask, restarts=5, seed=0):
    """max P(G) * 2^n / |G| over degree-<=deg phases, P ~ sin^2(phi/2)."""
    A = basis(n, deg)
    N, m = 2 ** n, int(mask.sum())
    rng = np.random.RandomState(seed)

    def neg(c):
        p = np.sin((A @ c) / 2.0) ** 2
        t = p.sum()
        return -(N / m) * p[mask].sum() / t if t > 1e-12 else 0.0

    best = 0.0
    for r in range(restarts):
        res = minimize(neg, rng.randn(A.shape[1]) * 0.8, method='BFGS',
                       options={'maxiter': 600, 'gtol': 1e-10})
        best = max(best, -res.fun)
    return best


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
MS = [1, 2, 4, 8]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 96)
print("SET-TARGETED SCORING — the walk against a reachable target")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. G_m is the best m corners.")
print(f"  Both columns are P(G_m) * 2^n / m: 1.0 is uniform, 2^n/m is a perfect")
print(f"  hit. 'ceil d1'/'ceil d2' are the best any degree-1 / degree-2 phase")
print(f"  could do on THIS landscape.")
print()
print(f"  {'problem':>15}{'blk':>4}{'m':>4}{'achieved':>10}{'ceil d1':>9}"
      f"{'ceil d2':>9}{'frac of d1':>12}{'2^n/m':>8}")
print("  " + "-" * 71)

agg = {m: [] for m in MS}
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
        order = np.array([int(''.join('1' if s[i] > 0 else '0'
                                      for i in range(n))[::-1], 2) for s in sig])
        e_by = np.empty(2 ** n)
        e_by[order] = vals

        # the shipped walk, once per block
        with contextlib.redirect_stdout(io.StringIO()):
            q = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                           merged_walk=False)
        q.reset_shot_stream()
        g1, g2 = sense_deg12(q, centre, R, act)
        counts = q.walk(centre, KS, DT, R, act, g1, np.zeros_like(g2), False, 0.0)
        P = np.zeros(2 ** n)
        for bs, c in counts.items():
            parts = bs.split()
            if len(parts) == 2 and parts[0][-1] == '1':
                P[int(parts[1].replace(" ", ""), 2)] += c
        P = P / max(P.sum(), 1)

        rank = np.argsort(e_by)
        for m in MS:
            if m >= 2 ** n:
                continue
            mask = np.zeros(2 ** n, dtype=bool)
            mask[rank[:m]] = True
            ach = (2 ** n / m) * float(P[mask].sum())
            c1 = ceiling_set(n, 1, mask, seed=bi)
            c2 = ceiling_set(n, 2, mask, seed=bi + 1)
            agg[m].append((ach, c1, c2))
            print(f"  {name if m == MS[0] else '':>15}{bi if m == MS[0] else '':>4}"
                  f"{m:>4}{ach:>10.3f}{c1:>9.3f}{c2:>9.3f}"
                  f"{ach / c1 if c1 > 1e-9 else 0:>12.2f}"
                  f"{2 ** n / m:>8.1f}", flush=True)
        print("  " + "." * 71)

print(f"\n  {'m':>5}{'mean achieved':>16}{'mean ceil d1':>15}{'mean ceil d2':>15}"
      f"{'achieved/d1':>13}")
print("  " + "-" * 64)
for m in MS:
    if not agg[m]:
        continue
    a = np.array([r[0] for r in agg[m]])
    c1 = np.array([r[1] for r in agg[m]])
    c2 = np.array([r[2] for r in agg[m]])
    print(f"  {m:>5}{a.mean():>16.3f}{c1.mean():>15.3f}{c2.mean():>15.3f}"
          f"{a.mean() / c1.mean():>13.2f}")

print()
print("  achieved/d1 near 1 means the walk is saturating its degree-1 ceiling and")
print("  the session's low scores were the TARGET's fault, not the circuit's.")
print("  Well below 1 means real headroom exists at degree 1 - which none of the")
print("  nine interventions found, and which would be worth chasing.")
print("  ceil d2 well above ceil d1 prices the pairwise upgrade for SET marking,")
print("  a different question from the point marking that v42/v42b/v43 tested.")
