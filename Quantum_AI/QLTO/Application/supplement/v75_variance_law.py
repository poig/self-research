"""Where does the 5.66x come from? Derive the two variances and check them.

v72 measured QLTO ahead of SPSA by 5.66x (N=4) and 5.21x (N=6) in 1-cos at
matched total shots. That is a benchmark number. This asks whether it follows from
a closed form, because a mechanism that can be written down predicts where it
holds and where it fails, and a benchmark number does not.

THE ESTIMATORS, stated exactly as implemented rather than as usually sketched.

V5 forms per-parameter marginals from ONE shot record, dividing by the EMPIRICAL
counts:

    g_i = [ mean(E | sigma_i=+1) - mean(E | sigma_i=-1) ] / (2R)

The empirical normalisation matters and is often dropped in sketches. Writing
Z_i = sigma_i E / R instead, and averaging, leaves a term E_0^2/R^2 in the
variance. The self-normalised form cancels E_0 EXACTLY on every shot record, so
that term is absent. It would dominate at small R, so the difference is not
cosmetic.

Under E(sigma) = E_0 + R sum_j g_j sigma_j with per-shot readout noise of
variance v:

    g_i^hat = g_i + (1/2) sum_{j!=i} g_j (sbar_j^+ - sbar_j^-) + (etabar_+ - etabar_-)/(2R)

    Var(g_i^V5)   ~ (1/T) [ C_i + v_q/R^2 ] ,        C_i = sum_{j!=i} g_j^2
    Var(g_i^SPSA) ~ (2 S_p/T) C_i + v_c/(T R^2)

THE MECHANISM, if the algebra is right: V5 draws a FRESH sigma on every shot,
while SPSA holds one sigma fixed for S_p shots. Its effective number of
perturbation samples is T/(2 S_p) against V5's T, so its cross-direction term
carries a factor 2 S_p.

WHAT THAT PREDICTS, AND WHY IT IS A PROBLEM. At S_p = 1 the factor is 2, so
sigma-resampling alone caps the ratio near 2 and CANNOT explain 5.66x. Either the
algebra is wrong or a second source exists. The candidate is that v is not shared:
V5's shot returns a bounded +-1 ancilla bit, variance at most 1 however much E
varies, whereas SPSA's shot returns an energy estimate with variance Var(H_g),
which is not bounded by 1 and grows with the Hamiltonian.

WHAT IS MEASURED. A synthetic landscape with g KNOWN EXACTLY, so estimator error
is separable from every other effect. Three things:
  1. the two closed forms against Monte Carlo, to check the algebra
  2. the ratio with v_q = v_c, isolating sigma-resampling alone
  3. the ratio with v_c set to a realistic Var(H_g) > 1, isolating the readout

WHAT WOULD FALSIFY THE DECOMPOSITION. If (2) already reaches ~5x, the readout
story is wrong and resampling is the whole mechanism. If (3) cannot reach 5x for
any plausible Var(H_g), then neither source explains v72 and the benchmark is
measuring something not in this model, which would be the most useful outcome of
the three.

This is a statement about ESTIMATOR VARIANCE on a linear landscape. It says
nothing about finite-R bias, which is the separate cR^2 term, nor about optimiser
progress.
"""
import numpy as np

rng = np.random.default_rng(17)


def cross(g, i):
    return float(np.sum(g ** 2) - g[i] ** 2)


def v5_run(g, E0, R, T, vq):
    """One V5 shot record: fresh sigma per shot, bounded readout noise vq,
    self-normalised marginals."""
    M = len(g)
    sig = rng.choice([-1.0, 1.0], size=(T, M))
    E = E0 + R * (sig @ g) + rng.normal(0.0, np.sqrt(vq), size=T)
    out = np.empty(M)
    for i in range(M):
        p = sig[:, i] > 0
        if p.all() or (~p).all():
            out[i] = 0.0
        else:
            out[i] = (E[p].mean() - E[~p].mean()) / (2 * R)
    return out


def spsa_run(g, E0, R, T, vc, Sp):
    """SPSA at matched TOTAL shots: K = T/(2 Sp) perturbations, Sp shots on each
    of the two energies, so each sigma is reused Sp times."""
    M = len(g)
    K = max(1, int(T // (2 * Sp)))
    acc = np.zeros(M)
    for _ in range(K):
        s = rng.choice([-1.0, 1.0], size=M)
        base = R * float(s @ g)
        ep = E0 + base + rng.normal(0.0, np.sqrt(vc / Sp))
        em = E0 - base + rng.normal(0.0, np.sqrt(vc / Sp))
        acc += ((ep - em) / (2 * R)) * s
    return acc / K


def empirical_var(fn, reps, g, *a):
    est = np.array([fn(g, *a) for _ in range(reps)])
    return est.var(axis=0, ddof=1), est.mean(axis=0)


M, R, T, REPS = 8, 0.45, 4096, 400
E0 = 1.7
g = rng.normal(0.0, 1.0, M)

print("=" * 96)
print("(1)  DO THE CLOSED FORMS MATCH MONTE CARLO?")
print("=" * 96)
print(f"  M={M}  R={R}  T={T}  E0={E0}  reps={REPS}. g known exactly.")
print("  Both arms given the SAME readout variance here, so only sigma-resampling")
print("  differs. Sp=1 is SPSA's best case, its largest number of perturbations.")
print()
v = 1.0
Sp = 1
var_q, mean_q = empirical_var(v5_run, REPS, g, E0, R, T, v)
var_s, mean_s = empirical_var(spsa_run, REPS, g, E0, R, T, v, Sp)
print(f"  {'i':>3}{'g_i':>9}{'V5 emp':>11}{'V5 pred':>11}"
      f"{'SPSA emp':>11}{'SPSA pred':>11}{'bias V5':>10}")
print("  " + "-" * 66)
for i in range(M):
    pq = (cross(g, i) + v / R ** 2) / T
    ps = (2 * Sp * cross(g, i)) / T + v / (T * R ** 2)
    print(f"  {i:>3}{g[i]:>9.4f}{var_q[i]:>11.5f}{pq:>11.5f}"
          f"{var_s[i]:>11.5f}{ps:>11.5f}{mean_q[i] - g[i]:>10.5f}")
print()
print("  If the two 'pred' columns track the two 'emp' columns, the algebra holds")
print("  and E_0 has cancelled in V5 despite E0 = %.1f being large." % E0)

print()
print("=" * 96)
print("(2)  RESAMPLING ALONE:  ratio with the readout variance SHARED")
print("=" * 96)
print(f"  {'Sp':>4}{'sum Var V5':>14}{'sum Var SPSA':>15}{'ratio':>9}")
print("  " + "-" * 42)
for Sp in (1, 2, 4, 8):
    vq2, _ = empirical_var(v5_run, REPS, g, E0, R, T, 1.0)
    vs2, _ = empirical_var(spsa_run, REPS, g, E0, R, T, 1.0, Sp)
    print(f"  {Sp:>4}{vq2.sum():>14.5f}{vs2.sum():>15.5f}"
          f"{vs2.sum() / vq2.sum():>9.2f}")
print()
print("  SPSA would choose the Sp that minimises its own variance, so the honest")
print("  comparison is against the best row. If that best row sits near 2, then")
print("  resampling alone does NOT explain 5.66x.")

print()
print("=" * 96)
print("(3)  UNSHARED READOUT:  V5 bounded at v<=1, SPSA paying Var(H_g)")
print("=" * 96)
print("  V5's shot is a bounded +-1 ancilla bit. SPSA's shot is an energy estimate")
print("  whose variance is Var(H_g) and is not bounded by 1.")
print()
print(f"  {'Var(H_g)':>10}{'Sp*':>6}{'sum Var V5':>14}{'sum Var SPSA':>15}{'ratio':>9}")
print("  " + "-" * 54)
vq_bounded = 1.0
for vc in (1.0, 4.0, 16.0, 64.0):
    vq3, _ = empirical_var(v5_run, REPS, g, E0, R, T, vq_bounded)
    best, best_sp = None, None
    for Sp in (1, 2, 4, 8, 16):
        vs3, _ = empirical_var(spsa_run, REPS, g, E0, R, T, vc, Sp)
        if best is None or vs3.sum() < best:
            best, best_sp = vs3.sum(), Sp
    print(f"  {vc:>10.1f}{best_sp:>6}{vq3.sum():>14.5f}{best:>15.5f}"
          f"{best / vq3.sum():>9.2f}")
print()
print("  Reaching ~5x here, at a Var(H_g) the tested Hamiltonians actually have,")
print("  would locate v72's margin in the readout rather than in the superposition")
print("  and would agree with v67, which found the superposition alone buys nothing.")
print("  Failing to reach it for any plausible Var(H_g) would mean this model does")
print("  not contain the effect v72 measured, which is worth more than either.")
