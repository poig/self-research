"""Does the smeared signal decay POLYNOMIALLY or EXPONENTIALLY in block width?

This is the single quantity that decides whether the batching advantage survives
to large M, and it needs no circuits at all.

T2 says the degree-1 estimator is unbiased at ANY shots-per-vertex, even fewer
than one, so the hypercube can be made arbitrarily large without biasing the
gradient. Nothing else in the algorithm has that property. The only thing that
degrades with block width n is the SIGNAL: smearing over more coordinates flattens
the gradient, measured 2.88 -> 1.61 from n=1 to n=16 in the cost study.

Cost for equal relative precision goes as ceil(M/n) * Var / ||g_sm(n)||^2, and Var
is flat in n (T4, b/a = -0.004). So everything hinges on how ||g_sm(n)|| decays:

    POLYNOMIAL  ||g_sm(n)|| ~ n^-p   -> cost ~ (M/n) * n^2p, still improving in n
                                        for p < 1/2, so batching keeps paying and
                                        large M is viable.
    EXPONENTIAL ||g_sm(n)|| ~ a^n    -> cost blows up, the optimal n saturates at
                                        a constant, and the advantage is a fixed
                                        factor no matter how big M gets.

Measured here by fitting both models to the exact smeared gradient over a range of
n on the largest ansatz that fits comfortably, and comparing residuals.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Statevector
import benchmark as B


def energies(ansatz, H, P):
    return np.array([float(np.real(Statevector(ansatz.assign_parameters(p))
                                   .expectation_value(H))) for p in P])


def smeared_norm(ansatz, H, c, R, act, n_samp, rng):
    """||g_smeared|| over `act`, sampling the +-R signs of the other actives.

    Exact enumeration costs 2^n; sampling is unbiased and is what the circuit
    does anyway.
    """
    n = len(act)
    S = rng.choice([-1.0, 1.0], size=(n_samp, n))
    g = np.zeros(n)
    for j, i in enumerate(act):
        Pp, Pm = [], []
        for s in S:
            b = c.copy(); b[act] = c[act] + R * s
            bp = b.copy(); bp[i] = c[i] + R; Pp.append(bp)
            bm = b.copy(); bm[i] = c[i] - R; Pm.append(bm)
        g[j] = (energies(ansatz, H, Pp).mean()
                - energies(ansatz, H, Pm).mean()) / (2.0 * R)
    return float(np.linalg.norm(g)), g


def exact_norm(ansatz, H, c, act):
    g = np.zeros(len(act))
    for j, i in enumerate(act):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[j] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp))
                                    .expectation_value(H)))
                      - float(np.real(Statevector(ansatz.assign_parameters(pm))
                                      .expectation_value(H))))
    return float(np.linalg.norm(g))


R = 0.6
N_SAMP = 400
print("=" * 84)
print("Signal attenuation vs block width - polynomial or exponential?")
print("=" * 84)
print(f"  R={R}, {N_SAMP} smearing samples, ratio is per-coordinate normalised")
print("  so that growing n does not inflate the norm by dimension alone.")

for pname, fn, widths in (
        ("Heisenberg N=4 (M=16)", lambda: B.get_heisenberg_problem(4),
         (1, 2, 4, 6, 8, 12, 16)),
        ("Heisenberg N=6 (M=24)", lambda: B.get_heisenberg_problem(6),
         (1, 2, 4, 6, 8, 12, 16, 24)),
        ("Heisenberg N=8 (M=32)", lambda: B.get_heisenberg_problem(8),
         (1, 2, 4, 8, 12, 16, 24, 32))):
    ansatz, H, _ = fn()
    M = ansatz.num_parameters
    print(f"\n  --- {pname} ---")
    print(f"  {'n':>4}{'|g_sm|/sqrt(n)':>16}{'|g_ex|/sqrt(n)':>16}{'ratio':>9}")
    print("  " + "-" * 45)
    ns, ratios = [], []
    for n in widths:
        acc_r = []
        for seed in (3, 11, 17):
            rng = np.random.RandomState(seed)
            c = rng.uniform(-np.pi, np.pi, M)
            act = list(range(n))
            sm, _ = smeared_norm(ansatz, H, c, R, act, N_SAMP, rng)
            ex = exact_norm(ansatz, H, c, act)
            acc_r.append(sm / max(ex, 1e-12))
        r = float(np.mean(acc_r))
        ns.append(n); ratios.append(r)
        print(f"  {n:>4}{'':>16}{'':>16}{r:>9.4f}", flush=True)

    ns_a = np.array(ns, float); ra = np.array(ratios)
    ok = ra > 1e-9
    # polynomial: log r = log A - p log n
    Ap = np.vstack([np.ones(ok.sum()), np.log(ns_a[ok])]).T
    cp, *_ = np.linalg.lstsq(Ap, np.log(ra[ok]), rcond=None)
    resid_p = np.linalg.norm(Ap @ cp - np.log(ra[ok]))
    # exponential: log r = log B + n log a
    Ae = np.vstack([np.ones(ok.sum()), ns_a[ok]]).T
    ce, *_ = np.linalg.lstsq(Ae, np.log(ra[ok]), rcond=None)
    resid_e = np.linalg.norm(Ae @ ce - np.log(ra[ok]))
    print(f"  polynomial fit  ratio ~ {np.exp(cp[0]):.3f} * n^({cp[1]:+.3f})"
          f"   residual {resid_p:.4f}")
    print(f"  exponential fit ratio ~ {np.exp(ce[0]):.3f} * "
          f"{np.exp(ce[1]):.4f}^n   residual {resid_e:.4f}")
    better = "POLYNOMIAL" if resid_p < resid_e else "EXPONENTIAL"
    print(f"  -> {better} fits better")
    # Report the conclusion implied by the fit that actually WON. An earlier
    # version printed the polynomial-derived exponent unconditionally, which
    # contradicted its own verdict whenever exponential won - which is always.
    if better == "POLYNOMIAL":
        p = -cp[1]
        print(f"  cost ~ (M/n)*n^(2p), p={p:.3f} -> exponent on n "
              f"{2*p - 1:+.3f}; negative means wider blocks keep paying")
    else:
        cR2 = -ce[1]                      # decay rate per active coordinate
        nstar = 1.0 / (2.0 * cR2) if cR2 > 0 else float('inf')
        print(f"  cost ~ (M/n)*exp(2*cR2*n) with cR2={cR2:.4f}")
        print(f"  -> optimal block width n* = 1/(2*cR2) = {nstar:.1f}"
              f"   (M={M}, so n*/M = {nstar/M:.2f})")
        print(f"  -> circuits per gradient at n*: M/n* = {M/nstar:.2f}")
        print("     n*/M roughly CONSTANT across sizes => circuits/gradient is")
        print("     constant and the advantage GROWS with M.")
        print("     n* roughly constant across sizes => the advantage is a fixed")
        print("     factor. Compare this row against the other problem sizes.")
