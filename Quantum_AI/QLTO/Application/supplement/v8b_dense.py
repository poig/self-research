"""Does T10's growing advantage survive DENSE coupling?

T10 found cR^2 ~ 1/N on Heisenberg, so n* ~ 0.65M, circuits per gradient stay at
~1.5, and the advantage over parameter-shift GROWS linearly in M. The mechanism is
locality: coordinate i's gradient is attenuated only by coordinates sharing a
Hamiltonian term with it, and for a 1D chain that neighbourhood is FIXED, so it
becomes a shrinking fraction of the system as N grows.

That mechanism makes a falsifiable prediction. For an ALL-TO-ALL Hamiltonian the
neighbourhood IS the whole system, so cR^2 should stay roughly CONSTANT in N
instead of falling as 1/N. Then n* is constant, circuits per gradient grow like M,
and the advantage collapses to a fixed factor.

The suite already contains the right test case: generate_frustrated_hamiltonian is
a transverse-field SK spin glass, sum_{i<j} J_ij Z_i Z_j + sum_i h_i X_i over EVERY
pair - fully connected, and the same efficient_su2 ansatz as Heisenberg, so the
comparison is apples to apples.

    Heisenberg      degree 2      cR^2 * N = 0.198, 0.185, 0.191  (constant)
    Frustrated      degree N-1    cR^2 itself should be constant instead

If the frustrated rows show cR^2 flat and cR^2*N rising, T10 is confirmed as a
LOCALITY DIVIDEND: local physics problems get a growing advantage, dense
combinatorial ones get a fixed factor. That is a sharper and more useful claim
than either extreme.
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
    return float(np.linalg.norm(g))


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


R, N_SAMP = 0.6, 400
print("=" * 86)
print("Does the growing advantage survive DENSE coupling? (T10 stress test)")
print("=" * 86)

summary = []
for fam, fn in (("Heisenberg  (1D, degree 2)", B.get_heisenberg_problem),
                ("Frustrated  (all-to-all)", lambda n: B.generate_frustrated_hamiltonian(n))):
    print(f"\n  ===== {fam} =====")
    print(f"  {'N':>3}{'M':>5}{'terms':>7}{'cR^2':>9}{'cR^2*N':>9}"
          f"{'n*':>8}{'n*/M':>7}{'M/n*':>7}")
    print("  " + "-" * 55)
    for N in (4, 6, 8):
        ansatz, H, _ = fn(N)
        M = ansatz.num_parameters
        widths = [w for w in (1, 2, 4, 8, 12, 16, 24, 32) if w <= M]
        ns, ratios = [], []
        for n in widths:
            acc = []
            for seed in (3, 11, 17):
                rng = np.random.RandomState(seed)
                c = rng.uniform(-np.pi, np.pi, M)
                act = list(range(n))
                sm = smeared_norm(ansatz, H, c, R, act, N_SAMP, rng)
                ex = exact_norm(ansatz, H, c, act)
                acc.append(sm / max(ex, 1e-12))
            ns.append(n); ratios.append(float(np.mean(acc)))
        ns_a = np.array(ns, float); ra = np.array(ratios)
        A = np.vstack([np.ones(len(ns_a)), ns_a]).T
        ce, *_ = np.linalg.lstsq(A, np.log(ra), rcond=None)
        cR2 = -ce[1]
        nstar = 1.0 / (2.0 * cR2) if cR2 > 0 else float('inf')
        summary.append((fam, N, M, cR2, nstar))
        print(f"  {N:>3}{M:>5}{len(H.paulis):>7}{cR2:>9.4f}{cR2*N:>9.4f}"
              f"{nstar:>8.1f}{nstar/M:>7.2f}{M/max(nstar,1e-9):>7.2f}", flush=True)

print()
print("=" * 86)
print("VERDICT")
print("=" * 86)
for fam in dict.fromkeys(f for f, *_ in summary):
    rows = [r for r in summary if r[0] == fam]
    cr = np.array([r[3] for r in rows]); Ns = np.array([r[1] for r in rows], float)
    flat_cr = cr.std() / cr.mean()
    flat_crN = (cr * Ns).std() / (cr * Ns).mean()
    print(f"  {fam}")
    print(f"    spread of cR^2   across N: {flat_cr:6.1%}")
    print(f"    spread of cR^2*N across N: {flat_crN:6.1%}")
    if flat_crN < flat_cr:
        print("    -> cR^2 ~ 1/N. n* scales with M, circuits/gradient constant,")
        print("       ADVANTAGE GROWS WITH M.")
    else:
        print("    -> cR^2 roughly constant. n* saturates, circuits/gradient ~ M,")
        print("       ADVANTAGE IS A FIXED FACTOR.")
