"""Is spec(i[rho,H]) always symmetric about zero?

Two hypotheses have now failed. The reachable interval of Theorem 2 stayed
exactly symmetric when the generator's spectrum was made asymmetric (rank-r
projectors, r = 1..15) and when the state was mixed (rank(M11) climbing 2 -> 8
over protocol cycles). Both times the symmetry survived.

That points at a property of M11 = (i/2)[rho, H] itself rather than at anything
downstream. Theorem 2's interval is

    [ sum_k lam_dn(M11) lam_up(Y) ,  sum_k lam_dn(M11) lam_dn(Y) ]

and if spec(M11) is symmetric about zero then reversing the pairing negates the
sum for EVERY Y - the interval is symmetric no matter what generator is used and
no matter how mixed the state is.

A PARTIAL ARGUMENT. In the eigenbasis of rho, with rho = diag(D),

    M_jk = (i/2)(D_j - D_k) H_jk ,     M_jj = 0 .

If H is REAL SYMMETRIC in that basis, then M^T = -M, so complex conjugation is an
antiunitary carrying M to -M and the spectrum must be symmetric. That covers the
simulations here, where H is built from real Pauli strings - but rho evolves under
e^{-iH tau} and need not stay real in any fixed basis, so it does not obviously
cover the general case.

THIS FILE SETTLES IT NUMERICALLY, which is the honest thing to do before either
claiming a no-go or abandoning one:

  (a) H real symmetric, rho arbitrary Hermitian  - the argument predicts symmetric
  (b) H complex Hermitian, rho complex Hermitian - unconstrained by the argument
  (c) the protocol's own operators, as a control

If (b) comes back symmetric too, the manuscript's symmetric-interval defect is not
a property of the unfiltered kick at all: it is a property of every protocol whose
first-order response is a commutator of the state with the Hamiltonian, and the
no-go is much wider than the paper currently claims.
"""
import numpy as np

TRIALS = 200
DIMS = (4, 8, 16)


def herm(n, rng, real=False):
    A = rng.normal(size=(n, n))
    if not real:
        A = A + 1j * rng.normal(size=(n, n))
    return (A + A.conj().T) / 2


def sym_defect(M):
    """max_k |lam_k + lam_{n-1-k}| - zero iff the spectrum is +/- paired."""
    ev = np.sort(np.linalg.eigvalsh(M))
    return float(np.max(np.abs(ev + ev[::-1])))


def scaled_defect(M):
    ev = np.linalg.eigvalsh(M)
    s = np.max(np.abs(ev))
    return sym_defect(M) / s if s > 1e-14 else 0.0


print("=" * 84)
print("IS spec(i[rho,H]) SYMMETRIC ABOUT ZERO?")
print("=" * 84)
print(f"  {TRIALS} random pairs per row. 'defect' is max_k |lam_k + lam_(n-1-k)|,")
print("  normalised by the largest |eigenvalue|. Zero means +/- paired spectrum.")
print()
print(f"  {'case':<40}{'dim':>6}{'max defect':>14}{'symmetric?':>13}")
print("  " + "-" * 73)

rng = np.random.RandomState(0)
for tag, h_real, r_real in (("(a) H real symmetric, rho complex", True, False),
                            ("(a') H real, rho real", True, True),
                            ("(b) H complex, rho complex", False, False),
                            ("(b') H complex, rho real", False, True)):
    for n in DIMS:
        worst = 0.0
        for _ in range(TRIALS):
            H = herm(n, rng, real=h_real)
            R = herm(n, rng, real=r_real)
            M = 1j * (R @ H - H @ R)
            worst = max(worst, scaled_defect(M))
        verdict = "YES" if worst < 1e-9 else "NO"
        print(f"  {tag if n == DIMS[0] else '':<40}{n:>6}{worst:>14.2e}{verdict:>13}")

print()
print("  (c) CONTROL — a rank-2 case, where symmetry is guaranteed by Corollary 3")
rng = np.random.RandomState(7)
worst = 0.0
for _ in range(TRIALS):
    n = 8
    H = herm(n, rng, real=True)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    v /= np.linalg.norm(v)
    M = 1j * (np.outer(v, v.conj()) @ H - H @ np.outer(v, v.conj()))
    worst = max(worst, scaled_defect(M))
print(f"  {'pure rho, H real symmetric':<40}{8:>6}{worst:>14.2e}"
      f"{('YES' if worst < 1e-9 else 'NO'):>13}")

print()
print("  If row (b) is NO, the symmetry is a consequence of H being real in the")
print("  simulations - an artefact of the Hamiltonian family, not a general law,")
print("  and an asymmetric interval is reachable with a complex Hamiltonian.")
print("  If row (b) is YES, then no protocol whose first-order response is")
print("  i[rho,H] can prefer cooling, for any state and any generator - which is a")
print("  far stronger statement than the manuscript currently makes.")
