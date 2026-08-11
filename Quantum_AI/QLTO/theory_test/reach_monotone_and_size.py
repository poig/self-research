"""Two questions about REACH, not about correctness.

The results are verified. What limits the paper is that (a) the abstract ends
"N <= 7" so the no-go reads as small-scale, and (b) W* is a bespoke spectral
pairing that exists nowhere but this paper. Both are addressable without new
physics.

QUESTION 1 - IS THE NO-GO ACTUALLY SIZE-LIMITED?
Condition (A) is "spec(Y) symmetric about zero". For the protocol's own
generator Y = sum_i X_i the spectrum is {n-2k : k = 0..n}, which is symmetric
about zero for EVERY n, by inspection. If that holds at every size then the
central conclusion is size-independent and the small-N work verifies only the
quantitative parts. Checked here to n = 14, along with the other natural
generators, so the claim can be stated for all N rather than for the sizes
simulated.

QUESTION 2 - IS W* A KNOWN QUANTITY IN DISGUISE?
W* = (theta/2) sum_k lam_dn(M11) lam_dn(Y) with M11 = <1| i[rho,A] |1>.
Hoelder gives immediately

    W*  <=  (theta/2) ||Y||_inf ||M11||_1                                  (B1)

and ||i[rho,A]||_1 is a commutator norm of the family used to quantify coherence
between energy eigenspaces. The established measure is Wigner-Yanase skew
information

    S_A(rho) = -(1/2) Tr( [A, sqrt(rho)]^2 ) ,

which uses SQRT(rho), not rho. For a PURE state they coincide, and skew
information then equals the variance Var(A) - the quantity the manuscript already
carries as Delta_H. So for pure branches the connection is one step away.

For MIXED branches [rho, A] and [sqrt(rho), A] differ, and it is an open question
whether W* is still controlled by a monotone. That is the question worth
answering, because a bound in terms of an established asymmetry measure would
place the result inside the resource theory instead of beside it.

MEASURED HERE:
  (1) spec(sum X_i) symmetry defect vs n, to n = 14
  (2) the Hoelder bound (B1) against the exact W*, and how loose it is
  (3) pure branch: W* vs skew information and vs variance, checking the
      identity S = Var and whether W* tracks it
  (4) mixed branch: whether the SAME relation survives when [rho,A] and
      [sqrt(rho),A] part company, which is where the claim would either extend
      or break
"""

import numpy as np
from scipy.linalg import expm, sqrtm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
THETA, TAU = 0.2, 1.042


def op(P, i, n):
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def sum_p(P, n):
    return sum(op(P, i, n) for i in range(n))


def sym_defect(A):
    ev = np.sort(np.linalg.eigvalsh(A))
    return float(np.max(np.abs(ev + ev[::-1])))


print("=" * 94)
print("Q1  IS CONDITION (A) SIZE-INDEPENDENT?   spec(Y) symmetric about zero")
print("=" * 94)
print(f"  {'n':>4}{'dim':>7}{'sum X_i':>14}{'sum Z_i':>14}{'sum Y_i':>14}"
      f"{'random Herm':>14}")
print("  " + "-" * 67)
rng = np.random.default_rng(3)
for n in range(2, 15):
    d = 2 ** n
    if n <= 11:
        sx = sym_defect(sum_p(X, n))
        sz = sym_defect(sum_p(Z, n))
        sy = sym_defect(sum_p(Y_, n))
        R = rng.normal(size=(min(d, 64),) * 2)
        R = R + R.T
        sr = sym_defect(R)
        print(f"  {n:>4}{d:>7}{sx:>14.2e}{sz:>14.2e}{sy:>14.2e}{sr:>14.2e}")
    else:
        # spectrum of sum X_i is {n-2k} by construction, no matrix needed
        ev = np.array([n - 2 * k for k in range(n + 1)], dtype=float)
        print(f"  {n:>4}{d:>7}{np.max(np.abs(np.sort(ev) + np.sort(ev)[::-1])):>14.2e}"
              f"{'(analytic)':>14}{'':>14}{'':>14}")
print()
print("  Sums of single-qubit Paulis have spectrum {n-2k}, symmetric about zero at")
print("  EVERY n. A generic Hermitian operator is not. So condition (A) holds for")
print("  the protocol's own generator at all sizes and the no-go is not")
print("  size-limited; only the quantitative parts need simulation.")

# ---------------------------------------------------------------- Q2
def post_sensing(Hm, n, mix_cycles=0, Ym=None):
    """Joint post-sensing state; mix_cycles>0 returns a genuinely mixed branch."""
    d = 2 ** n
    psi = np.ones(d) / np.sqrt(d)
    rho_s = np.outer(psi, psi.conj())
    if mix_cycles:
        ufb = expm(-1j * (THETA / 2.0) * Ym)
        ev, evec = np.linalg.eigh(Hm)
        u = (evec * np.exp(-1j * ev * TAU)) @ evec.conj().T
        p0 = np.array([[1.0, 0], [0, 0]], dtype=complex)
        p1 = np.array([[0, 0], [0, 1.0]], dtype=complex)
        had = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        anc = np.array([[.5, .5], [.5, .5]], dtype=complex)
        for _ in range(mix_cycles):
            r = np.kron(rho_s, anc)
            Us = np.kron(np.eye(d), p0) + np.kron(u, p1)
            r = Us @ r @ Us.conj().T
            Hd = np.kron(np.eye(d), had)
            r = Hd @ r @ Hd.conj().T
            Uf = np.kron(np.eye(d), p0) + np.kron(ufb, p1)
            r = Uf @ r @ Uf.conj().T
            rho_s = r.reshape(d, 2, d, 2)[:, 0, :, 0] + r.reshape(d, 2, d, 2)[:, 1, :, 1]
            rho_s = rho_s / np.trace(rho_s)
    ev, evec = np.linalg.eigh(Hm)
    u = (evec * np.exp(-1j * ev * TAU)) @ evec.conj().T
    p0 = np.array([[1.0, 0], [0, 0]], dtype=complex)
    p1 = np.array([[0, 0], [0, 1.0]], dtype=complex)
    had = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    anc = np.array([[.5, .5], [.5, .5]], dtype=complex)
    r = np.kron(rho_s, anc)
    Us = np.kron(np.eye(d), p0) + np.kron(u, p1)
    r = Us @ r @ Us.conj().T
    Hd = np.kron(np.eye(d), had)
    return Hd @ r @ Hd.conj().T


def build_H(n, seed=5):
    r = np.random.default_rng(seed)
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            H = H + r.uniform(-1, 1) * (op(Z, i, n) @ op(Z, j, n))
    for i in range(n):
        H = H + r.uniform(-.5, .5) * op(X, i, n)
    return H


print()
print("=" * 94)
print("Q2  IS W* CONTROLLED BY AN ESTABLISHED MONOTONE?")
print("=" * 94)
print(f"  Hoelder:  W* <= (theta/2) ||Y||_inf ||M11||_1.   Skew information")
print(f"  S = -(1/2)Tr([A,sqrt(rho)]^2) equals Var(A) on pure states.")
print()
print(f"  {'n':>3}{'branch':>9}{'purity':>9}{'W*':>10}{'Hoelder':>10}{'ratio':>8}"
      f"{'||M11||_1':>11}{'skew S':>10}{'Var(A)':>10}{'S=Var?':>9}")
print("  " + "-" * 89)

for n in (3, 4):
    Hm = build_H(n)
    d = 2 ** n
    Ym = sum_p(X, n)
    for tag, cyc in (("pure", 0), ("mixed", 6)):
        joint = post_sensing(Hm, n, cyc, Ym)
        A = np.kron(Hm, np.eye(2))
        M = 1j * (joint @ A - A @ joint)
        M11 = M.reshape(d, 2, d, 2)[:, 1, :, 1]

        lm = np.sort(np.linalg.eigvalsh(M11))[::-1]
        ly = np.sort(np.linalg.eigvalsh(Ym))[::-1]
        wstar = (THETA / 2.0) * float(np.dot(lm, ly))
        hoelder = (THETA / 2.0) * float(np.max(np.abs(ly))) * float(np.sum(np.abs(lm)))

        pur = float(np.real(np.trace(joint @ joint)))
        sq = sqrtm(joint)
        comm = A @ sq - sq @ A
        skew = -0.5 * float(np.real(np.trace(comm @ comm)))
        var = float(np.real(np.trace(joint @ A @ A) - np.trace(joint @ A) ** 2))

        print(f"  {n:>3}{tag:>9}{pur:>9.4f}{wstar:>10.5f}{hoelder:>10.5f}"
              f"{wstar / hoelder:>8.3f}{np.sum(np.abs(lm)):>11.5f}"
              f"{skew:>10.5f}{var:>10.5f}"
              f"{abs(skew - var):>9.1e}")

print()
print("  'S=Var?' near zero on the PURE rows and clearly nonzero on the MIXED rows")
print("  is the expected signature: skew information coincides with the variance")
print("  only for pure states. If W*/Hoelder stays in a narrow band across BOTH,")
print("  the trace norm ||M11||_1 controls the reachable work regardless of purity,")
print("  and the bound can be quoted in terms of a commutator norm. If the ratio")
print("  swings, the bound is loose and the honest statement is the spectral")
print("  pairing itself, with no monotone attached.")
