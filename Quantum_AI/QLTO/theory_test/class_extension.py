"""How wide is the no-go? Everything so far is one ancilla, product coupling,
one cycle.

The manuscript proves its statements for K = P_1 (x) Y: a single ancilla, a
coupling that factorises, and the first cycle. A no-go is worth as much as the
class it covers, and none of the three restrictions is obviously load-bearing.
Proposition (general response) already dropped the requirement that A be
I_A (x) H; this asks whether the rest of the structure can go too.

THREE EXTENSIONS, each testing whether the interval stays symmetric:

  (1) k ANCILLAS.  Replace the single control qubit by k, with feedback
      conditioned on the ancilla register. If the reachable interval is still
      symmetric the no-go is not a one-ancilla artefact, and multi-ancilla
      sensing - which the optimiser side uses - inherits it.

  (2) NON-PRODUCT COUPLING.  K = P_1 (x) Y factorises. Replace it with a general
      Hermitian K on the joint space that does NOT factorise, so the ancilla and
      system parts are entangled in the generator itself. The relevant question
      becomes whether spec(K) rather than spec(Y) is what has to be asymmetric.

  (3) MULTI-CYCLE.  Everything is first cycle. The directional sweep found the
      tilt peaking at four cycles and falling by twelve, which is a dynamical
      statement never made precise. Here the interval is recomputed at each
      cycle of a repeated sense-actuate-reset run to see whether symmetry is a
      property of the first cycle or of the whole trajectory.

WHAT WOULD MAKE THE PAPER BIGGER: symmetry surviving all three. Then the
statement is not "this protocol cannot cool" but "no conditional-feedback
protocol built from a spectrally symmetric generator can cool, at any ancilla
count, with or without product structure, at any point in a repeated run".

WHAT WOULD MAKE IT SMALLER, and is worth knowing: if any extension breaks the
symmetry, the no-go is narrower than the manuscript implies and the scope
sentence has to say so.
"""

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
THETA, TAU = 0.2, 1.042
N = 3
D = 2 ** N


def op(P, i, n):
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def sum_x(n):
    return sum(op(X, i, n) for i in range(n))


def build_H(seed=5, n=N):
    r = np.random.default_rng(seed)
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            H = H + r.uniform(-1, 1) * (op(Z, i, n) @ op(Z, j, n))
    for i in range(n):
        H = H + r.uniform(-.5, .5) * op(X, i, n)
    return H


def sym_defect(A):
    ev = np.sort(np.linalg.eigvalsh(A))
    return float(np.max(np.abs(ev + ev[::-1])))


def interval_from(M, K):
    lm = np.sort(np.linalg.eigvalsh(M))[::-1]
    lk = np.sort(np.linalg.eigvalsh(K))[::-1]
    hi = (THETA / 2.0) * float(np.dot(lm, lk))
    lo = (THETA / 2.0) * float(np.dot(lm, lk[::-1]))
    return lo, hi


Hm = build_H()

print("=" * 96)
print("(1)  k ANCILLAS  -  is the symmetric interval a one-ancilla artefact?")
print("=" * 96)
print("  Sensing controlled on each ancilla; feedback generator P_1^(x k) (x) sum X.")
print(f"  {'k':>3}{'joint dim':>11}{'spec(K) def':>14}{'W_lo':>11}{'W_hi':>11}"
      f"{'|lo+hi|':>11}{'verdict':>12}")
print("  " + "-" * 73)

for k in (1, 2, 3):
    dA = 2 ** k
    psi = np.ones(D) / np.sqrt(D)
    anc = np.ones(dA) / np.sqrt(dA)
    joint = np.kron(psi, anc)

    ev, evec = np.linalg.eigh(Hm)
    # controlled evolution: ancilla basis state a advances the system by a*tau
    U = np.zeros((D * dA, D * dA), dtype=complex)
    for a in range(dA):
        ua = (evec * np.exp(-1j * ev * TAU * a)) @ evec.conj().T
        Pa = np.zeros((dA, dA), dtype=complex)
        Pa[a, a] = 1.0
        U = U + np.kron(ua, Pa)
    joint = U @ joint
    had = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Hk = had
    for _ in range(k - 1):
        Hk = np.kron(Hk, had)
    joint = np.kron(np.eye(D), Hk) @ joint

    rho = np.outer(joint, joint.conj())
    A = np.kron(Hm, np.eye(dA))
    M = 1j * (rho @ A - A @ rho)

    Pk = np.zeros((dA, dA), dtype=complex)
    Pk[dA - 1, dA - 1] = 1.0
    K = np.kron(sum_x(N), Pk)

    lo, hi = interval_from(M, K)
    print(f"  {k:>3}{D * dA:>11}{sym_defect(K):>14.2e}{lo:>11.6f}{hi:>11.6f}"
          f"{abs(lo + hi):>11.1e}"
          f"{('SYMMETRIC' if abs(lo + hi) < 1e-9 else 'asym'):>12}")

print()
print("=" * 96)
print("(2)  NON-PRODUCT COUPLING  -  does spec(K) replace spec(Y)?")
print("=" * 96)
print("  K is a random Hermitian on the JOINT space, optionally symmetrised.")
print(f"  {'K type':>22}{'spec(K) def':>14}{'W_lo':>11}{'W_hi':>11}"
      f"{'|lo+hi|':>11}{'verdict':>12}")
print("  " + "-" * 81)

psi = np.ones(D) / np.sqrt(D)
anc = np.array([1.0, 1.0]) / np.sqrt(2)
ev, evec = np.linalg.eigh(Hm)
u = (evec * np.exp(-1j * ev * TAU)) @ evec.conj().T
p0 = np.array([[1.0, 0], [0, 0]], dtype=complex)
p1 = np.array([[0, 0], [0, 1.0]], dtype=complex)
had = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
j0 = np.kron(psi, anc)
j0 = (np.kron(np.eye(D), p0) + np.kron(u, p1)) @ j0
j0 = np.kron(np.eye(D), had) @ j0
rho0 = np.outer(j0, j0.conj())
A0 = np.kron(Hm, np.eye(2))
M0 = 1j * (rho0 @ A0 - A0 @ rho0)

rng = np.random.default_rng(2)
for tag, mk in [
    ("product P1 (x) sumX", lambda: np.kron(sum_x(N), p1)),
    ("random joint Herm", lambda: (lambda R: (R + R.conj().T) / 2)(
        rng.normal(size=(2 * D, 2 * D)) + 1j * rng.normal(size=(2 * D, 2 * D)))),
    ("random, symmetrised", lambda: (lambda R: np.block(
        [[np.zeros((D, D)), R], [R.conj().T, np.zeros((D, D))]]))(
        rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))),
]:
    K = mk()
    lo, hi = interval_from(M0, K)
    print(f"  {tag:>22}{sym_defect(K):>14.2e}{lo:>11.6f}{hi:>11.6f}"
          f"{abs(lo + hi):>11.1e}"
          f"{('SYMMETRIC' if abs(lo + hi) < 1e-9 else 'asym'):>12}")

print()
print("=" * 96)
print("(3)  MULTI-CYCLE  -  is symmetry a first-cycle property?")
print("=" * 96)
Ym = sum_x(N)
K = np.kron(Ym, p1)
ufb = expm(-1j * (THETA / 2.0) * Ym)
rho_s = np.outer(psi, psi.conj())
ancm = np.array([[.5, .5], [.5, .5]], dtype=complex)
print(f"  {'cycle':>7}{'purity':>10}{'spec(M) def':>14}{'W_lo':>11}{'W_hi':>11}"
      f"{'|lo+hi|':>11}{'verdict':>12}")
print("  " + "-" * 76)
for c in range(1, 9):
    r = np.kron(rho_s, ancm)
    Us = np.kron(np.eye(D), p0) + np.kron(u, p1)
    r = Us @ r @ Us.conj().T
    r = np.kron(np.eye(D), had) @ r @ np.kron(np.eye(D), had).conj().T
    M = 1j * (r @ A0 - A0 @ r)
    lo, hi = interval_from(M, K)
    pur = float(np.real(np.trace(rho_s @ rho_s)))
    print(f"  {c:>7}{pur:>10.4f}{sym_defect(M):>14.2e}{lo:>11.6f}{hi:>11.6f}"
          f"{abs(lo + hi):>11.1e}"
          f"{('SYMMETRIC' if abs(lo + hi) < 1e-9 else 'asym'):>12}")
    Uf = np.kron(np.eye(D), p0) + np.kron(ufb, p1)
    r = Uf @ r @ Uf.conj().T
    rho_s = r.reshape(D, 2, D, 2)[:, 0, :, 0] + r.reshape(D, 2, D, 2)[:, 1, :, 1]
    rho_s = rho_s / np.trace(rho_s)

print()
print("  SYMMETRIC on every row of all three blocks means the no-go covers any")
print("  ancilla count, any coupling with a spectrally symmetric generator, and")
print("  every cycle of a repeated run - a statement about a CLASS rather than")
print("  about one protocol. Any 'asym' row narrows it, and the scope sentence in")
print("  the manuscript then has to name the restriction that is doing the work.")
