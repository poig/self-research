"""At what temperature does the Landauer comparison actually hold?

The manuscript states twice that the work extracted per cycle stays below
k_B T ln2 * S(A), and never defines T. landauer_limit_test.py sets KBT = 1.0 with
its own comment that "choosing k_B T is a modeling assumption rather than a
derived physical temperature", because the Hamiltonian couplings are
dimensionless draws from U(-1,1) and U(-0.5,0.5). An inequality against an
undefined scale is not falsifiable: it is true for large enough T and false for
small enough T, and nothing in the paper says which side it is on.

So compute the crossing. For each sensing time,

    T_req(tau) = W(tau) / (ln2 * S_A(tau))

is the smallest k_B T at which the erasure cost still covers the work at that
tau. The claim "work stays below the erasure cost at every tau" is therefore
exactly the claim k_B T >= max_tau T_req(tau), and that maximum is a number this
file reports. Quoting it turns an unfalsifiable sentence into a conditional one.

Everything is computed exactly from the statevector - same protocol, same
Hamiltonian seed, same theta and tau grid as landauer_limit_test.py - so the
number is commensurable with Fig. 5 rather than with a re-derivation.

  sense     ancilla |+>, controlled e^{-iH tau}
  correlate H on the ancilla, phase -> population
  feedback  CRX(theta) from ancilla onto every system qubit
  W         -Delta<H> across the feedback
  S_A       von Neumann entropy of the ancilla marginal, in BITS
"""
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

N, THETA, TAU_STEPS, MAX_TAU = 4, 0.2, 20, 1.5
LN2 = np.log(2)


def op(P, i, n):
    m = np.ones((1, 1), dtype=complex)
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def build_H(n):
    """Same construction and seed as landauer_limit_test.py."""
    np.random.seed(42)
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        J = np.random.uniform(-1, 1)
        H = H + J * (op(Z, i, n) @ op(Z, i + 1, n))
    for i in range(n):
        h = np.random.uniform(-0.5, 0.5)
        H = H + h * op(X, i, n)
    return H


def entropy_bits(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-12]
    return float(-np.sum(w * np.log2(w)))


H = build_H(N)
d = 2 ** N
A = np.kron(np.eye(2), H)

# |psi> = |+>^n, the protocol's system input
psi = np.ones(d, dtype=complex) / np.sqrt(d)

P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)
Ykick = sum(op(X, i, N) for i in range(N))
Ufb = np.kron(P0, np.eye(d)) + np.kron(P1, expm(-1j * (THETA / 2) * Ykick))

print("=" * 88)
print("THRESHOLD TEMPERATURE FOR THE LANDAUER COMPARISON")
print("=" * 88)
print(f"  N={N}, theta={THETA}, tau grid {TAU_STEPS} points to {MAX_TAU}, seed 42.")
print(f"  T_req(tau) = W / (ln2 * S_A): the smallest k_B T at which the erasure")
print(f"  cost still covers the work at that tau. Energy units = Hamiltonian units.")
print()
print(f"  {'tau':>7}{'W':>12}{'S_A (bits)':>13}{'cost@kT=1':>12}{'T_req':>12}")
print("  " + "-" * 56)

best_T, best_tau = -np.inf, None
for tau in np.linspace(0, MAX_TAU, TAU_STEPS):
    Utau = expm(-1j * H * tau)
    Us = np.kron(P0, np.eye(d)) + np.kron(P1, Utau)
    state = Us @ np.kron(np.array([1, 1]) / np.sqrt(2), psi)
    state = np.kron(HAD, np.eye(d)) @ state          # phase -> population

    e0 = float(np.real(state.conj() @ A @ state))
    out = Ufb @ state
    e1 = float(np.real(out.conj() @ A @ out))
    W = e0 - e1                                       # extracted work

    rho = np.outer(state, state.conj())
    rho_A = np.array([[np.trace(rho[i * d:(i + 1) * d, j * d:(j + 1) * d])
                       for j in range(2)] for i in range(2)])
    SA = entropy_bits(rho_A)

    cost1 = LN2 * SA
    Treq = W / cost1 if cost1 > 1e-12 else (np.inf if W > 0 else -np.inf)
    if np.isfinite(Treq) and Treq > best_T:
        best_T, best_tau = Treq, tau
    print(f"  {tau:>7.3f}{W:>12.5f}{SA:>13.5f}{cost1:>12.5f}"
          f"{Treq:>12.4f}")

print()
print(f"  THRESHOLD  k_B T* = {best_T:.4f}  (attained at tau = {best_tau:.3f})")
print()
print(f"  The manuscript's statement is therefore true exactly when")
print(f"      k_B T  >=  {best_T:.3f}   in units of the Hamiltonian couplings,")
print(f"  which are dimensionless draws from U(-1,1) and U(-0.5,0.5). At the")
print(f"  hardcoded k_B T = 1.0 the claim holds with {1.0 / best_T:.1f}x margin;")
print(f"  below k_B T = {best_T:.3f} it fails and the figure's conclusion reverses.")
print()
print("  Quote the threshold, not the inequality. The inequality alone is true by")
print("  choice of an undefined scale and says nothing about the protocol.")
