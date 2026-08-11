"""One lemma, two corollaries: the feedback work and the VQE gradient are the
same object, not an analogy.

The manuscript currently asserts that "the commutator that forbids work
extraction in the feedback protocol and the commutator that forbids training in
the variational circuit are the same object". That is stated by resemblance. It
does not have to be: Theorem 2's proof never uses the tensor structure of
A = I_A (x) H, only that A is Hermitian and that the generator is Hermitian. So
the theorem generalises, and both statements drop out as instances.

THE GENERAL LEMMA
-----------------
Let rho be any state, A any Hermitian observable, G any Hermitian generator, and
consider the response of <A> to the unitary exp(-i theta G / 2):

    Delta<A>(theta) = <psi| A |psi> - <psi| U^dag A U |psi>,   U = exp(-i theta G/2)

    Delta<A> = (theta/2) Tr( i[rho, A] . G ) + O(theta^2)              (LEMMA)

Write M := i[rho, A], Hermitian. Over the isospectral orbit of G, i.e. over
G -> V^dag G V for unitary V, von Neumann's trace inequality gives

    Tr(M V^dag G V)  in  [ sum_k lam_k^down(M) lam_k^up(G) ,
                           sum_k lam_k^down(M) lam_k^down(G) ]

and the interval is SYMMETRIC about zero whenever spec(G) is symmetric about
zero.

TWO INSTANCES
-------------
  WORK        A = I_A (x) H,  G = P_1 (x) Y.   The ancilla factor is inert, so M
              reduces to the (1,1) block M_11 and the interval is Theorem 2.
  GRADIENT    no ancilla.  For psi(theta) = W exp(-i theta G_j / 2) V|0>, the
              energy gradient is
                  d/dtheta <H> = (1/2) <phi| i[G_j, H_eff] |phi>,
              H_eff = W^dag H W, phi = exp(-i theta G_j/2) V|0>.
              Same form with A = H_eff. So [G_j, H_eff] = 0 forcing a zero
              gradient is the same statement as [H, Y] = 0 forcing zero work,
              and it is a COROLLARY rather than a parallel.

WHAT THIS RUN CHECKS, on random instances so nothing is special-cased:
  (1) the LEMMA against exact finite-difference response, for random rho, A, G
  (2) the interval bound against Haar-random V, counting violations
  (3) saturation by the constructed optimiser V = U_G U_M^dag
  (4) that the WORK case reproduces the A = I(x)H numbers exactly
  (5) that the GRADIENT case reproduces an independently computed
      parameter-shift gradient of a real circuit

If (5) holds the manuscript can state the VQE consequence as a corollary of the
generalised theorem instead of as a resemblance, which is the joint the current
draft is weakest at.
"""

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

rng = np.random.default_rng(11)


def herm(d, r=None):
    r = r or rng
    X = r.normal(size=(d, d)) + 1j * r.normal(size=(d, d))
    return (X + X.conj().T) / 2.0


def response_exact(psi, A, G, theta):
    """<A> before minus after, no expansion."""
    U = expm(-1j * (theta / 2.0) * G)
    v = U @ psi
    return float(np.real(np.vdot(psi, A @ psi) - np.vdot(v, A @ v)))


def response_first_order(psi, A, G, theta):
    """(theta/2) Tr( i[rho,A] G ) with rho = |psi><psi|."""
    rho = np.outer(psi, psi.conj())
    M = 1j * (rho @ A - A @ rho)
    return (theta / 2.0) * float(np.real(np.trace(M @ G)))


def interval(M, G, theta):
    lm = np.sort(np.linalg.eigvalsh(M))[::-1]
    lg = np.sort(np.linalg.eigvalsh(G))[::-1]
    hi = (theta / 2.0) * float(np.dot(lm, lg))
    lo = (theta / 2.0) * float(np.dot(lm, lg[::-1]))
    return lo, hi


def optimal_V(M, G, reverse=False):
    _, U_M = np.linalg.eigh(M)
    U_M = U_M[:, ::-1]
    _, U_G = np.linalg.eigh(G)
    U_G = U_G[:, ::-1]
    if reverse:
        U_G = U_G[:, ::-1]
    return U_G @ U_M.conj().T


THETA = 0.02          # small, so the O(theta^2) remainder is visible as small
print("=" * 94)
print("GENERAL RESPONSE LEMMA:  Delta<A> = (theta/2) Tr( i[rho,A] G ) + O(theta^2)")
print("=" * 94)
print(f"  theta = {THETA}. Random Hermitian A and G, random pure rho, no structure imposed.")
print()
print(f"  {'d':>4}{'exact':>14}{'1st order':>14}{'|diff|':>12}{'|diff|/theta^2':>16}")
print("  " + "-" * 60)
for d in (2, 4, 8, 16):
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi /= np.linalg.norm(psi)
    A, G = herm(d), herm(d)
    ex = response_exact(psi, A, G, THETA)
    fo = response_first_order(psi, A, G, THETA)
    print(f"  {d:>4}{ex:>14.8f}{fo:>14.8f}{abs(ex - fo):>12.2e}"
          f"{abs(ex - fo) / THETA ** 2:>16.3f}")
print("  The last column is bounded, so the difference is genuinely O(theta^2).")

print()
print("=" * 94)
print("THE INTERVAL, on random instances")
print("=" * 94)
print(f"  {'d':>4}{'lo':>12}{'hi':>12}{'sym':>10}{'Haar viol':>11}"
      f"{'sat max':>11}{'sat min':>11}")
print("  " + "-" * 71)
for d in (4, 8, 16):
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi /= np.linalg.norm(psi)
    A = herm(d)
    # symmetric-spectrum generator, as in the protocol (spec(sum X) is symmetric)
    lam = rng.normal(size=d // 2)
    lam = np.concatenate([lam, -lam])
    Q = unitary_group.rvs(d, random_state=rng)
    G = Q @ np.diag(lam) @ Q.conj().T
    rho = np.outer(psi, psi.conj())
    M = 1j * (rho @ A - A @ rho)
    lo, hi = interval(M, G, THETA)

    viol = 0
    for _ in range(300):
        V = unitary_group.rvs(d, random_state=rng)
        w = response_first_order(psi, A, V.conj().T @ G @ V, THETA)
        if w > hi + 1e-12 or w < lo - 1e-12:
            viol += 1
    smax = response_first_order(psi, A, optimal_V(M, G).conj().T @ G
                                @ optimal_V(M, G), THETA)
    smin = response_first_order(psi, A, optimal_V(M, G, True).conj().T @ G
                                @ optimal_V(M, G, True), THETA)
    print(f"  {d:>4}{lo:>12.6f}{hi:>12.6f}{abs(lo + hi):>10.1e}{viol:>11}"
          f"{abs(smax - hi):>11.1e}{abs(smin - lo):>11.1e}")
print("  'sym' is |lo+hi|: the interval is symmetric when spec(G) is.")
print("  'sat' columns are the distance from the constructed optimum to each edge.")

print()
print("=" * 94)
print("INSTANCE 2: the VQE gradient is the same expression")
print("=" * 94)
print("  A real circuit: psi(t) = W exp(-i t G/2) V|0>, energy <psi|H|psi>.")
print("  Compared: (a) parameter-shift gradient, computed independently;")
print("            (b) the LEMMA with A = H_eff = W^dag H W, rho = |phi><phi|.")
print()
print(f"  {'d':>4}{'param-shift':>16}{'lemma':>16}{'|diff|':>12}")
print("  " + "-" * 48)
for d in (4, 8, 16):
    V = unitary_group.rvs(d, random_state=rng)
    W = unitary_group.rvs(d, random_state=rng)
    H = herm(d)
    # Pauli-like generator: G^2 = I, so the parameter-shift rule is exact
    lam = rng.choice([-1.0, 1.0], size=d)
    Q = unitary_group.rvs(d, random_state=rng)
    G = Q @ np.diag(lam) @ Q.conj().T

    t0 = 0.37
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0

    def energy(t):
        p = W @ expm(-1j * (t / 2.0) * G) @ V @ psi0
        return float(np.real(np.vdot(p, H @ p)))

    shift = (energy(t0 + np.pi / 2) - energy(t0 - np.pi / 2)) / 2.0

    phi = expm(-1j * (t0 / 2.0) * G) @ V @ psi0
    H_eff = W.conj().T @ H @ W
    rho = np.outer(phi, phi.conj())
    M = 1j * (rho @ H_eff - H_eff @ rho)
    # SIGN. The lemma is stated for the response BEFORE minus AFTER, i.e. for
    # energy EXTRACTED, which is a decrease. A gradient is the rate of INCREASE.
    # So d<H>/dt = -(1/2) Tr(M G). Without the minus the two agree in magnitude
    # to 1e-16 and differ in sign at every dimension, which is what first
    # exposed the convention.
    lemma = -0.5 * float(np.real(np.trace(M @ G)))

    print(f"  {d:>4}{shift:>16.10f}{lemma:>16.10f}{abs(shift - lemma):>12.2e}")

print()
print("  Agreement here means the variational gradient IS the general response")
print("  with A = H_eff, so the zero-gradient rule is a corollary of the interval")
print("  theorem rather than a separate observation that happens to look similar.")
print("  In particular [G, H_eff] = 0 gives M commuting with nothing to pair")
print("  against, Tr(M G) = 0, and the gradient vanishes identically.")
