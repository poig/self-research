"""Breaking both conditions is necessary. Is it SUFFICIENT to be useful?

two_conditions.py established the qualitative statement: the reachable interval
is symmetric unless BOTH spec(Y) and spec(M11) are asymmetric, and it exhibited
one configuration where breaking both does produce an asymmetric interval:

    projector generator, mixed branch, complex H, 4 cycles
        W in [-0.01910, +0.01921]                             ASYMMETRIC

That is the manuscript's escape route, and read carelessly it says the protocol
can be repaired. Read carefully it says almost nothing, because the asymmetry is
0.00011 on an interval of width 0.038 - SIX TENTHS OF ONE PERCENT. A protocol
whose reachable work is [-0.01910, +0.01921] still heats essentially as readily
as it cools. Necessary is not sufficient, and the manuscript currently stops at
necessary.

THE QUANTITY THAT DECIDES IT is the directional fraction

    D  :=  (W_hi + W_lo) / (W_hi - W_lo)   in  [-1, +1]

D = 0 is a symmetric interval and no preferred direction. |D| = 1 is a fully
one-sided interval, which is what a cooling protocol needs. The last row above
has D = 0.0029. The question this file asks is whether D can be pushed to O(1)
by breaking the two conditions HARDER, or whether it stays near zero for every
conditional unitary feedback of this form.

WHAT IS SWEPT. Both conditions are broken continuously rather than switched:

    generator   Y(a) = (1-a) sum_i X_i  +  a P,  P a rank-1 projector
                a = 0 is condition (A) intact, a = 1 is maximally asymmetric
    branch      mixedness via c cycles of sense-actuate-reset, and a complex
                Hamiltonian with a Dzyaloshinskii-Moriya term of strength g
                (g = 0 keeps H real, which keeps spec(M11) symmetric)

D is reported over the grid. If max |D| stays small, the honest reading is that
breaking both conditions is necessary but nowhere near sufficient, and the
manuscript's escape route needs the frequency filter for a reason stronger than
the one currently given: not merely that an unfiltered kick fails to prefer a
direction, but that no unitary conditional kick can prefer one by more than a
few percent.

That is a STRONGER no-go than the paper currently claims, and it is the honest
conclusion if the numbers say so.
"""

import numpy as np
from scipy.linalg import expm

N = 3
D = 2 ** N
TAU, THETA = 1.042, 0.2

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def op(P, i, n=N):
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def sum_x(n=N):
    return sum(op(X, i, n) for i in range(n))


def build_H(g, seed=7):
    """Heisenberg-like chain; g>0 adds a Dzyaloshinskii-Moriya term, making H
    complex and breaking the time-reversal symmetry that keeps spec(M11)
    symmetric."""
    r = np.random.default_rng(seed)
    H = np.zeros((D, D), dtype=complex)
    for i in range(N - 1):
        j = i + 1
        Jc = r.uniform(0.5, 1.5)
        H = H + Jc * (op(Z, i) @ op(Z, j) + op(X, i) @ op(X, j))
        if g:
            # DM term  g (X_i Y_j - Y_i X_j) : Hermitian, complex, breaks T
            H = H + g * (op(X, i) @ op(Y, j) - op(Y, i) @ op(X, j))
    for i in range(N):
        H = H + r.uniform(-0.5, 0.5) * op(Z, i)
    return H


def generator(a):
    """Y(a) interpolates from the symmetric sum X_i to a rank-1 projector."""
    S = sum_x()
    P = np.zeros((D, D), dtype=complex)
    P[0, 0] = 1.0
    return (1 - a) * S + a * P


def m11_from_branch(rho_s, Hm):
    """Post-sensing joint state from a possibly mixed system state, then the
    (anc=1,anc=1) block of i[rho, A] with A = I_A (x) H."""
    ev, evec = np.linalg.eigh(Hm)
    u = (evec * np.exp(-1j * ev * TAU)) @ evec.conj().T
    p0 = np.array([[1.0, 0], [0, 0]], dtype=complex)
    p1 = np.array([[0, 0], [0, 1.0]], dtype=complex)
    had = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    anc = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)   # |+><+|
    rho = np.kron(rho_s, anc)
    Us = np.kron(np.eye(D), p0) + np.kron(u, p1)
    rho = Us @ rho @ Us.conj().T
    Hd = np.kron(np.eye(D), had)
    rho = Hd @ rho @ Hd.conj().T
    A = np.kron(Hm, np.eye(2))
    M = 1j * (rho @ A - A @ rho)
    return M.reshape(D, 2, D, 2)[:, 1, :, 1]


def mixed_branch(Hm, Ym, cycles):
    """Run the protocol `cycles` times with reset, returning the system state."""
    rho_s = np.full((D, D), 1.0 / D, dtype=complex)      # |+>^n
    ufb = expm(-1j * (THETA / 2.0) * Ym)
    ev, evec = np.linalg.eigh(Hm)
    u = (evec * np.exp(-1j * ev * TAU)) @ evec.conj().T
    p0 = np.array([[1.0, 0], [0, 0]], dtype=complex)
    p1 = np.array([[0, 0], [0, 1.0]], dtype=complex)
    had = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    anc = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    for _ in range(cycles):
        rho = np.kron(rho_s, anc)
        Us = np.kron(np.eye(D), p0) + np.kron(u, p1)
        rho = Us @ rho @ Us.conj().T
        Hd = np.kron(np.eye(D), had)
        rho = Hd @ rho @ Hd.conj().T
        Ufb = np.kron(np.eye(D), p0) + np.kron(ufb, p1)
        rho = Ufb @ rho @ Ufb.conj().T
        rho_s = rho.reshape(D, 2, D, 2)[:, 0, :, 0] + rho.reshape(D, 2, D, 2)[:, 1, :, 1]
        rho_s = rho_s / np.trace(rho_s)
    return rho_s


def interval(M, Ym):
    lm = np.sort(np.linalg.eigvalsh(M))[::-1]
    ly = np.sort(np.linalg.eigvalsh(Ym))[::-1]
    hi = (THETA / 2.0) * float(np.dot(lm, ly))
    lo = (THETA / 2.0) * float(np.dot(lm, ly[::-1]))
    return lo, hi


def defect(M):
    ev = np.sort(np.linalg.eigvalsh(M))
    return float(np.max(np.abs(ev + ev[::-1])))


AS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
GS = (0.0, 0.3, 1.0)
CS = (1, 4, 12)

print("=" * 100)
print("HOW DIRECTIONAL CAN AN UNFILTERED CONDITIONAL KICK GET?")
print("=" * 100)
print(f"  N={N}, tau={TAU}, theta={THETA}.  D = (W_hi + W_lo)/(W_hi - W_lo).")
print(f"  D = 0 is a symmetric interval (no preferred direction);")
print(f"  |D| = 1 is one-sided, which is what cooling needs.")
print()
print(f"  {'DM g':>6}{'cycles':>8}{'a':>6}{'spec(Y) def':>13}{'spec(M11) def':>15}"
      f"{'W_lo':>10}{'W_hi':>10}{'D':>10}")
print("  " + "-" * 78)

best = (0.0, None)
for g in GS:
    Hm = build_H(g)
    for c in CS:
        for a in AS:
            Ym = generator(a)
            rho_s = mixed_branch(Hm, generator(0.0), c) if c > 1 else \
                np.full((D, D), 1.0 / D, dtype=complex)
            M = m11_from_branch(rho_s, Hm)
            lo, hi = interval(M, Ym)
            width = hi - lo
            Dfrac = (hi + lo) / width if width > 1e-14 else 0.0
            if abs(Dfrac) > abs(best[0]):
                best = (Dfrac, (g, c, a))
            print(f"  {g:>6.1f}{c:>8}{a:>6.2f}{defect(Ym):>13.2e}"
                  f"{defect(M):>15.2e}{lo:>10.5f}{hi:>10.5f}{Dfrac:>10.4f}",
                  flush=True)
        print("  " + "." * 78)

print()
print(f"  LARGEST |D| over the grid: {abs(best[0]):.4f}  at "
      f"(g, cycles, a) = {best[1]}")
print()
print("  A largest |D| of a few percent means breaking both conditions is")
print("  NECESSARY BUT NOWHERE NEAR SUFFICIENT: the interval tilts, but a")
print("  protocol whose reachable work is still nearly symmetric heats almost as")
print("  readily as it cools. That would strengthen the manuscript's conclusion")
print("  rather than weaken it: the frequency filter is needed not merely because")
print("  an unfiltered kick has no preferred direction, but because no amount of")
print("  breaking these two conditions buys a useful one.")
print("  A |D| approaching 1 would say the opposite - that the escape route is")
print("  real and the filter is one option among several.")
