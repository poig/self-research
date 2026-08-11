"""Does the escape route scale? The three numbers measured at small N.

The theorems are already size-independent: the identity holds in any finite
dimension (formalised that way in Lean), von Neumann's inequality holds in any
dimension, and condition (A) holds at every n because spec(sum_i X_i) = {n-2k} is
symmetric by construction. Scaling cannot break the no-go.

What was measured at ONE size is the quantitative story, and each number implies
something different if it moves:

  |D| <= 0.0147        the directional fraction a unitary kick can reach.
                       SHRINKING with N turns the no-go into an asymptotic
                       statement - the obstruction tightens with system size.
                       GROWING would mean the escape route is more real at scale
                       than the manuscript implies.

  ~32 filter samples   the number of distinct Hamiltonian evolutions the filtered
                       repair needs per cycle. THIS IS THE ONE THAT MATTERS.
                       POLYNOMIAL in N: the repair is viable, the paper's message
                       stays "use the filter, here is the price". EXPONENTIAL:
                       the only known repair is itself unscalable and the verdict
                       becomes "this protocol family does not scale, use phase
                       estimation" - a much larger claim, and the one that would
                       interest people deciding whether to build at all.

  94.5% to ground      how far the filtered protocol actually gets. Expected to
                       degrade, since there are exponentially many states to cool
                       through; the rate is the practical question.

NOTE WHICH ANSWER IS WORTH MORE. A scalable filter is what everyone already
assumes. An unscalable one is news. The pessimistic outcome is the valuable one
here, which is worth saying before the numbers arrive rather than after.

The filter threshold is defined operationally: the smallest sample count at which
the discretised K still COOLS (final energy below the start). Below threshold the
sign inverts and the protocol heats, so the threshold is sharp rather than a
matter of tolerance.
"""

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Yp = np.array([[0, -1j], [1j, 0]], dtype=complex)
RAISE = np.array([[0, 0], [1, 0]], dtype=complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)

THETA_FB, TAU = 0.2, 1.042
MU, SIG = -1.6, 0.8


def op(P, i, n):
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def sum_p(P, n):
    return sum(op(P, i, n) for i in range(n))


def build_H(n, g=0.0, seed=5):
    r = np.random.default_rng(seed)
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            H = H + r.uniform(-1, 1) * (op(Z, i, n) @ op(Z, j, n))
            if g:
                H = H + g * (op(X, i, n) @ op(Yp, j, n) - op(Yp, i, n) @ op(X, j, n))
    for i in range(n):
        H = H + r.uniform(-.5, .5) * op(X, i, n)
    return H


# ---------------------------------------------------------------- |D| vs N
def directional(n, cycles=4, a=1.0, g=1.0):
    """Largest directional fraction with BOTH conditions broken as hard as the
    N=3 sweep allowed: maximally asymmetric generator, complex H, mixed branch."""
    d = 2 ** n
    Hm = build_H(n, g=g)
    S = sum_p(X, n)
    P = np.zeros((d, d), dtype=complex)
    P[0, 0] = 1.0
    Ym = (1 - a) * S + a * P

    ev, evec = np.linalg.eigh(Hm)
    u = (evec * np.exp(-1j * ev * TAU)) @ evec.conj().T
    p0 = np.array([[1.0, 0], [0, 0]], dtype=complex)
    p1 = np.array([[0, 0], [0, 1.0]], dtype=complex)
    had = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    anc = np.array([[.5, .5], [.5, .5]], dtype=complex)
    ufb = expm(-1j * (THETA_FB / 2.0) * sum_p(X, n))

    rho_s = np.full((d, d), 1.0 / d, dtype=complex)
    for _ in range(cycles):
        r = np.kron(rho_s, anc)
        Us = np.kron(np.eye(d), p0) + np.kron(u, p1)
        r = Us @ r @ Us.conj().T
        Hd = np.kron(np.eye(d), had)
        r = Hd @ r @ Hd.conj().T
        Uf = np.kron(np.eye(d), p0) + np.kron(ufb, p1)
        r = Uf @ r @ Uf.conj().T
        rho_s = r.reshape(d, 2, d, 2)[:, 0, :, 0] + r.reshape(d, 2, d, 2)[:, 1, :, 1]
        rho_s = rho_s / np.trace(rho_s)

    r = np.kron(rho_s, anc)
    Us = np.kron(np.eye(d), p0) + np.kron(u, p1)
    r = Us @ r @ Us.conj().T
    Hd = np.kron(np.eye(d), had)
    r = Hd @ r @ Hd.conj().T
    A = np.kron(Hm, np.eye(2))
    M = 1j * (r @ A - A @ r)
    M11 = M.reshape(d, 2, d, 2)[:, 1, :, 1]

    lm = np.sort(np.linalg.eigvalsh(M11))[::-1]
    ly = np.sort(np.linalg.eigvalsh(Ym))[::-1]
    hi = (THETA_FB / 2.0) * float(np.dot(lm, ly))
    lo = (THETA_FB / 2.0) * float(np.dot(lm, ly[::-1]))
    w = hi - lo
    return (hi + lo) / w if w > 1e-14 else 0.0


# ---------------------------------------------------------- filter vs N
def filtered_K(Hm, lam, V, A, n_times=None, T=14.0):
    Ae = V.conj().T @ A @ V
    w = lam[:, None] - lam[None, :]
    if n_times is None:
        F = np.exp(-(w - MU) ** 2 / (2 * SIG ** 2)).astype(complex)
    else:
        s = np.linspace(-T, T, n_times)
        ds = s[1] - s[0]
        f = np.exp(-SIG ** 2 * s ** 2 / 2.0) * np.exp(-1j * MU * s)
        F = np.zeros_like(Ae)
        for sv, fv in zip(s, f):
            F = F + fv * np.exp(1j * w * sv) * ds
        F = F * (SIG / np.sqrt(2 * np.pi))
    return V @ (F * Ae) @ V.conj().T


def collide_energy(rho, Kop, Hm, theta, cycles):
    d = Hm.shape[0]
    Gj = np.kron(Kop, RAISE) + np.kron(Kop.conj().T, LOWER)
    U = expm(-1j * theta * Gj)
    anc = np.zeros((2, 2), dtype=complex)
    anc[0, 0] = 1.0
    for _ in range(cycles):
        r = np.kron(rho, anc)
        r = U @ r @ U.conj().T
        rho = r.reshape(d, 2, d, 2)[:, 0, :, 0] + r.reshape(d, 2, d, 2)[:, 1, :, 1]
        rho = rho / np.trace(rho)
    return float(np.real(np.trace(rho @ Hm)))


print("=" * 98)
print("DOES THE ESCAPE ROUTE SCALE?")
print("=" * 98)
print("  The theorems already do. These three numbers were measured at one size.")
print()
print(f"  {'N':>3}{'dim':>6}{'|D|':>16}{'cool% @40cyc':>13}"
      f"{'cyc to 80%':>12}{'samples@90%':>13}{'gap ratio':>14}")
print("  " + "-" * 78)

SAMPLES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
CYC, TH = 40, 0.3
rows = []
for n in (3, 4, 5, 6):
    d = 2 ** n
    Hm = build_H(n)
    lam, V = np.linalg.eigh(Hm)
    A = sum_p(X, n)
    psi = np.ones(d) / np.sqrt(d)
    rho0 = np.outer(psi, psi.conj())
    E0 = float(np.real(np.trace(rho0 @ Hm)))
    Egs = float(lam[0])

    Kid = filtered_K(Hm, lam, V, A)
    e_ideal = collide_energy(rho0.copy(), Kid, Hm, TH, CYC)
    frac_ideal = 100 * (E0 - e_ideal) / (E0 - Egs)

    # SAMPLES TO REACH 90% OF THE IDEAL FILTER'S OWN PERFORMANCE.
    # The earlier threshold was "smallest count that cools at all", which is not
    # comparable across N: the start state sits further above the ground state as
    # N grows, so "moves down at all" becomes EASIER and the threshold fell with
    # N for a reason that had nothing to do with the filter. Measuring against
    # the ideal filter's own result at the same N removes that.
    target = E0 - 0.90 * (E0 - e_ideal)
    thresh = None
    for ns in SAMPLES:
        Kd = filtered_K(Hm, lam, V, A, n_times=ns)
        e = collide_energy(rho0.copy(), Kd, Hm, TH, CYC)
        if e <= target:
            thresh = ns
            break

    # CYCLES the ideal filter needs to cover 80% of the distance to the ground
    # state, so the cooling-fraction column is not just "40 cycles is less of the
    # way when there is more to cool".
    cyc80 = None
    rr = rho0.copy()
    Gj = np.kron(Kid, RAISE) + np.kron(Kid.conj().T, LOWER)
    Uc = expm(-1j * TH * Gj)
    ancz = np.zeros((2, 2), dtype=complex)
    ancz[0, 0] = 1.0
    for c in range(1, 401):
        r = np.kron(rr, ancz)
        r = Uc @ r @ Uc.conj().T
        rr = r.reshape(d, 2, d, 2)[:, 0, :, 0] + r.reshape(d, 2, d, 2)[:, 1, :, 1]
        rr = rr / np.trace(rr)
        if float(np.real(np.trace(rr @ Hm))) <= E0 - 0.80 * (E0 - Egs):
            cyc80 = c
            break

    Dfrac = directional(n)
    # spectral scale the filter must resolve: full width over min nonzero gap
    gaps = np.diff(np.sort(lam))
    gaps = gaps[gaps > 1e-9]
    ratio = float((lam[-1] - lam[0]) / np.min(gaps)) if len(gaps) else float('nan')

    rows.append((n, abs(Dfrac), frac_ideal, thresh, cyc80, ratio))
    print(f"  {n:>3}{d:>6}{abs(Dfrac):>16.4f}{frac_ideal:>12.1f}%"
          f"{str(cyc80):>12}{str(thresh):>13}{ratio:>14.1f}", flush=True)

print()
print("  READING IT.")
print("  |D| falling with N   -> the obstruction TIGHTENS at scale; the no-go")
print("                          becomes asymptotic rather than small-system.")
print("  min samples ~ poly N -> the filtered repair is viable and the message")
print("                          stays 'use the filter, here is the price'.")
print("  min samples blowing  -> the ONLY known repair is itself unscalable, and")
print("                          the verdict becomes a statement about the whole")
print("                          protocol family rather than about one design.")
print("  'gap ratio' is the spectral width divided by the smallest gap: the")
print("  resolution the filter would need if it had to separate individual")
print("  levels. If min samples tracks N while the gap ratio explodes, the filter")
print("  is only making a COARSE positive/negative cut and stays cheap - which is")
print("  the optimistic outcome and the one worth checking carefully.")
