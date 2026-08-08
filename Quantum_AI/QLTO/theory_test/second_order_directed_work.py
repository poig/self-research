"""The second-order term, derived and checked link by link.

Theorem 2 stops at first order, where W = (theta/2) Tr(M_11 Y) is ODD in Y, so
reversing the von Neumann pairing negates it and the interval is symmetric. That
symmetry is the whole basis for reading conditions (A)/(B) as an obstruction to
cooling. exact_interval_asymmetry.py measured the symmetry failing as
A(theta) = 2.137 theta. This file derives WHY and asks what it is good for.

DERIVATION. With U = exp(-i(theta/2)K), K = P_1 (x) Y, A = I (x) H,

    U^dag A U = A + i(theta/2)[K,A] - (theta^2/8)[K,[K,A]] + O(theta^3)
    W = <A> - <U^dag A U>
      = (theta/2)<i[A,K]> + (theta^2/8)<[K,[K,A]]> + O(theta^3).

The nested commutator collapses because K and A share the P_1 structure:

    [K,A] = P_1 (x) [Y,H]
    [K,[K,A]] = [P_1 (x) Y, P_1 (x) [Y,H]] = P_1 (x) [Y,[Y,H]]

and the |1> branch of |Psi_1> = (|0>|psi> + |1>|phi>)/sqrt2 carries weight 1/2:

    W = (theta/2) Tr(M_11 Y)  +  (theta^2/16) <phi| [Y,[Y,H]] |phi>  + O(theta^3)
         W_1 theta, ODD in Y        W_2 theta^2, EVEN in Y

THE MECHANISM, in one line: reversing the pairing flips the sign of the linear
term and leaves the quadratic one alone, so the two endpoints do not negate each
other and the interval tilts by 2 W_2 theta^2.

WHY IT MIGHT MATTER FOR ENGINEERING. Tr(M_11 Y) needs M_11, which needs the joint
state - the paper's own "implementable frames" paragraph says W* is a bound
rather than a prescription for exactly this reason. But <phi|ad_Y^2(H)|phi>
depends on no alignment at all. In the eigenbasis of Y,

    ad_Y^2(H) = sum_{k,l} (y_k - y_l)^2 H_{kl} |k><l|,

a reweighting of H that is manifestly independent of how spec(Y) is ORDERED
against spec(M_11). A FIXED generator therefore extracts directed work at second
order. For the single-qubit case Y = X, H = Z: [X,[X,Z]] = 4Z, so
W_2 ~ <Z> - extraction proportional to the current excess energy, which is
exponential relaxation rather than a symmetric kick.

CHECKS, each falsifiable
  (1) W_exact - (W_1 theta + W_2 theta^2) is O(theta^3)
  (2) the measured c = 2.137 is PREDICTED by W_1 and W_2 at the two optimising V
  (3) <phi|ad_Y^2(H)|phi> is sign-definite over random states, or it is not
  (4) the second-order term is alignment-free: its spread over V is small next to
      the first-order term's, which sweeps the whole interval
"""
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Yp = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def op(P, i, n):
    m = np.ones((1, 1), dtype=complex)
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def heisenberg(n):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        for P in (X, Yp, Z):
            H = H + op(P, i, n) @ op(P, i + 1, n)
    return H


def comm(a, b):
    return a @ b - b @ a


def exact_W(Psi, H, Yr, theta, n):
    d = 2 ** n
    A = np.kron(np.eye(2), H)
    K = np.kron(np.diag([0.0, 1.0]).astype(complex), Yr)
    U = expm(-1j * (theta / 2) * K)
    return float(np.real(Psi.conj() @ (A - U.conj().T @ A @ U) @ Psi))


def W1_W2(Psi, H, Yr, n):
    """The two derived coefficients, computed from the operators directly."""
    d = 2 ** n
    A = np.kron(np.eye(2), H)
    rho = np.outer(Psi, Psi.conj())
    M11 = (1j * (rho @ A - A @ rho))[d:, d:]
    W1 = 0.5 * float(np.real(np.trace(M11 @ Yr)))
    phi = Psi[d:] * np.sqrt(2.0)                     # |1> branch, normalised
    dc = comm(Yr, comm(Yr, H))
    W2 = float(np.real(phi.conj() @ dc @ phi)) / 16.0
    return W1, W2


def pairing_V(M11, Yg, reverse):
    """Theorem 2's optimising V = U_Y U_M^dag, matched or reversed ordering."""
    wm, Um = np.linalg.eigh(M11)
    wy, Uy = np.linalg.eigh(Yg)
    om = np.argsort(wm)[::-1]
    oy = np.argsort(wy)
    if not reverse:
        oy = oy[::-1]
    return Uy[:, oy] @ Um[:, om].conj().T


n, tau = 3, 0.35
H = heisenberg(n)
Yg = sum(op(X, i, n) for i in range(n))
d = 2 ** n
psi = np.linalg.qr(np.random.RandomState(3).randn(d, d))[0][:, 0]
Psi = np.concatenate([psi, expm(-1j * H * tau) @ psi]) / np.sqrt(2.0)
A = np.kron(np.eye(2), H)
rho = np.outer(Psi, Psi.conj())
M11 = (1j * (rho @ A - A @ rho))[d:, d:]

print("=" * 92)
print("THE SECOND-ORDER TERM — derivation checked link by link")
print("=" * 92)
print(f"  n={n}, tau={tau}, real Heisenberg, Y = sum_i X_i. Same state as")
print(f"  exact_interval_asymmetry.py, which measured A(theta) = 2.137 theta.")

# ── (1) does the expansion reproduce the exact W? ────────────────────────────
Vmax = pairing_V(M11, Yg, reverse=False)
Yr = Vmax.conj().T @ Yg @ Vmax
W1, W2 = W1_W2(Psi, H, Yr, n)
print(f"\n  (1) EXPANSION vs EXACT, at the first-order optimising V.")
print(f"      W_1 = {W1:+.6f}   W_2 = {W2:+.6f}")
print(f"  {'theta':>9}{'W_exact':>13}{'W1*th+W2*th^2':>16}{'residual':>13}"
      f"{'resid/th^3':>13}")
print("  " + "-" * 64)
for th in (0.005, 0.01, 0.02, 0.04, 0.08):
    we = exact_W(Psi, H, Yr, th, n)
    wp = W1 * th + W2 * th * th
    r = we - wp
    print(f"  {th:>9.4f}{we:>13.6f}{wp:>16.6f}{r:>13.2e}{r / th ** 3:>13.4f}")
print("      resid/theta^3 constant => the O(theta^3) truncation is what is left,")
print("      and the two derived coefficients are correct.")

# ── (2) does it PREDICT the measured c = 2.137? ──────────────────────────────
Vmin = pairing_V(M11, Yg, reverse=True)
W1p, W2p = W1_W2(Psi, H, Vmax.conj().T @ Yg @ Vmax, n)
W1m, W2m = W1_W2(Psi, H, Vmin.conj().T @ Yg @ Vmin, n)
c_pred = (W2p + W2m) / (W1p - W1m)
print(f"\n  (2) PREDICTING THE MEASURED SLOPE.")
print(f"      W_max ~ W1p*th + W2p*th^2   W1p={W1p:+.6f}  W2p={W2p:+.6f}")
print(f"      W_min ~ W1m*th + W2m*th^2   W1m={W1m:+.6f}  W2m={W2m:+.6f}")
print(f"      A(th) = (W_max+W_min)/(W_max-W_min) = c*th,")
print(f"              c = (W2p+W2m)/(W1p-W1m) = {c_pred:.4f}")
print(f"      MEASURED (exact_interval_asymmetry.log): 2.1369, 2.1372, 2.1384")
print(f"      -> {'AGREES' if abs(c_pred - 2.137) < 0.05 else 'DISAGREES'}"
      f"   (difference {abs(c_pred - 2.137):.4f})")

# ── (3) is the second-order term sign-definite? ──────────────────────────────
print(f"\n  (3) THE BARE GENERATOR IS DEAD AT EVERY ORDER, and it should be.")
rs = np.random.RandomState(17)
dc_bare = comm(Yg, comm(Yg, H))
print(f"      ||[Y,H]||         = {np.linalg.norm(comm(Yg, H)):.3e}")
print(f"      ||ad_Y^2(H)||     = {np.linalg.norm(dc_bare):.3e}")
print(f"      Heisenberg has SU(2) symmetry, so [sum_i X_i, H] = 0 EXACTLY and")
print(f"      Corollary 1 gives W = 0 to all orders for the unrotated generator.")
print(f"      Only the rotated Y~ = V^dag Y V is live, which is why every number")
print(f"      in (1), (2) and (4) is computed at a rotated generator.")

# single-qubit sanity check of the worked example in the docstring
print(f"\n      sanity, Y=X and H=Z on one qubit: [X,[X,Z]] = "
      f"{np.real(comm(X, comm(X, Z))[0, 0]):.0f} Z  (derivation says 4 Z)")

# ── (4) is it alignment-free? ────────────────────────────────────────────────
print(f"\n  (4) ALIGNMENT DEPENDENCE. Sweep V over the isospectral orbit and")
print(f"      compare how much each order moves.")
w1s, w2s = [], []
for t in range(200):
    Hh = rs.randn(d, d) + 1j * rs.randn(d, d)
    Hh = (Hh + Hh.conj().T) / 2
    V = expm(1j * Hh * 0.7)
    a, b = W1_W2(Psi, H, V.conj().T @ Yg @ V, n)
    w1s.append(a); w2s.append(b)
w1s, w2s = np.array(w1s), np.array(w2s)
print(f"      W_1 over V : mean {w1s.mean():+.5f}  sd {w1s.std():.5f}"
      f"  range [{w1s.min():+.5f}, {w1s.max():+.5f}]")
print(f"      W_2 over V : mean {w2s.mean():+.5f}  sd {w2s.std():.5f}"
      f"  range [{w2s.min():+.5f}, {w2s.max():+.5f}]")
rel1 = w1s.std() / max(abs(w1s.mean()), 1e-12)
rel2 = w2s.std() / max(abs(w2s.mean()), 1e-12)
print(f"      relative spread: W_1 {rel1:.2f},  W_2 {rel2:.2f}")
print(f"      W_1 changes SIGN over the orbit: {np.any(w1s > 0) and np.any(w1s < 0)}")
print(f"      W_2 changes SIGN over the orbit: {np.any(w2s > 0) and np.any(w2s < 0)}")

print()
print("  If W_2 keeps one sign while W_1 sweeps through zero, then a fixed")
print("  generator has a guaranteed direction at second order and none at first,")
print("  which is the engineering claim: second-order extraction needs no")
print("  knowledge of M_11 and therefore no knowledge of the joint state. If W_2")
print("  also changes sign, that claim fails and the second order buys nothing")
print("  a first-order protocol could not already get by choosing V.")

# ── (5) does it survive repeated cycles, which is the only thing that matters ─
print("\n  (5) ACCUMULATION OVER CYCLES — the engineering question.")
print("      A blind protocol repeats a FIXED rotated generator. The first-order")
print("      term has whatever sign the current alignment gives it and that")
print("      alignment drifts as the state evolves; the second-order term is")
print("      sign-definite. So does <H> fall monotonically, and does the total")
print("      track n*W_2*theta^2? Compared against a matched RANDOM-generator arm.")
P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)
Utau = expm(-1j * H * tau)


def cycles(Yr, theta, ncyc, seedstate):
    """Repeat sense+feedback with a FIXED generator, tracing the ancilla out."""
    p = np.linalg.qr(np.random.RandomState(seedstate).randn(d, d))[0][:, 0]
    rho_S = np.outer(p, p.conj())
    Us = np.kron(P0, np.eye(d)) + np.kron(P1, Utau)
    Ufb = np.kron(P0, np.eye(d)) + np.kron(P1, expm(-1j * (theta / 2) * Yr))
    plus = 0.5 * np.ones((2, 2), dtype=complex)
    traj = [float(np.real(np.trace(rho_S @ H)))]
    for c in range(ncyc):
        r = Us @ np.kron(plus, rho_S) @ Us.conj().T
        r = Ufb @ r @ Ufb.conj().T
        rho_S = r[:d, :d] + r[d:, d:]
        traj.append(float(np.real(np.trace(rho_S @ H))))
    return np.array(traj)


print("      ONE random V proves nothing, so 24 are drawn per theta and the")
print("      FRACTION THAT COOL is reported. The crossover prediction is that a")
print("      blind generator is directed only once theta exceeds |W_1|/W_2.")
NC, NV = 12, 24
Yal = Vmax.conj().T @ Yg @ Vmax
w1a, w2a = W1_W2(Psi, H, Yal, n)
print(f"\n      aligned generator: W_1 {w1a:+.4f}, W_2 {w2a:+.4f},"
      f"  |W_1|/W_2 = {abs(w1a) / w2a:.3f}")

Vs = []
for t in range(NV):
    Hh = rs.randn(d, d) + 1j * rs.randn(d, d)
    Hh = (Hh + Hh.conj().T) / 2
    Vs.append(expm(1j * Hh * 0.7))
w1r = np.array([W1_W2(Psi, H, V.conj().T @ Yg @ V, n)[0] for V in Vs])
w2r = np.array([W1_W2(Psi, H, V.conj().T @ Yg @ V, n)[1] for V in Vs])
xover = np.median(np.abs(w1r) / w2r)
print(f"      random generators: median |W_1|/W_2 = {xover:.3f}"
      f"   <- predicted crossover theta")

print(f"\n  {'theta':>8}{'aligned dE':>12}{'random dE mean':>16}"
      f"{'frac cooling':>14}{'frac monotone':>15}")
print("  " + "-" * 65)
for theta in (0.05, 0.1, 0.2, 0.3, 0.4, 0.6):
    dEa = cycles(Yal, theta, NC, 3)
    dEa = dEa[-1] - dEa[0]
    ds, ms = [], []
    for V in Vs:
        tr = cycles(V.conj().T @ Yg @ V, theta, NC, 3)
        ds.append(tr[-1] - tr[0])
        ms.append(bool(np.all(np.diff(tr) <= 1e-9)))
    ds = np.array(ds)
    print(f"  {theta:>8.2f}{dEa:>12.5f}{ds.mean():>16.5f}"
          f"{np.mean(ds < 0):>14.1%}{np.mean(ms):>15.1%}")

print()
print("  dE NEGATIVE means cooling. 'frac cooling' rising from ~50% to 100% as")
print("  theta crosses |W_1|/W_2 is the signature of the second-order term taking")
print("  over: below it a blind generator is a coin flip, above it the direction")
print("  is guaranteed without knowing M_11. That would make theta a DESIGN")
print("  VARIABLE rather than a small parameter, and it would say the protocol")
print("  should be run hard rather than gently - the opposite of the usual")
print("  weak-kick instinct. If frac cooling stays near 50% at every theta, the")
print("  second order is not usable blind and only alignment buys direction.")
