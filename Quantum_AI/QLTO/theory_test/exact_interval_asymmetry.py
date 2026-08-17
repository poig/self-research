"""Is the SYMMETRY of the achievable interval an artefact of first order?

The paper's negative result - a symmetric interval means no directed work, so
cooling needs (A) and (B) to fail together - rests on Theorem 2, which bounds the
FIRST-ORDER term only. The Scope paragraph already concedes the rest:

    "Theorem 2 bounds the first-order term only. The exact work at the optimising
     V falls short of the first-order endpoint --- by ~10% at theta = 0.2, n = 4
     --- and the O(theta^2) term breaks the exact symmetry of Eq. (interval)."

So the symmetry that the whole classification turns on is known to fail at second
order, and nobody has measured by how much. Two possibilities, with opposite
consequences for the paper:

  the asymmetry is NEGLIGIBLE at usable theta
      the first-order picture is the operational one, the classification stands
      as written, and this file just quantifies a caveat.

  the asymmetry GROWS FAST ENOUGH
      then directed work is available at finite theta with NO symmetry breaking
      at all - neither (A) nor (B) - and "break both conditions" is advice about
      a measure-zero limit. |W| grows like theta while the asymmetry grows like
      theta^2, so their ratio grows linearly: there would be an operating point.

MEASURED HERE. For each theta, the EXACT interval

    W(theta, V) = <Psi_1| A - U^dag(theta,V) A U(theta,V) |Psi_1>

is maximised and minimised over system-local V by direct optimisation over U(d)
(parametrised as expm(i*Hermitian), with random restarts), rather than by the
von Neumann pairing, which is only valid for the linear term. Reported against
the first-order prediction:

    W_max, W_min      exact endpoints found
    asym              (W_max + W_min) / (W_max - W_min), zero iff symmetric
    fo_endpoint       theta/2 * sum lam_down(M11) lam_down(Y), the Theorem-2 value
    shortfall         how far the exact optimum falls short of it

The optimisation returns a LOWER bound on the true interval, so any asymmetry it
finds is real; only its size could be understated.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def op(P, i, n):
    m = np.ones((1, 1), dtype=complex)
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def heisenberg(n, dm=0.0):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        for P in (X, Y, Z):
            H = H + op(P, i, n) @ op(P, i + 1, n)
        if dm:
            H = H + dm * (op(X, i, n) @ op(Y, i + 1, n))
    return H


def herm_from(v, d):
    """Hermitian matrix from d^2 real parameters."""
    A = np.zeros((d, d), dtype=complex)
    idx = 0
    for i in range(d):
        A[i, i] = v[idx]; idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            A[i, j] = v[idx] + 1j * v[idx + 1]
            A[j, i] = v[idx] - 1j * v[idx + 1]
            idx += 2
    return A


def build_state(n, H, tau, seed=3):
    """Post-sensing joint state |Psi_1>, pure, as in the protocol."""
    d = 2 ** n
    psi = np.linalg.qr(np.random.RandomState(seed).randn(d, d))[0][:, 0]
    Utau = expm(-1j * H * tau)
    # (|0>|psi> + |1> e^{-iH tau}|psi>)/sqrt2
    return np.concatenate([psi, Utau @ psi]) / np.sqrt(2.0)


def work(Psi, H, Yg, theta, V, n):
    """Exact W for feedback generator V^dag Yg V, all orders in theta."""
    d = 2 ** n
    Yr = V.conj().T @ Yg @ V
    A = np.kron(np.eye(2), H)
    K = np.kron(np.diag([0.0, 1.0]).astype(complex), Yr)
    U = expm(-1j * (theta / 2) * K)
    return float(np.real(Psi.conj() @ (A - U.conj().T @ A @ U) @ Psi))


def extremise(Psi, H, Yg, theta, n, sign, restarts=4, seed=0):
    """max (sign=-1) or min (sign=+1) of W over system-local V."""
    d = 2 ** n
    best = None
    rng = np.random.RandomState(seed)
    for r in range(restarts):
        v0 = rng.randn(d * d) * 0.6
        res = minimize(lambda v: sign * work(Psi, H, Yg, theta,
                                             expm(1j * herm_from(v, d)), n),
                       v0, method='BFGS',
                       options={'maxiter': 300, 'gtol': 1e-10})
        val = sign * res.fun
        if best is None or (sign < 0 and val > best) or (sign > 0 and val < best):
            best = val
    return best


def first_order_endpoint(Psi, H, Yg, theta, n):
    d = 2 ** n
    rho = np.outer(Psi, Psi.conj())
    A = np.kron(np.eye(2), H)
    M = (1j * (rho @ A - A @ rho))[d:, d:]
    a = np.sort(np.real(np.linalg.eigvalsh(M)))[::-1]
    b = np.sort(np.real(np.linalg.eigvalsh(Yg)))[::-1]
    return float(np.sum(a * b)) * theta / 2


n, tau = 3, 0.35
H = heisenberg(n)
Yg = sum(op(X, i, n) for i in range(n))
Psi = build_state(n, H, tau)

# THE SATURATION BOUND, without which this whole file is misleading.
# W = <A> - <U^dag A U>, and U^dag A U ranges over the isospectral orbit of A, so
#     W_floor = <A> - lam_max(A)      W_ceil = <A> - lam_min(A)
# and once an endpoint reaches its bound the "achievable interval" has stopped
# measuring the protocol and is just reporting the spectral range of H. A first
# run of this file swept theta to 2.0 and found asymmetry rising to 0.93 with a
# directed part 25000x the paper's - all of it saturation: both endpoints were
# pinned and their span was exactly lam_max - lam_min. Only the theta where
# NEITHER endpoint is near its bound says anything about the O(theta^2) term.
# and the bound is HALF the spectral range, not the whole of it. Only the |1>
# branch is rotated and it carries weight 1/2:
#     |Psi_1> = (|0>|psi> + |1>|phi>)/sqrt2,  U = P_0 (x) I + P_1 (x) V
#     W = 1/2 [ <phi|H|phi> - <phi|V^dag H V|phi> ]
# so W ranges over 1/2 [<phi|H|phi> - lam_max, <phi|H|phi> - lam_min]. A first
# version of this bound omitted the 1/2 and reported saturation at half its true
# value, which made theta=0.05 look clean at 20% when it is 40%. The pinned
# endpoints of the original sweep, -0.10275 and +2.89725 with span exactly 3.000
# against a spectral range of 6.000, are the check.
# Note <phi|H|phi> = <psi|H|psi> because [U_tau, H] = 0, so the centre is <A>.
Amat = np.kron(np.eye(2), H)
Aexp = float(np.real(Psi.conj() @ Amat @ Psi))
lam = np.real(np.linalg.eigvalsh(H))
W_FLOOR, W_CEIL = (Aexp - lam.max()) / 2, (Aexp - lam.min()) / 2

print("=" * 96)
print("EXACT INTERVAL vs THE FIRST-ORDER ONE — is the symmetry an artefact of O(theta)?")
print("=" * 96)
print(f"  n={n}, tau={tau}, real Heisenberg, Y = sum_i X_i (so (A) holds),")
print(f"  pure branch (so (B) holds). Theorem 2 therefore predicts a SYMMETRIC")
print(f"  interval at first order. Exact endpoints found by optimisation over U(8).")
print()
print(f"  spectral bounds: W_floor {W_FLOOR:.5f}, W_ceil {W_CEIL:.5f}"
      f"  (span {W_CEIL - W_FLOOR:.5f} = lam_max - lam_min)")
print("  'sat' columns are how far each endpoint has run into its bound. Any row")
print("  with sat above ~20% is reporting the spectrum, not the protocol.")
print()
print(f"  {'theta':>8}{'W_min':>11}{'W_max':>11}{'sat-':>7}{'sat+':>7}"
      f"{'asym':>10}{'fo_endpt':>11}{'asym/theta':>11}")
print("  " + "-" * 76)

rows = []
for theta in (0.0025, 0.005, 0.01, 0.02, 0.04):
    hi = extremise(Psi, H, Yg, theta, n, -1, restarts=8, seed=1)
    lo = extremise(Psi, H, Yg, theta, n, +1, restarts=8, seed=2)
    span = hi - lo
    asym = (hi + lo) / span if span > 1e-14 else 0.0
    fo = first_order_endpoint(Psi, H, Yg, theta, n)
    sm, sp = lo / W_FLOOR, hi / W_CEIL
    rows.append((theta, lo, hi, asym, fo, max(sm, sp)))
    print(f"  {theta:>8.4f}{lo:>11.5f}{hi:>11.5f}{sm:>7.1%}{sp:>7.1%}"
          f"{asym:>10.4f}{fo:>11.5f}{asym / theta:>11.4f}", flush=True)

print()
print("  (2) HOW BIG IS THE DIRECTED PART, and does it scale as O(theta^2)?")
print("      NOT compared against the paper's both-conditions-broken interval:")
print("      that uses a rank-2 projector generator whose spectral range differs")
print("      from sum_i X_i, so the magnitudes are not commensurable. An earlier")
print("      version of this file made that comparison and reported multipliers")
print("      up to 25000x. Those were saturation against an incommensurable")
print("      baseline and are withdrawn. What IS comparable is the SCALING.")
print(f"  {'theta':>8}{'directed |W_max+W_min|':>24}{'asym/theta':>12}{'usable?':>11}")
print("  " + "-" * 55)
for theta, lo, hi, asym, fo, sat in rows:
    ok = "yes" if sat < 0.20 else "SATURATED"
    print(f"  {theta:>9.4f}{abs(hi + lo):>23.6f}{asym / theta:>12.4f}{ok:>11}")

clean = [r for r in rows if r[5] < 0.20]
if len(clean) >= 2:
    s = [r[3] / r[0] for r in clean]
    print(f"\n  asym/theta over the UNSATURATED rows: "
          f"{', '.join(f'{v:.4f}' for v in s)}")
    print(f"  spread {max(s) - min(s):.4f} over a "
          f"{clean[-1][0] / clean[0][0]:.0f}x range of theta.")
    print("  A constant here is the signature of a genuine O(theta^2) directed")
    print("  term: asym = (W_2/W_1) theta exactly. Saturation or an optimiser")
    print("  artefact would not hold a constant across that range.")

print()
print("  If asym stays ~0 across theta, the first-order picture is operational and")
print("  the classification stands. If asym grows and the directed part exceeds")
print("  the paper's own both-conditions-broken result, then finite theta supplies")
print("  directed work WITHOUT breaking (A) or (B), and Sec. 'Two independent")
print("  causes' needs a scope line saying so - the conditions govern the")
print("  linearised protocol, not the protocol.")
