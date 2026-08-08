"""Does the (A)/(B) classification explain QLTO's walk? Tested, not assumed.

The paper's Proposition (two sufficient conditions) says the first-order
achievable interval is symmetric about zero if EITHER spec(Y) is symmetric (A) or
spec(M_11) is symmetric (B), and Corollary (pure branch) says a pure |1> branch
forces (B) for every generator. Sec. 'Two independent causes' then reads
Ref. ding2024single as breaking both at once: the filter breaks (A), and "the
reset that makes the scheme Lindbladian rather than unitary supplies the
mixedness that breaks (B)".

QLTO's walk looks like an instance of the SAME structure, arrived at independently
and for a different purpose:

  * one ancilla is prepared in |+> and controls ALL k walk steps, so the walk
    unitary is U = P_0 (x) I + P_1 (x) prod_s V_s - exactly the paper's class
    K = P_1 (x) Y with Y_eff = i log prod_s V_s.
  * the product factorises over param qubits and every SU(2) element has
    generator spectrum +-theta/2, so spec(Y_eff) should be SYMMETRIC => (A).
  * everything is unitary from a pure input, so the |1> branch is PURE => (B) by
    the pure-branch corollary.

If that holds, the shipped walk provably has a symmetric reachable interval and
therefore no intrinsic direction, which would explain - as a theorem rather than
as tuning - why its measured transfer function is erratic and only weakly
anti-correlated with its own input (corr = -0.376, 8 sign turns, and the wrong
sign at g = -1.0; supplement/results/v37_ancilla_reset.log). Resetting the walk
ancilla every step makes the branch mixed, which should break (B), and the same
log measures corr going to -0.857 with 2 turns.

THE OBVIOUS WAY THIS FAILS, and the reason it has to be run: the paper also
reports that under REAL Hamiltonians spec M_11 stays symmetric across repeated
cycles, collapsing below 1e-15 whenever rho and H are simultaneously real. The
Heisenberg Hamiltonian QLTO is benchmarked on IS real. So the reset may NOT break
(B) here, in which case the walk improvement has some other cause and the paper
connection must not be written. That is the point of this file.

REPORTED, per variant and per cycle count
  purity        Tr(rho_1^2) of the |1> branch - 1.0 means (B) holds by corollary
  asym(M_11)    ||lam + reverse(lam)|| / ||lam||, the paper's spectral asymmetry
  asym(Y)       same for the walk's effective generator - condition (A)
  interval      [W_min, W_max] from von Neumann pairing, and its own asymmetry
"""
import numpy as np
from scipy.linalg import expm, logm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def op(pauli, i, n):
    m = np.ones((1, 1), dtype=complex)
    for q in range(n):
        m = np.kron(m, pauli if q == i else I2)
    return m


def heisenberg(n, dm=0.0):
    """Real Heisenberg; dm>0 adds a Dzyaloshinskii-Moriya X_i Y_j term, which is
    the paper's own way of making H complex."""
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        for P in (X, Y, Z):
            H = H + op(P, i, n) @ op(P, i + 1, n)
        if dm:
            H = H + dm * (op(X, i, n) @ op(Y, i + 1, n))
    return H


def spectral_asymmetry(M):
    lam = np.sort(np.real(np.linalg.eigvalsh(M)))
    nrm = np.linalg.norm(lam)
    if nrm < 1e-14:
        return 0.0
    return float(np.linalg.norm(lam + lam[::-1]) / nrm)


def branch_block(rho, n):
    """The |1><1| ancilla block, as a system operator. Ancilla is factor 0."""
    d = 2 ** n
    return rho[d:, d:]


def m11(rho, H, n):
    d = 2 ** n
    A = np.kron(np.eye(2), H)
    C = 1j * (rho @ A - A @ rho)
    return C[d:, d:]


def interval(M, Yg, theta=0.2):
    """von Neumann pairing endpoints of Eq. (interval)."""
    a = np.sort(np.real(np.linalg.eigvalsh(M)))[::-1]     # decreasing
    b = np.sort(np.real(np.linalg.eigvalsh(Yg)))
    lo = float(np.sum(a * b)) * theta / 2                  # opposite order
    hi = float(np.sum(a * b[::-1])) * theta / 2            # same order
    return lo, hi


def walk_generator(n, k, dt, R, g):
    """The walk's effective system generator Y_eff = i log prod_s V_s.

    Built exactly as nisq_v3._execute_walk builds it, so spec(Y_eff) is the
    condition-(A) object for the SHIPPED circuit rather than for a stand-in.
    """
    gain = 1.0 / np.sqrt(R)
    U = np.eye(2 ** n, dtype=complex)
    for step in range(k):
        s = (step + 0.5) / k
        be = (1.0 - s) * np.pi * dt
        V = np.ones((1, 1), dtype=complex)
        for i in range(n):
            al = g[i] * (s * np.pi * dt) * 0.5 * np.pi * gain
            th = float(np.hypot(al, be))
            if th < 1e-15:
                V = np.kron(V, I2)
                continue
            nz, nx = al / th, be / th
            V = np.kron(V, np.cos(th / 2) * I2
                        - 1j * np.sin(th / 2) * (nz * Z + nx * X))
        U = V @ U
    return np.real_if_close(1j * logm(U)), U


def cycles(n, H, ncyc, tau, theta, Yg, reset):
    """Run ncyc sense+feedback cycles, sharing or resetting the ancilla."""
    d = 2 ** n
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    psi = np.linalg.qr(np.random.RandomState(3).randn(d, d))[0] @ psi
    rho_S = np.outer(psi, psi.conj())

    Utau = expm(-1j * H * tau)
    P0 = np.diag([1.0, 0.0]).astype(complex)
    P1 = np.diag([0.0, 1.0]).astype(complex)
    Ufb = np.kron(P0, np.eye(d)) + np.kron(P1, expm(-1j * (theta / 2) * Yg))

    plus = 0.5 * np.ones((2, 2), dtype=complex)
    Us = np.kron(P0, np.eye(d)) + np.kron(P1, Utau)      # controlled sensing
    out = []

    if not reset:
        # SHARED ANCILLA, as shipped: sense ONCE, then every feedback step is
        # controlled on that same never-reset ancilla. The joint state is carried
        # forward untouched, so it stays PURE and the k steps compose coherently.
        rho = Us @ np.kron(plus, rho_S) @ Us.conj().T
        for c in range(ncyc):
            b = branch_block(rho, n)
            tr = float(np.real(np.trace(b)))
            pur = float(np.real(np.trace(b @ b))) / max(tr ** 2, 1e-18)
            M = m11(rho, H, n)
            lo, hi = interval(M, Yg, theta)
            out.append((c + 1, pur, spectral_asymmetry(M), lo, hi))
            rho = Ufb @ rho @ Ufb.conj().T
        return out

    # RESET: fresh |+> ancilla every step, so each step re-senses and the ancilla
    # is traced out afterwards. That trace-out is what makes rho_S MIXED and is
    # the only difference from the arm above.
    for c in range(ncyc):
        rho = Us @ np.kron(plus, rho_S) @ Us.conj().T
        b = branch_block(rho, n)
        tr = float(np.real(np.trace(b)))
        pur = float(np.real(np.trace(b @ b))) / max(tr ** 2, 1e-18)
        M = m11(rho, H, n)
        lo, hi = interval(M, Yg, theta)
        out.append((c + 1, pur, spectral_asymmetry(M), lo, hi))
        rho = Ufb @ rho @ Ufb.conj().T
        rho_S = rho[:d, :d] + rho[d:, d:]                # trace out the ancilla
    return out


n, tau, theta, R, dt, k = 3, 0.35, 0.2, 0.6, 0.5, 15
rng = np.random.RandomState(11)
g = rng.uniform(-1, 1, n)

print("=" * 94)
print("DOES THE (A)/(B) CLASSIFICATION EXPLAIN QLTO's WALK?")
print("=" * 94)

print("\n  (1) CONDITION (A) — is the walk's own generator spectrally symmetric?")
Yeff, Uw = walk_generator(n, k, dt, R, g)
Ysum = sum(op(X, i, n) for i in range(n))
print(f"      spec asymmetry of Y_eff (walk, k={k})   : "
      f"{spectral_asymmetry(Yeff):.3e}")
print(f"      spec asymmetry of sum_i X_i (paper's Y) : "
      f"{spectral_asymmetry(Ysum):.3e}")
print(f"      => (A) {'HOLDS' if spectral_asymmetry(Yeff) < 1e-9 else 'FAILS'} "
      f"for the shipped walk generator")

print("\n  (2) CONDITION (B) — purity and spec(M_11), shared vs reset ancilla.")
for label, Hm in (("Heisenberg (REAL)", heisenberg(n)),
                  ("Heisenberg + DM (COMPLEX)", heisenberg(n, dm=0.7))):
    print(f"\n      {label}")
    print(f"      {'variant':>9}{'cyc':>5}{'purity':>10}{'asym(M11)':>12}"
          f"{'W_min':>10}{'W_max':>10}{'asym(W)':>10}")
    print("      " + "-" * 66)
    for reset in (False, True):
        for c, pur, am, lo, hi in cycles(n, Hm, 4, tau, theta, Ysum, reset):
            aw = abs(hi + lo) / max(hi - lo, 1e-18)
            print(f"      {'reset' if reset else 'shared':>9}{c:>5}{pur:>10.4f}"
                  f"{am:>12.3e}{lo:>10.5f}{hi:>10.5f}{aw:>10.3e}")
        print("      " + "." * 66)

print("\n  (3) VERDICT INPUTS")
print("      (A) holding for the walk generator means NO choice of state gives an")
print("      asymmetric interval, and (B) holding means no choice of generator")
print("      does. Both must fail together. If the reset rows show purity < 1 but")
print("      asym(M11) still ~1e-15 under the REAL Hamiltonian, then mixedness")
print("      alone does NOT break (B) here, the paper's real-H observation covers")
print("      this case, and QLTO's walk improvement needs a different explanation")
print("      - in which case do NOT write the connection into the paper.")
