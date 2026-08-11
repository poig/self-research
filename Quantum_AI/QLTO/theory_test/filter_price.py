"""The fix, built and priced. Does the filter actually cool, and what does it cost?

The manuscript proves no unitary conditional kick has a preferred direction, shows
that breaking both symmetry conditions buys |D| <= 0.0147, and then points at the
frequency-filtered jump operators of Ding-Chen-Lin as the repair. It never builds
one. A no-go that names its own escape route and does not test it is asking the
reader to take the escape on faith.

WHAT THE FILTER ACTUALLY IS, and it matters more than the manuscript says:

    K = int f(s) e^{iHs} A e^{-iHs} ds ,      fhat(omega) = 0 for omega >= 0

In the eigenbasis of H this is (K)_ij = fhat(lam_i - lam_j) A_ij, so K carries
amplitude only from higher to lower energy. THAT OPERATOR IS NOT HERMITIAN. The
filter does not satisfy the two conditions of Prop. twoconditions by some clever
choice of spectrum - it LEAVES THE CLASS those conditions describe. The no-go is
about Hermitian generators of conditional unitaries; a filtered jump operator is
not one.

If that reading is right it sharpens the paper's conclusion considerably. The
escape is not "pick a better generator", which the |D| <= 0.0147 sweep already
rules out. It is "stop using a Hermitian generator", which is a structural change
of protocol rather than a tuning choice.

WHAT IS MEASURED HERE
  (1) the filtered K is non-Hermitian, and lowers energy only: matrix elements
      vanish above the diagonal in the energy-ordered eigenbasis
  (2) collision dynamics with the filtered K cool monotonically toward the ground
      state, while the unfiltered Hermitian coupling does not
  (3) THE PRICE. Building K needs e^{iHs} at a range of s, i.e. Hamiltonian
      simulation as a subroutine. Counted here as the number of distinct
      evolution times a discretised filter needs, which is the quantity that
      would be multiplied by a Trotter cost on hardware.

The price is the interesting part. The protocol exists because phase estimation
was too deep to run; if the only repair needs Hamiltonian simulation inside every
cycle, then the fix costs the thing the protocol was avoiding, and that is worth
saying plainly.
"""

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
# CONVENTION, and getting it backwards makes the protocol heat instead of cool.
# For cooling the SYSTEM must lose energy to the ancilla, so the lowering
# operator K is paired with the ancilla RAISING operator |1><0|.
RAISE = np.array([[0, 0], [1, 0]], dtype=complex)   # |1><0| , excites the ancilla
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|

N = 3
D = 2 ** N


def op(P, i, n=N):
    m = np.array([[1.0 + 0j]])
    for q in range(n):
        m = np.kron(m, P if q == i else I2)
    return m


def build_H(seed=5):
    r = np.random.default_rng(seed)
    H = np.zeros((D, D), dtype=complex)
    for i in range(N):
        for j in range(i + 1, N):
            H = H + r.uniform(-1, 1) * (op(Z, i) @ op(Z, j))
    for i in range(N):
        H = H + r.uniform(-.5, .5) * op(X, i)
    return H


H = build_H()
lam, V = np.linalg.eigh(H)
A = sum(op(X, i) for i in range(N))               # coupling operator


MU, SIG = -1.6, 0.8          # filter centred on a NEGATIVE frequency: lowering


def filtered_K(n_times=None, T=14.0):
    """K = int f(s) e^{iHs} A e^{-iHs} ds, i.e. K_ij = fhat(lam_i-lam_j) A_ij.

    A GAUSSIAN filter is used rather than an ideal step, because the transform
    pair is then exact and the discretisation error is the only error:

        f(s)    = exp(-SIG^2 s^2 / 2) exp(-i MU s)
        fhat(w) = int f(s) e^{iws} ds  ~  exp(-(w-MU)^2 / (2 SIG^2))

    With MU < 0 the weight sits on lam_i - lam_j < 0, so K carries amplitude
    downward in energy only. n_times=None builds fhat directly; otherwise the
    TIME INTEGRAL is discretised over n_times samples, which is what an
    implementation pays for - each sample is one Hamiltonian evolution."""
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
        F = F * (SIG / np.sqrt(2 * np.pi))          # normalise to the same peak
    return V @ (F * Ae) @ V.conj().T


print("=" * 96)
print("(1)  THE FILTERED OPERATOR IS NOT HERMITIAN, AND ONLY LOWERS ENERGY")
print("=" * 96)
K = filtered_K()
Ke = V.conj().T @ K @ V
# eigh returns lam ASCENDING, so i<j means lam_i<lam_j: the strict UPPER triangle
# is the energy-LOWERING part and the lower triangle is the raising part.
lowering = np.max(np.abs(np.triu(Ke, 1)))
raising = np.max(np.abs(np.tril(Ke, -1)))
print(f"  ||K - K^dag||_F / ||K||_F      : {np.linalg.norm(K - K.conj().T) / np.linalg.norm(K):.4f}")
print(f"  ||A - A^dag||_F                : {np.linalg.norm(A - A.conj().T):.2e}  (the unfiltered coupling IS Hermitian)")
print(f"  max |K_ij| lowering (lam_i<lam_j): {lowering:.4f}")
print(f"  max |K_ij| raising  (lam_i>lam_j): {raising:.2e}   (suppressed by the filter)")
print()
print("  A non-Hermitian generator is OUTSIDE the class Prop. twoconditions")
print("  describes. The filter does not satisfy the conditions - it leaves them.")

print()
print("=" * 96)
print("(2)  DOES IT COOL?  collision dynamics, ancilla reset each cycle")
print("=" * 96)


def collide(rho, Kop, theta, cycles):
    """One ancilla, joint unitary exp(-i theta (K (x) sigma+ + K^dag (x) sigma-)),
    trace out and reset. This is the standard single-ancilla Lindblad simulation."""
    Gj = np.kron(Kop, RAISE) + np.kron(Kop.conj().T, LOWER)
    U = expm(-1j * theta * Gj)
    out = []
    for _ in range(cycles):
        anc = np.zeros((2, 2), dtype=complex)
        anc[0, 0] = 1.0
        r = np.kron(rho, anc)
        r = U @ r @ U.conj().T
        rho = r.reshape(D, 2, D, 2)[:, 0, :, 0] + r.reshape(D, 2, D, 2)[:, 1, :, 1]
        rho = rho / np.trace(rho)
        out.append(float(np.real(np.trace(rho @ H))))
    return out


psi = np.ones(D) / np.sqrt(D)
rho0 = np.outer(psi, psi.conj())
E0 = float(np.real(np.trace(rho0 @ H)))
Egs = float(lam[0])
CYC, TH = 40, 0.3

ef = collide(rho0.copy(), K, TH, CYC)
eu = collide(rho0.copy(), A, TH, CYC)

print(f"  E(start) = {E0:+.5f}   E(ground) = {Egs:+.5f}   theta = {TH}, {CYC} cycles")
print()
print(f"  {'cycle':>7}{'filtered K':>14}{'unfiltered A':>15}")
print("  " + "-" * 36)
for c in [0, 4, 9, 19, 29, 39]:
    print(f"  {c + 1:>7}{ef[c]:>14.5f}{eu[c]:>15.5f}")
mono_f = all(ef[i + 1] <= ef[i] + 1e-12 for i in range(len(ef) - 1))
mono_u = all(eu[i + 1] <= eu[i] + 1e-12 for i in range(len(eu) - 1))
print()
print(f"  filtered   : final {ef[-1]:+.5f}, {100 * (E0 - ef[-1]) / (E0 - Egs):5.1f}% of the way to"
      f" the ground state, monotone {mono_f}")
print(f"  unfiltered : final {eu[-1]:+.5f}, {100 * (E0 - eu[-1]) / (E0 - Egs):5.1f}% of the way to"
      f" the ground state, monotone {mono_u}")

print()
print("=" * 96)
print("(3)  THE PRICE:  how much Hamiltonian simulation does the filter need?")
print("=" * 96)
print("  K is built from e^{iHs} at a range of s. A discretised filter uses")
print("  n_times distinct evolution times, EVERY CYCLE. Fidelity of the")
print("  discretised K against the ideal one, and the cooling it achieves:")
print()
print(f"  {'n_times':>9}{'rel err vs ideal':>18}{'final E':>12}{'% to ground':>13}")
print("  " + "-" * 52)
for nt in (8, 16, 32, 64, 128):
    Kd = filtered_K(n_times=nt)
    Kd = Kd / (np.linalg.norm(Kd) + 1e-15) * np.linalg.norm(K)
    rel = float(np.linalg.norm(Kd - K) / np.linalg.norm(K))
    e = collide(rho0.copy(), Kd, TH, CYC)
    print(f"  {nt:>9}{rel:>18.4f}{e[-1]:>12.5f}"
          f"{100 * (E0 - e[-1]) / (E0 - Egs):>12.1f}%")

print()
print("  CAVEAT ON READING THE COUNT. This filter is a Gaussian BANDPASS centred on a")
print("  single negative frequency, not the LOW-PASS of Ding-Chen-Lin, which has")
print("  support across the whole negative spectrum. A narrow window needs few samples")
print("  BECAUSE it is narrow, and for the same reason it drives fewer of the available")
print("  transitions as the spectrum widens with N (scaling_verdict.py: cycles to 80%")
print("  of the way to the ground state go 9, 24, >400, >400 for N=3..6). The low cost")
print("  and the failure to converge are the SAME FACT. This count is the cost of a")
print("  filter at one size, not the cost of one that works at scale.")
print()
print("  Each of those n_times entries is a distinct Hamiltonian evolution, needed")
print("  once per cycle. The protocol exists because phase estimation was too deep")
print("  to run; if the repair puts Hamiltonian simulation back inside every cycle,")
print("  the fix costs the very thing the protocol was avoiding. That is the honest")
print("  closing statement, and it is stronger than 'use a filter'.")
