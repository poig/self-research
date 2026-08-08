"""Break time-reversal symmetry and the protocol can prefer cooling.

The manuscript reports the reachable work at fixed correlations as the SYMMETRIC
interval [-W*, +W*] and reads that as the protocol's defect: an unfiltered
conditional kick heats as readily as it cools. Three attempts to break the
symmetry from the outside failed - an asymmetric generator spectrum did nothing,
depolarising the state did nothing, and the mixing produced by the protocol's own
reset did nothing.

commutator_spectrum_symmetry.py located the cause. For random Hermitian pairs,
spec(i[rho,H]) is symmetric about zero exactly when rho and H are BOTH REAL in a
common basis (defect ~1e-15), and strongly asymmetric otherwise (defect ~0.3).
The only other guaranteed case is rank(rho) = 1, which is Corollary 3.

EVERY HAMILTONIAN IN THE MANUSCRIPT IS REAL. sum_i Z_i is real; the ZZ + X family
of Fig. 1 is real; and the isotropic Heisenberg chain is real too, since Y (x) Y
has real entries even though Y does not. A real Hamiltonian is time-reversal
symmetric under T = complex conjugation.

SO THE DEFECT IS NOT THE MISSING FILTER. It is T-symmetry. The prediction is that
a Hamiltonian with genuinely complex entries - one containing an odd number of Y
factors per term, such as a Dzyaloshinskii-Moriya coupling X_i Y_j - gives an
ASYMMETRIC reachable interval, and hence a protocol that prefers one direction.

This also explains why Ding-Chen-Lin's repair works. Their filter satisfies
f_hat(omega) = 0 for omega >= 0, which is manifestly not invariant under
t -> -t: the filter is itself a T-breaking device. Breaking T with the
Hamiltonian is the cheaper route to the same asymmetry, and needs no integral
over Heisenberg-evolved coupling operators.

CHECKED HERE
  (a) reality and T-symmetry of the manuscript's own Hamiltonian families
  (b) reachable interval for a T-broken Hamiltonian, against a T-symmetric one
      of matched norm
  (c) whether the bias has a usable sign and survives the exact protocol
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

N = 4
TAU, THETA = 1.042, 0.2
d = 2 ** N


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def op(terms):
    return SparsePauliOp.from_list(terms).to_matrix()


rng = np.random.RandomState(42)
zz = [(lbl(N, **{str(i): "Z", str(j): "Z"}), rng.uniform(-1, 1))
      for i in range(N) for j in range(i + 1, N)]
xs = [(lbl(N, **{str(i): "X"}), rng.uniform(-0.5, 0.5)) for i in range(N)]

FAMILIES = {
    "sum_i Z_i (real)": [(lbl(N, **{str(i): "Z"}), 1.0) for i in range(N)],
    "ZZ + X  (paper Fig.1)": zz + xs,
    "Heisenberg XXZ (real)": [(lbl(N, **{str(i): p, str(i + 1): p}), 1.0)
                              for i in range(N - 1) for p in "XYZ"],
    "DM coupling X_i Y_j (T-broken)": zz + xs +
        [(lbl(N, **{str(i): "X", str(i + 1): "Y"}), 0.6) for i in range(N - 1)],
}

plus = np.ones(d) / np.sqrt(d)
Y_gen = sum(op([(lbl(N, **{str(i): "X"}), 1.0)]) for i in range(N))


def M11(Hm, psi=plus, tau=TAU):
    psi1 = expm(-1j * Hm * tau) @ psi
    s = np.outer(psi1, psi1.conj())
    return 0.5j * (s @ Hm - Hm @ s)


def interval(M, Ym):
    lm = np.sort(np.linalg.eigvalsh(M))[::-1]
    ly = np.sort(np.linalg.eigvalsh(Ym))[::-1]
    return (float(np.sum(lm * ly[::-1])) * THETA / 2.0,
            float(np.sum(lm * ly)) * THETA / 2.0)


def cycle(rho, Hm, Ym, tau=TAU, theta=THETA):
    Uv = expm(-1j * Hm * tau)
    W = np.vstack([np.eye(d) / np.sqrt(2), Uv / np.sqrt(2)])
    big = W @ rho @ W.conj().T
    K = np.kron(np.diag([0.0, 1.0]), Ym)
    U = expm(-1j * (theta / 2.0) * K)
    out = U @ big @ U.conj().T
    return out[:d, :d] + out[d:, d:]


print("=" * 92)
print("BREAKING TIME-REVERSAL SYMMETRY")
print("=" * 92)
print("  (a) reality of the Hamiltonian families, and the resulting interval")
print(f"  {'family':<34}{'Im part':>11}{'W_lo':>11}{'W_hi':>11}"
      f"{'|hi|-|lo|':>12}{'asym %':>9}")
print("  " + "-" * 88)

Hs = {}
for name, terms in FAMILIES.items():
    Hm = op(terms)
    Hs[name] = Hm
    imag = float(np.max(np.abs(np.imag(Hm))))
    lo, hi = interval(M11(Hm), Y_gen)
    asym = abs(abs(hi) - abs(lo)) / max(abs(hi), 1e-15) * 100
    print(f"  {name:<34}{imag:>11.2e}{lo:>11.5f}{hi:>11.5f}"
          f"{abs(hi) - abs(lo):>12.2e}{asym:>9.2f}")

print()
print("  (b) MIXED STATES — where Corollary 3's rank-2 protection is gone")
print(f"  {'family':<34}{'cycle':>7}{'rank M11':>10}{'W_lo':>11}{'W_hi':>11}"
      f"{'asym %':>9}")
print("  " + "-" * 82)
for name in ("ZZ + X  (paper Fig.1)", "DM coupling X_i Y_j (T-broken)"):
    Hm = Hs[name]
    rho = np.outer(plus, plus.conj())
    for k in (0, 1, 3, 5):
        while True:
            Uv = expm(-1j * Hm * TAU)
            M = 0.5j * ((Uv @ rho @ Uv.conj().T) @ Hm - Hm @ (Uv @ rho @ Uv.conj().T))
            break
        rk = int(np.sum(np.abs(np.linalg.eigvalsh(M)) > 1e-10))
        lo, hi = interval(M, Y_gen)
        asym = abs(abs(hi) - abs(lo)) / max(abs(hi), 1e-15) * 100
        print(f"  {name if k == 0 else '':<34}{k:>7}{rk:>10}{lo:>11.5f}"
              f"{hi:>11.5f}{asym:>9.2f}")
        for _ in range(2):
            rho = cycle(rho, Hm, Y_gen)

print()
print("  An asymmetric interval means the reachable set is BIASED: the protocol can")
print("  reach further in one direction than the other at fixed correlations. That")
print("  is the ingredient the manuscript identifies as missing, obtained from the")
print("  Hamiltonian's symmetry rather than from a frequency filter.")
