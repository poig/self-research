"""What breaks the sign symmetry? A spectral criterion for cooling.

Theorem 2 gives the reachable work at fixed correlations as

    W in (theta/2)[ sum_k lam_dn(M11) lam_up(Y) , sum_k lam_dn(M11) lam_dn(Y) ]

and the interval collapses to [-W*, +W*] exactly when spec(Y) is symmetric about
zero, because the ascending list is then the negation of the descending one. So
the protocol's defect - heats as readily as it cools - is not about filtering per
se. It is a statement about the SPECTRUM OF THE FEEDBACK GENERATOR.

TWO CONSEQUENCES, and the second is the useful one.

1. SHIFTING DOES NOT HELP. Y -> Y + cI sends K -> K + c P_1, and since P_1
   commutes with K,
       U = e^{-i(theta/2)(K + c P_1)} = e^{-i(theta/2) c P_1} e^{-i(theta/2)K} .
   The extra factor is diagonal in the ancilla and commutes with A = I (x) H, so
   U^dag A U is unchanged and W is INVARIANT. Recentring the spectrum is not a
   design knob.

2. TRACELESS PAULI SUMS ARE ALL DISQUALIFIED. Any sum of non-identity Pauli
   strings is traceless with a spectrum symmetric under negation whenever the
   Paulis anticommute pairwise or pair up - and for the standard choices
   (sum_i X_i, single X_j, sum_i Z_i) the spectrum is literally {n-2k}, symmetric.
   So EVERY natural generator in this protocol class gives a symmetric interval,
   which is why the defect looked structural rather than like a bad choice.

WHAT HAS AN ASYMMETRIC SPECTRUM. A projector Pi onto a subspace of dimension d
has spectrum {0,1} with multiplicities (2^n - d, d). Its centred spectrum is
symmetric only when d = 2^n / 2. Away from half filling, the interval is
asymmetric - and a projector is exactly what a measurement-conditioned feedback
implements.

THIS FILE CHECKS
  (a) shift invariance: Y and Y + cI must give identical W
  (b) the interval for traceless Pauli generators is symmetric, as claimed
  (c) rank-r projectors give an ASYMMETRIC interval, maximally away from r = 2^{n-1}
  (d) whether the asymmetry has the useful sign - an interval biased toward
      negative W is a generator that cools more than it heats
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


def paper_H(n, seed=42):
    rng = np.random.RandomState(seed)
    ops = []
    for i in range(n):
        for j in range(i + 1, n):
            ops.append((lbl(n, **{str(i): "Z", str(j): "Z"}), rng.uniform(-1, 1)))
    for i in range(n):
        ops.append((lbl(n, **{str(i): "X"}), rng.uniform(-0.5, 0.5)))
    return SparsePauliOp.from_list(ops).to_matrix()


Hm = paper_H(N)
plus = np.ones(d) / np.sqrt(d)


def post_sensing(psi, tau=TAU):
    psi1 = expm(-1j * Hm * tau) @ psi
    P = np.zeros(2 * d, dtype=complex)
    P[:d] = psi / np.sqrt(2)
    P[d:] = psi1 / np.sqrt(2)
    return P


def work(Ym, psi=plus, theta=THETA):
    P = post_sensing(psi)
    A = np.kron(np.eye(2), Hm)
    K = np.kron(np.diag([0.0, 1.0]), Ym)
    U = expm(-1j * (theta / 2.0) * K)
    return float(np.real(P.conj() @ (A - U.conj().T @ A @ U) @ P))


def M11(psi=plus):
    """<1| i[rho, A] |1> = (i/2)[|psi_1><psi_1|, H]."""
    psi1 = expm(-1j * Hm * TAU) @ psi
    s = np.outer(psi1, psi1.conj())
    return 0.5j * (s @ Hm - Hm @ s)


def interval(Ym):
    """Theorem 2: extremes of Tr(M11 V^dag Y V) over the unitary orbit."""
    lm = np.sort(np.linalg.eigvalsh(M11()))[::-1]
    ly = np.sort(np.linalg.eigvalsh(Ym))[::-1]
    hi = float(np.sum(lm * ly)) * THETA / 2.0
    lo = float(np.sum(lm * ly[::-1])) * THETA / 2.0
    return lo, hi


print("=" * 90)
print("ASYMMETRIC GENERATORS — what actually breaks the sign degeneracy")
print("=" * 90)

print("\n  (a) SHIFT INVARIANCE — Y and Y + cI must agree")
Y0 = sum(SparsePauliOp(lbl(N, **{str(i): "X"})).to_matrix() for i in range(N))
print(f"  {'generator':<22}{'W':>13}")
print("  " + "-" * 36)
for c in (0.0, 1.0, 5.0):
    print(f"  {'sum X + ' + str(c) + 'I':<22}{work(Y0 + c * np.eye(d)):>13.8f}")

print("\n  (b) TRACELESS PAULI GENERATORS — spectrum and interval")
print(f"  {'generator':<24}{'spec min':>10}{'spec max':>10}{'W_lo':>11}{'W_hi':>11}"
      f"{'|hi|-|lo|':>12}")
print("  " + "-" * 78)
gens = [("sum_i X_i", Y0),
        ("X_0", SparsePauliOp(lbl(N, **{"0": "X"})).to_matrix()),
        ("Z_0 Z_1", SparsePauliOp(lbl(N, **{"0": "Z", "1": "Z"})).to_matrix()),
        ("sum_i Z_i", sum(SparsePauliOp(lbl(N, **{str(i): "Z"})).to_matrix()
                          for i in range(N)))]
for name, Ym in gens:
    ev = np.linalg.eigvalsh(Ym)
    lo, hi = interval(Ym)
    print(f"  {name:<24}{ev.min():>10.2f}{ev.max():>10.2f}{lo:>11.5f}{hi:>11.5f}"
          f"{abs(hi) - abs(lo):>12.2e}")

print("\n  (c) RANK-r PROJECTORS — asymmetric away from half filling")
print(f"  {'generator':<24}{'rank':>6}{'W_lo':>11}{'W_hi':>11}"
      f"{'|hi|-|lo|':>12}{'bias':>9}")
print("  " + "-" * 73)
evals, evecs = np.linalg.eigh(Hm)          # project onto low-energy subspaces
for r in (1, 2, 4, 8, 12, 15):
    V = evecs[:, :r]
    Pi = V @ V.conj().T
    lo, hi = interval(Pi)
    bias = "cools" if abs(lo) > abs(hi) else "heats"
    print(f"  {'Pi_low, r=' + str(r):<24}{r:>6}{lo:>11.5f}{hi:>11.5f}"
          f"{abs(hi) - abs(lo):>12.5f}{bias:>9}")

print("\n  (d) DOES THE ASYMMETRY SURVIVE THE EXACT PROTOCOL?")
print(f"  {'generator':<24}{'1st-order lo':>14}{'1st-order hi':>14}{'exact W':>12}")
print("  " + "-" * 66)
for r in (1, 4, 8, 15):
    V = evecs[:, :r]
    Pi = V @ V.conj().T
    lo, hi = interval(Pi)
    print(f"  {'Pi_low, r=' + str(r):<24}{lo:>14.5f}{hi:>14.5f}{work(Pi):>12.5f}")

print()
print("  A symmetric interval means no protocol of this class can prefer cooling.")
print("  If rank-r projectors give |W_lo| != |W_hi| the degeneracy is broken by the")
print("  GENERATOR'S SPECTRUM alone - no frequency filter required - and the sign")
print("  of the bias says which direction the protocol favours.")
