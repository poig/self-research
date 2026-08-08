"""The symmetric interval has TWO independent sufficient causes, not one.

Four attempts to break the symmetry failed, and the scan explains why: each broke
only one of two conditions, either of which is enough on its own.

Theorem 2's endpoints are

    hi = sum_k lam_dn(M11) lam_dn(Y) ,     lo = sum_k lam_dn(M11) lam_up(Y) .

CONDITION A - spec(Y) symmetric about zero. Then lam_up(Y) = -lam_dn(Y) termwise,
so lo = -hi for ANY M11 whatsoever. This is the condition the manuscript states.

CONDITION B - spec(M11) symmetric about zero. Then reversing the pairing negates
the sum for ANY Y. Guaranteed when the branch is pure, since rank(M11) <= 2 with
eigenvalues +/-Delta_H (Corollary 3), and also whenever the branch and H are
simultaneously real - measured, and consistent with random Hermitian pairs being
asymmetric at ~0.3 only when at least one is complex.

WHY EACH ATTEMPT FAILED, in hindsight:

  asymmetric generator (projectors)   broke A, but the state was pure  -> B held
  depolarising the state              [I,H]=0, M11 merely rescaled      -> B held
  protocol's multi-cycle mixing       branch and H both real           -> B held
  breaking T-symmetry of H (DM term)  M11 asymmetric, but Y = sum X_i  -> A held

So the escape needs BOTH broken at once: an asymmetric generator spectrum AND a
branch whose commutator with H is spectrally asymmetric. Neither alone suffices,
and testing them one at a time can only ever return a symmetric interval.

This file tests the conjunction.
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

N = 3
d = 2 ** N
TAU, THETA = 1.042, 0.2
rng = np.random.RandomState(11)


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def op(t):
    return SparsePauliOp.from_list(t).to_matrix()


H_real = op([(lbl(N, **{str(i): "Z", str(j): "Z"}), 0.7)
             for i in range(N) for j in range(i + 1, N)] +
            [(lbl(N, **{str(i): "X"}), 0.4) for i in range(N)])
H_cplx = H_real + op([(lbl(N, **{str(i): "X", str(i + 1): "Y"}), 0.6)
                      for i in range(N - 1)])

Y_sym = sum(op([(lbl(N, **{str(i): "X"}), 1.0)]) for i in range(N))   # spec symmetric
ev, evec = np.linalg.eigh(H_cplx)
Y_asym = evec[:, :2] @ evec[:, :2].conj().T                          # rank-2 projector

plus = np.ones(d) / np.sqrt(d)


def spec_defect(M):
    e = np.sort(np.linalg.eigvalsh(M))
    s = np.max(np.abs(e))
    return float(np.max(np.abs(e + e[::-1])) / s) if s > 1e-13 else 0.0


def interval(M, Ym):
    lm = np.sort(np.linalg.eigvalsh(M))[::-1]
    ly = np.sort(np.linalg.eigvalsh(Ym))[::-1]
    return (float(np.sum(lm * ly[::-1])) * THETA / 2.0,
            float(np.sum(lm * ly)) * THETA / 2.0)


def cycle(rho, Hm, Ym, tau=TAU, theta=THETA):
    Uv = expm(-1j * Hm * tau)
    W = np.vstack([np.eye(d) / np.sqrt(2), Uv / np.sqrt(2)])
    big = W @ rho @ W.conj().T
    U = expm(-1j * (theta / 2.0) * np.kron(np.diag([0.0, 1.0]), Ym))
    out = U @ big @ U.conj().T
    return out[:d, :d] + out[d:, d:]


def branch(Hm, cycles, Ym):
    rho = np.outer(plus, plus.conj())
    for _ in range(cycles):
        rho = cycle(rho, Hm, Ym)
    Uv = expm(-1j * Hm * TAU)
    return Uv @ rho @ Uv.conj().T


print("=" * 98)
print("BOTH CONDITIONS MUST BREAK — spec(Y) asymmetric AND spec(M11) asymmetric")
print("=" * 98)
print(f"  {'A broken?':<12}{'B broken?':<12}{'generator':<14}{'branch':<26}"
      f"{'defect M11':>12}{'W_lo':>10}{'W_hi':>10}{'interval':>12}")
print("  " + "-" * 96)

setups = [
    ("no", "no", "sum X_i", "pure", Y_sym, np.outer(plus, plus.conj()), H_real),
    ("YES", "no", "projector", "pure", Y_asym, np.outer(plus, plus.conj()), H_real),
    ("no", "no", "sum X_i", "mixed, H real, 4 cyc", Y_sym,
     branch(H_real, 4, Y_sym), H_real),
    ("no", "YES", "sum X_i", "mixed, H complex, 4 cyc", Y_sym,
     branch(H_cplx, 4, Y_sym), H_cplx),
    ("YES", "YES", "projector", "mixed, H complex, 4 cyc", Y_asym,
     branch(H_cplx, 4, Y_sym), H_cplx),
]

for a, b, gname, bname, Ym, br, Hm in setups:
    M = 0.5j * (br @ Hm - Hm @ br)
    lo, hi = interval(M, Ym)
    asym = abs(abs(hi) - abs(lo)) / max(abs(hi), 1e-15)
    verdict = "SYMMETRIC" if asym < 1e-9 else "ASYMMETRIC"
    print(f"  {a:<12}{b:<12}{gname:<14}{bname:<26}{spec_defect(M):>12.2e}"
          f"{lo:>10.5f}{hi:>10.5f}{verdict:>12}")

print()
print("  Only the last row breaks both. If it is the only ASYMMETRIC entry, the")
print("  manuscript's condition is incomplete: spec(Y) symmetric is sufficient for")
print("  a symmetric interval, but so is spec(M11) symmetric, and the latter holds")
print("  automatically for every pure branch by Corollary 3. A protocol that starts")
print("  pure therefore cannot be repaired by the generator alone, and one that uses")
print("  a symmetric generator cannot be repaired by the state alone.")
