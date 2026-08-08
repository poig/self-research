"""Why does the Heisenberg pairing give zero work - the state, or the symmetry?

The manuscript attributes it to the INITIAL STATE: |+>^n is the fully polarised
state, hence an exact eigenstate of the isotropic Heisenberg chain, so the
controlled evolution produces only a global phase and W = 0.

The asymmetry diagnostic gives a different reason. It measures [H, Y] = 0
identically for that pairing, because the isotropic Heisenberg Hamiltonian is
SU(2)-symmetric and therefore commutes with every total-spin operator, including
Y = sum_i X_i. Corollary 1 then forces W = 0 for EVERY state, not only for
eigenstates.

Both explanations predict zero at |+>^n. They differ on a NON-eigenstate initial
state: the eigenstate reading predicts nonzero work, the symmetry reading
predicts exactly zero. One measurement separates them.
"""
import numpy as np
from scipy.linalg import expm
from asymmetry_diagnostic import build_H, sum_X, N

TAU, THETA = 1.042, 0.2

H = build_H(N, "heisenberg")
Hm = H.to_matrix()
Ym = sum_X(N).to_matrix()
d = 2 ** N
plus = np.ones(d) / np.sqrt(d)


def eig_residual(psi):
    return float(np.linalg.norm(Hm @ psi - (psi.conj() @ Hm @ psi) * psi))


def work(psi, tau=TAU, theta=THETA):
    psi1 = expm(-1j * Hm * tau) @ psi
    Psi1 = np.zeros(2 * d, dtype=complex)
    Psi1[:d] = psi / np.sqrt(2)
    Psi1[d:] = psi1 / np.sqrt(2)
    A = np.kron(np.eye(2), Hm)
    K = np.kron(np.diag([0.0, 1.0]), Ym)
    U = expm(-1j * (theta / 2.0) * K)
    return float(np.real(Psi1.conj() @ (A - U.conj().T @ A @ U) @ Psi1))


print("=" * 78)
print("HEISENBERG ZERO — is it the eigenstate or the symmetry?")
print("=" * 78)
print(f"  |+>^n eigenstate residual : {eig_residual(plus):.2e}")
print(f"  ||[H, sum_i X_i]||        : {np.linalg.norm(Hm @ Ym - Ym @ Hm):.2e}")
print()
print("  Eigenstate reading predicts NONZERO W off the eigenstate.")
print("  Symmetry reading predicts EXACTLY ZERO for every state.")
print()
print(f"  {'initial state':<20}{'eigenstate?':>13}{'exact W':>13}")
print("  " + "-" * 46)
print(f"  {'|+>^n':<20}{'yes':>13}{work(plus):>13.2e}")
for k in range(4):
    rng = np.random.RandomState(100 + k)
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi = v / np.linalg.norm(v)
    tag = 'yes' if eig_residual(psi) < 1e-9 else 'no'
    print(f"  {'random ' + str(k + 1):<20}{tag:>13}{work(psi):>13.2e}")

print()
print("  Zero on non-eigenstates settles it: the cause is [H,Y]=0, i.e. SU(2)")
print("  symmetry of the isotropic chain, not the choice of initial state. The")
print("  degeneracy is therefore a property of the H/Y PAIRING and cannot be")
print("  repaired by changing the input - it needs a generator outside the")
print("  symmetry algebra.")
