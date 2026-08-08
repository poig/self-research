"""
Closed form for the work achievable at fixed system-ancilla correlation.

The isospectral construction (isospectral_family.py) exhibits ONE family on
which I(S:A) and E_N are invariant while W varies.  This script establishes the
general statement: the full set of work values reachable without changing any
correlation measure, in closed form, for an arbitrary system-local rotation.

DERIVATION
----------
Rotating the Hamiltonian and initial state together by a system-local V while
leaving the feedback generator fixed in the lab frame is equivalent to holding
H fixed and conjugating the feedback generator:

    K = P_1 (x) sum_i X_i        (P_1 = |1><1| on the ancilla)
    Ktilde = (I (x) V^dag) K (I (x) V) = P_1 (x) V^dag (sum_i X_i) V

The ancilla factor is untouched, so only the system factor moves, and it moves
over the isospectral orbit of sum_i X_i.  To first order in theta,

    W(V) = (theta/2) <Psi_1| i[A, Ktilde] |Psi_1|>            A = I_A (x) H
         = (theta/2) Tr( i[rho, A] . Ktilde )                 (cyclicity)
         = (theta/2) Tr( M_11 . Y )                           Y = V^dag (sum X) V

where M_11 is the (anc=1, anc=1) block of the Hermitian operator i[rho, A].
So the reachable work is a trace of a FIXED Hermitian operator against an
operator ranging over an isospectral orbit -- exactly the setting of von
Neumann's trace inequality:

    max_Y Tr(M_11 Y) = sum_k lambda_k^down(M_11) . lambda_k^down(sum X)
    min_Y Tr(M_11 Y) = sum_k lambda_k^down(M_11) . lambda_k^up(sum X)

sum_i X_i has spectrum {n-2k} symmetric about zero, so the ascending list is the
negation of the descending one and the interval is exactly [-W*, +W*].

TWO CONSEQUENCES
----------------
1. Zero work is ALWAYS reachable at unchanged correlation: conjugate Ktilde into
   an eigenbasis of H, where it commutes with H and the first-order work vanishes
   identically.  This is not a property of a chosen example -- no protocol of
   this form can avoid it.

2. The optimum is attained by an explicit V.  With U_X diagonalising sum X and
   U_M diagonalising M_11, both sorted descending, V = U_X U_M^dag gives
   Y = U_M Lambda_X^down U_M^dag and saturates the upper bound; reversing the
   sort attains the lower one.

CHECKS RUN HERE
---------------
  (a) closed form vs. the observed range of the one-parameter R_y sweep
  (b) Haar-random V: sampled work must lie inside [-W*, W*]
  (c) the constructed optimal V must attain the endpoints
  (d) I(S:A) and E_N must be unchanged at the optimum, or the bound is vacuous
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from scipy.stats import unitary_group

N = 4
THETA = 0.2
TAU = 1.042
N_HAAR = 400
S_POINTS = 181


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def build(n, fam):
    ops = []
    if fam == "sum-Z":
        for i in range(n):
            ops.append((lbl(n, **{str(i): "Z"}), 1.0))
    elif fam == "paper-fig1":
        np.random.seed(42)
        for i in range(n):
            for j in range(i + 1, n):
                ops.append((lbl(n, **{str(i): "Z", str(j): "Z"}),
                            np.random.uniform(-1, 1)))
        for i in range(n):
            ops.append((lbl(n, **{str(i): "X"}), np.random.uniform(-0.5, 0.5)))
    return SparsePauliOp.from_list(ops).to_matrix()


def sum_x(n):
    return sum(SparsePauliOp(lbl(n, **{str(i): "X"})).to_matrix() for i in range(n))


def sum_y(n):
    return sum(SparsePauliOp(lbl(n, **{str(i): "Y"})).to_matrix() for i in range(n))


def feedback_generator(n):
    ops = []
    for i in range(n):
        ops.append((lbl(n + 1, **{str(i + 1): "X"}), 0.5))
        ops.append((lbl(n + 1, **{str(i + 1): "X", "0": "Z"}), -0.5))
    return SparsePauliOp.from_list(ops).to_matrix()


def vn_entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-13]
    return float(-np.sum(ev * np.log2(ev)))


def log_negativity(joint, d):
    rho = np.outer(joint, joint.conj()).reshape(d, 2, d, 2)
    rho_pt = rho.transpose(0, 3, 2, 1).reshape(2 * d, 2 * d)
    return float(np.log2(np.sum(np.abs(np.linalg.eigvals(rho_pt)))))


def post_sensing(Hm, V, n, tau):
    """|Psi_1(V)> for the rotated configuration (V H V^dag, V|+>^n)."""
    d = 2 ** n
    psi = V @ (np.ones(d) / np.sqrt(d))
    H_s = V @ Hm @ V.conj().T
    joint = np.kron(psi, np.array([1.0, 1.0]) / np.sqrt(2.0))
    evals, evecs = np.linalg.eigh(H_s)
    u_sys = (evecs * np.exp(-1j * evals * tau)) @ evecs.conj().T
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    joint = (np.kron(np.eye(d), p0) + np.kron(u_sys, p1)) @ joint
    had = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    return np.kron(np.eye(d), had) @ joint, H_s


def observables(joint, H_s, K, u_fb, n):
    d = 2 ** n
    A = np.asarray(joint).reshape(d, 2)
    info = vn_entropy(A @ A.conj().T) + vn_entropy(A.conj().T @ A)
    e_n = log_negativity(joint, d)
    A_op = np.kron(H_s, np.eye(2))
    uv = u_fb @ joint
    work = float(np.real(np.vdot(joint, A_op @ joint) - np.vdot(uv, A_op @ uv)))
    comm = 1j * (A_op @ K - K @ A_op)
    work_fo = (THETA / 2.0) * float(np.real(np.vdot(joint, comm @ joint)))
    return info, e_n, work, work_fo


def m11_block(joint, H_s, n):
    """M_11 = (anc=1, anc=1) block of i[rho, A], with joint index = sys*2 + anc."""
    d = 2 ** n
    rho = np.outer(joint, joint.conj())
    A_op = np.kron(H_s, np.eye(2))
    M = 1j * (rho @ A_op - A_op @ rho)
    return M.reshape(d, 2, d, 2)[:, 1, :, 1]


def closed_form(M11, n):
    lam_M = np.sort(np.linalg.eigvalsh(M11))[::-1]
    lam_X = np.sort(np.linalg.eigvalsh(sum_x(n)))[::-1]
    w_max = (THETA / 2.0) * float(np.dot(lam_M, lam_X))
    w_min = (THETA / 2.0) * float(np.dot(lam_M, lam_X[::-1]))
    return w_min, w_max, lam_M, lam_X


def optimal_V(M11, n, reverse=False):
    """V with V^dag (sum X) V diagonal in M_11's eigenbasis, sorted to match."""
    _, U_M = np.linalg.eigh(M11)
    U_M = U_M[:, ::-1]                       # eigenvectors, eigenvalues descending
    _, U_X = np.linalg.eigh(sum_x(n))
    U_X = U_X[:, ::-1]
    if reverse:
        U_X = U_X[:, ::-1]
    return U_X @ U_M.conj().T


def main():
    K = feedback_generator(N)
    u_fb = expm(-1j * (THETA / 2.0) * K)
    G = sum_y(N) / 2.0

    for fam in ["sum-Z", "paper-fig1"]:
        Hm = build(N, fam)
        print("=" * 94)
        print(f"{fam}   N = {N}   theta = {THETA}   tau = {TAU}")
        print("=" * 94)

        # Reference configuration (V = I) fixes M_11 and all correlations.
        joint0, H0 = post_sensing(Hm, np.eye(2 ** N), N, TAU)
        info0, en0, w0, wfo0 = observables(joint0, H0, K, u_fb, N)
        M11 = m11_block(joint0, H0, N)
        w_min, w_max, lam_M, lam_X = closed_form(M11, N)

        print(f"  reference:  I(S:A) = {info0:.8f}   E_N = {en0:.8f}   "
              f"W = {w0:+.6f}  (1st order {wfo0:+.6f})")
        spec_str = np.array2string(lam_X, precision=1, max_line_width=200)
        print(f"  spectrum of sum X : {spec_str}")
        print(f"\n  CLOSED FORM   W in [{w_min:+.6f}, {w_max:+.6f}]"
              f"    symmetric: {abs(w_min + w_max):.2e}")

        # (a) one-parameter R_y sweep
        svals = np.linspace(0.0, 2.0, S_POINTS)
        ry = []
        for sv in svals:
            V = expm(-1j * (sv * np.pi) * G)
            j, h = post_sensing(Hm, V, N, TAU)
            ry.append(observables(j, h, K, u_fb, N)[3])
        ry = np.array(ry)
        print(f"  (a) R_y family        [{ry.min():+.6f}, {ry.max():+.6f}]"
              f"    saturation: {ry.max() / w_max * 100:5.1f}% of bound")

        # (b) Haar-random V
        rng = np.random.default_rng(0)
        haar = []
        for _ in range(N_HAAR):
            V = unitary_group.rvs(2 ** N, random_state=rng)
            j, h = post_sensing(Hm, V, N, TAU)
            haar.append(observables(j, h, K, u_fb, N)[3])
        haar = np.array(haar)
        viol = int(np.sum((haar > w_max + 1e-9) | (haar < w_min - 1e-9)))
        print(f"  (b) {N_HAAR} Haar V       [{haar.min():+.6f}, {haar.max():+.6f}]"
              f"    violations: {viol}")

        # (c) constructed optimum
        for tag, rev in [("max", False), ("min", True)]:
            V = optimal_V(M11, N, reverse=rev)
            j, h = post_sensing(Hm, V, N, TAU)
            i_v, e_v, w_v, wfo_v = observables(j, h, K, u_fb, N)
            target = w_max if not rev else w_min
            print(f"  (c) constructed {tag}   W_1st = {wfo_v:+.6f}   "
                  f"target {target:+.6f}   err {abs(wfo_v - target):.2e}")
            # (d) correlations must be untouched, else the bound means nothing
            print(f"      correlations   dI = {abs(i_v - info0):.2e}   "
                  f"dE_N = {abs(e_v - en0):.2e}   (exact W = {w_v:+.6f})")
        print()


if __name__ == "__main__":
    main()
