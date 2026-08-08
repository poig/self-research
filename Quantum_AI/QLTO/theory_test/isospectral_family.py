"""
Isospectral construction: an explicit one-parameter family on which every
system-ancilla correlation measure is EXACTLY constant while the extracted work
sweeps continuously through zero.

This supersedes the matched-information pair.  That comparison found two
configurations that happened to sit at the same mutual information; this one
constructs a continuum where the information cannot vary, so no function
W = f(I(S:A)) can exist -- and the same argument disposes of W = f(E_N).

THE CONSTRUCTION
----------------
Rotate the Hamiltonian and the initial state together by a system-local unitary
V(s), leaving the feedback generator fixed in the lab frame:

    H(s) = V(s) H V(s)^dag ,    |psi(s)> = V(s) |+>^n ,    K unchanged.

The controlled evolution then gives, for the post-sensing joint state,

    e^{-i H(s) tau} |psi(s)> = V e^{-iH tau} V^dag V |+> = V e^{-iH tau} |+>

so the whole joint state is

    |Psi_1(s)> = (I_A (x) V(s)) |Psi_1(0)> .

A local unitary on one subsystem leaves the reduced spectra unchanged, hence
I(S:A), S(A), S(S) and the logarithmic negativity are invariant in s -- exactly,
not to within numerical tolerance.

The work is not, because the feedback did not rotate:

    W(s) = <Psi_1(0)| A - Utilde^dag A Utilde |Psi_1(0)> ,  Utilde = (I(x)V^dag) U (I(x)V)

    W(s) = (theta/2) <i[H, V(s)^dag K V(s)]> + O(theta^2)

With H = sum Z_i and V(s) = exp(-i s sum Y_i / 2), the rotation carries
sum X_i -> cos(s) sum X_i -/+ sin(s) sum Z_i, and the sin term commutes with H.
So W(s) is proportional to cos(s) and passes through exactly zero at s = pi/2
while the information sits on a flat line.

Reported for two Hamiltonians: the clean H = sum Z case, where the cosine is
analytic, and the paper's own Fig. 1 instance, where the same invariance holds
but the s-dependence has no closed form -- showing the construction is a
property of the protocol, not of a special Hamiltonian.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

N = 4
THETA = 0.2
S_POINTS = 41            # s / pi from 0 to 1
TAU_DEFAULT = 1.042      # near the peak of |W| for the sum-Z family at N=4


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


def rotation_generator(n):
    """G = sum_i Y_i / 2, so V(s) = exp(-i s G) is a product of R_y(s)."""
    g = sum(SparsePauliOp(lbl(n, **{str(i): "Y"})).to_matrix() for i in range(n))
    return g / 2.0


def feedback_generator(n):
    """K = sum_i |1><1|_anc (x) X_i on n+1 qubits, ancilla = qubit 0."""
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
    """E_N = log2 || rho^{T_A} ||_1 for the 2 x d bipartition (ancilla | system)."""
    rho = np.outer(joint, joint.conj()).reshape(d, 2, d, 2)
    rho_pt = rho.transpose(0, 3, 2, 1).reshape(2 * d, 2 * d)
    return float(np.log2(np.sum(np.abs(np.linalg.eigvals(rho_pt)))))


def cycle(Hm, V, K, u_fb, n, tau):
    d = 2 ** n
    psi = V @ (np.ones(d) / np.sqrt(d))                 # |psi(s)> = V |+>^n
    H_s = V @ Hm @ V.conj().T                           # H(s) = V H V^dag
    joint = np.kron(psi, np.array([1.0, 1.0]) / np.sqrt(2.0))

    evals, evecs = np.linalg.eigh(H_s)
    u_sys = (evecs * np.exp(-1j * evals * tau)) @ evecs.conj().T
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    joint = (np.kron(np.eye(d), p0) + np.kron(u_sys, p1)) @ joint

    had = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    joint = np.kron(np.eye(d), had) @ joint             # |Psi_1(s)>

    A = np.asarray(joint).reshape(d, 2)
    info = vn_entropy(A @ A.conj().T) + vn_entropy(A.conj().T @ A)
    e_n = log_negativity(joint, d)

    A_op = np.kron(H_s, np.eye(2))                      # I_A (x) H(s)
    uv = u_fb @ joint
    work = float(np.real(np.vdot(joint, A_op @ joint) - np.vdot(uv, A_op @ uv)))
    comm = 1j * (A_op @ K - K @ A_op)
    work_fo = (THETA / 2.0) * float(np.real(np.vdot(joint, comm @ joint)))
    return info, e_n, work, work_fo


def run(fam, tau=TAU_DEFAULT):
    Hm = build(N, fam)
    G = rotation_generator(N)
    K = feedback_generator(N)
    u_fb = expm(-1j * (THETA / 2.0) * K)

    svals = np.linspace(0.0, 1.0, S_POINTS)             # in units of pi
    rows = []
    for sv in svals:
        V = expm(-1j * (sv * np.pi) * G)
        rows.append(cycle(Hm, V, K, u_fb, N, tau))
    return svals, np.array(rows)


def main():
    print("=" * 92)
    print("ISOSPECTRAL FAMILY  --  I(S:A) and E_N exactly invariant, W sweeps through zero")
    print(f"  N = {N}   theta = {THETA}   tau = {TAU_DEFAULT}   "
          f"V(s) = exp(-i s sum Y_i / 2),  s in [0, pi]")
    print("=" * 92)

    for fam in ["sum-Z", "paper-fig1"]:
        svals, rows = run(fam)
        info, e_n, work, work_fo = (rows[:, k] for k in range(4))

        print(f"\n  {fam}")
        print(f"  {'s/pi':>6} {'I(S:A)':>14} {'E_N':>14} {'W':>13} {'W_1st':>13} "
              f"{'cos(s)·W(0)':>13}")
        for idx in range(0, S_POINTS, S_POINTS // 8):
            pred = np.cos(svals[idx] * np.pi) * work[0]
            print(f"  {svals[idx]:>6.3f} {info[idx]:>14.8f} {e_n[idx]:>14.8f} "
                  f"{work[idx]:>+13.6f} {work_fo[idx]:>+13.6f} {pred:>+13.6f}")

        print(f"\n    I(S:A)  range over the whole sweep : {np.ptp(info):.3e}"
              f"   (value {info[0]:.8f})")
        print(f"    E_N     range over the whole sweep : {np.ptp(e_n):.3e}"
              f"   (value {e_n[0]:.8f})")
        print(f"    W       range over the whole sweep : {np.ptp(work):.6f}"
              f"   from {work.min():+.6f} to {work.max():+.6f}")
        zero_idx = int(np.argmin(np.abs(work)))
        print(f"    |W| minimum {abs(work[zero_idx]):.3e} at s/pi = "
              f"{svals[zero_idx]:.3f}, where I = {info[zero_idx]:.8f}")

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, fam in zip(axes, ["sum-Z", "paper-fig1"]):
            svals, rows = run(fam)
            ax.plot(svals, rows[:, 0], color="#1f77b4", linewidth=2.5,
                    label="$I(S{:}A)$ [bits]")
            ax.plot(svals, rows[:, 1], color="#17becf", linewidth=1.6,
                    linestyle="--", label="$E_N$")
            ax.plot(svals, rows[:, 2], color="#d62728", linewidth=2.5,
                    label=r"extracted work $-\Delta\langle H\rangle$")
            ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
            ax.set_xlabel(r"$s/\pi$", fontsize=12)
            ax.set_title(fam, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="center left")
        axes[0].set_ylabel("bits  /  energy", fontsize=12)
        fig.suptitle("Correlations held exactly fixed by construction; "
                     "work sweeps continuously to zero", fontsize=12)
        plt.tight_layout()
        plt.savefig("isospectral_family.png", dpi=150, bbox_inches="tight")
        print("\n[Saved] isospectral_family.png")
    except Exception as exc:
        print(f"\n[figure skipped] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
