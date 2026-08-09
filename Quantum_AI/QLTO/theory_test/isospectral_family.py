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

        # SIZED FOR THE PAGE, NOT FOR THE SCREEN. The manuscript is two-column
        # revtex; this goes in a full-width figure* at \textwidth ~ 7.0in, so the
        # figure is authored at exactly that width and the font sizes below are
        # the sizes that actually appear in print. The earlier 12x4.6 version
        # rendered its 12pt labels at ~3.4pt once scaled into a single column.
        # TWIN AXES, and the reason is not cosmetic. The correlations sit near
        # 2.0 and 1.0 while the work sits near +-0.2, so on a shared axis the
        # work curve - the only thing that MOVES, and the whole argument - is
        # compressed into the bottom sixth of the panel and its sign change is
        # invisible. Plotting it on its own right-hand scale lets both render at
        # full amplitude, which is what the figure is actually claiming: one pair
        # of quantities pinned, another sweeping through zero.
        TITLES = {"sum-Z": r"$H=\sum_i Z_i$",
                  "paper-fig1": r"random all-to-all $ZZ$ + transverse field"}
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
        for ax, fam in zip(axes, ["sum-Z", "paper-fig1"]):
            svals, rows = run(fam)
            l1, = ax.plot(svals, rows[:, 0], color="#1f77b4", linewidth=1.8,
                          label="$I(S{:}A)$ [bits]")
            l2, = ax.plot(svals, rows[:, 1], color="#17becf", linewidth=1.5,
                          linestyle="--", label="$E_N$")
            ax.set_ylim(0.0, 2.35)
            ax.set_xlabel(r"$s/\pi$", fontsize=9)
            ax.set_title(TITLES[fam], fontsize=9.5)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)

            axw = ax.twinx()
            l3, = axw.plot(svals, rows[:, 2], color="#d62728", linewidth=2.0,
                           label=r"work $-\Delta\langle H\rangle$")
            w = float(np.max(np.abs(rows[:, 2]))) * 1.35
            axw.set_ylim(-w, w)
            axw.axhline(0, color="#d62728", linestyle=":", alpha=0.55,
                        linewidth=1.0)
            axw.tick_params(labelsize=8, colors="#d62728")
            axw.spines["right"].set_color("#d62728")

            # mark the sign change: the point the whole construction exists for
            sgn = np.where(np.sign(rows[:-1, 2]) != np.sign(rows[1:, 2]))[0]
            if len(sgn):
                k = int(sgn[0])
                axw.plot([svals[k]], [rows[k, 2]], marker="o", ms=4.5,
                         color="#d62728", zorder=5)
                axw.annotate("work $=0$", xy=(svals[k], rows[k, 2]),
                             xytext=(svals[k] + 0.06, w * 0.42), fontsize=7.5,
                             color="#d62728",
                             arrowprops=dict(arrowstyle="->", color="#d62728",
                                             lw=0.8))
            handles = [l1, l2, l3]
        # ONE legend for the pair, below the axes. Per-panel legends land on the
        # E_N line in the right panel whichever corner they are put in, because
        # the work curve occupies the opposite corner in each.
        axes[0].set_ylabel("correlation  [bits]", fontsize=9)
        fig.legend(handles=handles, fontsize=8, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), frameon=False)
        fig.text(0.995, 0.58, r"extracted work  $-\Delta\langle H\rangle$",
                 rotation=270, va="center", fontsize=9, color="#d62728")
        plt.tight_layout(rect=(0, 0.07, 0.975, 1))
        plt.savefig("isospectral_family.png", dpi=300, bbox_inches="tight")
        print("\n[Saved] isospectral_family.png")
    except Exception as exc:
        print(f"\n[figure skipped] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
