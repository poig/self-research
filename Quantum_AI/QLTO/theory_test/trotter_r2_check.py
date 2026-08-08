"""
Is the ordered arm's R^2(W ~ I) a property of the dynamics or of LieTrotter(reps=1)?

thermo_scrambling_crash.py evolves with PauliEvolutionGate(LieTrotter(reps=1))
over 15 tau points and reports ordered R^2 in the 0.14-0.32 band.
harmonized_sweep.py evolves exactly over 20 tau points and reports the same
quantity in the 0.005-0.65 band, clearing the R2_MIN = 0.60 acceptance gate at
N = 7.  Same Hamiltonian (all-pairs J = -1, uniform h = 1.0), same tau range.

This script runs both evolutions through an identical measurement pipeline so
the only difference is exact vs. Trotterized, and reports R^2 side by side.
Written directly in numpy rather than through Qiskit circuits so that the
Trotter term ordering is explicit and matches get_hamiltonian()'s emission
order (all ZZ pairs, then all X).
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from scipy.stats import linregress

MAX_TAU = 1.5
THETA = 0.2
N_RANGE = [3, 4, 5, 6, 7]


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def build(n, fam, seed=42):
    rng = np.random.default_rng(seed)
    ops = []
    for i in range(n):
        for j in range(i + 1, n):
            J = -1.0 if fam == "ordered" else rng.uniform(-1, 1)
            ops.append((lbl(n, **{str(i): "Z", str(j): "Z"}), J))
    for i in range(n):
        h = 1.0 if fam == "ordered" else rng.uniform(-1, 1)
        ops.append((lbl(n, **{str(i): "X"}), h))
    return SparsePauliOp.from_list(ops)


def u_exact(spec, tau):
    evals, evecs = spec
    return (evecs * np.exp(-1j * evals * tau)) @ evecs.conj().T


def u_trotter(terms, tau, reps=1):
    """LieTrotter: product of exp(-i c P tau/reps) in emission order, reps times."""
    d = terms[0][1].shape[0]
    step = np.eye(d, dtype=complex)
    for coeff, pmat in terms:
        step = expm(-1j * coeff * pmat * (tau / reps)) @ step
    out = np.eye(d, dtype=complex)
    for _ in range(reps):
        out = step @ out
    return out


def feedback(n):
    ops = []
    for i in range(n):
        ops.append((lbl(n + 1, **{str(i + 1): "X"}), 0.5))
        ops.append((lbl(n + 1, **{str(i + 1): "X", "0": "Z"}), -0.5))
    k = SparsePauliOp.from_list(ops).to_matrix()
    return expm(-1j * (THETA / 2.0) * k)


def vn_entropy(rho):
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log2(ev)))


def cycle(Hm, u_sys, u_fb, n):
    """Ancilla is qubit 0 (least significant): joint index = sys * 2 + anc."""
    d = 2 ** n
    psi = np.ones(d) / np.sqrt(d)                      # |+>^n
    joint = np.kron(psi, np.array([1.0, 1.0]) / np.sqrt(2.0))

    p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    joint = (np.kron(np.eye(d), p0) + np.kron(u_sys, p1)) @ joint

    had = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    joint = np.kron(np.eye(d), had) @ joint

    A = joint.reshape(d, 2)
    rS = A @ A.conj().T
    rA = A.conj().T @ A
    info = vn_entropy(rS) + vn_entropy(rA)             # pure joint state
    e_before = float(np.real(np.trace(rS @ Hm)))

    A2 = (u_fb @ joint).reshape(d, 2)
    e_after = float(np.real(np.trace((A2 @ A2.conj().T) @ Hm)))
    return info, e_before - e_after


def run(n, fam, mode, n_taus):
    H = build(n, fam)
    Hm = H.to_matrix()
    terms = [(np.real(c), SparsePauliOp(p).to_matrix())
             for p, c in zip(H.paulis.to_labels(), H.coeffs)]
    spec = np.linalg.eigh(Hm)
    u_fb = feedback(n)

    taus = np.linspace(0.05, MAX_TAU, n_taus)
    data = []
    for t in taus:
        u = u_exact(spec, t) if mode == "exact" else u_trotter(terms, t)
        data.append(cycle(Hm, u, u_fb, n))
    data = np.array(data)
    fit = linregress(data[:, 0], data[:, 1])
    return fit.rvalue ** 2, fit.slope, np.abs(data[:, 1]).max()


def main():
    print("=" * 88)
    print("EXACT vs LieTrotter(reps=1) -- identical pipeline, only the evolution differs")
    print(f"  H: all-pairs ZZ (J=-1) + uniform X (h=1.0)   tau in [0.05, {MAX_TAU}]   "
          f"theta = {THETA}   init |+>^n")
    print("=" * 88)
    for fam in ["ordered", "chaotic"]:
        print(f"\n  {fam.upper()}")
        print(f"  {'N':>3} {'R2 exact/20pt':>15} {'R2 trot/15pt':>14} "
              f"{'R2 trot/20pt':>14} {'R2 exact/15pt':>15}")
        for n in N_RANGE:
            e20, _, _ = run(n, fam, "exact", 20)
            t15, _, _ = run(n, fam, "trotter", 15)
            t20, _, _ = run(n, fam, "trotter", 20)
            e15, _, _ = run(n, fam, "exact", 15)
            print(f"  {n:>3} {e20:>15.3f} {t15:>14.3f} {t20:>14.3f} {e15:>15.3f}")

    print("\n" + "=" * 88)
    print("Reference: thermo_scrambling_crash.py reported ordered R^2 = "
          "0.005 / 0.143 / 0.324 / 0.300 / 0.262 at N = 3-7")
    print("=" * 88)


if __name__ == "__main__":
    main()
