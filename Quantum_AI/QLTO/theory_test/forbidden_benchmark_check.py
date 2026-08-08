"""
Corollary 1 applied to the benchmark suite, and a corrected alignment measurement.

PART 1 -- a falsifiable prediction about existing benchmarks.

The protocol's feedback is one CR_X(theta) per system qubit, i.e. the unitary
exp(-i (theta/2) P_1 (x) sum_i X_i), so its generator is Y = sum_i X_i.  The
Heisenberg chain sum_i (X_iX_{i+1} + Y_iY_{i+1} + Z_iZ_{i+1}) has full SU(2)
symmetry, so sum_i X_i is a conserved quantity and [H, sum_i X_i] = 0 exactly.

By Corollary 1 the extracted work must then be identically zero -- for every
sensing time, every feedback strength, every initial state.  The Heisenberg
problems in the benchmark suite are therefore cases on which the ancilla feedback
mechanism can contribute nothing, and any convergence observed on them came from
the separate classical gradient path.

This part checks the prediction through the actual protocol pipeline rather than
through the commutator alone.

PART 2 -- the alignment decomposition, on a Hamiltonian that is not symmetric
under the generator.  Transverse-field Ising with random couplings does not
commute with sum X, and single-qubit generators (what a real ansatz parameter
carries) are used alongside it.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

THETA = 0.2
TAUS = np.linspace(0.05, 1.5, 12)
N_SAMPLES = 200
N_RANGE = range(2, 11)


def lbl(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def heisenberg(n):
    ops = []
    for i in range(n - 1):
        for p in "XYZ":
            ops.append((lbl(n, **{str(i): p, str(i + 1): p}), 1.0))
    return SparsePauliOp.from_list(ops).to_matrix()


def tfim(n, seed=42):
    rng = np.random.default_rng(seed)
    ops = []
    for i in range(n - 1):
        ops.append((lbl(n, **{str(i): "Z", str(i + 1): "Z"}), rng.uniform(-1, 1)))
    for i in range(n):
        ops.append((lbl(n, **{str(i): "X"}), rng.uniform(-1, 1)))
    return SparsePauliOp.from_list(ops).to_matrix()


def pauli(n, **kw):
    return SparsePauliOp(lbl(n, **kw)).to_matrix()


def sum_x(n):
    return sum(pauli(n, **{str(i): "X"}) for i in range(n))


def protocol_work(Hm, n, tau, theta=THETA):
    """Full sense -> lock -> feedback cycle; ancilla is qubit 0 (least significant)."""
    d = 2 ** n
    psi = np.ones(d) / np.sqrt(d)
    joint = np.kron(psi, np.array([1.0, 1.0]) / np.sqrt(2.0))
    evals, evecs = np.linalg.eigh(Hm)
    u_sys = (evecs * np.exp(-1j * evals * tau)) @ evecs.conj().T
    p0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    joint = (np.kron(np.eye(d), p0) + np.kron(u_sys, p1)) @ joint
    had = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    joint = np.kron(np.eye(d), had) @ joint

    K = np.kron(sum_x(n), p1)
    u_fb = expm(-1j * (theta / 2.0) * K)
    A = np.kron(Hm, np.eye(2))
    uv = u_fb @ joint
    return float(np.real(np.vdot(joint, A @ joint) - np.vdot(uv, A @ uv)))


def part1():
    print("=" * 94)
    print("PART 1  --  Corollary 1 on the benchmark Hamiltonians")
    print("  feedback generator Y = sum_i X_i   (one CR_X per system qubit)")
    print("=" * 94)
    print(f"  {'H':>12} {'N':>3} {'||[H, sum X]||':>16} {'max |W| over tau':>18}")
    for name, maker in [("Heisenberg", heisenberg), ("TFIM(random)", tfim)]:
        for n in [2, 3, 4, 5, 6]:
            Hm = maker(n)
            G = sum_x(n)
            comm_norm = float(np.max(np.abs(np.linalg.eigvalsh(
                1j * (Hm @ G - G @ Hm)))))
            works = [abs(protocol_work(Hm, n, t)) for t in TAUS]
            print(f"  {name:>12} {n:>3} {comm_norm:>16.3e} {max(works):>18.3e}")
        print()


def part2():
    rng = np.random.default_rng(3)
    print("=" * 94)
    print("PART 2  --  alignment decomposition on a NON-symmetric Hamiltonian")
    print("  H = random transverse-field Ising;  alpha = |grad| / (Delta_H * range(G))")
    print("=" * 94)

    for gname in ["sum_X (protocol)", "Y_0 (single param)"]:
        print(f"\n  generator: {gname}")
        print(f"  {'N':>3} {'Delta_H':>11} {'|grad|':>13} {'ceiling':>11} "
              f"{'alpha':>12} {'alpha drop':>11}")
        prev = None
        for n in N_RANGE:
            Hm = tfim(n)
            if gname.startswith("sum_X"):
                G, rg = sum_x(n), 2.0 * n
            else:
                G, rg = pauli(n, **{"0": "Y"}), 2.0
            dHs, grads, alphas = [], [], []
            for _ in range(N_SAMPLES):
                v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
                v /= np.linalg.norm(v)
                e1 = float(np.real(v.conj() @ (Hm @ v)))
                e2 = float(np.real(v.conj() @ (Hm @ (Hm @ v))))
                dH = float(np.sqrt(max(e2 - e1 ** 2, 0.0)))
                gr = abs(float(np.real(v.conj() @ ((1j * (Hm @ G - G @ Hm)) @ v))))
                dHs.append(dH)
                grads.append(gr)
                alphas.append(gr / max(dH * rg, 1e-18))
            dH_m, gr_m, al_m = np.mean(dHs), np.mean(grads), np.mean(alphas)
            drop = "" if prev is None else f"{prev / max(al_m, 1e-18):>11.2f}"
            print(f"  {n:>3} {dH_m:>11.4f} {gr_m:>13.3e} {dH_m * rg:>11.4f} "
                  f"{al_m:>12.3e} {drop}")
            prev = al_m


if __name__ == "__main__":
    part1()
    part2()
