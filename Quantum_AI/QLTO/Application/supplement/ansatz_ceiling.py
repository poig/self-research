"""
ansatz_ceiling.py

Which ansatz should QLTO be running on?

Two questions, both answerable without spending a single quantum circuit:

  1. CEILING - what is the lowest energy this ansatz can represent at all?
     Found by optimising the statevector classically to convergence from
     several restarts. This is the floor the optimiser is trying to reach; no
     improvement to gradient estimation can cross it.

  2. COST - what would QLTO pay per epoch on this ansatz?
     V2:        (2M - N) gradient + 2B walk/energy
     V3 layered: 2B + 1          (one sensing + one walk per block)
     V3 global:  3               (one sensing + one walk + one energy, all M
                                  parameters in a single param register)

Motivation: on Heisenberg N=4, V2 (-6.08) and V3 (-6.07) stop in the same place
despite a 4x difference in gradient cost and ~25x in gradient quality. When two
optimisers with very different signal quality plateau together, the ansatz is
the binding constraint, not the optimiser.

Also reports the Bowles et al. (arXiv:2306.14962) commuting-block conditions,
since those decide whether the 2B-1 gradient protocol is available at all:

  Cond A - generators within a block mutually commute
  Cond B - between any two blocks, generators ALL commute or ALL anticommute

Note B is only worth having when blocks hold many parameters. An ansatz with
one parameter per block has B = M, so 2B-1 = 2M-1 and the protocol buys
nothing over parameter-shift regardless of whether the conditions hold.

Usage:  python ansatz_ceiling.py [N ...]      (default: 4 6)
"""

import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonians
# ─────────────────────────────────────────────────────────────────────────────

def heisenberg(N) -> SparsePauliOp:
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def frustrated_ising(N, seed=999) -> SparsePauliOp:
    rng = np.random.RandomState(seed)
    ops = []
    for i in range(N):
        for j in range(i + 1, N):
            s = ["I"] * N
            s[i] = s[j] = "Z"
            ops.append(("".join(s), rng.uniform(-1.0, 1.0)))
    for i in range(N):
        s = ["I"] * N
        s[i] = "X"
        ops.append(("".join(s), rng.uniform(-1.0, 1.0)))
    return SparsePauliOp.from_list(ops)


def _group(N, pauli) -> SparsePauliOp:
    """Sum of nearest-neighbour <pauli><pauli> terms - mutually commuting."""
    ops = []
    for i in range(N - 1):
        s = ["I"] * N
        s[i] = s[i + 1] = pauli
        ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def _field(N, pauli) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [("".join("I" * i + pauli + "I" * (N - i - 1)), 1.0) for i in range(N)])


# ─────────────────────────────────────────────────────────────────────────────
# Ansatz candidates
# ─────────────────────────────────────────────────────────────────────────────

def hea(N, reps):
    """Hardware-efficient. Blocks are the rotation layers: B = 2(reps+1)."""
    a = efficient_su2(N, reps=reps)
    blocks = []
    for _ in range(reps + 1):
        for axis in ("Y", "Z"):
            blocks.append([_single(N, axis, q) for q in range(N)])
    return a, blocks


def _single(N, pauli, qubit) -> SparsePauliOp:
    s = ["I"] * N
    s[N - 1 - qubit] = pauli
    return SparsePauliOp("".join(s))


def hva(N, p, groups: List[SparsePauliOp], init: Callable[[QuantumCircuit], None]):
    """Hamiltonian Variational Ansatz: p layers of exp(-i theta G) per group.

    One parameter per group, so every block holds exactly one parameter.
    """
    qc = QuantumCircuit(N)
    init(qc)
    blocks = []
    for layer in range(p):
        for g_idx, G in enumerate(groups):
            th = Parameter(f"t_{layer}_{g_idx}")
            qc.append(PauliEvolutionGate(G, time=th, synthesis=LieTrotter(reps=1)),
                      range(N))
            blocks.append([G])
    return qc, blocks


def neel(qc):
    for q in range(0, qc.num_qubits, 2):
        qc.x(q)


def plus(qc):
    qc.h(range(qc.num_qubits))


# ─────────────────────────────────────────────────────────────────────────────
# Ceiling
# ─────────────────────────────────────────────────────────────────────────────

def ceiling(ansatz, H, restarts=4, seed=0, maxiter=600) -> Tuple[float, float]:
    """Lowest energy this ansatz can represent, by classical optimisation."""
    M = ansatz.num_parameters
    H_mat = H.to_matrix()
    rng = np.random.RandomState(seed)

    def energy(p):
        sv = Statevector.from_instruction(ansatz.assign_parameters(p)).data
        return float(np.real(np.vdot(sv, H_mat @ sv)))

    def grad(p):
        g = np.zeros(M)
        for i in range(M):
            pp = p.copy(); pp[i] += np.pi / 2
            pm = p.copy(); pm[i] -= np.pi / 2
            g[i] = 0.5 * (energy(pp) - energy(pm))
        return g

    best = np.inf
    t0 = time.time()
    for r in range(restarts):
        p0 = rng.uniform(-np.pi, np.pi, M)
        try:
            # parameter-shift gradients are exact for Pauli rotations; for the
            # HVA's multi-term generators they are not, so fall back there.
            res = minimize(energy, p0, jac=grad, method="L-BFGS-B",
                           options={"maxiter": maxiter})
            best = min(best, float(res.fun))
            res2 = minimize(energy, p0, method="L-BFGS-B",
                            options={"maxiter": maxiter})
            best = min(best, float(res2.fun))
        except Exception as e:
            print(f"      restart {r} failed: {e}")
    return best, time.time() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Bowles conditions
# ─────────────────────────────────────────────────────────────────────────────

def _zero(op) -> bool:
    return np.allclose(op.coeffs, 0, atol=1e-9)


def bowles_conditions(blocks) -> Tuple[bool, bool]:
    cond_a = True
    for gens in blocks:
        for i in range(len(gens)):
            for j in range(i + 1, len(gens)):
                if not _zero((gens[i] @ gens[j] - gens[j] @ gens[i]).simplify()):
                    cond_a = False

    cond_b = True
    for b1 in range(len(blocks)):
        for b2 in range(b1 + 1, len(blocks)):
            g1, g2 = blocks[b1], blocks[b2]
            tot = len(g1) * len(g2)
            nc = sum(1 for x in g1 for y in g2
                     if _zero((x @ y - y @ x).simplify()))
            na = sum(1 for x in g1 for y in g2
                     if _zero((x @ y + y @ x).simplify()))
            if nc != tot and na != tot:
                cond_b = False
    return cond_a, cond_b


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(label, ansatz, blocks, H, N, exact, restarts):
    M = ansatz.num_parameters
    B = len(blocks)
    e, dt = ceiling(ansatz, H, restarts=restarts)
    cond_a, cond_b = bowles_conditions(blocks)

    v2 = (2 * M - N) + 2 * B
    v3_layered = 2 * B + 1
    v3_global = 3
    gap = e - exact

    print(f"  {label:<24}{M:>4}{B:>4}{e:>11.4f}{gap:>9.4f}"
          f"{v2:>7}{v3_layered:>7}{v3_global:>7}"
          f"{('Y' if cond_a else 'n'):>4}{('Y' if cond_b else 'n'):>3}"
          f"{dt:>7.0f}s")
    return dict(label=label, M=M, B=B, energy=e, gap=gap,
                v2=v2, v3_layered=v3_layered, v3_global=v3_global,
                cond_a=cond_a, cond_b=cond_b)


def run(N, restarts=3):
    print(f"\n{'=' * 92}")
    print(f"Heisenberg N={N}")
    print("=" * 92)
    H = heisenberg(N)
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    print(f"exact ground state: {exact:.6f}\n")
    print(f"  {'ansatz':<24}{'M':>4}{'B':>4}{'ceiling':>11}{'gap':>9}"
          f"{'V2':>7}{'V3-L':>7}{'V3-G':>7}{'A':>4}{'B':>3}{'time':>8}")
    print("  " + "-" * 88)

    rows = []
    for reps in (1, 2, 3):
        a, blk = hea(N, reps)
        rows.append(evaluate(f"efficient_su2 reps={reps}", a, blk, H, N, exact, restarts))

    groups = [_group(N, p) for p in ("X", "Y", "Z")]
    for p in (1, 2, 4, 6):
        a, blk = hva(N, p, groups, neel)
        rows.append(evaluate(f"HVA p={p} (Neel)", a, blk, H, N, exact, restarts))

    print(f"\n{'=' * 92}")
    print(f"Frustrated Ising N={N}")
    print("=" * 92)
    Hi = frustrated_ising(N)
    exact_i = float(np.min(np.linalg.eigvalsh(Hi.to_matrix())))
    print(f"exact ground state: {exact_i:.6f}\n")
    print(f"  {'ansatz':<24}{'M':>4}{'B':>4}{'ceiling':>11}{'gap':>9}"
          f"{'V2':>7}{'V3-L':>7}{'V3-G':>7}{'A':>4}{'B':>3}{'time':>8}")
    print("  " + "-" * 88)

    for reps in (1, 2, 3):
        a, blk = hea(N, reps)
        evaluate(f"efficient_su2 reps={reps}", a, blk, Hi, N, exact_i, restarts)

    zz = SparsePauliOp.from_list([(str(p), c.real) for p, c in
                                  zip(Hi.paulis, Hi.coeffs) if "X" not in str(p)])
    xs = _field(N, "X")
    for p in (1, 2, 4, 6):
        a, blk = hva(N, p, [zz, xs], plus)
        evaluate(f"QAOA-style p={p}", a, blk, Hi, N, exact_i, restarts)

    return rows


if __name__ == "__main__":
    Ns = [int(x) for x in sys.argv[1:]] or [4, 6]
    print("Ceiling = lowest energy the ansatz can represent (classical optimisation).")
    print("V2 / V3-L / V3-G = circuits per epoch. A / B = Bowles conditions.")
    for N in Ns:
        run(N)
    print("\nNo quantum circuits were run. These are representability and cost")
    print("bounds only - they say what is reachable, not how hard it is to reach.")
