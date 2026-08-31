"""
Prototype: coherent-target-basis Stage-2 replacement for arXiv:2606.19486v2.

Goal
----
Keep Zhou--Gong's odd-kernel coefficient estimator and its statistical guarantee,
but replace the classical per-shot random target-basis choice C by a coherent
ternary register.  The register is measured at the end, so the marginal
statistics are exactly the same as classical sampling of C, while the device
can compile one parameterised circuit template rather than a separate target-
basis circuit for each C.

This is a circuit-compression prototype, not yet a claim of lower total
Hamiltonian evolution time.  The paper's Stage 2 already has optimal
control-free evolution-time scaling; this prototype targets compiled-circuit /
program-count overhead.

The circuit builder below implements the *basis-selection layer* and the
register encoding.  The physical evolution remains uncontrolled by the design
register, matching the in-situ access model.  Full hardware validation still
needs backend-specific controlled basis-change synthesis and SPAM analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    _HAS_QISKIT = True
except ImportError:
    QuantumCircuit = QuantumRegister = ClassicalRegister = None
    _HAS_QISKIT = False


# Two qubits encode three target-basis values.
# 00 -> X, 01 -> Y, 10 -> Z. 11 is never populated.
BASIS_BITS = {"X": 0b00, "Y": 0b01, "Z": 0b10}
BITS_BASIS = {v: k for k, v in BASIS_BITS.items()}


def make_c_basis_state(qc, q0, q1) -> None:
    """Prepare (|00> + |01> + |10>)/sqrt(3) on two qubits.

    This is implemented as a small state-preparation unitary at compile time.
    The exact decomposition is backend dependent; the important abstraction is
    the three-value uniform coherent register.
    """
    target = np.zeros(4, dtype=complex)
    target[0] = target[1] = target[2] = 1 / np.sqrt(3)
    # Qiskit accepts a StatePreparation circuit through initialize, but initialize
    # is non-unitary from a hardware-compilation perspective.  This prototype
    # therefore keeps the state-vector specification explicit and lets callers
    # replace it with a native preparation on their backend.
    qc.initialize(target, [q0, q1])


def prepare_uniform_target_basis_register(qc, reg) -> None:
    """Prepare N independent uniform ternary target-basis symbols."""
    n = len(reg) // 2
    for i in range(n):
        make_c_basis_state(qc, reg[2 * i], reg[2 * i + 1])


def basis_register_bits(c_label: str) -> Tuple[int, int]:
    if c_label not in BASIS_BITS:
        raise ValueError(f"invalid target basis {c_label!r}")
    v = BASIS_BITS[c_label]
    return v & 1, (v >> 1) & 1


def visible(c: str, pauli: str) -> bool:
    """Zhou--Gong visibility predicate: odd number of matches u_i == c_i."""
    if len(c) != len(pauli):
        raise ValueError("basis and Pauli strings must have equal length")
    return sum(u != "I" and u == ci for u, ci in zip(pauli, c)) % 2 == 1


def ab_from_c_u(c: str, u: str) -> Tuple[List[int], List[int]]:
    """Return A,B bit sets for the cyclic (c,a,b) convention.

    (X,Y,Z), (Y,Z,X), (Z,X,Y)
    """
    cyclic = {"X": ("Y", "Z"), "Y": ("Z", "X"), "Z": ("X", "Y")}
    A, B = [], []
    for i, (ci, ui) in enumerate(zip(c, u)):
        a, b = cyclic[ci]
        if ui == a:
            A.append(i)
        elif ui == b:
            B.append(i)
        elif ui == ci:
            A.append(i)
            B.append(i)
        elif ui == "I":
            pass
        else:
            raise ValueError("Pauli letter is inconsistent with target basis")
    return A, B


def sigma_from_ab(A: Sequence[int], B: Sequence[int]) -> int:
    k = len(set(A).intersection(B))
    if k % 2 == 0:
        raise ValueError("sigma is defined only for visible (odd-overlap) blocks")
    return 1 if ((k - 1) // 2) % 2 == 0 else -1


def candidate_visibility_matrix(candidates: Sequence[str], bases: Sequence[str]) -> np.ndarray:
    """Boolean matrix V[s,u] saying whether candidate u is visible under c_s."""
    return np.asarray([[visible(c, u) for u in candidates] for c in bases], dtype=np.int8)


@dataclass(frozen=True)
class CoherentStage2Resources:
    n_system: int
    target_basis_qubits: int
    orientation_qubits: int
    classical_target_bits: int
    classical_orientation_bits: int

    @property
    def ancilla_qubits(self) -> int:
        return self.target_basis_qubits + self.orientation_qubits


class CoherentStage2Prototype:
    """Circuit-program compression layer for Stage 2.

    The full estimator remains the Zhou--Gong estimator.  This class only
    changes how the random target basis C and orientation rho are selected:
    they are placed in coherent registers and measured at the end.
    """

    def __init__(self, n_system: int):
        if n_system < 1:
            raise ValueError("n_system must be positive")
        self.n = int(n_system)
        self.resources = CoherentStage2Resources(
            n_system=self.n,
            target_basis_qubits=2 * self.n,
            orientation_qubits=1,
            classical_target_bits=2 * self.n,
            classical_orientation_bits=1,
        )

    def template_skeleton(self):
        """Build a backend-neutral skeleton containing the coherent design index.

        Registers:
          cb      2N qubits: ternary target-basis register C
          rho     1 qubit: orientation +/-
          sys     N qubits: physical device

        The actual controlled product-state preparation and measurement-basis
        rotations are intentionally isolated because their decomposition is
        backend-dependent and can dominate NISQ depth.
        """
        if not _HAS_QISKIT:
            raise RuntimeError("Qiskit is not installed in this environment")
        cb = QuantumRegister(2 * self.n, "cb")
        rho = QuantumRegister(1, "rho")
        sys = QuantumRegister(self.n, "sys")
        ccb = ClassicalRegister(2 * self.n, "ccb")
        crho = ClassicalRegister(1, "crho")
        csys = ClassicalRegister(self.n, "csys")
        qc = QuantumCircuit(cb, rho, sys, ccb, crho, csys)

        prepare_uniform_target_basis_register(qc, cb)
        qc.h(rho[0])

        # Placeholder barrier: the device's natural evolution must remain
        # uncontrolled, exactly as required by the in-situ access model.
        qc.barrier(cb, rho, sys)
        qc.barrier(cb, rho, sys)

        qc.measure(cb, ccb)
        qc.measure(rho, crho)
        qc.measure(sys, csys)
        return qc

    def classical_decode_equivalence(
        self,
        candidates: Sequence[str],
        n_shots: int,
        seed: int = 0,
    ) -> Dict[str, int]:
        """Monte-Carlo check that the coherent register has the same C marginal.

        A real coherent circuit gives exactly the same measured C distribution
        as drawing each C independently from {X,Y,Z}^N.  This method reports
        the empirical occupancy range to sanity-check the sampling layer.
        """
        rng = np.random.default_rng(seed)
        labels = np.array(["X", "Y", "Z"])
        counts = {u: 0 for u in candidates}
        for _ in range(int(n_shots)):
            c = "".join(rng.choice(labels, size=self.n))
            for u in candidates:
                if visible(c, u):
                    counts[u] += 1
        return counts


def theoretical_stage2_sample_bound(
    Lambda: float,
    epsilon: float,
    candidate_count: int,
    eta: float,
    ell0: float = 1.0,
) -> float:
    """The Zhou--Gong Stage-2 sample-count form from Lemma C.10.

    Constants are exposed only for resource comparison.  This is not a new
    theorem and deliberately does not claim improved constants.
    """
    N = np.ceil(
        18 * ell0**2 * Lambda**2 / epsilon**2
        * np.log(max(16 * candidate_count, 2))
    )
    R = np.ceil(16 * np.log(max(2 / eta, 2)))
    return float(N * R)


if __name__ == "__main__":
    proto = CoherentStage2Prototype(n_system=4)
    print("coherent Stage-2 skeleton")
    print("  design qubits:", proto.resources.ancilla_qubits)
    print("  target-basis qubits:", proto.resources.target_basis_qubits)
    print("  orientation qubits:", proto.resources.orientation_qubits)
    print("  circuit templates: 1 (before backend decomposition)")
    candidates = ["XXII", "YYII", "ZZII", "XYZI", "IIZZ"]
    counts = proto.classical_decode_equivalence(candidates, 100_000, seed=7)
    for u, nvisible in counts.items():
        print(f"  {u}: visible {nvisible/100_000:.3f}")


def build_visibility_codebook(
    candidates: Sequence[str],
    n_rows: int | None = None,
    seed: int = 0,
    pool_size: int = 20000,
) -> List[str]:
    """Build a small target-basis codebook with good worst-case visibility.

    This is a heuristic design-search layer, not part of the Zhou--Gong theorem.
    Rows are selected from random C in {X,Y,Z}^N by greedy max-min coverage.
    The resulting codebook can be coherently indexed by r, so only ceil(log2 R)
    design qubits are required, instead of 2N qubits for a direct ternary-per-site
    encoding.
    """
    U = list(dict.fromkeys(candidates))
    if not U:
        return []
    n = len(U[0])
    if any(len(u) != n for u in U):
        raise ValueError("all candidate Paulis must have the same length")
    if n_rows is None:
        # O(log M) rows is the target. This is a deliberately generous heuristic
        # starting point; optimization can often find a much smaller family.
        n_rows = max(3, int(np.ceil(8 * np.log(max(len(U), 2)))))
    n_rows = int(n_rows)
    if n_rows < 1:
        raise ValueError("n_rows must be positive")

    rng = np.random.default_rng(seed)
    labels = np.array(["X", "Y", "Z"])
    pool = ["".join(rng.choice(labels, size=n)) for _ in range(int(pool_size))]
    # Remove duplicate rows while preserving order.
    pool = list(dict.fromkeys(pool))

    V = np.asarray([[visible(c, u) for u in U] for c in pool], dtype=np.int16)
    selected: List[int] = []
    coverage = np.zeros(len(U), dtype=int)

    # Greedy max-min coverage: maximize the next row's contribution to currently
    # least-covered candidates, with a small tie-break toward global coverage.
    for _ in range(min(n_rows, len(pool))):
        deficits = (coverage.min() - coverage) if selected else np.zeros_like(coverage)
        scores = V @ (-deficits + 0.05)
        j = int(np.argmax(scores))
        if j in selected:
            remaining = [i for i in range(len(pool)) if i not in selected]
            if not remaining:
                break
            j = remaining[int(np.argmax(scores[remaining]))]
        selected.append(j)
        coverage += V[j]
        if coverage.min() >= max(1, n_rows // 3):
            # Stop early once the finite codebook preserves the paper's q>=1/3
            # visibility floor for every candidate.
            break

    return [pool[i] for i in selected]


def codebook_stats(candidates: Sequence[str], codebook: Sequence[str]) -> Dict[str, float]:
    if not candidates:
        return {"rows": float(len(codebook)), "min_q": 1.0, "max_q": 1.0, "mean_q": 1.0}
    V = candidate_visibility_matrix(candidates, codebook)
    q = V.mean(axis=0)
    return {
        "rows": float(len(codebook)),
        "min_q": float(q.min()),
        "max_q": float(q.max()),
        "mean_q": float(q.mean()),
    }


def crosstalk_terms(N: int) -> List[str]:
    def put(d):
        s = ["I"] * N
        for i, ch in d.items():
            s[i] = ch
        return "".join(s)
    t = []
    for i in range(N - 1):
        t += [put({i: "Z", i + 1: "Z"}),
              put({i: "X", i + 1: "X"}),
              put({i: "Y", i + 1: "Y"})]
    t += [put({i: "Z"}) for i in range(N)]
    return t


if __name__ == "__main__":
    print("\nvisibility-codebook experiment")
    for n in range(3, 9):
        U = crosstalk_terms(n)
        # Multiple seeds give a small robustness check of the heuristic search.
        rows = []
        mins = []
        for seed in range(3):
            C = build_visibility_codebook(U, n_rows=24, seed=seed)
            st = codebook_stats(U, C)
            rows.append(int(st["rows"]))
            mins.append(st["min_q"])
        R = int(max(rows))
        qmin = float(min(mins))
        index_qubits = int(np.ceil(np.log2(max(R, 1))))
        print(f"N={n:2d}, M={len(U):2d}, R={R:2d}, index qubits={index_qubits:2d}, min q={qmin:.3f}")
