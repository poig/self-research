"""Are spectral concentration and DLA dimension INCOMPARABLE certificates?

TRACTABILITY_CERTIFICATES.md reduces "extend the catalogue" to one question. Two
candidate predicates on a Hamiltonian's Pauli list:

    Pi_DLA(H, ansatz)  = 1  iff  dim g(H, ansatz) <= poly(n)
    Pi_conc,k(H)       = 1  iff  PR(H) <= k ,   PR = 1 / sum_S p(S)^2 ,
                                 p(S) = hhat(S)^2 / sum_T hhat(T)^2

PR is the participation ratio of the Pauli spectrum: the effective number of terms
carrying the weight. It is computable in O(#terms) straight off the coefficient
list, so condition (C1) of Definition 1 is immediate for it and NOT immediate for
Pi_DLA, which needs a Lie closure.

THREE OUTCOMES, and only one makes the programme non-empty:

  (1) Pi_conc => Pi_DLA          concentration is a cheap pre-check, nothing more
  (2) Pi_DLA => Pi_conc          dimension counting is strictly stronger
  (3) INCOMPARABLE               each certifies cases the other misses, so
                                 Pi_conc OR Pi_DLA strictly enlarges the certified
                                 set C, which by Corollary 1 is exactly what
                                 progress means

Outcome (3) is decided by exhibiting ONE Hamiltonian of each kind:

  A. concentrated spectrum, LARGE algebra
     A single Pauli term H = Z_1 Z_2 ... Z_n has PR = 1 - maximally concentrated.
     Paired with a hardware-efficient ansatz the algebra is not small. If the
     gradient variance is healthy, concentration certified a case dimension
     counting would reject.

  B. spread spectrum, SMALL algebra
     A sum of many commuting single-qubit Z terms has PR ~ n - spread - while
     generating an abelian algebra of dimension n. If the variance is healthy,
     the algebra certified a case concentration would reject.

WHAT IS MEASURED. dim g by explicit Lie closure over the Pauli group (exact, not
estimated), PR from the coefficient list, and Var(d_1 E) over random parameter
draws as the operational stand-in for trainability. The claim under test is only
about the RELATION between the two predicates; neither is being asserted sound.

A caution stated in advance: finding one example each way establishes logical
incomparability and says nothing about coverage. Two predicates can be
incomparable and still jointly certify a negligible fraction of anything anyone
wants to run. Coverage is a separate measurement and is not attempted here.
"""
import sys, os, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli

I2 = np.eye(2, dtype=complex)


def pauli_str(n, **kw):
    s = ["I"] * n
    for i, p in kw.items():
        s[int(i)] = p
    return "".join(s[::-1])


def participation_ratio(H):
    """PR of the Pauli spectrum: effective number of terms carrying the weight."""
    c = np.abs(np.asarray(H.coeffs, dtype=complex)) ** 2
    if c.sum() <= 0:
        return 0.0
    p = c / c.sum()
    return float(1.0 / np.sum(p ** 2))


def dla_dimension(generators, max_dim=4096):
    """Exact Lie closure over the Pauli group, up to a cap.

    Paulis close under commutation up to phase, so the closure can be computed on
    Pauli LABELS rather than matrices: [P,Q] is proportional to PQ when P,Q
    anticommute and vanishes when they commute. Dimension is the number of
    distinct labels generated."""
    def commutes(a, b):
        # Pauli labels anticommute iff an odd number of positions have differing
        # non-identity letters
        d = 0
        for x, y in zip(a, b):
            if x != "I" and y != "I" and x != y:
                d += 1
        return d % 2 == 0

    def prod_label(a, b):
        out = []
        for x, y in zip(a, b):
            if x == "I":
                out.append(y)
            elif y == "I":
                out.append(x)
            elif x == y:
                out.append("I")
            else:
                out.append({"XY": "Z", "YX": "Z", "YZ": "X", "ZY": "X",
                            "XZ": "Y", "ZX": "Y"}[x + y])
        return "".join(out)

    basis = set(generators)
    frontier = set(generators)
    while frontier and len(basis) < max_dim:
        new = set()
        for a in frontier:
            for b in basis:
                if not commutes(a, b):
                    lab = prod_label(a, b)
                    if lab not in basis and set(lab) != {"I"}:
                        new.add(lab)
        if not new:
            break
        basis |= new
        frontier = new
    return len(basis), len(basis) >= max_dim


def ansatz_generators(n):
    """efficient_su2: RY and RZ on every qubit, plus the CX entanglers' effect is
    captured by including the single-qubit generators (the closure grows from
    these under commutation with H's terms)."""
    g = []
    for i in range(n):
        g.append(pauli_str(n, **{str(i): "Y"}))
        g.append(pauli_str(n, **{str(i): "Z"}))
    return g


def grad_variance(H, n, draws=40, seed=0):
    """Var(d_1 E) over random parameter draws: the operational stand-in."""
    a = efficient_su2(n, reps=2)
    Hm = H.to_matrix()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        th = rng.uniform(-np.pi, np.pi, a.num_parameters)
        g = 0.0
        for s in (+1, -1):
            t = th.copy()
            t[0] += s * np.pi / 2
            v = Statevector(a.assign_parameters(t)).data
            g += s * float(np.real(np.conj(v) @ (Hm @ v))) / 2
        vals.append(g)
    return float(np.var(vals))


N = 6
print("=" * 100)
print("ARE Pi_conc AND Pi_DLA INCOMPARABLE?")
print("=" * 100)
print(f"  n = {N}. PR = participation ratio of the Pauli spectrum (C1-checkable")
print(f"  in O(#terms)); dim g by exact Pauli-label Lie closure, capped at 4096.")
print(f"  Var(d_1 E) over 40 random parameter draws as the trainability proxy.")
print()

cases = {}

# A. concentrated spectrum, one global Pauli term
cases["A  single global ZZ..Z"] = SparsePauliOp.from_list(
    [(pauli_str(N, **{str(i): "Z" for i in range(N)}), 1.0)])

# B. spread spectrum, commuting single-qubit terms (abelian algebra)
cases["B  sum of local Z_i"] = SparsePauliOp.from_list(
    [(pauli_str(N, **{str(i): "Z"}), 1.0 + 0.1 * i) for i in range(N)])

# reference points
cases["   Heisenberg chain"] = SparsePauliOp.from_list(
    [(pauli_str(N, **{str(i): p, str(i + 1): p}), 1.0)
     for i in range(N - 1) for p in "XYZ"])
rng = np.random.default_rng(3)
cases["   random 2-local"] = SparsePauliOp.from_list(
    [(pauli_str(N, **{str(i): p, str(j): q}), float(rng.normal()))
     for i in range(N) for j in range(i + 1, N)
     for p in "XYZ" for q in "XYZ"][:60])

print(f"  {'Hamiltonian':>24}{'#terms':>8}{'PR':>8}{'dim g':>9}{'capped':>8}"
      f"{'Var(d1 E)':>13}")
print("  " + "-" * 70)

rows = {}
for tag, H in cases.items():
    labels = [p.to_label() for p in H.paulis]
    gens = sorted(set(labels) | set(ansatz_generators(N)))
    dim, capped = dla_dimension(gens)
    pr = participation_ratio(H)
    var = grad_variance(H, N)
    rows[tag] = (pr, dim, var)
    print(f"  {tag:>24}{len(labels):>8}{pr:>8.2f}{dim:>9}{str(capped):>8}"
          f"{var:>13.3e}")

print()
prA, dimA, varA = rows["A  single global ZZ..Z"]
prB, dimB, varB = rows["B  sum of local Z_i"]
print(f"  A: PR = {prA:.2f} (maximally concentrated), dim g = {dimA}")
print(f"  B: PR = {prB:.2f} (spread over {len(cases['B  sum of local Z_i'].paulis)}"
      f" terms), dim g = {dimB}")
print()
print("  INCOMPARABLE requires A concentrated with LARGE dim g, and B spread with")
print("  SMALL dim g. If both hold, neither predicate implies the other and their")
print("  disjunction certifies strictly more than either alone - outcome (3), the")
print("  one that makes the catalogue programme non-empty. If instead PR and dim g")
print("  move together across all four rows, one predicate is a proxy for the")
print("  other and Pi_conc adds nothing but speed.")
print()
print("  NOTE the variance column is context, not evidence: it says whether the")
print("  case is trainable at all, not which predicate deserves credit. Soundness")
print("  of Pi_conc is untouched by this run and remains open.")
