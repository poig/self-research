"""Can G be pushed below what Qiskit's grouper returns? The problem is NP-hard.

v119 measured G(general) ~ N^3.79 against G(qwc) ~ N^4.24 using Qiskit's
`SparsePauliOp.group_commuting`. Then a literature check found the technique is
well established - Verteletskyi, Yen & Izmaylov, arXiv:1907.03358 (2019) - so
v119 rediscovered a known result and its value is the sizing, not the idea.

BUT THAT PAPER ALSO SAYS SOMETHING v119 DID NOT USE. Its central observation is

    "the problem of the optimal grouping is equivalent to finding a minimum
     clique cover (MCC) for the Hamiltonian graph"

and MCC is **NP-hard**, so the paper tests "several polynomial heuristic
algorithms to solve it approximately". Which means Qiskit's grouper is ONE
heuristic among many, its answer is an upper bound rather than the optimum, and
the gap between heuristics is free to take.

THE GRAPH. Grouping mutually commuting terms = partitioning the commutation graph
into cliques = COLOURING the ANTICOMMUTATION graph, since a proper colouring puts
anticommuting terms in different classes and each colour class is then a mutually
commuting group. Number of colours = number of measurement settings = G.

Anticommutation is a symplectic inner product, so the whole adjacency matrix is
one matrix multiply over GF(2):

    A = (Z @ X.T + X @ Z.T) mod 2        A[i,j] = 1  <=>  P_i, P_j anticommute

and for QWC the relation is per-qubit disagreement instead, computed the same way.

WHAT THIS MEASURES. Qiskit's grouper against standard colouring heuristics -
largest-first (Welsh-Powell), smallest-last, DSATUR (saturation-largest-first),
independent-set - on v30's Jordan-Wigner Hamiltonians, for both QWC and general
commuting. Whichever wins, the honest framing is "a better heuristic on a known
NP-hard problem", not a new method.

WHAT WOULD BE WORTH SOMETHING. A reduction large enough to matter at the sizes
V6 would actually run, and a check of whether the ORDERING of heuristics is
stable across N - because if it is, the best one can just be adopted, and if it
is not, the grouper should try several and keep the best, which costs seconds of
classical time against circuits saved on every gradient for the whole run.

TIER (project rule R1): tier C, sanctioned - group counts are structural facts
about the Hamiltonian with no state evolution. No circuits built here.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.quantum_info import SparsePauliOp, Pauli

# v30's generator, imported not reimplemented (v119 learned this the hard way)
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_v30", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "v30_chemistry_scaling.py"))
    _v30 = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_v30)
es_hamiltonian_terms = _v30.es_hamiltonian_terms

try:
    import networkx as nx
    HAVE_NX = True
except ImportError:
    HAVE_NX = False


def zx(labels):
    Z = np.array([Pauli(t).z.astype(np.int8) for t in labels])
    X = np.array([Pauli(t).x.astype(np.int8) for t in labels])
    return Z, X


def anticommute_matrix(Z, X, qubit_wise=False):
    """A[i,j] = 1 iff P_i and P_j may NOT share a measurement setting."""
    if not qubit_wise:
        A = (Z @ X.T + X @ Z.T) % 2
    else:
        # qubit-wise: they clash if on ANY qubit both are non-identity and differ
        nz_i = (Z | X).astype(bool)                    # non-identity mask
        # encode each qubit's letter as 0..3 (I,X,Z,Y) via (z,x)
        code = (Z * 2 + X).astype(np.int8)
        T, n = code.shape
        A = np.zeros((T, T), dtype=np.int8)
        for q in range(n):
            c = code[:, q]
            both = np.outer(nz_i[:, q], nz_i[:, q])
            diff = c[:, None] != c[None, :]
            A |= (both & diff).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def colour_counts(A, strategies):
    """Number of colours for the anticommutation graph under each strategy."""
    out = {}
    G = nx.from_numpy_array(A)
    for s in strategies:
        t0 = time.time()
        col = nx.coloring.greedy_color(G, strategy=s)
        out[s] = (len(set(col.values())), time.time() - t0)
    return out


STRATS = ['largest_first', 'smallest_last', 'saturation_largest_first']


print("=" * 100)
print("v120  MINIMUM CLIQUE COVER:  is Qiskit's grouper leaving G on the table?")
print("=" * 100)
print("  Grouping = clique cover of the commutation graph = COLOURING the")
print("  anticommutation graph. MCC is NP-hard (arXiv:1907.03358), so every")
print("  grouper is a heuristic and its answer is an upper bound.")
print("  TIER C: structural, no circuits.")
print("  networkx available: %s" % HAVE_NX)
print()

if not HAVE_NX:
    print("  networkx missing - cannot run the colouring comparison.")
    sys.exit(0)

for qw in (True, False):
    tag = "QUBIT-WISE" if qw else "GENERAL"
    print("=" * 100)
    print("%s COMMUTING" % tag)
    print("=" * 100)
    print("     N   terms   qiskit   largest_1st  smallest_last   DSATUR   best   vs qiskit")
    print("   " + "-" * 88)
    for N in (6, 8, 10, 12):
        labels = es_hamiltonian_terms(N)
        op = SparsePauliOp.from_list([(s, 1.0) for s in labels])
        gq = len(op.group_commuting(qubit_wise=qw))
        Z, X = zx(labels)
        A = anticommute_matrix(Z, X, qubit_wise=qw)
        res = colour_counts(A, STRATS)
        vals = [res[s][0] for s in STRATS]
        best = min(vals + [gq])
        print("   %4d  %6d   %6d   %10d   %12d   %6d   %5d   %+6.1f%%"
              % (N, len(labels), gq, res['largest_first'][0],
                 res["smallest_last"][0], res["saturation_largest_first"][0],
                 best, 100.0 * (best - gq) / gq))
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  IF a standard heuristic beats Qiskit's grouper, the saving is free: it is")
print("  seconds of classical preprocessing against circuits saved on EVERY gradient")
print("  for the whole optimisation run. nisq_v6._group would take the best of")
print("  several rather than whatever `group_commuting` returns.")
print()
print("  IF Qiskit already wins or ties, its grouper is doing as well as the standard")
print("  greedy family and the remaining gap to the true MCC optimum is not")
print("  reachable this cheaply - which is itself worth knowing, because it closes")
print("  the question rather than leaving it as a vague 'maybe better heuristics'.")
print()
print("  EITHER WAY THIS IS NOT A NEW METHOD. The MCC formulation and these")
print("  heuristics are arXiv:1907.03358's, from 2019. What is measured here is the")
print("  gap on THIS Hamiltonian family at THESE sizes, which is a sizing result.")
print()
print("  Scope: v30's JW generator, four sizes, coefficients ignored (neither")
print("  grouping depends on them), greedy heuristics only - no exact MCC, no")
print("  simulated annealing, no LP relaxation.")
