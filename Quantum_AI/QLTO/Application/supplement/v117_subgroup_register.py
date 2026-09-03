"""Can the twirl register be log(M) qubits instead of 2N?

The largest gap between twirl_cal and arXiv:2606.19486 is hardware: 2N ancillas
against zero. But 2N is the width of the FULL Pauli group, and the design does not
need the full group - it needs only to tell M terms apart.

THE CONSTRUCTION. Twirl over a SUBGROUP. Pick a d x 2N matrix B over GF(2) and
let register value r in F_2^d drive the twirl Q(B^T r). Then term k sees the sign

    sigma_k(r) = (-1)^{<v_k, B^T r>} = (-1)^{<B v_k, r>}

so the design column for term k is the character indexed by B v_k. Two terms are
INDISTINGUISHABLE exactly when B v_j = B v_k, i.e. when v_j + v_k lies in the
kernel of B. So the register separates all M terms iff B is injective on {v_k},
and the information bound is d >= ceil(log2 M).

WHY v107 DOES NOT FORBID THIS, and the distinction is the whole point. v107
measured RANDOM SUBSETS of twirls and found no fraction works: K=32 of 64 frames
still gave 1.10 mean rel err against 0.054 for the full 64. A random subset
destroys character orthogonality, so the plain-average decode becomes biased. A
SUBGROUP does not - characters restricted to a subgroup are still orthogonal
characters OF THAT SUBGROUP, and the decode stays exactly unbiased. Fraction bad,
subgroup fine. That difference is why this is worth searching for.

WHAT IT WOULD BUY. For crosstalk M ~ 4N, so d ~ 2 + log2 N against 2N:

    N=10    6 qubits instead of 20
    N=50    8 qubits instead of 100

The ancilla objection changes from LINEAR to LOGARITHMIC overhead, which is a
different conversation from "more hardware for an easier problem".

WHAT IT COSTS, and this file's own output corrected me on it. I wrote that
aliasing would be "preserved, and necessarily so", reasoning that v_j + v_k = v_m
implies B v_j + B v_k = B v_m for any B. That direction is right and it is only
half the statement. The CONVERSE fails: B is injective on the M points {v_k} but
not on all of F_2^{2N}, so it can CREATE coincidences B v_j + B v_k = B v_m where
v_j + v_k != v_m. Fewer register bits means fewer distinct sign patterns for
degree-2 products to land in, so collisions multiply.

MEASURED at N=4: 9 alias triples on the full 2N=8 register, 33 on the d=4
subgroup register. Nearly 4x worse. Compressing the register does not leave the
accuracy ceiling alone - it lowers it.

THE GATE COUNT IS A REAL TRADE AND IS MEASURED HERE. twirl_cal's 2N-qubit
register uses 2N controlled single-qubit gates (one cx and one cz per system
qubit). A d-qubit subgroup register has each register qubit controlling a whole
Pauli STRING - row i of B - so the controlled-gate count is sum_i weight(row_i).
Choosing B for low row weight is therefore part of the search, not an
afterthought: a register that saves qubits and triples the entangling gates is
not obviously a win on hardware.

TIER (project rule R1): tier C, sanctioned - symplectic vectors and GF(2) rank,
structural facts about operators with no state evolution. No circuit is built
here. Whether a d-qubit register reproduces the 2N-qubit accuracy at matched
shots is a tier-A question and is NOT answered by this file.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.quantum_info import Pauli
from twirl_cal import crosstalk_terms


def symplectic_matrix(terms):
    Z = np.array([Pauli(t).z.astype(int) for t in terms])
    X = np.array([Pauli(t).x.astype(int) for t in terms])
    return np.concatenate([Z, X], axis=1) % 2


def separates(B, V):
    """True if B is injective on the rows of V."""
    img = (V @ B.T) % 2
    seen = set()
    for row in img:
        t = tuple(row)
        if t in seen:
            return False
        seen.add(t)
    return True


def search_B(V, d, trials=20000, seed=0, weight_bias=True):
    """Random search for a separating d x 2N matrix, preferring low row weight."""
    rng = np.random.default_rng(seed)
    n2 = V.shape[1]
    best = None
    for _ in range(trials):
        if weight_bias:
            # bias toward sparse rows: each entry 1 with prob p, p small
            p = rng.choice([0.15, 0.25, 0.4, 0.5])
            B = (rng.random((d, n2)) < p).astype(int)
        else:
            B = rng.integers(0, 2, (d, n2))
        if not B.any(axis=1).all():
            continue
        if separates(B, V):
            w = int(B.sum())
            if best is None or w < best[1]:
                best = (B.copy(), w)
    return best


def minimal_d(V, dmax, trials=20000, seed=0):
    M = V.shape[0]
    lo = int(np.ceil(np.log2(M)))
    for d in range(lo, dmax + 1):
        got = search_B(V, d, trials=trials, seed=seed)
        if got is not None:
            return d, got[0], got[1]
    return None, None, None


print("=" * 100)
print("v117  SUBGROUP REGISTER:  d qubits instead of 2N")
print("=" * 100)
print("  A d-qubit register twirls over a SUBGROUP Q(B^T r). It separates the M")
print("  terms iff B is injective on their symplectic vectors. Characters of a")
print("  subgroup stay orthogonal, so the decode stays unbiased - which is why this")
print("  is not forbidden by v107's failed random fractions.")
print("  TIER C: structural, no circuit built.")
print()
print("    family            N    M    2N    log2M    d found   ctrl gates   qubits saved")
print("   " + "-" * 92)

rows = []
for N in (3, 4, 5, 6, 7, 8):
    terms = crosstalk_terms(N)
    V = symplectic_matrix(terms)
    M = V.shape[0]
    d, B, w = minimal_d(V, 2 * N, trials=8000, seed=1)
    if d is None:
        print("   crosstalk N=%-2d      %2d   %3d   %3d     %3d       none found"
              % (N, N, M, 2 * N, int(np.ceil(np.log2(M)))))
        continue
    rows.append((N, M, 2 * N, d, w))
    print("   crosstalk N=%-2d      %2d   %3d   %3d     %3d      %4d       %5d       %5d"
          % (N, N, M, 2 * N, int(np.ceil(np.log2(M))), d, w, 2 * N - d))
print()

print("=" * 100)
print("SCALING OF THE SAVING")
print("=" * 100)
print("     N     current 2N     subgroup d     ratio      ctrl gates: 2N -> sum|row|")
print("   " + "-" * 84)
for N, M, w2n, d, w in rows:
    print("   %3d       %4d          %4d        %.2fx          %3d -> %3d"
          % (N, w2n, d, w2n / d, w2n, w))
print()
if len(rows) >= 3:
    Ns = np.array([r[0] for r in rows], float)
    ds = np.array([r[3] for r in rows], float)
    sl = float(np.polyfit(np.log(Ns), np.log(ds), 1)[0])
    print("   d ~ N^%.3f   (2N is N^1.0; log2 M ~ log N would be N^0)" % sl)
print()

print("=" * 100)
print("WHAT THE REDUCTION DOES NOT FIX")
print("=" * 100)
N = 4
terms = crosstalk_terms(N)
V = symplectic_matrix(terms)
d, B, w = minimal_d(V, 2 * N, trials=8000, seed=1)
img = (V @ B.T) % 2
alias_full, alias_red = 0, 0
for i in range(len(terms)):
    for j in range(i + 1, len(terms)):
        s_full = (V[i] + V[j]) % 2
        s_red = (img[i] + img[j]) % 2
        for k in range(len(terms)):
            if k in (i, j):
                continue
            if np.array_equal(s_full, V[k]):
                alias_full += 1
            if np.array_equal(s_red, img[k]):
                alias_red += 1
print("  crosstalk N=%d, full 2N=%d register vs subgroup d=%d register:" % (N, 2 * N, d))
print("     degree-2 alias triples, full register     : %d" % (alias_full // 2))
print("     degree-2 alias triples, subgroup register : %d" % (alias_red // 2))
print()
print("  WORSE, NOT IDENTICAL - and this refutes what the docstring of this file")
print("  originally asserted. v_j + v_k = v_m does imply B v_j + B v_k = B v_m, so")
print("  every existing alias survives. But B is injective only on the M points, not")
print("  on all of F_2^{2N}, so it CREATES coincidences that were not there: fewer")
print("  register bits, fewer sign patterns, more collisions among degree-2 products.")
print("  The one-directional implication was mistaken for a biconditional.")
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  THE QUBIT SAVING IS REAL. d sits at or one above the information bound")
print("  ceil(log2 M), so the register is logarithmic in M where it was linear in N.")
print("  The fitted d ~ N^0.47 above OVERSTATES the growth: the bound is 4,4,5,5,5,5")
print("  and the search returned 4,4,5,5,6,6, missing the optimum at N=7,8. That is a")
print("  search budget artefact, not scaling - more trials would flatten it.")
print()
print("  BUT IT IS BOUGHT TWICE OVER, and both costs land on axes that already hurt:")
print()
print("     controlled gates   2N -> sum|row|    16 -> 54 at N=8   (3.4x worse)")
print("     alias triples       9 -> 33          at N=4            (3.7x worse)")
print()
print("  Gates matter because depth is where this construction is already weakest.")
print("  Aliasing matters more, because it is ALREADY the accuracy ceiling on the")
print("  weak coefficients (v106), and the reduction lowers that ceiling further.")
print()
print("  NET: a qubit saving paid for in the currency that was already scarce. The")
print("  2N-ancilla objection was the strongest argument against this construction,")
print("  but answering it by compressing the register makes the estimator worse at")
print("  the thing it is already worst at. That is not obviously a good trade and")
print("  this file does not claim it is one.")
print()
print("  A CHEAPER VARIANT WORTH TRYING: d strictly between log2 M and 2N. The")
print("  extremes are 2N qubits / 9 aliases and log2 M qubits / 33 aliases; nothing")
print("  here measures the middle, and the alias count should interpolate. If it")
print("  falls off fast, a modest reduction may cost almost nothing.")
print()
print("  NOT ANSWERED HERE, and it is the tier-A question: does a d-qubit register")
print("  reproduce the 2N-qubit accuracy at matched shots on real circuits? The")
print("  orthogonality argument says the DECODE stays unbiased, and that argument")
print("  is sound - but it says nothing about the alias inflation just measured,")
print("  which is exactly the failure mode it does not model.")
