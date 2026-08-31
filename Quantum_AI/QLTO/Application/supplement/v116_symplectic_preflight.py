"""One GF(2) computation that predicts two obstructions, before any circuit is built.

Two separate walls were hit in this session, several experiments apart, and they
turn out to be the same algebraic fact.

  v106  DEGREE-2 ALIASING. sigma_j sigma_k is itself a Walsh character - the one
        indexed by v_j + v_k - so whenever v_j + v_k = v_m for a term m already
        in the set, a degree-2 effect is indistinguishable from term m's degree-1
        effect. Measured for crosstalk: v_XX + v_YY = v_ZZ on every bond, and the
        weak ZZ coefficients are exactly the ones the estimator recovers worst.

  v115  NO TWIRL TIME REVERSAL. The band-limited derivative kernel of
        arXiv:2606.19486 is ODD, so half-range sampling tau >= 0 is valid only if
        the correlator is odd; otherwise negative evolution times are required.
        A Pauli Q with Q H Q^dag = -H would supply them for free, since then
        Q e^{-iHt} Q^dag = e^{+iHt}. For crosstalk no such Q exists.

THE COMMON CAUSE. Write v_k in F_2^{2N} for the symplectic vector of P_k and let
V be the M x 2N matrix of them. Conjugation by Q(a,b) flips the sign of term k
exactly when (V w)_k = 1, with w = (a,b).

    ALIASING          v_j + v_k = v_m  =>  sigma_j sigma_k = sigma_m
    TIME REVERSAL     needs V w = 1 (all ones) to be solvable

and these interact. If some subset S has sum_{k in S} v_k = 0, then for ANY w

    sum_{k in S} (V w)_k = (sum_{k in S} v_k) . w = 0

so the number of sign flips inside S is always EVEN. If |S| is ODD, all-minus is
unreachable and no time-reversing twirl exists. The same null combinations are
the aliasing relations. One dependency structure, both obstructions.

AND M > 2N FORCES IT. rank(V) <= 2N, so M > 2N guarantees null combinations
exist. Whether an ODD one exists is what decides the time-reversal question, and
that is what this check computes.

WHAT THIS IS FOR. A pre-flight test. Given a term set, before writing any
circuit, it says: will a degree-1 Walsh decode be confounded, and is the cheap
time-reversal fix available. This session lost two builds to questions this
answers in milliseconds.

TIER (project rule R1): tier C, and legitimately so - this is a structural fact
about operators with no state evolution, which R1 lists explicitly as sanctioned
NumPy use ("commutant rank, symplectic vectors, DLA dimension"). No circuit, no
shots, no accuracy claim.
"""
import sys, os, itertools
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.quantum_info import Pauli
from twirl_cal import crosstalk_terms


# ---- GF(2) linear algebra ----------------------------------------------------
def _rref(A):
    """Row-reduce over GF(2). Returns (R, pivots)."""
    R = A.copy() % 2
    rows, cols = R.shape
    piv, r = [], 0
    for c in range(cols):
        p = None
        for i in range(r, rows):
            if R[i, c]:
                p = i
                break
        if p is None:
            continue
        R[[r, p]] = R[[p, r]]
        for i in range(rows):
            if i != r and R[i, c]:
                R[i] = (R[i] + R[r]) % 2
        piv.append(c)
        r += 1
        if r == rows:
            break
    return R, piv


def _solve(A, b):
    """Solve A w = b over GF(2). Returns w or None."""
    A = A % 2
    b = b % 2
    aug = np.concatenate([A, b.reshape(-1, 1)], axis=1) % 2
    R, piv = _rref(aug)
    n = A.shape[1]
    for i in range(R.shape[0]):                 # inconsistent row 0...0 | 1
        if not R[i, :n].any() and R[i, n]:
            return None
    w = np.zeros(n, dtype=int)
    for i, c in enumerate(piv):
        if c < n:
            w[c] = R[i, n]
    return w


def symplectic_matrix(terms):
    Z = np.array([Pauli(t).z.astype(int) for t in terms])
    X = np.array([Pauli(t).x.astype(int) for t in terms])
    return np.concatenate([Z, X], axis=1) % 2


def preflight(terms, max_subset=4):
    """Pre-flight report for a twirl-multiplexed degree-1 Walsh decode.

    Returns a dict with:
      rank           rank(V) over GF(2)
      forced_dep     True if M > rank, i.e. null combinations must exist
      alias_triples  [(i,j,k)] with v_i + v_j = v_k  - degree-2 aliases degree-1
      odd_nulls      odd-size subsets S with sum_{k in S} v_k = 0, up to
                     max_subset - each one blocks time reversal
      time_reversal  a w with V w = 1 if one exists, else None
    """
    V = symplectic_matrix(terms)
    M = V.shape[0]
    rank = int(np.linalg.matrix_rank(V))         # over R; refined below
    R, piv = _rref(V)
    rank = len(piv)

    alias = []
    for i in range(M):
        for j in range(i + 1, M):
            s = (V[i] + V[j]) % 2
            for k in range(M):
                if k not in (i, j) and np.array_equal(s, V[k]):
                    alias.append((i, j, k))

    odd_nulls = []
    for size in range(3, max_subset + 1, 2):
        for S in itertools.combinations(range(M), size):
            if not (V[list(S)].sum(axis=0) % 2).any():
                odd_nulls.append(S)

    w = _solve(V, np.ones(M, dtype=int))
    return {'V': V, 'M': M, 'rank': rank, 'forced_dep': M > rank,
            'alias_triples': alias, 'odd_nulls': odd_nulls, 'time_reversal': w}


def report(terms, name):
    r = preflight(terms)
    M, V = r['M'], r['V']
    N = len(terms[0])
    print("  %-22s N=%d  M=%-3d rank(V)=%-2d  2N=%d" % (name, N, M, r['rank'], 2 * N))
    print("     dependencies forced by M > rank : %s" % r['forced_dep'])
    print("     degree-2 aliasing triples       : %d%s"
          % (len(r['alias_triples']) // 2,
             ("   e.g. %s + %s = %s" % (terms[r['alias_triples'][0][0]],
                                        terms[r['alias_triples'][0][1]],
                                        terms[r['alias_triples'][0][2]]))
             if r['alias_triples'] else ""))
    print("     odd null subsets (block Q)      : %d" % len(r['odd_nulls']))
    if r['time_reversal'] is not None:
        w = r['time_reversal']
        a, b = w[:N], w[N:]
        chk = np.all((V @ w) % 2 == 1)
        print("     time-reversing twirl Q          : YES  a=%s b=%s  verified=%s"
              % (''.join(map(str, a)), ''.join(map(str, b)), chk))
        print("     -> kernel half-range sampling is AVAILABLE for free")
    else:
        print("     time-reversing twirl Q          : NO")
        print("     -> kernel needs the cross-Pauli readout; no cheap fix")
    print()


print("=" * 96)
print("v116  SYMPLECTIC PRE-FLIGHT:  one GF(2) check, two obstructions")
print("=" * 96)
print("  For a twirl-multiplexed degree-1 Walsh decode, the term set alone decides")
print("  whether the decode is confounded and whether time reversal is free.")
print("  TIER C: operator structure, no state evolution. R1-sanctioned.")
print()

print("=" * 96)
print("THE FAMILIES")
print("=" * 96)
for N in (3, 4, 5):
    report(crosstalk_terms(N), "crosstalk N=%d" % N)

report([''.join('Z' if q == i else 'I' for q in range(3)) for i in range(3)],
       "single-Z N=3")
report(['ZII', 'IZI', 'IIZ', 'XII', 'IXI'], "mixed independent")
report(['ZZI', 'IZZ', 'ZIZ'], "ZZ ring N=3")
report(['XII', 'IXI', 'IIX', 'ZZI', 'IZZ'], "TFIM-like N=3")

print("=" * 96)
print("READING IT")
print("=" * 96)
print("  THE RULE. If any ODD-size subset of terms has symplectic vectors summing")
print("  to zero, then the flip parity over that subset is fixed at even for every")
print("  twirl - so all-minus is unreachable, no Q reverses time, and the same null")
print("  combination confounds a degree-2 product with a degree-1 effect.")
print()
print("  crosstalk fails on both counts at every N, and for one reason:")
print("     v(ZZ) + v(XX) + v(YY) = 0   on every bond")
print("  The XX/YY/ZZ triple is the whole story - it caps the weak ZZ coefficients")
print("  (v106) and blocks the free kernel fix (v115) simultaneously.")
print()
print("  USE IT AS A PRE-FLIGHT. Before building a twirl-multiplexed estimator for")
print("  a new term set:")
print("     odd_nulls empty  -> decode is clean at degree 2 AND time reversal is")
print("                         free, so the band-limited kernel drops straight in")
print("     odd_nulls found  -> expect a weak-coefficient ceiling, and budget for")
print("                         the cross-Pauli restructure rather than a patch")
print()
print("  This session lost two builds to questions answered here in milliseconds.")
print()
print("  Scope: alias search is over PAIRS and odd nulls up to size 4. Larger even")
print("  dependencies exist whenever M > rank and are not enumerated - they affect")
print("  higher-degree confounding, not the two obstructions above.")
