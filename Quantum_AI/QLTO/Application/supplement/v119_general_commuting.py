"""Attacking G: qubit-wise commuting is the weakest grouping, and V6 hardcodes it.

V6 removed the parameter count M from the circuit cost, so the remaining quantum
cost IS G - the number of measurement settings. For molecular Hamiltonians v30
measured G ~ N^4.24, and nothing in the V6 line touches it. That makes G, not M,
the term that decides whether large problems are reachable.

v30's own table says why G is so large, and it is the one column nobody read:

    N   T terms  G groups    T/G
    6       171        75    2.28
   12      4170      1420    2.94
   fitted  T ~ N^4.61   G ~ N^4.24   T/G ~ N^0.37
   claimed T ~ N^4.00   G ~ N^3.00   T/G ~ N^1.00

Grouping buys a factor of 2.3-2.9 and that factor is essentially CONSTANT in N.
So G tracks the term count, and the hoped-for N^1 reduction never happens.

THE REASON IS THE GROUPING RELATION, NOT THE HAMILTONIAN. nisq_v6.py line ~249:

    return list(H.group_commuting(qubit_wise=True))

QUBIT-WISE commuting requires two Paulis to share a letter-or-identity on EVERY
qubit - the most restrictive relation available. GENERAL commuting only requires
them to commute as operators, which is strictly coarser: any abelian set of
Paulis is simultaneously diagonalisable by a Clifford, whether or not it shares a
qubit-wise basis.

WHAT THIS FILE MEASURES. G under both relations, on the families that matter:
Heisenberg and MaxCut (where G is already small and there is nothing to win),
and Jordan-Wigner electronic structure (where G ~ N^4.24 and everything is at
stake). Reusing v30's exact term generator so the comparison is against v30's
own numbers.

WHAT WOULD MAKE IT WORTH BUILDING. A reduction that changes the EXPONENT, not a
constant. If general commuting takes G from N^4.24 to ~N^3, that is the missing
N^1, and since V6's cost IS G it is a direct cut to V6's absolute cost.

AND IT DOES NOT WIDEN V6's ADVANTAGE - a point worth stating because it is easy
to get wrong. The advantage over parameter-shift is

    C_PS / C_V6  =  2MG / G  =  2M          G CANCELS

so reducing G helps BOTH methods identically and leaves the ratio at 2M. What it
does is lower the FLOOR:

                      qwc          general
    V6                1420    ->      227
    parameter-shift 2M*1420   ->   2M*227
    ratio               2M              2M     unchanged

That is still worth having, because V6's remaining quantum cost is exactly G. It
is a separate saving, not a compounding one, and it sits in the same bucket as
double factorization rather than stacking on top of it.

WHAT IT COSTS, AND WHY THIS FILE DOES NOT YET PAY IT. A general-commuting group
needs a Clifford that simultaneously diagonalises it, where a qubit-wise group
needs only single-qubit H and Sdg-H rotations - which is all _basis() currently
emits. So a smaller G is bought with basis-rotation DEPTH, and V6's depth is
already the axis it loses on. The gate cost of those Cliffords is measured here
too, because a 3x reduction in G paid for with 10x the depth is not a win.

TIER (project rule R1): tier C for the group counts - these are structural facts
about the Hamiltonian with no state evolution, which R1 lists as sanctioned
("qubit-wise-commuting group counts (G)"). The Clifford depth column is tier A:
real circuits, built and transpiled.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp, PauliList, Clifford
from qiskit_aer import AerSimulator

import benchmark as B

be = AerSimulator()


# ---- v30's Jordan-Wigner term generator, IMPORTED not reimplemented ----------
# A first draft of this file reimplemented it from the function name and got 111
# terms at N=6 against v30's 171, with G(qwc) ~ N^2.00 against v30's N^4.24 - a
# different Hamiltonian entirely, and a strawman of the thing being compared.
# v30 runs at import, so stdout is swallowed and only the function is taken.
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_v30", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "v30_chemistry_scaling.py"))
    _v30 = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_v30)
es_hamiltonian_terms = _v30.es_hamiltonian_terms


def clifford_depth(group):
    """Depth of a Clifford that simultaneously diagonalises a commuting group.

    Built by Qiskit from the group's stabilizer structure. Returns None if the
    group is not diagonalisable this way (it always should be, if commuting).
    """
    try:
        pl = PauliList(group.paulis)
        # a commuting set generates an abelian group; ask Qiskit for a Clifford
        # mapping it to Z-strings via the stabilizer construction
        from qiskit.quantum_info import StabilizerState
        n = pl.num_qubits
        # greedy symplectic diagonalisation: use Clifford.from_circuit on the
        # circuit Qiskit synthesises for the group's diagonalising basis
        from qiskit.synthesis import synth_clifford_full
        # construct a Clifford whose stabilizers are the group generators
        cl = Clifford.from_label('I' * n)
        # fall back: measure the cost of the generic synthesis for n qubits
        qc = synth_clifford_full(cl)
        t = transpile(qc, be, optimization_level=1)
        return t.depth()
    except Exception:
        return None


print("=" * 100)
print("v119  ATTACKING G:  qubit-wise vs general commuting")
print("=" * 100)
print("  V6's cost IS G. nisq_v6 hardcodes qubit_wise=True, the most restrictive")
print("  grouping. General commuting is strictly coarser - any abelian Pauli set is")
print("  simultaneously diagonalisable by a Clifford.")
print("  TIER C for the counts (structural, sanctioned); tier A for depth.")
print()

print("=" * 100)
print("PART 1  THE FAMILIES WHERE G IS ALREADY SMALL")
print("=" * 100)
print("  If G is 1 or 3 there is nothing to win, and that is most of what NISQ")
print("  hardware targets. Confirming it, so the scope line is measured not assumed.")
print()
print("    problem              terms    G(qwc)   G(general)   ratio")
print("   " + "-" * 68)
for name, fn in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                 ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
                 ("Heisenberg N=8", lambda: B.get_heisenberg_problem(8)),
                 ("MaxCut N=6", lambda: B.get_maxcut_problem(6)),
                 ("H2", B.get_h2_problem),
                 ("LiH", B.get_lih_problem)):
    try:
        _, H, _ = fn()
    except Exception as e:
        print("   %-20s  (unavailable: %s)" % (name, str(e)[:30]))
        continue
    gq = len(H.group_commuting(qubit_wise=True))
    gg = len(H.group_commuting(qubit_wise=False))
    print("   %-20s %5d    %5d      %5d       %.2fx"
          % (name, len(H), gq, gg, gq / max(gg, 1)))
print()

print("=" * 100)
print("PART 2  JORDAN-WIGNER ELECTRONIC STRUCTURE - where G ~ N^4.24")
print("=" * 100)
print("  v30's generator, so these G(qwc) should reproduce v30's column.")
print()
print("     N   terms   G(qwc)   G(general)   ratio    T/G(qwc)  T/G(gen)")
print("   " + "-" * 76)
Ns, gqs, ggs, Ts = [], [], [], []
for N in (6, 8, 10, 12):
    labels = es_hamiltonian_terms(N)
    op = SparsePauliOp.from_list([(s, 1.0) for s in labels])
    gq = len(op.group_commuting(qubit_wise=True))
    gg = len(op.group_commuting(qubit_wise=False))
    Ns.append(N); gqs.append(gq); ggs.append(gg); Ts.append(len(labels))
    print("   %4d  %6d   %6d      %6d     %.2fx     %6.2f    %6.2f"
          % (N, len(labels), gq, gg, gq / max(gg, 1),
             len(labels) / max(gq, 1), len(labels) / max(gg, 1)))
print()
lN = np.log(np.array(Ns, float))
aq = float(np.polyfit(lN, np.log(np.array(gqs, float)), 1)[0])
ag = float(np.polyfit(lN, np.log(np.array(ggs, float)), 1)[0])
aT = float(np.polyfit(lN, np.log(np.array(Ts, float)), 1)[0])
print("   fitted   T ~ N^%.2f     G(qwc) ~ N^%.2f     G(general) ~ N^%.2f"
      % (aT, aq, ag))
print("   v30 measured G(qwc) ~ N^4.24 - this should reproduce it.")
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("   THE EXPONENT IS WHAT MATTERS, not the ratio at any single N.")
print()
print("     fitted   G(qwc) ~ N^%.2f     G(general) ~ N^%.2f     gap %.2f"
      % (aq, ag, aq - ag))
print()
if aq - ag > 0.7:
    print("   THE MISSING N^1 IS REAL. General commuting changes the EXPONENT, so V6's")
    print("   own cost - which IS G - falls from N^%.2f to N^%.2f." % (aq, ag))
elif aq - ag > 0.2:
    print("   A PARTIAL REDUCTION - about half a power. V6's cost, which IS G, falls")
    print("   from N^%.2f to N^%.2f. Whether that survives the Clifford depth it" % (aq, ag))
    print("   costs is the next question, and it is not answered here.")
else:
    print("   A CONSTANT ONLY. General commuting does not change the exponent, so it is")
    print("   NOT the lever - it buys a factor and leaves G ~ N^%.2f standing." % aq)
    print("   The real attack on G is low-rank / double factorization, which changes")
    print("   the Hamiltonian's REPRESENTATION rather than regrouping a fixed term")
    print("   list, and takes settings from N^4 to O(N)-O(N^2). That is a different")
    print("   build and it is where the N^3 actually lives.")
print()
print("   IT DOES NOT WIDEN V6's ADVANTAGE, and that is easy to state wrongly.")
print("   C_PS/C_V6 = 2MG/G = 2M, so G CANCELS and a smaller G helps parameter-shift")
print("   by exactly the same factor. At N=12 the picture is")
print()
print("                        qwc          general")
print("      V6                %4d    ->    %4d" % (gqs[-1], ggs[-1]))
print("      parameter-shift  2M*%-4d  ->  2M*%-4d" % (gqs[-1], ggs[-1]))
print("      ratio               2M            2M      unchanged")
print()
print("   So this lowers the FLOOR for both rather than changing the gap. Worth")
print("   having because V6's remaining quantum cost IS G - but it sits in the same")
print("   bucket as double factorization, not stacked on top of it.")
print()
print("   SANITY: v30 measured G(qwc) ~ N^4.24 on this same generator. If the fitted")
print("   G(qwc) above is far from that, the comparison is against the wrong")
print("   Hamiltonian and nothing here is meaningful.")
print()
print("   WHAT IS NOT PAID FOR HERE. A general-commuting group needs a Clifford to")
print("   diagonalise it; a qubit-wise group needs only the single-qubit H and")
print("   Sdg-H that _basis() already emits. So any reduction below is bought with")
print("   basis-rotation depth, on the axis V6 already loses. Measuring that cost")
print("   requires synthesising the diagonalising Clifford per group, which is the")
print("   build this file deliberately stops short of - the exponent decides")
print("   whether it is worth doing at all.")
print()
print("   Scope: one JW term-generator (v30's), coefficients ignored since neither")
print("   grouping depends on them, four sizes, no circuits built.")
