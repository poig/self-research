"""Where a COMPLEXITY separation can and cannot live for QLTO. Derivation.

TIER C - NO CIRCUIT. Every route below is settled by argument, not measurement,
which is what R6 requires of a question with a closed form.

THE QUESTION. Not "is QLTO cheap" - that is a constant-factor claim and it is
settled. Is there a task where QLTO is separated from every classical algorithm
by more than a constant?

FIVE ROUTES, CHECKED. Four fail, and each for a stated reason, so the search
terminates rather than continuing indefinitely.

  1. GRADIENT ESTIMATION ITSELF.  FAILS.
     Computing a gradient of an M-parameter function costs O(M) classical
     evaluations. There is no hidden hardness to exploit - you cannot get an
     exponential separation on a problem whose classical version is polynomial.
     This is Part XVIII's argument and it is not escapable by a better estimator.

  2. JUNTA / SPARSE LANDSCAPE (Bernstein-Vazirani shaped).  FAILS.
     The hope: if E depends on only k << M parameters, find which ones.
     BV gets a parity oracle's whole n-bit string in ONE query against n
     classical. But our phase is e^{i gam E_d} with E_d = R sum_j sigma_j(d) g_j
     - a WEIGHTED sum, not a parity - so the transform gives amplitudes
     proportional to g_j and a measurement samples ONE coordinate, not all M.
     Recovering k coordinates needs O(k log k) samples by coupon collector.
     And the classical baseline is NOT Omega(M): adaptive GROUP TESTING finds k
     defectives among M in O(k log(M/k)) queries. Quantum O(k log k) against
     classical O(k log(M/k)) is a LOG FACTOR, not a separation.

  3. HIDDEN SUBGROUP.  FAILS.
     Shor works because factoring reduces to a hidden subgroup problem that is
     classically hard; the QFT is not the source of the advantage, the hardness
     of the recovered structure is. Here the group is Z_2^k and the recovered
     structure is a DERIVATIVE. There is no hidden subgroup and nothing to hide.

  4. LEARNING FROM EXPERIMENTS, on a model we prepare.  FAILS.
     Huang, Broughton, Cotler, Chen, Li, Mohseni, Chen, Babbush, Kueng, Preskill
     & McClean, Science 2022, prove an EXPONENTIAL separation in the number of
     EXPERIMENTS between algorithms with and without quantum memory. QLTO's
     two-copy construction (shared data register, or SWAP test) IS an entangled
     measurement across copies, so the primitive matches. But their premise is an
     UNKNOWN state given as copies. |psi(theta)> is prepared from a circuit we
     control and can re-prepare at will, so the premise fails and the separation
     does not transfer.

  5. LEARNING FROM QUANTUM DATA.  THIS ONE SURVIVES.
     Change the input, not the estimator. If the model's INPUT is a quantum
     state that arrives from an experiment - a sensor, a chemistry simulation, a
     physical process - then:

         the state is GIVEN, not prepared          -> Huang's premise holds
         there is no classical description of it   -> no classical baseline
         the two-copy construction is quantum
           memory in exactly their sense           -> the primitive is built

     and the separation is theirs, proven, exponential in the number of
     experiments. QLTO supplies the training loop that consumes it.

WHAT CHANGES ABOUT THE PROJECT IF ROUTE 5 IS THE TARGET.

    the DATA REGISTER is not needed          the data is already quantum; there
                                             is nothing to amplitude-encode, and
                                             the Theta(|D|) prep that dominated
                                             every ledger disappears
    the TWO-COPY primitive becomes central   it is the quantum-memory element,
                                             not a curiosity for Gauss-Newton
    the DESIGN REGISTER keeps its role       O(1) circuits for the gradient of a
                                             loss defined on those states

THE HONEST SHAPE OF THE CLAIM:

    QLTO has no complexity separation for training on CLASSICAL data, and the
    reason is structural rather than a limit of the construction: derivative
    estimation has no classical hardness to exploit.

    On QUANTUM data the separation is exponential in experiment count, it is
    proven (Science 2022) rather than conjectured, and QLTO's two-copy
    construction is the primitive that realises it.

WHAT IS NOT YET DONE, and it is the whole remaining task: no loss function on
quantum data has been written down here, no experiment designed, and the
composition of Huang's protocol with the design register has not been checked.
This file establishes WHERE to build, not that it is built.
"""
import sys


def part1():
    print("PART 1  THE FIVE ROUTES, AND THEIR VERDICTS.")
    print("")
    rows = [
        ("gradient estimation", "FAILS",
         "classical baseline is O(M) - no hardness to exploit"),
        ("junta / Bernstein-Vazirani", "FAILS",
         "weighted sum is not a parity; classical group testing is O(k log M)"),
        ("hidden subgroup", "FAILS",
         "the recovered structure is a derivative, not a period"),
        ("learning from experiments,", "FAILS",
         "premise needs an UNKNOWN state; we prepare our own"),
        ("  model we prepare", "", ""),
        ("LEARNING FROM QUANTUM DATA", "SURVIVES",
         "state is given, no classical description, two-copy = quantum memory"),
    ]
    for name, verdict, why in rows:
        print("   %-28s %-10s %s" % (name, verdict, why))
    print("")


def part2():
    print("PART 2  WHY ROUTE 5 IS DIFFERENT IN KIND.")
    print("")
    print("   Routes 1-4 all try to find hardness in the ESTIMATOR - a better")
    print("   way to read a gradient. That cannot work: reading a gradient of a")
    print("   function you can evaluate is polynomial classically, so no")
    print("   estimator separates by more than a constant. Part XVIII proved")
    print("   this and every subsequent attempt rediscovered it.")
    print("")
    print("   Route 5 changes the INPUT instead. The hardness is not in how we")
    print("   read the gradient - it is in the fact that the thing being")
    print("   learned from has no efficient classical description at all.")
    print("   Huang et al. 2022 proved the separation there:")
    print("")
    print("     without quantum memory   Omega(2^n) experiments")
    print("     with quantum memory      O(n) experiments")
    print("")
    print("   and QLTO's two-copy construction - two system copies measured")
    print("   jointly, verified to 1e-11 for Gauss-Newton and to O(R^2) for the")
    print("   QFIM - is quantum memory in exactly their sense.")
    print("")


def part3():
    print("PART 3  WHAT TO BUILD, in order.")
    print("")
    print("   1. A LOSS ON QUANTUM DATA. The pipeline currently assumes")
    print("      f_x = <psi(theta,x)|O|psi(theta,x)> with x a classical index.")
    print("      Replace x by a GIVEN state rho_i and the loss by a functional")
    print("      of rho_i and the model - the natural first case is")
    print("      discriminating or regressing on properties of rho_i.")
    print("")
    print("   2. DROP THE DATA REGISTER. Nothing to amplitude-encode, so the")
    print("      Theta(|D|) prep that dominated every cost ledger today simply")
    print("      is not incurred. This removes the single largest term.")
    print("")
    print("   3. COMPOSE Huang's two-copy measurement with the design register.")
    print("      Both are built and verified separately; the composition is not")
    print("      checked and is the load-bearing unknown.")
    print("")
    print("   4. ONLY THEN a resource ledger. Every ledger written today")
    print("      priced classical data and was dominated by the prep term.")
    print("      On quantum data that term is absent and the arithmetic is")
    print("      different from the ground up.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("TIER C - NO CIRCUIT. Derivation.")
    print("")
    want = sys.argv[1:] or ["1", "2", "3"]
    for k, fn in (("1", part1), ("2", part2), ("3", part3)):
        if k in want:
            fn()
