"""Is the data prep one-time? No - and that decides the whole training claim.

TIER C - NO CIRCUIT. Derivation with hardware numbers.

THE ARGUMENT UNDER TEST. "Classical must run through all the data every epoch,
which is why a frontier run needs a $100M data centre. If the quantum side
prepares the data ONCE, it does not have that limit and wins exponentially."

The first half is right. The second half fails on one physical fact:

    A QUANTUM STATE IS DESTROYED BY MEASUREMENT.

Every shot collapses the register, so the Theta(|D|) state preparation runs
AGAIN for every shot. It is amortised over nothing. And because the weights
w_x = l'(f_x, y_x) depend on the CURRENT residuals, the prep angles change every
epoch, so even the compiled circuit is not reused.

    prep cost   Theta(|D|) gates  PER SHOT, not per epoch, not once

So the honest per-epoch comparison is operations x time-per-operation, and the
two factors pull in opposite directions:

    OPERATION COUNT favours quantum
        classical   |D| x M'        every sample x every parameter
        quantum     c_ep x S x (|D| + M)   circuits x shots x gates

    TIME PER OPERATION favours classical, by nine orders of magnitude
        FLOP                 ~1e-15 s   (1e15 FLOP/s)
        logical quantum gate ~1e-6  s   (error-corrected, distillation-bound)

The question is which factor is bigger, and it is not close.
"""
import sys
import numpy as np

C_EP = 9.9
F = 1e15
T_GATE = 1e-6
T_FLOP = 1.0 / F


def part1():
    print("PART 1  ONE EPOCH, IN SECONDS. kappa = 100, M = 1e6, S shots.")
    print("")
    print("   %10s %8s %16s %16s %14s"
          % ("|D|", "shots", "quantum (s)", "classical (s)", "verdict"))
    kappa, M = 100, 1e6
    for D in (1e9, 1e12):
        for S in (1e4, 1e2):
            q = C_EP * S * (D + M) * T_GATE
            c = D * kappa * M * T_FLOP
            v = "quantum" if q < c else "classical %.3gx" % (q / c)
            print("   %10.0e %8.0e %16.4g %16.4g %14s" % (D, S, q, c, v))
    print("")
    print("   The operation COUNT ratio favours quantum by ~kappa*M/(c_ep*S),")
    print("   but each quantum operation costs 1e9 times more seconds. The")
    print("   speed gap wins.")
    print("")


def part2():
    print("PART 2  WHAT WOULD HAVE TO BE TRUE. Solve for each factor alone.")
    kappa, M, D, S = 100, 1e6, 1e12, 1e2
    q = C_EP * S * (D + M) * T_GATE
    c = D * kappa * M * T_FLOP
    short = q / c
    print("        |D|=1e12  M=1e6  kappa=100  S=1e2 (AE readout)")
    print("        quantum %.4g s   classical %.4g s   SHORTFALL %.3gx"
          % (q, c, short))
    print("")
    print("   %-26s %14s %16s" % ("factor", "now", "needed alone"))
    print("   %-26s %14.3g %16.3g" % ("logical gate time (s)", T_GATE,
                                      T_GATE / short))
    print("   %-26s %14.3g %16.3g" % ("shots S", S, S / short))
    print("   %-26s %14.3g %16.3g" % ("Abbas ratio kappa", kappa,
                                      kappa * short))
    print("   %-26s %14.3g %16.3g" % ("prep gates (frac of |D|)", 1.0,
                                      1.0 / short))
    print("")
    print("   The gate-time row is the one that matters: 1e-6 s would have to")
    print("   become %.1e s, which is BELOW a physical superconducting gate"
          % (T_GATE / short))
    print("   (1e-8 s) let alone an error-corrected logical one. No algorithm")
    print("   reaches it, because the quantum side is already down to ONE")
    print("   circuit family - there is nothing left to remove.")
    print("")


def part3():
    print("PART 3  THE ONLY ESCAPE, AND ITS PRICE.")
    print("")
    print("   The Theta(|D|) prep is per-shot because measurement destroys the")
    print("   state. The one construction that would amortise it is QRAM: a")
    print("   device answering |x> -> |x>|w_x> in O(log|D|) TIME.")
    print("")
    print("   QRAM's own cost is the known objection and it is not resolved by")
    print("   this project: the bucket-brigade architecture needs Theta(|D|)")
    print("   PHYSICAL COMPONENTS, all of which must be maintained coherently")
    print("   for the duration of the query. The Theta(|D|) does not vanish -")
    print("   it moves from circuit DEPTH into HARDWARE COUNT, which is the")
    print("   same relocation Part V found for QPE (count -> depth) and the")
    print("   same one Part XXV found for routing (width -> routing).")
    print("")
    print("   WITH a working QRAM the ledger changes completely:")
    kappa, M, D, S = 100, 1e6, 1e12, 1e2
    q_now = C_EP * S * (D + M) * T_GATE
    q_qram = C_EP * S * (np.log2(D) + M) * T_GATE
    c = D * kappa * M * T_FLOP
    print("       quantum, prep per shot   %.4g s" % q_now)
    print("       quantum, with QRAM       %.4g s" % q_qram)
    print("       classical                %.4g s" % c)
    print("       QRAM verdict: %s"
          % ("QUANTUM CHEAPER by %.3gx" % (c / q_qram) if q_qram < c
             else "classical still %.3gx" % (q_qram / c)))
    print("")
    print("   So the exponential claim is TRUE CONDITIONAL ON QRAM and false")
    print("   without it. That is the honest statement, and QRAM is an")
    print("   unsolved hardware problem, not a QLTO one.")
    print("")


def part4():
    print("PART 4  WHAT THIS DOES NOT TOUCH.")
    print("")
    print("   Everything above is about loading CLASSICAL data. It says")
    print("   nothing about the case where the data is ALREADY QUANTUM -")
    print("   states from an experiment, a sensor, a chemistry simulation.")
    print("   There is no Theta(|D|) prep because there is nothing to load,")
    print("   the classical column does not exist at any budget, and QLTO's")
    print("   parameter-factor saving applies with no offsetting cost.")
    print("")
    print("   THE TWO REGIMES, kept separate:")
    print("     classical data   Theta(|D|) prep per shot dominates. Quantum")
    print("                      loses by ~1e5x and needs QRAM to compete.")
    print("     quantum data     no prep. QLTO trains a model no classical")
    print("                      machine can evaluate, at O(1) circuits per")
    print("                      epoch and kappa*M/3 fewer operations.")
    print("")
    print("   A frontier LLM is classical data. A chemistry surrogate or a")
    print("   sensor-data model is not. The $100M data centre is not replaced")
    print("   by this; the calculation nobody can do at all is.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("TIER C - NO CIRCUIT. Derivation with hardware numbers.")
    print("")
    want = sys.argv[1:] or ["1", "2", "3", "4"]
    for k, fn in (("1", part1), ("2", part2), ("3", part3), ("4", part4)):
        if k in want:
            fn()
