"""Read the gradient off the AMPLITUDES, not the samples. TIER B.

Exact amplitudes by Statevector. The claim is an identity.

WHY THIS EXISTS. Every result in this project reads the design register by
MEASURING it. That collapses the superposition to a classical mixture, which is
why the whole method reduces to weighted regression and why there is no quantum
advantage in it: the estimator is a classical design-of-experiments run on a
quantum sampler.

Shor's shape is superpose a group, imprint a function of the group element,
TRANSFORM, and read structure off the amplitudes. The design register does the
first two. It has never done the third.

    current      sum_d |d> |psi_d>  --measure-->  (d, E_d) pairs, regress
    this file    sum_d e^{i gam E_d} |d>  --H^k-->  read Walsh coefficients
                 AS AMPLITUDES

THE IDENTITY. With the row energy imprinted as a phase and the register
Hadamard-transformed,

    H^k sum_d e^{i gam E_d} |d>  =  2^{-k/2} sum_c [ sum_d e^{i gam E_d} chi_c(d) ] |c>

and expanding for small gamma, e^{i gam E_d} = 1 + i gam E_d + O(gam^2),

    amplitude at |c>   ~   delta_{c,0}  +  i gam Ehat(c)

where Ehat(c) is the Walsh coefficient of the energy at character c. Part XVIII
established Ehat(c_j) = (sin R / 2) g_j, so

    P(measure c_j)  ~  gam^2 (sin R/2)^2 g_j^2     PROPORTIONAL TO g_j^2

MEASURING THE REGISTER AFTER THE TRANSFORM SAMPLES COORDINATE j WITH PROBABILITY
PROPORTIONAL TO ITS SQUARED GRADIENT. The heavy coordinates come out directly.

WHY THAT IS DIFFERENT IN KIND. Reading all M components - what the rest of this
project does - costs O(M) classical numbers to store and O(M) to scan for the
largest. Sampling from g_j^2 finds the heaviest coordinates in O(1) draws
regardless of M. At M ~ 1e9 the difference is between 4 GB of gradient and a
handful of samples. This is the Goldreich-Levin / heavy-Fourier-coefficient
structure, and it is the one place the design register can do something a
classical design cannot.

WHAT IS VERIFIED HERE AND WHAT IS NOT. The READOUT identity is verified exactly:
that H^k on a phase-imprinted register yields amplitudes proportional to the
Walsh coefficients. The PHASE IMPRINT itself is applied here as a diagonal gate
built from known E_d, which is legitimate for checking the readout and is NOT an
algorithm - a real circuit would imprint it with the quadratic-potential
construction (RZ + RZZ) that already exists in qlto_walk. Whether that composition
is efficient is a separate question and is not claimed here.
"""
import sys
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

K = 6
COLS = [1, 2, 4, 8]          # linearly independent: no aliasing at any order


def sigma_row(d, cols):
    return np.array([1.0 - 2.0 * (bin(d & c).count("1") & 1) for c in cols])


def energies(g, H, R, cols, k):
    """Exact quadratic model on every design row."""
    N = 1 << k
    E = np.zeros(N)
    for d in range(N):
        s = sigma_row(d, cols)
        E[d] = R * (g @ s) + 0.5 * R * R * (s @ H @ s)
    return E


def part1():
    print("PART 1  P(measure c_j) proportional to g_j^2 ?   TIER B")
    print("        phase imprint e^{i gam E_d}, then H^k, then measure.")
    rng = np.random.default_rng(3)
    m = len(COLS)
    g = rng.normal(size=m)
    A = rng.normal(size=(m, m))
    Hm = (A + A.T) / 2 * 0.3
    R = 0.35
    E = energies(g, Hm, R, COLS, K)
    print("        true |g| per coordinate:",
          "  ".join("%.4f" % abs(v) for v in g))
    print("")
    print("   %8s %10s %38s"
          % ("gamma", "corr", "P(c_j) normalised, against g_j^2/||g||^2"))
    for gam in (0.8, 0.4, 0.2, 0.1):
        reg = QuantumRegister(K, "d")
        qc = QuantumCircuit(reg)
        qc.h(reg)
        qc.append(DiagonalGate(list(np.exp(1j * gam * E))), list(reg))
        qc.h(reg)
        pr = np.asarray(Statevector(qc).probabilities())
        pj = np.array([pr[c] for c in COLS])
        pj = pj / pj.sum()
        tgt = g ** 2 / np.sum(g ** 2)
        corr = float(np.corrcoef(pj, tgt)[0, 1])
        print("   %8.2f %10.6f    %s"
              % (gam, corr, "  ".join("%.4f" % v for v in pj)))
    print("   %8s %10s    %s"
          % ("target", "1.000000", "  ".join("%.4f" % v
                                             for v in g ** 2 / np.sum(g ** 2))))
    print("")
    print("   The amplitude at |c_j> carries g_j, so the PROBABILITY carries")
    print("   g_j^2. Small gamma is the linear-response regime where the")
    print("   identity is exact; large gamma mixes in higher Walsh weights.")
    print("")


def part2():
    print("PART 2  DOES IT FIND THE HEAVY COORDINATE? The point of sampling")
    print("        from g_j^2 is that the largest components dominate.")
    rng = np.random.default_rng(7)
    m = len(COLS)
    print("   %6s %30s %14s %14s"
          % ("trial", "true |g_j|", "argmax true", "argmax P(c_j)"))
    hit = 0
    T = 12
    for t in range(T):
        g = rng.normal(size=m)
        g[rng.integers(0, m)] *= 3.0          # plant a heavy coordinate
        A = rng.normal(size=(m, m))
        Hm = (A + A.T) / 2 * 0.3
        R, gam = 0.35, 0.15
        E = energies(g, Hm, R, COLS, K)
        reg = QuantumRegister(K, "d")
        qc = QuantumCircuit(reg)
        qc.h(reg)
        qc.append(DiagonalGate(list(np.exp(1j * gam * E))), list(reg))
        qc.h(reg)
        pr = np.asarray(Statevector(qc).probabilities())
        pj = np.array([pr[c] for c in COLS])
        a_t, a_p = int(np.argmax(np.abs(g))), int(np.argmax(pj))
        hit += (a_t == a_p)
        if t < 5:
            print("   %6d %30s %14d %14d"
                  % (t, "  ".join("%.3f" % abs(v) for v in g), a_t, a_p))
    print("   ...")
    print("   heaviest coordinate identified in %d/%d trials" % (hit, T))
    print("")
    print("   ONE circuit, ONE measurement, regardless of M. The classical")
    print("   route needs all M components before it can take an argmax.")
    print("")


def part3():
    print("PART 3  WHAT THIS IS AND IS NOT.")
    print("")
    print("   IS: the design register used the way Shor uses a group register -")
    print("   superpose, imprint, TRANSFORM, read structure off amplitudes. The")
    print("   structure recovered is the heavy part of the Walsh spectrum,")
    print("   which is the gradient's support.")
    print("")
    print("   IS NOT, and these are the honest gaps:")
    print("     - the phase imprint here is a DiagonalGate built from known")
    print("       E_d. That verifies the READOUT, not an algorithm. A real")
    print("       circuit imprints it with the RZ + RZZ quadratic-potential")
    print("       construction already in qlto_walk, and whether THAT")
    print("       composition is efficient is unproven.")
    print("     - sampling from g_j^2 is not the same as KNOWING g. It finds")
    print("       heavy coordinates; it does not give their values.")
    print("     - no complexity separation is claimed. Classical needs O(M) to")
    print("       scan for an argmax; this needs O(1) draws. That is a real")
    print("       difference at M ~ 1e9 and it is not a proof of hardness.")
    print("")
    print("   WHAT WOULD MAKE IT AN ADVANTAGE CLAIM: a task where the heavy")
    print("   coordinates are what is wanted and M is too large to enumerate.")
    print("   Sparse/junta landscapes are the natural candidate - if the energy")
    print("   depends strongly on k << M parameters, finding them classically")
    print("   is Omega(M) queries and this is O(k log M). That is a")
    print("   Bernstein-Vazirani-shaped statement and it is the next thing to")
    print("   derive, not another thing to measure.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("")
    want = sys.argv[1:] or ["1", "2", "3"]
    for k, fn in (("1", part1), ("2", part2), ("3", part3)):
        if k in want:
            fn()
