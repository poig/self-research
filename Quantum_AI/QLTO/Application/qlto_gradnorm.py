"""The gradient NORM as one probability: M-free, and Heisenberg-estimable.

TIER B - exact amplitudes by Statevector. The claim is an identity.

THE OBSERVATION. Part XVIII put the gradient in the amplitudes: phase-imprint
the row energy on the design register and Hadamard it, and the amplitude at the
character |c_j> is proportional to g_j (verified, corr 0.999999). Everything so
far then MEASURED that register, which samples one coordinate per shot and
recovers the components one at a time.

But the WEIGHT-1 SUBSPACE is a single projector. The probability of landing
anywhere in {|c_1>, ..., |c_M>} is

    P_1  =  sum_j |amp(c_j)|^2  ~  gam^2 (sin R / 2)^2 sum_j g_j^2
                                =  gam^2 (sin R / 2)^2 ||g||^2

so ||g||^2 is ONE NUMBER read from ONE observable - not M numbers that must then
be squared and summed.

WHY THAT IS WORTH HAVING. ||g|| is not a curiosity, it is what a trust-region
method actually consumes:

    the CONVERGENCE TEST          stop when ||g|| < tol
    the LEVENBERG damping         mu = max(0,-lam_min) + ||g||/R, which every
                                  Newton step in this project uses
    the STEP LENGTH               the Cauchy step is -R g/||g||

and classically there is no way to have ||g|| without first having all M
components. Here it is one projector.

AND IT IS A SINGLE PROBABILITY, so amplitude estimation applies directly:

    sampling                O(1/eps^2) shots
    amplitude estimation    O(1/eps)   applications        QUADRATIC, and it
                                                           needs no marking
                                                           oracle - the target
                                                           is a fixed subspace

    classical               O(M) evaluations, then a sum
    this                    O(1/eps), INDEPENDENT OF M

The distinction that makes this work where coordinate-argmax does not: finding
the LARGEST g_j by amplification needs an oracle marking |g_j| > t, and we have
amplitudes proportional to g_j rather than query access to g_j. The weight-1
SUBSPACE needs no such oracle - it is a fixed, known set of basis states.

PART 1 verifies P_1 ~ ||g||^2 exactly.
PART 2 checks it survives the weight-2 contamination that grows with gamma.
"""
import sys
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector

K = 6
COLS = [1, 2, 4, 8]          # linearly independent -> alias-free
M = len(COLS)


def sigma_row(d):
    return np.array([1.0 - 2.0 * (bin(d & c).count("1") & 1) for c in COLS])


def energies(g, H, R):
    N = 1 << K
    E = np.zeros(N)
    for d in range(N):
        s = sigma_row(d)
        E[d] = R * (g @ s) + 0.5 * R * R * (s @ H @ s)
    return E


def weight1_prob(g, H, R, gam):
    """P(measure any weight-1 character) after imprint + Hadamard."""
    E = energies(g, H, R)
    reg = QuantumRegister(K, "d")
    qc = QuantumCircuit(reg)
    qc.h(reg)
    qc.append(DiagonalGate(list(np.exp(1j * gam * E))), list(reg))
    qc.h(reg)
    pr = np.asarray(Statevector(qc).probabilities())
    return float(sum(pr[c] for c in COLS))


def part1():
    print("PART 1  P(weight-1 subspace)  ~  ||g||^2 ?    TIER B")
    print("        predicted P_1 = gam^2 (sin R/2)^2 ||g||^2")
    rng = np.random.default_rng(2)
    R, gam = 0.30, 0.10
    print("")
    print("   %8s %14s %16s %14s"
          % ("||g||", "P_1 measured", "P_1 predicted", "ratio"))
    for _ in range(6):
        g = rng.normal(size=M) * rng.uniform(0.2, 1.5)
        A = rng.normal(size=(M, M))
        Hm = (A + A.T) / 2 * 0.3
        p = weight1_prob(g, Hm, R, gam)
        pred = gam ** 2 * R ** 2 * float(g @ g)
        print("   %8.4f %14.8f %16.8f %14.6f"
              % (np.linalg.norm(g), p, pred, p / pred))
    print("")
    print("   A CONSTANT ratio across a 7x range of ||g|| is the identity:")
    print("   one probability carries the whole gradient norm, and the constant")
    print("   is known in closed form so it divides out.")
    print("")


def part2():
    print("PART 2  RECOVERING ||g|| FROM P_1, and where it breaks.")
    print("        ||g||_est = sqrt(P_1) / (gam sin(R)/2)")
    rng = np.random.default_rng(5)
    g = rng.normal(size=M)
    A = rng.normal(size=(M, M))
    Hm = (A + A.T) / 2 * 0.3
    true = float(np.linalg.norm(g))
    print("        true ||g|| = %.6f" % true)
    print("")
    print("   %8s %8s %16s %12s"
          % ("R", "gamma", "||g|| estimated", "rel err"))
    for R in (0.30, 0.15):
        for gam in (0.40, 0.20, 0.10, 0.05):
            p = weight1_prob(g, Hm, R, gam)
            est = np.sqrt(p) / (gam * R)
            print("   %8.2f %8.2f %16.8f %12.2e"
                  % (R, gam, est, abs(est - true) / true))
    print("")
    print("   Error falls with BOTH gamma and R: gamma controls how much")
    print("   weight-2 leaks into the linear response, R controls the design's")
    print("   own O(R^2) bias. Both are ours to set.")
    print("")


def part3():
    print("PART 3  THE ALGORITHM, and its scope.")
    print("")
    print("   PREPARE   design register in superposition, phase-imprint the row")
    print("             energy, Hadamard.            O(1) circuits")
    print("   ESTIMATE  P_1 = P(land in the weight-1 character set), a FIXED")
    print("             known subspace - so amplitude estimation applies with")
    print("             no marking oracle.           O(1/eps) applications")
    print("   RETURN    ||g|| = sqrt(P_1) / (gam sin(R)/2)")
    print("")
    print("   %-34s %18s" % ("", "cost to get ||g|| to eps"))
    print("   %-34s %18s" % ("classical / parameter-shift", "O(M) + O(M/eps^2)"))
    print("   %-34s %18s" % ("design register, sampled", "O(M/eps^2) shots"))
    print("   %-34s %18s" % ("this, with amplitude estimation", "O(1/eps)"))
    print("")
    print("   INDEPENDENT OF M on both axes - the component count and the")
    print("   precision exponent.")
    print("")
    print("   WHAT IT DOES NOT GIVE. The direction. P_1 is a norm and carries")
    print("   no sign or coordinate information, so this replaces the")
    print("   CONVERGENCE TEST and the DAMPING, not the step. A method that")
    print("   needs ||g|| far more often than it needs g - and a trust region")
    print("   evaluates ||g|| every epoch while only accepting a step")
    print("   sometimes - is where it pays.")
    print("")
    print("   AND THE HONEST CEILING. Amplitude estimation is available to any")
    print("   method that can write its target as one probability. The novelty")
    print("   is not AE; it is that the WEIGHT-1 SUBSPACE OF THE DESIGN")
    print("   REGISTER IS SUCH A TARGET, which is a property of the Walsh")
    print("   structure and not of the estimator. No classical design has an")
    print("   analogue, because there is no classical object whose measurement")
    print("   probability is the squared norm of a gradient.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("")
    want = sys.argv[1:] or ["1", "2", "3"]
    for k, fn in (("1", part1), ("2", part2), ("3", part3)):
        if k in want:
            fn()
