"""The landscape's Fourier weight spectrum: one probability per level.

TIER B - exact amplitudes by Statevector. The claim is an identity.

THE QUANTITY. For E(theta + R sigma) on sigma in {+-1}^M, the WEIGHT-d FOURIER
ENERGY is

    W_d  =  sum_{|S| = d} Ehat(S)^2

the total squared Walsh coefficient at degree d. It is the norm of the d-th
derivative tensor, and v137 measured that a real variational landscape's weight
spectrum is ~ Binomial(M, 1/2) - CONCENTRATED AT d ~ M/2. So the bulk of the
landscape's structure lives in the middle of the spectrum.

THE QUANTUM SIDE, and the catch that has to be checked. With UNIT columns
c_j = 2^j (k = M), the weight-d character c_{j1} ^ ... ^ c_{jd} has Hamming
weight exactly d. So the weight-d subspace is "register states of Hamming weight
d", which is recognised by an efficient weight computation - NOT by enumerating
the C(M,d) characters, which at d ~ M/2 would be 2^M work just to define the
measurement. That is the catch, and unit columns remove it.

    prepare   H^k, phase-imprint e^{i gam E_d}, H^k        O(1) circuits
    project   Hamming weight of the register == d          O(M) gates
    estimate  one probability                              O(1/eps) by AE

Note what is given up: unit columns mean k = M, so the log-width advantage of
the design register is GONE here. This construction is about the SPECTRUM, not
about cheap gradients, and the two use the register differently.

THE CLASSICAL SIDE. Three routes, all exponential at d ~ M/2:

    direct enumeration      C(M,d) ~ 2^M coefficients
    finite differences      2^d ~ 2^{M/2} evaluations per coefficient
    noise operator          <f, T_rho f> = sum_S rho^{|S|} fhat(S)^2, so M+1
                            values of rho and a Vandermonde inversion - and the
                            conditioning is MEASURED exponential: log2 cond
                            4.30 -> 39.69 over M = 4..32 even with optimal
                            Chebyshev placement, i.e. ~2^{1.27 M}

NOT A PROOF OF CLASSICAL HARDNESS. Three natural algorithms were checked and all
are exponential; no lower bound is claimed. That distinction is the difference
between this and Shor, where factoring's hardness is a standing assumption
rather than an observation about three algorithms.

PART 1 verifies P_d ~ W_d for every d.
PART 2 shows the spectrum recovered, against the exact one.
"""
import sys
import itertools
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector


def landscape(M, rng, depth=3):
    """A function on {+-1}^M with genuine high-weight content: a product of
    random linear forms, which spreads Fourier energy across many degrees."""
    A = rng.normal(size=(depth, M)) * 0.7
    b = rng.normal(size=depth) * 0.3

    def E(sig):
        v = 1.0
        for r in range(depth):
            v *= np.tanh(A[r] @ sig + b[r])
        return v
    return E


def all_energies(E, M):
    N = 1 << M
    out = np.zeros(N)
    for d in range(N):
        sig = np.array([1.0 - 2.0 * ((d >> j) & 1) for j in range(M)])
        out[d] = E(sig)
    return out


def exact_weight_spectrum(vals, M):
    """Walsh transform, then sum squares by Hamming weight of the character."""
    N = 1 << M
    f = vals.copy()
    h = 1
    while h < N:                       # fast Walsh-Hadamard
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                x, y = f[j], f[j + h]
                f[j], f[j + h] = x + y, x - y
        h *= 2
    f = f / N
    W = np.zeros(M + 1)
    for c in range(N):
        W[bin(c).count("1")] += f[c] ** 2
    return W


def quantum_weight_probs(vals, M, gam):
    """P(register has Hamming weight d) after imprint + Hadamard."""
    reg = QuantumRegister(M, "d")
    qc = QuantumCircuit(reg)
    qc.h(reg)
    qc.append(DiagonalGate(list(np.exp(1j * gam * vals))), list(reg))
    qc.h(reg)
    pr = np.asarray(Statevector(qc).probabilities())
    P = np.zeros(M + 1)
    for c in range(1 << M):
        P[bin(c).count("1")] += pr[c]
    return P


def part1():
    print("PART 1  P(Hamming weight d)  ~  W_d = sum_{|S|=d} Ehat(S)^2 ?")
    print("        TIER B. gamma small -> linear response.")
    rng = np.random.default_rng(7)
    M, gam = 8, 0.05
    E = landscape(M, rng)
    vals = all_energies(E, M)
    W = exact_weight_spectrum(vals, M)
    P = quantum_weight_probs(vals, M, gam)
    print("        M = %d, gamma = %.2f" % (M, gam))
    print("")
    print("   %4s %16s %16s %14s"
          % ("d", "W_d exact", "P_d / gam^2", "ratio"))
    for d in range(1, M + 1):
        if W[d] < 1e-12:
            continue
        est = P[d] / gam ** 2
        print("   %4d %16.8f %16.8f %14.6f" % (d, W[d], est, est / W[d]))
    print("")
    print("   Ratio ~ 1 at EVERY degree is the identity: one probability per")
    print("   weight level, and the level is read by a Hamming-weight")
    print("   computation rather than by enumerating C(M,d) characters.")
    print("")


def part2():
    print("PART 2  THE SPECTRUM, recovered. This is what costs 2^M classically.")
    rng = np.random.default_rng(3)
    M, gam = 10, 0.04
    E = landscape(M, rng, depth=4)
    vals = all_energies(E, M)
    W = exact_weight_spectrum(vals, M)
    P = quantum_weight_probs(vals, M, gam)
    tot = W[1:].sum()
    print("        M = %d.  C(M,M/2) = %d characters at the peak degree."
          % (M, len(list(itertools.combinations(range(M), M // 2)))))
    print("")
    print("   %4s %14s %14s %10s %s"
          % ("d", "W_d exact", "W_d quantum", "share", "bar"))
    for d in range(1, M + 1):
        est = P[d] / gam ** 2
        share = W[d] / tot
        print("   %4d %14.8f %14.8f %9.1f%% %s"
              % (d, W[d], est, 100 * share, "#" * int(60 * share)))
    print("")
    print("   The mass sits in the MIDDLE of the spectrum, exactly as v137")
    print("   measured for a real variational landscape - and the middle is")
    print("   where every classical route costs 2^M.")
    print("")


def part3():
    print("PART 3  THE LEDGER, and the honest gap.")
    print("")
    print("   %-30s %22s %16s" % ("route", "cost for W_d at d~M/2", "verdict"))
    print("   %-30s %22s %16s"
          % ("direct enumeration", "C(M,d) ~ 2^M", "exponential"))
    print("   %-30s %22s %16s"
          % ("finite differences per coeff", "2^d ~ 2^{M/2}", "exponential"))
    print("   %-30s %22s %16s"
          % ("noise operator + Vandermonde", "cond ~ 2^{1.27 M}", "exponential"))
    print("   %-30s %22s %16s"
          % ("THIS (AE on one subspace)", "O(1/eps)", "polynomial"))
    print("")
    print("   WHAT IS GIVEN UP. Unit columns mean k = M, so the design")
    print("   register's log-width advantage does not apply here. This")
    print("   construction reads the SPECTRUM; the log-width one reads the")
    print("   GRADIENT. Same register, different use, different scaling.")
    print("")
    print("   WHAT IS NOT PROVEN. Classical HARDNESS. Three natural algorithms")
    print("   were checked and all are exponential; that is evidence, not a")
    print("   lower bound. Shor rests on factoring's assumed hardness, which is")
    print("   a standing conjecture examined for decades - this rests on three")
    print("   algorithms examined for an afternoon. The right next step is a")
    print("   query-complexity lower bound for weight-d Fourier energy, and")
    print("   until that exists this is a CANDIDATE separation, not one.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("")
    want = sys.argv[1:] or ["1", "2", "3"]
    for k, fn in (("1", part1), ("2", part2), ("3", part3)):
        if k in want:
            fn()
