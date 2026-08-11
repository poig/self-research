"""What does a Shor-like Hamiltonian look like to a gradient? Is there a signature?

The question: given a Hamiltonian built from a periodic (hidden-subgroup) function,
does the VQE landscape look different from one built from a generic function, and
is the difference detectable in the gradient?

THE PREDICTION, worked out before measuring.

For a DIAGONAL Hamiltonian H = sum_x f(x) |x><x| the landscape is

    E(theta) = sum_x f(x) p_theta(x) = sum_S fhat(S) <Z_S>_theta

where Z_S is the Pauli-Z string on the subset S. So the WALSH COEFFICIENTS OF f
ARE EXACTLY THE PAULI-Z COEFFICIENTS OF H - the same numbers, viewed twice.

Now the hidden-subgroup structure. Simon's promise is f(x) = f(x XOR s) for a
fixed s (the hypercube analogue of Shor's periodicity over Z_N). A function
invariant under XOR by s has

    fhat(T) = 0   unless   T . s = 0  (mod 2)

i.e. its spectrum is supported ENTIRELY on the annihilator subgroup of s, which
has exactly half the subsets. So the prediction is:

    Shor/Simon-like H  ->  Pauli support is a SUBGROUP, half the strings, and
                           every surviving string is orthogonal to s
    generic H          ->  Pauli support spread over all 2^n subsets

and since d_i E = sum_S fhat(S) d_i <Z_S>, the GRADIENT is built from the same
restricted set. The signature, if it exists, should be visible without running
any optimisation at all.

WHY THIS MATTERS EITHER WAY. If the signature is there, it says exactly what a
Fourier-sampling optimiser could in principle key on. If it is there AND trivially
readable from H's Pauli list, then it is the notes' obstruction restated in the
sharpest possible form: the structure that a quantum algorithm would need to
discover is structure you must already possess in order to write the Hamiltonian
down. Shor does not have that problem because its function is given by an ORACLE
(modular exponentiation) whose period is not visible in the circuit that computes
it. A VQE Hamiltonian is given by its Pauli list, which is the spectrum itself.

Measured here:
  (1) Pauli-Z support of a Simon-structured H vs a generic diagonal H
  (2) whether every surviving string is orthogonal to the hidden shift s
  (3) whether the gradient of a real ansatz inherits the same restriction
"""

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector

N = 6
D = 2 ** N
rng = np.random.default_rng(4)


def walsh(f):
    """Fast Walsh-Hadamard transform, normalised so fhat(S) are the Z-string
    coefficients of diag(f)."""
    a = f.astype(float).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j], a[j + h] = x + y, x - y
        h *= 2
    return a / len(a)


def parity(t, x):
    return bin(t & x).count("1") & 1


print("=" * 96)
print("DOES A HIDDEN-SUBGROUP HAMILTONIAN HAVE A GRADIENT SIGNATURE?")
print("=" * 96)
print(f"  N = {N}. Diagonal H = sum_x f(x)|x><x|; fhat(S) ARE H's Pauli-Z coefficients.")
print()

s = 0b101101                      # the hidden shift
print(f"  hidden shift s = {s:0{N}b}")
print()

# Simon-structured f: constant on cosets of {0, s}
base = rng.normal(size=D)
f_simon = np.array([base[min(x, x ^ s)] for x in range(D)])
f_generic = rng.normal(size=D)

for tag, f in (("Simon-structured", f_simon), ("generic diagonal", f_generic)):
    fh = walsh(f)
    big = np.abs(fh) > 1e-10
    support = int(big.sum())
    orth = [t for t in range(D) if big[t] and parity(t, s) == 0]
    nonorth = [t for t in range(D) if big[t] and parity(t, s) == 1]
    print(f"  {tag:>18} : {support:>3} / {D} Pauli strings nonzero"
          f"   ({100 * support / D:5.1f}%)")
    print(f"  {'':>18}   orthogonal to s : {len(orth):>3}"
          f"    NOT orthogonal : {len(nonorth):>3}")

print()
print("  A Simon-structured H should keep exactly the strings T with T.s = 0,")
print("  which is half of them, and kill every other. That is the signature.")

# ------------------------------------------------------------------ gradient
print()
print("=" * 96)
print("DOES THE GRADIENT INHERIT IT?")
print("=" * 96)
ansatz = efficient_su2(N, reps=2)
M = ansatz.num_parameters
theta = rng.uniform(-np.pi, np.pi, M)


def grad_of(f):
    Hm = np.diag(f)
    g = np.zeros(M)
    for i in range(M):
        for sg in (+1, -1):
            t = theta.copy()
            t[i] += sg * np.pi / 2
            v = Statevector(ansatz.assign_parameters(t)).data
            g[i] += sg * float(np.real(np.conj(v) @ (Hm @ v))) / 2
    return g


gs, gg = grad_of(f_simon), grad_of(f_generic)
print(f"  {'':>18}{'||grad||':>12}{'max |g_i|':>12}{'min |g_i|':>12}"
      f"{'zero comps':>12}")
print("  " + "-" * 66)
for tag, g in (("Simon-structured", gs), ("generic diagonal", gg)):
    z = int(np.sum(np.abs(g) < 1e-12))
    print(f"  {tag:>18}{np.linalg.norm(g):>12.5f}{np.max(np.abs(g)):>12.5f}"
          f"{np.min(np.abs(g)):>12.2e}{z:>12}")

print()
print("  READING IT. If the Pauli support collapses to the annihilator subgroup")
print("  but the GRADIENT looks statistically ordinary, then the structure is")
print("  real and present in H yet invisible in the quantity an optimiser reads.")
print("  That is the sharpest form of the obstruction: the periodicity lives in")
print("  the Pauli list, which you must already know to specify the problem, and")
print("  the ansatz scrambles it out of the landscape. Shor escapes only because")
print("  its function arrives as an ORACLE whose period is not written in the")
print("  circuit that evaluates it.")
