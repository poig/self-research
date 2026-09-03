"""The correction to v118: a thin barrier is NOT exponentially expensive to build.

v118 decomposed diag(E(w)) into Z-strings, found a thin barrier needs ~2^n of
them, and concluded the tunneling advantage is self-defeating. That conclusion is
WRONG. A diagonal phase oracle is not implemented as Pauli rotations - the
standard route is reversible arithmetic: compute w into an ancilla with an adder,
phase-kick, uncompute. This file measures both and shows the arithmetic route is
FLAT in barrier width. The Pauli decomposition is a true statement about the
operator and a false one about the circuit.

TIER (project rule R1): tier A for the gate counts - real Qiskit circuits, built
and transpiled. No shots needed; these are resource counts, not accuracy claims.
"""
import numpy as np
from math import comb
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import WeightedAdder
from qiskit_aer import AerSimulator

N = 8
be = AerSimulator()


def spike_energy(n, h, width, centre=None):
    w = np.arange(n + 1).astype(float)
    c = (n // 4) if centre is None else centre
    return w + h * np.exp(-((w - c) ** 2) / (2.0 * width ** 2))


def z_coefficients(n, E):
    ck = np.zeros(n + 1)
    for k in range(n + 1):
        tot = 0.0
        for wt in range(n + 1):
            s = 0.0
            for j in range(0, min(k, wt) + 1):
                if wt - j > n - k:
                    continue
                s += ((-1.0) ** j) * comb(k, j) * comb(n - k, wt - j)
            tot += E[wt] * s
        ck[k] = tot / (2 ** n)
    return ck


def pauli_cost(n, E, frac=0.99):
    ck = z_coefficients(n, E)
    mass = np.array([abs(ck[k]) * comb(n, k) for k in range(n + 1)])
    m = mass[1:] / max(mass[1:].sum(), 1e-30)
    kmax = int(np.searchsorted(np.cumsum(m), frac) + 1)
    return sum(comb(n, k) for k in range(1, kmax + 1)), kmax


def oracle_cost(n, E, dt):
    """Ancilla route: WeightedAdder computes w, phases on the sum register, uncompute."""
    add = WeightedAdder(n, [1] * n)
    nsum = add.num_sum_qubits
    q = QuantumRegister(n, 's')
    a = QuantumRegister(add.num_qubits - n, 'a')
    qc = QuantumCircuit(q, a)
    qc.compose(add, qubits=list(q) + list(a), inplace=True)
    # phase e^{-i E(w) dt}: one multi-controlled phase per value of w
    sumq = list(a)[:nsum]
    for w in range(n + 1):
        ang = -E[w] * dt
        if abs(ang) < 1e-12:
            continue
        bits = [(w >> i) & 1 for i in range(nsum)]
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(sumq[i])
        if nsum == 1:
            qc.p(ang, sumq[0])
        else:
            qc.mcp(ang, sumq[:-1], sumq[-1])
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(sumq[i])
    qc.compose(add.inverse(), qubits=list(q) + list(a), inplace=True)
    t = transpile(qc, be, optimization_level=1)
    return t.size(), t.depth(), qc.num_qubits - n


print("Cost of ONE potential step e^{-iE(w)dt}, n=%d qubits" % N)
print()
print("  width   Pauli-rotation route      ancilla-arithmetic route")
print("          gates   (body-order)      gates   depth   ancillas")
print("  " + "-" * 68)
for width in (0.5, 1.0, 2.0, 4.0, 8.0):
    E = spike_energy(N, h=20.0, width=width)
    pg, kmax = pauli_cost(N, E)
    og, od, na = oracle_cost(N, E, 0.1)
    print("  %5.1f   %5d       (k<=%d)        %5d   %5d      %2d"
          % (width, pg, kmax, og, od, na))
print()
print("The Pauli route grows as the barrier thins (2^n in the limit).")
print("The arithmetic route does NOT depend on the shape of E at all - it is the")
print("same circuit with different rotation angles.")
