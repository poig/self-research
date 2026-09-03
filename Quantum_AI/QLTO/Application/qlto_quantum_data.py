"""Gradient of a NONLINEAR functional of quantum data, from one design register.

TIER B - exact amplitudes by Statevector. The claim is an identity.

WHY THIS IS THE ONE THAT MATTERS. qlto_separation.py checked five routes to a
complexity separation. Four fail because they look for hardness in the
ESTIMATOR, and reading the gradient of a function you can evaluate is polynomial
classically. The one that survives changes the INPUT: learning from QUANTUM
DATA, where Huang, Broughton, Cotler, Chen, Li, Mohseni, Chen, Babbush, Kueng,
Preskill & McClean (Science 2022) prove an EXPONENTIAL separation in the number
of experiments between algorithms with and without quantum memory -

    without quantum memory   Omega(2^n) experiments
    with quantum memory      O(n) experiments

and the separation exists precisely for properties NONLINEAR in the state, which
single-copy measurement cannot reach and two-copy entangled measurement can.

THE LOAD-BEARING UNKNOWN, and it is what this file settles. QLTO's design
register was built and verified for observables LINEAR in the state -
<psi|O|psi>. Every identity in the project (Part XVIII's Walsh spectrum, the
order-d tensors) assumes that form. A nonlinear functional is a different object
and there was no reason to expect the register to survive it.

THE TEST OBJECT. Subsystem purity of the transformed data state,

    L(theta) = tr[ rho_A(theta)^2 ],   rho_A(theta) = tr_B[ U(theta) rho U(theta)^dag ]

which is QUADRATIC in rho, requires two copies (a SWAP test on the A subsystems),
and is what one trains to disentangle. rho here is GIVEN - it stands for a state
arriving from an experiment, not one we prepare from classical data.

THE CONSTRUCTION. One design register SHARED by both copies, so both are
perturbed by the SAME row sigma(d):

    |Psi> = 2^{-k/2} sum_d |d> (x) U(th+R sig(d)) rho U^dag (x) U(th+R sig(d)) rho U^dag

then a SWAP test on the A subsystems. The claim:

    < sigma_j(d) (x) Z_anc >  =  (sin R / 2) dL/dtheta_j  +  O(R^3)

i.e. THE SAME WALSH IDENTITY, with a two-copy observable in place of a
single-copy one. If it holds, the design register composes with quantum memory
and the O(1)-circuit gradient carries over to the regime where the separation
lives.
"""
import sys
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, SparsePauliOp, partial_trace

NS, NA, K = 2, 1, 4          # system qubits per copy, subsystem A size, design
COLS = [1, 2]                # linearly independent -> no aliasing
ECOLS = [4, 8]


def data_state(qc, sq):
    """The GIVEN quantum data. Stands for a state from an experiment - fixed,
    not derived from any classical vector."""
    qc.ry(1.1, sq[0])
    qc.ry(0.7, sq[1])
    qc.cx(sq[0], sq[1])
    qc.rz(0.4, sq[1])


def model(qc, sq, theta, reg=None, scr=None, R=0.0):
    """U(theta), optionally with the design perturbation on each parameter."""
    prev_c = prev_e = 0
    for j in range(len(theta)):
        if reg is not None:
            for b in range(K):
                if (COLS[j] ^ prev_c) >> b & 1:
                    qc.cx(reg[b], scr[0])
                if (ECOLS[j] ^ prev_e) >> b & 1:
                    qc.cx(reg[b], scr[1])
            prev_c, prev_e = COLS[j], ECOLS[j]
            qc.ry(theta[j] + R, sq[j % NS])
            qc.cry(-R, scr[0], sq[j % NS])
            qc.cry(-R, scr[1], sq[j % NS])
        else:
            qc.ry(theta[j], sq[j % NS])
        if j + 1 < len(theta):
            qc.cx(sq[0], sq[1])
    if reg is not None:
        for b in range(K):
            if prev_c >> b & 1:
                qc.cx(reg[b], scr[0])
            if prev_e >> b & 1:
                qc.cx(reg[b], scr[1])


def exact_purity(theta):
    """L(theta) = tr[rho_A^2], computed exactly."""
    sq = QuantumRegister(NS, "s")
    qc = QuantumCircuit(sq)
    data_state(qc, sq)
    model(qc, sq, theta)
    rho = partial_trace(Statevector(qc), list(range(NA, NS)))
    m = np.asarray(rho.data)
    return float(np.real(np.trace(m @ m)))


def build(theta, R):
    """Two copies, ONE shared design register, SWAP test on subsystem A."""
    reg = QuantumRegister(K, "d")
    scr = QuantumRegister(2, "c")
    sA = QuantumRegister(NS, "sA")
    sB = QuantumRegister(NS, "sB")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(reg, scr, sA, sB, anc)
    qc.h(reg)
    for sq in (sA, sB):
        data_state(qc, sq)
        model(qc, sq, theta, reg, scr, R)
    qc.h(anc[0])
    for i in range(NA):
        qc.cswap(anc[0], sA[i], sB[i])
    qc.h(anc[0])
    return qc, (reg, sA, sB, anc)


def obs(qc, regs, j):
    """sigma_j(d) on the design register, times Z on the SWAP-test ancilla."""
    reg, sA, sB, anc = regs
    n = qc.num_qubits
    idx = {q: i for i, q in enumerate(qc.qubits)}
    terms = []
    for u in (COLS[j], ECOLS[j]):
        lab = ["I"] * n
        for b in range(K):
            if (u >> b) & 1:
                lab[idx[reg[b]]] = "Z"
        lab[idx[anc[0]]] = "Z"
        terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def part1():
    print("PART 1  Does the Walsh identity survive a NONLINEAR functional?")
    print("        <sigma_j (x) Z_anc> == (sin R/2) dL/dtheta_j ?   TIER B")
    rng = np.random.default_rng(5)
    theta = rng.uniform(-np.pi, np.pi, 2)
    h = 1e-5
    gt = np.zeros(2)
    for j in range(2):
        p, m = np.array(theta), np.array(theta)
        p[j] += h
        m[j] -= h
        gt[j] = (exact_purity(p) - exact_purity(m)) / (2 * h)
    print("        L(theta) = tr[rho_A^2] = %.6f" % exact_purity(theta))
    print("        exact dL/dtheta = [%.6f, %.6f]" % (gt[0], gt[1]))
    print("")
    print("   %6s %4s %16s %16s %11s"
          % ("R", "j", "from |Psi>", "exact", "|rel|"))
    for R in (0.30, 0.15, 0.075):
        qc, regs = build(theta, R)
        sv = Statevector(qc)
        for j in range(2):
            v = complex(sv.expectation_value(obs(qc, regs, j))).real
            est = v / (0.5 * np.sin(R))
            print("   %6.3f %4d %16.8f %16.8f %11.2e"
                  % (R, j, est, gt[j], abs(est - gt[j]) / max(abs(gt[j]), 1e-12)))
    print("")
    print("   Error must fall as O(R^2). If it does, the design register reads")
    print("   the gradient of a TWO-COPY functional exactly as it reads a")
    print("   single-copy one, and the O(1)-circuit gradient carries into the")
    print("   regime where Huang et al.'s separation lives.")
    print("")


def part2():
    print("PART 2  THE CONTROL. Break the SHARING of the design register -")
    print("        give each copy its own - and the estimator must stop")
    print("        measuring a derivative of L, because the two copies are then")
    print("        perturbed differently and the SWAP test no longer sees")
    print("        rho_A(theta + R sig) against ITSELF.")
    rng = np.random.default_rng(5)
    theta = rng.uniform(-np.pi, np.pi, 2)
    h = 1e-5
    gt = np.zeros(2)
    for j in range(2):
        p, m = np.array(theta), np.array(theta)
        p[j] += h
        m[j] -= h
        gt[j] = (exact_purity(p) - exact_purity(m)) / (2 * h)

    reg = QuantumRegister(K, "d")
    reg2 = QuantumRegister(K, "d2")
    scr, scr2 = QuantumRegister(2, "c"), QuantumRegister(2, "c2")
    sA, sB = QuantumRegister(NS, "sA"), QuantumRegister(NS, "sB")
    anc = QuantumRegister(1, "anc")
    R = 0.15
    qc = QuantumCircuit(reg, reg2, scr, scr2, sA, sB, anc)
    qc.h(reg)
    qc.h(reg2)
    data_state(qc, sA); model(qc, sA, theta, reg, scr, R)
    data_state(qc, sB); model(qc, sB, theta, reg2, scr2, R)
    qc.h(anc[0])
    for i in range(NA):
        qc.cswap(anc[0], sA[i], sB[i])
    qc.h(anc[0])
    sv = Statevector(qc)
    idx = {q: i for i, q in enumerate(qc.qubits)}
    print("")
    print("   %4s %16s %16s" % ("j", "unshared est", "exact dL/dj"))
    for j in range(2):
        terms = []
        for u in (COLS[j], ECOLS[j]):
            lab = ["I"] * qc.num_qubits
            for b in range(K):
                if (u >> b) & 1:
                    lab[idx[reg[b]]] = "Z"
            lab[idx[anc[0]]] = "Z"
            terms.append(("".join(reversed(lab)), 0.5))
        v = complex(sv.expectation_value(
            SparsePauliOp.from_list(terms))).real
        print("   %4d %16.8f %16.8f" % (j, v / (0.5 * np.sin(R)), gt[j]))
    print("   If the unshared column does NOT track the exact gradient, the")
    print("   SHARED register is what makes the identity work - the same")
    print("   mechanism as the Gauss-Newton construction, where sharing turns a")
    print("   product of means into a mean of products.")
    print("")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("")
    want = sys.argv[1:] or ["1", "2"]
    for k, fn in (("1", part1), ("2", part2)):
        if k in want:
            fn()
