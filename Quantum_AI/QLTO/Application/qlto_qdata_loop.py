"""Training on QUANTUM DATA, end to end, with shots. TIER A.

Every circuit built, transpiled, run on AerSimulator with finite shots. The
reference is the exact optimum of the same objective, found by BFGS - a DERIVED
reference, which is the admissible kind.

WHY THIS OBJECTIVE. qlto_separation checked five routes to a complexity
separation; four fail because they seek hardness in the ESTIMATOR, and reading
the gradient of a function you can evaluate is polynomial classically. The route
that survives changes the INPUT. Huang, Broughton, Cotler, Chen, Li, Mohseni,
Chen, Babbush, Kueng, Preskill & McClean (Science 2022) prove an EXPONENTIAL
separation in the number of experiments between algorithms with and without
quantum memory, and it exists precisely for properties NONLINEAR in the state:

    single-copy measurement    Omega(2^n) experiments
    two-copy entangled         O(n) experiments

    L(theta) = tr[ rho_A(theta)^2 ],  rho_A = tr_B[ U(theta) rho U(theta)^dag ]

is their canonical hard case - subsystem purity, quadratic in rho, unreachable
single-copy, and physically the thing one trains to disentangle. rho is GIVEN:
it stands for a state arriving from an experiment, not one built from classical
data, so the Theta(|D|) amplitude-encoding term that dominated every cost ledger
is simply not incurred.

WHAT WAS ALREADY VERIFIED (tier B, exact amplitudes): the design register reads
the gradient of this nonlinear two-copy functional with error falling as O(R^2)
(4.47e-2 -> 1.12e-2 -> 2.81e-3, ratios 3.99 and 3.99), and breaking the SHARING
of the design register between the two copies halves the estimate exactly, as
d/dtheta tr[rho_A^2] = 2 tr[rho_A d rho_A] demands.

WHAT THIS FILE ADDS: shots, and a loop. Does the gradient survive shot noise,
and does the loop reach the exact optimum?
"""
import sys
import time
import numpy as np
from scipy.optimize import minimize

from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    transpile)
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator

NS, NA, K = 3, 1, 8
COLS = [1, 2, 4, 8]           # linearly independent -> alias-free at all orders
ECOLS = [16, 32, 64, 128]
M = len(COLS)


def data_state(qc, sq):
    """The GIVEN quantum data - a fixed entangled state standing for one
    arriving from an experiment."""
    qc.ry(1.1, sq[0])
    qc.ry(0.7, sq[1])
    qc.ry(1.4, sq[2])
    qc.cx(sq[0], sq[1])
    qc.cx(sq[1], sq[2])
    qc.rz(0.4, sq[2])


def model(qc, sq, theta, reg=None, scr=None, R=0.0):
    prev_c = prev_e = 0
    for j in range(M):
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
        if j + 1 < M:
            qc.cx(sq[j % NS], sq[(j + 1) % NS])
    if reg is not None:
        for b in range(K):
            if prev_c >> b & 1:
                qc.cx(reg[b], scr[0])
            if prev_e >> b & 1:
                qc.cx(reg[b], scr[1])


def exact_purity(theta):
    sq = QuantumRegister(NS, "s")
    qc = QuantumCircuit(sq)
    data_state(qc, sq)
    model(qc, sq, theta)
    rho = partial_trace(Statevector(qc), list(range(NA, NS)))
    m = np.asarray(rho.data)
    return float(np.real(np.trace(m @ m)))


def sigma_of(d):
    a = np.array([1.0 - 2.0 * (bin(d & c).count("1") & 1) for c in COLS])
    b = np.array([1.0 - 2.0 * (bin(d & e).count("1") & 1) for e in ECOLS])
    return 0.5 * (a + b)


def sense(be, theta, R, shots):
    """ONE circuit: shared design register, two copies, SWAP test.
    Returns all M gradient components from one shot record."""
    reg, scr = QuantumRegister(K, "d"), QuantumRegister(2, "c")
    sA, sB = QuantumRegister(NS, "sA"), QuantumRegister(NS, "sB")
    anc = QuantumRegister(1, "a")
    qc = QuantumCircuit(reg, scr, sA, sB, anc,
                        ClassicalRegister(1, "ca"), ClassicalRegister(K, "cd"))
    qc.h(reg)
    for sq in (sA, sB):
        data_state(qc, sq)
        model(qc, sq, theta, reg, scr, R)
    qc.h(anc[0])
    for i in range(NA):
        qc.cswap(anc[0], sA[i], sB[i])
    qc.h(anc[0])
    qc.measure(anc, qc.cregs[0])
    qc.measure(reg, qc.cregs[1])
    t = transpile(qc, be, optimization_level=1)
    cnt = be.run(t, shots=shots).result().get_counts()
    acc = np.zeros(M)
    tot = 0
    for key, v in cnt.items():
        parts = key.split()
        d = int(parts[0], 2)                 # design reg, created last
        z = 1.0 - 2.0 * int(parts[1], 2)     # Z on the SWAP ancilla
        acc += sigma_of(d) * z * v
        tot += v
    s2 = 0.5                                  # <sigma_j^2> for the 3-level design
    return (acc / max(tot, 1)) / s2 / np.sin(R)


def measure_purity(be, theta, shots):
    """SWAP test only - one circuit, used by the line search."""
    sA, sB = QuantumRegister(NS, "sA"), QuantumRegister(NS, "sB")
    anc = QuantumRegister(1, "a")
    qc = QuantumCircuit(sA, sB, anc, ClassicalRegister(1, "ca"))
    for sq in (sA, sB):
        data_state(qc, sq)
        model(qc, sq, theta)
    qc.h(anc[0])
    for i in range(NA):
        qc.cswap(anc[0], sA[i], sB[i])
    qc.h(anc[0])
    qc.measure(anc, qc.cregs[0])
    t = transpile(qc, be, optimization_level=1)
    cnt = be.run(t, shots=shots).result().get_counts()
    z = sum((1.0 - 2.0 * int(k, 2)) * v for k, v in cnt.items())
    return z / max(sum(cnt.values()), 1)


def part1():
    print("PART 1  Does the gradient survive SHOTS?  TIER A.")
    be = AerSimulator(seed_simulator=5)
    rng = np.random.default_rng(4)
    th = rng.uniform(-np.pi, np.pi, M)
    h = 1e-5
    gt = np.zeros(M)
    for j in range(M):
        p, m = np.array(th), np.array(th)
        p[j] += h
        m[j] -= h
        gt[j] = (exact_purity(p) - exact_purity(m)) / (2 * h)
    print("        L = tr[rho_A^2] = %.6f,  ||g_true|| = %.6f"
          % (exact_purity(th), np.linalg.norm(gt)))
    print("")
    print("   %8s %8s %12s %14s %12s"
          % ("shots", "R", "cos(g)", "|g|/|g_true|", "circuits"))
    for shots in (1 << 12, 1 << 14, 1 << 16):
        for R in (0.4, 0.25):
            g = sense(be, th, R, shots)
            cs = float(g @ gt / (np.linalg.norm(g) * np.linalg.norm(gt)))
            print("   %8d %8.2f %12.6f %14.4f %12d"
                  % (shots, R, cs, np.linalg.norm(g) / np.linalg.norm(gt), 1))
    print("   ONE circuit returns all %d components. The count does not depend"
          % M)
    print("   on M - that is the design register - and there is no data-prep")
    print("   term because the state is GIVEN.")
    print("")


def part2():
    print("PART 2  THE LOOP. Maximise subsystem purity - disentangle the data.")
    be = AerSimulator(seed_simulator=5)
    rng = np.random.default_rng(4)
    th = rng.uniform(-np.pi, np.pi, M)
    best = max(float(-minimize(lambda t: -exact_purity(t),
                               np.random.default_rng(100 + s)
                               .uniform(-np.pi, np.pi, M), method="BFGS").fun)
               for s in range(20))
    start = exact_purity(th)
    print("        exact optimum (20 BFGS restarts) = %.6f" % best)
    print("        random start                     = %.6f" % start)
    print("")
    print("   %6s %12s %12s %10s %10s"
          % ("epoch", "purity", "below opt", "step", "circuits"))
    t0 = time.time()
    R, shots, ls = 0.35, 1 << 14, [0]
    for ep in range(1, 41):
        g = sense(be, th, R, shots)
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        u = g / ng
        base = measure_purity(be, th, shots)
        t, best_t = 0.6, 0.0
        for _ in range(8):                    # backtracking line search
            ls[0] += 1
            if measure_purity(be, th + t * u, shots) > base:
                best_t = t
                break
            t *= 0.5
        if best_t > 0:
            th = th + best_t * u
        if ep % 8 == 0:
            print("   %6d %12.6f %12.6f %10.4f %10d"
                  % (ep, exact_purity(th), best - exact_purity(th), best_t,
                     1 + 1 + ls[0] // ep))
    got = exact_purity(th)
    print("")
    print("   reached %.6f of an optimum %.6f  -  %.1f%% of the gap, %.1f s"
          % (got, best, 100.0 * (got - start) / max(best - start, 1e-12),
             time.time() - t0))
    print("   circuits per epoch: 1 sense + 1 base + %.1f line search = %.1f,"
          % (ls[0] / 40.0, 2 + ls[0] / 40.0))
    print("   all independent of M and with NO data-encoding term.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("")
    want = sys.argv[1:] or ["1", "2"]
    for k, fn in (("1", part1), ("2", part2)):
        if k in want:
            fn()
