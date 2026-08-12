"""Hamiltonian learning on V6's LOG register - the task where G=1 by construction.

v6_hamlearn showed the pivot works, but with a LINEAR parameter register: one
qubit per coefficient, exactly the thing V6 exists to remove. At M=5 that is 5
qubits and nobody cares. The claim only bites as M grows, and this measures where.

WHY THIS TASK AND NOT VQE. In VQE the readout is a Pauli sum, so a gradient costs
G circuits - 3 for Heisenberg. Here the loss is a RETURN PROBABILITY: prepare
|+..+>, evolve under the unknown device e^{-i H_true T}, apply the model's inverse
e^{+i H(theta) T}, and ask whether the probe came back. That is ONE bit from ONE
measurement setting, so G = 1 STRUCTURALLY, not as a property of some Hamiltonian
family. Theta(G) -> Theta(1). It is the cleanest form the cost claim ever takes,
and it is a quantum-DATA task, which is the setting where sample-complexity
advantage is provable rather than hoped for.

WHAT IS ACTUALLY UNDER TEST. Not "does Hamiltonian learning work" - v6_hamlearn
settled that (cos 0.99943, recovery to 0.034/coefficient). The question is whether
the degree-1 Walsh marginal survives being read off a resolution-IV Hadamard
design on ceil(log2(M+1))+1 qubits instead of M independent parameter qubits.
The design ALIASES: with M parameters in 2^(m_row+1) runs the columns are
orthogonal by construction, but only if the sign reconstruction is right, and a
wrong Gray/foldover convention produces a plausible-looking gradient that is
silently wrong on a subset of coordinates. Hence cosine against the exactly
enumerated smeared gradient at every M, not just recovery.

WIDTH IS THE POINT. linear = M + N. design = ceil(log2(M+1)) + 1 + N + n_scratch.
They cross once M exceeds roughly log2(M) + 5, i.e. almost immediately, and the
gap then grows without bound. At M=5 the design is NOT cheaper and the table
should show that rather than start at a flattering size.

All terms are Z-type so they commute and the Trotterisation is exact - a
disagreement here is the estimator's fault and cannot be blamed on Trotter error.
"""
import sys, os, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from nisq_v6 import _design_spec, _design_sign

T = 1.0
SHOTS = 8192
R = 0.4
BACKEND = AerSimulator(method='statevector')
RNG = np.random.default_rng(7)


def make_terms(N, M):
    """M distinct Z-type strings on N qubits: singles first, then ZZ pairs."""
    out = []
    for i in range(N):
        s = ['I'] * N
        s[i] = 'Z'
        out.append(''.join(s))
    for i, j in itertools.combinations(range(N), 2):
        s = ['I'] * N
        s[i] = s[j] = 'Z'
        out.append(''.join(s))
    if len(out) < M:
        raise ValueError(f"N={N} supplies only {len(out)} Z-type terms, need {M}")
    return out[:M]


def h_of(terms, theta):
    return SparsePauliOp.from_list(list(zip(terms, theta))).simplify()


def return_prob(terms, c_true, theta, N):
    qc = QuantumCircuit(N)
    qc.h(range(N))
    qc.append(PauliEvolutionGate(h_of(terms, c_true), time=T), range(N))
    qc.append(PauliEvolutionGate(h_of(terms, theta), time=-T), range(N))
    qc.h(range(N))
    return float(abs(Statevector(qc).data[0]) ** 2)


def sense_design(terms, c_true, center, N, n_scratch=2, shots=SHOTS):
    """One circuit on the LOG register. Returns the gradient estimate.

    theta_i = center_i + R * s_i with s_i = _design_sign(row, fold, cols[i]).
    Base evolution at center_i + R, then a controlled -2R increment fired by the
    parity bit, so s_i = +1 when the parity is even. That is the same convention
    _design_sign uses, and getting it backwards flips every coordinate at once -
    which the cosine would catch, but silently, so it is stated here.
    """
    M = len(terms)
    m_row, cols = _design_spec(M, n_scratch)
    nreg = m_row + 1
    ns = max(1, min(n_scratch, M))

    param = QuantumRegister(nreg, 'p')
    sysr = QuantumRegister(N, 's')
    scr = QuantumRegister(ns, 'a')
    qc = QuantumCircuit(param, sysr, scr,
                        ClassicalRegister(nreg, 'cp'), ClassicalRegister(N, 'cs'))
    qc.h(param)
    qc.h(sysr)
    qc.append(PauliEvolutionGate(h_of(terms, c_true), time=T), sysr)

    P = [SparsePauliOp.from_list([(t, 1.0)]) for t in terms]
    for i in range(M):
        s = i % ns
        # base at center_i + R
        qc.append(PauliEvolutionGate(P[i], time=-(center[i] + R) * T), sysr)
        # parity of (row & cols[i]) xor foldover -> scratch qubit s
        for b in range(m_row):
            if (cols[i] >> b) & 1:
                qc.cx(param[b], scr[s])
        qc.cx(param[m_row], scr[s])
        # fire -2R when parity is 1
        qc.append(PauliEvolutionGate(P[i], time=+2.0 * R * T).control(1),
                  [scr[s]] + list(sysr))
        # uncompute
        qc.cx(param[m_row], scr[s])
        for b in range(m_row):
            if (cols[i] >> b) & 1:
                qc.cx(param[b], scr[s])

    qc.h(sysr)
    qc.measure(param, qc.cregs[0])
    qc.measure(sysr, qc.cregs[1])

    width = qc.num_qubits
    counts = BACKEND.run(transpile(qc, BACKEND, optimization_level=1),
                         shots=shots).result().get_counts()

    tot, acc = 0, np.zeros(M)
    for bs, cnt in counts.items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        sys_bits, par_bits = parts[0], parts[1]     # cs printed first
        ret = 1.0 if set(sys_bits) == {'0'} else 0.0
        if ret == 0.0:
            tot += cnt
            continue
        xb = par_bits[::-1]                          # little-endian
        row = sum(1 << b for b in range(m_row)
                  if b < len(xb) and xb[b] == '1')
        fold = 1 if (m_row < len(xb) and xb[m_row] == '1') else 0
        sg = np.array([_design_sign(row, fold, cols[i]) for i in range(M)])
        acc += ret * sg * cnt
        tot += cnt
    return acc / max(tot, 1) / R, width, m_row


MC_CAP = 4096


def smeared_exact(terms, c_true, center, N, M):
    """The reference smeared gradient, EXACT energies, no shot noise.

    Full enumeration of the +-R hypercube costs 2^M evaluations, which is 65536
    at M=16 and dominates everything else here. Above MC_CAP the same quantity is
    estimated by sampling sign vectors uniformly - still the degree-1 Walsh
    coefficient, still computed from exact statevector energies, so the only
    thing introduced is a sampling error on the REFERENCE. That error is
    O(1/sqrt(MC_CAP)) ~ 1.6%, an order below the cosines being reported, and it
    is symmetric so it cannot flatter the design.
    """
    if 2 ** M <= MC_CAP:
        sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                        for v in range(2 ** M)])
    else:
        sig = np.random.default_rng(11).choice([-1.0, 1.0], size=(MC_CAP, M))
    Ev = np.array([return_prob(terms, c_true, center + R * s, N) for s in sig])
    return np.array([np.mean(Ev * sig[:, i]) for i in range(M)]) / R


def cos(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d > 0 else 0.0


print("=" * 92)
print("PART 1.  Does the degree-1 marginal survive the LOG design register?")
print("=" * 92)
print("  cosine against the exactly enumerated smeared gradient, so a wrong")
print("  Gray/foldover convention cannot hide behind a plausible descent.")
print()
print(f"  {'N':>3}{'M':>4}{'m_row':>7}{'linear w':>10}{'design w':>10}"
      f"{'saved':>7}{'cos':>9}{'circuits':>10}")
print("  " + "-" * 62)

for N, M in ((4, 5), (4, 8), (5, 12), (6, 16)):
    terms = make_terms(N, M)
    c_true = np.round(RNG.uniform(-0.8, 0.8, M), 2)
    center = c_true + RNG.uniform(-0.3, 0.3, M)
    g, width, m_row = sense_design(terms, c_true, center, N)
    ex = smeared_exact(terms, c_true, center, N, M)
    lin_w = M + N
    print(f"  {N:>3}{M:>4}{m_row:>7}{lin_w:>10}{width:>10}"
          f"{lin_w - width:>7}{cos(g, ex):>9.5f}{1:>10}")

print()
print("  Width crosses early and then diverges: the design register is")
print("  ceil(log2(M+1))+1 whatever M is, the linear one is M.")

print()
print("=" * 92)
print("PART 2.  Recovery from a wrong start, one circuit per epoch")
print("=" * 92)
N, M = 6, 16
terms = make_terms(N, M)
c_true = np.round(RNG.uniform(-0.8, 0.8, M), 2)
theta = c_true + RNG.uniform(-0.4, 0.4, M)
print(f"  N={N}  M={M}  terms {terms[0]} ... {terms[-1]}")
print(f"  start ||theta-c|| = {np.linalg.norm(theta - c_true):.4f}"
      f"   return prob = {return_prob(terms, c_true, theta, N):.4f}")
print()
print(f"  {'epoch':>7}{'R':>8}{'||theta-c||':>14}{'return prob':>14}")
print("  " + "-" * 43)

r, alpha, circuits = 0.5, 0.35, 0
for ep in range(1, 31):
    globals()['R'] = r
    g, _, _ = sense_design(terms, c_true, theta, N)
    circuits += 1
    mx = float(np.max(np.abs(g)))
    if mx > 1e-12:
        theta = theta + alpha * r * g / mx     # ASCEND: maximise return prob
    if ep % 5 == 0:
        print(f"  {ep:>7}{r:>8.3f}{np.linalg.norm(theta - c_true):>14.4f}"
              f"{return_prob(terms, c_true, theta, N):>14.4f}")
    r = max(r * 0.93, 1e-3)

print()
print(f"  max |error| per coefficient = {np.max(np.abs(theta - c_true)):.4f}")
print(f"  circuits used = {circuits} (one per epoch, all {M} coefficients each)")
print(f"  parameter-shift equivalent = {2 * M * circuits}"
      f"  ->  {2 * M}x")
print(f"  design register carried {M} coefficients on "
      f"{_design_spec(M, 2)[0] + 1} qubits")
