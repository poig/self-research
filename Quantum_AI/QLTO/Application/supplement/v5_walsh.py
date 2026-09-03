"""Does a degree-2 drift term have anything to carry?

T1 established that sense_gradient returns the degree-1 Walsh coefficients of the
energy on the +-R hypercube, and that the walk's CRZ writes a phase LINEAR in the
param bits - a degree-1 model of the energy. The natural upgrade is a degree-2
model: add CRZZ(Ehat({i,j})*gamma) terms, whose coefficients are estimable from
the SAME shots at zero extra circuit cost, because every Walsh coefficient is an
expectation over the same samples (T2).

But that is only worth building if the degree-2 weight EXISTS. By Parseval, the
variance of E over the hypercube decomposes exactly by degree:

    Var(E) = sum_{S != empty} Ehat(S)^2

so the fraction of the landscape's variation living at each degree is a five-line
classical calculation. If degree-1 carries nearly everything, a quadratic drift
model buys nothing and idea #1 is dead before any circuit is written.

PART 2 asks the separate question of whether the degree-2 coefficients are
MEASURABLE from a real sensing run at usable precision, since being free in
circuits is not the same as being free in shots.
"""
import sys, os, contextlib, io
import itertools
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import QFT, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def corner_energies(ansatz, H, c, R, act):
    """<H> at every one of the 2^n corners of the block's hypercube."""
    n = len(act)
    sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(n)]
                    for v in range(2 ** n)])
    E = np.empty(len(sig))
    for v, s in enumerate(sig):
        p = c.copy()
        p[act] = c[act] + R * s
        E[v] = float(np.real(Statevector(ansatz.assign_parameters(p))
                             .expectation_value(H)))
    return sig, E


def walsh_spectrum(sig, E):
    """All Walsh coefficients, keyed by subset."""
    n = sig.shape[1]
    out = {}
    for d in range(n + 1):
        for S in itertools.combinations(range(n), d):
            chi = np.ones(len(E))
            for i in S:
                chi = chi * sig[:, i]
            out[S] = float(np.mean(E * chi))
    return out


print("=" * 84)
print("PART 1. Walsh variance decomposition by degree (Parseval, exact)")
print("=" * 84)
print("  fraction of Var(E) over the +-R hypercube living at each degree")
print()

for pname, fn in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                  ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
                  ("H2", B.get_h2_problem)):
    ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=8192)
    BLK = [b['params'] for b in q.layers]
    print(f"  --- {pname} ---")
    for R in (0.6, 0.3):
        # one fraction-vector per (centre, block) sample, then average those
        samples = []
        for seed in (3, 7, 11):
            c = np.random.RandomState(seed).uniform(-np.pi, np.pi,
                                                    ansatz.num_parameters)
            for act in BLK:
                sig, E = corner_energies(ansatz, H, c, R, act)
                W = walsh_spectrum(sig, E)
                tot = sum(v ** 2 for S, v in W.items() if S)
                if tot <= 0:
                    continue
                nmax = len(act)
                f = np.zeros(nmax + 1)
                for S, v in W.items():
                    if S:
                        f[len(S)] += v ** 2 / tot
                samples.append(f)
        Fm = np.mean(np.array(samples), axis=0)
        print(f"    R={R}: " + "  ".join(f"deg{d}={Fm[d]:6.3f}"
                                        for d in range(1, len(Fm))))
    print()

print("=" * 84)
print("PART 2. Are the degree-2 coefficients MEASURABLE from a sensing run?")
print("=" * 84)


def sense_walsh(q, c, R, act, reps=6):
    """Degree-1 and degree-2 Walsh coefficients from QPE sensing shots.

    Both are empirical means over the same shot record - no extra circuit.
    """
    n, k = len(act), q.num_ancillas
    anc = AncillaRegister(k, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(k, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    for a in range(k):
        qc.append(PauliEvolutionGate(
            q.H_sense, time=(2 ** a) * q.tau0,
            synthesis=SuzukiTrotter(order=2, reps=max(1, (2 ** a) // 2))
        ).control(1), [anc[a]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])

    d1s, d2s = [], []
    for _ in range(reps):
        counts = q._run(qc)
        tot = 0
        s1 = np.zeros(n); s2 = {}
        for bs, cnt in counts.items():
            parts = bs.split()
            if len(parts) != 2:
                continue
            m = int(parts[0], 2); phi = m / (2 ** k)
            if phi >= 0.5:
                phi -= 1.0
            e = -2.0 * np.pi * phi / (q.tau0 + 1e-12)
            xb = parts[1][::-1]
            sg = np.array([1.0 if (i < len(xb) and xb[i] == '1') else -1.0
                           for i in range(n)])
            s1 += e * sg * cnt
            for i in range(n):
                for j in range(i + 1, n):
                    s2[(i, j)] = s2.get((i, j), 0.0) + e * sg[i] * sg[j] * cnt
            tot += cnt
        d1s.append(s1 / tot)
        d2s.append(np.array([s2[(i, j)] / tot for i in range(n)
                             for j in range(i + 1, n)]))
    return np.array(d1s), np.array(d2s)


ansatz, H, _ = B.get_heisenberg_problem(4)
q = Q(ansatz, H, shot_budget=8192, num_ancillas=4)
BLK = [b['params'] for b in q.layers]
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
R = 0.6
print(f"  {'blk':<5}{'|d1| exact':>12}{'|d2| exact':>12}"
      f"{'|d1| meas':>11}{'|d2| meas':>11}{'d1 SNR':>9}{'d2 SNR':>9}")
print("  " + "-" * 69)
for bi, act in enumerate(BLK):
    sig, E = corner_energies(ansatz, H, c, R, act)
    W = walsh_spectrum(sig, E)
    n = len(act)
    ex1 = np.array([W[(i,)] for i in range(n)])
    ex2 = np.array([W[(i, j)] for i in range(n) for j in range(i + 1, n)])
    m1, m2 = sense_walsh(q, c, R, act)
    n1 = np.linalg.norm(m1.mean(axis=0)); n2 = np.linalg.norm(m2.mean(axis=0))
    sd1 = np.mean([np.linalg.norm(r - m1.mean(axis=0)) for r in m1])
    sd2 = np.mean([np.linalg.norm(r - m2.mean(axis=0)) for r in m2])
    print(f"  {bi:<5}{np.linalg.norm(ex1):>12.4f}{np.linalg.norm(ex2):>12.4f}"
          f"{n1:>11.4f}{n2:>11.4f}{n1/max(sd1,1e-9):>9.2f}"
          f"{n2/max(sd2,1e-9):>9.2f}", flush=True)
print()
print("  SNR = ||coefficient vector|| / per-run scatter. Degree-2 is only worth")
print("  wiring into the walk if its exact weight is a real fraction of degree-1")
print("  AND its SNR at 8192 shots is above ~1.")
