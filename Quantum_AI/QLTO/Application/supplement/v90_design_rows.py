"""v88's aliasing, and whether extra design rows buy the fidelity back.

v88 measured the log register's gradient cosine falling from 0.997 at M=5 to
0.827 at M=12 on Hamiltonian learning. That is not shot noise - M=12 enumerates
its own hypercube exactly - and v89 ruled out the other candidate: under the
degree-wise law (*), R-smoothing preserves single-degree direction exactly, so
the radius cannot be blamed either. What is left is ALIASING in the design.

THE MECHANISM. The design assigns parameter i a column c_i in GF(2)^m and reads
its sign as (-1)^(popcount(row & c_i) + fold). A subset S of parameters is
CONFOUNDED when XOR_{i in S} c_i = 0: their effects become inseparable. The
foldover bit makes every column effectively odd, killing odd-size confounding,
which is what makes the design resolution IV - main effects clear of 2-factor
interactions. It does NOT clear them of 3-factor interactions, and the return
probability |<psi|psi'>|^2 is strongly non-linear over an R=0.4 hypercube, so
degree-3 content is large and lands directly on the main effects.

_design_spec currently takes the MINIMUM m_row = ceil(log2(n+1)) and fills it
with Gray codes, which guarantees distinct non-zero columns and nothing more.
At M=12 that is m_row=4, i.e. 12 parameters in 15 available columns - saturated,
so short confounding relations are unavoidable by counting alone.

THE FIX UNDER TEST. Spend extra row bits and choose columns so that

    (a) no column equals the XOR of two others      -> no 3-term relation
    (b) all pairwise XORs are distinct              -> no 4-term relation

which is the standard resolution-V construction. It costs width: (b) limits the
number of usable columns to roughly 2^(m/2), so m grows about twice as fast as
the ceil(log2(n+1)) minimum. The question is whether the cosine recovers enough
to justify that, and whether the result is still far below the LINEAR register's
M qubits - because if resolution V costs as much width as one qubit per
parameter, the log register has no reason to exist on this task.

Reported against the exactly enumerated smeared gradient, same as v88, so the
numbers are directly comparable to that log.
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

T, SHOTS, R = 1.0, 8192, 0.4
BACKEND = AerSimulator(method='statevector')
MC_CAP = 4096


def gray_cols(n):
    """What _design_spec ships: minimum width, Gray-code columns."""
    m = max(1, int(np.ceil(np.log2(n + 1))))
    return m, [(t ^ (t >> 1)) for t in range(1, n + 1)]


def resv_cols(n, m):
    """Greedy resolution-V column set in GF(2)^m, or None if m is too small.

    (a) a candidate that is already a pairwise XOR of chosen columns would
        create a 3-term relation, so it is skipped;
    (b) a candidate whose XOR with any chosen column duplicates an existing
        pairwise XOR would create a 4-term relation, so it is rejected.
    """
    cols, pair = [], set()
    for c in range(1, 1 << m):
        if c in pair or c in cols:
            continue
        new, ok = set(), True
        for d in cols:
            x = c ^ d
            if x == 0 or x in pair or x in new or x in cols:
                ok = False
                break
            new.add(x)
        if ok:
            cols.append(c)
            pair |= new
            if len(cols) == n:
                return cols
    return None


def min_resv_m(n):
    for m in range(2, 16):
        c = resv_cols(n, m)
        if c is not None:
            return m, c
    return None, None


def confound_profile(cols):
    """Smallest subset size whose columns XOR to zero. Larger is better."""
    n = len(cols)
    for size in (2, 3, 4):
        for S in itertools.combinations(range(n), size):
            x = 0
            for i in S:
                x ^= cols[i]
            if x == 0:
                return size
    return 5


def make_terms(N, M):
    out = []
    for i in range(N):
        s = ['I'] * N
        s[i] = 'Z'
        out.append(''.join(s))
    for i, j in itertools.combinations(range(N), 2):
        s = ['I'] * N
        s[i] = s[j] = 'Z'
        out.append(''.join(s))
    return out[:M]


def h_of(terms, th):
    return SparsePauliOp.from_list(list(zip(terms, th))).simplify()


def return_prob(terms, c_true, th, N):
    qc = QuantumCircuit(N)
    qc.h(range(N))
    qc.append(PauliEvolutionGate(h_of(terms, c_true), time=T), range(N))
    qc.append(PauliEvolutionGate(h_of(terms, th), time=-T), range(N))
    qc.h(range(N))
    return float(abs(Statevector(qc).data[0]) ** 2)


def sense(terms, c_true, center, N, m_row, cols, n_scratch=2):
    """One circuit on a design register of m_row+1 qubits with GIVEN columns."""
    M = len(terms)
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
        qc.append(PauliEvolutionGate(P[i], time=-(center[i] + R) * T), sysr)
        for b in range(m_row):
            if (cols[i] >> b) & 1:
                qc.cx(param[b], scr[s])
        qc.cx(param[m_row], scr[s])
        qc.append(PauliEvolutionGate(P[i], time=+2.0 * R * T).control(1),
                  [scr[s]] + list(sysr))
        qc.cx(param[m_row], scr[s])
        for b in range(m_row):
            if (cols[i] >> b) & 1:
                qc.cx(param[b], scr[s])
    qc.h(sysr)
    qc.measure(param, qc.cregs[0])
    qc.measure(sysr, qc.cregs[1])
    width = qc.num_qubits
    counts = BACKEND.run(transpile(qc, BACKEND, optimization_level=1),
                         shots=SHOTS).result().get_counts()
    tot, acc = 0, np.zeros(M)
    for bs, cnt in counts.items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        sysb, parb = parts[0], parts[1]
        ret = 1.0 if set(sysb) == {'0'} else 0.0
        tot += cnt
        if ret == 0.0:
            continue
        xb = parb[::-1]
        row = sum(1 << b for b in range(m_row) if b < len(xb) and xb[b] == '1')
        fold = 1 if (m_row < len(xb) and xb[m_row] == '1') else 0
        sg = np.array([_design_sign(row, fold, cols[i]) for i in range(M)])
        acc += ret * sg * cnt
    return acc / max(tot, 1) / R, width


def smeared_ref(terms, c_true, center, N, M):
    if 2 ** M <= MC_CAP:
        sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                        for v in range(2 ** M)])
    else:
        sig = np.random.default_rng(11).choice([-1.0, 1.0], size=(MC_CAP, M))
    E = np.array([return_prob(terms, c_true, center + R * s, N) for s in sig])
    return np.array([np.mean(E * sig[:, i]) for i in range(M)]) / R


def cosf(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d > 0 else 0.0


print("=" * 100)
print("Does resolution V recover v88's lost cosine, and what does it cost?")
print("=" * 100)
print("  shipped = _design_spec's Gray columns at minimum width (what v88 ran)")
print("  res-V   = greedy columns with no 3-term and no 4-term confounding")
print("  min |S| = smallest parameter subset whose columns XOR to zero")
print()
print(f"  {'N':>3}{'M':>4}  {'design':<9}{'m_row':>6}{'width':>7}"
      f"{'min|S|':>8}{'cos':>9}{'linear w':>10}")
print("  " + "-" * 62)

rng = np.random.default_rng(7)
for N, M in ((4, 8), (5, 12), (6, 16)):
    terms = make_terms(N, M)
    c_true = np.round(rng.uniform(-0.8, 0.8, M), 2)
    center = c_true + rng.uniform(-0.3, 0.3, M)
    ref = smeared_ref(terms, c_true, center, N, M)

    m_g, cols_g = gray_cols(M)
    g, w = sense(terms, c_true, center, N, m_g, cols_g)
    print(f"  {N:>3}{M:>4}  {'shipped':<9}{m_g:>6}{w:>7}"
          f"{confound_profile(cols_g):>8}{cosf(g, ref):>9.5f}{M + N:>10}",
          flush=True)

    m_v, cols_v = min_resv_m(M)
    if cols_v is not None:
        g2, w2 = sense(terms, c_true, center, N, m_v, cols_v)
        print(f"  {N:>3}{M:>4}  {'res-V':<9}{m_v:>6}{w2:>7}"
              f"{confound_profile(cols_v):>8}{cosf(g2, ref):>9.5f}{M + N:>10}",
              flush=True)
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  If res-V's cosine recovers toward 1 while its width stays well under")
print("  the linear register's M+N, the aliasing diagnosis is confirmed AND the")
print("  fix is worth porting into _design_spec as an optional extra-rows knob.")
print("  If the cosine does not move, aliasing was the wrong diagnosis and v88's")
print("  degradation has another cause that is still unidentified.")
print("  If the cosine recovers but width reaches M+N, the log register has no")
print("  advantage left on this task and the honest conclusion is that the")
print("  linear register is the right choice for Hamiltonian learning.")
