"""Does multi-probe Hamiltonian learning actually help, or is one probe enough?

v6_hamlearn recovered 5 planted coefficients to 0.034 worst-case from ONE probe
(|+..+>) at ONE evolution time. The multi-probe idea says several experiments
should be fitted jointly because a single one may constrain some directions in
coefficient space only weakly.

That is a STATISTICAL question and it must be settled before any parallel circuit
is built: if one probe already suffices, sharing a param register across two
system copies buys nothing but qubits.

So test the statistics first, sequentially - P probes means P circuits per epoch
and the gradients averaged, which is mathematically identical to the parallel
per-shot linear combination for the GRADIENT. Only the circuit count differs, and
that is the engineering question to answer afterwards.

FAIR COMPARISON: hold TOTAL SHOTS PER EPOCH fixed. P probes get shots/P each, so
more probes buy diversity at the cost of precision per probe. If multi-probe still
wins under that constraint, the diversity is real and not just extra measurement.
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

N = 4
TERMS = ["ZZII", "IZZI", "IIZZ", "ZIII", "IIZI"]
C_TRUE = np.array([0.80, -0.55, 0.35, 0.60, -0.40])
M = len(TERMS)
BACKEND = AerSimulator(method='statevector')
P_OPS = [SparsePauliOp.from_list([(t, 1.0)]) for t in TERMS]

# (probe basis rotation, evolution time). 'x' = H on all (|+..+>),
# 'mix' = H on even qubits only, so some qubits start in |0> and see a different
# slice of the diagonal spectrum. 'y' = S H, a third independent basis.
PROBES = [('x', 1.0), ('mix', 1.0), ('x', 2.0), ('y', 1.5)]


def h_of(theta):
    return SparsePauliOp.from_list(list(zip(TERMS, theta))).simplify()


def prep(qc, reg, kind):
    if kind == 'x':
        qc.h(reg)
    elif kind == 'mix':
        for i in range(0, N, 2):
            qc.h(reg[i])
        for i in range(1, N, 2):
            qc.h(reg[i]); qc.s(reg[i])
    elif kind == 'y':
        qc.h(reg); qc.s(reg)


def unprep(qc, reg, kind):
    if kind == 'x':
        qc.h(reg)
    elif kind == 'mix':
        for i in range(1, N, 2):
            qc.sdg(reg[i]); qc.h(reg[i])
        for i in range(0, N, 2):
            qc.h(reg[i])
    elif kind == 'y':
        qc.sdg(reg); qc.h(reg)


def return_prob_exact(theta, probe):
    kind, T = probe
    qc = QuantumCircuit(N)
    prep(qc, qc.qubits, kind)
    qc.append(PauliEvolutionGate(h_of(C_TRUE), time=T), range(N))
    qc.append(PauliEvolutionGate(h_of(theta), time=-T), range(N))
    unprep(qc, qc.qubits, kind)
    return float(abs(Statevector(qc).data[0]) ** 2)


def sense(center, R, probe, shots):
    """Degree-1 Walsh marginal of the return bit for ONE probe."""
    kind, T = probe
    param = QuantumRegister(M, 'p'); sysr = QuantumRegister(N, 's')
    qc = QuantumCircuit(param, sysr, ClassicalRegister(M, 'cp'),
                        ClassicalRegister(N, 'cs'))
    qc.h(param)
    prep(qc, sysr, kind)
    qc.append(PauliEvolutionGate(h_of(C_TRUE), time=T), sysr)
    for k in range(M):
        qc.append(PauliEvolutionGate(P_OPS[k], time=-(center[k] - R) * T), sysr)
        qc.append(PauliEvolutionGate(P_OPS[k], time=-2.0 * R * T).control(1),
                  [param[k]] + list(sysr))
    unprep(qc, sysr, kind)
    qc.measure(param, qc.cregs[0]); qc.measure(sysr, qc.cregs[1])
    counts = BACKEND.run(transpile(qc, BACKEND, optimization_level=1),
                         shots=shots).result().get_counts()
    tot = 0; s1 = np.zeros(M)
    for bs, cnt in counts.items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        ret = 1.0 if set(parts[0]) == {'0'} else 0.0
        xb = parts[1][::-1]
        sg = np.array([1.0 if (i < len(xb) and xb[i] == '1') else -1.0
                       for i in range(M)])
        s1 += ret * sg * cnt
        tot += cnt
    return (s1 / max(tot, 1)) / R


def recover(n_probes, seed, epochs=40, total_shots=16384):
    probes = PROBES[:n_probes]
    per = max(total_shots // n_probes, 256)
    theta = C_TRUE + np.random.RandomState(seed).uniform(-0.6, 0.6, M)
    for ep in range(epochs):
        R = max(0.4 * (0.93 ** ep), 0.02)
        g = np.mean([sense(theta, R, pr, per) for pr in probes], axis=0)
        nrm = np.linalg.norm(g)
        if nrm > 1e-9:
            theta = theta + 0.7 * R * g / nrm
    return theta


print("=" * 82)
print("Multi-probe Hamiltonian learning, TOTAL SHOTS PER EPOCH HELD FIXED")
print("=" * 82)
print(f"  {M} coefficients, c_true = {np.array2string(C_TRUE, precision=2)}")
print(f"  probes available: {PROBES}")
print(f"  16384 shots/epoch split across P probes, 40 epochs, 4 seeds")
print()
print(f"  {'P':>3}{'shots/probe':>13}{'circuits/ep':>13}"
      f"{'max |err|':>12}{'rms |err|':>12}{'worst seed':>12}")
print("  " + "-" * 65)
for P in (1, 2, 3, 4):
    errs = []
    for seed in (0, 1, 2, 3):
        th = recover(P, seed)
        errs.append(np.abs(th - C_TRUE))
    errs = np.array(errs)
    print(f"  {P:>3}{16384//P:>13}{P:>13}{errs.mean(axis=0).max():>12.4f}"
          f"{np.sqrt((errs**2).mean()):>12.4f}{errs.max():>12.4f}", flush=True)
print()
print("  If P=1 is already as good, one probe suffices and the parallel-register")
print("  construction is unnecessary. If error falls with P at FIXED total shots,")
print("  probe diversity is doing real work and the parallel version is worth")
print("  building - it would deliver the same P gradients in ONE circuit.")
