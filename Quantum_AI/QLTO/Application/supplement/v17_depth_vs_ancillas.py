"""Is QLTO's depth reducible, or is it structural?

v16 measured QLTO sensing circuits at depth 614-1976 against parameter-shift's
10-14. That is the finding that decides whether this is a NISQ method at all, so
it is worth knowing where the depth comes from.

QPE with k ancillas applies controlled-U^(2^a) for a = 0..k-1, so the evolution
time summed over the register is (2^k - 1) * tau0. The depth is therefore
EXPONENTIAL in the ancilla count, and k is the precision knob. k=4 was chosen
because it won the benchmark; this asks what it cost.

k=1 degenerates to a single controlled evolution - the Hadamard test. If that is
within a small factor of parameter-shift's depth, then there is a NISQ-viable
configuration of the method and the depth problem is a precision TRADE, not a
structural defect.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import AncillaRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
import benchmark as B
import nisq_v3


def build_k1(q, c, R, act):
    """The k=1 Hadamard-test sensing circuit, as sense_gradient builds it."""
    anc = AncillaRegister(1, 'anc')
    param = QuantumRegister(len(act), 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(len(act), 'c_param'),
                        ClassicalRegister(1, 'c_anc'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    qc.append(PauliEvolutionGate(q.H_sense, time=q.tau,
                                 synthesis=LieTrotter(reps=2)).control(1),
              [anc[0]] + list(sysr))
    qc.sdg(anc); qc.h(anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    return qc

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']

PROBLEMS = [
    ("H2",             B.get_h2_problem),
    ("MaxCut N=4",     lambda: B.get_maxcut_problem(4)),
    ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
]

print("=" * 92)
print("DEPTH vs ANCILLA COUNT — where QLTO's circuit depth comes from")
print("=" * 92)

for pname, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    M = ansatz.num_parameters
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, M)
    G = len(H.group_commuting(qubit_wise=True))

    a = ansatz.assign_parameters(c); a.measure_all()
    t = transpile(a, basis_gates=BASIS, optimization_level=1)
    d_ps, cx_ps, n_ps = t.depth(), t.count_ops().get('cx', 0), 2 * M * G

    print(f"\n  ===== {pname} | M={M} | G={G} | p-shift: "
          f"{n_ps} circuits, depth {d_ps}, {cx_ps} cx =====")
    print(f"  {'k':<5}{'circuits':>9}{'depth':>8}{'cx':>7}{'x p-shift depth':>17}"
          f"{'TOTAL cx':>10}{'x p-shift cx':>14}")
    print("  " + "-" * 70)

    for k in (1, 2, 3, 4):
        q = Q(ansatz, H, shot_budget=1024, num_ancillas=k)
        ds, cxs, n = [], [], 0
        for blk in q.layers:
            if not blk['params']:
                continue
            qc = (q._build_qpe_sensing_circuit(c, 0.6, blk['params']) if k > 1
                  else build_k1(q, c, 0.6, blk['params']))
            tq = transpile(qc, basis_gates=BASIS, optimization_level=1)
            ds.append(tq.depth()); cxs.append(tq.count_ops().get('cx', 0)); n += 1
        md, mc = float(np.mean(ds)), float(np.mean(cxs))
        print(f"  {k:<5}{n:>9}{int(md):>8}{int(mc):>7}{md/max(d_ps,1):>17.1f}"
              f"{int(sum(cxs)):>10}{sum(cxs)/max(n_ps*cx_ps,1):>14.2f}")

print()
print("  Depth should roughly double per ancilla: the controlled-U^(2^a) ladder")
print("  costs (2^k - 1)*tau0 of evolution. If k=1 lands within ~5x of")
print("  parameter-shift, the NISQ-viable setting exists and the k=4 default is")
print("  buying accuracy with coherence.")
