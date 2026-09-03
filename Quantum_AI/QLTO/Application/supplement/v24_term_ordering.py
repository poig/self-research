"""Is V3's Theta(N) depth just the ORDER the Hamiltonian terms are listed in?

v23 refuted the shared-ancilla explanation: controlled and uncontrolled evolution
both grow linearly, at a fixed 1.6x ratio, so the control is a constant factor and
not the critical path. But the uncontrolled numbers are suspiciously clean -
357, 621, 885, 1149, 1413, i.e. exactly +264 per two qubits - and a 1D chain has
no business costing linear depth.

WHY IT SHOULD BE CONSTANT. In sum_i (X_i X_i+1 + Y_i Y_i+1 + Z_i Z_i+1), the EVEN
bonds (0,1),(2,3),(4,5)... act on disjoint qubit pairs and therefore commute AND
can execute simultaneously. So can the odd bonds. A Trotter step is two parallel
layers, and its depth is O(1) in N no matter how many bonds there are.

WHY IT ISN'T. PauliEvolutionGate emits terms in the order they appear in the
SparsePauliOp, and benchmark.get_heisenberg_problem builds them as (0,1), (1,2),
(2,3), ... - a chain in which EVERY CONSECUTIVE PAIR SHARES A QUBIT. That is a
serial dependency chain of length N-1, and the transpiler cannot reorder across
it because it must respect the gate sequence it was given. The parallelism is
real but the ordering hides it.

THE FIX, if so, is free: sort the terms into disjoint layers before handing them
to PauliEvolutionGate. Same gates, same Trotter error (reordering within a single
Trotter step changes the product formula's error constant but not its order), just
a schedule the transpiler can exploit.

  depth Theta(N) -> Theta(1)     for all N, no extra qubits, no approximation

This tests the diagnosis by rebuilding the SAME operator with terms grouped into
disjoint layers and reading the depth scaling. It does NOT change gate count, so
it is a money fix and not a fidelity fix - fidelity is charged on cx, which is
unchanged.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp, Operator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']


def support(pauli):
    lbl = str(pauli)[::-1]
    return {i for i, c in enumerate(lbl) if c != 'I'}


def layer_sort(op):
    """Greedy partition into layers of mutually disjoint-support terms."""
    items = list(zip(op.paulis, op.coeffs))
    layers = []
    for p, c in items:
        s = support(p)
        for lay in layers:
            if not (s & lay['used']):
                lay['terms'].append((p, c)); lay['used'] |= s
                break
        else:
            layers.append({'terms': [(p, c)], 'used': set(s)})
    out = []
    for lay in layers:
        out.extend(lay['terms'])
    return SparsePauliOp.from_list([(str(p), c) for p, c in out]), len(layers)


def stats(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return t.depth(), t.count_ops().get('cx', 0)


print("=" * 96)
print("TERM ORDERING — is V3's linear depth a scheduling artefact?")
print("=" * 96)
print("  Same operator, same Trotter settings, terms regrouped into layers of")
print("  mutually disjoint support. Depth is the critical path, so a schedule the")
print("  transpiler can parallelise should collapse it.")
print()
print(f"  {'N':>4}{'T':>5}{'layers':>8}{'depth chain':>13}{'depth sorted':>14}"
      f"{'speedup':>9}{'cx chain':>10}{'cx sorted':>11}{'unitary err':>13}")
print("  " + "-" * 87)

rows = []
for N in (4, 6, 8, 10, 12):
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = B.get_heisenberg_problem(N)
    q = Q(ansatz, H, shot_budget=1024)
    Hs = q.H_sense
    T = len(Hs.paulis)
    a = q.num_ancillas - 1
    t_evo = (2 ** a) * q.tau0
    reps = int(max(1, (2 ** a) // 2, np.ceil(t_evo / 2.0)))
    syn = lambda: SuzukiTrotter(order=2, reps=reps)

    sysr = QuantumRegister(N, 'sys')
    qc_a = QuantumCircuit(sysr)
    qc_a.append(PauliEvolutionGate(Hs, time=t_evo, synthesis=syn()), list(sysr))
    d_a, cx_a = stats(qc_a)

    Hs2, nlay = layer_sort(Hs)
    qc_b = QuantumCircuit(sysr)
    qc_b.append(PauliEvolutionGate(Hs2, time=t_evo, synthesis=syn()), list(sysr))
    d_b, cx_b = stats(qc_b)

    # the two circuits are different product formulas; check they still agree
    # with each other to Trotter order, on the smallest cases only (2^N matrices)
    err = float('nan')
    if N <= 8:
        err = float(np.linalg.norm(Operator(qc_a).data - Operator(qc_b).data, 2))

    rows.append((N, d_a, d_b, cx_a, cx_b))
    print(f"  {N:>4}{T:>5}{nlay:>8}{d_a:>13}{d_b:>14}{d_a/max(d_b,1):>9.1f}"
          f"{cx_a:>10}{cx_b:>11}{err:>13.2e}", flush=True)

ns = np.array([r[0] for r in rows], float)
aa = np.polyfit(np.log(ns), np.log([r[1] for r in rows]), 1)[0]
ab = np.polyfit(np.log(ns), np.log([r[2] for r in rows]), 1)[0]
print()
print(f"  depth growth   chain order N^{aa:.2f}     layer-sorted N^{ab:.2f}")
print()
print("  A sorted exponent near 0 confirms the diagnosis: the depth was never the")
print("  physics, it was the term ORDER, and the fix costs nothing and holds for")
print("  every N. cx is unchanged by construction, so this is a MONEY fix only -")
print("  V3's fidelity problem is the gate count and is untouched.")
print()
print("  unitary err is the spectral distance between the two product formulas.")
print("  Reordering within a Trotter step changes the error CONSTANT, not the")
print("  order, so a small nonzero value is expected and correct - it is the same")
print("  approximation, not the same circuit.")
