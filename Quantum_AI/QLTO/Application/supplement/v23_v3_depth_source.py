"""WHERE is V3's depth, and is it structural or an artefact of one shared control?

V3's cost decomposes as circuits x shots x (rep_delay + depth*t_gate). Circuits
are 9 and constant; shots are Theta(N) and every method pays that. So the entire
excess is DEPTH, measured at 408N - 472, and the question is what sets the slope.

TWO CANDIDATE CAUSES, and they have opposite implications:

  (a) INHERENT. Simulating T = Theta(N) Hamiltonian terms takes Theta(N) gates,
      so the depth is the physics and cannot be removed.
  (b) SERIALISATION. A 1D chain's bonds mostly COMMUTE - all even bonds are
      disjoint, all odd bonds are disjoint - so an uncontrolled Trotter step has
      depth O(1) in N, applying N/2 bonds in parallel. But every gate in the QPE
      ladder is controlled on the SAME ancilla, and gates sharing a qubit cannot
      run in parallel. So the ancilla would be a critical path forcing Theta(N)
      depth out of an O(1)-depth operation.

These notes already assert (b) for a different circuit - the W-dagger analysis
concluded "depth is untouched because the ancilla is the critical path: it takes
part in all 2*n*k controlled gates while each param qubit sees only 2k". If that
generalises to the QPE ladder, V3's depth is an IMPLEMENTATION artefact and the
fix is ancilla fan-out: copy the control onto F ancillas with CNOTs (legitimate,
the control is diagonal so copying is not cloning), let disjoint terms proceed in
parallel, uncompute. Depth Theta(N) -> Theta(log N), for ALL N, at F extra qubits.

THE TEST. Build the same evolution twice - once controlled, once not - and read
the depth scaling of each. Cause (a) predicts both grow linearly. Cause (b)
predicts the uncontrolled one is flat and only the controlled one grows.

This measures the DIAGNOSIS, not the fix. It says nothing about gate COUNT, which
is what fidelity is charged on and which fan-out does not reduce.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.circuit import AncillaRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']


def stats(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return t.depth(), t.count_ops().get('cx', 0)


print("=" * 96)
print("WHERE V3's DEPTH COMES FROM — controlled vs uncontrolled evolution")
print("=" * 96)
print("  One Trotter layer of the sensing evolution at a=kappa-1 (the longest),")
print("  built identically except for .control(1). Heisenberg chain.")
print()
print(f"  {'N':>4}{'T terms':>9}{'uncontrolled':>14}{'controlled':>12}"
      f"{'ratio':>8}{'unctrl cx':>11}{'ctrl cx':>9}{'cx ratio':>10}")
print("  " + "-" * 79)

rows = []
for N in (4, 6, 8, 10, 12):
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = B.get_heisenberg_problem(N)
    q = Q(ansatz, H, shot_budget=1024)
    T = len(q.H_sense.paulis)
    a = q.num_ancillas - 1
    t_evo = (2 ** a) * q.tau0
    reps = int(max(1, (2 ** a) // 2, np.ceil(t_evo / 2.0)))

    sysr = QuantumRegister(N, 'sys')
    anc = AncillaRegister(1, 'anc')

    qc_u = QuantumCircuit(sysr)
    qc_u.append(PauliEvolutionGate(q.H_sense, time=t_evo,
                synthesis=SuzukiTrotter(order=2, reps=reps)), list(sysr))
    d_u, cx_u = stats(qc_u)

    qc_c = QuantumCircuit(anc, sysr)
    qc_c.append(PauliEvolutionGate(q.H_sense, time=t_evo,
                synthesis=SuzukiTrotter(order=2, reps=reps)).control(1),
                [anc[0]] + list(sysr))
    d_c, cx_c = stats(qc_c)

    rows.append((N, d_u, d_c, cx_u, cx_c))
    print(f"  {N:>4}{T:>9}{d_u:>14}{d_c:>12}{d_c/max(d_u,1):>8.1f}"
          f"{cx_u:>11}{cx_c:>9}{cx_c/max(cx_u,1):>10.1f}", flush=True)

ns = np.array([r[0] for r in rows], float)
au = np.polyfit(np.log(ns), np.log([r[1] for r in rows]), 1)[0]
ac = np.polyfit(np.log(ns), np.log([r[2] for r in rows]), 1)[0]
gu = np.polyfit(np.log(ns), np.log([r[3] for r in rows]), 1)[0]
gc = np.polyfit(np.log(ns), np.log([r[4] for r in rows]), 1)[0]

print()
print(f"  growth exponents   depth: uncontrolled N^{au:.2f}, controlled N^{ac:.2f}")
print(f"                     cx   : uncontrolled N^{gu:.2f}, controlled N^{gc:.2f}")
print()
print("  If the uncontrolled depth exponent is ~0 and the controlled one ~1, the")
print("  Theta(N) depth is the SHARED ANCILLA serialising an operation that is")
print("  intrinsically parallel, and ancilla fan-out removes it for all N.")
print("  If both are ~1, the depth is the physics and no fan-out helps.")
print()
print("  NOTE the cx columns. Fan-out changes SCHEDULING, not gate count, so")
print("  whatever the cx exponent is, it is unchanged by the fix - and cx is what")
print("  fidelity is charged on. A depth fix does not revive V3 on noisy hardware.")
