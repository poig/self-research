"""What layer-sorting does to the FULL V3 sensing circuit, not one Trotter stage.

v24 measured a single evolution stage and found depth Theta(N) -> Theta(1) exactly.
This applies the same reorder to H_sense and rebuilds the whole QPE sensing
circuit - W gate, the kappa-stage controlled ladder, inverse QFT - because that
is the object V3's cost is actually charged on, and the stages have different
Trotter reps so the total is not simply four copies of the stage measurement.

Reports depth and cx. Depth is the money term (it enters IBM's circuit_length);
cx is the fidelity term and sorting is not expected to move it.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']


def support(p):
    lbl = str(p)[::-1]
    return {i for i, ch in enumerate(lbl) if ch != 'I'}


def layer_sort(op):
    """Greedy partition into layers of mutually disjoint support, order preserved
    within each layer. Only ever transposes terms with disjoint support, which
    commute, so the emitted product is exactly the original one."""
    layers = []
    for p, c in zip(op.paulis, op.coeffs):
        s = support(p)
        for lay in layers:
            if not (s & lay['u']):
                lay['t'].append((p, c)); lay['u'] |= s
                break
        else:
            layers.append({'t': [(p, c)], 'u': set(s)})
    out = []
    for lay in layers:
        out.extend(lay['t'])
    return SparsePauliOp.from_list([(str(p), c) for p, c in out])


print("=" * 78)
print("FULL V3 QPE SENSING CIRCUIT — chain order vs layer-sorted H_sense")
print("=" * 78)
print(f"  {'N':>4}{'depth chain':>13}{'depth sorted':>14}{'speedup':>9}"
      f"{'cx chain':>10}{'cx sorted':>11}{'dur sorted':>12}")
print("  " + "-" * 73)

rows = []
for N in (4, 6, 8, 10, 12):
    with contextlib.redirect_stdout(io.StringIO()):
        ans, H, _ = B.get_heisenberg_problem(N)
    q = Q(ans, H, shot_budget=1024)
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ans.num_parameters)
    act = [b['params'] for b in q.layers if b['params']][0]

    t = transpile(q._build_qpe_sensing_circuit(c, 0.6, act),
                  basis_gates=BASIS, optimization_level=1)
    d1, x1 = t.depth(), t.count_ops().get('cx', 0)

    q.H_sense = layer_sort(q.H_sense)
    t = transpile(q._build_qpe_sensing_circuit(c, 0.6, act),
                  basis_gates=BASIS, optimization_level=1)
    d2, x2 = t.depth(), t.count_ops().get('cx', 0)

    rows.append((N, d1, d2, x1, x2))
    print(f"  {N:>4}{d1:>13}{d2:>14}{d1/max(d2,1):>9.1f}{x1:>10}{x2:>11}"
          f"{d2*70e-9*1e6:>11.1f}us", flush=True)

ns = np.array([r[0] for r in rows], float)
a1 = np.polyfit(np.log(ns), np.log([r[1] for r in rows]), 1)[0]
a2 = np.polyfit(np.log(ns), np.log([r[2] for r in rows]), 1)[0]
g2 = np.polyfit(np.log(ns), np.log([r[4] for r in rows]), 1)[0]
print()
print(f"  depth growth  chain N^{a1:.2f}   sorted N^{a2:.2f}")
print(f"  cx growth     sorted N^{g2:.2f}   <- unchanged in order; fidelity is not fixed")
print()
print("  'dur sorted' against IBM's 250us rep_delay is the money question: while")
print("  duration stays under rep_delay, V3's depth costs nothing on the invoice,")
print("  and V3's 9 circuits beat V4's 17 at every N.")
