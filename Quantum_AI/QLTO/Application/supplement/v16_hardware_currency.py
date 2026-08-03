"""What a gradient actually costs on hardware: circuits, depth, and two-qubit gates.

v15 showed QLTO's classical overhead is 40-100x parameter-shift's, dominated by
rebuild+transpile. That raised the obvious follow-up: circuit COUNT is only one
of the three things hardware charges for, and QLTO buys its low count by making
each circuit much deeper. So count the currency properly.

Three costs, and QLTO sits differently in each:

  CIRCUITS   fixed per-job overhead: load, calibration context, readout setup.
             QLTO wins here, by construction - one circuit serves all M.
  2Q GATES   the error budget. total = circuits * shots * cx_per_circuit is the
             proxy for accumulated infidelity, and depth * shots is the proxy
             for wall-clock QPU time. QLTO pays a controlled time evolution
             plus a W gate that parameter-shift does not pay at all.
  DECOHERENCE depth per circuit against T2. A 32-circuit job of depth 14 and a
             4-circuit job of depth 250 are not interchangeable even at equal
             total gate count - the deep one may simply not fit in coherence.

I report all three rather than picking the flattering one.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit_aer import AerSimulator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']
BACKEND = AerSimulator()
SHOTS = 1024


def stats(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=1)
    ops = t.count_ops()
    return t.depth(), ops.get('cx', 0)


PROBLEMS = [
    ("H2",             B.get_h2_problem),
    ("MaxCut N=4",     lambda: B.get_maxcut_problem(4)),
    ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
    ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
]

print("=" * 100)
print("HARDWARE CURRENCY PER GRADIENT — circuits, depth, two-qubit gates")
print("=" * 100)
print(f"  basis {BASIS}, opt level 1, {SHOTS} shots per circuit")

for pname, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=SHOTS)
    M = ansatz.num_parameters
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, M)

    # --- parameter-shift: one circuit per (component, sign, group)
    groups = H.group_commuting(qubit_wise=True)
    G = len(groups)
    a = ansatz.assign_parameters(c)
    a.measure_all()
    d_ps, cx_ps = stats(a)
    n_ps = 2 * M * G

    # --- QLTO: one sensing circuit per live block
    d_ql, cx_ql, n_ql = [], [], 0
    for blk in q.layers:
        if not blk['params']:
            continue
        qc = q._build_qpe_sensing_circuit(c, 0.6, blk['params'])
        d, cx = stats(qc)
        d_ql.append(d); cx_ql.append(cx); n_ql += 1

    tot_ps_cx = n_ps * cx_ps
    tot_ql_cx = sum(cx_ql)
    tot_ps_d = n_ps * d_ps
    tot_ql_d = sum(d_ql)

    print(f"\n  ===== {pname} | M={M} | G={G} | qubits {ansatz.num_qubits} =====")
    print(f"  {'method':<18}{'circuits':>9}{'depth/circ':>12}{'cx/circ':>9}"
          f"{'TOTAL cx':>11}{'TOTAL depth':>13}")
    print("  " + "-" * 74)
    print(f"  {'parameter-shift':<18}{n_ps:>9}{d_ps:>12}{cx_ps:>9}"
          f"{tot_ps_cx:>11}{tot_ps_d:>13}")
    print(f"  {'QLTO':<18}{n_ql:>9}{int(np.mean(d_ql)):>12}"
          f"{int(np.mean(cx_ql)):>9}{tot_ql_cx:>11}{tot_ql_d:>13}")
    print(f"  {'QLTO / p-shift':<18}{n_ql/n_ps:>9.3f}"
          f"{np.mean(d_ql)/max(d_ps,1):>12.1f}{np.mean(cx_ql)/max(cx_ps,1):>9.1f}"
          f"{tot_ql_cx/max(tot_ps_cx,1):>11.2f}{tot_ql_d/max(tot_ps_d,1):>13.2f}")

print()
print("  circuits < 1 is the win. TOTAL cx and TOTAL depth > 1 are the price.")
print("  depth/circ is the hard constraint: it must fit inside T2 regardless of")
print("  how favourable the totals look.")
