"""Does the fan-out, once wired into nisq_v3, do what the prototype promised?

v25 measured a standalone prototype: depth N^0.55 -> N^0.03, unitary error
0.00e+00. That prototype built ONE Trotter step. The shipped version has to sit
inside _build_qpe_sensing_circuit, which runs kappa stages at different times and
rep counts, alongside the W gate and the inverse QFT - and it builds Suzuki-2
EXPLICITLY rather than calling SuzukiTrotter, because the fan-out needs per-term
control assignment that the synthesis object does not expose.

That last point is the risk. My hand-rolled symmetric formula is a valid Suzuki-2
but is NOT guaranteed to be Qiskit's term ordering, so the two are different
product formulas of the same order. The unitary distance measures how different,
and that is the number that decides whether fanout=True is a drop-in or a change
of approximation requiring its own accuracy validation.

Reports, against the shipped default (sorted, kappa=3, no fan-out):
  depth, cx, width       what it costs and buys
  unitary distance       is it the same circuit, or merely the same order
  gradient cosine        does the gradient still point the same way
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']


def strip(qc):
    out = qc.copy_empty_like()
    for instr in qc.data:
        if instr.operation.name not in ('measure', 'barrier'):
            out.append(instr)
    return out


def exact_grad(ansatz, H, c, act):
    g = np.zeros(len(act))
    for j, i in enumerate(act):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[j] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp)).expectation_value(H)))
                      - float(np.real(Statevector(ansatz.assign_parameters(pm)).expectation_value(H))))
    return g


PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
            ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
            ("Heisenberg N=8", lambda: B.get_heisenberg_problem(8))]
SHOTS = 16384
REP = 4

print("=" * 96)
print("FAN-OUT AS SHIPPED — inside _build_qpe_sensing_circuit, kappa=3, sorted")
print("=" * 96)
print(f"  {'problem':<17}{'arm':>9}{'width':>7}{'depth':>8}{'cx':>7}"
      f"{'depth x':>9}{'cx x':>7}{'cos vs exact':>14}")
print("  " + "-" * 78)

rows = []
for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)

    base = {}
    for tag, fo in (('default', False), ('fanout', True)):
        q = Q(ansatz, H, shot_budget=SHOTS, fanout=fo, sim_seed=7)
        act = [b['params'] for b in q.layers if b['params']][0]
        qc = q._build_qpe_sensing_circuit(c, 0.6, act)
        t = transpile(qc, basis_gates=BASIS, optimization_level=1)
        d, cx, w = t.depth(), t.count_ops().get('cx', 0), qc.num_qubits
        gx = exact_grad(ansatz, H, c, act)
        cs = []
        for r in range(REP):
            q.reset_shot_stream()
            g = q.sense_gradient(c, 0.6, act)[act]
            cs.append(float(g @ gx / (np.linalg.norm(g) * np.linalg.norm(gx) + 1e-15)))
        if tag == 'default':
            base = dict(d=d, cx=cx)
        print(f"  {name if tag == 'default' else '':<17}{tag:>9}{w:>7}{d:>8}{cx:>7}"
              f"{d / base['d']:>9.2f}{cx / max(base['cx'], 1):>7.2f}"
              f"{np.mean(cs):>14.4f}", flush=True)
        rows.append((name, tag, d))

print()
print("  UNITARY DISTANCE between the two constructions (same order, possibly")
print("  different term sequence). Computed where the matrix is affordable.")
print(f"  {'problem':<18}{'spectral dist':>15}{'verdict':>28}")
print("  " + "-" * 61)
for name, fn in PROBLEMS[:3]:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)
    qa = Q(ansatz, H, shot_budget=1024, fanout=False)
    qb = Q(ansatz, H, shot_budget=1024, fanout=True)
    act = [b['params'] for b in qa.layers if b['params']][0]
    Ua = Operator(strip(qa._build_qpe_sensing_circuit(c, 0.6, act))).data
    Ub = Operator(strip(qb._build_qpe_sensing_circuit(c, 0.6, act))).data
    na, nb = qa.ansatz.num_qubits, len(act)
    F = qb._fanout_width()
    # helpers sit at qubits k..k+F-2, so the helpers-|0> block is every index
    # whose helper bits are zero. Qiskit register 0 is least significant.
    k = qa.num_ancillas
    dim_rest = 2 ** (na + nb)
    idx = []
    for anc_ in range(2 ** k):
        for rest in range(dim_rest):
            idx.append(anc_ + (rest << (k + max(F - 1, 0))))
    idx = np.array(idx)
    ia = np.arange(Ua.shape[0])
    d = float(np.linalg.norm(Ua[np.ix_(ia, ia)] - Ub[np.ix_(idx, idx)], 2))
    verdict = 'IDENTICAL' if d < 1e-9 else 'same order, different formula'
    print(f"  {name:<18}{d:>15.3e}{verdict:>28}")

print()
print("  If the distance is 0 the flag is a drop-in and needs no accuracy re-test.")
print("  If it is small but nonzero, fanout is a DIFFERENT product formula of the")
print("  same order, and the cosine column is what says whether that matters.")
