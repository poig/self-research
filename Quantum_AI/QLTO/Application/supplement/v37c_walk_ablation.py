"""Which part of the walk does the closed form miss? Ablate the circuit.

v37b wrote the walk's decode in closed form - anc=0 branch is exactly the
identity, anc=1 branch is a product of 2x2 rotations, decode is R(2P(x_i=1)-1)
under the anc=1-conditioned distribution - and reproduced v36's QUALITATIVE
findings (oscillatory in g, non-separable across coordinates) but missed
v36's numbers by 36.5% of range.

That gap has two possible causes and they point in opposite directions:

  the model is WRONG          an ordering or convention error, in which case
                              nothing derived from it can be trusted
  the model is INCOMPLETE     it omits the W gate (which entangles param with
                              sys) and the controlled H_sense imprint (which is
                              where energy enters), so the gap MEASURES what
                              those two contribute

Guessing between them is what this session has been doing wrong. So build the
ladder and measure each rung. Every arm uses the merged-walk step, identical
angles, identical decode; the arms differ only in which pieces are present.

  A  bare      h(param), k merged steps, h(anc)          <- exactly the closed form
  B  +W        A plus the W gate                          <- adds param-sys entanglement
  C  +imprint  B plus controlled exp(-i H_sense dt pi)    <- this IS the shipped walk

A must agree with the closed form to shot noise. If it does not, v37b is wrong
and its conclusions - including the aliasing diagnosis - are withdrawn. If it
does, then B-A and C-B are the honest decomposition of what the quantum parts of
the walk contribute, which is the question the notes have never answered.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit import transpile
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp
import nisq_v3

sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):    # v37b prints its own report
    from v37b_walk_closed_form import decode as closed_form


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


N, R, DT, KS, SHOTS, REPS = 4, 0.6, 0.5, 15, 65536, 3
H = heis(N)
ansatz = efficient_su2(N, reps=1)
M = ansatz.num_parameters

with contextlib.redirect_stdout(io.StringIO()):
    q = nisq_v3.QLTOv3(ansatz, H, shot_budget=SHOTS, sim_seed=17)
act = [b['params'] for b in q.layers if b['params']][0]
n = len(act)
centre = np.random.RandomState(7).uniform(-np.pi, np.pi, M)


def walk(gvec, use_w, use_imprint, seed):
    """One walk, with pieces switchable. Angles identical to the shipped walk."""
    grad_local = np.asarray(gvec, dtype=float)
    gain = 1.0 / np.sqrt(R)

    anc = AncillaRegister(1, 'anc')
    param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(ansatz.num_qubits, 'sys')
    c_param = ClassicalRegister(n, 'c_param')
    c_anc = ClassicalRegister(1, 'c_anc')
    qc = QuantumCircuit(anc, param, sysr, c_param, c_anc)

    qc.h(anc)
    qc.h(param)
    if use_w:
        qc.append(q.build_w_gate(param, sysr, centre, R, act),
                  list(param) + list(sysr))
    if use_imprint:
        qc.append(PauliEvolutionGate(q.H_sense, time=DT * np.pi,
                                     synthesis=LieTrotter(reps=1)).control(1),
                  [anc[0]] + list(sysr))

    for step in range(KS):
        s = (step + 0.5) / KS
        gamma = s * np.pi * DT
        beta = (1.0 - s) * np.pi * DT
        for i in range(n):
            al = grad_local[i] * gamma * 0.5 * np.pi * gain
            th = float(np.hypot(al, beta))
            ph = float(np.arctan2(beta, al))
            qc.ry(-ph, param[i])
            qc.crz(th, anc[0], param[i])
            qc.ry(ph, param[i])

    qc.h(anc)
    qc.measure(param, c_param)
    qc.measure(anc, c_anc)

    backend = q._backend_for(qc.num_qubits)
    t_qc = transpile(qc, backend, optimization_level=0)
    counts = backend.run(t_qc, shots=SHOTS, seed_simulator=seed).result().get_counts()

    blk = q._decode_walk(counts, centre, act, R)
    return blk - centre[act]


ARMS = [('A bare', False, False), ('B +W', True, False), ('C +imprint', True, True)]
grid = [-2.0, -1.5, -1.0, -0.6, -0.3, -0.15, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0]

print("=" * 96)
print("WALK ABLATION — which piece does the closed form miss?")
print("=" * 96)
print(f"  N={N}, n_active={n}, R={R}, dt={DT}, k={KS}, {SHOTS} shots x {REPS}")
print("  Arm C is the shipped walk. 'closed' is v37b, evaluated with no simulator.")
print()
print(f"  {'g_0':>8}{'closed':>11}" + "".join(f"{a:>13}" for a, _, _ in ARMS))
print("  " + "-" * (19 + 13 * len(ARMS)))

cols = {a: [] for a, _, _ in ARMS}
cf = []
for g0 in grid:
    gv = np.zeros(n); gv[0] = g0
    c = closed_form(gv, KS, DT, R)[0]
    cf.append(c)
    row = []
    for name, uw, ui in ARMS:
        vals = [walk(gv, uw, ui, 1000 + 37 * r)[0] for r in range(REPS)]
        v = float(np.mean(vals))
        cols[name].append(v)
        row.append(v)
    print(f"  {g0:>8.2f}{c:>11.5f}" + "".join(f"{v:>13.5f}" for v in row),
          flush=True)

cf = np.array(cf)
xs = np.array(grid)
print(f"\n  {'arm':>12}{'vs closed':>12}{'turns':>8}{'monotone':>10}{'|d| max':>10}")
print("  " + "-" * 52)
for name, _, _ in ARMS:
    y = np.array(cols[name])
    err = float(np.mean(np.abs(y - cf)))
    d = np.diff(y)
    turns = int(np.sum(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
    mono = bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))
    print(f"  {name:>12}{err:>12.5f}{turns:>8}{str(mono):>10}"
          f"{np.max(np.abs(y)):>10.4f}")

print()
print("  Arm A vs closed is the CORRECTNESS check on v37b: they are the same")
print("  circuit, so they must agree to shot noise or the derivation is wrong.")
print("  A->B is what the W gate's param-sys entanglement adds, B->C is what the")
print("  energy imprint adds. Whichever step moves the numbers is the one the")
print("  closed form was missing, and the one any explanation has to include.")
