"""Does QLTO's design register work on QCNN and DisCoCat ansaetze? Built, not argued.

Both architectures are trained by gradient descent, so both need a gradient, and
QLTO produces gradients. That is the reason to expect a fit. The reason to CHECK
is that neither looks like efficient_su2, and V6's construction has a precise
requirement that is easy to state and easy to violate.

THE REQUIREMENT, exactly. For parameter i, _direct_template emits

    op(theta_i - R)   then   controlled-op(2R)  on the scratch wire holding sigma_i

which equals op(theta_i + sigma_i R) if and only if

    op(a) op(b) = op(a + b)

i.e. the gate is a ONE-PARAMETER ABELIAN GROUP. Every Pauli rotation is
(exp(-iaG/2) exp(-ibG/2) = exp(-i(a+b)G/2)), which is why _CTRL is
{rx, ry, rz, p, u1}. Anything decomposing into Pauli rotations therefore
qualifies, and both architectures below do. So the gate algebra is not the
question. The STRUCTURE is, and each poses a different one.

QCNN - WEIGHT SHARING. Cong, Choi & Lukin (Nature Physics 15, 1273, 2019) get
O(log N) parameters by SHARING each convolution weight across every site in its
layer. So one Parameter object appears at many places in ansatz.data. V6's loop
resolves gi = self._pidx[...] per occurrence, so a shared theta_i hits the SAME
column c = cols[pos[gi]] every time, and every occurrence is shifted by the same
sigma_i * R. That is exactly right for a shared weight - and the decoded marginal
is then d/dtheta_i of the whole circuit, which by the chain rule is the sum over
occurrences. The scratch-wire XOR trick (`c ^ prev[s]`) re-derives the column
each time the wire is reused in between, so distant occurrences still work.

  Predicted to work. Never tested, because every ansatz in this project so far
  has had one occurrence per parameter.

DisCoCat - POST-SELECTION. The Oxford/Quantinuum compositional model (Coecke,
Sadrzadeh & Clark) compiles a pregroup grammar into a circuit: word states are
parameterised preparations, and grammatical `cups` are Bell effects, compiled as
CX + H followed by POST-SELECTION on |0>. V6 measures the whole system register
after rotating into the observable's basis, so the cup qubits - which are outside
the observable's support and so are not rotated - are measured in Z. Filtering
the shot record on cup == 0 is then a classical post-processing step the decode
can absorb.

  Predicted to work, with the post-selection acceptance rate as a MULTIPLIER on
  the shot cost. That multiplier is DisCoCat's own scaling problem, not QLTO's.

WHAT IS MEASURED, AND WHAT IS NOT. For each architecture: M, G, circuits per
gradient, register width, and cos(V6 gradient, exact gradient). A high cosine
means the design register resolves every shared/post-selected parameter
correctly; a low one means the structure defeats the encoding and the reason is
the finding.

  THIS FILE TESTS THE GRADIENT AXIS ONLY. No data enters either circuit - qcnn()
  emits conv/pool layers and a Z observable, discocat() emits word states and
  cups. Both are BARE ANSAETZE. Nothing here says anything about data encoding,
  and QCNN does not solve that problem: its tree is the PROCESSING structure and
  acts on a state already in the register. Amplitude encoding still costs
  Theta(2^N) (v125) and angle encoding still fits only N features on N qubits.

  Of the three axes a working QML system needs - encoding, gradient, optimisation
  step - this closes the middle one for these two architectures and leaves the
  other two exactly where they were. See Part IV of ../RESEARCH_NOTES.md.

TIER (project rule R1): tier A for the V6 gradients - circuits on AerSimulator
with finite shots. The exact reference gradient is tier B (Statevector, central
difference) and is the reference only.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from nisq_v6 import QLTOv6

SHOTS = 32768
SEEDS = (0, 1, 2)
FD = 1e-4


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def exact_grad(anz, obs, theta):
    """Central finite difference on the exact expectation. TIER B reference.

    Finite difference rather than parameter-shift because a SHARED parameter
    appears at many sites, and the two-point shift rule is stated for a single
    occurrence. The difference quotient is agnostic to how often theta appears.
    """
    M = len(theta)
    g = np.zeros(M)
    for i in range(M):
        for s, sg in ((+FD, +1.0), (-FD, -1.0)):
            t = np.array(theta, float)
            t[i] += s
            qc = anz.assign_parameters(t)
            g[i] += sg * float(np.real(Statevector(qc).expectation_value(obs)))
    return g / (2.0 * FD)


# ---------------------------------------------------------------- QCNN

def qcnn(n_qubits, seed=0):
    """QCNN with SHARED convolution weights - the O(log N) parameter count.

    Unitary pooling: the pooling step is a parameterised two-qubit rotation after
    which one wire is simply dropped from later layers. No mid-circuit
    measurement, so the whole circuit stays unitary.
    """
    qc = QuantumCircuit(n_qubits)
    active = list(range(n_qubits))
    params = []
    layer = 0
    while len(active) > 1:
        # one shared (conv_a, conv_b, pool) triple for the ENTIRE layer
        ca = Parameter(f'c{layer}a'); cb = Parameter(f'c{layer}b')
        pl = Parameter(f'p{layer}')
        params += [ca, cb, pl]
        for a, b in zip(active[0::2], active[1::2]):
            qc.ry(ca, a); qc.ry(cb, b)          # SHARED across every pair
            qc.cx(a, b)
            qc.rz(pl, b)                         # SHARED pooling angle
            qc.cx(a, b)
        active = active[1::2]                    # drop the pooled-out wires
        layer += 1
    lbl = ['I'] * n_qubits
    lbl[n_qubits - 1 - active[0]] = 'Z'
    return qc, SparsePauliOp.from_list([(''.join(lbl), 1.0)]), active[0]


# ---------------------------------------------------------------- DisCoCat

def discocat(seed=0):
    """A 'noun verb noun' sentence circuit: word states joined by cups.

    Wires:  0 = noun1, 1 = verb-left, 2 = verb-right, 3 = noun2, 4 = sentence
    Cups (Bell effects) join noun1-verbLeft and verbRight-noun2, compiled as
    CX + H and post-selected on |00>. Wire 4 carries the sentence meaning.
    """
    qc = QuantumCircuit(5)
    ps = []
    for i, w in enumerate([0, 1, 2, 3, 4]):
        a = Parameter(f'w{i}a'); b = Parameter(f'w{i}b')
        ps += [a, b]
        qc.ry(a, w); qc.rz(b, w)
    qc.cx(2, 4)                       # verb couples into the sentence wire
    qc.ry(Parameter('v0'), 4)
    # cup 1: qubits 0,1   cup 2: qubits 2,3
    qc.cx(0, 1); qc.h(0)
    qc.cx(3, 2); qc.h(3)
    lbl = ['I'] * 5
    lbl[5 - 1 - 4] = 'Z'
    # post-selection wires, in little-endian bit positions
    return qc, SparsePauliOp.from_list([(''.join(lbl), 1.0)]), [0, 1, 2, 3]


print("=" * 100)
print("v133  QLTO ON QCNN AND DisCoCat")
print("=" * 100)
print("  V6 needs op(a)op(b) = op(a+b) - a one-parameter abelian group. Every")
print("  Pauli rotation is one, so the gate algebra is never the question. The")
print("  STRUCTURE is: weight sharing (QCNN) and post-selection (DisCoCat).")
print("  TIER A, %d shots. Exact reference is tier B (Statevector)." % SHOTS)
print()

print("-" * 100)
print("PART 1  QCNN - does the design register resolve SHARED weights?")
print("-" * 100)
print("     N   M   occurrences/param   G   circuits   width   cos(V6, exact)")
print("   " + "-" * 82)
ok1 = True
for N in (4, 8):
    anz, obs, out = qcnn(N)
    M = anz.num_parameters
    occ = {}
    for inst in anz.data:
        for p in inst.operation.params:
            if hasattr(p, 'parameters'):
                for q in p.parameters:
                    occ[q] = occ.get(q, 0) + 1
    cs = []
    for sd in SEEDS:
        th = np.random.default_rng(sd).uniform(-np.pi, np.pi, M)
        q = QLTOv6(anz, obs, shot_budget=SHOTS, sim_seed=20 + sd,
                   backend=AerSimulator(seed_simulator=20 + sd))
        with contextlib.redirect_stdout(io.StringIO()):
            g, _e = q.sense(th, 0.25, list(range(M)))
        cs.append(cosine(g, exact_grad(anz, obs, th)))
    t = q._direct_template(list(range(M)), q.groups[0])[0]
    c = float(np.mean(cs))
    ok1 &= c > 0.85
    if N == 8:
        qcnn_M8 = M
    print("   %3d  %3d        %2d-%-2d          %2d      %2d       %3d      %+.4f"
          % (N, M, min(occ.values()), max(occ.values()), len(q.groups),
             len(q.groups), t.num_qubits, c))
print()
print("   Parameter-shift on the N=8 circuit would be 2M = %d circuits."
      % (2 * qcnn_M8))
print()

print("-" * 100)
print("PART 2  DisCoCat - does it survive POST-SELECTION on the cups?")
print("-" * 100)
anz, obs, cups = discocat()
M = anz.num_parameters
print("   sentence circuit: %d qubits, %d params, cups post-selected on qubits %s"
      % (anz.num_qubits, M, cups))
print()
cs = []
for sd in SEEDS:
    th = np.random.default_rng(30 + sd).uniform(-np.pi, np.pi, M)
    q = QLTOv6(anz, obs, shot_budget=SHOTS, sim_seed=30 + sd,
               backend=AerSimulator(seed_simulator=30 + sd))
    with contextlib.redirect_stdout(io.StringIO()):
        g, _e = q.sense(th, 0.25, list(range(M)))
    cs.append(cosine(g, exact_grad(anz, obs, th)))
c2 = float(np.mean(cs))
t2 = q._direct_template(list(range(M)), q.groups[0])[0]
print("      G   circuits   width   cos(V6, exact)")
print("   " + "-" * 48)
print("     %2d      %2d       %3d      %+.4f"
      % (len(q.groups), len(q.groups), t2.num_qubits, c2))
print()
print("   NOTE: this measures the gradient WITHOUT post-selecting the shot record")
print("   - the cups are left as unitaries and the observable is read on the")
print("   sentence wire. That is the honest scope: it shows the design register")
print("   handles the circuit STRUCTURE. Post-selection is a classical filter on")
print("   the same shots and multiplies the shot cost by 1/P(accept), which is")
print("   DisCoCat's own scaling problem and is not measured here.")
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  QCNN: cos %s at N=4,8 with each parameter appearing at up to %d sites."
      % ("passes" if ok1 else "FAILS", max(occ.values())))
if ok1:
    print("  WEIGHT SHARING IS NOT A PROBLEM, and the reason is structural rather")
    print("  than lucky: gi = _pidx[param] is resolved per OCCURRENCE, so every")
    print("  site carrying theta_i gets the same column cols[pos[gi]] and hence")
    print("  the same sigma_i * R shift. The marginal then decodes d/dtheta_i of")
    print("  the whole circuit, which is the chain-rule sum over sites - exactly")
    print("  what a shared weight's gradient is.")
else:
    print("  Weight sharing DEFEATS the encoding. That is the finding and it would")
    print("  block QCNN outright - investigate before any QCNN work proceeds.")
print()
print("  G = 1 in both cases, because a QML readout is a single Pauli. So the")
print("  cost is ONE circuit per gradient against parameter-shift's 2M, and with")
print("  readout='qpe' (v132) it stays one circuit whatever the observable.")
print()
print("  THE REAL TENSION FOR QCNN IS NOT FIT, IT IS LEVERAGE. QCNN's whole point")
print("  is O(log N) parameters, and QLTO's advantage is 2M. Cheap encoding and")
print("  large-M leverage pull against each other: at N=8 QCNN the saving is")
print("  2M = %d circuits (M=%d), which is real but far from the 2M at M~2000 of"
      % (2 * qcnn_M8, qcnn_M8))
print("  a dense ansatz.")
print("  QLTO fits QCNN; QCNN is simply not where QLTO is worth the most.")
print()
print("  SCOPE. N=4,8 QCNN and one 5-qubit DisCoCat sentence, %d seeds, R=0.25,"
      % len(SEEDS))
print("  %d shots, no noise model, no hardware, no training run - this measures" % SHOTS)
print("  GRADIENT FIDELITY at a point, not convergence.")
