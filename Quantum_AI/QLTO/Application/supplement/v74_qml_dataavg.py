"""Does the v72 readout margin survive when the objective is a DATA-AVERAGED loss?

v72 located the whole of QLTO's advantage over SPSA in the readout: 5.66x at N=4
and 5.21x at N=6 at matched shots, the margin widening with budget, because the
bounded +-1 ancilla bit removes the cross-coordinate term |grad E|^2 - (d_iE)^2
that SPSA cannot escape. That was measured against a FIXED observable. QML trains
against a loss averaged over data, and whether the margin survives that change has
not been run here. The answer splits, structurally.

PART 1. LINEAR LOSS. For a feature map V(x) applied before the ansatz,

    L(theta) = (1/S) sum_x <0|V(x)^dag U(theta)^dag O U(theta) V(x)|0>

is linear in the per-sample expectation, hence the expectation of O on the mixed
input rho_D = (1/S) sum_x V(x)|0><0|V(x)^dag. Realise it by holding a data
register in uniform superposition, entangling it into the system by CRY, and never
touching the register again: it is traced out at measurement, so <O tensor I> IS
the batch mean. Sample x is the bit pattern of the register and its feature angle
on system qubit j is sum_d x_d alpha[j,d], a standard angle encoding of a binary
feature vector. The objective becomes a single observable on a larger register and
QLTO applies unmodified. This part re-runs v72's comparison on that object, so any
movement in the ratio is attributable to batching and to nothing else.

PART 2. NONLINEAR LOSS. Mean-squared error does not reduce that way:

    L = (1/S) sum_x (f_x - y_x)^2,  dL/dth = (2/S) sum_x (f_x - y_x) df_x/dth

reweights each sample by a factor depending on f itself, so no single observable
carries it and one shared readout cannot return it. What the readout does return
is the gradient of the linear surrogate (1/S) sum_x f_x. Since an optimiser needs
a direction rather than a value, the useful question is whether the two point the
same way, measured as a cosine along an actual descent trajectory where the
weights move as f moves.

WHAT WOULD CONFIRM THE TRANSFER: Part 1 ratio comparable to v72's 5.2-5.7x.
WHAT WOULD KILL IT: Part 1 ratio collapsing toward 1, meaning the readout
advantage was an artefact of a single sharp observable and evaporates under
averaging. That would take QML off the list and would be the first evidence
against the v72 mechanism itself, so it is worth as much as the other outcome.

TWO NOTES ON THE SETUP, both of which affect how the numbers should be read.
  * O is a single Pauli (Z on one system qubit, the standard QML readout), so
    G = 1. SPSA is charged 2 circuits per sigma sample against QLTO's 1 per block.
    That is a different cost regime from v72's Heisenberg observable, where G was
    large; the cross-coordinate mechanism under test does not depend on G, but the
    absolute circuit counts do.
  * The batched expectation equals the plain mean of per-sample expectations, so
    every exact and SPSA quantity is computed from S small N-qubit simulations
    rather than one (N+D)-qubit simulation. That is an identity, not an
    approximation, and it is what makes this affordable. The batched circuit is
    still what QLTO itself runs, since the claim is about that readout. Its
    encoding uses n_sys*n_data CRY gates deliberately: a per-sample controlled
    unitary needs multi-controlled rotations, and collapsing those into a dense
    UnitaryGate is worse, since Qiskit then Shannon-decomposes it into thousands
    of CX on every circuit build.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, RYGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
import nisq_v5


D_QUBITS = 2                       # 4 data samples
R = 0.45
REPEATS = 4
BUDGETS = (2 ** 15, 2 ** 17)
K_FACTORS = (0.5, 1.0, 2.0)


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def dataset(n_sys, n_data, seed=5):
    """alpha[j,d] are the encoding weights; sample x is the bit pattern of x, so
    its feature angle on qubit j is sum_d x_d alpha[j,d]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, (n_sys, n_data))


def sample_angles(alpha, x, n_data):
    bits = [(x >> d) & 1 for d in range(n_data)]
    return alpha @ np.array(bits, dtype=float)


def batched_circuit(n_sys, n_data, alpha):
    """Data register in superposition, angle encoding applied by CRY, never
    uncomputed. Sample x gets rotation sum_d x_d alpha[j,d] on system qubit j, so
    the encoding costs n_sys*n_data two-qubit gates and no multi-controls. A
    per-sample controlled unitary would need multi-controlled rotations, and
    collapsing those into a dense UnitaryGate is worse still, since Qiskit then
    Shannon-decomposes it into thousands of CX on every circuit build."""
    qc = QuantumCircuit(n_sys + n_data)
    dat = list(range(n_sys, n_sys + n_data))
    for q in dat:
        qc.h(q)
    for j in range(n_sys):
        for d in range(n_data):
            # CRY written out as ry/cx/ry/cx rather than qc.cry. Exact: on |0> the
            # two rotations cancel, on |1> the X conjugation flips the sign of the
            # second so they add to RY(a). Written this way because a `cry` gate is
            # outside _CTRL, which forces V5 to decompose the WHOLE circuit, and
            # that turns the variational ry/rz into `u` gates it cannot control.
            a = float(alpha[j, d])
            qc.ry(a / 2.0, j)
            qc.cx(dat[d], j)
            qc.ry(-a / 2.0, j)
            qc.cx(dat[d], j)
    qc.compose(efficient_su2(n_sys, reps=2), qubits=list(range(n_sys)), inplace=True)
    return qc


def batched_observable(n_sys, n_data):
    s = ["I"] * (n_sys + n_data)
    s[0] = "Z"
    return SparsePauliOp.from_list([("".join(s[::-1]), 1.0)])


def sample_circuits(n_sys, n_data, alpha):
    """Per-sample N-qubit circuits: the same encoding with x fixed, no controls.
    The batched expectation is exactly the mean over these."""
    out = []
    for x in range(2 ** n_data):
        ang = sample_angles(alpha, x, n_data)
        c = QuantumCircuit(n_sys)
        for j in range(n_sys):
            c.ry(float(ang[j]), j)
        c.compose(efficient_su2(n_sys, reps=2), inplace=True)
        out.append(c)
    return out


def sys_observable(n_sys):
    return SparsePauliOp.from_list([("".join((["I"] * (n_sys - 1)) + ["Z"]), 1.0)])


def per_sample_vals(circs, Om, theta):
    return np.array([float(np.real(np.conj(v) @ (Om @ v)))
                     for v in (Statevector(c.assign_parameters(theta)).data
                               for c in circs)])


def batch_val(circs, Om, theta):
    return float(np.mean(per_sample_vals(circs, Om, theta)))


def sample_grad(circ, Om, theta):
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            v = Statevector(circ.assign_parameters(t)).data
            g[i] += s * float(np.real(np.conj(v) @ (Om @ v))) / 2
    return g


def batch_grad(circs, Om, theta):
    return np.mean([sample_grad(c, Om, theta) for c in circs], axis=0)


def batch_val_sampled(circs, Om, Om2, theta, shots, rng):
    """Batch mean with honest shot noise; variance taken exactly from the
    statevector, no estimator subsidy. O is one Pauli so there is one group."""
    tot = 0.0
    for c in circs:
        v = Statevector(c.assign_parameters(theta)).data
        m1 = float(np.real(np.conj(v) @ (Om @ v)))
        m2 = float(np.real(np.conj(v) @ (Om2 @ v)))
        tot += m1 + rng.normal(0.0, np.sqrt(max(m2 - m1 * m1, 0.0) / max(shots, 1)))
    return tot / len(circs)


print("=" * 100)
print("(1)  DATA-AVERAGED LINEAR LOSS:  does the v72 readout margin survive batching?")
print("=" * 100)
print(f"  {2 ** D_QUBITS} samples on a {D_QUBITS}-qubit register held in superposition and never")
print("  uncomputed, so <O tensor I> is the batch mean. Same R and scoring as v72;")
print("  SPSA charged 2 circuits per sigma sample (G=1 here) and given its best K.")
print()
print(f"  {'Nsys':>5}{'M':>4}{'T total':>10}{'cos QLTO':>11}{'cos SPSA':>11}"
      f"{'1-cos QL':>11}{'1-cos SP':>11}{'ratio':>8}{'winner':>9}")
print("  " + "-" * 80)

for N in (4, 6):
    angles = dataset(N, D_QUBITS)
    circ = batched_circuit(N, D_QUBITS, angles)
    H = batched_observable(N, D_QUBITS)
    circs = sample_circuits(N, D_QUBITS, angles)
    Osys = sys_observable(N)
    Om = Osys.to_matrix()
    Om2 = (Osys @ Osys).simplify().to_matrix()

    M = circ.num_parameters
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = batch_grad(circs, Om, theta)

    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v5.QLTOv5(circ, H, shot_budget=1024, gradient_mode='direct')
    G = len(probe.groups)
    blocks = [b['params'] for b in probe.layers if b['params']]
    L = len(blocks)

    with contextlib.redirect_stdout(io.StringIO()):
        qs = [nisq_v5.QLTOv5(circ, H, shot_budget=1024, gradient_mode='direct',
                             sim_seed=700 + r) for r in range(REPEATS)]

    for T in BUDGETS:
        Sq = max(1, T // (G * L))
        cq = []
        for q in qs:
            q.shot_budget = int(Sq)
            gh = np.zeros(M)
            for act in blocks:
                gi, _ = q.sense(theta, R, act)
                gh += gi
            cq.append(cosine(gh, g_ex))

        best_sp = -2.0
        for kf in K_FACTORS:
            K = max(1, int(kf * M))
            Sp = T // (2 * G * K)
            if Sp < 1:
                continue
            cs = []
            for rep in range(REPEATS):
                rng = np.random.RandomState(9000 + rep)
                g_sp = np.zeros(M)
                for _ in range(K):
                    sig = rng.choice([-1.0, 1.0], size=M)
                    ep = batch_val_sampled(circs, Om, Om2, theta + R * sig, Sp, rng)
                    em = batch_val_sampled(circs, Om, Om2, theta - R * sig, Sp, rng)
                    g_sp += ((ep - em) / (2.0 * R)) * sig
                g_sp /= K
                cs.append(cosine(g_sp, g_ex))
            best_sp = max(best_sp, float(np.mean(cs)))

        mq, ms = float(np.mean(cq)), best_sp
        eq, es = max(1 - mq, 1e-9), max(1 - ms, 1e-9)
        win = 'QLTO' if eq < es else 'SPSA'
        if abs(eq - es) / max(eq, es) < 0.05:
            win = 'tie'
        print(f"  {N:>5}{M:>4}{T:>10}{mq:>11.4f}{ms:>11.4f}{eq:>11.5f}"
              f"{es:>11.5f}{es / eq:>8.2f}{win:>9}", flush=True)
    print("  " + "." * 80)

print()
print("  v72 on a fixed observable reached 5.66x at N=4 and 5.21x at N=6 at its")
print("  largest budget. A comparable ratio here means batching costs the readout")
print("  nothing and QML sits on the same footing as the other applications.")

print()
print("=" * 100)
print("(2)  NONLINEAR LOSS:  is the shared-readout direction usable for MSE?")
print("=" * 100)
print("  One shared readout returns grad of the linear surrogate mean_x f_x. The MSE")
print("  gradient reweights sample x by (f_x - y_x). They are different vectors; the")
print("  question is whether they point the same way, measured along a real descent")
print("  trajectory since the weights move as f moves.")
print()

N = 4
angles = dataset(N, D_QUBITS)
circs = sample_circuits(N, D_QUBITS, angles)
Om = sys_observable(N).to_matrix()
S = len(circs)
M = circs[0].num_parameters
theta = np.random.default_rng(11).uniform(-np.pi, np.pi, M)
labels = np.array([0.6, -0.6, 0.6, -0.6])[:S]

print(f"  {'epoch':>6}{'MSE':>10}{'|g_mse|':>11}{'|g_lin|':>11}"
      f"{'cos(g_lin, g_mse)':>20}")
print("  " + "-" * 60)
for ep in range(10):
    f = per_sample_vals(circs, Om, theta)
    gs = np.array([sample_grad(c, Om, theta) for c in circs])
    g_mse = (2.0 / S) * ((f - labels) @ gs)
    g_lin = gs.mean(axis=0)
    if ep % 3 == 0:
        print(f"  {ep:>6}{float(np.mean((f - labels) ** 2)):>10.5f}"
              f"{np.linalg.norm(g_mse):>11.5f}{np.linalg.norm(g_lin):>11.5f}"
              f"{cosine(g_lin, g_mse):>20.4f}", flush=True)
    theta = theta - 0.35 * g_mse

print()
print("  A cosine near 1 means the shared readout is a usable descent direction for")
print("  MSE and the primitive transfers with a bias caveat. Near 0 or negative means")
print("  nonlinear losses need their own construction. Note the linear surrogate does")
print("  not depend on the labels at all, so agreement is a property of the landscape")
print("  and not of the task, and is not evidence that the method solves the")
print("  supervised problem.")
