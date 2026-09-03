"""The last gap: shot-estimate the WEIGHTS too, and see how it scales with |D|.

v122 ran the full QML stack on circuits and got cos 0.9768 against the exact MSE
gradient in 3 circuits per epoch. But it cheated in one place, stated in its own
scope line: Z+ and Z- used EXACT weights. The weights are w_x = f_x - y_x, they
change every epoch, and on hardware they must be measured.

They come from the same multiplexing trick. MEASURE the data register instead of
tracing it out, and the joint counts over (x, outcome) give every f_x at once -
|D| values from one circuit. So the scheme stays at 3 circuits. What it does not
stay free of is NOISE: each f_x sees only S/|D| of the shots, so its error grows
as sqrt(|D|/S), and those weights multiply the whole gradient.

THAT IS THE THING THIS FILE BREAKS OR CONFIRMS. Two questions:

  1. Does the stack survive shot-estimated weights at all?
  2. How does it degrade as |D| grows? Weight error ~ sqrt(|D|/S), so at fixed
     budget the data axis has a ceiling, and finding where it bites is the whole
     point of a DATA axis.

A GENERAL d-QUBIT WEIGHTED STATE PREP IS NEEDED, and it cannot use Qiskit's
state-preparation or `cry` gates: both carry params outside V6's _CTRL, which
trips the decompose loop in QLTOv6.__init__ and mangles the parameterised core
into `u` gates (v122 hit exactly that). So the prep is built from uniformly
controlled RY decomposed into fixed-angle ry and cx via the Gray-code Walsh
transform - the same transform this project's decode already runs, used here for
synthesis instead. PART 0 verifies it against target amplitudes before use.

TIER (project rule R1): PART 0 tier B - exact amplitudes, verifying a circuit
component. PART 1 tier A - real circuits, AerSimulator, finite shots, weights
included.
"""
import sys, os, contextlib, io
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

from nisq_v6 import QLTOv6

N_SYS = 3
EPOCHS = 6
LR = 0.35
SHOTS = 1 << 15
SEED = 5


def mux_ry(qc, thetas, controls, target):
    """Uniformly controlled RY: RY(thetas[j]) when the controls read j.

    Decomposed into fixed-angle ry + cx by the Gray-code Walsh transform, so
    nothing carries a symbolic param and V6 passes it through untouched.
    """
    k = len(controls)
    n = 1 << k
    # Moettoenen: alpha_i = 2^-k sum_j (-1)^{b_j . g_i} theta_j, where g_i is the
    # Gray code of i. Writing (-1)^{b_i . g_j} instead is the TRANSPOSE, which is
    # symmetric for k=1 and wrong for k>=2 - so d=2 passed while d=3,4 failed at
    # 4.3e-3 and 1.1e-1. Caught by PART 0.
    A = np.empty((n, n))
    for i in range(n):
        g = i ^ (i >> 1)
        for j in range(n):
            A[i, j] = (-1.0) ** (bin(j & g).count('1'))
    alpha = (A @ np.asarray(thetas, float)) / n
    for i in range(n):
        qc.ry(float(alpha[i]), target)
        c = min(((i + 1) & -(i + 1)).bit_length() - 1, k - 1)
        qc.cx(controls[c], target)


def prep_weights(qc, p, dq):
    """Prepare sum_x sqrt(p_x)|x> on dq using only ry/cx.

    ENDIANNESS, fourth time in this project. The statevector index x has qubit q
    as BIT q, so the most significant bit of x is qubit d-1, not qubit 0. The
    tree decides the most significant bit first, so level lvl must rotate
    dq[d-1-lvl] and be controlled on the already-decided HIGHER qubits
    dq[d-lvl .. d-1]. A first draft rotated dq[lvl] controlled on dq[0..lvl-1],
    which prepares the bit-reversed distribution: PART 0 measured
    max|amp^2 - p| of 0.30, 0.55, 0.11 at d = 2, 3, 4.

    Control ORDER matters too: mux_ry reads controls[i] as bit i of j, and j is
    indexed with its bit 0 being the LOWEST decided qubit, so controls must be
    [dq[d-lvl], ..., dq[d-1]] in that order.
    """
    d = len(dq)
    p = np.asarray(p, float)
    p = p / max(p.sum(), 1e-300)
    for lvl in range(d):
        blk = 1 << (d - lvl - 1)
        thetas = []
        for j in range(1 << lvl):
            lo = j * (blk << 1)
            tot = p[lo:lo + (blk << 1)].sum()
            hi = p[lo + blk:lo + (blk << 1)].sum()
            r = hi / tot if tot > 1e-300 else 0.0
            thetas.append(2.0 * np.arcsin(np.sqrt(np.clip(r, 0.0, 1.0))))
        target = dq[d - 1 - lvl]
        if lvl == 0:
            qc.ry(float(thetas[0]), target)
        else:
            mux_ry(qc, thetas, [dq[d - lvl + i] for i in range(lvl)], target)


def _cry(qc, a, c, t):
    qc.ry(a / 2.0, t); qc.cx(c, t); qc.ry(-a / 2.0, t); qc.cx(c, t)


def dataset(n_sys, n_data, seed=SEED):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1.0, 1.0, (n_sys, n_data)),
            rng.integers(0, 2, 2 ** n_data) * 2.0 - 1.0)


def sample_angles(alpha, x, n_data):
    return alpha @ np.array([(x >> d) & 1 for d in range(n_data)], dtype=float)


def batched(alpha, p, core, measure_register=False):
    n_sys, n_data = alpha.shape
    dq = QuantumRegister(n_data, 'd'); sq = QuantumRegister(n_sys, 's')
    qc = QuantumCircuit(dq, sq)
    prep_weights(qc, p, dq)
    for j in range(n_sys):
        for d in range(n_data):
            _cry(qc, float(alpha[j, d]), dq[d], sq[j])
    qc.compose(core, qubits=list(sq), inplace=True)
    return qc


def obs(n_sys, n_data):
    lbl = ['I'] * (n_sys + n_data)
    lbl[n_sys - 1] = 'Z'                     # system qubit 0, little-endian
    return SparsePauliOp.from_list([(''.join(lbl), 1.0)])


def f_exact(alpha, x, theta, core, O_sys):
    n_sys, n_data = alpha.shape
    qc = QuantumCircuit(n_sys)
    ang = sample_angles(alpha, x, n_data)
    for j in range(n_sys):
        qc.ry(float(ang[j]), j)
    qc.compose(core.assign_parameters(np.asarray(theta, float)), inplace=True)
    return float(np.real(Statevector(qc).expectation_value(O_sys)))


def f_from_shots(alpha, theta, core, backend, shots):
    """ONE circuit: measure register AND system, read every f_x from the joint."""
    n_sys, n_data = alpha.shape
    S = 2 ** n_data
    dq = QuantumRegister(n_data, 'd'); sq = QuantumRegister(n_sys, 's')
    cd = ClassicalRegister(n_data, 'cd'); cs = ClassicalRegister(1, 'cs')
    qc = QuantumCircuit(dq, sq, cd, cs)
    prep_weights(qc, np.ones(S) / S, dq)
    for j in range(n_sys):
        for d in range(n_data):
            _cry(qc, float(alpha[j, d]), dq[d], sq[j])
    qc.compose(core.assign_parameters(np.asarray(theta, float)),
               qubits=list(sq), inplace=True)
    qc.measure(dq, cd); qc.measure(sq[0], cs[0])
    counts = backend.run(transpile(qc, backend, optimization_level=1),
                         shots=shots).result().get_counts()
    num = np.zeros(S); den = np.zeros(S)
    for bit, c in counts.items():
        parts = bit.split()
        sysb, regb = parts[0], parts[1]
        x = int(regb, 2)
        num[x] += (1.0 if sysb[-1] == '0' else -1.0) * c
        den[x] += c
    return np.divide(num, den, out=np.zeros(S), where=den > 0), den


print("=" * 100)
print("v123  SHOT-ESTIMATED WEIGHTS:  the last thing v122 assumed away")
print("=" * 100)
print("  v122 got cos 0.9768 with EXACT weights. Here they come from circuit 1,")
print("  whose budget splits |D| ways, so weight error grows as sqrt(|D|/S).")
print()

print("=" * 100)
print("PART 0  DOES THE GENERAL d-QUBIT WEIGHTED PREP ACTUALLY PREPARE sqrt(p)?")
print("=" * 100)
print("  Built from uniformly controlled RY via the Gray-code Walsh transform,")
print("  in fixed-angle ry/cx only so V6 does not decompose the core into `u`.")
print()
print("     d   |D|   max |amp^2 - p|     verdict")
print("   " + "-" * 56)
ok0 = True
rng = np.random.default_rng(11)
for d in (2, 3, 4):
    S = 1 << d
    p = rng.dirichlet(np.ones(S))
    dq = QuantumRegister(d, 'd')
    qc = QuantumCircuit(dq)
    prep_weights(qc, p, dq)
    amp = np.abs(Statevector(qc).data) ** 2
    e = float(np.max(np.abs(amp - p)))
    ok0 &= e < 1e-9
    print("   %3d  %4d      %.2e        %s" % (d, S, e, "ok" if e < 1e-9 else "FAIL"))
print()
print("   PASS" if ok0 else "   FAIL - prep is wrong, nothing below is meaningful")
print()
if not ok0:
    sys.exit(0)

print("=" * 100)
print("PART 1  FULL STACK, WEIGHTS FROM SHOTS, SWEPT OVER |D|")
print("=" * 100)
print("  TIER A. %d shots per circuit; circuit 1's budget splits |D| ways." % SHOTS)
print()
print("    |D|   epoch      MSE     cos(stack)   weight err   shots/sample")
print("   " + "-" * 76)
summary = []
for d in (2, 3, 4):
    S = 1 << d
    alpha, y = dataset(N_SYS, d)
    core = efficient_su2(N_SYS, reps=1)
    M = core.num_parameters
    O_full = obs(N_SYS, d)
    O_sys = SparsePauliOp.from_list([('I' * (N_SYS - 1) + 'Z', 1.0)])
    theta = np.random.default_rng(1).uniform(-np.pi, np.pi, M)
    cs_all, we_all, wm_all = [], [], []
    for ep in range(EPOCHS):
        f_ex = np.array([f_exact(alpha, x, theta, core, O_sys) for x in range(S)])
        be0 = AerSimulator(seed_simulator=500 + ep)
        f_sh, den = f_from_shots(alpha, theta, core, be0, SHOTS)
        w_ex, w_sh = f_ex - y, f_sh - y
        # RMS is what sqrt(|D|/S) predicts; max is a max over |D| samples and is
        # systematically larger. Report both so the comparison in READING IT is
        # like-for-like - a first draft printed max against the per-sample
        # prediction, and the 1.6x/2.0x excess at |D|=8,16 looked like a scaling
        # break when it was extreme-value inflation.
        werr = float(np.sqrt(np.mean((w_sh - w_ex) ** 2)))
        wmax = float(np.max(np.abs(w_sh - w_ex)))
        mse = float(np.mean(w_ex ** 2))

        gs = np.zeros((S, M))
        for x in range(S):
            for i in range(M):
                for sh, sg in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
                    t = np.array(theta, float); t[i] += sh
                    gs[x, i] += sg * 0.5 * f_exact(alpha, x, t, core, O_sys)
        g_true = (2.0 / S) * (w_ex[:, None] * gs).sum(axis=0)

        g_stack = np.zeros(M)
        for mask, sgn in ((w_sh > 0, +1.0), (w_sh < 0, -1.0)):
            if not mask.any():
                continue
            pw = np.abs(w_sh) * mask
            Z = pw.sum()
            if Z < 1e-12:
                continue
            anz = batched(alpha, pw / Z, core)
            be = AerSimulator(seed_simulator=100 + ep)
            q = QLTOv6(anz, O_full, shot_budget=SHOTS, sim_seed=100 + ep,
                       backend=be)
            with contextlib.redirect_stdout(io.StringIO()):
                g, _ = q.sense(theta, 0.45, list(range(M)))
            g_stack += sgn * Z * g
        g_stack *= 2.0 / S

        c = float(np.dot(g_stack, g_true) /
                  (np.linalg.norm(g_stack) * np.linalg.norm(g_true) + 1e-30))
        cs_all.append(c); we_all.append(werr); wm_all.append(wmax)
        if ep in (0, EPOCHS - 1):
            print("   %4d   %5d   %.5f      %+.4f     %.4f   %.4f      %6d"
                  % (S, ep, mse, c, werr, wmax, int(den.mean())))
        theta = theta - LR * g_true / max(np.max(np.abs(g_true)), 1e-12)
    summary.append((S, float(np.mean(cs_all)), float(np.mean(we_all)),
                    float(np.mean(wm_all))))
    print("   %4d   mean            %+.4f     %.4f   %.4f"
          % (S, summary[-1][1], summary[-1][2], summary[-1][3]))
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("     |D|    mean cos      w rms     predicted sqrt(|D|/S)     w max   max/rms")
print("   " + "-" * 72)
for S, c, we, wm in summary:
    print("   %5d     %+.4f       %.4f          %.4f              %.4f    %.2f"
          % (S, c, we, np.sqrt(S / SHOTS), wm, wm / max(we, 1e-30)))
print()
print("  v122 with EXACT weights reached cos 0.9768 at |D|=4. The |D|=4 row here is")
print("  the same thing with the weights measured, so the drop is the cost of not")
print("  cheating. The |D| column is the data axis's actual scaling: the w rms")
print("  column is what sqrt(|D|/S) predicts and is the like-for-like comparison.")
print("  The w max column is a max over |D| draws, runs higher, and is the number")
print("  that decides whether the WORST sample's weight flips sign - which is what")
print("  actually breaks the two-branch positive/negative split.")
print()
print("  Circuits stay at 3 per epoch throughout - flat in |D| AND in M, since a QML")
print("  readout is one Pauli so G=1. That is the claim; the cos column is whether")
print("  it is worth anything.")
print()
print("  Scope: N_sys=%d, one dataset per |D|, one R, %d shots, no noise model, no"
      % (N_SYS, SHOTS))
print("  seed averaging, |D| only to 16. The per-sample gradients gs are exact - the")
print("  V6 estimate enters through g_stack alone.")
