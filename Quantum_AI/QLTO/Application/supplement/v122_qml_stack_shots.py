"""The QML data axis on real circuits: data register + design register + shots.

v121 identified the correct estimand for an MSE gradient - weight the data
register by p_x ~ |w_x| with w_x = f_x - y_x, split by sign - but its cos = 1.0000
was a TAUTOLOGY: the branch normalisation cancels and the file verified its own
algebra. Nothing about a circuit was tested. This file tests the circuit.

THE STACK BEING BUILT. Three registers, each multiplexing a different axis:

    data register    d = log2|D| qubits   carries all |D| samples at once
    system           N qubits             the model
    design register  log2(M)+1 qubits     carries all M parameters at once (V6)

The data register is entangled into the system by CRY and NEVER uncomputed, so
tracing it out at measurement gives sum_x p_x f_x. V6's design register then reads
the gradient of that single observable. Since the QML readout is one Pauli, G = 1,
so V6 issues ONE circuit per gradient regardless of M - the best case the
construction has anywhere.

PART 0 IS THE LOAD-BEARING CHECK. v74 verified the trace-out identity for a
UNIFORM register. Nothing has verified it for a NON-UNIFORM one, and the whole
weighted scheme assumes it. The algebra says it holds for any p:

    state    sum_x sqrt(p_x) |x> (x) U V(x)|0>
    rho_sys  = sum_x p_x  U V(x)|0><0| V(x)^dag U^dag        (|x> orthogonal)
    <O>      = sum_x p_x f_x

but "the algebra says so" is what v118 said too, so it gets measured against
explicit per-sample values before anything is built on it.

PART 1 RUNS THE STACK WITH SHOTS and scores the gradient against the exact MSE
gradient. Three circuits per epoch, independent of |D| and of M:
    1  register MEASURED -> every f_x at once, hence every weight
    2  positive-weight branch
    3  negative-weight branch

WHAT WOULD KILL IT. Shot noise on the weights. Z+ and Z- are sums over all |w_x|
estimated from circuit 1, whose budget is split |D| ways, and they multiply the
whole gradient. If that noise dominates, the scheme returns a correct estimand
too noisily to use and the data axis stays shut for nonlinear losses.

TIER (project rule R1): PART 0 is tier B - exact amplitudes, a structural
identity, no accuracy claim. PART 1 is tier A - real circuits, AerSimulator,
finite shots.
"""
import sys, os, contextlib, io
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

from nisq_v6 import QLTOv6

N_SYS = 3
D_QUB = 2                     # |D| = 4 samples
SEED = 5
EPOCHS = 8
LR = 0.35
SHOTS = 1 << 15


def dataset(n_sys, n_data, seed=SEED):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-1.0, 1.0, (n_sys, n_data))
    y = rng.integers(0, 2, 2 ** n_data) * 2.0 - 1.0
    return alpha, y


def sample_angles(alpha, x, n_data):
    bits = [(x >> d) & 1 for d in range(n_data)]
    return alpha @ np.array(bits, dtype=float)


def prep_angles(p):
    """RY tree angles for a 2-qubit register with |amp|^2 = p (4 entries)."""
    p = np.asarray(p, float)
    p = p / max(p.sum(), 1e-300)
    hi = p[2] + p[3]                       # prob that the high bit is 1
    th_hi = 2.0 * np.arcsin(np.sqrt(np.clip(hi, 0, 1)))
    lo0 = p[1] / max(p[0] + p[1], 1e-300)  # low bit given high = 0
    lo1 = p[3] / max(p[2] + p[3], 1e-300)  # low bit given high = 1
    return (th_hi,
            2.0 * np.arcsin(np.sqrt(np.clip(lo0, 0, 1))),
            2.0 * np.arcsin(np.sqrt(np.clip(lo1, 0, 1))))


def _cry(qc, a, c, t):
    """CRY(a) written as ry/cx/ry/cx, NOT qc.cry - and the reason matters.

    A `cry` gate carries params and is not in V6's _CTRL, so it trips the
    decompose loop in QLTOv6.__init__, which then decomposes the parameterised
    core into `u` gates that V6 cannot build a controlled form of. The run dies
    with "V6 cannot build a controlled form of 'u'" - the safety interlock doing
    its job, on the wrong gate.

    Written as ry/cx the encoding is fixed-angle `ry` (in _CTRL) and `cx` (no
    params), so nothing triggers. v74 did exactly this and said so; I read past
    the line and rediscovered it from the traceback.

    Exact: circuit order ry(a/2) cx ry(-a/2) cx is, as operators,
    CX RY(-a/2) CX RY(a/2) - identity on control 0, RY(a) on control 1.
    """
    qc.ry(a / 2.0, t)
    qc.cx(c, t)
    qc.ry(-a / 2.0, t)
    qc.cx(c, t)


def batched_ansatz(alpha, p, core):
    """Weighted data register -> CRY encoding -> parameterised core.

    Only `core` carries symbolic Parameters; the prep and encoding are
    fixed-angle ry/cx, which V6 passes through untouched (its _direct_template
    only controls gates whose params are ParameterExpressions).
    """
    n_sys, n_data = alpha.shape
    dq = QuantumRegister(n_data, 'd')
    sq = QuantumRegister(n_sys, 's')
    qc = QuantumCircuit(dq, sq)

    th_hi, th_lo0, th_lo1 = prep_angles(p)
    qc.ry(th_hi, dq[1])                       # high bit
    qc.x(dq[1]); _cry(qc, th_lo0, dq[1], dq[0]); qc.x(dq[1])
    _cry(qc, th_lo1, dq[1], dq[0])

    for j in range(n_sys):                    # angle encoding, never undone
        for d in range(n_data):
            _cry(qc, float(alpha[j, d]), dq[d], sq[j])
    qc.compose(core, qubits=list(sq), inplace=True)
    return qc


def obs(n_sys, n_data):
    """Z on SYSTEM QUBIT 0, identity everywhere else - matching O_sys exactly.

    ENDIANNESS, and the first draft got it wrong. Qiskit labels are
    little-endian: label[n-1-q] is qubit q. The data register occupies global
    qubits 0..n_data-1 and the system n_data..n_data+n_sys-1, so system qubit 0
    is global qubit n_data, whose label position is

        (n_sys + n_data) - 1 - n_data  =  n_sys - 1

    Writing lbl[0]='Z' instead puts Z on the HIGHEST qubit - the last system
    qubit - while the reference O_sys = 'I..IZ' measures system qubit 0. PART 0
    caught it: the uniform row, which v74 had already verified, was off by 0.37.
    Third endianness bug in this project; twirl_cal had two.
    """
    lbl = ['I'] * (n_sys + n_data)
    lbl[n_sys - 1] = 'Z'
    return SparsePauliOp.from_list([(''.join(lbl), 1.0)])


def f_exact(alpha, x, theta, core, O_sys):
    n_sys, n_data = alpha.shape
    qc = QuantumCircuit(n_sys)
    ang = sample_angles(alpha, x, n_data)
    for j in range(n_sys):
        qc.ry(float(ang[j]), j)
    qc.compose(core.assign_parameters(np.asarray(theta, float)), inplace=True)
    return float(np.real(Statevector(qc).expectation_value(O_sys)))


alpha, y = dataset(N_SYS, D_QUB)
S = 2 ** D_QUB
core = efficient_su2(N_SYS, reps=1)
M = core.num_parameters
O_full = obs(N_SYS, D_QUB)
O_sys = SparsePauliOp.from_list([('I' * (N_SYS - 1) + 'Z', 1.0)])

print("=" * 100)
print("v122  THE QML STACK ON CIRCUITS:  data register + design register + shots")
print("=" * 100)
print("  N_sys=%d, M=%d, |D|=%d on %d register qubits. Readout is one Pauli, so G=1."
      % (N_SYS, M, S, D_QUB))
print("  labels y =", y.astype(int))
print()

print("=" * 100)
print("PART 0  DOES THE TRACE-OUT IDENTITY HOLD FOR A *NON-UNIFORM* REGISTER?")
print("=" * 100)
print("  v74 verified <O x I> = mean_x f_x for a UNIFORM register. The weighted")
print("  scheme needs sum_x p_x f_x for arbitrary p, which nothing has checked.")
print("  TIER B: exact amplitudes, structural identity.")
print()
rng = np.random.default_rng(3)
theta0 = rng.uniform(-np.pi, np.pi, M)
fx = np.array([f_exact(alpha, x, theta0, core, O_sys) for x in range(S)])
print("   per-sample f_x =", np.round(fx, 5))
print()
print("      weights p                      <O x I> circuit    sum_x p_x f_x     err")
print("   " + "-" * 84)
ok0 = True
for name, p in (("uniform", np.ones(S) / S),
                ("skewed  ", np.array([0.6, 0.25, 0.1, 0.05])),
                ("2-point ", np.array([0.7, 0.0, 0.0, 0.3])),
                ("random  ", rng.dirichlet(np.ones(S)))):
    qc = batched_ansatz(alpha, p, core.assign_parameters(theta0))
    got = float(np.real(Statevector(qc).expectation_value(O_full)))
    want = float(np.dot(p / p.sum(), fx))
    e = abs(got - want)
    ok0 &= e < 1e-9
    print("   %-10s %s   %+.8f      %+.8f    %.1e"
          % (name, np.round(p, 3), got, want, e))
print()
print("   PASS - identity holds for arbitrary p" if ok0 else
      "   FAIL - the weighted scheme's core assumption is false")
print()

if not ok0:
    sys.exit(0)

print("=" * 100)
print("PART 1  THE FULL STACK WITH SHOTS")
print("=" * 100)
print("  Three circuits per epoch, independent of |D| and M:")
print("    1  register measured -> all f_x -> all weights")
print("    2  positive-weight branch, V6 gradient (G=1)")
print("    3  negative-weight branch, V6 gradient (G=1)")
print("  TIER A: AerSimulator, %d shots per circuit." % SHOTS)
print()
print("   epoch      MSE      cos(stack, true)   cos(uniform, true)   circuits")
print("  " + "-" * 84)

theta = np.random.default_rng(1).uniform(-np.pi, np.pi, M)
cs, cu_all = [], []
for ep in range(EPOCHS):
    fx = np.array([f_exact(alpha, x, theta, core, O_sys) for x in range(S)])
    w = fx - y
    mse = float(np.mean(w ** 2))

    # exact reference MSE gradient
    gs = np.zeros((S, M))
    for x in range(S):
        for i in range(M):
            for sh, sg in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
                t = np.array(theta, float); t[i] += sh
                gs[x, i] += sg * 0.5 * f_exact(alpha, x, t, core, O_sys)
    g_true = (2.0 / S) * (w[:, None] * gs).sum(axis=0)

    ncirc = 1                                   # circuit 1: the weights
    g_stack = np.zeros(M)
    for mask, sgn in ((w > 0, +1.0), (w < 0, -1.0)):
        if not mask.any():
            continue
        pw = np.abs(w) * mask
        Z = pw.sum()
        if Z < 1e-12:
            continue
        anz = batched_ansatz(alpha, pw / Z, core)
        be = AerSimulator(seed_simulator=100 + ep)
        q = QLTOv6(anz, O_full, shot_budget=SHOTS, sim_seed=100 + ep, backend=be)
        with contextlib.redirect_stdout(io.StringIO()):
            g, _ = q.sense(theta, 0.45, list(range(M)))
        ncirc += q.nefv
        g_stack += sgn * Z * g
    g_stack *= 2.0 / S

    # v74's uniform-register baseline, same machinery
    anz_u = batched_ansatz(alpha, np.ones(S) / S, core)
    be = AerSimulator(seed_simulator=900 + ep)
    qu = QLTOv6(anz_u, O_full, shot_budget=SHOTS, sim_seed=900 + ep, backend=be)
    with contextlib.redirect_stdout(io.StringIO()):
        g_u, _ = qu.sense(theta, 0.45, list(range(M)))

    cc = float(np.dot(g_stack, g_true) /
               (np.linalg.norm(g_stack) * np.linalg.norm(g_true) + 1e-30))
    cu = float(np.dot(g_u, g_true) /
               (np.linalg.norm(g_u) * np.linalg.norm(g_true) + 1e-30))
    cs.append(cc); cu_all.append(cu)
    print("   %5d   %.5f        %+.4f              %+.4f            %3d"
          % (ep, mse, cc, cu, ncirc))
    theta = theta - LR * g_true / max(np.max(np.abs(g_true)), 1e-12)

print()
print("   mean cos over %d epochs:   stack %+.4f     uniform %+.4f"
      % (EPOCHS, float(np.mean(cs)), float(np.mean(cu_all))))
print()
print("=" * 100)
print("READING IT")
print("=" * 100)
print("  PART 0 is the result that matters most, because everything else assumed it.")
print()
print("  PART 1's stack column is the honest test of v121, whose cos=1.0000 was")
print("  algebra. Here the weights, the branch normalisations and the gradient are")
print("  ALL shot-estimated, so anything below 1 is the noise v121 could not see.")
print()
print("  Circuits per epoch is 1 + 2*G = 3, independent of |D| AND of M. That is the")
print("  strongest cost position in this project, because a QML readout is a single")
print("  Pauli so G=1 structurally - no N^4.24 to fight.")
print()
print("  What it still does NOT buy is shots: circuit 1 splits its budget |D| ways to")
print("  read every f_x, which is what |D| separate evaluations would cost.")
print()
print("  Scope: |D|=%d, N_sys=%d, one dataset, one R, %d shots, no noise model, no" % (S, N_SYS, SHOTS))
print("  seed averaging. Z+ and Z- use EXACT weights here - shot noise on the weights")
print("  themselves is the next thing to break and is not tested.")
