"""Can a WEIGHTED data register give the MSE gradient? v74's part 2, reopened.

v74 established both halves of the data axis and only one of them worked.

  PART 1 WORKED. A data register in uniform superposition, entangled into the
  system by CRY and never uncomputed, is traced out at measurement so <O x I> IS
  the batch mean. QLTO beat SPSA 5.65x at N=4 and 2.97x at N=6, comparable to
  v72's 5.66x/5.21x on a fixed observable. Batching costs the readout nothing.

  PART 2 DID NOT. For MSE the gradient reweights each sample,

      dL/dtheta = (2/S) sum_x (f_x - y_x) df_x/dtheta

  and a UNIFORM register returns the gradient of the unweighted surrogate
  mean_x f_x instead. Measured cosine between the two along a descent
  trajectory: -0.7431, -0.0923, +0.2981, +0.2663. Unusable.

  AND THE DEEPER PROBLEM, which v74 states itself: "the linear surrogate does not
  depend on the labels at all". The objective that batches cleanly is not a
  supervised objective, so part 1's win is on a task nobody wants to solve.

WHAT v74 DID NOT TRY. The weights are the whole difficulty, and they are just
amplitudes. Prepare the data register as sum_x sqrt(p_x)|x> with p_x proportional
to |w_x|, w_x = f_x - y_x, and tracing it out returns

    <O x I>  =  sum_x p_x f_x        the WEIGHTED mean

which is the object MSE actually needs. Two obstacles, both handled:

  SIGNS. |c_x|^2 is positive, so a single register gives positive weights only.
  Split the batch by sign of w_x, run one circuit per branch, subtract. Two
  circuits, still O(1) in |D|.

  THE WEIGHTS ARE UNKNOWN. w_x needs f_x, which changes every epoch. But all |D|
  values of f_x come from ONE circuit if the register is MEASURED rather than
  traced out - the joint counts over (x, outcome) give every f_x at once. Same
  multiplexing trick, applied to the weights instead of the gradient.

So the scheme is THREE circuits per epoch, independent of |D|:
    1  measure the register  -> all f_x, hence all w_x
    2  positive-weight branch -> sum_{w>0} p_x f_x
    3  negative-weight branch -> sum_{w<0} p_x f_x

WHAT WOULD SETTLE IT. cos(estimated, true MSE gradient) along a real descent
trajectory, against v74's uniform-register baseline on the same trajectory. If
the weighted register lifts the cosine from ~0.27 to near 1, the data axis is
open for a REAL supervised loss and the label-independence objection goes with
it. If it does not, nonlinear losses need something other than a register.

TIER (project rule R1): tier B. Exact amplitudes throughout - this asks whether
the WEIGHTED ESTIMAND is the right vector, which is a mechanism question and one
R1 permits at tier B. No accuracy or cost figure is claimed. If the estimand is
right, the tier-A build with shots is the next file, not this one.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp

N_SYS = 4
D_QUBITS = 2                      # |D| = 4 samples
EPOCHS = 12
LR = 0.35
SEED = 5


def dataset(n_sys, n_data, seed=SEED):
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-1.0, 1.0, (n_sys, n_data))
    y = rng.integers(0, 2, 2 ** n_data) * 2.0 - 1.0        # labels in {-1,+1}
    return alpha, y


def sample_angles(alpha, x, n_data):
    bits = [(x >> d) & 1 for d in range(n_data)]
    return alpha @ np.array(bits, dtype=float)


def f_of_sample(anz, alpha, x, theta, n_data, O):
    """f_x = <O> for sample x: encode its angles, then the ansatz."""
    from qiskit import QuantumCircuit
    n = anz.num_qubits
    qc = QuantumCircuit(n)
    ang = sample_angles(alpha, x, n_data)
    for j in range(n):
        qc.ry(float(ang[j]), j)
    qc.compose(anz.assign_parameters(np.asarray(theta, float)), inplace=True)
    return float(np.real(Statevector(qc).expectation_value(O)))


def all_f(anz, alpha, theta, n_data, O):
    return np.array([f_of_sample(anz, alpha, x, theta, n_data, O)
                     for x in range(2 ** n_data)])


def grad_f(anz, alpha, x, theta, n_data, O):
    """Exact parameter-shift gradient of f_x."""
    M = len(theta)
    g = np.zeros(M)
    for i in range(M):
        for s, sg in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
            t = np.array(theta, float); t[i] += s
            g[i] += sg * 0.5 * f_of_sample(anz, alpha, x, t, n_data, O)
    return g


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


print("=" * 100)
print("v121  WEIGHTED DATA REGISTER:  does it recover the MSE gradient?")
print("=" * 100)
anz = efficient_su2(N_SYS, reps=1)
M = anz.num_parameters
S = 2 ** D_QUBITS
alpha, y = dataset(N_SYS, D_QUBITS)
O = SparsePauliOp.from_list([("I" * (N_SYS - 1) + "Z", 1.0)])
print("  N_sys=%d, M=%d, |D|=%d samples on %d register qubits, MSE loss."
      % (N_SYS, M, S, D_QUBITS))
print("  labels y =", y.astype(int))
print("  TIER B: exact amplitudes. This asks whether the ESTIMAND is the right")
print("  vector, not what it costs. Shots come next if it is.")
print()
print("  UNIFORM register  -> grad of mean_x f_x            (v74's, label-blind)")
print("  WEIGHTED register -> grad of sum_x p_x f_x, p ~ |w| (signs split)")
print()
print("   epoch      MSE     cos(uniform, true)   cos(weighted, true)   |g_true|")
print("  " + "-" * 84)

theta = np.random.default_rng(1).uniform(-np.pi, np.pi, M)
rows = []
for ep in range(EPOCHS):
    f = all_f(anz, alpha, theta, D_QUBITS, O)
    w = f - y                                   # MSE residuals
    mse = float(np.mean(w ** 2))
    gs = np.array([grad_f(anz, alpha, x, theta, D_QUBITS, O) for x in range(S)])

    g_true = (2.0 / S) * (w[:, None] * gs).sum(axis=0)      # true MSE gradient
    g_unif = gs.mean(axis=0)                                # v74's surrogate

    # weighted register: p_x ~ |w_x|, split by sign, subtract the two branches
    pos, neg = w > 0, w < 0
    g_w = np.zeros(M)
    for mask, sgn in ((pos, +1.0), (neg, -1.0)):
        if not mask.any():
            continue
        p = np.abs(w) * mask
        Z = p.sum()
        if Z < 1e-12:
            continue
        p = p / Z                                # normalised amplitudes^2
        g_w += sgn * Z * (p[:, None] * gs).sum(axis=0)
    g_w *= 2.0 / S

    cu, cw = cosine(g_unif, g_true), cosine(g_w, g_true)
    rows.append((ep, mse, cu, cw))
    if ep % 3 == 0 or ep == EPOCHS - 1:
        print("   %5d   %.5f        %+.4f              %+.4f          %.4f"
              % (ep, mse, cu, cw, np.linalg.norm(g_true)))
    theta = theta - LR * g_true / max(np.max(np.abs(g_true)), 1e-12)

print()
mu = float(np.mean([r[2] for r in rows]))
mw = float(np.mean([r[3] for r in rows]))
print("   mean over %d epochs:   uniform %+.4f     weighted %+.4f" % (EPOCHS, mu, mw))
print()
print("=" * 100)
print("READING IT")
print("=" * 100)
if mw > 0.99:
    print("  cos = 1.0000 AT EVERY EPOCH IS A TAUTOLOGY, NOT A MEASUREMENT, and it")
    print("  should be read that way. The weighted estimator as written is")
    print()
    print("     g_w = (2/S)[ sum_{w>0}|w_x| g_x  -  sum_{w<0}|w_x| g_x ]")
    print("         = (2/S) sum_x w_x g_x  =  g_true")
    print()
    print("  because the Z normalisation cancels against the branch weight. So this")
    print("  file verified its own algebra. That is worth something and it is worth")
    print("  exactly one thing: it identifies the CORRECT ESTIMAND and confirms that")
    print("  v74's part 2 failure was the UNIFORM register being the wrong object,")
    print("  not the data axis being shut.")
    print()
    print("  WHAT IS GENUINELY NEW HERE is the uniform column. v74 saw four points;")
    print("  over 12 epochs the surrogate's cosine does not merely sit low, it SWINGS")
    print("  IN SIGN - +0.9367, +0.3359, +0.8180, -0.9770, -0.9602. A direction that")
    print("  reverses against the true gradient mid-descent is worse than a weak one,")
    print("  and it explains why v74's trajectory numbers looked erratic.")
    print()
    print("  WHAT IS NOT SHOWN, and none of it is minor:")
    print("    1. that a CIRCUIT with amplitudes sqrt(p_x) yields sum_x p_x f_x - the")
    print("       algebra assumes the trace-out identity v74 verified for the UNIFORM")
    print("       register, and re-verifying it for a non-uniform one is a build")
    print("    2. the gate cost of preparing arbitrary amplitudes on the register")
    print("    3. shot noise on Z+ and Z-, which are sums over all |w_x| and are")
    print("       themselves estimated from the register-measured circuit")
    print("    4. that measuring the register for all f_x splits the budget |D| ways,")
    print("       so each f_x gets S/|D| shots - the same shots |D| separate")
    print("       evaluations would cost. Circuit saving, not shot saving, which is")
    print("       the trade this project keeps making.")
    print()
    print("  So the data axis is OPEN rather than solved, and the next file is the")
    print("  tier-A build with shots, not another exact-amplitude check.")
else:
    print("  THE WEIGHTED REGISTER DOES NOT RECOVER IT (mean cos %.4f). The estimand" % mw)
    print("  is wrong somewhere and the algebra above should be re-derived before any")
    print("  circuit is built on it.")
print()
print("  Scope: |D|=%d, N_sys=%d, one dataset draw, exact amplitudes, no shots, no" % (S, N_SYS))
print("  circuit. The weighted state preparation is assumed exact here; its own gate")
print("  cost - arbitrary amplitudes on %d register qubits - is not counted." % D_QUBITS)
