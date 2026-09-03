"""Can the energy log be read off the sensing ancilla instead of its own circuit?

run_walk spends one extra evaluation per epoch purely to report the energy - and
it is not the 1 circuit the notes call it. It is self.estimator.run(...), a full
expectation value, which on hardware costs G CIRCUITS, one per qubit-wise-
commuting group. At global mode that is G against the 2 the optimiser itself
needs, so the reporting can cost more than the optimisation.

The sensing shots already contain it. Each QPE shot returns (vertex x, decoded
energy E(theta_x)), so the DEGREE-0 Walsh coefficient - the plain mean of the
decoded energies - is available at zero extra cost by exactly the argument that
gives the gradient (T2: every Walsh coefficient is an expectation over the same
samples).

What it is NOT is E(theta_c). Expanding over the hypercube with E[sigma_i] = 0
and E[sigma_i sigma_j] = delta_ij:

    Ehat(empty) = E(theta_c) + (R^2/2) * sum_i d^2E/dtheta_i^2 + O(R^4)

so it is the centre energy plus HALF R-SQUARED TIMES THE TRACE OF THE BLOCK'S
DIAGONAL HESSIAN. And that bias is NOT separately measurable at one bit per
parameter, because sigma_i^2 = 1 identically folds the diagonal curvature into
the degree-0 term - the same degeneracy T9c found, which needs two radii to
break.

Two things worth knowing, so measure both:
  * how big the bias actually is at the shipping schedule R = 0.6*0.9^epoch,
    since R^2/2 falls to 0.0027 by epoch 20 and the bias may be irrelevant;
  * whether the ORDERING is preserved - a monitoring log only has to rank
    epochs, and a smooth positive offset preserves rank.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def degree0_from_sensing(q, center, R, act):
    """Mean decoded energy over the hypercube = the degree-0 Walsh coefficient."""
    qc = q._build_qpe_sensing_circuit(center, R, act)
    counts = q._run(qc)
    k = q.num_ancillas
    tot, n = 0.0, 0
    for bitstr, cnt in counts.items():
        parts = bitstr.split()
        if len(parts) != 2:
            continue
        m = int(parts[0], 2)
        phi = m / (2 ** k)
        if phi >= 0.5:
            phi -= 1.0
        tot += (-2.0 * np.pi * phi / (q.tau0 + 1e-12)) * cnt
        n += cnt
    return tot / max(n, 1)


PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4))]
RADII = (0.6, 0.4, 0.2, 0.1, 0.073)      # 0.073 = R at epoch 20
SHOTS = 32768

print("=" * 94)
print("FREE ENERGY LOG — degree-0 Walsh coefficient vs the exact centre energy")
print("=" * 94)
print("  H_sense is traceless, so compare against <H_sense>(theta_c), not <H>.")
print("  Predicted bias (R^2/2)*tr(diag Hessian) is shown against the measured one.")

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=SHOTS, sim_seed=11)
    act = [b['params'] for b in q.layers if b['params']][0]
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)

    sv = Statevector(ansatz.assign_parameters(c))
    exact = float(np.real(sv.expectation_value(q.H_sense)))

    # exact diagonal Hessian over the active block, by central difference
    h = 1e-4
    tr = 0.0
    for i in act:
        pp = c.copy(); pp[i] += h
        pm = c.copy(); pm[i] -= h
        ep = float(np.real(Statevector(ansatz.assign_parameters(pp)).expectation_value(q.H_sense)))
        em = float(np.real(Statevector(ansatz.assign_parameters(pm)).expectation_value(q.H_sense)))
        tr += (ep - 2 * exact + em) / h ** 2

    print(f"\n  ===== {name} | exact <H_sense>(c) = {exact:.5f} | "
          f"tr(diag Hess) = {tr:.4f} =====")
    print(f"  {'R':>7}{'degree-0':>12}{'bias':>10}{'predicted':>12}"
          f"{'rel err':>10}")
    print("  " + "-" * 51)
    for R in RADII:
        q.reset_shot_stream()
        d0 = degree0_from_sensing(q, c, R, act)
        bias = d0 - exact
        pred = 0.5 * R ** 2 * tr
        print(f"  {R:>7.3f}{d0:>12.5f}{bias:>+10.5f}{pred:>+12.5f}"
              f"{abs(bias / (abs(exact) + 1e-12)):>10.4f}")

print()
print("  If measured bias tracks the prediction, the free log is understood and")
print("  usable: at the shipping schedule R falls to 0.073 by epoch 20, where")
print("  R^2/2 = 0.0027 and the offset is negligible for monitoring.")
print()
print("  SCOPE. This replaces a HARDWARE log, saving G circuits per epoch. It does")
print("  NOT replace the benchmark's reported energy - the fairness audit requires")
print("  reporting to be noiseless and identical across methods, which is what")
print("  REPORT_ESTIMATOR is for and which costs no circuits at all.")
