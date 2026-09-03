"""Track the free degree-0 energy against the true one across a real run.

v31 showed the degree-0 Walsh coefficient of the sensing shots is a biased
estimate of the centre energy, and that on Heisenberg the bias FLOORS at +0.35
independent of R - the QPE decode's Trotter error, which the gradient escapes
because it is a DIFFERENCE and the degree-0 term does not because it is an
ABSOLUTE energy.

That kills it as a replacement for the reported energy. It does not kill it as a
CONVERGENCE MONITOR, and those are different jobs:

    reported energy   needs the right VALUE, once, at the end
    convergence log   needs the right SHAPE, every epoch, to see progress

A systematic offset destroys the first and preserves the second - PROVIDED the
offset is stable as the state moves. The notes say the Trotter bias is
STATE-dependent, so that is exactly what cannot be assumed and has to be
measured over a trajectory rather than at one point.

The sequencing costs nothing extra. The sensing circuit measures the hypercube
around the CURRENT centre, before the walk moves it, so epoch e+1's first
sensing circuit already reports epoch e's post-walk parameters. A one-epoch
delay, zero circuits.

Measures per epoch: the free degree-0 estimate, the exact energy, their
difference, and the rank correlation of the two curves - which is what decides
whether "stop when it flattens" gives the same answer either way.
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


def degree0(q, counts):
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
EPOCHS, SHOTS = 20, 16384

print("=" * 90)
print("FREE LOG OVER A TRAJECTORY — degree-0 from sensing vs the exact energy")
print("=" * 90)
print("  The degree-0 value is read from the FIRST block's sensing circuit each")
print("  epoch, which costs nothing: that circuit is run for the gradient anyway.")

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=SHOTS, sim_seed=5)
    q.reset_shot_stream()
    BLK = [b['params'] for b in q.layers if b['params']]
    p = np.random.RandomState(42).uniform(-np.pi, np.pi, ansatz.num_parameters)

    free, true = [], []
    for ep in range(EPOCHS):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for bi, act in enumerate(BLK):
            qc = q._build_qpe_sensing_circuit(p, R, act)
            counts = q._run(qc)
            if bi == 0:
                free.append(degree0(q, counts))
                true.append(float(np.real(Statevector(
                    ansatz.assign_parameters(p)).expectation_value(q.H_sense))))
            g = q._decode_gradient_qpe(counts, p, act, R)
            p = q._execute_walk(p, 15, dt, R, act, g)

    free, true = np.array(free), np.array(true)
    off = free - true
    # rank correlation: does "stop when it flattens" fire at the same epoch?
    rf = np.argsort(np.argsort(free)); rt = np.argsort(np.argsort(true))
    rho = float(np.corrcoef(rf, rt)[0, 1])
    pear = float(np.corrcoef(free, true)[0, 1])

    print(f"\n  ===== {name} =====")
    print(f"  {'epoch':>7}{'free (deg-0)':>15}{'exact':>11}{'offset':>10}")
    print("  " + "-" * 43)
    for e in (0, 4, 9, 14, 19):
        print(f"  {e:>7}{free[e]:>15.4f}{true[e]:>11.4f}{off[e]:>+10.4f}")
    print(f"  offset: mean {off.mean():+.4f}  std {off.std():.4f}  "
          f"drift {off[-1] - off[0]:+.4f}")
    print(f"  Pearson r {pear:.4f}   Spearman rho {rho:.4f}   "
          f"exact improved {true[0] - true[-1]:+.4f}")

print()
print("  A SMALL offset std against a LARGE exact improvement means the free log")
print("  tracks convergence even though its value is wrong - usable as a monitor.")
print("  A drifting offset, or rho well below 1, means the bias moves with the")
print("  state and the free curve can flatten when the true one has not.")
