"""Which probe statistic drives the R schedule: the magnitude ratio or the cosine?

I wired adaptive_radius to the magnitude ratio ||g(R)||/||g(R/2)||, on the
strength of v9b's calibration table. That was a mapping error. v9b compares two
BIT-LEVELS INSIDE ONE CIRCUIT, so both levels smear over the same coordinates by
the same amount and the T10 attenuation exp(-c R^2 n) cancels in their ratio,
leaving only the nonlinearity. Two SEPARATE circuits at R and R/2 smear by
different amounts, so their magnitude ratio carries BOTH effects, pulling in
opposite directions:

    attenuation   ||g(R)|| < ||g(R/2)||     pushes the ratio BELOW 1
    nonlinearity  cubic term inflates g(R)  pushes it ABOVE 1

Measured at R=0.6 the ratio came back 0.911-0.916 where the v9b mapping predicted
1.106, and the T10 attenuation prediction exp(-c R^2 n * 3/4) = 0.862 accounts
for it. So the diagnostic is dominated by the wrong term.

THE FIX FOLLOWS FROM THE SAME REASONING, and needs no new circuits: attenuation
is a UNIFORM SCALING of the gradient vector, so it cancels exactly in a COSINE
and not at all in a magnitude. cos(g(R), g(R/2)) therefore responds to direction
change - which is what nonlinearity causes and what the walk actually consumes -
and ignores the smearing.

This measures both statistics against the ground truth cos(g(R), grad E) that
neither can see in a real run, to set the threshold for adaptive_radius.
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

PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4))]
RADII = (0.2, 0.4, 0.6, 1.0, 1.5)

print("=" * 84)
print("PROBE STATISTIC — magnitude ratio vs cosine, against the truth")
print("=" * 84)
print("  probe cos / probe ratio are computable in a real run. true cos is not.")
print("  A usable diagnostic must TRACK true cos across R.")

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=65536, sim_seed=9)
    act = [b['params'] for b in q.layers if b['params']][0]
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)

    gx = np.zeros(len(act))
    for j, i in enumerate(act):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        gx[j] = 0.5 * (
            float(np.real(Statevector(ansatz.assign_parameters(pp)).expectation_value(H)))
            - float(np.real(Statevector(ansatz.assign_parameters(pm)).expectation_value(H))))

    print(f"\n  ===== {name} | block width n={len(act)} =====")
    print(f"  {'R':>6}{'probe cos':>12}{'true cos':>11}{'mag ratio':>11}")
    print("  " + "-" * 40)
    for R in RADII:
        q.reset_shot_stream()
        pc, mr = q.probe_linearity(c, R, act)
        q.reset_shot_stream()
        g = q.sense_gradient(c, R, act)[act]
        ng = np.linalg.norm(g)
        tc = float(g @ gx / (ng * np.linalg.norm(gx))) if ng > 1e-12 else 0.0
        print(f"  {R:>6.2f}{pc:>12.4f}{tc:>11.4f}{mr:>11.3f}")

print()
print("  The statistic to drive the schedule is whichever column moves WITH")
print("  'true cos'. If probe cos tracks it and mag ratio does not, adaptive_radius")
print("  must be rewired to the cosine - and the threshold read off this table.")
