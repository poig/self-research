"""Do landscape features get THINNER as the barren-plateau regime is entered?

v99 established the walk's mechanism and then measured it not engaging:

    "the open question is NOT a better mixer. It is whether an ansatz can be
     designed whose landscape has tall THIN barriers. Measured width on
     efficient_su2(reps=1) Heisenberg N=4/N=6 was 0.96 / 0.99 of the path -
     WIDE, so the mechanism does not engage there."

THAT MEASUREMENT WAS TAKEN IN THE WRONG REGIME, and this file tests whether it
was. efficient_su2 at reps=1, N=4 and N=6 is a SHALLOW, LOW-PARAMETER ansatz -
precisely the regime with NO barren plateau. Barren plateaus arrive with width
and depth. So v99 measured feature width exactly where the pathology it would
have to exploit does not exist.

WHY THIS MATTERS, and the two roles must not be conflated. QLTO has two separate
components with separate jobs:

    the SENSING oracle (V3/V5/V6)   job: COST. G circuits instead of 2MG,
                                    M parameters on log M qubits. v89 proves the
                                    smoothing attenuates every Fourier component,
                                    |grad E_R| <= |grad E| - the sensing oracle
                                    never claimed to help a plateau and provably
                                    does not.

    the WALK (V3's quantum walk)    job: THE LANDSCAPE. v99 measured the
                                    mechanism real - classical annealing costs
                                    ~exp(height), tunnelling ~exp(width), and
                                    classical success collapses 1.00 -> 0.00 over
                                    height 2 -> 20 while quantum transmission
                                    stays flat.

Applying the sensing result to a landscape question is a category error. Plateau
escape is the WALK's job and only the walk's.

THE HYPOTHESIS. The barren-plateau literature does not describe a flat desert
alone: where plateaus exist, minima sit in exponentially NARROW GORGES. Flat
almost everywhere, with thin deep features. That is not an obstacle to a walk -
it is the exact shape a walk is good at, because tunnelling cost scales with
WIDTH and a narrow gorge is narrow by construction. Gradient descent fails there
for the same reason: no gradient to follow on the flat part.

So the prediction is that feature width should FALL as N and depth grow, in
lockstep with the gradient variance falling. If it does, v99's 0.96/0.99 is a
small-ansatz artefact and the walk's regime is the large one.

WHAT IS MEASURED, per (N, reps):

    gradient variance   Var over random theta of dE/dtheta_0 - the standard
                        barren-plateau indicator; exponential decay in N is the
                        signature.
    feature width       Take a random 1-D cut theta(t) = theta_a + t(theta_b -
                        theta_a), t in [0,1]. Sample E(t), subtract the mean, and
                        take the normalised autocorrelation. The lag at which it
                        first falls below 1/e, as a FRACTION OF THE PATH, is the
                        characteristic feature width - the same "of the path"
                        units v99 quoted.

TIER (project rule R1): tier B. Exact energies from Statevector on built
circuits, no shots. This asks a STRUCTURAL question about the landscape - how
wide its features are - which R1 permits at tier B. No accuracy or cost figure is
claimed, and no walk is run: this measures whether the walk's PRECONDITION is met
at scale, not whether the walk wins. If width falls, building the walk in that
regime is the tier-A follow-up.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector

GRID = 512          # samples along the 1-D cut
CUTS = 12           # random cuts averaged per configuration
VAR_SAMPLES = 200   # random theta for the gradient-variance estimate
FD = 1e-4


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in 'XYZ':
            s = ['I'] * N
            s[i] = s[i + 1] = p
            ops.append((''.join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def energy(anz, H, theta):
    """H is a SparsePauliOp (local cost) or the string 'global'.

    'global' is C = |<0...0|psi>|^2, the standard global cost in the
    barren-plateau literature. It is computed from the amplitude directly rather
    than from a Pauli decomposition, which would need 2^N terms.
    """
    sv = Statevector(anz.assign_parameters(theta))
    if isinstance(H, str):
        return float(abs(sv.data[0]) ** 2)
    return float(np.real(sv.expectation_value(H)))


def grad_variance(anz, H, rng, n=VAR_SAMPLES):
    """Var over random theta of dE/dtheta_0. Exponential decay in N = plateau."""
    M = anz.num_parameters
    g = np.empty(n)
    for k in range(n):
        t = rng.uniform(-np.pi, np.pi, M)
        tp = t.copy(); tp[0] += FD
        tm = t.copy(); tm[0] -= FD
        g[k] = (energy(anz, H, tp) - energy(anz, H, tm)) / (2.0 * FD)
    return float(np.var(g))


def feature_width(anz, H, rng, cuts=CUTS, grid=GRID):
    """Autocorrelation length of E along random 1-D cuts, as a fraction of path.

    A value near 1.0 means the energy varies over the WHOLE path - one broad
    feature, which is what v99 found at reps=1. A small value means the profile
    decorrelates quickly: many thin features.
    """
    M = anz.num_parameters
    ws = []
    for _ in range(cuts):
        a = rng.uniform(-np.pi, np.pi, M)
        b = rng.uniform(-np.pi, np.pi, M)
        ts = np.linspace(0.0, 1.0, grid)
        E = np.array([energy(anz, H, a + t * (b - a)) for t in ts])
        E = E - E.mean()
        nrm = float(E @ E)
        if nrm < 1e-24:
            ws.append(1.0)
            continue
        # normalised autocorrelation, positive lags only
        ac = np.correlate(E, E, mode='full')[grid - 1:] / nrm
        below = np.where(ac < np.exp(-1.0))[0]
        lag = int(below[0]) if len(below) else grid
        ws.append(lag / float(grid))
    return float(np.mean(ws)), float(np.std(ws))


print("=" * 100)
print("v134  FEATURE WIDTH vs THE BARREN-PLATEAU REGIME")
print("=" * 100)
print("  v99 measured width 0.96 / 0.99 of the path at efficient_su2(reps=1),")
print("  N=4/6 - a shallow low-M ansatz with NO plateau. This asks whether width")
print("  falls as the plateau regime is entered, which is where the walk's")
print("  precondition would actually be met.")
print()
print("  The SENSING oracle's job is cost and v89 proves it cannot help a")
print("  plateau. Landscape navigation is the WALK's job. TIER B - exact")
print("  Statevector energies, no shots, no walk run.")
print()
print("  TWO COST FUNCTIONS, because the choice decides whether a plateau exists")
print("  at all. A LOCAL cost (Heisenberg, 2-local) plateaus only at depth")
print("  ~poly(N); a GLOBAL cost |<0..0|psi>|^2 plateaus at shallow depth. A first")
print("  run used the local cost alone, and its variance fell only 5x from N=4 to")
print("  N=10 - it never entered the regime, so it could say nothing about it.")
print()

ALLROWS = {}
for tag, mk in (("LOCAL  (Heisenberg)", lambda N: heis(N)),
                ("GLOBAL |<0..0|psi>|^2", lambda N: 'global')):
    print("  %s" % tag)
    print("      N   reps    M    Var(dE/dtheta_0)    feature width (frac of path)")
    print("   " + "-" * 80)
    rows = []
    for N in (4, 6, 8, 10):
        H = mk(N)
        for reps in (1, 4):
            anz = efficient_su2(N, reps=reps)
            M = anz.num_parameters
            rng = np.random.default_rng(1000 + 10 * N + reps)
            v = grad_variance(anz, H, rng)
            w, ws = feature_width(anz, H, rng)
            rows.append((N, reps, M, v, w, ws))
            print("   %4d %5d %5d      %.3e          %.4f +- %.4f"
                  % (N, reps, M, v, w, ws))
        print()
    ALLROWS[tag] = rows
rows = ALLROWS["GLOBAL |<0..0|psi>|^2"]

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  DID EITHER COST ACTUALLY REACH A PLATEAU? Variance decay per qubit,")
print("  reps=4 (a plateau is exponential decay; flat means no plateau):")
for tag, rs in ALLROWS.items():
    r4t = [r for r in rs if r[1] == 4]
    x = np.array([r[0] for r in r4t], float)
    y = np.log(np.maximum([r[3] for r in r4t], 1e-300))
    sl = float(np.polyfit(x, y, 1)[0])
    print("      %-24s d(log Var)/dN = %+.3f   %s"
          % (tag, sl, "PLATEAU" if sl < -0.35 else "no plateau in range"))
print()

r1 = [r for r in rows if r[1] == 1]
r4 = [r for r in rows if r[1] == 4]


def slope(rs, idx):
    x = np.array([r[0] for r in rs], float)
    y = np.log(np.maximum([r[idx] for r in rs], 1e-300))
    return float(np.polyfit(x, y, 1)[0])


print("  GLOBAL-cost arm, fitted d(log)/dN:")
print("      reps=1   variance %+.3f   width %+.3f"
      % (slope(r1, 3), slope(r1, 4)))
print("      reps=4   variance %+.3f   width %+.3f"
      % (slope(r4, 3), slope(r4, 4)))
print()
w_small = r1[0][4]
w_big = r4[-1][4]
v_small = r1[0][3]
v_big = r4[-1][3]
print("  Corner to corner: N=%d reps=1 -> N=%d reps=4"
      % (r1[0][0], r4[-1][0]))
print("      variance  %.3e -> %.3e   (%.0fx smaller)"
      % (v_small, v_big, v_small / max(v_big, 1e-300)))
print("      width     %.4f -> %.4f   (%.2fx narrower)"
      % (w_small, w_big, w_small / max(w_big, 1e-12)))
print()
if w_big < 0.5 * w_small and v_big < v_small:
    print("  FEATURES NARROW AS THE PLATEAU DEEPENS. v99's 0.96 / 0.99 is then a")
    print("  SMALL-ANSATZ ARTEFACT: it measured width exactly where no plateau")
    print("  exists, and reported the walk's precondition unmet in the only regime")
    print("  where it was never going to be met. The regime where gradient descent")
    print("  fails - flat almost everywhere, thin deep features - is the regime")
    print("  whose geometry the walk is built for, since tunnelling cost scales")
    print("  with WIDTH and these features are narrow by construction.")
    print()
    print("  THE TIER-A FOLLOW-UP is to run the walk itself at (N, reps) in the")
    print("  narrow regime and measure transmission against a classical descent,")
    print("  which is v99's PART 4 comparison moved to where the geometry lives.")
elif w_big >= 0.5 * w_small:
    print("  FEATURES DO NOT NARROW (%.4f -> %.4f). The plateau deepens without"
          % (w_small, w_big))
    print("  the landscape developing thin structure, so v99's finding survives")
    print("  the change of regime and is not a small-ansatz artefact. The walk's")
    print("  precondition is unmet at scale as well, and that is a stronger")
    print("  negative than v99 had - record it as such.")
else:
    print("  The two indicators do not move together, so no clean statement")
    print("  follows. Do not read a verdict out of these numbers.")
print()
print("  SCOPE. efficient_su2 only, Heisenberg only, N <= 10, reps in (1,4),")
print("  %d cuts x %d grid points per cell, %d samples for the variance."
      % (CUTS, GRID, VAR_SAMPLES))
print("  Autocorrelation length is ONE definition of feature width; a different")
print("  one (barrier height between local minima, gorge volume) could disagree,")
print("  and no walk is run here - this measures the PRECONDITION only.")
