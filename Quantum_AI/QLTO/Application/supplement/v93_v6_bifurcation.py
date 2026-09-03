"""Does V6's own iteration period-double, and if so where?

The Feigenbaum work in QLTO/Feigenbaum establishes that P(|1>) = sin^2(phi/2)
from a Hadamard test - verified to 4.44e-16 - and that the resulting map
x -> r sin^2(pi x) has a period-doubling cascade with the universal delta:
bifurcations at r = 0.6277, 0.7066, 0.7259, 0.7301 and a ratio reaching 4.694
against delta = 4.6692.

Paper 1's |Psi_1> = (|0>|psi> + |1>|chi>)/sqrt2 IS a Hadamard test, so its
sense-actuate-reset cycle iterates a sin^2 nonlinearity with a QUADRATIC
MAXIMUM - the hypothesis of Feigenbaum universality, not an analogy to it.
That protocol's multi-cycle log shows monotone convergence over eight cycles,
which is the period-1 regime at the small fixed theta used there; nobody swept
the control parameter up.

THE QUESTION HERE IS NARROWER AND IS ABOUT V6, NOT ABOUT THE PAPER. V6's update

    p <- p - alpha R g / |g|_max                                            (V6)

is a discrete dynamical system whose control parameter is alpha*R. The tuned
r0 = 0.6 sits suspiciously close to the map's first bifurcation at 0.6277. That
is SUGGESTIVE AND NOTHING MORE: the map's r is a multiplicative gain in
r sin^2(pi x) while alpha*R is a step size, and two parameters landing near 0.6
is exactly the coincidence that produces a confident wrong answer. So this
measures rather than asserts.

DESIGN, and each choice is there to stop something masquerading as the effect.

  EXACT ENERGIES, NO SHOTS. Shot noise makes every orbit look aperiodic. The
      degree-1 Walsh gradient is computed by FULL enumeration of the 2^M
      hypercube, so the iteration is deterministic and any period is real.
  FIXED R. V6 normally decays R each epoch, which sweeps the control parameter
      and would smear a bifurcation across the run. Here R is held.
  SMALL SYSTEM. N=2, reps=1 gives M=8 and a 256-point hypercube, enumerable
      exactly. Bifurcation structure is a property of the map, not of problem
      size, so nothing is lost by making the map cheap.
  MAX-NORMALISATION KEPT. This is V6's actual step rule and it BOUNDS the
      update at alpha*R per coordinate. That bound may well destroy the
      nonlinearity that period-doubling needs - which is a real possible
      outcome and is reported as one rather than tuned away.

THREE OUTCOMES, all informative.
  (a) bifurcation near alpha*R ~ 0.63 with ratios approaching 4.669: r0 = 0.6
      becomes a DERIVED quantity - the largest step staying in period 1.
  (b) bifurcation elsewhere: the dynamics are real, the parameters do not map,
      and the 0.6/0.6277 proximity was coincidence.
  (c) no bifurcation: max-normalisation linearises the step and the Feigenbaum
      connection stops at paper 1's protocol, not reaching V6.
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp

N, REPS = 2, 1
R_FIXED = 0.6
N_TRANS, N_SAMPLE = 400, 80
TOL = 1e-7


def build():
    ansatz = efficient_su2(N, reps=REPS)
    H = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])
    return ansatz, H.to_matrix(), ansatz.num_parameters


ANSATZ, HM, M = build()
SIGNS = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                  for v in range(2 ** M)])


def energy(th):
    v = Statevector(ANSATZ.assign_parameters(th)).data
    return float(np.real(np.conj(v) @ (HM @ v)))


def walsh_grad(th, R):
    """Degree-1 Walsh coefficient / R, FULL enumeration - deterministic."""
    E = np.array([energy(th + R * s) for s in SIGNS])
    return (SIGNS * E[:, None]).mean(axis=0) / R


def v6_iterate(p0, R, gain, n_trans=N_TRANS, n_sample=N_SAMPLE):
    """V6's step rule with control parameter `gain` = alpha*R, R held fixed."""
    p = p0.copy()
    for _ in range(n_trans):
        g = walsh_grad(p, R)
        mx = float(np.max(np.abs(g)))
        if mx > 1e-12:
            p = p - gain * g / mx
    out = []
    for _ in range(n_sample):
        g = walsh_grad(p, R)
        mx = float(np.max(np.abs(g)))
        if mx > 1e-12:
            p = p - gain * g / mx
        out.append(energy(p))
    return np.array(out)


def detect_period(traj, tol=TOL):
    """Smallest k in 1..16 with traj[i] ~ traj[i+k] throughout the tail."""
    tail = traj[-40:]
    for k in (1, 2, 4, 8, 16):
        if len(tail) <= k:
            break
        if np.max(np.abs(tail[:-k] - tail[k:])) < tol:
            return k
    return 0                                  # 0 = aperiodic / not detected


print("=" * 94)
print("V6 ITERATION AS A DYNAMICAL SYSTEM")
print("=" * 94)
print(f"  N={N} reps={REPS} M={M}, hypercube {2 ** M} points enumerated exactly.")
print(f"  R held at {R_FIXED}; control parameter is gain = alpha*R.")
print(f"  V6's default is alpha=0.9, r0=0.6 -> gain = 0.54.")
print(f"  Map's first bifurcation (Feigenbaum dir) is at r = 0.627688.")
print()
print(f"  {'gain':>8}{'period':>9}{'E range in attractor':>24}{'E mean':>12}")
print("  " + "-" * 55)

p0 = np.random.RandomState(7).uniform(0, 2 * np.pi, M)
rows = []
for gain in np.arange(0.10, 2.01, 0.05):
    traj = v6_iterate(p0, R_FIXED, float(gain))
    per = detect_period(traj)
    spread = float(traj.max() - traj.min())
    rows.append((float(gain), per, spread, float(traj.mean())))
    tag = {0: 'aperiodic'}.get(per, str(per))
    print(f"  {gain:>8.2f}{tag:>9}{spread:>24.3e}{traj.mean():>12.6f}",
          flush=True)

print()
print("=" * 94)
print("BIFURCATION POINTS")
print("=" * 94)
prev = rows[0][1]
bifs = []
for gain, per, spread, mean in rows[1:]:
    if per != prev and per != 0 and prev != 0:
        bifs.append((prev, per, gain))
        print(f"  period {prev:>2} -> {per:<2}  at gain ~ {gain:.3f}")
    prev = per
if not bifs:
    print("  NONE DETECTED in the swept range.")
    print("  Outcome (c): the max-normalisation bounds the step at gain per")
    print("  coordinate, which linearises the update and removes the quadratic")
    print("  maximum period-doubling requires. The Feigenbaum structure then")
    print("  belongs to paper 1's protocol and does NOT reach V6's optimiser.")
else:
    ds = [bifs[i + 1][2] - bifs[i][2] for i in range(len(bifs) - 1)]
    if len(ds) >= 2:
        print()
        print("  ratios (should approach delta = 4.6692 if universal):")
        for i in range(len(ds) - 1):
            if ds[i + 1] > 1e-12:
                print(f"    {ds[i] / ds[i + 1]:.4f}")
    print()
    print(f"  V6's default gain is 0.54; first bifurcation at ~{bifs[0][2]:.3f}.")
    print("  If those bracket the tuned value, r0 = 0.6 is the largest radius")
    print("  that keeps the iteration in period 1 - a derived quantity, not a")
    print("  fitted one. If they do not, outcome (b): real dynamics, wrong map.")
