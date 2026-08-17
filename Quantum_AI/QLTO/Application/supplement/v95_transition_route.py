"""Which route to chaos, and is the transition window smaller than v93's grid?

v93 swept gain in steps of 0.05 and saw period 2 at 0.30, aperiodic at 0.35, with
nothing between. Two readings, and they are distinguishable.

  (i)  RESOLUTION. A Feigenbaum cascade is geometrically compressed: from the
       first bifurcation to the accumulation point spans only delta/(delta-1)
       ~ 1.27 times the FIRST gap. In the sin^2 map of QLTO/Feigenbaum that
       whole cascade is 0.6277 -> 0.7313, a window of 0.104. If V6's window is
       narrower than 0.05, a full cascade fits inside one grid step of v93 and
       is invisible.

  (ii) WRONG ROUTE. Period-doubling requires a REAL eigenvalue of the update
       Jacobian to cross -1. V6's Jacobian is M x M, so complex pairs are
       generic, and a complex pair crossing the unit circle is a
       Neimark-Sacker bifurcation: fixed point -> invariant torus ->
       quasi-periodic -> chaos, WITH NO DOUBLING CASCADE. Quasi-periodic motion
       has no finite period, so v93's detector would report 'aperiodic'
       immediately - exactly what it did.

These predict different things and this measures both:

  PART 1  gain scanned at 0.002 through the transition, 25x finer than v93. If
          intermediate periods 4, 8, 16 appear, reading (i) is right and delta
          can be extracted. If period 2 goes straight to aperiodic even here,
          (i) is excluded at that resolution.

  PART 2  the Jacobian's leading eigenvalues at each gain, classified. A real
          eigenvalue passing -1 is period-doubling; a complex pair leaving the
          unit circle with |Im| well away from 0 is Neimark-Sacker. This
          decides the ROUTE independently of whether the scan resolves periods,
          which is why it is the more reliable half of the test.
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp

N, REPS, R = 2, 1, 0.6
FD = 1e-4
ANSATZ = efficient_su2(N, reps=REPS)
M = ANSATZ.num_parameters
HM = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)]).to_matrix()
SIGNS = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                  for v in range(2 ** M)])


def energy(th):
    v = Statevector(ANSATZ.assign_parameters(th)).data
    return float(np.real(np.conj(v) @ (HM @ v)))


def walsh(th):
    E = np.array([energy(th + R * s) for s in SIGNS])
    return (SIGNS * E[:, None]).mean(axis=0) / R


def ghat(th):
    g = walsh(th)
    mx = float(np.max(np.abs(g)))
    return g / mx if mx > 1e-14 else np.zeros_like(g)


def orbit(p0, gain, n_trans=600, n_samp=120):
    p = p0.copy()
    for _ in range(n_trans):
        p = p - gain * ghat(p)
    out = []
    for _ in range(n_samp):
        p = p - gain * ghat(p)
        out.append(energy(p))
    return np.array(out), p


def period_of(traj, tol=1e-7):
    tail = traj[-60:]
    for k in (1, 2, 4, 8, 16, 32):
        if len(tail) <= k:
            break
        if np.max(np.abs(tail[:-k] - tail[k:])) < tol:
            return k
    return 0


def jac(th, gain):
    J = np.eye(M)
    for j in range(M):
        tp, tm = th.copy(), th.copy()
        tp[j] += FD
        tm[j] -= FD
        J[:, j] -= gain * (ghat(tp) - ghat(tm)) / (2 * FD)
    return J


p0 = np.random.RandomState(7).uniform(0, 2 * np.pi, M)

print("=" * 98)
print("PART 1.  Fine scan through the transition, step 0.002 (25x finer than v93)")
print("=" * 98)
print(f"  {'gain':>7}{'period':>10}{'E spread':>13}"
      f"{'max|lam|':>11}{'lam type':>26}")
print("  " + "-" * 67)

rows = []
for gain in np.arange(0.290, 0.362, 0.002):
    traj, pend = orbit(p0, float(gain))
    per = period_of(traj)
    J = jac(pend, float(gain))
    lam = np.linalg.eigvals(J)
    lead = lam[np.argmax(np.abs(lam))]
    mx = float(np.abs(lead))
    if abs(lead.imag) < 1e-6:
        kind = f"real {lead.real:+.4f}"
    else:
        kind = f"complex {lead.real:+.3f}{lead.imag:+.3f}i"
    rows.append((float(gain), per, mx, lead))
    tag = 'aperiodic' if per == 0 else str(per)
    print(f"  {gain:>7.3f}{tag:>10}{traj.max() - traj.min():>13.3e}"
          f"{mx:>11.4f}{kind:>26}", flush=True)

print()
print("=" * 98)
print("PART 2.  ROUTE")
print("=" * 98)
seen = sorted({r[1] for r in rows if r[1] != 0})
print(f"  periods observed in the window: {seen}")
inter = [p for p in seen if p not in (1, 2)]
if inter:
    print(f"  INTERMEDIATE periods {inter} present -> cascade was hidden by")
    print("  v93's 0.05 grid. Reading (i): resolution. delta is extractable.")
else:
    print("  NO intermediate period between 2 and aperiodic even at 0.002.")
    print("  Reading (i) is excluded at this resolution.")

cross = [r for r in rows if r[2] > 1.0]
if cross:
    first = cross[0]
    lead = first[3]
    print()
    print(f"  |lambda| first exceeds 1 at gain = {first[0]:.3f},"
          f" leading eigenvalue {lead.real:+.4f}{lead.imag:+.4f}i")
    if abs(lead.imag) < 1e-6 and lead.real < 0:
        print("  REAL and NEGATIVE -> period-doubling bifurcation.")
    elif abs(lead.imag) > 1e-6:
        ang = float(np.angle(lead))
        print(f"  COMPLEX PAIR, argument {ang:+.4f} rad"
              f" ({ang / np.pi:+.3f} pi) -> NEIMARK-SACKER.")
        print("  The fixed point loses stability to an invariant TORUS, giving")
        print("  quasi-periodic motion with no finite period and no doubling")
        print("  cascade. v93's jump from 2 to 'aperiodic' is then the correct")
        print("  physical picture, not a resolution artefact, and there is no")
        print("  Feigenbaum delta to extract because the route is not that one.")
    else:
        print("  REAL and POSITIVE -> saddle-node / transcritical, not doubling.")
else:
    print("  |lambda| never exceeds 1 in this window; the aperiodicity is not")
    print("  a local linear instability of the fixed point.")
