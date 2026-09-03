"""Is bifurcation order a property of the STEP SIZE, or of the LANDSCAPE?

v93 sweeps gain = alpha*R and looks for a cascade. That is the obvious reading
and it is also the uninteresting one: every gradient method destabilises when
the learning rate grows, and finding period-doubling in it restates that.

THE HYPOTHESIS UNDER TEST HERE IS DIFFERENT AND IS NOT MINE. At FIXED gain, the
effective nonlinearity varies from point to point on the landscape, so the
bifurcation order measures WHERE YOU ARE rather than HOW BIG YOUR STEPS ARE.
The predicted ordering is

    near a minimum        period 1  - the gradient direction barely turns
    higher on the surface higher periods - it turns faster per step
    barren plateau        aperiodic - g -> 0 but g/|g|_max stays O(1), so the
                          direction is set by vanishing structure and rotates
                          almost freely

If that holds, bifurcation order is a TRAINABILITY DIAGNOSTIC, and it composes
with v89 rather than contradicting it: R-smoothing provably cannot ESCAPE a
barren plateau, but the local period could DETECT one.

WHY THE JACOBIAN AND NOT AN ORBIT. Iterating from a high-energy start does not
measure the high-energy landscape - the orbit runs downhill and reports wherever
it lands. The local object is the update map's Jacobian

    T(p) = p - gain * g(p)/|g(p)|_max,     J = dT/dp,

and period-doubling is exactly an eigenvalue of J crossing -1. So this samples
points at many energies, computes J by central differences, and asks whether
lambda_min(J) < -1 correlates with height. Everything is exact-energy; shot
noise would fake instability everywhere.

BARREN-PLATEAU ARM. A deep random circuit at the same qubit count gives the
suppressed-gradient regime. If the hypothesis is right its Jacobian should be
the WORST behaved, not the best, because normalisation divides by a vanishing
maximum.
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp

GAIN, R = 0.54, 0.6          # V6 defaults: alpha=0.9, r0=0.6
FD = 1e-4
N_SAMPLE = 40


def make(n, reps):
    a = efficient_su2(n, reps=reps)
    terms = [("ZZ" + "I" * (n - 2), 1.0)] if n > 2 else [("ZZ", 1.0)]
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        terms.append(("".join(s), 0.5))
    return a, SparsePauliOp.from_list(terms).to_matrix()


def energy(a, Hm, th):
    v = Statevector(a.assign_parameters(th)).data
    return float(np.real(np.conj(v) @ (Hm @ v)))


def walsh_grad(a, Hm, th, M, signs):
    E = np.array([energy(a, Hm, th + R * s) for s in signs])
    return (signs * E[:, None]).mean(axis=0) / R


def step_dir(a, Hm, th, M, signs):
    g = walsh_grad(a, Hm, th, M, signs)
    mx = float(np.max(np.abs(g)))
    return g / mx if mx > 1e-14 else np.zeros_like(g)


def jacobian(a, Hm, th, M, signs):
    """J = d/dp [ p - gain * ghat(p) ], central differences."""
    J = np.eye(M)
    for j in range(M):
        tp, tm = th.copy(), th.copy()
        tp[j] += FD
        tm[j] -= FD
        dd = (step_dir(a, Hm, tp, M, signs)
              - step_dir(a, Hm, tm, M, signs)) / (2 * FD)
        J[:, j] -= GAIN * dd
    return J


print("=" * 96)
print("BIFURCATION ORDER vs POSITION ON THE LANDSCAPE, at FIXED gain")
print("=" * 96)
print(f"  gain = {GAIN} (V6 default alpha*r0), R = {R} held.")
print("  period-doubling <=> an eigenvalue of J crosses -1.")
print("  'unstable' counts eigenvalues with Re(lambda) < -1.")
print()

for n, reps, tag in ((2, 1, "shallow  N=2 reps=1"),
                     (2, 3, "deeper   N=2 reps=3"),
                     (3, 4, "deep     N=3 reps=4  (plateau-ward)")):
    a, Hm = make(n, reps)
    M = a.num_parameters
    if M > 14:
        print(f"  {tag}: M={M} too wide for exact enumeration, skipped")
        continue
    signs = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                      for v in range(2 ** M)])
    ev = np.linalg.eigvalsh(Hm)
    Emin, Emax = float(ev.min()), float(ev.max())

    rng = np.random.default_rng(11)
    rows = []
    for t in range(N_SAMPLE):
        th = rng.uniform(0, 2 * np.pi, M)
        E = energy(a, Hm, th)
        g = walsh_grad(a, Hm, th, M, signs)
        J = jacobian(a, Hm, th, M, signs)
        lam = np.linalg.eigvals(J)
        rows.append((
            (E - Emin) / (Emax - Emin),          # normalised height
            float(np.linalg.norm(g)),
            int(np.sum(np.real(lam) < -1.0)),
            float(np.max(np.abs(lam))),
        ))

    rows.sort()
    lo = [r for r in rows if r[0] < 0.4]
    hi = [r for r in rows if r[0] >= 0.6]
    print(f"  {tag}   M={M}")
    print(f"    {'height band':<16}{'n':>4}{'mean|g|':>11}"
          f"{'mean unstable':>15}{'mean max|lam|':>15}")
    print("    " + "-" * 61)
    for lab, band in (("low  (<0.4)", lo), ("high (>=0.6)", hi)):
        if not band:
            print(f"    {lab:<16}{0:>4}{'-':>11}{'-':>15}{'-':>15}")
            continue
        print(f"    {lab:<16}{len(band):>4}"
              f"{np.mean([b[1] for b in band]):>11.4f}"
              f"{np.mean([b[2] for b in band]):>15.2f}"
              f"{np.mean([b[3] for b in band]):>15.3f}")
    print(f"    all samples: mean|g| {np.mean([r[1] for r in rows]):.4f}"
          f"   mean unstable {np.mean([r[2] for r in rows]):.2f}"
          f"   mean max|lam| {np.mean([r[3] for r in rows]):.3f}")
    print()

print("=" * 96)
print("READING IT")
print("=" * 96)
print("  The hypothesis predicts THREE things, and each can fail separately:")
print("    1. within a circuit, 'unstable' rises from the low to the high band;")
print("    2. across circuits, |g| falls as depth grows (approaching plateau);")
print("    3. and max|lambda| RISES as |g| falls, because normalisation divides")
print("       by a vanishing maximum - the plateau is the least stable regime,")
print("       not the most.")
print("  If (1) fails the landscape does not set the control parameter and the")
print("  bifurcation is a step-size artefact, as v93 assumes. If (3) fails in")
print("  the other direction - plateau most stable - then normalisation is")
print("  damping rather than amplifying and the diagnostic reads backwards.")
