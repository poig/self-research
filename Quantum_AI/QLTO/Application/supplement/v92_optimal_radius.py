"""Is the radius a fitted constant or a computed one?

r0 = 0.6 is the last purely INDUCTIVE number in V6: swept on a grid, chosen by
score, with no derivation behind it. v89 changed that. From the degree-wise law

    E_s[s_i E(theta+Rs)] = sin(R) sum_{T ni i} cos^(|T|-1)(R) d_i E_T          (*)

the Walsh shot noise is independent of R while the 1/R division is not, so

    SNR(R) ~ sin(R) cos^(d-1)(R) sqrt(T) / sigma,

maximised where d/dR[sin R cos^(d-1) R] = 0, i.e.

    tan^2 R* = 1/(d-1),        R* = arctan(1/sqrt(d-1)),

with d the EFFECTIVE Fourier degree the ansatz-plus-Hamiltonian actually carries.
That predicts R* ~ 0.20 at d=24 and ~0.18 at d=32, well BELOW the tuned 0.6.

THE HONEST DIFFICULTY, stated before the numbers. d is not a quantity anyone
handed us. (*) gives a different attenuation per degree, and a real ansatz mixes
degrees, so "the" effective degree is itself an inference. This script therefore
does not assume d: it MEASURES the attenuation curve with exact energies, fits
the single d that best explains it, and only then predicts R*. If that predicted
R* lands on the empirically best radius, the derivation has earned the right to
replace the sweep. If it does not, the derivation is missing something and the
tuned 0.6 stands - which is a result too, and the more likely one, since 0.6 was
selected against actual convergence rather than against gradient SNR.

  PART 1  attenuation |g_R|/|grad E| vs R, EXACT energies, no shots. Fit d.
  PART 2  cos(g_R, grad E) vs R at the REAL shot budget. Where is the empirical
          optimum, and does it sit at the predicted R*?
  PART 3  end-to-end: V6 from r0 = R* against r0 = 0.6, matched circuits.

PART 1 and PART 2 measure different things on purpose. Attenuation is pure
signal loss and is monotone in R; the SNR optimum only exists once shot noise is
in play, because that is what punishes small R.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

import benchmark as B
from nisq_v6 import QLTOv6
from qiskit.quantum_info import Statevector

RADII = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 1.00)
SEEDS = 6
MC = 3000


def energy_exact(ansatz, Hm, th):
    v = Statevector(ansatz.assign_parameters(th)).data
    return float(np.real(np.conj(v) @ (Hm @ v)))


def grad_exact(ansatz, Hm, th):
    g = np.zeros(len(th))
    for i in range(len(th)):
        tp, tm = th.copy(), th.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        g[i] = (energy_exact(ansatz, Hm, tp) - energy_exact(ansatz, Hm, tm)) / 2.0
    return g


def grad_smoothed_exact(ansatz, Hm, th, R, rng):
    """Degree-1 Walsh coefficient over the +-R hypercube / R, exact energies."""
    M = len(th)
    S = rng.choice([-1.0, 1.0], size=(MC, M))
    E = np.array([energy_exact(ansatz, Hm, th + R * s) for s in S])
    return (S * E[:, None]).mean(axis=0) / R


def cosf(u, v):
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d > 0 else 0.0


def predict_Rstar(d):
    return float(np.arctan(1.0 / np.sqrt(max(d - 1, 1e-9))))


PROBS = [B.get_heisenberg_problem(4), B.get_heisenberg_problem(6)]

print("=" * 96)
print("PART 1.  Attenuation vs R, exact energies.  Fit the effective degree d.")
print("=" * 96)
print("  model:  |g_R| / |grad E|  =  sin(R) cos^(d-1)(R) / R")
print()

fitted = {}
for ansatz, H, name in PROBS:
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    rng = np.random.default_rng(5)
    th = np.random.RandomState(42).uniform(0, 2 * np.pi, M)
    ge = grad_exact(ansatz, Hm, th)
    nge = np.linalg.norm(ge)

    print(f"  {name}  M={M}")
    print(f"    {'R':>7}{'|g_R|/|g|':>12}{'model d fit':>14}")
    print("    " + "-" * 33)
    ratios = []
    for R in RADII:
        gr = grad_smoothed_exact(ansatz, Hm, th, R, rng)
        ratio = float(np.linalg.norm(gr) / nge) if nge > 0 else float('nan')
        ratios.append(ratio)
        print(f"    {R:>7.2f}{ratio:>12.5f}")

    # least-squares fit of d over the model, on radii where signal survives
    best_d, best_err = None, None
    for dcand in np.arange(2.0, 80.0, 0.25):
        pred = [np.sin(R) * np.cos(R) ** (dcand - 1) / R for R in RADII]
        err = sum((p - r) ** 2 for p, r in zip(pred, ratios))
        if best_err is None or err < best_err:
            best_err, best_d = err, dcand
    fitted[name] = best_d
    print(f"    fitted d = {best_d:.2f}   (M = {M})"
          f"   -> predicted R* = {predict_Rstar(best_d):.4f}")
    print()

print("=" * 96)
print("PART 2.  cos(g_R, grad E) at the REAL shot budget.  Where is the optimum?")
print("=" * 96)
print(f"  V6's own sense(), {B.SHOTS} shots, {SEEDS} seeds, one global block.")
print()

for ansatz, H, name in PROBS:
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    Rstar = predict_Rstar(fitted[name])
    print(f"  {name}   predicted R* = {Rstar:.4f}   tuned r0 = 0.60")
    print(f"    {'R':>7}{'mean cos':>11}{'sd':>9}")
    print("    " + "-" * 27)
    rows = []
    for R in RADII:
        cs = []
        for s in range(SEEDS):
            th = np.random.RandomState(100 + s).uniform(0, 2 * np.pi, M)
            ge = grad_exact(ansatz, Hm, th)
            with contextlib.redirect_stdout(io.StringIO()):
                q = QLTOv6(ansatz, H, shot_budget=B.SHOTS, sim_seed=s)
                blocks = [b['params'] for b in q.layers if b['params']]
                g = np.zeros(M)
                for act in blocks:
                    gi, _ = q.sense(th, R, act)
                    g += gi
            cs.append(cosf(g, ge))
        rows.append((R, float(np.mean(cs)), float(np.std(cs))))
        print(f"    {R:>7.2f}{np.mean(cs):>11.4f}{np.std(cs):>9.4f}", flush=True)
    best = max(rows, key=lambda r: r[1])
    print(f"    empirical best R = {best[0]:.2f}  (cos {best[1]:.4f})")
    print(f"    predicted  R* = {Rstar:.4f}"
          f"   -> {'AGREES' if abs(best[0] - Rstar) <= 0.10 else 'DISAGREES'}"
          f" within 0.10")
    print()

print("=" * 96)
print("READING IT")
print("=" * 96)
print("  PART 1 must be monotone decreasing: attenuation only ever loses signal.")
print("  A fitted d far from M means the ansatz does not carry degree-M content,")
print("  which is itself worth knowing - (*) says the attenuation base reports")
print("  the EFFECTIVE degree, not the parameter count.")
print()
print("  PART 2 is the test. If the empirical argmax sits at the predicted R*,")
print("  the radius becomes a computed quantity and the sweep is redundant. If")
print("  the argmax sits near 0.60 instead, the SNR derivation does not govern")
print("  what the tuned value was selected for, and 0.6 stays a fitted constant")
print("  - in which case the honest statement is that V6 still has one knob")
print("  with no theory behind it.")
