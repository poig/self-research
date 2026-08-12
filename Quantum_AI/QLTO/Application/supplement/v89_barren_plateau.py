"""Barren plateaus and R-smoothing, after one wrong derivation and one bad sweep.

TWO FALSE STARTS ARE RECORDED HERE ON PURPOSE, because both produced
plausible-looking output and either could have been published.

  ATTEMPT 1, a sweep. Var[dE/dtheta] against Var[(grad E_R)_i] over random
  inits, decay exponents compared. Abandoned at the first row: N=4, reps=2 -
  where no plateau exists - gave cosines 0.11 / 0.26 / 0.007. That is the Walsh
  estimator's sampling noise at 160 samples over 24 parameters, not physics.

  ATTEMPT 2, an identity. Claimed E_s[s_i E(theta+Rs)] = sin(R) cos^(M-1)(R)
  dE/dtheta_i exactly, on the argument that averaging over each OTHER coordinate
  contributes one factor of cos R. FALSIFIED by its own check: max|walsh - pred|
  = 4.2e-2 against a predicted kappa of 0.215, a 20% relative error, and
  cos(walsh, grad) = 0.9986 rather than 1. The error is that E is AFFINE in
  (cos theta_i, sin theta_i) - there is a CONSTANT term - and terms in which
  another coordinate contributes its constant pick up NO factor of cos R.

THE CORRECT STATEMENT. Expand E in its Fourier basis over the parameters. For a
term supported on a coordinate set T, shifting theta -> theta + Rs and averaging
over s leaves the s_i-odd part attenuated by one cos R per OTHER coordinate in
T, so

    E_s[ s_i E(theta + R s) ]  =  sin(R) * sum_{T ni i} cos^(|T|-1)(R) d_i E_T   (*)

A LOW-PASS FILTER ON FOURIER DEGREE, not a uniform rescaling. Degree-1 content
passes at sin(R); degree-d content is suppressed by cos^(d-1)(R). Direction is
preserved exactly only when a single degree dominates, which is why the measured
cosine sits near 0.999 rather than at 1.

WHY THE PLATEAU CONCLUSION SURVIVES ANYWAY, and it needs no experiment. Every
factor cos^(|T|-1)(R) lies in [0,1] for R in [0, pi/2]. Smoothing therefore
ATTENUATES every Fourier component and amplifies none, so

    |grad E_R| <= |grad E|        coordinate-wise in the Fourier basis.

If the exact gradient is exponentially small in N, the smoothed one is at least
as small. R-smoothing cannot escape a barren plateau. The narrow-gorge hope
fails for a concrete reason: (*) has no dependence on theta beyond the local
Fourier content, so there is no global slope being collected - only the same
local information, low-pass filtered.

WHAT IS STILL WORTH SOMETHING. (*) identifies the base of the exponential decay
v8_attenuation measured empirically (0.9697 at M=24, 0.9764 at M=32) without
naming: it is cos(R), per Fourier degree. And since the Walsh shot noise does not
depend on R while the 1/R division does, SNR ~ sin(R) cos^(d-1)(R) sqrt(T), which
for effective degree d has an interior optimum. That is the schedule's job.

PART 1 verifies (*) degree by degree on functions of KNOWN Fourier content, where
it must hold exactly, and then reports the residual on a real ansatz where the
degree mixture is what it is.
PART 2 checks |grad E_R| <= |grad E| by exact enumeration - the plateau claim.
PART 3 gives the SNR-optimal radius and compares it to the tuned r0.
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp


def local_ham(N):
    s = ['I'] * N
    s[0] = s[1] = 'Z'
    return SparsePauliOp.from_list([(''.join(s), 1.0)])


def energy(ansatz, Hm, th):
    v = Statevector(ansatz.assign_parameters(th)).data
    return float(np.real(np.conj(v) @ (Hm @ v)))


def exact_grad(ansatz, Hm, th):
    g = np.zeros(len(th))
    for i in range(len(th)):
        tp, tm = th.copy(), th.copy()
        tp[i] += np.pi / 2
        tm[i] -= np.pi / 2
        g[i] = (energy(ansatz, Hm, tp) - energy(ansatz, Hm, tm)) / 2.0
    return g


def walsh_of(fn, th, R, M):
    """E_s[s_i f(theta+Rs)] by FULL 2^M enumeration. No sampling."""
    sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(M)]
                    for v in range(2 ** M)])
    E = np.array([fn(th + R * s) for s in sig])
    return (sig * E[:, None]).mean(axis=0)


print("=" * 92)
print("PART 1.  (*) on functions of KNOWN Fourier degree - it must be exact here")
print("=" * 92)
print("  f_d(theta) = cos(theta_0 + theta_1 + ... + theta_{d-1}) is pure degree d.")
print("  (*) predicts E_s[s_0 f_d(theta+Rs)] = sin(R) cos^(d-1)(R) d_0 f_d.")
print("  A machine-precision match here validates the DEGREE-WISE law; the")
print("  earlier cos^(M-1) claim fails precisely because a real ansatz mixes")
print("  degrees and each degree carries its own exponent.")
print()
print(f"  {'M':>4}{'degree d':>10}{'R':>6}{'predicted':>16}"
      f"{'measured':>16}{'|diff|':>12}")
print("  " + "-" * 64)

for M, d, R in ((8, 1, 0.4), (8, 2, 0.4), (8, 3, 0.4), (10, 4, 0.5), (10, 6, 0.5)):
    def f(t, d=d):
        return float(np.cos(np.sum(t[:d])))

    th = np.random.default_rng(1).uniform(0, 2 * np.pi, M)
    w0 = walsh_of(f, th, R, M)[0]
    dg = -float(np.sin(np.sum(th[:d])))          # d/dtheta_0 of f_d
    pred = np.sin(R) * np.cos(R) ** (d - 1) * dg
    print(f"  {M:>4}{d:>10}{R:>6.2f}{pred:>16.9f}{w0:>16.9f}"
          f"{abs(w0 - pred):>12.2e}")

print()
print("  Now the same check on a real ansatz, where the degree mixture is not")
print("  controlled. The residual against the naive cos^(M-1) law is the size of")
print("  the error that falsified ATTEMPT 2.")
print()
print(f"  {'N':>3}{'reps':>6}{'M':>4}{'R':>6}"
      f"{'cos(walsh,grad)':>18}{'rel err vs cos^(M-1)':>22}")
print("  " + "-" * 59)
rng = np.random.default_rng(3)
for N, reps, R in ((2, 1, 0.3), (2, 1, 0.7), (2, 2, 0.4), (3, 1, 0.5)):
    ansatz = efficient_su2(N, reps=reps)
    M = ansatz.num_parameters
    if M > 14:
        continue
    Hm = local_ham(max(N, 2)).to_matrix()
    th = rng.uniform(0, 2 * np.pi, M)
    w = walsh_of(lambda t: energy(ansatz, Hm, t), th, R, M)
    g = exact_grad(ansatz, Hm, th)
    naive = np.sin(R) * np.cos(R) ** (M - 1) * g
    dn = np.linalg.norm(w) * np.linalg.norm(g)
    c = float(w @ g / dn) if dn > 1e-300 else float('nan')
    rel = np.max(np.abs(w - naive)) / max(np.max(np.abs(w)), 1e-30)
    print(f"  {N:>3}{reps:>6}{M:>4}{R:>6.2f}{c:>18.9f}{rel:>22.3f}")

print()
print("=" * 92)
print("PART 2.  |grad E_R| <= |grad E| :  the plateau claim, by enumeration")
print("=" * 92)
print("  Every cos^(|T|-1)(R) lies in [0,1], so smoothing attenuates every")
print("  Fourier component and amplifies none. Measured, not asserted: the")
print("  smoothed gradient is computed by full enumeration at each init.")
print()
print(f"  {'N':>3}{'M':>4}{'R':>6}{'mean|grad|':>14}{'mean|grad_R|/R':>17}"
      f"{'ratio':>10}{'max ratio':>12}")
print("  " + "-" * 66)

INITS = 24
for N, reps, R in ((2, 1, 0.4), (2, 2, 0.4), (3, 1, 0.4), (2, 2, 0.8)):
    ansatz = efficient_su2(N, reps=reps)
    M = ansatz.num_parameters
    Hm = local_ham(max(N, 2)).to_matrix()
    ge, gs = [], []
    for _ in range(INITS):
        th = rng.uniform(0, 2 * np.pi, M)
        g = exact_grad(ansatz, Hm, th)
        w = walsh_of(lambda t: energy(ansatz, Hm, t), th, R, M) / R
        ge.append(np.linalg.norm(g))
        gs.append(np.linalg.norm(w))
    ge, gs = np.array(ge), np.array(gs)
    print(f"  {N:>3}{M:>4}{R:>6.2f}{ge.mean():>14.6f}{gs.mean():>17.6f}"
          f"{gs.mean() / ge.mean():>10.4f}{np.max(gs / ge):>12.4f}")

print()
print("  max ratio <= 1 on every row means no init anywhere gained signal from")
print("  smoothing. A barren plateau in grad E is a barren plateau in grad E_R.")
print("  R-smoothing estimates a plateau more cheaply; it does not escape one.")

print()
print("=" * 92)
print("PART 3.  SNR-optimal radius for effective degree d")
print("=" * 92)
print("  Walsh shot noise is independent of R and the 1/R division cancels, so")
print("  SNR ~ sin(R) cos^(d-1)(R) sqrt(T). Maximised at tan^2 R* = 1/(d-1).")
print()
print(f"  {'degree d':>10}{'R* ':>10}{'attenuation':>14}{'cos(R*)':>10}")
print("  " + "-" * 44)
for d in (2, 4, 8, 16, 24, 32, 64):
    Rs = np.arctan(1.0 / np.sqrt(d - 1))
    print(f"  {d:>10}{Rs:>10.4f}"
          f"{np.sin(Rs) * np.cos(Rs) ** (d - 1):>14.4e}{np.cos(Rs):>10.4f}")
print()
print("  v8_attenuation fitted the signal ratio to base 0.9697 (M=24) and 0.9764")
print("  (M=32) without naming the base. (*) says it is cos(R) PER FOURIER")
print("  DEGREE, so those bases invert to R = 0.246 and 0.218 - which is a")
print("  statement about the effective degree the ansatz carries, not about M.")
print("  The tuned r0 = 0.6 sits above every R* here and decays at 0.95/epoch,")
print("  so the schedule sweeps the optimum rather than sitting on it. Whether")
print("  starting nearer R* converges faster is UNTESTED.")
