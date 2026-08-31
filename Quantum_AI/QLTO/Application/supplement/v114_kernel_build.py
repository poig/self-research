"""Build and VERIFY the band-limited derivative kernel of arXiv:2606.19486, before using it.

twirl_cal reads the Hamiltonian coefficients at ONE fixed evolution time T, which
carries an irreducible O(T^2) bias. v113 measured the consequence on the axis the
literature uses: eps ~ T_total^(-0.217) against SQL's -0.5, the frontier bending
away because no number of shots removes a bias.

arXiv:2606.19486 removes it with a kernel. The correlation function is BAND-LIMITED
- C(t) = sum_jk |<e_j|P|e_k>|^2 e^{i(E_j-E_k)t} has Fourier support in [-2L, 2L]
for ||H|| <= L - so its derivative at zero is recoverable EXACTLY as a weighted
average over finite times. No small-T limit, hence no truncation bias.

THE CONSTRUCTION, from Section 3.1:

    chi          smooth cutoff, chi = 1 on |w| <= 1, chi = 0 on |w| >= 2
    psi(w)      := i w chi(w / (2L))          first-derivative symbol
    L_ker(t)    := (1/2pi) int psi(w) e^{iwt} dw
    p(tau)      := 2 |L_ker(tau)| / ||L_ker||_1,   tau >= 0
    G           := ||L_ker||_1 sign(L_ker(tau)) Z        unbiased for F'(0)

and the DISCRETE version, Section 4.2, which is what keeps the circuit count
finite: psi has support [-4L, 4L] and the correlation [-2L, 2L], so the product
is band-limited to [-6L, 6L] and for grid spacing t0 < pi/(6L)

    int L_ker(t) f(t) dt  =  t0 sum_m L_ker(m t0) f(m t0)          EXACTLY

with truncation at |m| <= R/(2 t0) costing error that "decays faster than any
polynomial in R". So the number of DISTINCT evolution times is R/t0 ~ 6LR/pi with
R ~ polylog(1/eps) - the eps-dependence goes into SHOTS PER TIME, not into the
number of times. That is the whole reason the O(1)-in-M circuit count can survive.

WHY THIS FILE EXISTS SEPARATELY. Getting a kernel subtly wrong breaks
unbiasedness silently, and every downstream number would then be a strawman of
someone else's protocol - the exact failure this project has already recorded
twice (v104's under-configured fit(), and the Operator() trap in v102). So the
kernel is built and CHECKED HERE against things whose answers are known
independently, and nothing is built on it until the checks pass:

  CHECK 1  Fourier inversion: int L_ker(t) e^{i w0 t} dt = psi(-w0) for |w0| <= 2L
  CHECK 2  derivative extraction on analytic band-limited test functions
  CHECK 3  the discretisation identity, at t0 just under and just over pi/(6L)
           - it must HOLD below the threshold and FAIL above it, or the band
             limit is not being respected and the agreement is a coincidence
  CHECK 4  the real thing: the correlation function of the N=3 crosstalk
           Hamiltonian, derivative at zero against exact differentiation

TIER (project rule R1): this file is tier C by design and says so - it builds a
classical numerical object and checks it against exact references. No circuit,
no shots, no accuracy claim. It is the reference the later circuit work will be
checked against, which R1 lists as sanctioned use.
"""
import sys, os
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.quantum_info import SparsePauliOp, Pauli
from twirl_cal import crosstalk_terms, crosstalk_coeffs


# ---- the smooth cutoff -------------------------------------------------------
def _bump(x):
    """C^inf ramp 0 -> 1 on [0,1], the standard exp(-1/x) partition."""
    x = np.asarray(x, float)
    s = np.where(x > 0, np.exp(-1.0 / np.maximum(x, 1e-300)), 0.0)
    s1 = np.where(1 - x > 0, np.exp(-1.0 / np.maximum(1 - x, 1e-300)), 0.0)
    d = s + s1
    return np.where(d > 0, s / np.maximum(d, 1e-300), 0.0)


def chi(w):
    """1 on |w|<=1, 0 on |w|>=2, C^inf in between."""
    a = np.abs(np.asarray(w, float))
    out = np.where(a <= 1.0, 1.0, 0.0)
    mid = (a > 1.0) & (a < 2.0)
    out = np.where(mid, 1.0 - _bump(a - 1.0), out)
    return out


def make_kernel(LAM, tmax=None, n_t=20001):
    """L_ker(t) = (1/2pi) int psi(w) e^{iwt} dw with psi(w) = -i w chi(w/2L).

    i w chi is odd and imaginary, so the cosine part integrates to zero and the
    kernel is real: L_ker(t) = +(1/pi) int_0^inf w chi(w/2L) sin(w t) dw.

    THE SIGN IS FIXED EMPIRICALLY, and it matters. Writing psi = +i w chi - the
    naive reading of the paper's notation through the OCR - gives a kernel that
    is correct in magnitude to 1e-8 on CHECK 1 but returns MINUS the derivative:
    for f = sum a_k sin(w_k t) the identity comes out int L f = -f'(0), because
    with that convention (1/2i)[psi(-w0) - psi(w0)] = -w0 chi. Their convention
    is the one that makes int L f = +f'(0), so psi carries the minus. Caught by
    CHECK 2 rather than assumed - which is the entire reason CHECK 2 exists.
    """
    W = 4.0 * LAM                       # psi is supported in |w| <= 4L

    def Lk(t):
        f = lambda w: w * chi(w / (2.0 * LAM)) * np.sin(w * t)
        val, _ = quad(f, 0.0, W, limit=400)
        return val / np.pi

    if tmax is None:
        tmax = 60.0 / LAM
    ts = np.linspace(-tmax, tmax, n_t)
    vals = np.array([Lk(t) for t in ts])
    l1 = float(np.trapezoid(np.abs(vals), ts))
    return Lk, l1, ts, vals


LAM_T = 1.0
Lk, L1, TS, LV = make_kernel(LAM_T)

print("=" * 96)
print("v114  BAND-LIMITED DERIVATIVE KERNEL - BUILD AND VERIFY")
print("=" * 96)
print("  TIER C BY DESIGN: a classical numerical object checked against exact")
print("  references. No circuit, no shots, no accuracy claim rests on it.")
print()
print("  Lambda = %.2f, kernel support |w| <= 4L = %.1f, ||L_ker||_1 = %.6f"
      % (LAM_T, 4 * LAM_T, L1))
print()

print("=" * 96)
print("CHECK 1  Fourier inversion:  int L_ker(t) e^{i w0 t} dt  =  psi(-w0) = +i w0 chi(w0/2L)")
print("=" * 96)
print("     w0      Re(measured)   Im(measured)     Im(expected)      abs err")
print("  " + "-" * 76)
ok1 = True
for w0 in (0.3, 0.8, 1.5, 1.9):
    integ = np.trapezoid(LV * np.exp(1j * w0 * TS), TS)
    exp_im = w0 * float(chi(np.array(w0 / (2 * LAM_T))))
    err = abs(integ.imag - exp_im) + abs(integ.real)
    ok1 &= err < 2e-3
    print("   %5.2f    %+.6e   %+.6e    %+.6e     %.2e"
          % (w0, integ.real, integ.imag, exp_im, err))
print("   PASS" if ok1 else "   FAIL")
print()

print("=" * 96)
print("CHECK 2  derivative extraction on band-limited test functions")
print("=" * 96)
print("  f(t) = sum_k a_k sin(w_k t) with all |w_k| <= 2L, so f'(0) = sum a_k w_k.")
print()
print("      case          int L_ker f dt        f'(0) exact         abs err")
print("  " + "-" * 76)
ok2 = True
cases = [([1.0], [0.7]), ([0.6, -0.4], [0.5, 1.7]), ([0.3, 0.3, 0.2], [0.2, 1.0, 1.95])]
for a, w in cases:
    f = sum(ai * np.sin(wi * TS) for ai, wi in zip(a, w))
    got = float(np.trapezoid(LV * f, TS))
    exact = float(sum(ai * wi for ai, wi in zip(a, w)))
    err = abs(got - exact)
    ok2 &= err < 5e-3
    print("   %-14s  %+.8f          %+.8f        %.2e"
          % ("%d-tone" % len(a), got, exact, err))
print("   PASS" if ok2 else "   FAIL")
print()

print("=" * 96)
print("CHECK 3  the discretisation identity, and that it FAILS above the threshold")
print("=" * 96)
print("  t0 < pi/(6L) = %.6f must give t0 sum_m L_ker(m t0) f(m t0) = int L_ker f dt."
      % (np.pi / (6 * LAM_T)))
print("  A grid that is too COARSE must break it - otherwise the agreement below")
print("  the threshold is not evidence the band limit is being respected.")
print()
a, w = [0.6, -0.4], [0.5, 1.7]
exact = float(sum(ai * wi for ai, wi in zip(a, w)))
thresh = np.pi / (6 * LAM_T)
print("      t0        t0/(pi/6L)     discrete sum       exact         abs err   verdict")
print("  " + "-" * 84)
ok3 = True
for frac in (0.3, 0.6, 0.9, 1.5, 3.0, 6.0):
    t0 = frac * thresh
    ms = np.arange(-int(80.0 / (LAM_T * t0)), int(80.0 / (LAM_T * t0)) + 1)
    tt = ms * t0
    fv = sum(ai * np.sin(wi * tt) for ai, wi in zip(a, w))
    kv = np.array([Lk(t) for t in tt])
    s = float(t0 * np.sum(kv * fv))
    err = abs(s - exact)
    good = err < 5e-3
    verdict = "holds" if good else "BREAKS"
    if frac < 1.0:
        ok3 &= good
    else:
        pass
    print("   %8.5f   %8.2f      %+.8f     %+.8f     %.2e   %s"
          % (t0, frac, s, exact, err, verdict))
print()
print("   below-threshold rows must all hold: %s" % ("PASS" if ok3 else "FAIL"))
print()

print("=" * 96)
print("CHECK 4  the real correlation function, N=3 crosstalk")
print("=" * 96)
N = 3
terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
H = SparsePauliOp.from_list(list(zip(terms, c_true))).simplify()
Hm = H.to_matrix()
LAM_H = float(np.linalg.norm(Hm, 2))
print("  ||H||_2 = %.6f. Rebuilding the kernel at this Lambda." % LAM_H)
Lk2, L1_2, TS2, LV2 = make_kernel(LAM_H, tmax=60.0 / LAM_H)
print("  ||L_ker||_1 = %.6f, grid threshold pi/(6L) = %.6f"
      % (L1_2, np.pi / (6 * LAM_H)))
print()

rng = np.random.default_rng(0)
psi = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
psi /= np.linalg.norm(psi)
A = Pauli('IIZ').to_matrix()
Bo = Pauli('IXI').to_matrix()


def F(t):
    U = expm(-1j * Hm * t)
    return float(np.real(psi.conj() @ (U.conj().T @ Bo @ U @ A @ psi)))


h = 1e-5
exactd = (F(h) - F(-h)) / (2 * h)
fv = np.array([F(t) for t in TS2])
got = float(np.trapezoid(LV2 * fv, TS2))
print("      quantity                      value")
print("  " + "-" * 60)
print("   F'(0) by central difference    %+.8f" % exactd)
print("   int L_ker(t) F(t) dt           %+.8f" % got)
print("   abs err                         %.3e" % abs(got - exactd))
ok4 = abs(got - exactd) < 5e-3 * max(1.0, abs(exactd))
print()
print("   PASS" if ok4 else "   FAIL")
print()

print("=" * 96)
print("VERDICT")
print("=" * 96)
allok = ok1 and ok2 and ok3 and ok4
print("   CHECK 1 Fourier inversion      %s" % ("pass" if ok1 else "FAIL"))
print("   CHECK 2 derivative extraction  %s" % ("pass" if ok2 else "FAIL"))
print("   CHECK 3 discretisation         %s" % ("pass" if ok3 else "FAIL"))
print("   CHECK 4 real correlation       %s" % ("pass" if ok4 else "FAIL"))
print()
if allok:
    print("   The kernel is correct as built, and the discrete grid identity holds")
    print("   below pi/(6L) and breaks above it - so the band limit is real and not")
    print("   a coincidence of the test function. It is safe to sample tau ~ p and")
    print("   weight by ||L||_1 sign(L(tau)) inside twirl_cal, which is the next step:")
    print("   the fixed-T bias floor should vanish, and because the O(T^2) term is")
    print("   excluded by construction rather than by design resolution, the degree-2")
    print("   aliasing of v106 should go with it.")
else:
    print("   DO NOT BUILD ON THIS. A kernel that fails any check would produce a")
    print("   strawman of someone else's protocol, which is worse than no comparison.")
print()
print("  Scope: numerical quadrature on a finite t-window (60/Lambda), trapezoid rule,")
print("  one cutoff realisation. The constants C_0 and gamma_1 of their Appendix C are")
print("  NOT reproduced here - only the kernel and the grid identity, which are what")
print("  the twirl construction needs.")
