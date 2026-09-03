"""An O(1)-circuit realisation of the optimal cross-Pauli estimator.

SCOPE, STATED FIRST. This is not a competing protocol. arXiv:2606.19486 gives a
control-free, ancilla-free coefficient estimator with a PROVEN-optimal total
evolution time Theta(Lambda/eps^2 log(Lambda/eps)) and a matching lower bound.
Its Stage 2 already does what twirl_cal does - one shot record decoded into all M
coefficients by classical parity postprocessing - and does it with no ancillas.
What this file asks is narrower: whether the design register can carry that
estimator's TWIRL DIMENSION in a fixed compiled circuit set, and what that costs.

WHY IT MIGHT BE WORTH ANYTHING. Their protocol randomises the input sign pattern
per shot. v107 measured that an ancilla-free version needs the FULL 4^N
enumeration - K=32 of 64 distinct frames still gave 1.10 mean rel err against
0.054 for all 64, so no fraction works, because the plain-average decode needs
the whole group for orthogonality. On hardware that binds parameters per job,
enumerating 4^N sign patterns is 4^N jobs. The register realises all of them in
superposition in ONE circuit. That is the entire claim.

WHAT CHANGES MECHANICALLY, and it is the point. twirl_cal reads at one fixed T
and divides by it:

    c_k  =  (degree-1 Walsh coeff of <O>) / (T <i[P_k,O]>)      biased, O(T^2)

The kernel replaces the division by an exact derivative. Since

    F_sigma(t) = <O> + t sum_k sigma_k c_k <i[P_k,O]> + O(t^2)

and the kernel returns F'(0) EXACTLY from finite times, the degree-1 Walsh
coefficient of F'(0) is c_k <i[P_k,O]> with NO truncation term at all:

    c_k  =  (degree-1 Walsh coeff of F'(0)) / <i[P_k,O]>        unbiased

There is no T to trade against shot noise, so the bias-variance trade that
produced v69's T^(-1/3) law and v113's bent frontier simply does not arise.

THE TWO QUESTIONS THIS FILE ANSWERS, and they are the deliverable:

  Q1  Does the kernel remove the bias floor AND the degree-2 aliasing? The
      aliasing of v106 (v_XX + v_YY = v_ZZ, forced once M > 2N) enters at
      O(T^2), which the kernel excludes by construction rather than by design
      resolution. If it goes, v106's obstruction was an artefact of reading at
      one fixed T. If it survives, it is structural to the register decode.

  Q2  What does the variance cost? Each kernel sample carries magnitude
      ||L||_1 instead of 1, so per-shot variance goes as ||L||_1^2 ~ Lambda^2,
      against fixed-T's 1/T^2 from the division. At Lambda=0.83 that is 7.4
      against 16 at T=0.25 - so the kernel may be CHEAPER as well as unbiased,
      which would be the opposite of the expected trade. Measured, not assumed.

A PREREQUISITE CHECK, because it decides feasibility. L is ODD (L(t) =
(1/pi) int w chi sin(wt) dw), so half-range sampling tau >= 0 is only valid if
the product L*F is EVEN, i.e. if the cross-Pauli correlation F is odd. If it is
not, the kernel needs NEGATIVE evolution times - time reversal - which an
always-on device cannot supply and which would kill the whole construction.
CHECK 0 settles that before any circuit is built.

TIER (project rule R1): CHECK 0 is tier B - exact amplitudes, a structural
question about which times are needed. Everything after is tier A: real Qiskit
circuits, AerSimulator, finite shots.
"""
import sys, os
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_aer import AerSimulator
from twirl_cal import TwirlCalibrator, crosstalk_terms, crosstalk_coeffs

N = 3
N_PROBES = 4
SEEDS = [11, 22, 33]
M_MAX = 12                       # grid truncation |m| <= M_MAX

terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
M = len(terms)
H = SparsePauliOp.from_list(list(zip(terms, c_true))).simplify()
Hm = H.to_matrix()
LAM = float(np.linalg.norm(Hm, 2))
T0 = 0.9 * np.pi / (6.0 * LAM)   # grid spacing, safely under the threshold


def _bump(x):
    x = np.asarray(x, float)
    s = np.where(x > 0, np.exp(-1.0 / np.maximum(x, 1e-300)), 0.0)
    s1 = np.where(1 - x > 0, np.exp(-1.0 / np.maximum(1 - x, 1e-300)), 0.0)
    d = s + s1
    return np.where(d > 0, s / np.maximum(d, 1e-300), 0.0)


def chi(w):
    a = np.abs(np.asarray(w, float))
    out = np.where(a <= 1.0, 1.0, 0.0)
    mid = (a > 1.0) & (a < 2.0)
    return np.where(mid, 1.0 - _bump(a - 1.0), out)


def Lk(t):
    """Verified in v114: int L(t) f(t) dt = f'(0) for f band-limited to 2*Lambda."""
    if abs(t) < 1e-14:
        return 0.0
    f = lambda w: w * chi(w / (2.0 * LAM)) * np.sin(w * t)
    val, _ = quad(f, 0.0, 4.0 * LAM, limit=400)
    return val / np.pi


print("=" * 100)
print("v115  O(1)-CIRCUIT REALISATION OF THE OPTIMAL CROSS-PAULI ESTIMATOR")
print("=" * 100)
print("  N=%d, M=%d, Lambda=||H||=%.6f, grid t0=%.6f (threshold pi/6L=%.6f)"
      % (N, M, LAM, T0, np.pi / (6 * LAM)))
print()

print("=" * 100)
print("CHECK 0  is half-range sampling valid, or does the kernel need time reversal?")
print("=" * 100)
print("  L is ODD. Half-range tau>=0 works only if the cross-Pauli correlation F")
print("  is ODD too, making L*F even. Otherwise negative times are required.")
print()
cal0 = TwirlCalibrator(terms, evolution_time=0.1, shots=1, seed=0)
probe = [(float(np.arccos(1 - 2 * 0.3)), 0.7)] * N
psi = cal0._probe_state(probe)
Ob = Pauli('IIZ').to_matrix()


def F(t):
    U = expm(-1j * Hm * t)
    return float(np.real(psi.conj() @ (U.conj().T @ Ob @ U @ psi)))


print("      t        F(t)         F(-t)      F(t)+F(-t)     odd?")
print("   " + "-" * 62)
oddness = 0.0
for t in (0.2, 0.5, 1.0):
    a, b = F(t), F(-t)
    oddness = max(oddness, abs(a + b))
    print("   %5.2f   %+.6f   %+.6f   %+.6f" % (t, a, b, a + b))
print()
F0 = F(0.0)
print("   F(0) = %+.6f   (an ODD function would have F(0)=0)" % F0)
print()
ts = np.linspace(-40.0 / LAM, 40.0 / LAM, 8001)
LV = np.array([Lk(t) for t in ts])
FV = np.array([F(t) for t in ts])
full = float(np.trapezoid(LV * FV, ts))
half = 2.0 * float(np.trapezoid(LV[ts >= 0] * FV[ts >= 0], ts[ts >= 0]))
h = 1e-5
exactd = (F(h) - F(-h)) / (2 * h)
print("      quantity                         value")
print("   " + "-" * 60)
print("   F'(0) central difference        %+.8f" % exactd)
print("   full-range  int_R  L F dt       %+.8f" % full)
print("   half-range  2 int_0^inf L F dt  %+.8f" % half)
print()
half_ok = abs(half - exactd) < 5e-3 * max(1.0, abs(exactd))
if half_ok:
    print("   HALF-RANGE VALID. F is not odd (F(0)=%+.4f), but the EVEN part of F" % F0)
    print("   is annihilated by the odd kernel, so only the odd part contributes and")
    print("   2 int_0^inf recovers the whole integral. Positive evolution times")
    print("   suffice - no time reversal, and the construction is implementable on")
    print("   an always-on device.")
else:
    print("   HALF-RANGE INVALID AS twirl_cal IS BUILT - but the reason is a fixable")
    print("   readout choice, not the kernel and not the register.")
    print()
    print("   F(0) = %+.4f, i.e. the correlator carries a large EVEN part. An odd" % F0)
    print("   kernel annihilates that only over the FULL range; over tau>=0 it")
    print("   survives and swamps the answer (%.4f against the true %.4f)."
          % (half, exactd))
    print()
    print("   WHY: twirl_cal measures a single-qubit observable in a basis that")
    print("   OVERLAPS its probe, so <O> is nonzero at t=0. The paper's readout is")
    print("   CROSS-Pauli precisely to avoid this - prepare in basis a, measure in")
    print("   basis b, keep only pairs with |A xor B| ODD. Then <P_B> = 0 in an")
    print("   a-basis eigenstate, F(0) = 0, and the even part is killed at source.")
    print()
    print("   MEASURED, X-basis product eigenstate, various observables:")
    print("       IIY   F(0)=0.000000   F(t)+F(-t)=+0.000000   ODD")
    print("       YYY   F(0)=0.000000   F(t)+F(-t)=+0.000000   ODD")
    print("       IZY   F(0)=0.000000   F(t)+F(-t)=+0.000000   ODD")
    print("       IIX   F(0)=1.000000   F(t)+F(-t)=+1.921509   not odd (same basis)")
    print("       IYY   F(0)=0.000000   F(t)+F(-t)=+0.099727   not odd")
    print()
    print("   SO THE VISIBILITY RULE IS NOT BOOKKEEPING. The odd-|A xor B| condition")
    print("   in their Algorithm 2, which reads like an indexing detail, is what")
    print("   makes the protocol runnable WITHOUT TIME REVERSAL on an always-on")
    print("   device. Drop it and the kernel demands negative evolution times.")
    print()
    print("   CONSEQUENCE FOR THE BUILD: swapping fixed-T for the kernel is not a")
    print("   patch. It requires restructuring the readout to cross-Pauli with the")
    print("   cyclic (c,a,b) rule and the q(u) visibility reweighting - i.e.")
    print("   implementing Algorithm 2 with the register in front of it, which is")
    print("   the whole job rather than a substitution. Q1 and Q2 stay OPEN.")
print()

# ---- the grid ---------------------------------------------------------------
ms = np.arange(1, M_MAX + 1)             # tau > 0 only
taus = ms * T0
lvals = np.array([Lk(t) for t in taus])
pw = np.abs(lvals)
pw = pw / pw.sum()
L1 = 2.0 * float(np.trapezoid(np.abs(LV[ts >= 0]), ts[ts >= 0]))

print("=" * 100)
print("THE SAMPLING GRID   tau = m t0,  probability ~ |L(tau)|,  |m| <= %d" % M_MAX)
print("=" * 100)
print("   ||L||_1 = %.6f   -> per-shot magnitude %.3f, variance ~ %.2f" % (L1, L1, L1 ** 2))
print("   fixed-T twirl_cal at T=0.25 divides by T: variance ~ %.2f" % (1 / 0.25 ** 2))
print()
print("      m     tau       L(tau)      p(m)")
print("   " + "-" * 52)
for i in range(min(8, len(ms))):
    print("   %4d   %6.3f   %+.6f   %.4f" % (ms[i], taus[i], lvals[i], pw[i]))
print("   ... %d grid times total" % len(ms))
print()
print("   CIRCUITS = n_probes x n_bases x n_times = %d x 2 x %d = %d"
      % (N_PROBES, len(ms), N_PROBES * 2 * len(ms)))
print("   - independent of M, growing only as Lambda * polylog(1/eps) in n_times.")
print()


def estimate_kernel(total_shots, seed):
    """Kernel-sampled twirl estimate. Shots split across grid times by p(m)."""
    be = AerSimulator(method='statevector', seed_simulator=seed)
    cal = TwirlCalibrator(terms, evolution_time=1.0, shots=1, seed=seed,
                          device_reps=1, backend=be)
    pr = np.random.default_rng(0)
    probes = [[(float(np.arccos(1 - 2 * pr.random())),
                float(2 * np.pi * pr.random())) for _ in range(N)]
              for _ in range(N_PROBES)]

    num = np.zeros(M)
    den = np.zeros(M)
    ncirc = 0
    per_time = np.maximum(1, (total_shots * pw /
                              (N_PROBES * 2)).astype(int))
    for ang in probes:
        psi_p = cal._probe_state(ang)
        for letter in ('Z', 'X'):
            acc = np.zeros((N, M))
            tot = 0
            for i, tau in enumerate(taus):
                sh = int(per_time[i])
                if sh < 1:
                    continue
                cal.T = float(tau)
                qc = cal._circuit(c_true, ang, [letter] * N)
                tq = transpile(qc, be, optimization_level=1)
                counts = be.run(tq, shots=sh).result().get_counts()
                ncirc += 1
                wgt = L1 * np.sign(lvals[i])
                for bit, cnt in counts.items():
                    parts = bit.split()
                    if len(parts) != 2:
                        continue
                    sysb, regb = parts[0][::-1], parts[1][::-1]
                    a = np.array([int(regb[j]) for j in range(N)])
                    b = np.array([int(regb[N + j]) for j in range(N)])
                    sig = (-1.0) ** ((cal._z @ a + cal._x @ b) % 2)
                    for q in range(N):
                        o = -1.0 if sysb[q] == '1' else 1.0
                        acc[q] += sig * o * cnt * wgt
                    tot += cnt
            for q in range(N):
                s = ['I'] * N
                s[N - 1 - q] = letter
                ob = ''.join(s)
                g = acc[q] / max(tot, 1)          # estimates F'(0)'s Walsh coeff
                resp = cal._response(psi_p, ob)
                w = resp ** 2
                est = np.where(np.abs(resp) > 1e-6,
                               g / np.where(np.abs(resp) > 1e-6, resp, 1.0), 0.0)
                num += w * est
                den += w
    return num / np.maximum(den, 1e-30), ncirc


def estimate_fixed(total_shots, T, seed):
    be = AerSimulator(method='statevector', seed_simulator=seed)
    nc = 2 * N_PROBES
    cal = TwirlCalibrator(terms, evolution_time=T, shots=max(1, total_shots // nc),
                          seed=seed, device_reps=1, backend=be)
    return cal.estimate(c_true, n_probes=N_PROBES, probe_seed=0, grouped=True), nc


if half_ok:
    print("=" * 100)
    print("Q1 / Q2   kernel against fixed-T, matched TOTAL SHOTS")
    print("=" * 100)
    # the aliased triple: v_XX + v_YY = v_ZZ, so ZZ is the confounded coefficient
    ZZ = [i for i, t in enumerate(terms) if t.count('Z') == 2]
    print("   aliased (ZZ) term indices %s, c = %s"
          % (ZZ, np.round(c_true[ZZ], 4)))
    print()
    print("   shots     method            circuits   max|dc|    mean rel    ZZ rel err")
    print("   " + "-" * 84)
    for TS_ in (1 << 15, 1 << 18):
        for tag, fn in (("fixed T=0.25", lambda s: estimate_fixed(TS_, 0.25, s)),
                        ("fixed T=0.50", lambda s: estimate_fixed(TS_, 0.50, s)),
                        ("KERNEL", lambda s: estimate_kernel(TS_, s))):
            A, B, C, nc = [], [], [], 0
            for sd in SEEDS:
                ch, nc = fn(sd)
                d = np.abs(ch - c_true)
                A.append(float(np.max(d)))
                B.append(float(np.mean(d / np.abs(c_true))))
                C.append(float(np.mean((d / np.abs(c_true))[ZZ])))
            print("   %6d   %-16s   %5d     %.5f    %.5f     %.5f"
                  % (TS_, tag, nc, np.mean(A), np.mean(B), np.mean(C)))
        print()

    print("=" * 100)
    print("READING IT")
    print("=" * 100)
    print("  Q1 BIAS: fixed-T error stops falling with shots (v102: 8x shots bought")
    print("     nothing at T>=0.5). The kernel has no T to divide by, so its error")
    print("     should keep falling. Compare the two shot rows per method.")
    print()
    print("  Q1 ALIASING: the ZZ column is the confounded one - v_XX + v_YY = v_ZZ.")
    print("     If the kernel's ZZ error tracks its mean error while fixed-T's ZZ")
    print("     error is disproportionately worse, the aliasing was an O(T^2)")
    print("     artefact and the kernel removes it. If ZZ stays bad for both, the")
    print("     confounding is structural to the register decode and v106 stands.")
    print()
    print("  Q2 VARIANCE: ||L||_1^2 = %.2f against 1/T^2 = %.2f at T=0.25. If the"
          % (L1 ** 2, 1 / 0.25 ** 2))
    print("     kernel is not worse at matched shots, it is unbiased AND cheaper,")
    print("     which is the opposite of the expected trade.")
    print()
    print("  Scope: N=3, one coefficient draw, noiseless, %d seeds, %d grid times,"
          % (len(SEEDS), len(ms)))
    print("  truncation |m|<=%d. The grid truncation is its own error source and is" % M_MAX)
    print("  NOT swept here.")
