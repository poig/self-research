"""Is twirl_cal's 3.0% the ESTIMATOR's truncation, or the simulated DEVICE's?

twirl_cal reports 3.0% mean relative error at T=0.25 on N=3 crosstalk and
attributes the residual entirely to first-order truncation in T - "every estimate
at larger T is systematically low". That attribution is plausible and untested,
and there is a second error source sitting underneath it.

THE DEVICE IN THE SIMULATION IS ITSELF A TROTTER CIRCUIT. twirl_cal._circuit
appends

    PauliEvolutionGate(H, time=T, synthesis=SuzukiTrotter(order=2, reps=12))

to stand in for an always-on chip Hamiltonian. On hardware that evolution is exact
by definition - the device IS its own Hamiltonian, which is the whole point of the
construction and why it carries no product formula. In simulation it is
synthesised, so the circuit implements exp(-i H_eff T) for some H_eff that is not
H(c_true). The estimator is then scored against c_true.

So the reported error is a SUM of two things that have never been separated:

    err_total     = |chat - c_true|     what the commit reports
    err_device    = |c_eff - c_true|    the simulation's own synthesis error
    err_estimator = |chat - c_eff|      what the estimator achieves against the
                                        thing it actually measured

THE DECOMPOSITION IS AVAILABLE EXACTLY. U = exp(-i H_eff T) is the Trotter
circuit's unitary, so H_eff = i log(U)/T and c_eff_k = Tr(P_k H_eff)/2^N. At N=3
that is an 8x8 matrix logarithm, well conditioned for small T since every
eigenvalue of -i H T sits near zero.

TIER (project rule R1). PART 1 is the reference calculation a circuit is checked
against: the Trotter circuit is BUILT as a Qiskit circuit and read exactly by
Operator, then compared against scipy's expm - dense linear algebra used as ground
truth, which is the sanctioned use. No accuracy claim rests on it. PART 2 is
tier A throughout: real circuits, AerSimulator, finite shots.

AN R1 TRAP, HIT WHILE WRITING THIS, AND WORTH RECORDING. The obvious way to get
the device unitary is Operator(qc) on the circuit holding the PauliEvolutionGate.
That returns the EXACT evolution - 1.147e-16 against expm at every reps, including
reps=1 - because Operator reads the gate's own matrix and never touches the
synthesis. Depth still scales 70 -> 1680, so the circuit really is being built;
only the readout bypasses it. Decomposing first shows the true error: 6.897e-04 at
reps=1, 1.721e-04 at reps=2 (a ratio of 4.01, second order confirmed), 4.778e-06
at reps=12. Operator(transpile(qc)) agrees with the decomposed value, which is why
PART 2 was never affected - AerSimulator executes the real circuit. Same failure
family as the StatevectorEstimator trap in RESEARCH_NOTES.md: an object that
looks like it measures the circuit and instead measures the ideal it stands for.
Hence decompose() below, not bare Operator().

WHAT WOULD SETTLE IT.

  err_device << err_total at reps=12, AND err_total flat in reps
      -> the commit's attribution is right, 3.0% is the estimator's own, and the
         headline stands unchanged.

  err_total FALLS as reps grows
      -> part of the reported 3.0% belongs to the simulation's device model, the
         headline is pessimistic, and the operating-point table needs re-reading.

  err_total RISES as reps grows
      -> the two errors partially cancel at reps=12, which would make the shipped
         default accidentally flattering and is the least comfortable outcome.

A PREDICTION, STATED BEFORE RUNNING. Second-order Suzuki error scales as
O(T^3/reps^2), which at T=0.25 and reps=12 should sit far below 3%. So I expect
err_device to be negligible and err_total to be flat. The reason to run it anyway
is that the estimator DIVIDES BY T: a coefficient is recovered as
g/(T <i[P,O]>), so an error eps in the unitary is amplified to about eps/T = 4x at
this operating point, and "negligible" has to be checked against that
amplification rather than against 3% directly.

LEAKAGE. Trotter error does not only shift the modelled coefficients, it generates
Pauli terms OUTSIDE span(terms). Those cannot be represented by any c_eff, so they
are reported separately: a device that is not in the model class at all is a
different problem from one whose coefficients are merely shifted.
"""
import sys, os
import numpy as np
from scipy.linalg import logm, expm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator

from twirl_cal import TwirlCalibrator, crosstalk_terms, crosstalk_coeffs

N = 3
T_MAIN = 0.25
SHOTS = 1 << 16
N_PROBES = 4
REPS = [1, 2, 4, 8, 12, 24]
SEEDS = [11, 22, 33]

terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
M = len(terms)
H = SparsePauliOp.from_list(list(zip(terms, c_true))).simplify()
IN_MODEL = set(terms)


def all_paulis(n):
    out = []
    for i in range(4 ** n):
        s, t = "", i
        for _ in range(n):
            s += "IXYZ"[t & 3]
            t >>= 2
        out.append(s)
    return out


ALL = all_paulis(N)


def effective_coeffs(T, reps):
    """c_eff and out-of-model leakage of the Trotter circuit at this reps.

    The circuit is BUILT (Qiskit) and read exactly; expm is the reference.
    DECOMPOSE FIRST - bare Operator(qc) returns the gate's exact matrix and
    silently discards the synthesis. See the docstring.
    """
    qc = QuantumCircuit(N)
    qc.append(PauliEvolutionGate(
        H, time=T, synthesis=SuzukiTrotter(order=2, reps=reps)), range(N))
    U = Operator(qc.decompose(reps=6)).data
    Heff = 1j * logm(U) / T
    c_eff = np.array([np.real(np.trace(Pauli(t).to_matrix() @ Heff)) / (2 ** N)
                      for t in terms])
    leak = 0.0
    for p in ALL:
        if p in IN_MODEL or p == "I" * N:
            continue
        leak += abs(np.real(np.trace(Pauli(p).to_matrix() @ Heff)) / (2 ** N)) ** 2
    Uex = expm(-1j * H.to_matrix() * T)
    return c_eff, np.sqrt(leak), np.linalg.norm(U - Uex, 2)


def relerr(a, b):
    return np.abs(a - b) / np.abs(b)


def trotter_depth(T, reps):
    qc = QuantumCircuit(N)
    qc.append(PauliEvolutionGate(
        H, time=T, synthesis=SuzukiTrotter(order=2, reps=reps)), range(N))
    return qc.decompose(reps=3).depth()


print("=" * 98)
print("v102  DEVICE_REPS:  is the reported 3.0% the estimator, or the simulated device?")
print("=" * 98)
print("  N=%d crosstalk, M=%d terms, T=%.2f, shots=%d, n_probes=%d -> %d circuits/estimate"
      % (N, M, T_MAIN, SHOTS, N_PROBES, N_PROBES * 2 * N))
print("  c_true =", np.round(c_true, 4))
print()

print("=" * 98)
print("PART 1  WHAT THE SIMULATED DEVICE ACTUALLY IMPLEMENTS")
print("        TIER B/C REFERENCE - circuit built and read exactly, expm as ground")
print("        truth, no sampling. Supports no accuracy claim on its own.")
print("=" * 98)
print("  H_eff = i log(U)/T from the Trotter circuit's own unitary; c_eff is its")
print("  projection onto the modelled terms. 'leak' is the RMS weight on Paulis")
print("  OUTSIDE span(terms) - error no choice of c can absorb.")
print()
print("   reps   mean |c_eff-c_true|/|c_true|    max |dc_k|      leak      ||U-U_exact||_2")
print("   " + "-" * 90)
c_eff_at = {}
for r in REPS:
    c_eff, leak, d = effective_coeffs(T_MAIN, r)
    c_eff_at[r] = c_eff
    rel = relerr(c_eff, c_true)
    print("   %4d          %10.3e             %10.3e   %9.3e     %9.3e"
          % (r, np.mean(rel), np.max(np.abs(c_eff - c_true)), leak, d))
print()

print("=" * 98)
print("PART 2  DOES THE MEASURED ERROR MOVE WITH reps?   TIER A - real circuits, shots")
print("=" * 98)
print("  Same probes on every row (probe_seed=0), so only the device synthesis and")
print("  the shot draw differ. %d seeded simulator runs per row; +- is the sd."
      % len(SEEDS))
print()
print("   reps   circuits   mean rel err vs c_true     vs c_eff(reps)     depth")
print("   " + "-" * 90)
rows = []
for r in REPS:
    tot, dev = [], []
    for sd in SEEDS:
        be = AerSimulator(method="statevector", seed_simulator=sd)
        cal = TwirlCalibrator(terms, evolution_time=T_MAIN, shots=SHOTS,
                              seed=sd, device_reps=r, backend=be)
        chat = cal.estimate(c_true, n_probes=N_PROBES, probe_seed=0,
                            grouped=False)   # PINNED: these logs predate v105
        tot.append(np.mean(relerr(chat, c_true)))
        dev.append(np.mean(relerr(chat, c_eff_at[r])))
    rows.append((r, np.mean(tot), np.std(tot), np.mean(dev), np.std(dev)))
    print("   %4d     %4d       %.4f +- %.4f         %.4f +- %.4f      %5d"
          % (r, N_PROBES * 2 * N, np.mean(tot), np.std(tot),
             np.mean(dev), np.std(dev), trotter_depth(T_MAIN, r)))
print()

base = [x for x in rows if x[0] == 12][0]
spread = max(x[1] for x in rows) - min(x[1] for x in rows)
noise = float(np.mean([x[2] for x in rows]))

print("=" * 98)
print("READING IT")
print("=" * 98)
print("  shipped default reps=12 :  %.4f +- %.4f  vs c_true" % (base[1], base[2]))
print("  cheapest        reps=1  :  %.4f +- %.4f" % (rows[0][1], rows[0][2]))
print("  most converged  reps=24 :  %.4f +- %.4f" % (rows[-1][1], rows[-1][2]))
print("  spread across reps      :  %.4f" % spread)
print("  typical seed-to-seed sd :  %.4f" % noise)
print()
if spread <= 2.0 * noise:
    print("  FLAT within seed noise. The reported error does NOT come from the device")
    print("  model, so twirl_cal's attribution to first-order truncation in T stands,")
    print("  and the 3.0% headline is the estimator's own. device_reps is then free to")
    print("  lower for simulation speed.")
else:
    print("  MOVES with reps by more than seed noise. Part of the reported error is the")
    print("  simulation's device model rather than the estimator, and twirl_cal's")
    print("  operating-point table needs re-reading at converged reps.")
print()
print("  Scope: one coefficient draw (seed 7), N=3, noiseless, one T. Widening any of")
print("  those is a separate experiment; this one asks only whether reps is a confound.")
print()

# ---------------------------------------------------------------------------
# PART 3 - not the question this file set out to ask, but it fell out of PART 2
# and it is larger than the answer.
# ---------------------------------------------------------------------------
T_SWEEP = [0.10, 0.25, 0.50, 1.00]
SHOT_SET = [1 << 16, 1 << 19]
SEEDS3 = [11, 22, 33]

print("=" * 98)
print("PART 3  THE SEED SPREAD, AND WHAT IT DOES TO THE REPORTED OPERATING POINT")
print("=" * 98)
print("  PART 2 measures 0.067 +- 0.010 at exactly the configuration twirl_cal reports")
print("  0.0297 for. Reproducing that configuration directly over 4 unseeded trials")
print("  gives 0.0662 +- 0.0246. The commit discloses 'no seed averaging on the circuit")
print("  numbers'; this is what that disclosure costs. The estimator divides by")
print("  <i[P_k,O]>, so terms a probe barely sees amplify shot noise, and the")
print("  run-to-run spread is a large fraction of the mean.")
print()
print("  The reported table then rests on differences smaller than its own noise:")
print("  0.0297 at 65536 shots against 0.0331 at 524288 was read as 'no gain,")
print("  bias-limited'. Re-run with seeds, does that survive?")
print()
print("      T      shots    circuits    mean rel err (%d seeds)      max rel err"
      % len(SEEDS3))
print("   " + "-" * 88)
grid = {}
for Tv in T_SWEEP:
    for sh in SHOT_SET:
        ms, xs = [], []
        for sd in SEEDS3:
            be = AerSimulator(method="statevector", seed_simulator=sd)
            cal = TwirlCalibrator(terms, evolution_time=Tv, shots=sh, seed=sd,
                                  device_reps=12, backend=be)
            chat = cal.estimate(c_true, n_probes=N_PROBES, probe_seed=0,
                            grouped=False)   # PINNED: these logs predate v105
            rel = relerr(chat, c_true)
            ms.append(np.mean(rel))
            xs.append(np.max(rel))
        grid[(Tv, sh)] = (float(np.mean(ms)), float(np.std(ms)), float(np.mean(xs)))
        print("   %5.2f   %8d     %4d       %.4f +- %.4f            %.4f"
              % (Tv, sh, N_PROBES * 2 * N, np.mean(ms), np.std(ms), np.mean(xs)))
    print()

LO, HI = SHOT_SET
best_lo = min(T_SWEEP, key=lambda t: grid[(t, LO)][0])
best_hi = min(T_SWEEP, key=lambda t: grid[(t, HI)][0])

print("=" * 98)
print("READING PART 3")
print("=" * 98)
print("  (a) THE OPERATING POINT. Best T is %.2f at %d shots and %.2f at %d shots."
      % (best_lo, LO, best_hi, HI))
m25lo, s25lo, _ = grid[(0.25, LO)]
m50lo, s50lo, _ = grid[(0.50, LO)]
if abs(m25lo - m50lo) < max(s25lo, s50lo):
    print("      At %d shots T=0.25 and T=0.50 differ by %.4f against sd %.4f - INSIDE"
          % (LO, abs(m25lo - m50lo), max(s25lo, s50lo)))
    print("      the noise, so the low-shot row does not order them at all. The")
    print("      operating point is only resolvable at the higher shot count.")
print()

print("  (b) BIAS-LIMITED, OR SHOT-LIMITED? twirl_cal reads 0.0297 -> 0.0331 across")
print("      8x shots as 'no gain - bias-limited'. Per T, the seed-averaged ratio:")
print()
print("        T      %d shots      %d shots     ratio    verdict" % (LO, HI))
print("      " + "-" * 74)
for Tv in T_SWEEP:
    mlo = grid[(Tv, LO)][0]
    mhi = grid[(Tv, HI)][0]
    slo = grid[(Tv, LO)][1]
    ratio = mlo / max(mhi, 1e-12)
    verdict = "shot-limited" if (mlo - mhi) > 2 * slo else "bias-limited"
    print("      %5.2f     %.4f        %.4f       %5.2fx   %s"
          % (Tv, mlo, mhi, ratio, verdict))
print()

r25 = grid[(0.25, LO)][0] / max(grid[(0.25, HI)][0], 1e-12)
print("      At the operating point T=0.25 the error falls %.1fx for 8x shots." % r25)
if r25 > 1.5:
    print("      THE 'BIAS-LIMITED' READING IS REFUTED THERE. It is shot-limited, and")
    print("      the original conclusion came from comparing two single draws whose")
    print("      difference (0.0034) is an eighth of the seed spread (~0.025). The")
    print("      diagnosis IS right at large T, where truncation dominates - which is")
    print("      why it looked plausible.")
print()
print("  (c) THE HEADLINE. 3.0% was a favourable single draw from a distribution")
print("      centred near 6.6% at that shot count. Restated as a seed mean it is")
print("      %.1f%% at 65536 shots and %.1f%% at 524288 - and the second number is"
      % (100 * grid[(0.25, LO)][0], 100 * grid[(0.25, HI)][0]))
print("      BETTER than the figure originally claimed, reached honestly. The")
print("      comparison against QLTO fit()'s ~30% in 160 circuits survives either")
print("      way, which is why restating costs nothing.")
