"""twirl_cal against the in-house iterative path, on ONE statistic and ONE budget.

The commit for twirl_cal claims "roughly 10x the accuracy at 6.7x fewer circuits"
against qlto_hl.fit(). That comparison does not hold up as stated, for three
separate reasons found while checking it, and this file replaces it with one that
does.

WHY THE ORIGINAL COMPARISON IS NOT APPLES TO APPLES.

  1. DIFFERENT STATISTICS. twirl_cal's 3.0% is a MEAN relative error over all
     terms. fit()'s ~30% is quoted "on the weak ZZ terms" - a worst case on the
     smallest coefficients. twirl_cal's own max relative error at that setting is
     0.27, so max-against-max the gap is not 10x.

  2. DIFFERENT SHOT BUDGETS. fit() runs 160 circuits at 8192 shots = 1.31M shots.
     twirl_cal's best figure used 24 circuits at 524288 = 12.6M shots, 9.6x more.
     Circuit count is a real currency - vendors bill per task - but it is not the
     only one, and quoting the cheap axis while spending 9.6x on the other is the
     move this project's own README warns about.

  3. DIFFERENT PROBLEMS. benchmark_hl starts fit() from
     th0 = c_true + uniform(-0.4, 0.4): a WARM START within 0.4 of the answer.
     twirl_cal is a direct estimator and is handed nothing. An iterative method
     given the neighbourhood of the solution is solving an easier problem, and
     this asymmetry runs the OTHER way - it flatters fit(), not twirl_cal.

  4. AND v102 SHOWED THE 3.0% ITSELF WAS A SINGLE LUCKY DRAW. The seed mean at
     that configuration is 6.7%.

SO: same Hamiltonian, same statistic reported three ways, matched total shots,
and the warm start left in place for fit() because removing it would be a
different experiment - but named, so the reader can discount it.

TIER (project rule R1): tier A throughout. Both arms are real Qiskit circuits on
AerSimulator with finite shots.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit_aer import AerSimulator
from twirl_cal import TwirlCalibrator, crosstalk_terms, crosstalk_coeffs
from qlto_hl import QLTOHamiltonianLearner

N = 3
T_TWIRL = 0.25
N_PROBES = 4
SEEDS = [11, 22, 33]
EPOCHS = 30

terms = crosstalk_terms(N)
c_true = crosstalk_coeffs(N)
M = len(terms)
TWIRL_CIRCUITS = N_PROBES * 2 * N


def stats(chat):
    d = np.abs(chat - c_true)
    rel = d / np.abs(c_true)
    return float(np.max(d)), float(np.mean(rel)), float(np.max(rel))


print("=" * 100)
print("v104  HEAD TO HEAD:  twirl design vs the iterative path, one statistic, matched shots")
print("=" * 100)
print("  N=%d crosstalk, M=%d terms, c_true seed 7." % (N, M))
print("  %d seeds per arm. Reported three ways because the original comparison"
      % len(SEEDS))
print("  mixed a mean against a worst case.")
print()

# ---- arm 1: twirl design, at two shot budgets -------------------------------
rows = []
for shots in (1 << 16, 1 << 19):
    A, B, C = [], [], []
    for sd in SEEDS:
        be = AerSimulator(method='statevector', seed_simulator=sd)
        cal = TwirlCalibrator(terms, evolution_time=T_TWIRL, shots=shots, seed=sd,
                              device_reps=1, backend=be)
        chat = cal.estimate(c_true, n_probes=N_PROBES, probe_seed=0,
                            grouped=False)   # PINNED: these logs predate v105
        a, b, c = stats(chat)
        A.append(a); B.append(b); C.append(c)
    rows.append(('twirl design', TWIRL_CIRCUITS, shots,
                 TWIRL_CIRCUITS * shots, np.mean(A), np.mean(B), np.mean(C),
                 np.std(B), 'direct, no initial guess'))

# ---- arm 2: qlto_hl.fit(), warm start ---------------------------------------
# fit() spends ONE circuit per epoch, so epochs IS the circuit count. The
# twirl_cal commit cites fit() at 160 circuits, so 160 epochs is the configuration
# that figure refers to; 30 was under-configured and left the iterate essentially
# on its warm start (max|dc| 0.3935 against a +-0.4 start). Both are shown, and
# 320 too, to make it plain the baseline is given room rather than clipped.
for epochs in (30, 160, 320):
    for shots in (8192,):
        A, B, C, NC = [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(100 + sd)
            th0 = np.asarray(c_true, float) + rng.uniform(-0.4, 0.4, M)
            q = QLTOHamiltonianLearner(terms, evolution_time=1.0, shots=shots,
                                       seed=sd)
            th, tr = q.fit(c_true, th0, epochs=epochs)
            a, b, c = stats(np.asarray(th))
            A.append(a); B.append(b); C.append(c); NC.append(q.nefv)
        nc = int(np.mean(NC))
        rows.append(('qlto_hl.fit() e%d' % epochs, nc, shots, nc * shots,
                     np.mean(A), np.mean(B), np.mean(C), np.std(B),
                     'iterative, WARM START +-0.4'))

print("   method            circuits   shots/circ   total shots   max|dc|   mean rel   max rel")
print("   " + "-" * 94)
for nm, nc, sh, tot, a, b, c, bs, note in rows:
    print("   %-16s %6d     %8d    %10.2e   %7.4f   %7.4f    %6.4f"
          % (nm, nc, sh, tot, a, b, c))
print()
print("   notes:")
for nm, nc, sh, tot, a, b, c, bs, note in rows:
    print("     %-16s %s" % (nm, note))
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
tw_lo = rows[0]
tw_hi = rows[1]
fit_lo = [r for r in rows if r[0].startswith('qlto_hl.fit() e160')][0]

print("  MATCHED-SHOT ROW is the honest one: twirl at %d shots/circuit spends"
      % tw_lo[2])
print("  %.2e total shots against fit()'s %.2e - within %.1fx."
      % (tw_lo[3], fit_lo[3], max(tw_lo[3], fit_lo[3]) / min(tw_lo[3], fit_lo[3])))
print()
print("    circuits    %6d  vs %6d      %.1fx fewer"
      % (tw_lo[1], fit_lo[1], fit_lo[1] / tw_lo[1]))
print("    mean rel    %.4f  vs %.4f      %.1fx"
      % (tw_lo[5], fit_lo[5], fit_lo[5] / max(tw_lo[5], 1e-9)))
print("    max  rel    %.4f  vs %.4f      %.1fx"
      % (tw_lo[6], fit_lo[6], fit_lo[6] / max(tw_lo[6], 1e-9)))
print("    max |dc|    %.4f  vs %.4f      %.1fx"
      % (tw_lo[4], fit_lo[4], fit_lo[4] / max(tw_lo[4], 1e-9)))
print()
print("  AND fit() had a warm start within 0.4 of the answer while twirl had none,")
print("  so any twirl win here is an understatement and any fit() win is inflated.")
print()
print("  The defensible claim is whatever the max-rel row supports at matched shots,")
print("  stated with the warm-start asymmetry named. 'Roughly 10x the accuracy at")
print("  6.7x fewer circuits' is withdrawn: it compared a lucky-draw mean against a")
print("  worst case, across a 9.6x shot gap, on an easier problem.")
