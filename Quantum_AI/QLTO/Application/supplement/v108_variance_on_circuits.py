"""The two standing caveats, re-measured on REAL CIRCUITS instead of a synthetic landscape.

Two caveats are quoted beside every V6 cost claim, and both come from evidence
that cannot see a circuit:

  v82  "V6 variance exponent 1.94 vs parameter-shift 2.00, so at matched TOTAL
        SHOTS the advantage is circuits, not shots."
  v69  "gradient error ~ T^(-1/3) for V6 against the unbiased T^(-1/2)."

WHAT IS WRONG WITH CITING THEM THAT WAY.

  1. v82 IS TIER C. Zero QuantumCircuit, AerSimulator or Statevector constructs
     anywhere in the file - it is a synthetic Gaussian model of an estimator. Its
     own docstring says so: "an ESTIMATOR-level measurement on a synthetic
     landscape... It says nothing about circuit depth or hardware." Under project
     rule R1 that may support mechanism, never a cost or accuracy figure, and it
     is currently constraining a tier-A benchmark.

  2. v82 MEASURED TWO REGIMES AND ONLY ONE IS EVER QUOTED IN THE CAVEAT.
         per-component fixed  V6 1.940  PS 2.000   <- the caveat
         total norm fixed     V6 1.006  PS 2.000   <- PS/V6 reaches 26.3x at M=128
     The same README quotes the second under "what is measured" and the first
     under "what is not claimed", without saying which governs the benchmark.

  3. ON A REAL LANDSCAPE THE REGIME IS NOT A CHOICE. v82 imposed each regime by
     rescaling a synthetic g. A real ansatz has whatever regime it has, and that
     is measurable: track |grad E| and the per-component magnitude as M grows and
     see which synthetic picture the truth resembles.

  4. v69's "-1/3 against -1/2" ARE THE PREDICTIONS, NOT THE MEASUREMENTS. Its log
     fits -0.742 / -0.759 for QLTO against -0.941 / -0.921 for parameter-shift on
     (1-cos), which tracks error SQUARED. So the measured ERROR exponents are
     about -0.371 against -0.470: a gap of 0.099, not the 0.167 the "-1/3 vs
     -1/2" framing implies. QLTO beat its own prediction and parameter-shift
     missed its own.

WHAT THIS FILE MEASURES, all tier A except the reference.

  For a family of REAL problems with growing M, at MATCHED TOTAL SHOTS per
  gradient, with everything sampled from real circuits:

      tr(Cov)   spread of the estimate over repeats  - what v82 modelled
      bias^2    |mean(estimate) - g_exact|^2         - what v82 EXCLUDED by
                construction, and what the R-smearing actually costs
      MSE       bias^2 + tr(Cov)                     - the only honest total
      cos       direction quality, which is what a max-normalised step uses

  Reporting tr(Cov) alone favours a biased-but-tight estimator, which is exactly
  what V6 is, so the variance exponent on its own cannot settle the question the
  caveat claims to settle. MSE can.

  g_exact is computed by parameter-shift on Statevector - a dense reference the
  circuits are checked against, which R1 lists as sanctioned use.

WHICH REGIME THE BENCHMARK IS IN gets answered on the way: |grad E| and the mean
per-component magnitude are printed at every M.
"""
import sys, os, contextlib, io
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import benchmark as B
from nisq_v6 import QLTOv6

N = 4
REPS_LIST = [1, 2, 3, 4]           # M = 2N(r+1) = 16, 24, 32, 40
TOTAL_SHOTS = 1 << 17              # per gradient, matched across methods
REPEATS = 20
R_SENSE = 0.45
SEED0 = 1000

_, H, _ = B.get_heisenberg_problem(N)


def exact_gradient(ansatz, H, theta):
    """Parameter-shift on exact amplitudes. Dense reference, no sampling."""
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s, sgn in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
            t = np.array(theta, float)
            t[i] += s
            sv = Statevector(ansatz.assign_parameters(t))
            g[i] += sgn * 0.5 * float(np.real(sv.expectation_value(H)))
    return g


def energy_shots(backend, ansatz, groups, theta, shots):
    """Energy from real circuits: one circuit per commuting group."""
    from qiskit import transpile
    tot = 0.0
    for grp in groups:
        qc = ansatz.assign_parameters(np.asarray(theta, float)).copy()
        axis = {}
        for lbl in grp.paulis.to_labels():
            for q, ch in enumerate(reversed(lbl)):
                if ch != 'I':
                    axis[q] = ch
        for q, ch in axis.items():
            if ch == 'X':
                qc.h(q)
            elif ch == 'Y':
                qc.sdg(q); qc.h(q)
        qc.measure_all()
        tq = transpile(qc, backend, optimization_level=1)
        counts = backend.run(tq, shots=shots).result().get_counts()
        n = sum(counts.values())
        labels = grp.paulis.to_labels()
        coeffs = np.real(grp.coeffs)
        acc = 0.0
        for bit, c in counts.items():
            b = bit.replace(' ', '')[::-1]
            for lbl, co in zip(labels, coeffs):
                s = 1
                for q, ch in enumerate(reversed(lbl)):
                    if ch != 'I' and b[q] == '1':
                        s = -s
                acc += co * s * c
        tot += acc / max(n, 1)
    return tot


def pshift_gradient(backend, ansatz, groups, theta, shots_per_circuit):
    """2MG circuits, finite shots. The unbiased baseline."""
    M = len(theta)
    g = np.zeros(M)
    for i in range(M):
        tp = np.array(theta, float); tp[i] += np.pi / 2
        tm = np.array(theta, float); tm[i] -= np.pi / 2
        g[i] = 0.5 * (energy_shots(backend, ansatz, groups, tp, shots_per_circuit)
                      - energy_shots(backend, ansatz, groups, tm, shots_per_circuit))
    return g


print("=" * 104)
print("v108  THE TWO CAVEATS, ON REAL CIRCUITS")
print("=" * 104)
print("  Heisenberg N=%d, efficient_su2 reps=%s, matched TOTAL shots = %d per gradient,"
      % (N, REPS_LIST, TOTAL_SHOTS))
print("  %d repeats per point. TIER A except g_exact, which is the dense reference."
      % REPEATS)
print()
print("  WHICH REGIME IS THE REAL LANDSCAPE IN?")
print("     M    |grad E|   mean|g_i|    per-component magnitude vs M")
print("  " + "-" * 72)
rows = []
for r in REPS_LIST:
    anz = efficient_su2(N, reps=r)
    M = anz.num_parameters
    rng = np.random.default_rng(7)
    theta = rng.uniform(-np.pi, np.pi, M)
    gx = exact_gradient(anz, H, theta)
    rows.append((r, anz, M, theta, gx))
    print("   %4d   %8.4f   %9.5f" % (M, np.linalg.norm(gx), np.mean(np.abs(gx))))
print()
norms = np.array([np.linalg.norm(x[4]) for x in rows])
percomp = np.array([np.mean(np.abs(x[4])) for x in rows])
Ms = np.array([x[2] for x in rows], float)
a_norm = np.polyfit(np.log(Ms), np.log(norms), 1)[0]
a_perc = np.polyfit(np.log(Ms), np.log(percomp), 1)[0]
print("   |grad E| ~ M^%.3f      mean|g_i| ~ M^%.3f" % (a_norm, a_perc))
print("   v82's 'total norm fixed' predicts  0.0 and -0.5;")
print("   v82's 'per-component fixed' predicts  +0.5 and 0.0.")
print()

print("=" * 104)
print("THE ESTIMATORS, at matched total shots")
print("=" * 104)
print("     M   method     circuits  shots/circ    tr(Cov)      bias^2         MSE      cos")
print("  " + "-" * 96)
res = {'v6': [], 'ps': []}
for r, anz, M, theta, gx in rows:
    q0 = QLTOv6(anz, H, shot_budget=1, sim_seed=1)
    G = len(q0.groups)
    # --- V6: G circuits per gradient
    sh_v6 = max(1, TOTAL_SHOTS // G)
    ests = []
    for k in range(REPEATS):
        be = AerSimulator(seed_simulator=SEED0 + k)
        q = QLTOv6(anz, H, shot_budget=sh_v6, sim_seed=SEED0 + k, backend=be)
        with contextlib.redirect_stdout(io.StringIO()):
            g, _ = q.sense(theta, R_SENSE, list(range(M)))
        ests.append(g)
    ests = np.array(ests)
    cov = np.var(ests, axis=0, ddof=1).sum()
    bias2 = float(np.sum((ests.mean(0) - gx) ** 2))
    cos = float(np.dot(ests.mean(0), gx) /
                (np.linalg.norm(ests.mean(0)) * np.linalg.norm(gx)))
    res['v6'].append((M, cov, bias2, cov + bias2, cos))
    print("   %4d   V6         %5d    %8d   %.4e  %.4e  %.4e   %.4f"
          % (M, G, sh_v6, cov, bias2, cov + bias2, cos))

    # --- parameter-shift: 2MG circuits per gradient
    nc = 2 * M * G
    sh_ps = max(1, TOTAL_SHOTS // nc)
    ests = []
    for k in range(REPEATS):
        be = AerSimulator(seed_simulator=SEED0 + 500 + k)
        ests.append(pshift_gradient(be, anz, q0.groups, theta, sh_ps))
    ests = np.array(ests)
    cov = np.var(ests, axis=0, ddof=1).sum()
    bias2 = float(np.sum((ests.mean(0) - gx) ** 2))
    cos = float(np.dot(ests.mean(0), gx) /
                (np.linalg.norm(ests.mean(0)) * np.linalg.norm(gx)))
    res['ps'].append((M, cov, bias2, cov + bias2, cos))
    print("   %4d   p-shift    %5d    %8d   %.4e  %.4e  %.4e   %.4f"
          % (M, nc, sh_ps, cov, bias2, cov + bias2, cos))
    print()

print("=" * 104)
print("FITTED EXPONENTS  (tier A, real circuits, matched total shots)")
print("=" * 104)
lm = np.log(np.array([x[0] for x in res['v6']], float))
print("   quantity        V6 alpha    p-shift alpha     v82 said (tier C, synthetic)")
print("  " + "-" * 88)
for j, name in ((1, 'tr(Cov)'), (2, 'bias^2'), (3, 'MSE')):
    av = np.polyfit(lm, np.log(np.maximum([x[j] for x in res['v6']], 1e-30)), 1)[0]
    ap = np.polyfit(lm, np.log(np.maximum([x[j] for x in res['ps']], 1e-30)), 1)[0]
    extra = "V6 1.940 / 1.006,  PS 2.000" if name == 'tr(Cov)' else ""
    print("   %-14s  %+8.3f    %+8.3f          %s" % (name, av, ap, extra))
print()
print("=" * 104)
print("READING IT")
print("=" * 104)
print("  The caveat says the shot-side advantage is absent. On circuits that is a")
print("  claim about the MSE column, not the tr(Cov) column, because V6 is biased")
print("  by construction and a variance comparison alone flatters it. Compare the")
print("  MSE exponents, and compare cos, which is what a max-normalised step uses.")
print()
print("  The regime table above says which of v82's two synthetic worlds the real")
print("  landscape resembles - and therefore which of its two exponents, 1.940 or")
print("  1.006, the caveat was entitled to quote.")
print()
print("  Scope: one problem family, one N, one theta, %d repeats, reps 1-4. A")
print("  covariance trace from %d repeats carries roughly %.0f%% relative error on"
      % (REPEATS, REPEATS, 100 * np.sqrt(2.0 / (REPEATS - 1))))
print("  each variance, so exponents fitted over four points are indicative.")
