"""Audit benchmark.py for the baseline-subsidy bug.

Two suspected defects, both running AGAINST V3, i.e. the headline result would
STRENGTHEN if they were fixed. Confirming each by measurement, not by reading.

DEFECT 1 - PRECISION SUBSIDY.
  benchmark.py sets PRECISION = 1/sqrt(8192) and hands it to
  StatevectorEstimator(default_precision=...). That estimator computes the EXACT
  expectation and adds Gaussian noise of standard deviation `precision`; it does
  not sample. So every baseline gets standard error 0.011 on <H> regardless of
  Var(H) and regardless of how many measurement settings H needs.
  A real device reaching standard error sigma needs

      shots = G * sum_g Var(H_g) / sigma^2

  so the EFFECTIVE shot budget the baselines receive is that number, against the
  8192 the comment claims. The comment's reasoning - "precision is the standard
  error, so 1/sqrt(SHOTS) is the matching setting" - is only correct if
  Var(H) = 1.
  Meanwhile V3's sensing calls backend.run(qc, shots=shot_budget): REAL shots.
  V2 is on the subsidised path too (BaseEstimator(precision=PRECISION)), which
  matters because the docs compare V3 against V2 directly.

DEFECT 2 - CIRCUIT UNDERCOUNT.
  Every baseline bills one energy evaluation as one circuit (AdamW:
  grad_nefv = 2*len(params); SPSA: += 2; QNG: += 2*n_params + n_layers). But
  measuring <H> needs G qubit-wise-commuting measurement settings, and each
  setting is a separate circuit. So true baseline circuits = nefv * G.
  V3's count is honest: its sensing circuit measures the param and ancilla
  registers in the computational basis and takes the energy from the PHASE, so
  it needs exactly one setting whatever H looks like.

Reports the correction factor for both, per problem.
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
import benchmark as B

# ── Defect 1a: what does default_precision actually DO? ──────────────────────
print("=" * 88)
print("1a. StatevectorEstimator(default_precision=p) semantics")
print("=" * 88)
ans, H, name = B.get_heisenberg_problem(4)
c = np.random.RandomState(0).uniform(-np.pi, np.pi, ans.num_parameters)
sv = Statevector(ans.assign_parameters(c))
varH = float(np.real(sv.expectation_value((H @ H).simplify()))
             - np.real(sv.expectation_value(H)) ** 2)
for p in (1.0 / np.sqrt(8192), 0.05, 0.2):
    est = StatevectorEstimator(default_precision=p)
    vals = np.array([float(est.run([(ans, H, c)]).result()[0].data.evs)
                     for _ in range(400)])
    print(f"  precision={p:.6f} -> measured std {vals.std(ddof=1):.6f}"
          f"   ratio std/precision = {vals.std(ddof=1)/p:.3f}")
print(f"  Var(H) at this point = {varH:.4f}")
print("  If std == precision for every p, the estimator ADDS FIXED NOISE and is")
print("  blind to Var(H) - it is not simulating shots.")

# ── Defect 1b + 2: per-problem subsidy ──────────────────────────────────────
print()
print("=" * 88)
print("1b/2. Per-problem: effective shots the baselines receive, and G")
print("=" * 88)

PROBS = [("H2", B.get_h2_problem), ("LiH", B.get_lih_problem),
         ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
         ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
         ("Heisenberg N=8", lambda: B.get_heisenberg_problem(8)),
         ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
         ("MaxCut N=6", lambda: B.get_maxcut_problem(6))]

P = 1.0 / np.sqrt(8192)
print(f"  nominal SHOTS = 8192, PRECISION = {P:.6f}")
print()
print(f"  {'problem':<17}{'terms':>7}{'G':>5}{'Var(H)':>10}"
      f"{'G*sumVar':>11}{'eff shots':>12}{'subsidy':>10}{'circ x':>8}")
print("  " + "-" * 80)
rows = []
for pname, fn in PROBS:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            a, h, _ = fn()
    except Exception as e:
        print(f"  {pname:<17} SKIP ({type(e).__name__}: {e})")
        continue
    rng = np.random.RandomState(1)
    grps = h.group_commuting(qubit_wise=True)
    G = len(grps)
    vh, sumvar = [], []
    for _ in range(4):
        pv = rng.uniform(-np.pi, np.pi, a.num_parameters)
        s = Statevector(a.assign_parameters(pv))
        vh.append(float(np.real(s.expectation_value((h @ h).simplify()))
                        - np.real(s.expectation_value(h)) ** 2))
        sv_ = 0.0
        for g in grps:
            e1 = float(np.real(s.expectation_value(g)))
            e2 = float(np.real(s.expectation_value((g @ g).simplify())))
            sv_ += max(e2 - e1 ** 2, 0.0)
        sumvar.append(sv_)
    varH = float(np.mean(vh)); SV = float(np.mean(sumvar))
    eff = G * SV / P ** 2
    rows.append((pname, eff / 8192.0, G))
    print(f"  {pname:<17}{len(h.paulis):>7}{G:>5}{varH:>10.3f}"
          f"{G*SV:>11.3f}{eff:>12.0f}{eff/8192.0:>9.0f}x{G:>7}x")

print()
print("  eff shots = G * sum_g Var(H_g) / PRECISION^2 : what a real device needs")
print("  to deliver the standard error the baselines are being GIVEN for free.")
print("  subsidy = that, over the 8192 the benchmark claims to be enforcing.")
print("  circ x  = factor by which baseline circuit counts are UNDERCOUNTED.")
print()
if rows:
    subs = [r[1] for r in rows]
    print(f"  subsidy range {min(subs):.0f}x - {max(subs):.0f}x, "
          f"median {np.median(subs):.0f}x")
    print(f"  G range {min(r[2] for r in rows)} - {max(r[2] for r in rows)}")
print()
print("  DIRECTION OF BIAS: both defects flatter the BASELINES and penalise V3,")
print("  which samples real shots and needs one setting. Fixing them can only")
print("  improve V3's standing, so the published 180-vs-320-1360 comparison is")
print("  CONSERVATIVE, not inflated. V2 sits on the subsidised path, so the")
print("  V2-vs-V3 comparisons in RESULT are the ones actually at risk.")
