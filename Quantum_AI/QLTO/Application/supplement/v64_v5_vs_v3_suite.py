"""V5 against V3 across the full suite. V5 has had exactly one smoke test.

V5 was written as the standalone gradstep-only optimiser after v54 retired the
walk (0 of 7). It carries three changes worth auditing at suite scale:

  * WALK REMOVED. One circuit per block-epoch instead of two. v54's decision rule
    was "parity is a win" on that ratio alone, and it was established for V3's
    gradstep decoder - not for V5's reimplementation of it.
  * DIRECT READOUT DEFAULT. gradient_mode='direct' reads the marginal straight
    off the W-gate state; 'qpe' keeps the ancilla ladder. V3 only ever had the
    QPE path.
  * TWO BUGS FIXED SINCE. v60 found sense() was averaging the group
    contributions instead of summing them (a 1/G error, invisible under gradstep
    because it normalises by max|g|), and v63 found tau0 was scaled by H_range
    where V3 uses H0_norm - a full lost bit of QPE resolution on MaxCut/TFIM/H2,
    and a gradient cosine of 0.7389 against V3's 0.9584 on MaxCut N=6.

Both fixes are in as of this run, so this is the first measurement of V5 that is
not measuring a known defect. What it has to answer: does the direct path match
V3-gradstep on final energy at matched epochs, and what does it cost?

PROTOCOL. Same seven problems and the same 3 trials x 20 epochs as v54, same
initial parameters per (problem, trial) across all arms, MPS backend. Circuits
are each optimiser's own self-reported nefv, so the ratio is measured rather than
argued. Reported against the exact ground state so the rows are readable
independently of each other.

DECISION RULE, fixed before the numbers. V5-direct is the candidate default. It
wins if it is within the harness null scale (0.09, v54's) of V3-gradstep on a
clear majority of problems AND costs no more circuits. If it is behind on energy,
V3 stays the reference implementation and V5's docstring has to say so.
"""
import sys, os, contextlib, io, time, gc
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
import benchmark as B
import nisq_v3, nisq_v5
from qiskit_aer import AerSimulator

N_TRIALS = 3
EPOCHS = 20
SHOTS = 8192
NULL = 0.09                       # v54's harness null scale

PROBLEMS = [
    B.get_maxcut_problem(4, seed=101),
    B.get_maxcut_problem(6, seed=102),
    B.get_h2_problem(),
    B.get_heisenberg_problem(4),
    B.get_heisenberg_problem(6),
    B.get_heisenberg_problem(8),
    B.get_lih_problem(),
]


def run_v3(ansatz, H, bk, p0):
    q = nisq_v3.QLTOv3(ansatz, H, shot_budget=SHOTS, num_ancillas=3,
                       backend=bk)
    p = p0.copy()
    for ep in range(EPOCHS):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        p, _ = q.run_walk(p, k_steps=15, delta_t=dt, search_radius=R,
                          decoder='gradstep')
    return p, q.nefv


def run_v5(mode):
    def go(ansatz, H, bk, p0):
        q = nisq_v5.QLTOv5(ansatz, H, shot_budget=SHOTS, backend=bk,
                           gradient_mode=mode, num_ancillas=3)
        p, _ = q.minimize(p0.copy(), epochs=EPOCHS)
        return p, q.nefv
    return go


ARMS = [('V3 gradstep', run_v3),
        ('V5 direct', run_v5('direct')),
        ('V5 qpe', run_v5('qpe'))]

print("=" * 104)
print("V5 vs V3 — FULL SUITE")
print("=" * 104)
print(f"  {N_TRIALS} trials x {EPOCHS} epochs, {SHOTS} shots, shared p0 per")
print(f"  (problem, trial). Null scale {NULL}. Both sense() bugs fixed (v60 G")
print(f"  factor, v63 tau0). E is the mean final energy; C is self-reported nefv.")
print()
print(f"  {'problem':>16}{'exact':>9}" + "".join(f"{a:>14}" for a, _ in ARMS)
      + f"{'C v3':>8}{'C dir':>8}{'C qpe':>8}{'sec':>7}")
print("  " + "-" * 100)

score = {a: 0 for a, _ in ARMS}
for ansatz, H, name in PROBLEMS:
    t0 = time.time()
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    bk = AerSimulator(method='matrix_product_state')
    E, C = {}, {}
    for arm, fn in ARMS:
        fin, cst = [], []
        for t in range(N_TRIALS):
            rng = np.random.RandomState(42 + t)
            p0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    p, c = fn(ansatz, H, bk, p0)
                fin.append(B.report_energy(ansatz, H, p))
                cst.append(c)
            except Exception as e:
                print(f"  {name:>16}  {arm} failed: {type(e).__name__}: {e}",
                      flush=True)
            gc.collect()
        E[arm] = float(np.mean(fin)) if fin else float('nan')
        C[arm] = float(np.mean(cst)) if cst else float('nan')

    best = min((v for v in E.values() if v == v), default=float('nan'))
    for a in E:
        if E[a] == E[a] and E[a] <= best + NULL:
            score[a] += 1
    print(f"  {name:>16}{exact:>9.4f}"
          + "".join(f"{E[a]:>14.4f}" for a, _ in ARMS)
          + f"{C['V3 gradstep']:>8.0f}{C['V5 direct']:>8.0f}"
          f"{C['V5 qpe']:>8.0f}{time.time() - t0:>7.0f}", flush=True)
    del bk
    gc.collect()

print("  " + "-" * 100)
print(f"  rows within {NULL} of the row best (out of {len(PROBLEMS)}):")
for a, _ in ARMS:
    print(f"      {a:>14}  {score[a]}")
print()
print("  V5-direct at or near V3-gradstep on most rows, at no greater circuit")
print("  count, promotes it to the reference implementation. Behind on energy")
print("  means V3 stays the reference and V5's docstring must say so rather than")
print("  presenting itself as the successor.")
