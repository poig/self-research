"""Path A's theorem, checked numerically against what v5 actually does.

research_roadmap_final.md states Path A as a theorem to be written, with three
quantitative claims. The proof is separate work, but the claims are measurable and
worth checking BEFORE anyone writes them down:

    (1) "G * ceil(M/n*) ~ 1.5G circuits per epoch, constant in M"
    (2) "O(M/(R^2 eps^2)) total shots"
    (3) "circuit depth bounded by the ansatz depth + O(1)"

CLAIM 1 HAS A SUSPECTED GAP, and it is the reason for this file. The "~1.5G"
comes from T10's COST-OPTIMAL block width n* ~ 0.65M, which would give
ceil(M/n*) ~ 1.5 blocks. But v5's _layers() does not block by n* - it partitions
by DISJOINT QUBIT SUPPORT, i.e. by the ansatz's own layer structure. For
efficient_su2(N, reps=r) that is 2(r+1) rotation layers of N parameters each, so

    L = 2(r+1) = M/N,        not 1.5

and circuits per epoch is G*M/N rather than 1.5G. If so the theorem describes a
blocking strategy the implementation does not use, and either the statement or
_layers() has to change before it is written down.

CLAIM 3 SHOULD SPLIT BY MODE. Direct readout is H on param, W, one basis
rotation, measure - so ansatz depth + O(1) is plausible. QPE adds the
(2^k - 1)*tau0 ladder, which is not O(1) in anything.

CLAIM 2 is tested as a scaling: if eps ~ sqrt(M/(R^2 S)) then the L2 error of the
sensed gradient against the EXACT R-smoothed gradient - computed here by full
hypercube enumeration, which is the quantity the estimator actually targets -
should fall as 1/sqrt(S) and rise as sqrt(M).

Nothing here proves the theorem. It checks whether the numbers in it are the
numbers the code produces.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
import nisq_v5


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def E(a, H, p):
    return float(np.real(Statevector(a.assign_parameters(p)).expectation_value(H)))


def exact_smoothed_grad(a, H, centre, R, act):
    """The estimator's TARGET: degree-1 Walsh coefficients over the full
    hypercube, divided by R. Not the point gradient - the smoothed one."""
    n = len(act)
    sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    vals = np.empty(len(sig))
    for k, s in enumerate(sig):
        p = np.asarray(centre, float).copy()
        p[act] = p[act] + R * s
        vals[k] = E(a, H, p)
    return np.array([float(np.mean(vals * sig[:, i])) for i in range(n)]) / R


R = 0.6
BACK = AerSimulator()

print("=" * 96)
print("PATH A — the theorem's three claims, measured")
print("=" * 96)

# ── claims 1 and 3 ───────────────────────────────────────────────────────────
print("\n  (1) CIRCUITS PER EPOCH, and (3) DEPTH. Claim: ~1.5G circuits,")
print("      constant in M; depth = ansatz depth + O(1).")
print()
print(f"  {'N':>3}{'reps':>5}{'M':>5}{'G':>3}{'L':>4}{'n/blk':>7}"
      f"{'circ/ep qpe':>13}{'circ/ep dir':>13}{'1.5G':>7}"
      f"{'anz depth':>11}{'d qpe':>8}{'d dir':>8}")
print("  " + "-" * 88)

for N, reps in ((4, 1), (4, 2), (4, 3), (6, 1), (6, 2), (8, 1)):
    H = heis(N)
    a = efficient_su2(N, reps=reps)
    M = a.num_parameters
    G = len(list(H.group_commuting(qubit_wise=True)))
    anz_d = transpile(a.assign_parameters(np.zeros(M)), BACK,
                      optimization_level=1).depth()
    row = {}
    for mode in ('qpe', 'direct'):
        with contextlib.redirect_stdout(io.StringIO()):
            q = nisq_v5.QLTOv5(a, H, shot_budget=256, gradient_mode=mode,
                               sim_seed=3)
        L = len([b for b in q.layers if b['params']])
        nblk = len(q.layers[0]['params'])
        p = np.random.RandomState(1).uniform(-np.pi, np.pi, M)
        with contextlib.redirect_stdout(io.StringIO()):
            q.run_epoch(p, R)
        row[mode] = (q.nefv, q.max_circuit_depth, L, nblk)
    L, nblk = row['qpe'][2], row['qpe'][3]
    print(f"  {N:>3}{reps:>5}{M:>5}{G:>3}{L:>4}{nblk:>7}"
          f"{row['qpe'][0]:>13}{row['direct'][0]:>13}{1.5 * G:>7.1f}"
          f"{anz_d:>11}{row['qpe'][1]:>8}{row['direct'][1]:>8}", flush=True)

print()
print("  L is the number of blocks _layers() produces. If L tracks M/N rather")
print("  than sitting near 1.5, claim (1)'s constant describes T10's optimal")
print("  blocking, which v5 does not implement.")

# ── claim 2 ──────────────────────────────────────────────────────────────────
print("\n  (2) SHOT SCALING. Claim eps ~ sqrt(M/(R^2 S)), so L2 error of the")
print("      sensed gradient against the EXACT R-smoothed target should fall")
print("      as 1/sqrt(S). Direct mode, block 0, 4 repeats.")
print()
print(f"  {'N':>3}{'reps':>5}{'M':>5}{'n':>4}{'shots':>8}{'L2 err':>10}"
      f"{'err*sqrt(S)':>13}{'rel err':>9}")
print("  " + "-" * 55)

for N, reps in ((4, 1), (4, 2), (6, 1)):
    H = heis(N)
    a = efficient_su2(N, reps=reps)
    M = a.num_parameters
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v5.QLTOv5(a, H, shot_budget=256, gradient_mode='direct')
    act = probe.layers[0]['params']
    tgt = exact_smoothed_grad(a, H, centre, R, act)
    for S in (256, 1024, 4096, 16384):
        errs = []
        for r in range(4):
            with contextlib.redirect_stdout(io.StringIO()):
                q = nisq_v5.QLTOv5(a, H, shot_budget=S,
                                   gradient_mode='direct', sim_seed=100 + 17 * r)
            g, _ = q.sense(centre, R, act)
            errs.append(float(np.linalg.norm(g[act] - tgt)))
        e = float(np.mean(errs))
        print(f"  {N:>3}{reps:>5}{M:>5}{len(act):>4}{S:>8}{e:>10.5f}"
              f"{e * np.sqrt(S):>13.3f}{e / max(np.linalg.norm(tgt), 1e-12):>9.4f}",
              flush=True)
    print("  " + "." * 55)

print()
print("  err*sqrt(S) flat down a column confirms the 1/sqrt(S) half of claim (2).")
print("  Comparing that constant ACROSS blocks of different width tests the")
print("  sqrt(M) half - it should grow with n, since T4 gives")
print("  Var(g_i) = (1/S)[a + b(n-1)].")
