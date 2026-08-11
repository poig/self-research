"""Where is the cost-optimal block width, measured at matched TOTAL shots?

T10 predicts n* ~ 0.65 M from a purely CLASSICAL attenuation fit: the smeared
gradient's magnitude decays as exp(-c R^2 n), so cost/||g||^2 has an interior
minimum. _layers() does not implement it - it partitions by disjoint qubit
support, giving n = N and L = 2(r+1) (v65, exact on 12 rows) - so ~2.7x of
circuits has been sitting on the table unclaimed.

WHY THIS IS THE RIGHT TIME TO MEASURE IT. v69 established that at matched TOTAL
shots the picture is a bias-variance trade, not a plateau, and that QLTO's
advantage comes from spending its budget as FEWER CIRCUITS WITH MORE SHOTS EACH.
Block width is the same lever seen from the other side:

    wider blocks  ->  fewer circuits  ->  MORE shots each  ->  less variance
                  ->  more coordinates smeared at once  ->  MORE attenuation

So the optimum is a genuine interior trade at fixed budget, and T10's n* was
fitted WITHOUT the shot side - it minimised cost/||g||^2 using a noise model, not
a measured budget. This measures the whole thing directly.

PROTOCOL. Chunk the M parameters into consecutive blocks of width n, for n across
the range. Circuits per gradient = ceil(M/n) * G, so shots per circuit = T /
(ceil(M/n) * G). Score by cos(g_hat, grad E) against the exact gradient - the
common-target metric the CORRECTION section insisted on. R is swept and the best
taken at each width, so a width is not penalised for preferring a radius the grid
happened to centre elsewhere.

WHAT WOULD VINDICATE T10: an interior optimum near 0.65 M, clearly better than
both n = N (what ships) and n = M (block_mode='global'). WHAT WOULD RETIRE IT: a
monotone curve, meaning the shot side dominates the attenuation side and the
right answer is simply the widest block that fits - which would make
block_mode='global' the default and delete the whole n* question.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v5


def heis(N):
    o = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


RADII = (0.1, 0.2, 0.3, 0.45, 0.6, 0.9)
REPEATS = 5
T = 2 ** 17

print("=" * 100)
print("BLOCK WIDTH AT MATCHED TOTAL SHOTS — is T10's interior optimum real?")
print("=" * 100)
print(f"  Total budget T = {T} per gradient, split over ceil(M/n)*G circuits.")
print(f"  Wider block = fewer circuits = more shots each, but more attenuation.")
print(f"  Best R taken per width from {RADII}. {REPEATS} repeats.")
print()

for N in (6,):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()

    theta = np.random.RandomState(23).uniform(-np.pi, np.pi, M)
    g_ex = np.zeros(M)
    for i in range(M):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            v = Statevector(ansatz.assign_parameters(t)).data
            g_ex[i] += s * float(np.real(np.conj(v) @ (Hm @ v))) / 2

    with contextlib.redirect_stdout(io.StringIO()):
        qs = [nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct',
                             block_mode='global', sim_seed=300 + r)
              for r in range(REPEATS)]
    G = len(qs[0].groups)

    widths = sorted(set([1, 2, N, M // 4, M // 2, int(round(0.65 * M)),
                         3 * M // 4, M]))
    widths = [w for w in widths if 1 <= w <= M]

    print(f"  Heisenberg N={N}:  M={M}  G={G}   (shipped width n=N={N},"
          f"  T10 predicts n* ~ {0.65 * M:.0f})")
    print(f"  {'n':>5}{'n/M':>7}{'circuits':>10}{'S/circ':>9}{'R*':>7}"
          f"{'cos':>10}{'1-cos':>10}")
    print("  " + "-" * 58)

    best_overall = (-2.0, None)
    for n in widths:
        blocks = [list(range(i, min(i + n, M))) for i in range(0, M, n)]
        C = len(blocks) * G
        S = max(1, T // C)
        for q in qs:
            q.shot_budget = int(S)

        best = (-2.0, None)
        for R in RADII:
            cs = []
            for q in qs:
                gh = np.zeros(M)
                for act in blocks:
                    gi, _ = q.sense(theta, R, act)
                    gh += gi
                cs.append(cosine(gh, g_ex))
            m = float(np.mean(cs))
            if m > best[0]:
                best = (m, R)
        c, Rstar = best
        if c > best_overall[0]:
            best_overall = (c, n)
        print(f"  {n:>5}{n / M:>7.2f}{C:>10}{S:>9}{Rstar:>7.2f}"
              f"{c:>10.4f}{1 - c:>10.5f}", flush=True)

    bc, bn = best_overall
    print(f"  best width {bn}  (n/M = {bn / M:.2f}),  cos {bc:.4f}"
          f"   |  T10 predicted n/M ~ 0.65,  shipped n/M = {N / M:.2f}")
    print()

print("  An interior peak near n/M ~ 0.65 vindicates T10 and says the shipped")
print("  blocking leaves circuits on the table. A curve that keeps improving to")
print("  n = M says the shot side dominates, block_mode='global' should be the")
print("  default, and T10's optimum was an artefact of costing without a budget.")
