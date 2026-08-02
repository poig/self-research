"""Does a multi-bit grid help GLOBAL search at wide radius?

v7_bitsperparam asked whether a finer grid reveals basins the +-R corners miss,
at R=0.6 and 1.2 - both LOCAL radii. That was the wrong question for the
coarse-to-fine proposal, which is about RANGE: use extra bits to search wide,
identify the promising zone, then narrow with a single bit.

So ask it properly. At radius R the b=1 corners sit at c+-R while a b-bit grid
puts 2^b levels across the SAME span. Sweep R out to pi/2, which is the widest
useful radius - at R=pi the corners c+pi and c-pi coincide for a 2pi-periodic
parameter, so the encoding degenerates.

The decisive metric is not how many minima each grid contains but whether the
COARSE grid can reach what the FINE grid finds:

    best energy on the grid, vs b     if b=1 already attains the fine grid's best,
                                      the extra bits find nothing new
    minima count, vs b                structure the coarse grid cannot represent

Purely classical - no circuits needed to answer whether the structure is there.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def levels_for(b, R):
    """2^b evenly spaced offsets spanning [-R, +R] inclusive."""
    L = 2 ** b
    if L == 2:
        return [-R, R]
    return list(np.linspace(-R, R, L))


def grid_scan(ansatz, H, c, act, lv):
    pts = list(itertools.product(lv, repeat=len(act)))
    E = np.empty(len(pts))
    for v, off in enumerate(pts):
        p = c.copy(); p[act] = c[act] + np.array(off)
        E[v] = float(np.real(Statevector(ansatz.assign_parameters(p))
                             .expectation_value(H)))
    return np.array(pts), E


def count_minima(E, n, L):
    Eg = E.reshape((L,) * n)
    cnt = 0
    for midx in itertools.product(range(L), repeat=n):
        v = Eg[midx]; ok = True
        for d in range(n):
            for step in (-1, 1):
                j = midx[d] + step
                if 0 <= j < L and Eg[tuple(list(midx[:d]) + [j]
                                           + list(midx[d + 1:]))] < v:
                    ok = False; break
            if not ok:
                break
        if ok:
            cnt += 1
    return cnt


ansatz, H, _ = B.get_heisenberg_problem(4)
q = Q(ansatz, H, shot_budget=8192)
act = q.layers[0]['params']
n = len(act)
print("=" * 84)
print("Multi-bit grid for GLOBAL search: does the coarse grid miss the wide-R best?")
print("=" * 84)
print(f"  Heisenberg N=4, block of {n} params, 3 centres")
print()
print(f"  {'R':>7}{'b':>4}{'levels':>8}{'points':>9}{'best E':>10}"
      f"{'gap to b=3':>12}{'minima':>8}")
print("  " + "-" * 58)
for R in (0.6, 1.2, np.pi / 2):
    ref = {}
    for b in (1, 2, 3):
        lv = levels_for(b, R)
        L = len(lv)
        bests, minc = [], []
        for seed in (3, 11, 17):
            c = np.random.RandomState(seed).uniform(-np.pi, np.pi,
                                                    ansatz.num_parameters)
            pts, E = grid_scan(ansatz, H, c, act, lv)
            bests.append(E.min()); minc.append(count_minima(E, n, L))
        ref[b] = float(np.mean(bests))
        gap = ref[b] - ref.get(3, ref[b])
        print(f"  {R:>7.3f}{b:>4}{L:>8}{L**n:>9}{ref[b]:>10.4f}"
              f"{'':>12}{np.mean(minc):>8.1f}", flush=True)
    print(f"  {'':>7}{'':>4}{'':>8}{'':>9}{'':>10}"
          f"{'b=1 worse than b=3 by ' + format(ref[1]-ref[3], '.4f'):>34}")
    print()
print("  If the b=1 gap is ~0, the two-level corners already reach whatever the")
print("  finer grid reaches, and extra bits buy no global-search power - the")
print("  coarse-to-fine cascade then reduces to the R decay schedule that")
print("  _execute_walk already runs.")
