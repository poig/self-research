"""Is there basin structure a finer grid would see that the +-R corners miss?

The bits_per_param > 1 proposal is: use a multi-level encoding to seed a
multi-basin search, then drop to 1 bit for local descent. More bits buys either
finer resolution at fixed range, or wider range at fixed resolution.

Before building a multi-level W-gate, ask the question classically - exactly as
the degree-2 Walsh weight was checked before building any CRZZ. Evaluate the
energy on a b=2 grid (4 levels per parameter) and ask whether the b=1 corners
already capture it. Two things decide it:

  RECONSTRUCTION  fit the multilinear model the b=1 corners determine, then
                  predict the b=2 grid points the corners never saw. If the
                  prediction is good, the fine grid holds no new information and
                  the extra bits buy nothing.
  BASIN COUNT     count distinct local minima on each grid. If the fine grid finds
                  minima the coarse one misses, seeding has something to seed.

THE SHOT WALL, which applies whatever this says: b bits over n params gives
2^(b n) vertices, so at fixed S the shots-per-vertex fall exponentially in b.
n=4,b=2 -> 256 vertices, 32 shots each at S=8192: workable. n=8,b=2 -> 65536
vertices, 0.125 shots each: dead. Any b>1 phase is confined to SMALL blocks,
which is not where the cost advantage lives.
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


def grid_energies(ansatz, H, c, act, levels):
    """<H> on the full tensor grid of `levels` offsets per active parameter."""
    pts = list(itertools.product(levels, repeat=len(act)))
    E = np.empty(len(pts))
    for v, off in enumerate(pts):
        p = c.copy(); p[act] = c[act] + np.array(off)
        E[v] = float(np.real(Statevector(ansatz.assign_parameters(p))
                             .expectation_value(H)))
    return np.array(pts), E


def multilinear_fit(pts_c, E_c, R):
    """Fit the multilinear model the +-R corners determine exactly.

    E(s) = sum_S coeff_S prod_{i in S} s_i  with s_i = offset_i / R in {-1,+1}.
    """
    n = pts_c.shape[1]
    S_list = [S for d in range(n + 1) for S in itertools.combinations(range(n), d)]
    A = np.empty((len(pts_c), len(S_list)))
    for r, off in enumerate(pts_c):
        s = off / R
        for cix, S in enumerate(S_list):
            A[r, cix] = np.prod([s[i] for i in S]) if S else 1.0
    coef, *_ = np.linalg.lstsq(A, E_c, rcond=None)
    return S_list, coef


def predict(S_list, coef, pts, R):
    out = np.empty(len(pts))
    for r, off in enumerate(pts):
        s = off / R
        out[r] = sum(cf * (np.prod([s[i] for i in S]) if S else 1.0)
                     for S, cf in zip(S_list, coef))
    return out


def count_minima(E, n, L):
    """Local minima on an L^n tensor grid indexed in itertools.product order.

    Works on multi-indices rather than float coordinates - matching grid points
    by rounded value is fragile and was the source of a lookup failure.
    """
    Eg = E.reshape((L,) * n)
    cnt = 0
    for midx in itertools.product(range(L), repeat=n):
        v = Eg[midx]
        ok = True
        for d in range(n):
            for step in (-1, 1):
                j = midx[d] + step
                if 0 <= j < L:
                    nb = list(midx); nb[d] = j
                    if Eg[tuple(nb)] < v:
                        ok = False; break
            if not ok:
                break
        if ok:
            cnt += 1
    return cnt


print("=" * 84)
print("Does a b=2 grid hold structure the b=1 corners miss?")
print("=" * 84)
for pname, fn in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                  ("H2", B.get_h2_problem)):
    ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=8192)
    BLK = [b['params'] for b in q.layers]
    print(f"\n  --- {pname} ---")
    print(f"  {'R':>5}{'blk':>5}{'corners':>9}{'fine grid':>11}"
          f"{'pred err %':>12}{'min b=1':>9}{'min b=2':>9}")
    print("  " + "-" * 60)
    for R in (0.6, 1.2):
        coarse = [-R, R]
        fine = [-R, -R / 3.0, R / 3.0, R]        # 4 levels, same RANGE
        for bi, act in enumerate(BLK[:2]):
            c = np.random.RandomState(3).uniform(-np.pi, np.pi,
                                                 ansatz.num_parameters)
            pc, Ec = grid_energies(ansatz, H, c, act, coarse)
            pf, Ef = grid_energies(ansatz, H, c, act, fine)
            S_list, coef = multilinear_fit(pc, Ec, R)
            pred = predict(S_list, coef, pf, R)
            rel = 100.0 * np.linalg.norm(pred - Ef) / (np.linalg.norm(
                Ef - Ef.mean()) + 1e-12)
            nact = len(act)
            print(f"  {R:>5.1f}{bi:>5}{len(Ec):>9}{len(Ef):>11}"
                  f"{rel:>12.2f}{count_minima(Ec, nact, 2):>9}"
                  f"{count_minima(Ef, nact, 4):>9}", flush=True)
print()
print("  pred err % = how badly the model fitted to the CORNERS predicts the")
print("  points it never saw, relative to the fine grid's own spread.")
print("  Small => the corners already determine the landscape and extra bits")
print("  add no information. Large, or more minima at b=2, => seeding has")
print("  something to find.")
