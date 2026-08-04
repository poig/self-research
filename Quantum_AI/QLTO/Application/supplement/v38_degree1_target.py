"""Does the product mixer target the WRONG CORNER? Enumerated, not argued.

Today's closed form (v37b/c, validated to 0.00241) showed the walk's k steps
compose as a product of rotations with the angle ADDING - the same mechanism
Grover uses. So why is there no Grover-like concentration as k grows?

The hypothesis is structural, not statistical. Grover amplifies ONE GLOBAL
amplitude in a 2D invariant plane. The walk's mixer is a product, so its
parameter-register DLA is su(2)^(+)n and it amplifies n INDEPENDENT local
amplitudes. A product of per-coordinate updates can only converge to the product
of per-coordinate optima, which is the argmin of the DEGREE-1 TRUNCATION of the
Walsh expansion:

    E(x) = E_hat(0) + sum_i E_hat({i}) x_i + sum_{i<j} E_hat({i,j}) x_i x_j + ...
    product-mixer target:  x*_i = -sign(E_hat({i}))       [degree-1 argmin]
    true target:           argmin over the whole hypercube

These agree only when the degree>=2 part never flips a sign. T6 measured
deg1+deg2 = 99.6% of the landscape with deg2 EXCEEDING deg1 on 2 of 4 blocks, so
on those blocks the degree-2 part is the larger term and there is no reason to
expect the signs to survive it.

This file enumerates the full 2^n hypercube - no optimiser, no shots, no walk -
and compares three corners per block:

    x_true    argmin of the exact energy over all 2^n vertices
    x_deg1    -sign of the degree-1 Walsh coefficients  (what a product mixer can reach)
    x_deg2    argmin of the degree-<=2 truncation       (what a 2-body mixer could reach)

REPORTED
  hamming     distance from each surrogate corner to the true one
  E gap       energy left on the table by stopping at that corner
  regret      that gap as a fraction of the full hypercube energy range

If x_deg1 is far from x_true on exactly the blocks where deg2 > deg1, the
proposition holds and the shipped walk is aiming at the wrong corner BY
CONSTRUCTION - a reachability defect, which is not what Cerezo & Coles forbids.
If x_deg2 closes most of the gap, the body-order rule (mixer order = Walsh
degree) is quantified rather than asserted, and T7's failure is explained: it
raised the DRIFT to degree 2 while leaving the MIXER at degree 1, so the extra
information had nowhere to go.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def maxcut(N):
    ops = []
    for i in range(N - 1):
        s = ["I"] * N
        s[i] = s[i + 1] = "Z"
        ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def h2():
    return SparsePauliOp.from_list([("II", -1.0523), ("IZ", 0.3979),
                                    ("ZI", -0.3979), ("ZZ", -0.0113),
                                    ("XX", 0.1809)])


def E(ansatz, H, p):
    return float(np.real(Statevector(ansatz.assign_parameters(p)).expectation_value(H)))


def walsh(vals, sig, n):
    """Exact Walsh coefficients up to degree 2 from the full enumeration."""
    cols = [np.ones(len(sig))]
    keys = [()]
    for i in range(n):
        cols.append(sig[:, i]); keys.append((i,))
    for i in range(n):
        for j in range(i + 1, n):
            cols.append(sig[:, i] * sig[:, j]); keys.append((i, j))
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return dict(zip(keys, coef)), A, coef


R = 0.6
PROBLEMS = [("H2", h2(), 1), ("MaxCut N=4", maxcut(4), 1),
            ("Heisenberg N=4", heis(4), 1), ("Heisenberg N=6", heis(6), 1)]

print("=" * 100)
print("IS THE PRODUCT MIXER AIMING AT THE WRONG CORNER? Full hypercube enumeration.")
print("=" * 100)
print(f"  R={R}. No optimiser, no shots. x_deg1 is the best a product mixer can")
print(f"  reach; x_deg2 is the best a 2-body mixer could reach; x_true is exact.")
print()
print(f"  {'problem':>15}{'blk':>5}{'n':>3}{'deg1':>8}{'deg2':>8}"
      f"{'ham1':>6}{'ham2':>6}{'regret1':>9}{'regret2':>9}{'wrong?':>8}")
print("  " + "-" * 82)

n_wrong = n_tot = 0
for name, H, reps in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=reps)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v3.QLTOv3(ansatz, H, shot_budget=1024)
    BLK = [b['params'] for b in q.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        if n > 12:
            continue
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for k, s in enumerate(sig):
            p = centre.copy()
            p[act] = p[act] + R * s
            vals[k] = E(ansatz, H, p)

        coef, A, c = walsh(vals, sig, n)
        d1 = np.array([coef[(i,)] for i in range(n)])
        p1 = float(np.sum(d1 ** 2))
        p2 = float(np.sum([coef[(i, j)] ** 2
                           for i in range(n) for j in range(i + 1, n)]))

        i_true = int(np.argmin(vals))
        x_true = sig[i_true]
        x_d1 = np.where(d1 <= 0, 1.0, -1.0)      # x_i = -sign(Ehat_i)
        # degree-<=2 truncation, minimised by enumeration (exact at these n)
        trunc = A @ c
        x_d2 = sig[int(np.argmin(trunc))]

        def energy_of(x):
            return float(vals[int(np.argmin(np.sum(np.abs(sig - x), axis=1)))])

        rng_e = vals.max() - vals.min()
        e1, e2, et = energy_of(x_d1), energy_of(x_d2), vals[i_true]
        h1 = int(np.sum(x_d1 != x_true))
        h2_ = int(np.sum(x_d2 != x_true))
        r1 = (e1 - et) / rng_e if rng_e > 1e-12 else 0.0
        r2 = (e2 - et) / rng_e if rng_e > 1e-12 else 0.0
        wrong = "YES" if h1 > 0 else "no"
        n_wrong += (h1 > 0); n_tot += 1
        print(f"  {name:>15}{bi:>5}{n:>3}{p1:>8.4f}{p2:>8.4f}"
              f"{h1:>6}{h2_:>6}{r1:>9.3f}{r2:>9.3f}{wrong:>8}", flush=True)

print("  " + "-" * 82)
print(f"  product-mixer target differs from the true corner on "
      f"{n_wrong}/{n_tot} blocks")
print()
print("  'deg1'/'deg2' are the squared Walsh weights at each degree. 'ham' is the")
print("  Hamming distance from the surrogate corner to the true one; 'regret' is")
print("  the energy left on the table as a fraction of the hypercube's range.")
print()
print("  A product mixer cannot represent a target that is not the sign pattern of")
print("  the degree-1 coefficients, so ham1 > 0 is a REACHABILITY failure, not a")
print("  signal-to-noise one - and therefore outside what Cerezo & Coles forbids.")
print("  regret2 << regret1 would price the 2-body mixer exactly.")
