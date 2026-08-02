"""Why is the hypercube landscape exactly quadratic? Locality, and it is a theorem.

T6 measured degree1+degree2 = 99.6%+ with degree3 ~ 0.004. Here is the reason,
and it makes a sharper prediction than "small".

CLAIM. A block is single-qubit rotations on DISTINCT qubits. A k-local Hamiltonian
term is supported on k qubits, so its expectation depends on at most k of the
block's sigma variables - and any function of k binary variables has Walsh degree
<= k. Summing over terms: the energy on the +-R hypercube has Walsh degree <= k for
k-local H. Heisenberg is 2-local, hence degree <= 2 EXACTLY.

WHICH BLOCKS, and this is the falsifiable part. A block only sees a strictly
2-local observable if nothing entangling follows it. efficient_su2(reps=1)
decomposes to RY, RZ, CX, RY, RZ, giving blocks [Y, Z, Y, Z]:

  blk 3  final RZ layer      - nothing after it            -> exactly degree <= 2
  blk 2  RY before final RZ  - only single-qubit gates after, which preserve
                               support                      -> exactly degree <= 2
  blk 1  RZ before the CX    - H conjugated by CX spreads 2-local to 4-local
                                                            -> degree 3+ allowed
  blk 0  first RY layer      - same                         -> degree 3+ allowed

So the prediction is not "degree-3 is small" but "degree-3 is ZERO to machine
precision on blocks 2 and 3, and nonzero on 0 and 1". If that holds, T6 is a
consequence of locality rather than a numerical accident, and the amount of
landscape the degree-1 estimator misses is knowable A PRIORI from H's locality.
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


def spectrum_by_degree(ansatz, H, c, R, act):
    n = len(act)
    sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(n)]
                    for v in range(2 ** n)])
    E = np.empty(len(sig))
    for v, s in enumerate(sig):
        p = c.copy(); p[act] = c[act] + R * s
        E[v] = float(np.real(Statevector(ansatz.assign_parameters(p))
                             .expectation_value(H)))
    per = np.zeros(n + 1)
    for d in range(1, n + 1):
        for S in itertools.combinations(range(n), d):
            chi = np.ones(len(E))
            for i in S:
                chi = chi * sig[:, i]
            per[d] += float(np.mean(E * chi)) ** 2
    return per, float(np.sqrt(per.sum()))


print("=" * 84)
print("Walsh weight by degree, PER BLOCK (absolute, not normalised)")
print("=" * 84)
print("  prediction: deg3+ vanishes to machine precision on the blocks with no")
print("  entangler after them (blk 2, blk 3), and is nonzero on blk 0, blk 1")
print()
for pname, fn in (("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
                  ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))):
    ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=8192)
    BLK = [b['params'] for b in q.layers]
    AX = [b['axis'] for b in q.layers]
    print(f"  --- {pname} | blocks {[len(b) for b in BLK]} axes {AX} ---")
    hdr = "  " + f"{'blk':<5}{'axis':<6}" + "".join(
        f"{'deg%d' % d:>12}" for d in range(1, 5))
    print(hdr); print("  " + "-" * (11 + 12 * 4))
    for seed in (3, 11):
        for bi, act in enumerate(BLK):
            c = np.random.RandomState(seed).uniform(-np.pi, np.pi,
                                                    ansatz.num_parameters)
            per, _ = spectrum_by_degree(ansatz, H, c, 0.6, act)
            row = f"  {bi:<5}{AX[bi]:<6}"
            for d in range(1, 5):
                v = per[d] if d < len(per) else 0.0
                row += f"{v:>12.3e}"
            print(row, flush=True)
        print("  " + "." * 40)
    print()
print("  deg3 at ~1e-30 or below is exact zero in double precision;")
print("  deg3 at 1e-3 or above is real structure the CX spread created.")
