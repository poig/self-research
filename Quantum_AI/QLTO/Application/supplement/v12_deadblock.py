"""Are dead blocks the source of V3's MaxCut variance - and does skipping them fix it?

The DIAGONAL-HAMILTONIAN RULE says a final RZ block commutes with a diagonal H, so
its gradient is identically zero. The docs treat that as wasted effort. It may be
worse than wasted.

When grad_local = 0, _execute_walk sets every CRZ angle to zero and only the CRX
mixer runs. The param register is then mixed toward uniform and _decode_walk
returns a SHOT-NOISE-LIMITED estimate of the hypercube centre - so those
parameters take a small random walk every epoch. Those same parameters sit inside
the circuit that the LIVE blocks are optimising against, so the live blocks see a
landscape that jitters underneath them. That is a variance source, not just a
cost.

PART 1 confirms which blocks are dead, exactly, on MaxCut vs Heisenberg.
PART 2 A/Bs skipping them: same seeds, both arms, paired.

If skipping cuts the variance, the fix is three lines in run_walk and it also
saves 2 circuits per dead block per epoch - 25% of V3's circuits on MaxCut.
"""
import sys, os, contextlib, io, time
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


def exact_block_grad(ansatz, H, c, act):
    g = np.zeros(len(act))
    for j, i in enumerate(act):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[j] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp))
                                    .expectation_value(H)))
                      - float(np.real(Statevector(ansatz.assign_parameters(pm))
                                      .expectation_value(H))))
    return float(np.linalg.norm(g))


print("=" * 78)
print("PART 1. Which blocks are dead? Exact gradient norm per block.")
print("=" * 78)
PROBS = [("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
         ("MaxCut N=6", lambda: B.get_maxcut_problem(6)),
         ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))]
dead_map = {}
for pname, fn in PROBS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=8192)
    BLK = [b['params'] for b in q.layers]
    AX = [b['axis'] for b in q.layers]
    norms = []
    for bi, act in enumerate(BLK):
        v = np.mean([exact_block_grad(ansatz, H,
                                      np.random.RandomState(s).uniform(
                                          -np.pi, np.pi, ansatz.num_parameters),
                                      act) for s in (3, 11, 17)])
        norms.append(v)
    dead_map[pname] = [i for i, v in enumerate(norms) if v < 1e-9]
    print(f"  {pname:<16} axes {AX}")
    print(f"  {'':<16} |g| " + "  ".join(f"blk{i}={v:.3e}"
                                         for i, v in enumerate(norms)))
    print(f"  {'':<16} dead blocks: {dead_map[pname]}")
    print()

print("=" * 78)
print("PART 2. Does skipping dead blocks reduce variance? Paired A/B.")
print("=" * 78)


def run(prob, skip_dead, seed, epochs=20, k=15, shots=8192):
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    BLK = [b['params'] for b in q.layers]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    live = list(range(len(BLK)))
    if skip_dead:
        live = [i for i in range(len(BLK))
                if exact_block_grad(ansatz, H, p, BLK[i]) > 1e-9]
    ncirc = 0
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for bi in live:
            act = BLK[bi]
            g = q.sense_gradient(p, R, act)
            p = q._execute_walk(p, k, dt, R, act, g)
            ncirc += 2
    E = float(np.real(Statevector(ansatz.assign_parameters(p))
                      .expectation_value(H)))
    return E, ncirc


SEEDS = tuple(range(30, 38))
for pname, prob in (("MaxCut N=6", lambda: B.get_maxcut_problem(6)),
                    ("MaxCut N=4", lambda: B.get_maxcut_problem(4))):
    print(f"\n  --- {pname}, {len(SEEDS)} seeds, paired ---")
    t0 = time.time(); a_all, b_all, ca, cb = [], [], 0, 0
    for s in SEEDS:
        e0, c0 = run(prob, False, s)
        e1, c1 = run(prob, True, s)
        a_all.append(e0); b_all.append(e1); ca, cb = c0, c1
    a_all = np.array(a_all); b_all = np.array(b_all)
    d = b_all - a_all
    print(f"  {'all blocks':<14} mean {a_all.mean():>9.4f}  std {a_all.std(ddof=1):>8.4f}"
          f"  circuits {ca}")
    print(f"  {'skip dead':<14} mean {b_all.mean():>9.4f}  std {b_all.std(ddof=1):>8.4f}"
          f"  circuits {cb}")
    print(f"  paired diff (skip - all): {d.mean():+.4f} +- "
          f"{d.std(ddof=1)/np.sqrt(len(d)):.4f}")
    print(f"  variance ratio (skip/all): "
          f"{(b_all.std(ddof=1)/max(a_all.std(ddof=1),1e-12))**2:.3f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)
print()
print("  variance ratio < 1 => dead blocks were injecting noise, and skipping")
print("  them is a free fix that also cuts circuits.")
