"""Paired A/B: does resolution V survive more than one seed?

v90 measured res-5 beating res-4 on Heisenberg N=4 at identical circuit count,
-5.5921 -> -5.8329. That is ONE seed at 10 epochs, and a single draw is not a
result - the QN-SPSA sweep showed 2-trial noise on this harness is about 0.07,
comparable to that gap.

Rerunning the whole 8-problem 5-trial suite to find out costs hours and would
make every committed number stale. So this is the cheap discriminator first:
same problems, same seeds, same r0, same epoch count, only the design changes.
PAIRED - each seed is run under both resolutions and the DIFFERENCE is what is
reported, which removes the seed-to-seed variance that swamped the single-draw
comparison.

r0 IS NOT RE-TUNED HERE, and that biases against res-5: 0.6 was selected against
the resolution-4 design. If res-5 wins anyway the case is stronger than the
number shows; if it loses, a re-tune is the next thing to try rather than the
conclusion.

The suite run is justified only if the paired mean difference is clearly
positive and larger than its own standard error.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

import benchmark as B

PROBS = [B.get_heisenberg_problem(4), B.get_heisenberg_problem(6),
         B.get_maxcut_problem(4), B.get_h2_problem()]
SEEDS, EPOCHS = 5, 20


def run(ansatz, H, seed, res):
    M = ansatz.num_parameters
    np.random.seed(seed)
    p = np.random.uniform(-np.pi, np.pi, M)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = B.QLTOv6_Wrapper(ansatz, H, r0=B.TUNED['QLTO V6'],
                               design_resolution=res)
        B._mult(opt, 1)
        es = []
        for _ in range(EPOCHS):
            p = opt.step(p)
            es.append(B.report_energy(ansatz, H, p))
    return min(es), B.optimizer_circuits(opt), opt.max_circuit_depth


print("=" * 96)
print("PAIRED A/B:  design_resolution 4 vs 5,  same seeds, same r0=0.6")
print("=" * 96)
print(f"  {SEEDS} seeds x {EPOCHS} epochs. Negative diff = res-5 better"
      f" (lower energy).")
print()
print(f"  {'problem':<22}{'res4 mean':>12}{'res5 mean':>12}"
      f"{'paired diff':>13}{'sem':>9}{'nefv':>7}{'depth 4/5':>12}")
print("  " + "-" * 87)

for ansatz, H, name in PROBS:
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    d4, d5, n4, n5, dep4, dep5 = [], [], 0, 0, 0, 0
    for s in range(SEEDS):
        e4, n4, dep4 = run(ansatz, H, 42 + s, 4)
        e5, n5, dep5 = run(ansatz, H, 42 + s, 5)
        d4.append(e4)
        d5.append(e5)
    d4, d5 = np.array(d4), np.array(d5)
    diff = d5 - d4
    sem = float(np.std(diff) / np.sqrt(max(len(diff), 1)))
    print(f"  {name:<22}{d4.mean():>12.4f}{d5.mean():>12.4f}"
          f"{diff.mean():>13.4f}{sem:>9.4f}{n5:>7}"
          f"{str(dep4) + '/' + str(dep5):>12}", flush=True)

print()
print("=" * 96)
print("READING IT")
print("=" * 96)
print("  A paired difference more negative than ~2 SEM is a real improvement")
print("  and justifies re-tuning r0 for res-5 and rerunning the full suite.")
print("  Anything inside the noise means v90's single-seed -5.5921 -> -5.8329")
print("  was a draw of the dice, the aliasing fix helps the Hamiltonian-")
print("  learning task it was built for and NOT VQE, and the suite stands.")
print("  The depth column is the cost either way: res-5 spends more CNOTs on")
print("  longer parity chains, and that is charged to nobody in the NEFV table.")
