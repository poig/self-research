"""How reproducible is a v20-class row at all? The control the drift hunt needed.

THE HUNT, AND WHERE IT LANDED. v20's logged row (2026-08-03 02:30, Heisenberg
N=4, 256 shots, seeds 42-47) read walk -5.7904 / boltz -5.5951 / gradstep -5.8096.
Re-running v20's own code today gave -5.0514 / -5.4167 / -5.5801, a +0.739 move in
the walk arm that looked like a code change. Ruled out in order: merged_walk
(v53c, all under 2 sigma with the sign flipping), tau0 (one commit ever touched
that line, the original), sort_terms (v27, identical unitary to 0.000e+00),
num_ancillas (v20 pins it). Then v62 bisected nisq_v3 across its whole history:

    9e8290c 07-30    walk -5.4988   boltz -5.7166   gradstep -5.5401
    9f4d3a1 08-03    walk -5.5470   boltz -5.7575   gradstep -5.5218
    4dc62a4 08-04    walk -5.3851   boltz -4.9609   gradstep -5.0752
    108c945 08-05    walk -5.4305   boltz -5.2181   gradstep -5.5448
    WORKING today    walk -5.3177   boltz -4.8555   gradstep -5.3684

NO VERSION REPRODUCES -5.7904, including the two that PREDATE v20's log. So the
premise was wrong: there is no commit to find.

THE TELL IS IN THE COLUMN THAT CANNOT HAVE MOVED. boltz swings -5.72 -> -4.96 ->
-5.22 -> -4.86 across those versions, and boltz_step never calls nisq_v3's walk -
it decodes v20's own sensing table classically. A 0.9 swing in an arm with no
exposure to the code being bisected is not a code effect. And today's WORKING row
differs from v53b's run of THE SAME CODE, also today, by up to 0.56.

SO THE HYPOTHESIS IS THAT THE PROTOCOL ITSELF SCATTERS AT THIS SCALE, and every
"drift" read off these logs has been noise read as signal. v20 does not pass
sim_seed, so Aer draws its own shot entropy per run; only p0 is pinned, by
RandomState(seed). Six seeds at 256 shots then average six noisy optimisation
TRAJECTORIES, not six noisy measurements - shot noise steers each descent into a
different basin, so the spread compounds over 20 epochs rather than averaging
down.

THIS RUN IS THE CONTROL: the working tree, unchanged, K times over, same six
seeds, nothing varying but Aer's entropy. It reports the per-arm mean and SD
across repeats. If SD is ~0.3-0.5 then the +0.739 is ~1.5 SD, the bisect's
version-to-version spread is explained, and the finding is not about any commit -
it is that a v20-class row is a sample, not a measurement, and single numbers
from supplement/results/ must not be quoted as point estimates. If SD is ~0.05
then the scatter is real and something genuinely non-deterministic is loose.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
import nisq_v3

SRC = os.path.join(APP, 'supplement', 'v20_walk_shot_tolerance.py')
src = open(SRC).read()
cut = src.index('SIZES = (4, 6, 8)')
V20 = {'__name__': 'v66_control', '__file__': SRC}
exec(compile(src[:cut], SRC, 'exec'), V20)

SEEDS = (42, 43, 44, 45, 46, 47)
ARMS = V20['ARMS']
REPEATS = 6

print("=" * 92)
print("PROTOCOL SCATTER — the same code, run over and over")
print("=" * 92)
print("  Heisenberg N=4, 256 shots, seeds 42-47, 20 epochs, working tree.")
print("  Only Aer's shot entropy varies between repeats; p0 is pinned per seed.")
print(f"  v20 LOGGED:  walk -5.7904  boltz -5.5951  gradstep -5.8096")
print()
print(f"  {'repeat':>8}" + "".join(f"{a:>12}" for a in ARMS))
print("  " + "-" * 44)

allr = {a: [] for a in ARMS}
for rep in range(REPEATS):
    res = {a: [] for a in ARMS}
    for sd in SEEDS:
        ansatz, H = V20['heis'](4)
        with contextlib.redirect_stdout(io.StringIO()):
            q = nisq_v3.QLTOv3(ansatz, H, shot_budget=256, num_ancillas=4)
        BLK = [b['params'] for b in q.layers if b['params']]
        p0 = np.random.RandomState(sd).uniform(-np.pi, np.pi,
                                               ansatz.num_parameters)
        for arm in ARMS:
            p = p0.copy()
            for ep in range(20):
                R = max(0.6 * (0.9 ** ep), 1e-4)
                dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
                for act in BLK:
                    grad, table = V20['sense_table'](q, p, R, act)
                    if arm == 'walk':
                        p = q._execute_walk(p, 15, dt, R, act, grad)
                    elif arm.startswith('boltz'):
                        p = V20['boltz_step'](p, act, R, table,
                                              frac=float(arm[5:]))
                    else:
                        p = V20['grad_step'](p, act, R, grad)
            res[arm].append(V20['energy_at'](ansatz, H, p))
    for a in ARMS:
        allr[a].append(float(np.mean(res[a])))
    print(f"  {rep:>8}" + "".join(f"{allr[a][-1]:>12.4f}" for a in ARMS),
          flush=True)

print("  " + "-" * 44)
print(f"  {'mean':>8}" + "".join(f"{np.mean(allr[a]):>12.4f}" for a in ARMS))
print(f"  {'SD':>8}" + "".join(f"{np.std(allr[a], ddof=1):>12.4f}" for a in ARMS))
print(f"  {'min':>8}" + "".join(f"{np.min(allr[a]):>12.4f}" for a in ARMS))
print(f"  {'max':>8}" + "".join(f"{np.max(allr[a]):>12.4f}" for a in ARMS))

print()
logged = {'walk': -5.7904, 'boltz0.1': -5.5951, 'gradstep': -5.8096}
print("  v20's logged row expressed in units of this SD:")
for a in ARMS:
    m, s = float(np.mean(allr[a])), float(np.std(allr[a], ddof=1))
    print(f"      {a:>10}  logged {logged[a]:+.4f}   mean {m:+.4f}"
          f"   z = {(logged[a] - m) / s:+.2f}")
print()
print("  A z of ~1-2 on every arm means the logged row is an ordinary draw from")
print("  this distribution and there was never a drift to explain. It also means")
print("  the reporting convention has to change: six seeds of a 20-epoch")
print("  stochastic descent at 256 shots is a SAMPLE, and rows like this need")
print("  repeats and an SD, not a single number quoted to four decimals.")
