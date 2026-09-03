"""Is merged_walk harmful at LOW shots? Its validation was at 8192.

v53b showed v20's logged row does not reproduce, and the drift is concentrated in
the WALK arm:

    arm        v20 today   v20 logged   today-log   SEM
    walk         -5.0514      -5.7904     +0.7390   0.20   <- 3.7 SEM
    boltz0.1     -5.4167      -5.5951     +0.1784   ~0.09
    gradstep     -5.5801      -5.8096     +0.2295   ~0.09

merged_walk is the only changed default that touches the walk and nothing else,
and it is explicitly NOT an equivalent rewrite - nisq_v3 records "at the angles
actually used they differ by 0.813 in operator norm, so this is different
dynamics at lower depth". It was validated PAIRED at 12 seeds, -0.0032 +- 0.0101,
0.3 sigma, better on 7/12 - but at 8192 SHOTS. v20's row is at 256.

The BCH error of the merge scales as alpha*beta with
alpha = g gamma 0.5 pi / sqrt(R), and nisq_v3's own note records the measured max
alpha across the suite as 6.53 at Heisenberg N=4 - the largest in the suite. This
session then showed alpha is the per-step drift angle and that it WRAPS. A merge
whose error grows with alpha, validated only where the gradient is precise, is
exactly the thing that could degrade when the gradient gets noisy and alpha
scatters.

So: same protocol, same seeds, merged_walk toggled, across shot budgets. This is
a PAIRED test - both arms start each seed from identical parameters - because
this project has twice recorded sub-2-sigma results that reversed on replication.

If merged=True is worse at low shots and equal at high, the shipped default is
wrong outside the regime it was validated in, and that is a live defect in
nisq_v3 rather than a historical note.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

SRC = os.path.join(APP, 'supplement', 'v20_walk_shot_tolerance.py')
src = open(SRC).read()
cut = src.index('SIZES = (4, 6, 8)')
mod = {'__name__': 'v20_merged', '__file__': SRC}
exec(compile(src[:cut], SRC, 'exec'), mod)

import nisq_v3
_R = nisq_v3.QLTOv3


def run_walk(N, seed, shots, merged, epochs=20, k=15):
    ansatz, H = mod['heis'](N)
    with contextlib.redirect_stdout(io.StringIO()):
        q = _R(ansatz, H, shot_budget=shots, num_ancillas=4, merged_walk=merged)
    BLK = [b['params'] for b in q.layers if b['params']]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi,
                                            ansatz.num_parameters)
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            grad, _ = mod['sense_table'](q, p, R, act)
            p = q._execute_walk(p, k, dt, R, act, grad)
    return mod['energy_at'](ansatz, H, p)


SEEDS = (42, 43, 44, 45, 46, 47)
BUDGETS = (256, 1024, 8192)

print("=" * 88)
print("IS merged_walk HARMFUL AT LOW SHOTS? Its validation was at 8192.")
print("=" * 88)
print(f"  Heisenberg N=4, {len(SEEDS)} PAIRED seeds, 20 epochs, k=15, kappa=4.")
print(f"  Both arms start each seed from identical parameters.")
print(f"  merged_walk was validated at 8192 shots only: -0.0032 +- 0.0101, 0.3s.")
print()
print(f"  {'shots':>8}{'merged=True':>14}{'merged=False':>14}{'diff':>10}"
      f"{'sigma':>8}{'better':>9}")
print("  " + "-" * 63)

for S in BUDGETS:
    a, b = [], []
    for sd in SEEDS:
        a.append(run_walk(4, sd, S, True))
        b.append(run_walk(4, sd, S, False))
    a, b = np.array(a), np.array(b)
    d = a - b                       # positive => merged is WORSE (higher energy)
    sem = d.std() / np.sqrt(len(SEEDS))
    print(f"  {S:>8}{a.mean():>14.4f}{b.mean():>14.4f}{d.mean():>+10.4f}"
          f"{abs(d.mean()) / max(sem, 1e-9):>8.1f}"
          f"{f'{int((d < 0).sum())}/{len(SEEDS)}':>9}", flush=True)

print()
print("  POSITIVE diff means merged_walk reached HIGHER energy, i.e. is worse.")
print("  'better' counts seeds where merged won. A positive diff at 256 that")
print("  vanishes by 8192 means the shipped default is wrong outside the regime")
print("  it was validated in. Flat across budgets means merged_walk is not the")
print("  cause of v53b's +0.739 and the stale-log question is still open.")
print("  Read sigma against this project's record: under ~3 sigma on six seeds")
print("  has reversed twice before.")
