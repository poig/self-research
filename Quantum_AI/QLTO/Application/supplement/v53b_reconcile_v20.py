"""Does v20 still reproduce its OWN logged row? A stale-baseline check.

v20 logged, at Heisenberg N=4 / 256 shots / seeds 42-47, 6 paired seeds:

    walk -5.7904 +- 0.045   boltz0.1 -5.5951 +- 0.054   gradstep -5.8096 +- 0.035

v53 re-implements the same protocol - same sizes, seeds, epochs, k_steps,
num_ancillas=4, same R and dt schedules, same three arms sharing one sensing call
- and gets, at the identical configuration:

    walk -4.5406            boltz0.1 -5.0903            gradstep -5.3907

ALL THREE ARMS ARE DOWN BY ~1 HARTREE. That pattern points at the SHARED sensing
path rather than at any one decoder, since the arms differ only in how they
consume a gradient the sensing produces. And it is far outside the quoted SEMs -
1.25 Hartree against +-0.045 is not a sampling fluctuation.

One candidate was eliminated already: v53 replaced the deprecated
qiskit.circuit.library.QFT with qiskit.synthesis.qft.synth_qft_full, and the two
are EXACT to 0.000e+00 at k=3 and k=4.

The remaining candidate is that nisq_v3 HAS CHANGED since v20's log was written.
This session and earlier ones altered sort_terms, num_ancillas and merged_walk
defaults; v20 pins num_ancillas=4 explicitly but not the others, and merged_walk
in particular is documented as NOT an equivalent rewrite - "at the angles
actually used they differ by 0.813 in operator norm, so this is different
dynamics at lower depth". That would move the walk arm; it would not by itself
move gradstep, so if all three have moved the cause is upstream of the decoders.

THIS RUNS v20'S OWN FUNCTIONS, UNMODIFIED, at the single logged configuration, by
executing the file up to its experiment block and calling run_seed directly.

  reproduces the log      -> v53 has a bug, and v20's baseline stands
  does not reproduce      -> v20's log is STALE, its row cannot be used as a
                             baseline, and every verdict resting on it needs
                             re-establishing before the walk question is decided
"""
import sys, os
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)

SRC = os.path.join(APP, 'supplement', 'v20_walk_shot_tolerance.py')
src = open(SRC).read()
cut = src.index('SIZES = (4, 6, 8)')          # stop before v20's own grid runs
mod = {'__name__': 'v20_reconcile', '__file__': SRC}   # v20 reads __file__
exec(compile(src[:cut], SRC, 'exec'), mod)

print("=" * 88)
print("RECONCILING v53 AGAINST v20's LOGGED ROW")
print("=" * 88)
print("  v20's own code, unmodified, at Heisenberg N=4 / 256 shots / seeds 42-47.")
print()

ARMS = mod['ARMS']
res = {a: [] for a in ARMS}
prob = lambda: mod['heis'](4)
for sd in (42, 43, 44, 45, 46, 47):
    r = mod['run_seed'](prob, sd, 256)
    for a in ARMS:
        res[a].append(r[a])
    print(f"    seed {sd}: " + "  ".join(f"{a}={r[a]:+.4f}" for a in ARMS),
          flush=True)

print()
print(f"  {'arm':>10}{'v20 today':>13}{'v20 logged':>13}{'v53 today':>12}{'today-log':>12}")
print("  " + "-" * 60)
logged = {'walk': -5.7904, 'boltz0.1': -5.5951, 'gradstep': -5.8096}
v53 = {'walk': -4.5406, 'boltz0.1': -5.0903, 'gradstep': -5.3907}
for a in ARMS:
    m = float(np.mean(res[a]))
    print(f"  {a:>10}{m:>13.4f}{logged[a]:>13.4f}{v53[a]:>12.4f}"
          f"{m - logged[a]:>+12.4f}")

print()
print("  If 'v20 today' matches 'v20 logged', v53 has a bug and the baseline is")
print("  sound. If 'v20 today' matches 'v53 today' instead, nisq_v3 has changed")
print("  under both and the LOG is stale - in which case v20's row cannot be used")
print("  to decide anything, and neither can any other verdict that predates the")
print("  same defaults changes without being re-established.")
