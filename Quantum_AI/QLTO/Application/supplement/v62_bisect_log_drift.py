"""Which commit moved v20's logged row? Bisect nisq_v3 against its own history.

v20 logged, at Heisenberg N=4 / 256 shots / seeds 42-47 on 2026-08-03 02:30:
    walk -5.7904   boltz -5.5951   gradstep -5.8096
Re-running v20's OWN code today (v53b) gives
    walk -5.0514   boltz -5.4167   gradstep -5.5801
with the drift concentrated in the WALK arm: +0.739, about 3.7 SEM, against
+0.18 and +0.23 for the other two.

RULED OUT SO FAR:
  merged_walk   v53c, exonerated: -0.168 / -0.074 / +0.054 at 256/1024/8192
                shots, all under 2 sigma, and the sign flips.
  tau0          git says the line `self.tau0 = pi/(qpe_margin*H0_norm)` has been
                touched by exactly one commit - the original.
  sort_terms    v27 proved base and sorted are the SAME UNITARY to 0.000e+00,
                because the reordering is a disjoint-support partition and
                disjoint terms commute.
  num_ancillas  v20 pins num_ancillas=4 explicitly, so the 4->3 default change
                cannot reach it.

WHY THIS MATTERS BEYOND ONE ROW. supplement/results/ holds ~60 logs written
against a moving nisq_v3, and this session quoted several of them as settled
fact. If a commit moved a shipped code path, every log predating it needs
re-establishing before it can be cited. If instead nothing in nisq_v3 moved and
the difference is environmental, that is a different and milder problem.

METHOD. Extract nisq_v3.py at each commit into its own directory, import it
under a private module name, and run v20's protocol - v20's own sense_table and
boltz_step and grad_step, unmodified - against each. Same seeds, same shots, same
epochs. The arm that moves localises the change; the commit where it moves dates
it.

v20's sensing is SELF-CONTAINED (it builds its own QPE circuit with
SuzukiTrotter(order=2, reps=max(1, 2^a//2)) and reads q.H_sense and q.tau0), so
only the WALK arm calls into nisq_v3 for its dynamics - which is consistent with
the walk being the arm that moved.
"""
import sys, os, subprocess, importlib.util, contextlib, io, shutil
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(APP)))
sys.path.insert(0, APP)
os.chdir(APP)

REL = 'Quantum_AI/QLTO/Application/nisq_v3.py'
COMMITS = [('9e8290c', '07-30 docs modification'),
           ('9f4d3a1', '08-03 v3 correction'),
           ('4dc62a4', '08-04 update'),
           ('108c945', '08-05 v5 added'),
           ('WORKING', 'working tree (today)')]

TMP = '/tmp/v62_bisect'
os.makedirs(TMP, exist_ok=True)


def load_v3(tag):
    """Import nisq_v3.py as it stood at `tag`, under a private module name."""
    d = os.path.join(TMP, tag)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, 'nisq_v3.py')
    if tag == 'WORKING':
        shutil.copy(os.path.join(APP, 'nisq_v3.py'), dst)
    else:
        src = subprocess.run(['git', '-C', REPO, 'show', f'{tag}:{REL}'],
                             capture_output=True, text=True, check=True).stdout
        open(dst, 'w').write(src)
    name = f'nisq_v3_{tag}'
    spec = importlib.util.spec_from_file_location(name, dst)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# v20's own protocol functions, loaded once (they do not depend on nisq_v3's
# internals except through the QLTOv3 object we pass in).
SRC = os.path.join(APP, 'supplement', 'v20_walk_shot_tolerance.py')
src = open(SRC).read()
cut = src.index('SIZES = (4, 6, 8)')
V20 = {'__name__': 'v20_bisect', '__file__': SRC}
exec(compile(src[:cut], SRC, 'exec'), V20)

SEEDS = (42, 43, 44, 45, 46, 47)
ARMS = V20['ARMS']

print("=" * 92)
print("BISECTING v20's LOG DRIFT ACROSS nisq_v3 HISTORY")
print("=" * 92)
print(f"  Heisenberg N=4, 256 shots, seeds 42-47, 20 epochs, kappa=4.")
print(f"  v20 LOGGED (2026-08-03 02:30):  walk -5.7904  boltz -5.5951"
      f"  gradstep -5.8096")
print()
print(f"  {'commit':>10}{'when':>26}" + "".join(f"{a:>12}" for a in ARMS))
print("  " + "-" * 72)

for tag, when in COMMITS:
    try:
        v3 = load_v3(tag)
    except Exception as e:
        print(f"  {tag:>10}{when:>26}   load failed: {type(e).__name__}: {e}")
        continue

    res = {a: [] for a in ARMS}
    ok = True
    for sd in SEEDS:
        ansatz, H = V20['heis'](4)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                q = v3.QLTOv3(ansatz, H, shot_budget=256, num_ancillas=4)
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
        except Exception as e:
            print(f"  {tag:>10}{when:>26}   run failed: {type(e).__name__}: {e}")
            ok = False
            break
    if not ok:
        continue
    print(f"  {tag:>10}{when:>26}"
          + "".join(f"{np.mean(res[a]):>12.4f}" for a in ARMS), flush=True)

print()
print("  The commit at which the WALK column jumps from ~-5.79 to ~-5.05 is the")
print("  one that moved it. If no commit reproduces -5.7904, nisq_v3 is not the")
print("  cause and the difference is environmental - qiskit 2.2.3 is current, and")
print("  v20's log carries a QFT deprecation warning from qiskit 2.1, so a")
print("  library upgrade between the two is the remaining suspect.")
