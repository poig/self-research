"""Acceptance test for sim_seed: does the null experiment now return EXACTLY zero?

v26/v27 established that comparing the shipping configuration against a
term-sorted one is a NULL experiment - v27 proved the two are the same unitary to
0.000e+00 - and that the unseeded harness reported up to 3.3 sigma on it. The
fix is sim_seed plus reset_shot_stream(), which makes two arms issuing circuits
in the same order draw identical shot noise.

The acceptance criterion is not "smaller sigma". It is EXACTLY ZERO, bit for bit,
on every seed. Anything else means the shot stream is not actually aligned and
the fix does not do what it claims.

Also re-runs the same null WITHOUT seeding at the same seeds, so the before/after
sits in one table rather than across two logs.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.primitives import StatevectorEstimator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

EST = StatevectorEstimator()
def energy_at(ansatz, H, p):
    return float(EST.run([(ansatz, H, np.asarray([p]))]).result()[0].data.evs.ravel()[0])


def run(ansatz, H, sort, seed, sim_seed, epochs=20, k_steps=15, shots=8192):
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=3,
          sort_terms=sort, sim_seed=sim_seed)
    q.reset_shot_stream()
    BLK = [b['params'] for b in q.layers if b['params']]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            g = q.sense_gradient(p, R, act)
            p = q._execute_walk(p, k_steps, dt, R, act, g)
    return energy_at(ansatz, H, p)


PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
            ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))]
SEEDS = (42, 43, 44, 45, 46, 47)

print("=" * 92)
print("SEEDED NULL — base vs term-sorted, which v27 proved are the SAME UNITARY")
print("=" * 92)
print(f"  {len(SEEDS)} seeds, 20 epochs, kappa=3. The true difference is exactly 0.")
print("  UNSEEDED is the old harness; SEEDED uses sim_seed + reset_shot_stream().")
print()
print(f"  {'problem':<18}{'unseeded diff':>15}{'sigma':>8}{'seeded diff':>14}"
      f"{'max |diff|':>13}{'verdict':>10}")
print("  " + "-" * 78)

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()

    du = np.array([run(ansatz, H, True, s, None) - run(ansatz, H, False, s, None)
                   for s in SEEDS])
    semu = du.std(ddof=1) / np.sqrt(len(SEEDS))
    sigu = abs(du.mean()) / max(semu, 1e-12)

    ds = np.array([run(ansatz, H, True, s, 1000 + s) - run(ansatz, H, False, s, 1000 + s)
                   for s in SEEDS])
    ok = 'PASS' if np.max(np.abs(ds)) < 1e-12 else 'FAIL'
    print(f"  {name:<18}{du.mean():>+15.4f}{sigu:>8.1f}{ds.mean():>+14.2e}"
          f"{np.max(np.abs(ds)):>13.2e}{ok:>10}", flush=True)

print()
print("  PASS requires max |diff| = 0 to machine precision on every seed. A small")
print("  but nonzero seeded difference would mean the shot streams drift apart -")
print("  e.g. because the two arms issue different numbers of circuits - and the")
print("  pairing is still incomplete.")
