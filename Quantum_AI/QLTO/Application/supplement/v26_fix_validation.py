"""Do the proposed V3 fixes survive an END-TO-END optimisation, not just a
circuit-metric measurement?

Two fixes are on the table and they have different evidential status:

  TERM SORTING   v24 measured depth N^1.22 -> N^0.64 with unitary error
                 0.00e+00, so it is an EXACT rewrite - only commuting terms are
                 transposed. Accuracy MUST be unchanged. This arm is therefore a
                 falsification test of my own claim: if energies move, the
                 "exact" claim is wrong and something else changed with it.

  kappa = 4 -> 2 UNTESTED, and the one that could make V3 viable on noisy
                 hardware. Sigma_a r_a falls 8 -> 2, cutting gate count ~4x and
                 taking survival at N=6 from 8.6e-03 to ~0.30. The supporting
                 evidence is indirect: anomaly_e swept k from 3 to 7, moved the
                 QPE bin width 16x straddling the signal, and found the gradient
                 error UNMOVED at every block - which says resolution is not what
                 kappa is buying. But that was a gradient measurement at one
                 problem, not an optimisation across the suite.

Four arms, PAIRED on seeds so every arm starts each seed from identical initial
parameters. These notes record two sub-2-sigma results that reversed on
replication, both from comparing across runs.

    base      sorting off, kappa=4     <- shipping, the reference
    sorted    sorting on,  kappa=4     <- must tie base
    sorted-k3 sorting on,  kappa=3
    sorted-k2 sorting on,  kappa=2     <- the one that matters

Also reports transpiled depth and cx per config, because the whole point of the
kappa cut is gate count and a tie on energy is only interesting alongside the
resource saving it buys.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']
EST = StatevectorEstimator()


def energy_at(ansatz, H, p):
    return float(EST.run([(ansatz, H, np.asarray([p]))]).result()[0].data.evs.ravel()[0])


def support(p):
    lbl = str(p)[::-1]
    return {i for i, ch in enumerate(lbl) if ch != 'I'}


def layer_sort(op):
    """Greedy partition into layers of mutually disjoint support. Only ever
    transposes terms with disjoint support, which commute, so the product is
    unchanged - verified to 0.00e+00 in v24."""
    lays = []
    for p, c in zip(op.paulis, op.coeffs):
        s = support(p)
        for L in lays:
            if not (s & L['u']):
                L['t'].append((p, c)); L['u'] |= s
                break
        else:
            lays.append({'t': [(p, c)], 'u': set(s)})
    out = []
    for L in lays:
        out.extend(L['t'])
    return SparsePauliOp.from_list([(str(p), c) for p, c in out])


ARMS = [('base', False, 4), ('sorted', True, 4),
        ('sorted-k3', True, 3), ('sorted-k2', True, 2)]


def run(ansatz, H, sort, kappa, seed, shots=8192, epochs=20, k_steps=15):
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=kappa)
    if sort:
        q.H_sense = layer_sort(q.H_sense)
    BLK = [b['params'] for b in q.layers if b['params']]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            g = q.sense_gradient(p, R, act)
            p = q._execute_walk(p, k_steps, dt, R, act, g)
    return energy_at(ansatz, H, p)


def circuit_cost(ansatz, H, sort, kappa):
    q = Q(ansatz, H, shot_budget=1024, num_ancillas=kappa)
    if sort:
        q.H_sense = layer_sort(q.H_sense)
    act = [b['params'] for b in q.layers if b['params']][0]
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)
    t = transpile(q._build_qpe_sensing_circuit(c, 0.6, act),
                  basis_gates=BASIS, optimization_level=1)
    return t.depth(), t.count_ops().get('cx', 0)


PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
            ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6))]
SEEDS = (42, 43, 44, 45, 46, 47)
P2Q = 5e-3

print("=" * 100)
print("V3 FIX VALIDATION — term sorting and kappa reduction, end to end")
print("=" * 100)
print(f"  {len(SEEDS)} PAIRED seeds, 20 epochs, k_steps=15, 8192 shots.")
print("  'vs base' is the paired mean difference; POSITIVE = worse than shipping.")

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
    print(f"\n  ===== {name} | exact {exact:.4f} =====")
    print(f"  {'arm':<12}{'E_final':>10}{'vs base':>10}{'sigma':>7}"
          f"{'depth':>8}{'cx':>7}{'cx vs base':>12}{'survival':>11}")
    print("  " + "-" * 77)

    res = {}
    for arm, sort, kappa in ARMS:
        res[arm] = np.array([run(ansatz, H, sort, kappa, s) for s in SEEDS])
    base = res['base']
    _, cx_base = circuit_cost(ansatz, H, False, 4)

    for arm, sort, kappa in ARMS:
        d, cx = circuit_cost(ansatz, H, sort, kappa)
        diff = res[arm] - base
        sem = diff.std(ddof=1) / np.sqrt(len(SEEDS)) if arm != 'base' else 0.0
        sig = abs(diff.mean()) / sem if sem > 1e-12 else 0.0
        surv = (1 - P2Q) ** cx
        print(f"  {arm:<12}{res[arm].mean():>10.4f}{diff.mean():>+10.4f}{sig:>7.1f}"
              f"{d:>8}{cx:>7}{cx / max(cx_base, 1):>11.2f}x{surv:>11.3f}", flush=True)

print()
print("  READ IN THIS ORDER:")
print("   1. 'sorted' MUST tie 'base'. It is an exact rewrite; a real difference")
print("      would falsify v24 and invalidate the depth fix.")
print("   2. If sorted-k2 also ties, the kappa cut is free and V3's survival")
print("      improves by the cx ratio - the only thing that makes it viable on")
print("      noisy hardware.")
print("   3. Under 2 sigma at six seeds is a TIE in these notes, not a small")
print("      effect. Two prior results at 2.5-3 sigma reversed on replication.")
