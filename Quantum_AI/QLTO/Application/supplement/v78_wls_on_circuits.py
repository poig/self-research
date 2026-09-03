"""Does the WLS decoder's synthetic win appear on the ACTUAL sensing circuits?

v77 measured the decoder change on a synthetic landscape: regressing the shot
record on the realised design matrix beat the marginal decoder at every degree-2
weight, 0.00178 -> 0.00060 MSE at zero weight. That was a model. The landscape a
real circuit produces is not linear in sigma, the register outcomes are whatever
the hardware sampled, and the energies carry group structure. None of that is in
the synthetic test, so the win has to be re-measured where it would actually be
used before the default is changed.

This re-runs v72 unchanged except for the decoder, so the comparison is like for
like: same ansatz, same Heisenberg Hamiltonian, same R, same theta, same shot
accounting, same cosine scoring, SPSA still given its best K.

WHAT WOULD JUSTIFY CHANGING THE DEFAULT: wls at or below marginal in 1-cos at
every size and budget, with the QLTO/SPSA ratio rising. The decoder is free -
identical circuits, identical shots - so anything better than parity is worth
taking.
WHAT WOULD KILL IT: wls at or above marginal on real circuits. That would mean
the synthetic landscape carried the win and the real one does not, most likely
because E(theta + R sigma) has enough higher-order Walsh content that a
main-effects-only regression is misspecified where the marginal decoder is not.
Then the decoder stays opt-in and the synthetic result must be labelled as
model-only.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v5


def heis(N):
    o = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def exact_grad(ansatz, Hm, theta):
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            v = Statevector(ansatz.assign_parameters(t)).data
            g[i] += s * float(np.real(np.conj(v) @ (Hm @ v))) / 2
    return g


def energy_sampled(ansatz, gmats, theta, shots, rng):
    v = Statevector(ansatz.assign_parameters(theta)).data
    tot = 0.0
    for Hg, Hg2 in gmats:
        m1 = float(np.real(np.conj(v) @ (Hg @ v)))
        m2 = float(np.real(np.conj(v) @ (Hg2 @ v)))
        tot += m1 + rng.normal(0.0, np.sqrt(max(m2 - m1 * m1, 0.0) / max(shots, 1)))
    return tot


R, REPEATS = 0.45, 4
BUDGETS = (2 ** 15, 2 ** 17)

print("=" * 104)
print("WLS DECODER ON REAL CIRCUITS  (v72 rerun, decoder is the only change)")
print("=" * 104)
print(f"  R = {R}, {REPEATS} repeats. Identical circuits and shots for both")
print("  decoders; only the reduction of the shot record differs.")
print()
print(f"  {'N':>3}{'M':>4}{'T total':>10}{'1-cos marg':>12}{'1-cos wls':>11}"
      f"{'1-cos SPSA':>12}{'wls/marg':>10}{'SPSA/wls':>10}{'SPSA/marg':>11}")
print("  " + "-" * 83)

for N in (4, 6):
    ansatz = efficient_su2(N, reps=2)
    H = heis(N)
    M = ansatz.num_parameters
    Hm = H.to_matrix()
    theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
    g_ex = exact_grad(ansatz, Hm, theta)

    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct')
    groups = probe.groups
    G = len(groups)
    blocks = [b['params'] for b in probe.layers if b['params']]
    L = len(blocks)
    gmats = [(g.to_matrix(), (g @ g).simplify().to_matrix()) for g in groups]

    engines = {}
    for dec in ('marginal', 'wls'):
        with contextlib.redirect_stdout(io.StringIO()):
            engines[dec] = [nisq_v5.QLTOv5(ansatz, H, shot_budget=1024,
                                           gradient_mode='direct',
                                           sim_seed=700 + r, decoder=dec)
                            for r in range(REPEATS)]

    for T in BUDGETS:
        Sq = max(1, T // (G * L))
        res = {}
        for dec, qs in engines.items():
            cs = []
            for q in qs:
                q.shot_budget = int(Sq)
                gh = np.zeros(M)
                for act in blocks:
                    gi, _ = q.sense(theta, R, act)
                    gh += gi
                cs.append(cosine(gh, g_ex))
            res[dec] = max(1 - float(np.mean(cs)), 1e-9)

        best_sp = -2.0
        for kf in (0.5, 1.0, 2.0):
            K = max(1, int(kf * M))
            Sp = T // (2 * G * K)
            if Sp < 1:
                continue
            cs = []
            for rep in range(REPEATS):
                rng = np.random.RandomState(9000 + rep)
                g_sp = np.zeros(M)
                for _ in range(K):
                    sig = rng.choice([-1.0, 1.0], size=M)
                    ep = energy_sampled(ansatz, gmats, theta + R * sig, Sp, rng)
                    em = energy_sampled(ansatz, gmats, theta - R * sig, Sp, rng)
                    g_sp += ((ep - em) / (2.0 * R)) * sig
                g_sp /= K
                cs.append(cosine(g_sp, g_ex))
            best_sp = max(best_sp, float(np.mean(cs)))
        es = max(1 - best_sp, 1e-9)

        print(f"  {N:>3}{M:>4}{T:>10}{res['marginal']:>12.5f}{res['wls']:>11.5f}"
              f"{es:>12.5f}{res['wls'] / res['marginal']:>10.2f}"
              f"{es / res['wls']:>10.2f}{es / res['marginal']:>11.2f}",
              flush=True)
    print("  " + "." * 83)

print()
print("  'wls/marg' below 1 means the decoder helps; the last two columns are the")
print("  QLTO/SPSA ratio under each decoder, against v72's 5.66x (N=4) and 5.21x")
print("  (N=6) which were measured with the marginal decoder.")
print()
print("  A win here is free: the circuits, the shots and the counts are identical,")
print("  and only the arithmetic applied to them changes. A loss would mean the")
print("  real landscape has higher-order Walsh content that a main-effects-only")
print("  regression is misspecified against, which the synthetic test could not see.")
