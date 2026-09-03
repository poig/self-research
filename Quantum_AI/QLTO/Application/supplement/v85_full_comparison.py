"""V6 against every gradient method here, across the Richardson crossover.

V6 now carries all of its improvements as defaults: log-width design register,
one global block, parallel parity scratch, radius rescaled by block width, and
richardson='auto', which switches on the derived crossover rather than a flag.

This is the comparison that decides whether that default is right. It sweeps the
budget ACROSS the crossover, so the auto-switch can be checked instead of trusted:

    T_cross ~ 3.6e6 * a * c4^3 / c^5      (v84)

with the switch set at 10^6 total shots per gradient. Below it Richardson should
lose and 'auto' should decline; above it Richardson should win and 'auto' should
take it. If the switch fires on the wrong side, the threshold is wrong and the
default should change.

ARMS, all at MATCHED TOTAL SHOTS and scored by cos against the exact gradient:

    parameter-shift   2MG circuits, unbiased, T^(-1/2). The method to beat.
    SPSA              2GK circuits, K perturbations, best K taken.
    V5                one-hot register, layered blocks, G*M/N circuits.
    V6 plain          design register, global block, G circuits.
    V6 richardson     as above at two radii, 2G circuits.
    V6 auto           what a caller actually gets from the defaults.

WHAT WOULD CONFIRM THE DEFAULT: 'auto' tracking whichever of plain/richardson is
better at every budget.
WHAT WOULD KILL IT: 'auto' choosing the worse arm anywhere, which would mean the
10^6 threshold is on the wrong side of the real crossover for this problem, since
c and c4 are problem-dependent and the 3.6e6 constant is a scale rather than a
threshold.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v5
import nisq_v6


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


def ps_grad(ansatz, gmats, theta, shots, rng):
    g = np.zeros(len(theta))
    for i in range(len(theta)):
        for s in (+1, -1):
            t = theta.copy()
            t[i] += s * np.pi / 2
            g[i] += s * energy_sampled(ansatz, gmats, t, shots, rng) / 2
    return g


N, REPS = 4, 2
BUDGETS = (2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22)
RGRID = (1.2, 0.9, 0.6, 0.45, 0.3)

ansatz = efficient_su2(N, reps=2)
H = heis(N)
M = ansatz.num_parameters
Hm = H.to_matrix()
theta = np.random.RandomState(31).uniform(-np.pi, np.pi, M)
g_ex = exact_grad(ansatz, Hm, theta)
with contextlib.redirect_stdout(io.StringIO()):
    probe = nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct')
G = len(probe.groups)
L5 = len([b for b in probe.layers if b['params']])
gmats = [(g.to_matrix(), (g @ g).simplify().to_matrix()) for g in probe.groups]

print("=" * 104)
print(f"V6 AGAINST EVERY GRADIENT METHOD HERE   (N={N}, M={M}, G={G})")
print("=" * 104)
print(f"  Matched total shots. Richardson auto-switch set at 10^6 total shots per")
print(f"  gradient; the sweep crosses it between T=2^18 and T=2^22.")
print()
print(f"  {'T total':>10}{'method':>18}{'circuits':>10}{'1-cos':>10}{'R*':>7}"
      f"{'vs p-shift':>12}")
print("  " + "-" * 68)


def qlto_arm(cls, kw, ncirc, rgrid=RGRID):
    best, br = 2.0, None
    for Rv in rgrid:
        cs = []
        for s in range(REPS):
            with contextlib.redirect_stdout(io.StringIO()):
                q = cls(ansatz, H, shot_budget=max(1, T // ncirc),
                        sim_seed=700 + s, **kw)
            gh = np.zeros(M)
            for act in [b['params'] for b in q.layers if b['params']]:
                gi, _ = q.sense(theta, Rv, act)
                gh += gi
            cs.append(cosine(gh, g_ex))
        e = max(1 - float(np.mean(cs)), 1e-12)
        if e < best:
            best, br = e, Rv
    return best, br


for T in BUDGETS:
    rng0 = np.random.RandomState(9000)
    per_ps = max(1, T // (2 * M * G))
    cs = [cosine(ps_grad(ansatz, gmats, theta,
                         per_ps, np.random.RandomState(9000 + s)), g_ex)
          for s in range(REPS)]
    e_ps = max(1 - float(np.mean(cs)), 1e-12)
    print(f"  {T:>10}{'parameter-shift':>18}{2 * M * G:>10}{e_ps:>10.5f}"
          f"{'-':>7}{1.0:>12.2f}")

    best_sp = 2.0
    for kf in (0.5, 1.0, 2.0):
        K = max(1, int(kf * M))
        Sp = T // (2 * G * K)
        if Sp < 1:
            continue
        cs = []
        for s in range(REPS):
            rng = np.random.RandomState(9100 + s)
            gs = np.zeros(M)
            for _ in range(K):
                sig = rng.choice([-1.0, 1.0], size=M)
                ep = energy_sampled(ansatz, gmats, theta + 0.45 * sig, Sp, rng)
                em = energy_sampled(ansatz, gmats, theta - 0.45 * sig, Sp, rng)
                gs += ((ep - em) / 0.9) * sig
            cs.append(cosine(gs / K, g_ex))
        best_sp = min(best_sp, max(1 - float(np.mean(cs)), 1e-12))
    print(f"  {T:>10}{'SPSA':>18}{'2GK':>10}{best_sp:>10.5f}{'-':>7}"
          f"{e_ps / best_sp:>12.2f}")

    e5, r5 = qlto_arm(nisq_v5.QLTOv5, {'gradient_mode': 'direct'}, G * L5)
    print(f"  {T:>10}{'V5':>18}{G * L5:>10}{e5:>10.5f}{r5:>7.2f}"
          f"{e_ps / e5:>12.2f}")

    e6, r6 = qlto_arm(nisq_v6.QLTOv6, {'richardson': False}, G)
    print(f"  {T:>10}{'V6 plain':>18}{G:>10}{e6:>10.5f}{r6:>7.2f}"
          f"{e_ps / e6:>12.2f}")

    e6r, r6r = qlto_arm(nisq_v6.QLTOv6, {'richardson': True}, 2 * G)
    print(f"  {T:>10}{'V6 richardson':>18}{2 * G:>10}{e6r:>10.5f}{r6r:>7.2f}"
          f"{e_ps / e6r:>12.2f}")

    with contextlib.redirect_stdout(io.StringIO()):
        probe6 = nisq_v6.QLTOv6(ansatz, H, shot_budget=max(1, T // G))
    fired = probe6._use_richardson()
    ea, ra = qlto_arm(nisq_v6.QLTOv6, {}, (2 * G) if fired else G)
    ok = 'ok' if abs(ea - min(e6, e6r)) <= 0.25 * min(e6, e6r) else 'WRONG SIDE'
    print(f"  {T:>10}{'V6 auto':>18}{(2 * G) if fired else G:>10}{ea:>10.5f}"
          f"{ra:>7.2f}{e_ps / ea:>12.2f}   rich={fired} {ok}")
    print("  " + "." * 68, flush=True)

print()
print("  'vs p-shift' is parameter-shift error over the row's error: above 1 means")
print("  the row is more accurate at the same total shots. Circuits is the other")
print("  axis and does not appear in that ratio.")
print()
print("  The auto row should track whichever of plain/richardson is better. If it")
print("  is marked WRONG SIDE, the 10^6 threshold sits on the wrong side of this")
print("  problem's crossover and the default needs changing.")
