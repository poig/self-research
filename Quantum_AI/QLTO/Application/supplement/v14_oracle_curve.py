"""Gradient quality vs cost: the QLTO oracle against parameter-shift.

This is the comparison the whole method rests on, stripped of everything that
confounds it. No optimizer, no step size, no schedule, no epochs, no trajectory -
just: at a given budget, how well do you know the descent direction?

Two reasons this beats an optimizer benchmark for settling it:

  NO TUNING       an optimizer comparison needs a step size, and any result then
                  depends on how well each arm was tuned. Gradient quality has no
                  free parameter.
  STATISTICAL     an optimizer comparison inherits trajectory variance - MaxCut
  POWER           N=6 has std up to 1.5, which is why 5 seeds could not resolve
                  anything and why three earlier results reversed on replication.
                  A gradient comparison carries only shot noise, which averages
                  down cheaply.

METRIC: cos(g_hat, grad E) against the EXACT gradient. Not against the R-smeared
target - descent follows the true gradient, and the smearing bias is precisely
what the curve should expose.

The two estimators have different error structure, which is the point:
  parameter-shift  unbiased, but pays 2*M*G circuits per gradient
  QLTO             O(R^2) bias floor, but one circuit per block for ALL M
                   components. By the sinc identity the floor is exactly
                   sin(R)/R at n=1 and cos(R)^{~Kn/M} beyond it, so the
                   asymptote is predicted, not fitted.

Expect QLTO to dominate at low budget and parameter-shift to overtake once its
1/sqrt(shots) noise drops below QLTO's bias floor. Where that crossover sits is
the practical answer.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def exact_gradient(ansatz, H, c):
    """Parameter-shift on the statevector - the ground truth."""
    g = np.zeros(len(c))
    for i in range(len(c)):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[i] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp))
                                    .expectation_value(H)))
                      - float(np.real(Statevector(ansatz.assign_parameters(pm))
                                      .expectation_value(H))))
    return g


def qlto_gradient(q, c, R):
    """One sensing circuit per block -> all M components."""
    g = np.zeros(len(c))
    for blk in q.layers:
        act = blk['params']
        if act:
            g += q.sense_gradient(c, R, act)
    return g


def pshift_gradient(ansatz, H, c, est):
    """Sampled parameter-shift: 2M expectation values, G circuits each."""
    pubs = []
    for i in range(len(c)):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        pubs.append((ansatz, H, pp)); pubs.append((ansatz, H, pm))
    r = est.run(pubs).result()
    e = np.array([float(r[j].data.evs) for j in range(2 * len(c))])
    return 0.5 * (e[0::2] - e[1::2])


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 1e-15 and nv > 1e-15 else 0.0


PROBLEMS = [
    ("H2",              B.get_h2_problem),
    ("MaxCut N=4",      lambda: B.get_maxcut_problem(4)),
    ("Heisenberg N=4",  lambda: B.get_heisenberg_problem(4)),
    ("Heisenberg N=6",  lambda: B.get_heisenberg_problem(6)),
]
QLTO_SHOTS = (256, 1024, 4096, 16384, 65536)
PS_SHOTS   = (16, 64, 256, 1024, 4096)      # shots per group per expectation
REP = 5
R_DEFAULT = 0.6

print("=" * 96)
print("GRADIENT QUALITY vs COST — QLTO oracle against sampled parameter-shift")
print("=" * 96)
print(f"  metric: cos(g_hat, grad E) against the exact gradient, {REP} repeats, R={R_DEFAULT}")

for pname, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=8192)
    M = ansatz.num_parameters
    Gp = B.pauli_groups(H)
    nblk = len([b for b in q.layers if b['params']])
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, M)
    gx = exact_gradient(ansatz, H, c)

    print(f"\n  ===== {pname} | M={M} | blocks={nblk} | G={Gp} | "
          f"||H0||={q.H0_norm:.2f} =====")
    print(f"  {'method':<20}{'circuits':>9}{'shots':>10}{'cos':>9}{'rel err':>9}")
    print("  " + "-" * 57)

    for S in QLTO_SHOTS:
        qq = Q(ansatz, H, shot_budget=S)
        cs, re = [], []
        for _ in range(REP):
            g = qlto_gradient(qq, c, R_DEFAULT)
            cs.append(cos(g, gx))
            re.append(np.linalg.norm(g - gx) / np.linalg.norm(gx))
        print(f"  {'QLTO S=%d' % S:<20}{nblk:>9}{nblk*S:>10}"
              f"{np.mean(cs):>9.4f}{np.mean(re):>9.3f}", flush=True)

    for S in PS_SHOTS:
        est = BackendEstimatorV2(backend=AerSimulator(),
                                 options={'default_precision': 1.0/np.sqrt(S)})
        cs, re = [], []
        for _ in range(REP):
            g = pshift_gradient(ansatz, H, c, est)
            cs.append(cos(g, gx))
            re.append(np.linalg.norm(g - gx) / np.linalg.norm(gx))
        print(f"  {'p-shift S=%d' % S:<20}{2*M*Gp:>9}{2*M*Gp*S:>10}"
              f"{np.mean(cs):>9.4f}{np.mean(re):>9.3f}", flush=True)

print()
print("  Read across at matched SHOTS and at matched CIRCUITS. QLTO should lead")
print("  at low budget and flatten at its sinc(R)-set bias floor; parameter-shift")
print("  has no floor and must overtake eventually. The crossover is the answer.")
