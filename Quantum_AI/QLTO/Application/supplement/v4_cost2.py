"""Corrected cost comparison: charge parameter-shift honestly.

v4_cost.log's verdict table gave parameter-shift a variance of 1/S per energy
evaluation, because Aer's EstimatorV2 with precision=p returns the EXACT
expectation plus Gaussian noise of std p - it does not simulate shots. Tell:
measured V_ps = 5.9e-5 and 2*5.9e-5*8192 = 0.97 = precision^2 exactly, with no
dependence on Var(H) at all. That silently handed parameter-shift a ~28x shot
subsidy and inverted the verdict.

A real shot-based measurement of <H> must pay for two things QLTO's phase
estimation does not:

  Var(H)   the spread of the observable. Achieving standard error sigma needs
           Var(H)/sigma^2 shots, and Var(H) is extensive in N.
  G        the number of qubit-wise-commuting groups. Each is a separate
           MEASUREMENT SETTING, hence a separate circuit. Heisenberg needs 3
           (X-type, Y-type, Z-type); molecular Hamiltonians need hundreds.

QPE avoids BOTH. It reads the energy out of the phase of exp(-iHt), in one
setting, whatever H looks like. That is the structural advantage, and it is
independent of everything else I measured.

Fair accounting, allocating S shots per energy evaluation split across G groups:

    Var(<H>)   = (G/S) * sum_g Var(H_g)          [exact, from the statevector]
    Var(g_i)   = Var(<H>) / 2                    [two shift points, /2 each]
    circuits   = 2*M*G per full gradient         [2 shifts x G settings x M]
    shots      = 2*M*S

against QLTO's ceil(M/n) circuits and ceil(M/n)*S shots for ALL M components.

Cost index for equal RELATIVE precision on the descent direction (angle error is
||dg||/||g||, so each method is normalised by its OWN target norm):

    QLTO  ceil(M/n) * V(n)   / ||g_smeared||^2
    PS    2*M       * V_ps   / ||g_exact||^2
"""
import sys, os, contextlib, io
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return efficient_su2(N, reps=1), SparsePauliOp.from_list(ops)

def groups(M, n):
    return [list(range(i, min(i + n, M))) for i in range(0, M, n)]


def ps_variance(ansatz, H, c, S):
    """Exact shot-noise variance of a parameter-shift component, with grouping.

    Var(<H>) = (G/S) * sum_g Var(H_g) where the S shots are split evenly over
    the G qubit-wise-commuting groups. Averaged over the 2M shifted states the
    estimator would actually visit.
    """
    grps = H.group_commuting(qubit_wise=True)
    G = len(grps)
    M = ansatz.num_parameters
    tot = 0.0
    for i in range(M):
        for sgn in (+1.0, -1.0):
            p = c.copy(); p[i] += sgn * np.pi / 2
            sv = Statevector(ansatz.assign_parameters(p))
            v = 0.0
            for g in grps:
                e1 = float(np.real(sv.expectation_value(g)))
                e2 = float(np.real(sv.expectation_value((g @ g).simplify())))
                v += max(e2 - e1 ** 2, 0.0)
            tot += (G / S) * v
    # per component: two evaluations, each entering with weight 1/2
    return tot / (2 * M) / 2.0, G


N, R, S, REP = 4, 0.6, 8192, 8
ansatz, H = heis(N)
M = ansatz.num_parameters
c = np.random.RandomState(3).uniform(-np.pi, np.pi, M)

V_ps, G = ps_variance(ansatz, H, c, S)
sv = Statevector(ansatz.assign_parameters(c))
varH = float(np.real(sv.expectation_value((H @ H).simplify()))
             - np.real(sv.expectation_value(H)) ** 2)

# exact gradient norm for the parameter-shift target
gex = np.zeros(M)
for i in range(M):
    pp = c.copy(); pp[i] += np.pi / 2
    pm = c.copy(); pm[i] -= np.pi / 2
    gex[i] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp))
                                  .expectation_value(H)))
                    - float(np.real(Statevector(ansatz.assign_parameters(pm))
                                    .expectation_value(H))))
NGEX = float(np.linalg.norm(gex))

print("=" * 86)
print(f"CORRECTED COST COMPARISON.  Heisenberg N={N}, M={M}, S={S} shots/evaluation")
print("=" * 86)
print(f"  Var(H) at centre                  : {varH:.4f}")
print(f"  qubit-wise commuting groups G     : {G}")
print(f"  |g_exact|                         : {NGEX:.4f}")
print(f"  parameter-shift Var per component : {V_ps:.6e}   (fair, shot-based)")
print(f"    idealised value from v4_cost    : 5.9e-05      "
      f"({V_ps/5.9e-5:.0f}x subsidy)")
print(f"  circuits per full gradient        : 2*M*G = {2*M*G}")
C_ps = 2 * M * V_ps
rel_ps = C_ps / NGEX ** 2
print(f"  C_ps = 2M*V_ps                    : {C_ps:.6f}")
print(f"  cost/|g|^2                        : {rel_ps:.6e}")

print()
print(f"  {'method':<24}{'circuits':>10}{'V(n)':>12}{'cost':>11}"
      f"{'|g|':>8}{'cost/|g|^2':>13}{'vs PS':>9}")
print("  " + "-" * 87)
print(f"  {'parameter-shift':<24}{2*M*G:>10}{V_ps:>12.4e}{C_ps:>11.5f}"
      f"{NGEX:>8.3f}{rel_ps:>13.4e}{1.0:>9.2f}")

for tag, k, widths in (("QPE k=4", 4, (1, 2, 4, 8)),
                       ("Hadamard k=1", 1, (1, 2, 4, 8, 16))):
    q = Q(ansatz, H, shot_budget=S, num_ancillas=k)
    fit_n, fit_V, last_gn = [], [], None
    for n in widths:
        grps = groups(M, n)
        percomp, gn = [], np.zeros(M)
        for grp in grps:
            runs = np.array([q.sense_gradient(c, R, grp)[grp] for _ in range(REP)])
            percomp.extend(runs.var(axis=0, ddof=1).tolist())
            gn[grp] = runs.mean(axis=0)
        V = float(np.mean(percomp)); NG = float(np.linalg.norm(gn))
        fit_n.append(n); fit_V.append(V); last_gn = NG
        Cq = len(grps) * V
        rel = Cq / NG ** 2
        print(f"  {tag + ' n=' + str(n):<24}{len(grps):>10}{V:>12.4e}"
              f"{Cq:>11.5f}{NG:>8.3f}{rel:>13.4e}{rel/rel_ps:>9.2f}", flush=True)
    nn = np.array(fit_n, float)
    A = np.vstack([np.ones_like(nn), nn - 1.0]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(fit_V), rcond=None)
    Cg = a + b * (M - 1)
    print(f"  {tag + ' n=M pred':<24}{1:>10}{Cg:>12.4e}{Cg:>11.5f}"
          f"{last_gn:>8.3f}{Cg/last_gn**2:>13.4e}"
          f"{(Cg/last_gn**2)/rel_ps:>9.2f}")

print()
print("  vs PS < 1 means QLTO reaches the same relative gradient precision in")
print("  FEWER TOTAL SHOTS, i.e. a genuine information advantage, not just a")
print("  circuit-count one. The circuits column is the separate, larger win.")
print()
print("  G is the lever that generalises: it is 3 here and in the hundreds for")
print("  molecular Hamiltonians, and QPE pays it once regardless.")
for NN in (4, 6, 8):
    _, HH = heis(NN)
    print(f"    Heisenberg N={NN}: {len(HH.paulis)} terms -> "
          f"G = {len(HH.group_commuting(qubit_wise=True))}")
