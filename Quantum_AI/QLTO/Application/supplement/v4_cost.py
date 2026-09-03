"""Is QLTO actually CHEAPER, or just 1 circuit at the same shot cost?

The marginal estimator is LINEAR in the measured energy, so every shot informs
every coordinate and shot noise averages out regardless of how many vertices go
unsampled. That is why it survives at large block width where argmin/top-m/
Boltzmann die. But linearity does not make it free: the OTHER coordinates'
randomisation is noise for coordinate i, so

    Var(g_i) ~ (1/S) * [ a + b*(n-1) ]

  a  quantum measurement noise (1/tau^2 for Hadamard, Var(H) for QPE)
  b  per-coordinate landscape variation, ~R^2 * (dE/dtheta_j)^2

With ceil(M/n) circuits of width n at S shots each, the total shots to reach a
fixed per-component precision is proportional to

    C(n) = ceil(M/n) * [a + b*(n-1)]

    n=1 (fully layered): M*a
    n=M (global):        a + b*(M-1)

so GLOBAL BEATS LAYERED IFF b < a, and by a factor approaching M when b << a.
That is the whole question, and it is two measured constants.

Then the real comparison. Parameter-shift needs 2 circuits per component, 2M
total, each carrying only ITS OWN component. QLTO gets all n components of a
group from one circuit. Cost index in the same units:

    C_ps = 2M * V_ps      vs      C_qlto = ceil(M/n) * V(n)

If C_qlto < C_ps, QLTO is cheaper in TOTAL SHOTS - a real information advantage.
If C_qlto ~ C_ps, then the saving is purely CIRCUIT COUNT, i.e. it converts a
per-circuit overhead into shots, which is worth a lot on hardware and nothing in
complexity terms. Both outcomes are publishable; only one of them is a speedup.

CAVEAT kept explicit: QLTO estimates the R-SMEARED gradient, parameter-shift the
exact one. Different targets, so each cost is normalised by ITS OWN gradient norm
and the comparison is "shots to reach the same RELATIVE precision on a usable
descent direction", not on identical quantities.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = "/home/poig/project/self-research/Quantum_AI/QLTO/Application"
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
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

N, R, S, REP = 4, 0.6, 8192, 8
ansatz, H = heis(N)
M = ansatz.num_parameters
c = np.random.RandomState(3).uniform(-np.pi, np.pi, M)

print("=" * 84)
print(f"COST SCALING vs block width.  Heisenberg N={N}, M={M} params, "
      f"S={S} shots, {REP} reps")
print("=" * 84)

results = {}
for tag, k, widths in (("Hadamard k=1", 1, (1, 2, 4, 8, 16)),
                       ("QPE k=4", 4, (1, 2, 4, 8))):
    q = Q(ansatz, H, shot_budget=S, num_ancillas=k)
    print(f"\n  --- {tag} ---")
    print(f"  {'n':>4}{'circuits':>10}{'qubits':>8}{'V(n) per comp':>16}"
          f"{'C(n)=ceil(M/n)V':>18}{'|g| targ':>10}{'time':>7}")
    print("  " + "-" * 73)
    Vs, ns, GN = [], [], []
    for n in widths:
        t0 = time.time()
        grps = groups(M, n)
        percomp, gnorm = [], np.zeros(M)
        for grp in grps:
            runs = np.array([q.sense_gradient(c, R, grp)[grp] for _ in range(REP)])
            percomp.extend(runs.var(axis=0, ddof=1).tolist())
            gnorm[grp] = runs.mean(axis=0)
        V = float(np.mean(percomp))
        gn = float(np.linalg.norm(gnorm))
        Vs.append(V); ns.append(n); GN.append(gn)
        w = k + n + N
        print(f"  {n:>4}{len(grps):>10}{w:>8}{V:>16.6f}"
              f"{len(grps) * V:>18.6f}{gn:>10.4f}"
              f"{time.time()-t0:>6.0f}s", flush=True)
    ns = np.array(ns, float); Vs = np.array(Vs); GN = np.array(GN)
    A = np.vstack([np.ones_like(ns), ns - 1.0]).T
    (a, b), *_ = np.linalg.lstsq(A, Vs, rcond=None)
    results[tag] = (a, b, Vs, ns, GN)
    print(f"  fit V(n) = a + b(n-1):   a = {a:.6f}   b = {b:.6f}"
          f"   b/a = {b/max(a,1e-18):.4f}")
    if b < a:
        print(f"  => b < a, so GLOBAL is cheaper in total shots, by up to "
              f"{M*a/(a+b*(M-1)):.1f}x over fully-layered")
    else:
        print(f"  => b >= a, so LAYERED is cheaper; global costs "
              f"{(a+b*(M-1))/(M*a):.2f}x fully-layered")
    print(f"  predicted C(1)={M*a:.6f}  C(M={M})={a+b*(M-1):.6f}")

print()
print("=" * 84)
print("Parameter-shift baseline, same units")
print("=" * 84)
est = AerEstimator()
prec = 1.0 / np.sqrt(S)

def ps_grad():
    pubs = []
    for i in range(M):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        pubs.append((ansatz, H, pp)); pubs.append((ansatz, H, pm))
    r = est.run(pubs, precision=prec).result()
    e = np.array([float(r[j].data.evs) for j in range(2 * M)])
    return 0.5 * (e[0::2] - e[1::2])

t0 = time.time()
runs = np.array([ps_grad() for _ in range(REP)])
V_ps = float(np.mean(runs.var(axis=0, ddof=1)))
g_ps = runs.mean(axis=0)
C_ps = 2 * M * V_ps
print(f"  circuits per gradient : {2*M}")
print(f"  V_ps per component    : {V_ps:.6f}")
print(f"  C_ps = 2M * V_ps      : {C_ps:.6f}")
print(f"  |g_exact|             : {np.linalg.norm(g_ps):.4f}   ({time.time()-t0:.0f}s)")

print()
print("=" * 84)
print("VERDICT: shots to reach the same RELATIVE gradient precision")
print("=" * 84)
print("  cost / |g|^2 is the shot count for equal relative accuracy;")
print("  lower is cheaper. circuits column is the other axis.")
print()
print(f"  {'method':<22}{'circuits':>10}{'cost':>12}{'|g|':>9}"
      f"{'cost/|g|^2':>13}{'vs par-shift':>14}")
print("  " + "-" * 80)
base = C_ps / np.linalg.norm(g_ps) ** 2
print(f"  {'parameter-shift':<22}{2*M:>10}{C_ps:>12.6f}"
      f"{np.linalg.norm(g_ps):>9.4f}{base:>13.6f}{1.0:>14.2f}")
for tag, (a, b, Vs, ns, GN) in results.items():
    for n, V, gn in zip(ns.astype(int), Vs, GN):
        nc = int(np.ceil(M / n))
        Cq = nc * V
        rel = Cq / max(gn ** 2, 1e-18)
        print(f"  {tag + ' n=' + str(n):<22}{nc:>10}{Cq:>12.6f}"
              f"{gn:>9.4f}{rel:>13.6f}{rel / base:>14.2f}")
    Cg = a + b * (M - 1)
    gn = float(GN[-1])   # widest measured norm; smeared norm is ~flat in n
    rel = Cg / max(gn ** 2, 1e-18)
    print(f"  {tag + ' n=M pred':<22}{1:>10}{Cg:>12.6f}"
          f"{gn:>9.4f}{rel:>13.6f}{rel / base:>14.2f}")
print()
print("  NOTE the |g| columns: QLTO targets the SMEARED gradient, whose norm is")
print("  0.79-0.93 of the exact one at R=0.6, so a like-for-like relative")
print("  comparison must divide each method's cost by its OWN |g|^2. The raw")
print("  cost column is only comparable within a method.")
