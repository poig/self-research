"""The only cost question that matters: whose complexity GROWS slower?

Every cost number so far has been a constant factor at one problem size, and
constants are not the claim. The claim has always been that QLTO's per-epoch cost
is FLAT in the parameter count M and, with QPE, flat in the commuting-group count
G as well - while parameter-shift pays 2*M*G circuits, growing in both. If that
is true the advantage widens without bound and the depth penalty is a fixed toll
paid once. If it is false the whole thing is a constant-factor trick.

So: sweep N, fit the exponent, and let the slopes decide. Three methods:

  parameter-shift   2*M*G circuits, ansatz depth
  QLTO V3 (QPE)     2 circuits per block, flat in M and G, but the controlled
                    U^(2^a) ladder makes each one deep and the depth itself
                    grows with the term count of H
  QLTO V4 (direct)  (G+1) circuits per block - it TRADES BACK the G-independence
                    that QPE bought, in exchange for ansatz-scale depth

Four currencies, because they scale differently and the answer may differ by
currency:

  circuits        per epoch (QLTO) or per gradient (parameter-shift)
  depth/circuit   the coherence constraint - a hard wall, not a cost
  total 2q gates  circuits * gates, the error budget and the QPU-time proxy
  classical ms    build + decode, which is where I was told to expect QLTO to
                  win at scale and where v15 already showed decode is cheaper

Fitted as log(cost) ~ alpha * log(N); alpha is the growth exponent and the
smallest alpha wins asymptotically whatever the constant.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import benchmark as B
import nisq_v3
from supplement.v18_v4_direct_readout import build_direct, group_basis

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']
SIZES = (4, 6, 8, 10)
QPE_MAX_N = 8          # transpiling the controlled ladder past this is minutes


def tstats(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return t.depth(), t.count_ops().get('cx', 0)


rows = []
print("=" * 104)
print("COMPLEXITY GROWTH — Heisenberg chain, N = 4, 6, 8, 10")
print("=" * 104)

for N in SIZES:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = B.get_heisenberg_problem(N)
    q = Q(ansatz, H, shot_budget=1024)
    M = ansatz.num_parameters
    G = len(H.group_commuting(qubit_wise=True))
    T = len(H.paulis)
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, M)
    blocks = [b['params'] for b in q.layers if b['params']]
    Bk = len(blocks)

    # ---- parameter-shift: 2*M*G circuits at ansatz depth
    a = ansatz.assign_parameters(c); a.measure_all()
    t0 = time.perf_counter(); d_ps, cx_ps = tstats(a)
    build_ps = (time.perf_counter() - t0) * 2 * M * G
    n_ps = 2 * M * G

    # ---- QLTO V4 direct: (G+1) circuits per block  [G sensing + 1 walk]
    t0 = time.perf_counter()
    d4, cx4, n4 = [], [], 0
    groups = H.group_commuting(qubit_wise=True)
    for act in blocks:
        for g in groups:
            d, cx = tstats(build_direct(q, c, 0.6, act, g))
            d4.append(d); cx4.append(cx); n4 += 1
    build_4 = time.perf_counter() - t0

    # ---- QLTO V3 QPE: 2 circuits per block, flat in M and G
    if N <= QPE_MAX_N:
        t0 = time.perf_counter()
        d3, cx3 = [], []
        for act in blocks:
            d, cx = tstats(q._build_qpe_sensing_circuit(c, 0.6, act))
            d3.append(d); cx3.append(cx)
        build_3 = time.perf_counter() - t0
        n3 = 2 * Bk
        # the walk circuit is the second of the pair; it is NOT a QPE ladder
        tot3, dep3 = sum(cx3) * 2, float(np.mean(d3))
    else:
        n3, tot3, dep3, build_3 = np.nan, np.nan, np.nan, np.nan

    rows.append(dict(N=N, M=M, G=G, T=T, B=Bk,
                     n_ps=n_ps, d_ps=d_ps, cx_ps=n_ps * cx_ps, b_ps=build_ps * 1e3,
                     n3=n3, d3=dep3, cx3=tot3, b3=build_3 * 1e3,
                     n4=n4 + Bk, d4=float(np.mean(d4)), cx4=sum(cx4), b4=build_4 * 1e3))
    r = rows[-1]
    print(f"\n  ===== N={N} | M={M} | G={G} | H terms={T} | blocks={Bk} =====")
    print(f"  {'method':<20}{'circuits':>9}{'depth':>8}{'total 2q':>10}{'build ms':>10}")
    print("  " + "-" * 57)
    print(f"  {'parameter-shift':<20}{r['n_ps']:>9}{r['d_ps']:>8}"
          f"{r['cx_ps']:>10}{r['b_ps']:>10.1f}")
    if not np.isnan(r['n3']):
        print(f"  {'QLTO V3 (QPE)':<20}{int(r['n3']):>9}{int(r['d3']):>8}"
              f"{int(r['cx3']):>10}{r['b3']:>10.1f}")
    print(f"  {'QLTO V4 (direct)':<20}{int(r['n4']):>9}{int(r['d4']):>8}"
          f"{r['cx4']:>10}{r['b4']:>10.1f}", flush=True)


def alpha(key, mask=None):
    ns = np.array([r['N'] for r in rows], float)
    ys = np.array([r[key] for r in rows], float)
    ok = ~np.isnan(ys) & (ys > 0)
    if mask is not None:
        ok &= mask
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(ns[ok]), np.log(ys[ok]), 1)[0])


print("\n" + "=" * 104)
print("GROWTH EXPONENTS   cost ~ N^alpha   (smaller wins asymptotically)")
print("=" * 104)
print(f"  {'method':<20}{'circuits':>12}{'depth':>10}{'total 2q':>12}{'build ms':>12}")
print("  " + "-" * 66)
print(f"  {'parameter-shift':<20}{alpha('n_ps'):>12.2f}{alpha('d_ps'):>10.2f}"
      f"{alpha('cx_ps'):>12.2f}{alpha('b_ps'):>12.2f}")
print(f"  {'QLTO V3 (QPE)':<20}{alpha('n3'):>12.2f}{alpha('d3'):>10.2f}"
      f"{alpha('cx3'):>12.2f}{alpha('b3'):>12.2f}")
print(f"  {'QLTO V4 (direct)':<20}{alpha('n4'):>12.2f}{alpha('d4'):>10.2f}"
      f"{alpha('cx4'):>12.2f}{alpha('b4'):>12.2f}")

print()
print("  Heisenberg holds G=3 at every N, so this sweep CANNOT separate QPE's")
print("  G-independence from V4's G-linearity. That takes a family whose group")
print("  count grows:")
for nm, fn in (("H2", B.get_h2_problem), ("LiH", B.get_lih_problem)):
    with contextlib.redirect_stdout(io.StringIO()):
        an, HH, _ = fn()
    print(f"    {nm:<5} {an.num_qubits:>2} qubits, {len(HH.paulis):>4} terms -> "
          f"G = {len(HH.group_commuting(qubit_wise=True)):>3}   "
          f"parameter-shift {2*an.num_parameters*len(HH.group_commuting(qubit_wise=True)):>5} circuits, "
          f"V4 {len([b for b in Q(an, HH, shot_budget=1024).layers if b['params']])*(len(HH.group_commuting(qubit_wise=True))+1):>4}, "
          f"V3 {2*len([b for b in Q(an, HH, shot_budget=1024).layers if b['params']]):>3}")
