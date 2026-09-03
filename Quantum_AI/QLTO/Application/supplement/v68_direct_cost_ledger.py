"""Do the depth and classical-overhead penalties survive the DIRECT readout?

Two numbers have been quoted against QLTO all through these notes:

  DEPTH        v16: 19-141x parameter-shift, 2q gates 5-40x worse, "a depth-1976
               circuit on six qubits does not fit inside any current device's
               coherence, so at the shipping default this is not a NISQ method".
  CLASSICAL    v15: 40-109x worse, 535-753 ms per gradient, essentially all of it
               BUILD - "QLTO rebuilds and re-transpiles the whole sensing circuit
               every gradient because the W gate is constructed with the current
               parameters baked in as concrete angles."

BOTH WERE MEASURED ON THE QPE PATH, AND BOTH HAVE SINCE BEEN ADDRESSED.

  * The depth figure is the 2^a Trotter ladder, which the direct readout does not
    have. The notes already record the split without folding it back into the
    verdict: "depth = ansatz + O(1) holds for direct readout (offset 7-11) and
    fails for QPE (274 -> 658 across N)". The billing projection carries the same
    split - depth N+8 for parameter-shift, 408N-472 for QPE, N+15 for direct.
  * v15 named the build fix exactly - "the sensing circuit can be built once as a
    parameterised template and bound each epoch" - and marked it UNTESTED. It is
    now implemented: nisq_v5._direct_template builds the circuit with Parameter
    objects for both the angles and the radius, transpiles ONCE, and caches by
    (block, group). So every gradient after the first should pay binding cost,
    not transpilation cost.

Neither has been re-measured, and the ledger has gone on quoting the QPE numbers
for a default that is now direct. This fixes that.

WHAT IS MEASURED, all transpile-only - no shots, no simulation, nothing that
needs a backend to execute:

  depth / 2q gates   per circuit AND summed per gradient, basis {rz,sx,x,cx} at
                     optimisation level 1, matching v16 exactly so the numbers
                     are comparable to the ones they replace. Summed per gradient
                     is the honest total: QLTO runs 2G(r+1) circuits where
                     parameter-shift runs 2MG, so a per-circuit penalty and a
                     per-gradient total can point OPPOSITE WAYS.
  build ms           cold (empty cache, first gradient) against warm (every
                     subsequent one). The gap is the transpilation the template
                     cache is supposed to remove.

DECODE IS NOT IN QUESTION and is not re-measured: v15 found QLTO CHEAPER there on
every problem (1.1 ms against 3.4 ms), because reading n marginals out of one
shot table beats averaging 2MG separate expectation values - the T1 batching
argument showing up in CPU time.

PREDICTION, stated first. Per circuit the direct path should cost the ansatz plus
a handful of layers for the W gate's controlled rotations. Per GRADIENT it should
come out AHEAD of parameter-shift on both depth and 2q gates, because 2N fewer
circuits more than pays for ~2n extra controlled rotations in each. If that holds,
"not a NISQ method" is a statement about the QPE path only.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
import nisq_v5

BASIS = ['rz', 'sx', 'x', 'cx']
OPT = 1


def heis(N):
    o = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def maxcut(N):
    o = []
    for i in range(N):
        j = (i + 1) % N
        s = ["I"] * N
        s[i] = s[j] = "Z"
        o.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(o)


def h2():
    return SparsePauliOp.from_list([
        ("II", -1.0523), ("IZ", 0.3979), ("ZI", -0.3979),
        ("ZZ", -0.0113), ("XX", 0.1809)])


PROBS = [('H2', 2, h2()),
         ('MaxCut N=4', 4, maxcut(4)),
         ('Heisenberg N=4', 4, heis(4)),
         ('Heisenberg N=6', 6, heis(6))]

print("=" * 104)
print("THE DIRECT-READOUT COST LEDGER — replacing v15/v16's QPE numbers")
print("=" * 104)
print(f"  basis {BASIS}, optimisation_level {OPT}, one gradient.")
print(f"  'per circ' is one transpiled circuit; 'per grad' multiplies by the")
print(f"  circuits each method needs, which is where the 2N shows up.")
print()
print(f"  {'problem':>16}{'M':>4}{'G':>3}{'circ QL':>8}{'circ PS':>8}"
      f"{'dep QL':>8}{'dep PS':>8}{'2q QL':>7}{'2q PS':>7}"
      f"{'DEPTH/grad':>12}{'2Q/grad':>10}")
print("  " + "-" * 96)

for name, N, H in PROBS:
    ansatz = efficient_su2(N, reps=2)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct')
    G = len(q.groups)
    blocks = [b['params'] for b in q.layers if b['params']]

    # ---- QLTO: one transpiled template per (block, group) -------------------
    # _direct_template transpiles to the AER backend, whose basis keeps ry/cry as
    # single gates - that would make QLTO look shallow AND hide its controlled
    # rotations from a 'cx' count. Re-transpile to the HARDWARE basis so both
    # arms are measured in the same currency.
    q._direct_template_cache.clear()
    dq, gq = [], []
    for act in blocks:
        for grp in q.groups:
            t, _, _ = q._direct_template(act, grp)
            t = transpile(t, basis_gates=BASIS, optimization_level=OPT)
            dq.append(t.depth())
            gq.append(sum(v for k, v in t.count_ops().items() if k == 'cx'))
    cq = len(dq)

    # ---- parameter-shift: transpile the ansatz once per group ---------------
    dp, gp = [], []
    for grp in q.groups:
        qc = ansatz.copy()
        q._basis(qc, qc.qubits, grp)
        qc.measure_all()
        t = transpile(qc, basis_gates=BASIS, optimization_level=OPT)
        dp.append(t.depth())
        gp.append(sum(v for k, v in t.count_ops().items() if k == 'cx'))
    cp = 2 * M * G
    # each group's circuit is run for 2M shifts
    dep_ps = float(np.mean(dp))
    g2_ps = float(np.sum(gp)) * 2 * M

    dep_ql = float(np.mean(dq))
    g2_ql = float(np.sum(gq))

    dsum_ql, dsum_ps = dep_ql * cq, dep_ps * cp
    print(f"  {name:>16}{M:>4}{G:>3}{cq:>8}{cp:>8}{dep_ql:>8.0f}{dep_ps:>8.0f}"
          f"{g2_ql / cq:>7.0f}{g2_ps / cp:>7.0f}"
          f"{dsum_ps / dsum_ql:>11.2f}x{g2_ps / g2_ql:>9.2f}x", flush=True)

print()
print("  DEPTH/grad and 2Q/grad are PS/QLTO — above 1.0 means QLTO uses LESS in")
print("  total for one gradient. v16 reported the per-circuit ratio on the QPE")
print("  path and read it as a penalty; per gradient on the direct path is the")
print("  number that decides whether the circuits fit a device's coherence budget")
print("  in aggregate.")
print()
print("=" * 104)
print("  BUILD COST: is v15's 535-753 ms artefact removed by the template cache?")
print("=" * 104)
print(f"  Cold = empty cache (first gradient, pays transpilation).")
print(f"  Warm = median of the next 5 (should pay binding only).")
print(f"  v15 measured, on the QPE path: H2 553.8 / MaxCut 535.1 / Heis N=4 753.0")
print()
print(f"  {'problem':>16}{'cold ms':>10}{'warm ms':>10}{'speedup':>10}"
      f"{'v15 ms':>9}{'vs v15':>9}")
print("  " + "-" * 64)

V15 = {'H2': 553.8, 'MaxCut N=4': 535.1, 'Heisenberg N=4': 753.0}
for name, N, H in PROBS:
    ansatz = efficient_su2(N, reps=2)
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v5.QLTOv5(ansatz, H, shot_budget=1024, gradient_mode='direct')
    blocks = [b['params'] for b in q.layers if b['params']]
    centre = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)

    def one_gradient():
        t0 = time.perf_counter()
        for act in blocks:
            for grp in q.groups:
                t, th, rad = q._direct_template(act, grp)
                t.assign_parameters(
                    {**{th[i]: centre[i] for i in range(len(centre))},
                     rad: 0.6}, inplace=False)
        return (time.perf_counter() - t0) * 1e3

    q._direct_template_cache.clear()
    cold = one_gradient()
    warm = float(np.median([one_gradient() for _ in range(5)]))
    ref = V15.get(name)
    print(f"  {name:>16}{cold:>10.1f}{warm:>10.1f}{cold / warm:>9.1f}x"
          + (f"{ref:>9.1f}{ref / warm:>8.1f}x" if ref else f"{'-':>9}{'-':>9}"),
          flush=True)

print()
print("  A warm cost of a few ms would retire the '40-100x worse classical")
print("  overhead' row for the direct path: v15 already found DECODE cheaper for")
print("  QLTO (1.1 ms vs 3.4 ms), and build was the entire remaining gap.")
