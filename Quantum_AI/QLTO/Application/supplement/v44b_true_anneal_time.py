"""A real anneal-time sweep: scale BOTH totals, not one.

v44 normalised the drift span to pi at every k and left beta per-step, so the
total MIXING angle grew proportional to k while the drift stayed fixed. Degree-1
collapsed from 0.449 to -0.073 - that is a mixer-strength sweep, not an anneal,
and it is the same confound class as v37f.

The anneal identification itself is not in question: gamma = s pi dt ramps UP and
beta = (1-s) pi dt ramps DOWN, which is a Trotterised anneal from the transverse
field to the cost function. That is read off the schedule, not hypothesised. What
was untested is adiabaticity.

Physical anneal time T scales the whole generator: the integrated cost angle and
the integrated transverse angle grow TOGETHER, with k only large enough to keep
Trotter error small. So sweep T with

    span(drift) = T * pi        and        dt_eff = T * dt   (which sets beta)

at fixed k. Larger T is a slower, more adiabatic anneal.

    degree-1   n independent Landau-Zener sweeps, O(1) gap  -> flat in T
    degree-2   transverse-field Ising, needs T >> 1/gap^2    -> should RISE

If degree-2 does not rise here, the anneal reading survives as a DESCRIPTION of
the circuit but yields no usable handle, and degree-1 targeting should be recorded
as a fixed property of this circuit family after nine attempts.
"""
import sys, os, contextlib, io, itertools, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
import nisq_v3
sys.path.insert(0, os.path.join(APP, "supplement"))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12
    from v43_phase_offset import OffsetWalk

R, DT0, SHOTS, K = 0.6, 0.5, 65536, 30
TS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 92)
print("TRUE ANNEAL-TIME SWEEP — both totals scaled together, k fixed")
print("=" * 92)
print(f"  R={R}, k={K}, {SHOTS} shots. T scales the drift span AND dt (hence beta)")
print(f"  together, so the anneal slows instead of the mixer taking over.")
print()
print(f"  {'T':>7}{'mean corr d1':>15}{'mean corr d12':>15}{'enh d1':>10}"
      f"{'enh d12':>10}{'sec':>7}")
print("  " + "-" * 64)

blocks = []
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=64)
    BLK = [b["params"] for b in probe.layers if b["params"]]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)
    for act in BLK:
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for kk, sv in enumerate(sig):
            p = centre.copy()
            p[act] = p[act] + R * sv
            vals[kk] = E(ansatz, H, p)

        def idx_of(x, n=n):
            return int("".join("1" if x[i] > 0 else "0"
                               for i in range(n))[::-1], 2)

        e_by = np.empty(2 ** n)
        e_by[np.array([idx_of(s) for s in sig])] = vals
        blocks.append((ansatz, H, centre, act, n, e_by,
                       idx_of(sig[int(np.argmin(vals))])))

for T in TS:
    t0 = time.time()
    res = {False: [[], []], True: [[], []]}
    for ansatz, H, centre, act, n, e_by, i_true in blocks:
        for use2 in (False, True):
            with contextlib.redirect_stdout(io.StringIO()):
                q = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                               merged_walk=False)
            q.reset_shot_stream()
            g1, g2 = sense_deg12(q, centre, R, act)
            if not use2:
                g2 = np.zeros_like(g2)
            dt = DT0 * T
            acc = 0.5 * np.pi / np.sqrt(R) * np.pi * dt * K / 2.0
            l1 = float(np.sum(np.abs(g1)))
            l2 = float(np.sum(np.abs(np.triu(g2, 1))))
            raw = 2.0 * acc * (l1 + l2)
            if raw > 1e-12:
                nrm = (np.pi * T) / raw
                g1, g2 = g1 * nrm, g2 * nrm
            counts = q.walk(centre, K, dt, R, act, g1, g2, use2, 0.0)
            sel = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == "1":
                    sel[int(parts[1].replace(" ", ""), 2)] += c
            sel = sel / max(sel.sum(), 1)
            res[use2][0].append(float(np.corrcoef(sel, -e_by)[0, 1]))
            res[use2][1].append(sel[i_true] * 2 ** n)
    print(f"  {T:>7.2f}{np.mean(res[False][0]):>15.4f}"
          f"{np.mean(res[True][0]):>15.4f}{np.mean(res[False][1]):>10.3f}"
          f"{np.mean(res[True][1]):>10.3f}{time.time() - t0:>7.0f}", flush=True)

print()
print("  degree-2 rising with T is the prediction. Both flat, or both falling,")
print("  means the anneal reading describes the circuit but gives no handle.")
