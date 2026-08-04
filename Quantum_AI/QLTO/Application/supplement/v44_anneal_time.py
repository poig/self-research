"""The walk is an ANNEAL, not a Grover search. So sweep the anneal time.

Seven hypotheses for the degree-2 failure have been falsified (v39, v39b, v41b,
v41c, v42b, v42c, v43). This one is not proposed, it is read off the schedule:

    gamma = s * pi * dt        s = (step+0.5)/k     INCREASES  0 -> pi dt
    beta  = (1-s) * pi * dt                         DECREASES  pi dt -> 0

Alternating exp(-i gamma Phi_diag / 2) with exp(-i beta sum_i X_i / 2) while gamma
ramps up and beta ramps down is a TROTTERISED QUANTUM ANNEAL from the transverse
field to the cost function, on the parameter hypercube, wrapped in a Hadamard
test. The Grover framing used throughout these notes is the wrong analogy.

That accounts for the degree-1/degree-2 asymmetry with no new assumption:

    degree-1   H(s) = sum_i [ gamma phi_i Z_i + beta X_i ]
               n INDEPENDENT two-level systems. Each qubit runs its own
               Landau-Zener sweep with an O(1) gap, so any k suffices and every
               qubit lands on its local optimum - which IS the degree-1 argmin,
               and is exactly what v38/v39c measured the walk amplifying.

    degree-2   H(s) = sum gamma phi_i Z_i + sum gamma phi_ij Z_i Z_j
                      + beta sum_i X_i
               the transverse-field Ising model. Not separable, not integrable.
               The adiabatic theorem needs T >> 1/gap_min^2; at k=15 this is a
               QUENCH, which scrambles rather than concentrates. Adding pairwise
               terms therefore makes the oracle WORSE, which is what v42, v42b
               and v43 all found.

THE PREDICTION, and it is falsifiable in one sweep: with the TOTAL phase held
fixed - so that raising k slows the anneal instead of enlarging the angle -

    degree-1    flat in k        (already adiabatic at any k)
    degree-2    improves with k  (approaching adiabatic)

Holding the total phase fixed is essential and is why this is not just a repeat
of the k sweep in v4_schedule: there, raising k raised the accumulated angle
proportionally, so more steps meant more WRAP rather than a slower anneal. Here
the span is normalised to pi at every k, exactly as in v42b.

If degree-2 rises with k and degree-1 does not, the obstruction is anneal time,
the required time is set by the induced TFIM's minimum gap, and that gap is a
property of the Hamiltonian and ansatz - which makes the design problem-based in
a precise sense rather than a hopeful one.
"""
import sys, os, contextlib, io, itertools, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12
    from v43_phase_offset import OffsetWalk

R, DT, SHOTS, SPAN = 0.6, 0.5, 65536, np.pi
KS = [5, 15, 30, 60, 120]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 96)
print("ANNEAL TIME — degree-2 needs adiabaticity, degree-1 does not")
print("=" * 96)
print(f"  R={R}, dt={DT}, {SHOTS} shots. Phase SPAN normalised to pi at EVERY k,")
print(f"  so raising k slows the anneal rather than enlarging the angle. That is")
print(f"  the difference from the k sweep in v4_schedule, where it did both.")
print()
print(f"  {'problem':>15}{'blk':>4}{'d1ok':>6}{'k':>6}{'corr d1':>10}"
      f"{'corr d12':>10}{'enh d1':>9}{'enh d12':>9}{'sec':>7}")
print("  " + "-" * 76)

agg = {}
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=64)
    BLK = [b['params'] for b in probe.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for kk, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[kk] = E(ansatz, H, p)
        d1e = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)])
        d1ok = bool(np.all(np.where(d1e <= 0, 1.0, -1.0)
                           == sig[int(np.argmin(vals))]))

        def idx_of(x):
            return int(''.join('1' if x[i] > 0 else '0'
                               for i in range(n))[::-1], 2)
        i_true = idx_of(sig[int(np.argmin(vals))])
        e_by_idx = np.empty(2 ** n)
        e_by_idx[np.array([idx_of(s) for s in sig])] = vals

        for k in KS:
            t0 = time.time()
            out = {}
            for use2 in (False, True):
                with contextlib.redirect_stdout(io.StringIO()):
                    q = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                                   merged_walk=False)
                q.reset_shot_stream()
                g1, g2 = sense_deg12(q, centre, R, act)
                if not use2:
                    g2 = np.zeros_like(g2)
                # ACC depends on k, so the normalisation must be recomputed
                acc = 0.5 * np.pi / np.sqrt(R) * np.pi * DT * k / 2.0
                l1 = float(np.sum(np.abs(g1)))
                l2 = float(np.sum(np.abs(np.triu(g2, 1))))
                raw = 2.0 * acc * (l1 + l2)
                if raw > 1e-12:
                    nrm = SPAN / raw
                    g1, g2 = g1 * nrm, g2 * nrm
                counts = q.walk(centre, k, DT, R, act, g1, g2, use2, 0.0)
                sel = np.zeros(2 ** n)
                for bs, c in counts.items():
                    parts = bs.split()
                    if len(parts) == 2 and parts[0][-1] == '1':
                        sel[int(parts[1].replace(" ", ""), 2)] += c
                sel = sel / max(sel.sum(), 1)
                out[use2] = (float(np.corrcoef(sel, -e_by_idx)[0, 1]),
                             sel[i_true] * 2 ** n)
            agg.setdefault((k, False), []).append(out[False][0])
            agg.setdefault((k, True), []).append(out[True][0])
            print(f"  {name if k == KS[0] else '':>15}{bi if k == KS[0] else '':>4}"
                  f"{('Y' if d1ok else 'n') if k == KS[0] else '':>6}{k:>6}"
                  f"{out[False][0]:>10.4f}{out[True][0]:>10.4f}"
                  f"{out[False][1]:>9.3f}{out[True][1]:>9.3f}"
                  f"{time.time() - t0:>7.0f}", flush=True)
        print("  " + "." * 76)

print(f"\n  {'k':>6}{'mean corr d1':>15}{'mean corr d12':>15}{'d12 - d1':>11}")
print("  " + "-" * 47)
for k in KS:
    a = np.array(agg[(k, False)]); b = np.array(agg[(k, True)])
    print(f"  {k:>6}{a.mean():>15.4f}{b.mean():>15.4f}{b.mean() - a.mean():>11.4f}")

print()
print("  'd12 - d1' rising toward zero and beyond as k grows is the prediction.")
print("  Flat in k for BOTH would falsify the anneal reading and leave the")
print("  degree-2 failure genuinely unexplained after eight attempts, which is")
print("  itself worth recording - at that point the honest move is to stop")
print("  proposing mechanisms and treat degree-1 targeting as a fixed property")
print("  of this circuit family.")
