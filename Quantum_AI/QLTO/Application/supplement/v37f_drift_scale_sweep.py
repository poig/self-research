"""The cheap fix, on the real circuit: rescale the drift so it stops wrapping.

v37b/c/d/e established, in exact arithmetic validated against the simulator at
0.00241, that the shipped walk's drift angle accumulates as

    sum_s al_i = g_i * (pi dt k / 2) * 0.5 pi / sqrt(R)  ~  23.9 g_i

so its Bloch response is PERIODIC in g and wraps ~4 times over the gradient range
the benchmark actually produces. v37e compared two fixes on the bare model:

    rescale   divide the drift by 23.9/pi. 0 sign crossings, 0 turns,
              corr(d,g) = 0.98, knee at |g| = 1.28 - inside the operating range,
              so magnitude information SURVIVES. Costs one constant.
    reset     fresh ancilla per step. 0 crossings, better separability (0.067 vs
              0.147), but the knee falls to 0.26, which is bounded SIGN descent -
              magnitude-blind. Costs a mid-circuit reset and k energy imprints.

The bare model cannot settle which fix moves DOWNHILL, because v37c measured that
the energy imprint flips signs and the bare model omits it. That has to be run.

This file runs the rescale on the SHIPPED circuit, and it needs no new code: the
walk consumes the gradient only as a drift rate, so passing g*scale into
_execute_walk IS the rescale, exactly. Everything else - sensing, decode, W gate,
imprint, schedule - is untouched, so the sweep isolates one number.

scale = 1.0 is the shipped walk. scale = 0.1315 is v37e's prescription. The
intermediate values are there because the prescription was derived on the bare
model, and v37c says the imprint contributes more to the VALUES than the bare
model does - so the optimum on the real circuit need not sit where the bare model
puts it. Reporting the whole curve rather than the predicted point is the
difference between testing a prediction and confirming one.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


def E(ansatz, H, p):
    return float(np.real(Statevector(ansatz.assign_parameters(p)).expectation_value(H)))


N, R, DT, KS, SHOTS, EPOCHS, SEEDS = 4, 0.6, 0.5, 15, 8192, 20, 4
H = heis(N)
exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
ansatz = efficient_su2(N, reps=1)
M = ansatz.num_parameters
PRESCRIBED = np.pi / (np.pi * DT * KS / 2 * 0.5 * np.pi / np.sqrt(R))

print("=" * 96)
print(f"DRIFT SCALE SWEEP — the cheap fix on the shipped circuit. Heisenberg "
      f"N={N}, exact {exact:.4f}")
print("=" * 96)
print(f"  {EPOCHS} epochs, {SHOTS} shots, {SEEDS} seeds, reps=1 "
      f"(ansatz ceiling ~ -6.12).")
print(f"  scale=1.0 is shipped. v37e's bare-model prescription is "
      f"{PRESCRIBED:.4f}.")
print(f"  Sensing, decode, W gate, imprint and schedule are all untouched.")
print()
print(f"  {'scale':>8}{'E_final':>11}{'sigma':>9}{'best':>10}{'E@3':>10}"
      f"{'E@5':>10}{'E@10':>10}{'|g| mean':>10}{'sec':>7}")
print("  " + "-" * 78)

SCALES = [1.0, 0.5, 0.25, PRESCRIBED, 0.08, 0.04]
rows = []
for sc in SCALES:
    t0 = time.time()
    fin, e3, e5, e10, gn = [], [], [], [], []
    for sd in range(SEEDS):
        with contextlib.redirect_stdout(io.StringIO()):
            q = nisq_v3.QLTOv3(ansatz, H, shot_budget=SHOTS, sim_seed=5 + sd)
        q.reset_shot_stream()
        BLK = [b['params'] for b in q.layers if b['params']]
        p = np.random.RandomState(42 + sd).uniform(-np.pi, np.pi, M)
        for ep in range(EPOCHS):
            r = max(R * (0.9 ** ep), 1e-4)
            dt = max(DT * (0.95 ** (ep + 1)), 0.01)
            for act in BLK:
                g = q.sense_gradient(p, r, act)
                gn.append(float(np.linalg.norm(g[act])))
                # The walk uses the gradient ONLY as a drift rate, so this is
                # exactly a rescale of the accumulated angle - no other effect.
                p = q._execute_walk(p, KS, dt, r, act, g * sc)
            if ep == 2:
                e3.append(E(ansatz, H, p))
            if ep == 4:
                e5.append(E(ansatz, H, p))
            if ep == 9:
                e10.append(E(ansatz, H, p))
        fin.append(E(ansatz, H, p))
    rows.append((sc, np.mean(fin), np.std(fin)))
    tag = "  <- shipped" if sc == 1.0 else ("  <- predicted" if sc == PRESCRIBED
                                            else "")
    print(f"  {sc:>8.4f}{np.mean(fin):>11.4f}{np.std(fin):>9.4f}"
          f"{np.min(fin):>10.4f}{np.mean(e3):>10.4f}{np.mean(e5):>10.4f}"
          f"{np.mean(e10):>10.4f}{np.mean(gn):>10.4f}{time.time() - t0:>7.0f}"
          f"{tag}", flush=True)

best = min(rows, key=lambda r: r[1])
ship = [r for r in rows if r[0] == 1.0][0]
print(f"\n  best scale {best[0]:.4f} at {best[1]:.4f} +- {best[2]:.4f}")
print(f"  shipped      1.0000 at {ship[1]:.4f} +- {ship[2]:.4f}")
print(f"  difference   {best[1] - ship[1]:+.4f}")
print()
print("  This harness's spurious-difference scale is 0.03-0.09 (the kappa=4")
print("  term-sort null in RESEARCH_NOTES, where a provably identical unitary")
print("  returned up to 3.3 sigma), so read anything under ~0.1 as no result.")
print("  A win at small scale means the wrap was costing real energy. A flat")
print("  curve means the wrap is real but the schedule already absorbs it, and")
print("  the aliasing matters for EXPLANATION rather than for performance.")
