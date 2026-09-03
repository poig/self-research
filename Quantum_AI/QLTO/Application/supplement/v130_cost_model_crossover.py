"""When does QLTO's circuit saving actually pay? The break-even overhead, measured.

Every advantage claim in this project is a CIRCUIT-COUNT claim bought with SHOTS:
v109 measured ~4x more shots for 32x fewer circuits. Whether that is a good trade
depends entirely on the relative cost of a circuit and a shot, and that ratio is
a property of the MACHINE, not of the algorithm.

    total cost  =  r_circ * (number of circuits)  +  r_shot * (number of shots)

On a cloud QPU r_circ is enormous - queue time, compilation, waveform upload,
calibration drift between jobs - and dominates. On a hypothetical local
accelerator (photonic, NV-diamond, anything sitting inside the machine rather
than behind a queue) r_circ collapses toward the cost of reloading a pulse
sequence, and then the comparison is decided by SHOTS ALONE.

  THAT INVERTS THE VERDICT. If r_circ -> 0 the trade becomes 4x more shots for a
  saving worth nothing, and QLTO LOSES. The advantage is largest in exactly the
  regime people want to escape, and shrinks in exactly the regime they are
  building toward. This file measures where the crossover sits.

THE BASELINE IS PARAMETER-SHIFT ON THE SAME DATA REGISTER, not naive per-sample
parameter-shift. Both arms get the weighted register; only the gradient method
differs, so the V6 contribution is isolated:

    qlto_qml    1 f_hat  +  2 branches x G      =  3 circuits    (G=1, QML)
    PS-on-reg   1 f_hat  +  2 branches x 2M     =  1 + 4M circuits

At M=12 that is 3 against 49. The naive 2M|D| = 384 figure quoted elsewhere in
this project compares against a baseline nobody would run once the data register
exists, and overstates the advantage by |D|/2.

WHAT IS MEASURED. Shots are swept on each arm until the two reach the SAME cos
against the exact gradient. Then the cost model is solved for the break-even
r_circ / r_shot, reported in shot-equivalents: the per-circuit overhead, measured
in units of one shot, above which QLTO is the cheaper way to get that accuracy.

TIER (project rule R1): tier A for everything measured - circuits on
AerSimulator with finite shots, counted through qlto_qml's own instrumented
q.ncircuits and q.nshots. The exact gradient is tier B and is the reference. The
break-even arithmetic is algebra on measured counts, and the conversion to
seconds at the end is EXPLICITLY speculative and labelled as such - no hardware
was involved and no vendor timing was consulted.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit.circuit.library import efficient_su2
from qiskit_aer import AerSimulator

from qlto_qml import QLTOQML
from nisq_v6 import QLTOv6

N_SYS = 3
D_QUBITS = 3
EPOCHS = 5
SEEDS = (0, 1, 2)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def setup(seed):
    rng = np.random.default_rng(seed)
    D = 1 << D_QUBITS
    alpha = rng.uniform(-1.0, 1.0, (N_SYS, D_QUBITS))
    core = efficient_su2(N_SYS, reps=1)
    M = core.num_parameters
    probe = QLTOQML(core, alpha, np.zeros(D), shot_budget=4096, sim_seed=1)
    tstar = rng.uniform(-np.pi, np.pi, M)
    y = np.array([probe.f_exact(x, tstar) for x in range(D)])
    theta = rng.uniform(-np.pi, np.pi, M)
    return alpha, core, y, theta, M, D


def arm_qlto(shots, seed):
    """qlto_qml as shipped. Returns (cos, circuits, shots) per epoch, averaged."""
    alpha, core, y, theta, M, D = setup(seed)
    q = QLTOQML(core, alpha, y, shot_budget=shots, sim_seed=400 + seed,
                backend=AerSimulator(seed_simulator=400 + seed))
    cs = []
    for _ep in range(EPOCHS):
        g_true, _ = q.grad_exact(theta)
        c0, s0 = q.ncircuits, q.nshots
        with contextlib.redirect_stdout(io.StringIO()):
            f, _den = q.f_hat(theta)
            g, _ = q.gradient(theta, w=f - y)
        cs.append(_cos(g, g_true))
        nc, ns = q.ncircuits - c0, q.nshots - s0
        theta = theta - 0.3 * g_true / max(np.max(np.abs(g_true)), 1e-12)
    return float(np.mean(cs)), nc, ns


def arm_pshift(shots, seed):
    """Parameter-shift ON THE SAME REGISTER. 1 + 4M circuits per epoch."""
    alpha, core, y, theta, M, D = setup(seed)
    q = QLTOQML(core, alpha, y, shot_budget=shots, sim_seed=500 + seed,
                backend=AerSimulator(seed_simulator=500 + seed))
    cs = []
    for _ep in range(EPOCHS):
        g_true, _ = q.grad_exact(theta)
        with contextlib.redirect_stdout(io.StringIO()):
            f, _den = q.f_hat(theta, shots=shots)
        w = f - y
        nc, ns = 1, shots
        g = np.zeros(M)
        for mask, sgn in ((w > 0, +1.0), (w < 0, -1.0)):
            if not mask.any():
                continue
            pw = np.abs(w) * mask
            Z = pw.sum()
            if Z < 1e-12:
                continue
            anz = q.batched(pw / Z)
            gb = np.zeros(M)
            for i in range(M):
                for sh, sg in ((np.pi / 2, +1.0), (-np.pi / 2, -1.0)):
                    t = np.array(theta, float); t[i] += sh
                    bound = anz.assign_parameters(t)
                    bound.measure_all()
                    from qiskit import transpile
                    tq = transpile(bound, q.backend, optimization_level=1)
                    counts = q.backend.run(tq, shots=shots).result().get_counts()
                    nc += 1; ns += shots
                    tot = sum(counts.values())
                    ev = 0.0
                    for bits, ct in counts.items():
                        b = bits.replace(' ', '')
                        # batched() builds QuantumCircuit(dq, sq), so system
                        # qubit 0 is CIRCUIT qubit D_QUBITS, and get_counts is
                        # little-endian -> string index -(D_QUBITS+1). Writing
                        # b[-N_SYS] is off by one and gave cos 0.08, 0.02, 0.20,
                        # non-monotone in shots - broken, not noisy. Fifth
                        # endianness bug in this project.
                        ev += (1.0 if b[-(D_QUBITS + 1)] == '0' else -1.0) * ct
                    gb[i] += sg * 0.5 * (ev / tot)
            g += sgn * Z * gb
        g *= 2.0 / D
        cs.append(_cos(g, g_true))
        theta = theta - 0.3 * g_true / max(np.max(np.abs(g_true)), 1e-12)
    return float(np.mean(cs)), nc, ns


print("=" * 100)
print("v130  WHEN DOES THE CIRCUIT SAVING PAY?  the break-even overhead")
print("=" * 100)
print("  cost = r_circ * circuits + r_shot * shots. The circuits/shots trade is")
print("  measured; which side wins is a property of the MACHINE. TIER A.")
print("  Baseline is parameter-shift ON THE SAME DATA REGISTER (1+4M circuits),")
print("  not naive per-sample PS (2M|D|) - the latter overstates by |D|/2.")
print()

print("-" * 100)
print("STEP 1  find shot budgets that give the two arms the SAME accuracy")
print("-" * 100)
print("      arm            shots/circ   circuits   total shots   mean cos")
print("   " + "-" * 78)
qrows, prows = [], []
for sh in (1024, 4096, 16384):
    r = [arm_qlto(sh, s) for s in SEEDS]
    c = float(np.mean([x[0] for x in r]))
    qrows.append((sh, r[0][1], r[0][2], c))
    print("      qlto_qml       %7d      %3d       %8d      %+.4f"
          % (sh, r[0][1], r[0][2], c))
for sh in (256, 1024, 4096):
    r = [arm_pshift(sh, s) for s in SEEDS]
    c = float(np.mean([x[0] for x in r]))
    prows.append((sh, r[0][1], r[0][2], c))
    print("      param-shift    %7d      %3d       %8d      %+.4f"
          % (sh, r[0][1], r[0][2], c))
print()

print("-" * 100)
print("STEP 2  the break-even per-circuit overhead")
print("-" * 100)
print("  For a matched pair, QLTO is cheaper iff")
print("      r_circ * (C_ps - C_q)  >  r_shot * (S_q - S_ps)")
print("  i.e.  r_circ / r_shot  >  (S_q - S_ps) / (C_ps - C_q)   [shot-equivalents]")
print()
print("      target cos    qlto (C,S)        p-shift (C,S)      break-even r_circ/r_shot")
print("   " + "-" * 92)
pairs = []
for qs, qc, qsh, qcos in qrows:
    best = min(prows, key=lambda p: abs(p[3] - qcos))
    if abs(best[3] - qcos) > 0.05:
        continue
    ps, pc, psh, pcos = best
    dC, dS = pc - qc, qsh - psh
    if dC <= 0:
        continue
    be = dS / dC
    pairs.append((qcos, qc, qsh, pc, psh, be))
    print("      %+.4f      (%3d, %7d)   (%3d, %7d)      %12.1f shots"
          % (qcos, qc, qsh, pc, psh, be))
print()

print("=" * 100)
print("READING IT")
print("=" * 100)
if not pairs:
    print("  No accuracy-matched pair was found in the sampled budgets, so no")
    print("  break-even can be stated. Widen the sweeps before reading anything")
    print("  into the tables above.")
    sys.exit(0)
be = float(np.mean([p[5] for p in pairs]))
print("  BREAK-EVEN: QLTO is the cheaper route to the same accuracy only when one")
print("  circuit costs more than about %.0f shots." % be)
print()
print("  WHAT THAT MEANS ON REAL MACHINES. A superconducting shot is order 1-100")
print("  microseconds. %.0f shots is therefore roughly %.1f ms to %.1f s of pure"
      % (be, be * 1e-3, be * 1e-4 * 1e3 / 1e3))
print("  shot time. Cloud per-job overhead - queue, compile, waveform upload,")
print("  recalibration - is SECONDS TO MINUTES, so the inequality holds by orders")
print("  of magnitude and QLTO wins comfortably. That is the regime every number")
print("  in this project was measured in.")
print()
print("  AND IT INVERTS ON THE HARDWARE THE QUESTION IS ABOUT. A local accelerator")
print("  - photonic, NV-diamond, anything sitting inside the machine rather than")
print("  behind a queue - exists precisely to drive r_circ toward the cost of")
print("  reloading a pulse sequence, i.e. toward a few shot-equivalents. Below")
print("  %.0f the trade reverses and QLTO is simply paying ~4x the shots for a" % be)
print("  saving worth nothing.")
print()
print("  SO THE ADVANTAGE IS LARGEST IN THE REGIME PEOPLE WANT TO ESCAPE AND")
print("  SMALLEST IN THE ONE THEY ARE BUILDING TOWARD. That is not a reason to")
print("  stop - queues will exist for a long time - but it does mean the")
print("  circuit-count framing has a shelf life, and that any claim resting on it")
print("  should name r_circ/r_shot as the assumption it depends on.")
print()
print("  THIS SAYS NOTHING ABOUT QUANTUM VS CLASSICAL COST. Both are Theta(D) in")
print("  operations on the data axis (Part III), so a cost comparison reduces to")
print("  cost-per-operation, and that is a hardware-physics question this file")
print("  does not touch: it compares two QUANTUM methods on one machine.")
print()
print("  SCOPE. N_sys=%d, |D|=%d, M=12, G=1, %d seeds, %d epochs, no noise model,"
      % (N_SYS, 1 << D_QUBITS, len(SEEDS), EPOCHS))
print("  no hardware, no vendor timings. The microsecond figures above are")
print("  order-of-magnitude context, NOT measurements. Break-even scales with M:")
print("  more parameters means more circuits saved, so the threshold falls and")
print("  QLTO holds up longer as r_circ drops. At M=12 it is what it is.")
