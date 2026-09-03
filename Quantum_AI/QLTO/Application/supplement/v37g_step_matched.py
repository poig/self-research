"""The wrap fixes, compared AT MATCHED STEP SIZE. The previous two runs were not.

Both earlier results are confounded by step size, and this project has been caught
by that exact confound before.

  v37f swept the drift scale at FIXED dt and found monotone collapse below 0.25.
  But scaling the drift down also shrinks the step: v37e measured |d| at |g|=0.6
  as 0.417 for base and 0.055 for the rescale. So v37f compared "same schedule,
  7.6x smaller steps" over a fixed 20 epochs. That is not a test of the fix. The
  schedule entry in RESEARCH_NOTES warned about precisely this - "comparing the
  two at fixed dt is the same step-size confound that made the raw natural
  gradient look bad until natural_norm - normalising simply steps less far" - and
  v37f reproduced the error anyway.

  v37 found reset_cheap 0.52 AHEAD of base at epoch 3. But the closed form gives
  reset a LARGER maximum step than base, 0.558 against 0.417, so part of that
  lead may also be step size rather than the removal of the wrap.

So sweep dt per arm and report the mean per-coordinate displacement alongside the
energy. Two arms are comparable only where their |move| columns agree; the right
comparison is each arm at ITS OWN best dt, and additionally at matched |move|.

  base      shipped walk, drift scale 1.0
  rescale   drift scaled by pi/23.9 = 0.1315, wrap removed, post-selection kept
  reset     fresh ancilla per step, one imprint (the cheap variant that won early)

REPORTED
  E_final   mean over seeds
  E@3       early convergence, where reset's advantage appeared
  |move|    mean per-coordinate |delta_theta| over the whole run - THE CONTROL
  best      per-arm minimum over the dt sweep, which is the honest headline
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit import transpile
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
import nisq_v3


class FastRun:
    """Level-0 transpile; route reset circuits to density_matrix (see v37)."""

    _dm = None

    def _run(self, qc):
        if any(i.operation.name == 'reset' for i in qc.data):
            if FastRun._dm is None:
                FastRun._dm = AerSimulator(method='density_matrix')
            backend = FastRun._dm
        else:
            backend = self._backend_for(qc.num_qubits)
        t_qc = transpile(qc, backend, optimization_level=0)
        self.last_circuit_depth = t_qc.depth()
        self.max_circuit_depth = max(self.max_circuit_depth, self.last_circuit_depth)
        self.nefv += 1
        kwargs = {}
        if self.sim_seed is not None:
            kwargs['seed_simulator'] = int(self.sim_seed) + self._shot_index
            self._shot_index += 1
        return backend.run(t_qc, shots=self.shot_budget,
                           **kwargs).result().get_counts()


class Base(FastRun, nisq_v3.QLTOv3):
    pass


class ResetWalk(FastRun, nisq_v3.QLTOv3):
    """Fresh ancilla each step. Duplicated from v37 so this file stands alone."""

    def _execute_walk(self, center_params, k_steps, delta_t, radius,
                      active_indices, grad):
        n_active = len(active_indices)
        grad_local = grad[active_indices]
        drift_gain = 1.0 / np.sqrt(max(radius, 1e-9))

        anc = AncillaRegister(1, 'anc')
        param = QuantumRegister(n_active, 'param')
        sysr = QuantumRegister(self.ansatz.num_qubits, 'sys')
        c_param = ClassicalRegister(n_active, 'c_param')
        c_anc = ClassicalRegister(1, 'c_anc')
        qc = QuantumCircuit(anc, param, sysr, c_param, c_anc)

        qc.h(param)
        w = self.build_w_gate(param, sysr, center_params, radius, active_indices)
        qc.append(w, list(param) + list(sysr))
        imprint = PauliEvolutionGate(self.H_sense, time=delta_t * np.pi,
                                     synthesis=LieTrotter(reps=1)).control(1)

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * delta_t
            beta = (1.0 - s) * np.pi * delta_t
            qc.reset(anc)
            qc.h(anc)
            if step == 0:
                qc.append(imprint, [anc[0]] + list(sysr))
            for i in range(n_active):
                al = grad_local[i] * gamma * 0.5 * np.pi * drift_gain
                th = float(np.hypot(al, beta))
                ph = float(np.arctan2(beta, al))
                qc.ry(-ph, param[i])
                qc.crz(th, anc[0], param[i])
                qc.ry(ph, param[i])

        qc.h(anc)
        qc.measure(param, c_param)
        qc.measure(anc, c_anc)
        counts = self._run(qc)
        block = self._decode_walk(counts, center_params, active_indices, radius)
        new_params = center_params.copy()
        new_params[active_indices] = block
        return new_params


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


N, R, DT0, KS, SHOTS, EPOCHS, SEEDS = 4, 0.6, 0.5, 15, 8192, 20, 3
H = heis(N)
exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
ansatz = efficient_su2(N, reps=1)
M = ansatz.num_parameters
SCALE = np.pi / (np.pi * DT0 * KS / 2 * 0.5 * np.pi / np.sqrt(R))

ARMS = [('base', Base, 1.0), ('rescale', Base, SCALE), ('reset', ResetWalk, 1.0)]
DTMULT = [0.5, 1.0, 2.0, 4.0, 8.0]


def run_one(cls, scale, dtm, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        q = cls(ansatz, H, shot_budget=SHOTS, sim_seed=5 + seed)
    q.reset_shot_stream()
    BLK = [b['params'] for b in q.layers if b['params']]
    p = np.random.RandomState(42 + seed).uniform(-np.pi, np.pi, M)
    moves, e3 = [], None
    for ep in range(EPOCHS):
        r = max(R * (0.9 ** ep), 1e-4)
        dt = max(DT0 * dtm * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            g = q.sense_gradient(p, r, act)
            prev = p[act].copy()
            p = q._execute_walk(p, KS, dt, r, act, g * scale)
            moves.append(float(np.mean(np.abs(p[act] - prev))))
        if ep == 2:
            e3 = E(ansatz, H, p)
    return E(ansatz, H, p), e3, float(np.mean(moves))


print("=" * 96)
print(f"STEP-MATCHED COMPARISON — the wrap fixes with the confound controlled. "
      f"N={N}")
print("=" * 96)
print(f"  {EPOCHS} epochs, {SHOTS} shots, {SEEDS} seeds, reps=1, exact "
      f"{exact:.4f}, ansatz ceiling ~ -6.12.")
print(f"  rescale factor = {SCALE:.4f}. dt is swept because a fix that changes")
print(f"  the step size cannot be judged at one dt - that error produced v37f.")
print(f"  |move| is the CONTROL: arms are comparable only where it agrees.")
print()
print(f"  {'arm':>9}{'dt x':>7}{'E_final':>11}{'sigma':>9}{'E@3':>10}"
      f"{'|move|':>10}{'sec':>7}")
print("  " + "-" * 63)

best = {}
table = []
for name, cls, sc in ARMS:
    for dtm in DTMULT:
        t0 = time.time()
        fs, es, ms = [], [], []
        for sd in range(SEEDS):
            f, e3, mv = run_one(cls, sc, dtm, sd)
            fs.append(f); es.append(e3); ms.append(mv)
        row = (name, dtm, float(np.mean(fs)), float(np.std(fs)),
               float(np.mean(es)), float(np.mean(ms)))
        table.append(row)
        if name not in best or row[2] < best[name][2]:
            best[name] = row
        print(f"  {name:>9}{dtm:>7.1f}{np.mean(fs):>11.4f}{np.std(fs):>9.4f}"
              f"{np.mean(es):>10.4f}{np.mean(ms):>10.4f}"
              f"{time.time() - t0:>7.0f}", flush=True)
    print("  " + "." * 63, flush=True)

print(f"\n  BEST PER ARM (each at its own best dt — the honest headline)")
print(f"  {'arm':>9}{'dt x':>7}{'E_final':>11}{'sigma':>9}{'E@3':>10}{'|move|':>10}")
print("  " + "-" * 56)
for name, _, _ in ARMS:
    b = best[name]
    print(f"  {b[0]:>9}{b[1]:>7.1f}{b[2]:>11.4f}{b[3]:>9.4f}{b[4]:>10.4f}"
          f"{b[5]:>10.4f}")

print(f"\n  MATCHED |move| — for each base row, the nearest row of each other arm")
print(f"  {'base |move|':>13}{'base E':>10}{'rescale E':>12}{'d|move|':>9}"
      f"{'reset E':>10}{'d|move|':>9}")
print("  " + "-" * 63)
for r in [t for t in table if t[0] == 'base']:
    out = [f"  {r[5]:>13.4f}{r[2]:>10.4f}"]
    for other in ('rescale', 'reset'):
        cand = [t for t in table if t[0] == other]
        m = min(cand, key=lambda t: abs(t[5] - r[5]))
        out.append(f"{m[2]:>12.4f}{abs(m[5] - r[5]):>9.4f}")
    print("".join(out))

print()
print("  Read the matched-|move| block, not the raw rows. If an arm wins there,")
print("  the wrap was costing real energy. If every arm collapses onto the same")
print("  curve in |move|, then STEP SIZE is the only thing any of these knobs")
print("  controls, the aliasing is real but performance-neutral, and it belongs")
print("  in the mechanism section rather than the tuning section.")
print("  This harness's null scale is 0.03-0.09; treat smaller gaps as zero.")
