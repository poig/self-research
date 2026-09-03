"""Does resetting the walk ancilla each step remove the aliasing? Measured.

v36 measured the walk's input-output map and found it NON-MONOTONIC: at g=-1.0 the
step goes the wrong way, the tanh surrogate fits at 46% residual, and delta_theta_0
depends on the other gradient components with a spread of 0.29 on a step bounded by
R=0.6. No mechanism in the notes predicts any of that.

The mechanism is coherent phase accumulation on a SHARED ancilla. One ancilla is
prepared in |+> before the loop and every one of the k_steps drift/mix pairs is
controlled on it, so the |1> branch applies the PRODUCT of k rotations to each
param qubit. A product of rotations about a fixed-ish axis is a rotation by the SUM
of the angles, and population goes as cos^2(sum/2) - periodic, so it WRAPS.

    per-step drift angle   al_i = g_i * gamma * 0.5 pi / sqrt(R),  gamma = s pi dt
    accumulated over k     sum al_i ~ g_i * 23.9   at dt=0.5, k=15, R=0.6

so the first wrap is at |g| ~ pi/23.9 = 0.13, while the benchmark's own sensed |g|
is 0.58-0.97. The shipped configuration runs 4-7 wraps deep. The notes already
recorded this number - "the measured max alpha across the suite is ... Heisenberg
N=4 6.53" - as a BCH-error risk axis for merged_walk, not as a wrap.

THE FIX UNDER TEST: reset the ancilla every step. Each step becomes its own
Hadamard test, the k rotations stop composing into one large coherent angle, and
the measurement back-action turns a wrapping rotation into a sequence of bounded
biased kicks. Populations then move monotonically and saturate instead of
oscillating - the shape v36 expected and did not find.

Resetting costs the energy imprint, which lived in the ancilla's phase. Two arms
separate the two effects, because they have very different price tags:

  base          as shipped: one ancilla, one imprint, k coherent steps
  reset_cheap   reset each step, imprint ONCE - isolates the reset alone, and
                costs the same as base
  reset_full    reset each step AND re-imprint each step at dt*pi/k, so the total
                evolution time matches base. k controlled evolutions instead of 1:
                affordable only at small N, which is the regime tested here

Convergence runs FIRST because "it converges much faster" is the claim; the
transfer function follows as the mechanism check.

FAIRNESS NOTE: _run is overridden to transpile at optimization_level=0 for EVERY
arm. Level 1 costs ~75 s per reset circuit and dominates the wall clock. The
override is applied identically to all three arms, so it cannot favour one; it
does mean the depth column is not comparable to depths quoted elsewhere.
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
    """Transpile at level 0, and route reset circuits to the density-matrix method.

    Mid-circuit reset is stochastic under the statevector method, so Aer runs one
    trajectory PER SHOT - 8192 evolutions for one walk. Under density_matrix it is
    a deterministic CPTP map: one evolution, then sample the terminal measurements.

    Routing is by circuit CONTENT, not by arm, so base takes exactly the path it
    always takes. It also matters that the switch is per circuit: forcing
    density_matrix globally would drag in the 11-qubit SENSING circuits, which are
    identical in every arm and would just make the whole comparison slower.
    """

    _dm = None

    def _pick(self, qc):
        if not any(i.operation.name == 'reset' for i in qc.data):
            return self._backend_for(qc.num_qubits)
        if FastRun._dm is None:
            FastRun._dm = AerSimulator(method='density_matrix')
        return FastRun._dm

    def _run(self, qc):
        backend = self._pick(qc)
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
    """QLTO V3 with the walk ancilla reset every step.

    Only _execute_walk differs. Sensing, gradient decoding, the W gate and the
    schedule are inherited unchanged, so any difference measured here is the
    ancilla reset and nothing else.
    """

    reimprint = True

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

        # Total evolution time held equal to base's single imprint.
        t = delta_t * np.pi / k_steps if self.reimprint else delta_t * np.pi
        imprint = PauliEvolutionGate(self.H_sense, time=t,
                                     synthesis=LieTrotter(reps=1)).control(1)

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * delta_t
            beta = (1.0 - s) * np.pi * delta_t

            # RESET, NOT MEASURE-AND-RESET. Only the last step's ancilla feeds the
            # decode, and tracing out an ancilla gives the same reduced state on
            # param whether or not it was measured first - a unitary on the
            # ancilla alone cannot change the partial trace. So the intermediate
            # measurements are physically redundant, and dropping them keeps every
            # measurement terminal, which is what lets Aer sample the shots from a
            # single density-matrix evolution instead of 8192 trajectories.
            qc.reset(anc)                       # <- the whole point
            qc.h(anc)
            if self.reimprint or step == 0:
                qc.append(imprint, [anc[0]] + list(sysr))
            for i in range(n_active):
                al = grad_local[i] * gamma * 0.5 * np.pi * drift_gain
                th = float(np.hypot(al, beta))
                ph = float(np.arctan2(beta, al))
                qc.ry(-ph, param[i])
                qc.crz(th, anc[0], param[i])
                qc.ry(ph, param[i])

        qc.h(anc)                               # phase -> population, last step
        qc.measure(param, c_param)
        qc.measure(anc, c_anc)
        counts = self._run(qc)

        # Register layout is now identical to base's, so the SHIPPED decode runs
        # unmodified. Nothing about the decode differs between arms.
        block = self._decode_walk(counts, center_params, active_indices, radius)
        new_params = center_params.copy()
        new_params[active_indices] = block
        return new_params


def _mk(cls, ansatz, H, reimp=None, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        q = cls(ansatz, H, **kw)
    if reimp is not None:
        q.reimprint = reimp
    return q


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


N, R, DT, KS = 4, 0.6, 0.5, 15
H = heis(N)
exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))
ansatz1 = efficient_su2(N, reps=1)
M1 = ansatz1.num_parameters

ARMS = [('base',        Base,      None),
        ('reset_cheap', ResetWalk, False),
        ('reset_full',  ResetWalk, True)]

print("=" * 96)
print(f"ANCILLA RESET — does it remove the walk's aliasing? Heisenberg N={N}, "
      f"exact {exact:.4f}")
print("=" * 96)
tot_angle = np.pi * DT * KS / 2 * 0.5 * np.pi / np.sqrt(R)
print(f"  Predicted accumulated drift angle = g * {tot_angle:.1f};"
      f"  first wrap at |g| = {np.pi / tot_angle:.3f}")
print(f"  dt={DT}, k_steps={KS}, R={R}. Base shares one ancilla across all {KS} "
      f"steps.")

# ── (1) convergence — the claim ──────────────────────────────────────────────
SHOTS_C, EPOCHS, SEEDS = 8192, 20, 3
print(f"\n  (1) CONVERGENCE — {EPOCHS} epochs, {SHOTS_C} shots, {SEEDS} seeds, "
      f"reps=1 (ansatz ceiling ~ -6.12).")
print(f"  {'arm':>12}{'E_final':>11}{'sigma':>9}{'best':>10}{'E@3':>10}{'E@5':>10}"
      f"{'E@10':>10}{'depth':>8}{'sec':>8}")
print("  " + "-" * 78)

for name, cls, reimp in ARMS:
    t0 = time.time()
    fin, e3, e5, e10, dep = [], [], [], [], []
    for sd in range(SEEDS):
        q = _mk(cls, ansatz1, H, reimp, shot_budget=SHOTS_C, sim_seed=5 + sd)
        q.reset_shot_stream()
        BLK = [b['params'] for b in q.layers if b['params']]
        p = np.random.RandomState(42 + sd).uniform(-np.pi, np.pi, M1)
        for ep in range(EPOCHS):
            r = max(R * (0.9 ** ep), 1e-4)
            dt = max(DT * (0.95 ** (ep + 1)), 0.01)
            for act in BLK:
                g = q.sense_gradient(p, r, act)
                p = q._execute_walk(p, KS, dt, r, act, g)
            if ep == 2:
                e3.append(E(ansatz1, H, p))
            if ep == 4:
                e5.append(E(ansatz1, H, p))
            if ep == 9:
                e10.append(E(ansatz1, H, p))
        fin.append(E(ansatz1, H, p))
        dep.append(q.max_circuit_depth)
    print(f"  {name:>12}{np.mean(fin):>11.4f}{np.std(fin):>9.4f}"
          f"{np.min(fin):>10.4f}{np.mean(e3):>10.4f}{np.mean(e5):>10.4f}"
          f"{np.mean(e10):>10.4f}{int(np.mean(dep)):>8}"
          f"{time.time() - t0:>8.0f}", flush=True)

# ── (2) transfer function — the mechanism ────────────────────────────────────
SHOTS_T, REPS_T = 16384, 3
grid = [-2.0, -1.5, -1.0, -0.6, -0.3, -0.15, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0]
centre = np.random.RandomState(7).uniform(-np.pi, np.pi, M1)

print(f"\n  (2) TRANSFER FUNCTION — d_theta_0 vs g_0, others zero. "
      f"{SHOTS_T} shots x {REPS_T}.")
print(f"  {'g_0':>8}" + "".join(f"{a:>14}" for a, _, _ in ARMS))
print("  " + "-" * (8 + 14 * len(ARMS)))

qs, acts, curves = {}, {}, {a: [] for a, _, _ in ARMS}
for name, cls, reimp in ARMS:
    qs[name] = _mk(cls, ansatz1, H, reimp, shot_budget=SHOTS_T, sim_seed=17)
    acts[name] = [b['params'] for b in qs[name].layers if b['params']][0]

for g0 in grid:
    row = []
    for name, _, _ in ARMS:
        q, act = qs[name], acts[name]
        d = []
        for _ in range(REPS_T):
            q.reset_shot_stream()
            g = np.zeros(M1)
            g[act[0]] = g0
            p = q._execute_walk(centre, KS, DT, R, act, g)
            d.append(p[act[0]] - centre[act[0]])
        v = float(np.mean(d))
        curves[name].append(v)
        row.append(v)
    print(f"  {g0:>8.2f}" + "".join(f"{v:>14.5f}" for v in row), flush=True)

print(f"\n  {'arm':>12}{'turns':>8}{'monotone':>10}{'|d| max':>10}"
      f"{'small-g slope':>15}{'corr(d,g)':>12}")
print("  " + "-" * 67)
xs = np.array(grid)
for name, _, _ in ARMS:
    y = np.array(curves[name])
    d = np.diff(y)
    turns = int(np.sum(np.sign(d[:-1]) * np.sign(d[1:]) < 0))
    mono = bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))
    lin = np.abs(xs) <= 0.3
    slope = float(np.polyfit(xs[lin], y[lin], 1)[0])
    cc = float(np.corrcoef(xs, y)[0, 1])
    print(f"  {name:>12}{turns:>8}{str(mono):>10}{np.max(np.abs(y)):>10.4f}"
          f"{slope:>15.4f}{cc:>12.4f}")

print()
print("  'turns' counts sign changes in the slope: 0 means the map is monotone in")
print("  g, which is the minimum any usable update rule must satisfy. Base scored")
print("  many in v36. If reset_* is monotone where base is not, the aliasing")
print("  diagnosis is confirmed and the shipped walk has run wrapped since it was")
print("  written. reset_cheap vs reset_full then says whether the fix is free or")
print("  whether it has to buy back the energy imprint at k times the cost.")
