"""The Grover diffuser as the walk's mixer. First time the primitive is in the circuit.

v38 enumerated the full hypercube and found the product mixer aiming at the WRONG
CORNER on 7 of 16 blocks, with regret up to 0.889 of the hypercube's energy range,
failing exactly where deg2 >= deg1. The degree-<=2 target hit the true corner on
every block, regret2 = 0.000 throughout. That is a REACHABILITY defect - no number
of shots repairs it - and it is therefore outside what Cerezo & Coles forbids,
which is a statement about signal magnitude.

The fix the notes have described but never built: replace the product mixer
(x)_i CRX(beta)_i, whose parameter-register DLA is su(2)^(+)n and which can only
converge to the sign pattern of the degree-1 Walsh coefficients, with the GLOBAL
REFLECTION about the uniform superposition - Grover's diffuser.

    M(beta) = exp(-i beta |s><s|),   |s> = H^(x)n |0>

a rank-1 generator, hence maximally non-local: every one of the 2^n amplitudes
interferes with every other. At beta = pi it is the Grover reflection
I - 2|s><s| up to phase; at small beta it is a weak correlated mix, so beta
remains the continuous knob the schedule already tunes.

IMPLEMENTATION. |s><s| = H^(x)n |0><0| H^(x)n, and exp(-i beta |0><0|) phases only
the all-zeros string, so

    exp(-i beta |s><s|) = H^(x)n X^(x)n  MCPhase(-beta)  X^(x)n H^(x)n

and the ancilla-controlled version just adds the ancilla to the MCPhase controls,
since controlled-(V W V^dag) = V (controlled-W) V^dag leaves the H and X
conjugations uncontrolled. One (n+1)-controlled phase per step.

This is the mixer of Grover-Mixer QAOA (Baertschi & Eidenbenz, arXiv:2006.00354).
Its known property is a real limitation to keep in view: because |s><s| is
invariant under every permutation of basis states, the expectation depends only on
the MULTISET of cost values and not on their arrangement, so it cannot exploit
problem structure and gives Grover-like sqrt(N) scaling rather than better. That
is fine for the question here - the claim under test is REACHABILITY of the
correct corner, not scaling.

FAIRNESS. Both arms use merged_walk=False so the drift is a separate CRZ in each,
identical between arms; only the mixing operation differs.

REPORTED
  (1) per block: the sign pattern of the decoded step against x_true and x_deg1,
      on the blocks v38 flagged - does the global mixer reach what the product
      mixer provably cannot
  (2) convergence over 20 epochs, both arms, with |move| as the step-size control
"""
import sys, os, contextlib, io, itertools, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3


class GlobalMixerWalk(nisq_v3.QLTOv3):
    """QLTO V3 with Grover's diffuser in place of the product CRX mixer."""

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

        qc.h(anc)
        qc.h(param)
        w = self.build_w_gate(param, sysr, center_params, radius, active_indices)
        qc.append(w, list(param) + list(sysr))
        qc.append(PauliEvolutionGate(self.H_sense, time=delta_t * np.pi,
                                     synthesis=LieTrotter(reps=1)).control(1),
                  [anc[0]] + list(sysr))

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * delta_t
            beta = (1.0 - s) * np.pi * delta_t
            # DRIFT - identical to the shipped unmerged path
            for i in range(n_active):
                qc.crz(grad_local[i] * gamma * 0.5 * np.pi * drift_gain,
                       anc[0], param[i])
            # MIXER - global reflection instead of a product of CRX
            qc.h(param)
            qc.x(param)
            if n_active == 1:
                qc.cp(-2.0 * beta, anc[0], param[0])
            else:
                qc.mcp(-2.0 * beta, [anc[0]] + list(param[:-1]), param[-1])
            qc.x(param)
            qc.h(param)

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


def maxcut(N):
    return SparsePauliOp.from_list(
        [("".join("Z" if q in (i, i + 1) else "I" for q in range(N)), 1.0)
         for i in range(N - 1)])


def h2():
    return SparsePauliOp.from_list([("II", -1.0523), ("IZ", 0.3979),
                                    ("ZI", -0.3979), ("ZZ", -0.0113),
                                    ("XX", 0.1809)])


def E(ansatz, H, p):
    return float(np.real(Statevector(ansatz.assign_parameters(p)).expectation_value(H)))


def mk(cls, ansatz, H, **kw):
    kw.setdefault('merged_walk', False)          # fair: same drift in both arms
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(ansatz, H, **kw)


R, DT, KS, SHOTS = 0.6, 0.5, 15, 16384
PROBLEMS = [("H2", h2()), ("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 100)
print("GROVER'S DIFFUSER AS THE WALK MIXER — does it reach what a product mixer cannot?")
print("=" * 100)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. Both arms merged_walk=False so")
print(f"  the drift is identical; only the mixing operation differs.")
print()
print(f"  {'problem':>15}{'blk':>4}{'n':>3}{'deg1':>8}{'deg2':>8}"
      f"{'ham_prod':>10}{'ham_glob':>10}{'reg_prod':>10}{'reg_glob':>10}")
print("  " + "-" * 78)

tot = {'prod': 0, 'glob': 0, 'n': 0}
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    qb = mk(nisq_v3.QLTOv3, ansatz, H, shot_budget=SHOTS, sim_seed=17)
    qg = mk(GlobalMixerWalk, ansatz, H, shot_budget=SHOTS, sim_seed=17)
    BLK = [b['params'] for b in qb.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for k, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[k] = E(ansatz, H, p)
        x_true = sig[int(np.argmin(vals))]
        rng_e = vals.max() - vals.min()

        row = []
        for tag, q in (('prod', qb), ('glob', qg)):
            q.reset_shot_stream()
            g = q.sense_gradient(centre, R, act)
            p1 = q._execute_walk(centre, KS, DT, R, act, g)
            step = p1[act] - centre[act]
            xs = np.where(step >= 0, 1.0, -1.0)
            ham = int(np.sum(xs != x_true))
            e_at = vals[int(np.argmin(np.sum(np.abs(sig - xs), axis=1)))]
            reg = (e_at - vals.min()) / rng_e if rng_e > 1e-12 else 0.0
            row.append((ham, reg))
            tot[tag] += ham
        tot['n'] += n

        d1 = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)])
        p1w = float(np.sum(d1 ** 2))
        p2w = float(np.sum([np.mean(vals * sig[:, i] * sig[:, j]) ** 2
                            for i in range(n) for j in range(i + 1, n)]))
        print(f"  {name:>15}{bi:>4}{n:>3}{p1w:>8.4f}{p2w:>8.4f}"
              f"{row[0][0]:>10}{row[1][0]:>10}{row[0][1]:>10.3f}"
              f"{row[1][1]:>10.3f}", flush=True)

print("  " + "-" * 78)
print(f"  total sign errors: product {tot['prod']}/{tot['n']}"
      f"   global {tot['glob']}/{tot['n']}")

# ── convergence ──────────────────────────────────────────────────────────────
EPOCHS, SEEDS, SHOTS_C = 20, 3, 8192
print(f"\n  CONVERGENCE — {EPOCHS} epochs, {SHOTS_C} shots, {SEEDS} seeds.")
print(f"  {'problem':>15}{'mixer':>9}{'E_final':>11}{'sigma':>9}{'E@5':>10}"
      f"{'|move|':>9}{'depth':>8}{'sec':>7}")
print("  " + "-" * 68)
for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    for tag, cls in (('product', nisq_v3.QLTOv3), ('global', GlobalMixerWalk)):
        t0 = time.time()
        fin, e5, mv, dep = [], [], [], []
        for sd in range(SEEDS):
            q = mk(cls, ansatz, H, shot_budget=SHOTS_C, sim_seed=5 + sd)
            q.reset_shot_stream()
            BLK = [b['params'] for b in q.layers if b['params']]
            p = np.random.RandomState(42 + sd).uniform(-np.pi, np.pi, M)
            for ep in range(EPOCHS):
                r = max(R * (0.9 ** ep), 1e-4)
                dt = max(DT * (0.95 ** (ep + 1)), 0.01)
                for act in BLK:
                    g = q.sense_gradient(p, r, act)
                    prev = p[act].copy()
                    p = q._execute_walk(p, KS, dt, r, act, g)
                    mv.append(float(np.mean(np.abs(p[act] - prev))))
                if ep == 4:
                    e5.append(E(ansatz, H, p))
            fin.append(E(ansatz, H, p))
            dep.append(q.max_circuit_depth)
        print(f"  {name:>15}{tag:>9}{np.mean(fin):>11.4f}{np.std(fin):>9.4f}"
              f"{np.mean(e5):>10.4f}{np.mean(mv):>9.4f}{int(np.mean(dep)):>8}"
              f"{time.time() - t0:>7.0f}", flush=True)

print()
print("  ham_glob < ham_prod on the blocks v38 flagged would confirm that the")
print("  reachability defect is the mixer's locality and that Grover's diffuser")
print("  repairs it. Convergence is reported second and with |move| because a")
print("  mixer change moves the step size, which is the confound that invalidated")
print("  v37f - compare arms only where |move| agrees.")
