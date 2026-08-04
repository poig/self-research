"""Which oracle is the walk listening to? Decouple the two and sweep.

v39c read the walk's RAW parameter distribution and found real Grover-like
amplification - enhancement up to 6.6x over uniform - but aimed at the DEGREE-1
argmin. The separation against v38's enumeration was perfect: on the 5 blocks
where the degree-1 target is correct, enhance was 1.5-6.6 with mode == x_true; on
the 7 where it is wrong, enhance <= 1.55 and mode != x_true.

That is an ORACLE defect, not a mixer defect, which is why swapping in Grover's
diffuser (v39) and adding oracle-diffuser alternation (v39b) both changed nothing:
a diffuser amplifies whatever it is given. The walk writes

    phi(x) = A * sum_i g_i x_i   +   t * E(x)
             DEGREE-1 drift          the CORRECT all-degree oracle

with, at shipped settings, A ~ |g| * 23.9 ~ 16 rad against t * dE ~ 1.57 * 2 ~ 3
rad. The correct oracle is present and is drowned about fivefold by a degree-1
one.

THE OBSTRUCTION TO TESTING THIS is that delta_t scales all three of the drift,
the imprint and the mixer together, so the balance has never been movable. This
file adds two independent multipliers - drift_scale on alpha, imprint_scale on
the imprint time - changing nothing else, and sweeps them.

MEASURED, on the blocks v38 flagged and on control blocks where degree-1 is
already correct:

    enhance   P(x_true) / 2^-n from the raw param marginal; 1.0 is uniform
    mode      whether the distribution's MODE is the true corner
    e_deg1    the same enhancement for the DEGREE-1 argmin, so the two targets
              can be watched trading places

If lowering drift_scale and raising imprint_scale moves enhance from the degree-1
corner onto the true corner, the diagnosis is confirmed and the walk has a
correct oracle it has simply been shouting over. If enhance collapses toward 1.0
instead, the imprint is too weak to mark anything on its own and the phase spread
t * (max E - min E) is the quantity that has to grow - which costs evolution time
and therefore depth, and is a different engineering problem.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v3


class BalancedWalk(nisq_v3.QLTOv3):
    """Shipped walk with the drift and the imprint on independent knobs."""

    drift_scale = 1.0
    imprint_scale = 1.0

    def _execute_walk(self, center_params, k_steps, delta_t, radius,
                      active_indices, grad):
        n = len(active_indices)
        gl = grad[active_indices] * self.drift_scale
        gain = 1.0 / np.sqrt(max(radius, 1e-9))

        anc = AncillaRegister(1, 'anc')
        param = QuantumRegister(n, 'param')
        sysr = QuantumRegister(self.ansatz.num_qubits, 'sys')
        cp = ClassicalRegister(n, 'c_param')
        ca = ClassicalRegister(1, 'c_anc')
        qc = QuantumCircuit(anc, param, sysr, cp, ca)

        qc.h(anc)
        qc.h(param)
        qc.append(self.build_w_gate(param, sysr, center_params, radius,
                                    active_indices), list(param) + list(sysr))
        qc.append(PauliEvolutionGate(
            self.H_sense, time=delta_t * np.pi * self.imprint_scale,
            synthesis=LieTrotter(reps=1)).control(1), [anc[0]] + list(sysr))

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * delta_t
            beta = (1.0 - s) * np.pi * delta_t
            for i in range(n):
                qc.crz(gl[i] * gamma * 0.5 * np.pi * gain, anc[0], param[i])
            for i in range(n):
                qc.crx(beta, anc[0], param[i])

        qc.h(anc)
        qc.measure(param, cp)
        qc.measure(anc, ca)
        counts = self._run(qc)
        blk = self._decode_walk(counts, center_params, active_indices, radius)
        out = center_params.copy()
        out[active_indices] = blk
        return out


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


def E(a, H, p):
    return float(np.real(Statevector(a.assign_parameters(p)).expectation_value(H)))


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
GRID = [(1.0, 1.0), (0.3, 1.0), (0.1, 1.0), (0.0, 1.0),
        (0.1, 4.0), (0.0, 4.0), (0.0, 10.0)]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 100)
print("ORACLE BALANCE — degree-1 drift against the all-degree energy imprint")
print("=" * 100)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. Only two numbers change per row.")
print(f"  drift=0 removes the degree-1 oracle entirely, leaving E(x) as the only")
print(f"  marking. enhance = P(x_true)*2^n; e_deg1 = P(x_deg1)*2^n.")

for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        probe = nisq_v3.QLTOv3(ansatz, H, shot_budget=64, merged_walk=False)
    BLK = [b['params'] for b in probe.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        vals = np.empty(len(sig))
        for k, sv in enumerate(sig):
            p = centre.copy(); p[act] = p[act] + R * sv
            vals[k] = E(ansatz, H, p)
        x_true = sig[int(np.argmin(vals))]
        d1 = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)])
        x_d1 = np.where(d1 <= 0, 1.0, -1.0)
        agree = np.all(x_d1 == x_true)

        def idx_of(x):
            return int(''.join('1' if x[i] > 0 else '0'
                               for i in range(n))[::-1], 2)
        i_true, i_d1 = idx_of(x_true), idx_of(x_d1)

        print(f"\n  {name} block {bi}  (n={n}, degree-1 target "
              f"{'CORRECT' if agree else 'WRONG'})")
        print(f"  {'drift':>7}{'imprint':>9}{'enhance':>9}{'e_deg1':>9}"
              f"{'mode=x*':>9}{'H/Hmax':>9}")
        print("  " + "-" * 52)
        for ds, ims in GRID:
            q = None
            with contextlib.redirect_stdout(io.StringIO()):
                q = BalancedWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                                 merged_walk=False)
            q.drift_scale, q.imprint_scale = ds, ims
            q.reset_shot_stream()
            g = q.sense_gradient(centre, R, act)
            cap = {}
            orig = q._run
            def spy(qc, _o=orig, _c=cap):
                r = _o(qc); _c['last'] = r; return r
            q._run = spy
            q._execute_walk(centre, KS, DT, R, act, g)
            q._run = orig

            sel = np.zeros(2 ** n)
            for bs, c in cap['last'].items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    sel[int(parts[1].replace(" ", ""), 2)] += c
            sel = sel / max(sel.sum(), 1)
            qq = sel[sel > 0]
            ent = float(-np.sum(qq * np.log2(qq))) / n
            print(f"  {ds:>7.1f}{ims:>9.1f}{sel[i_true] * 2 ** n:>9.3f}"
                  f"{sel[i_d1] * 2 ** n:>9.3f}"
                  f"{str(int(np.argmax(sel)) == i_true):>9}{ent:>9.3f}",
                  flush=True)

print()
print("  On a WRONG block, enhance rising above e_deg1 as drift falls is the")
print("  result: the walk stops amplifying the linear surrogate and starts")
print("  amplifying the true minimum. On a CORRECT block the two columns should")
print("  stay together throughout, which is the control.")
