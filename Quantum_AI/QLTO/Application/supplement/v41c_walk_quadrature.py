"""The walk reads the WRONG QUADRATURE. One missing sdg.

v41 showed the energy imprint marks nothing on its own: with the drift removed the
parameter distribution went exactly uniform. v41b then killed the obvious
explanation - W-dagger changes NOTHING, and provably cannot, because W is
controlled on param and is therefore block-diagonal in the param basis, so it
cannot move param populations at all.

What v41b did show, once the drift was off, is that the energy reaches the
distribution with |corr(P, -E)| up to 0.59 but with an INCONSISTENT SIGN across
blocks: +0.50 on Heisenberg block 1, -0.59 on block 2. An oracle that marks low
energy on one block and high energy on the next is not reading energy.

The sensing circuit documents exactly this trap and avoids it:

    qc.sdg(anc)    # Y basis -> Im<U> ~ -tau<H>; a plain H would read
    qc.h(anc)      # Re<U> ~ 1 - tau^2<H^2>/2, the wrong observable

_execute_walk ends with a bare qc.h(anc) - no sdg. So the walk's ancilla reports
Re<e^{-iHt}> ~ 1 - t^2<H^2>/2. It marks by <H^2>, which is second order in t
(hence weak) and does not track <H> in sign (hence the sign flips). The sensing
path found this bug and fixed it; the walk never received the fix.

THE TEST. Add the sdg, change nothing else, and watch corr(P, -E) with the drift
at zero, where the energy is the only possible source of structure. A correct
energy oracle must make low-energy vertices MORE likely on EVERY block, so the
diagnostic is not the magnitude but the SIGN CONSISTENCY.

    y_basis False   Re<U> ~ 1 - t^2<H^2>/2     the shipped walk
    y_basis True    Im<U> ~ -t<H>              what sensing already does

If the sign becomes consistently positive, the walk has been marking the wrong
observable since it was written, and the degree-1 targeting measured in v38/v39c
follows: with the energy oracle miswired, the only coherent marking left is the
degree-1 drift, which is exactly what the walk was observed to amplify.
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
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E


class QuadWalk(nisq_v3.QLTOv3):
    drift_scale = 1.0
    imprint_scale = 1.0
    y_basis = False

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

        if self.y_basis:
            qc.sdg(anc)          # <- the one line the sensing path already has
        qc.h(anc)
        qc.measure(param, cp)
        qc.measure(anc, ca)
        counts = self._run(qc)
        blk = self._decode_walk(counts, center_params, active_indices, radius)
        out = center_params.copy()
        out[active_indices] = blk
        return out


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
GRID = [(1.0, 1.0, False), (1.0, 1.0, True),
        (0.0, 1.0, False), (0.0, 1.0, True),
        (0.0, 4.0, True), (0.3, 4.0, True)]
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 96)
print("WALK QUADRATURE — does the missing sdg explain the miswired oracle?")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. With drift=0 the energy is the")
print(f"  only possible source of structure. The diagnostic is SIGN CONSISTENCY")
print(f"  of corr(P,-E): a real energy oracle is positive on EVERY block.")

summary = {}
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

        def idx_of(x):
            return int(''.join('1' if x[i] > 0 else '0'
                               for i in range(n))[::-1], 2)
        i_true = idx_of(x_true)
        e_by_idx = np.empty(2 ** n)
        e_by_idx[np.array([idx_of(s) for s in sig])] = vals

        print(f"\n  {name} block {bi}  (n={n})")
        print(f"  {'drift':>7}{'imprint':>9}{'sdg':>7}{'corr(P,-E)':>12}"
              f"{'enhance':>9}{'H/Hmax':>9}")
        print("  " + "-" * 53)
        for ds, ims, yb in GRID:
            with contextlib.redirect_stdout(io.StringIO()):
                q = QuadWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                             merged_walk=False)
            q.drift_scale, q.imprint_scale, q.y_basis = ds, ims, yb
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
            cc = float(np.corrcoef(sel, -e_by_idx)[0, 1])
            summary.setdefault((ds, ims, yb), []).append(cc)
            print(f"  {ds:>7.1f}{ims:>9.1f}{str(yb):>7}{cc:>12.4f}"
                  f"{sel[i_true] * 2 ** n:>9.3f}{ent:>9.3f}", flush=True)

print("\n" + "=" * 96)
print("  SIGN CONSISTENCY of corr(P,-E) across all blocks")
print(f"  {'drift':>7}{'imprint':>9}{'sdg':>7}{'mean':>9}{'min':>9}{'max':>9}"
      f"{'frac>0':>9}")
print("  " + "-" * 59)
for (ds, ims, yb), cs in summary.items():
    cs = np.array(cs)
    print(f"  {ds:>7.1f}{ims:>9.1f}{str(yb):>7}{cs.mean():>9.4f}"
          f"{cs.min():>9.4f}{cs.max():>9.4f}{np.mean(cs > 0):>9.1%}")

print()
print("  frac>0 going to 100% with sdg=True at drift=0 is the result: the energy")
print("  oracle then marks low energy on every block, which is what it must do.")
print("  If the sign stays mixed, the quadrature is not the explanation and the")
print("  imprint's coupling to the param register has to be re-derived.")
