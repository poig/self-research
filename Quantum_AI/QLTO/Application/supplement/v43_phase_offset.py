"""The oracle folds. One phase offset on the ancilla unfolds it.

Four explanations for the degree-2 failure have now been falsified in order:
mixer locality (v39), missing alternation (v39b), the unbounded phase channel
(v42b - bounding helped degree-1 but never rescued degree-2), and estimator noise
(v42c - cos2 = 0.98-0.9997 on every block with appreciable degree-2 weight).

So the coefficients are exact, the channel is clean, and v38 proved by
enumeration that the degree-<=2 argmin IS the true argmin on all 16 blocks. The
failure has to be in how phase becomes probability.

IT IS. The drift is diagonal in the param basis, so with the anc=1
post-selection,

    P(x) ~ |<x|(I - U)|s>|^2 = |1 - e^{i phi(x)}|^2 = 4 sin^2( phi(x) / 2 ).

sin^2 IS EVEN. Vertices at +phi and -phi receive IDENTICAL probability. And
phi(x) = ACC * sum_i g_i x_i is symmetric about zero by construction, because x
ranges over +-1 and the coefficients are signed. The oracle therefore FOLDS the
low-energy half of the hypercube onto the high-energy half and is blind to the
sign of the phase it just computed.

That accounts for every observation: degree-1 works partially and by accident
(corr 0.45); degree-2 widens the spread and folds MORE (0.12); bounding the span
at pi helped degree-1 (0.54) by narrowing the fold; and bounding could not rescue
degree-2 because the span stayed centred on zero.

Grover has no such problem because it marks at exactly phi = pi, the maximum of
sin^2. The continuous analogue is to map the energy range affinely onto [0, pi]:

    normalise   span of phi -> pi, so phi in [-pi/2, +pi/2]
    offset      add phi_0 = pi/2 on the ancilla, so psi = phi + phi_0 in [0, pi]
    sign        choose it so psi is LARGE where E is LOW

then P ~ sin^2(psi/2) is MONOTONE in psi over the whole range, and monotone in E.
The offset is a single phase gate on the ancilla - qc.p(phi_0, anc) - because a
phase on the |1> branch is exactly a Z-rotation of the control.

SWEPT: offset and drift sign, at span pi, for degree-1 and degree-1+2. If the
right (offset, sign) lifts corr(P,-E) well above the 0.5352 that bounded degree-1
reached, the fold was the obstruction. If degree-1+2 then overtakes degree-1, the
degree-2 information becomes usable for the first time and T7 was measuring a
circuit that could not have used it.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12


class OffsetWalk(nisq_v3.QLTOv3):
    def walk(self, centre, k, dt, R, act, g1, g2, use2, offset):
        n = len(act)
        gain = 1.0 / np.sqrt(max(R, 1e-9))
        anc = AncillaRegister(1, 'anc')
        param = QuantumRegister(n, 'param')
        sysr = QuantumRegister(self.ansatz.num_qubits, 'sys')
        cp = ClassicalRegister(n, 'c_param')
        ca = ClassicalRegister(1, 'c_anc')
        qc = QuantumCircuit(anc, param, sysr, cp, ca)

        qc.h(anc)
        qc.h(param)
        qc.append(self.build_w_gate(param, sysr, centre, R, act),
                  list(param) + list(sysr))
        qc.append(PauliEvolutionGate(self.H_sense, time=dt * np.pi,
                                     synthesis=LieTrotter(reps=1)).control(1),
                  [anc[0]] + list(sysr))

        for step in range(k):
            s = (step + 0.5) / k
            gamma = s * np.pi * dt
            beta = (1.0 - s) * np.pi * dt
            sc = gamma * 0.5 * np.pi * gain
            for i in range(n):
                qc.crz(g1[i] * sc, anc[0], param[i])
            if use2:
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(g2[i, j]) < 1e-12:
                            continue
                        qc.cx(param[i], param[j])
                        qc.crz(2.0 * g2[i, j] * sc, anc[0], param[j])
                        qc.cx(param[i], param[j])
            for i in range(n):
                qc.crx(beta, anc[0], param[i])

        if abs(offset) > 1e-12:
            qc.p(offset, anc[0])      # <- unfold: shift psi into [0, pi]
        qc.h(anc)
        qc.measure(param, cp)
        qc.measure(anc, ca)
        return self._run(qc)


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
ACC = 0.5 * np.pi / np.sqrt(R) * np.pi * DT * KS / 2.0
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]
CFG = [('ship d1', False, None, +1, 0.0)]
for use2 in (False, True):
    for sgn in (+1, -1):
        for off in (0.0, np.pi / 2, np.pi):
            CFG.append((f"{'d12' if use2 else 'd1 '} s{sgn:+d} o{off:.2f}",
                        use2, np.pi, sgn, off))

print("=" * 96)
print("PHASE OFFSET — unfolding the sin^2 oracle")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. Span normalised to pi so")
print(f"  phi in [-pi/2, pi/2]; offset pi/2 then puts psi in [0, pi], where")
print(f"  sin^2(psi/2) is monotone. 'ship d1' is the shipped drift, unnormalised.")

summ = {}
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

        def idx_of(x):
            return int(''.join('1' if x[i] > 0 else '0'
                               for i in range(n))[::-1], 2)
        i_true = idx_of(sig[int(np.argmin(vals))])
        e_by_idx = np.empty(2 ** n)
        e_by_idx[np.array([idx_of(s) for s in sig])] = vals

        for tag, use2, span, sgn, off in CFG:
            with contextlib.redirect_stdout(io.StringIO()):
                q = OffsetWalk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                               merged_walk=False)
            q.reset_shot_stream()
            g1, g2 = sense_deg12(q, centre, R, act)
            if not use2:
                g2 = np.zeros_like(g2)
            if span is not None:
                l1 = float(np.sum(np.abs(g1)))
                l2 = float(np.sum(np.abs(np.triu(g2, 1))))
                raw = 2.0 * ACC * (l1 + l2)          # full SPAN, not the L1 bound
                if raw > 1e-12:
                    nrm = span / raw
                    g1, g2 = g1 * nrm, g2 * nrm
            g1, g2 = sgn * g1, sgn * g2
            counts = q.walk(centre, KS, DT, R, act, g1, g2, use2, off)

            sel = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    sel[int(parts[1].replace(" ", ""), 2)] += c
            sel = sel / max(sel.sum(), 1)
            cc = float(np.corrcoef(sel, -e_by_idx)[0, 1])
            summ.setdefault(tag, []).append(
                (cc, sel[i_true] * 2 ** n, int(np.argmax(sel)) == i_true))

print(f"\n  {'config':>16}{'mean corr':>12}{'min':>9}{'frac>0':>9}"
      f"{'mean enh':>10}{'mode hits':>11}")
print("  " + "-" * 67)
rows = []
for tag, *_ in CFG:
    a = np.array([s[0] for s in summ[tag]])
    b = np.array([s[1] for s in summ[tag]])
    m = np.array([s[2] for s in summ[tag]])
    rows.append((a.mean(), tag, a, b, m))
    print(f"  {tag:>16}{a.mean():>12.4f}{a.min():>9.4f}{np.mean(a > 0):>9.1%}"
          f"{b.mean():>10.3f}{f'{m.sum()}/{len(m)}':>11}")

best = max(rows, key=lambda r: r[0])
print(f"\n  best: {best[1]}  at mean corr {best[0]:.4f}")
print(f"  reference points: shipped degree-1 0.4535, bounded degree-1 0.5352,")
print(f"                    shipped degree-1+2 0.1170, bounded degree-1+2 0.2537")
print()
print("  A jump well past 0.5352 means the fold was the obstruction. degree-1+2")
print("  overtaking degree-1 there means the pairwise information is usable once")
print("  the oracle is monotone, and that every earlier degree-2 test was run on")
print("  a circuit incapable of using it.")
