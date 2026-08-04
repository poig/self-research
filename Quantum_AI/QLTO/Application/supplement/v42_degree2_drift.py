"""T7 redone with the instrument that was missing.

T7 built a degree-2 drift, measured FINAL ENERGY, saw nothing, and concluded that
"a product CRX mixer structurally cannot convert pairwise phase to correlated
motion". Today's chain says that conclusion was drawn with the wrong instrument
and probably for the wrong reason:

  v38   the product mixer converges to the DEGREE-1 argmin - wrong on 7/16
        blocks, regret to 0.889 - while the degree-<=2 target is EXACT on every
        block, regret2 = 0.000. So the target is reachable; something has to aim
        at it.
  v39   Grover's diffuser in place of the product mixer: 14/40 -> 14/40. The
        mixer is not what selects the target.
  v39b  adding oracle-diffuser alternation: no change either.
  v39c  the walk DOES amplify, up to 6.6x over uniform - onto the degree-1
        corner, with perfect separation against v38's verdict per block.
  v41   the DRIFT is the oracle: corr(P,-E) = 0.45 with the sign positive on
        100% of blocks, against 0.07 and coin-flip with the drift removed.

So the walk aims where the drift points, and the drift writes

    phi(x) = sum_i g_i x_i        the DEGREE-1 Walsh truncation of E

whose correlation with E is ceilinged by the degree-1 fraction of the spectrum.
T6 measured deg1+deg2 = 99.6% with deg2 EXCEEDING deg1 on 2 of 4 blocks. Adding
the degree-2 term should raise that ceiling, and v38 says the resulting target is
exact.

THE COEFFICIENTS ARE FREE. Every shot of the QPE sensing circuit carries a
decoded energy AND the full vertex bitstring, so

    E_hat({i})   = mean(e * x_i)          <- what the walk already uses
    E_hat({i,j}) = mean(e * x_i * x_j)    <- the same shots, one more product

by T2's linearity: any marginal of the same shot record is available at no extra
circuit cost. The degree-2 drift therefore costs GATES, not CIRCUITS.

IMPLEMENTATION. exp(-i th ZZ/2) = CX(i,j) RZ(th, j) CX(i,j), and controlling only
the RZ suffices because controlled-(V W V^dag) = V (controlled-W) V^dag leaves
the CX pair uncontrolled. So one CRZZ is cx, crz, cx - three gates per pair per
step, n(n-1)/2 pairs.

MEASURED, on the metrics that actually move:
    corr(P,-E)   does the walk's distribution track the true energy better
    enhance      P(x_true)/2^-n, and the same for the degree-1 corner, so the
                 two targets can be watched trading places
    depth        what the pairwise terms cost
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


def sense_deg12(q, centre, R, act):
    """Degree-1 AND degree-2 Walsh coefficients from ONE sensing circuit.

    Same shots, same decode of the QPE register into an energy; the only
    difference is which product of vertex bits each shot is weighted by.
    """
    qc = q._build_qpe_sensing_circuit(centre, R, act)
    counts = q._run(qc)
    k, n = q.num_ancillas, len(act)
    tot = 0.0
    s1 = np.zeros(n)
    s2 = np.zeros((n, n))
    for bitstr, cnt in counts.items():
        parts = bitstr.split()
        if len(parts) != 2:
            continue
        phi = int(parts[0], 2) / (2 ** k)
        if phi >= 0.5:
            phi -= 1.0
        e = -2.0 * np.pi * phi / (q.tau0 + 1e-12)
        xb = parts[1].replace(" ", "")[::-1]
        x = np.array([1.0 if (i < len(xb) and xb[i] == '1') else -1.0
                      for i in range(n)])
        tot += cnt
        s1 += cnt * e * x
        s2 += cnt * e * np.outer(x, x)
    d1 = s1 / max(tot, 1.0)
    d2 = s2 / max(tot, 1.0)
    np.fill_diagonal(d2, 0.0)          # x_i^2 = 1, that is the degree-0 term
    return d1 / R, d2 / R


class Deg2Walk(nisq_v3.QLTOv3):
    """Walk whose drift carries degree-1 AND (optionally) degree-2 phase."""

    use_deg2 = True

    def walk(self, centre, k_steps, dt, R, act, g1, g2):
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

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * dt
            beta = (1.0 - s) * np.pi * dt
            scale = gamma * 0.5 * np.pi * gain
            for i in range(n):
                qc.crz(g1[i] * scale, anc[0], param[i])
            if self.use_deg2:
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(g2[i, j]) < 1e-12:
                            continue
                        qc.cx(param[i], param[j])
                        qc.crz(2.0 * g2[i, j] * scale, anc[0], param[j])
                        qc.cx(param[i], param[j])
            for i in range(n):
                qc.crx(beta, anc[0], param[i])

        qc.h(anc)
        qc.measure(param, cp)
        qc.measure(anc, ca)
        return self._run(qc)


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 96)
print("DEGREE-2 DRIFT — T7 remeasured on the oracle, not on final energy")
print("=" * 96)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots. Both degree-1 and degree-2")
print(f"  coefficients come from the SAME sensing circuit - no extra circuits.")
print(f"  corr(P,-E) is the oracle quality; enhance is what it amplifies.")
print()
print(f"  {'problem':>15}{'blk':>4}{'d1ok':>6}{'drift':>7}{'corr(P,-E)':>12}"
      f"{'enh_true':>10}{'enh_deg1':>10}{'mode=x*':>9}{'depth':>7}")
print("  " + "-" * 80)

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
        x_true = sig[int(np.argmin(vals))]
        d1e = np.array([float(np.mean(vals * sig[:, i])) for i in range(n)])
        x_d1 = np.where(d1e <= 0, 1.0, -1.0)
        d1ok = bool(np.all(x_d1 == x_true))

        def idx_of(x):
            return int(''.join('1' if x[i] > 0 else '0'
                               for i in range(n))[::-1], 2)
        i_true, i_d1 = idx_of(x_true), idx_of(x_d1)
        e_by_idx = np.empty(2 ** n)
        e_by_idx[np.array([idx_of(s) for s in sig])] = vals

        for tag, use2 in (('deg1', False), ('deg1+2', True)):
            with contextlib.redirect_stdout(io.StringIO()):
                q = Deg2Walk(ansatz, H, shot_budget=SHOTS, sim_seed=17,
                             merged_walk=False)
            q.use_deg2 = use2
            q.reset_shot_stream()
            g1, g2 = sense_deg12(q, centre, R, act)
            counts = q.walk(centre, KS, DT, R, act, g1, g2)

            sel = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    sel[int(parts[1].replace(" ", ""), 2)] += c
            sel = sel / max(sel.sum(), 1)
            cc = float(np.corrcoef(sel, -e_by_idx)[0, 1])
            summ.setdefault(tag, []).append(cc)
            print(f"  {name if not use2 else '':>15}{bi if not use2 else '':>4}"
                  f"{('Y' if d1ok else 'n') if not use2 else '':>6}{tag:>7}"
                  f"{cc:>12.4f}{sel[i_true] * 2 ** n:>10.3f}"
                  f"{sel[i_d1] * 2 ** n:>10.3f}"
                  f"{str(int(np.argmax(sel)) == i_true):>9}"
                  f"{q.last_circuit_depth:>7}", flush=True)
        print("  " + "." * 80)

print(f"\n  {'drift':>9}{'mean corr':>12}{'min':>9}{'max':>9}{'frac>0':>9}")
print("  " + "-" * 48)
for tag in ('deg1', 'deg1+2'):
    cs = np.array(summ[tag])
    print(f"  {tag:>9}{cs.mean():>12.4f}{cs.min():>9.4f}{cs.max():>9.4f}"
          f"{np.mean(cs > 0):>9.1%}")

print()
print("  A rise in mean corr(P,-E) is the oracle improving. enh_true overtaking")
print("  enh_deg1 on the blocks marked 'n' is the walk changing what it aims at -")
print("  which is the claim T7 could not see, because final energy is downstream")
print("  of the decode and the decode averages the concentration away.")
