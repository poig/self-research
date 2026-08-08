"""Level 2 Walk Experiment: Correlated Drift + Correlated Mixer

Tests whether degree-2 drift AND a correlated (body-matched) mixer TOGETHER
can improve the parameter-register dynamics beyond su(2)^⊗n.

Previous experiments (v5_deg2walk, v42_degree2_drift) tested degree-2 drift alone
with a product mixer and saw degradation. The notes (RESEARCH_DIRECTIONS §2) predicted:
"a product mixer structurally cannot convert pairwise phase correlations into
correlated population motion... Level 2 is not 'drift plus optionally a mixer' —
it is a single upgrade with two halves."

This script runs a 2x2 factorial test:
  Arm A: deg1 drift, product mixer           (Level 1 control / v3 shipped walk)
  Arm B: deg1+2 drift, product mixer         (Reproduces v42 negative result)
  Arm C: deg1 drift, correlated mixer        (Isolates correlated mixer)
  Arm D: deg1+2 drift, correlated mixer      (Full Level 2 upgrade)

Measured on oracle metric corr(P, -E), enhancement over uniform, mode=x*, and depth.
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
    """Degree-1 AND degree-2 Walsh coefficients from ONE QPE sensing circuit."""
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
    np.fill_diagonal(d2, 0.0)
    return d1 / R, d2 / R


class Level2Walk(nisq_v3.QLTOv3):
    """Walk with options for degree-2 drift AND correlated (body-matched) mixer."""

    def walk(self, centre, k_steps, dt, R, act, g1, g2,
             use_deg2_drift=False, use_corr_mixer=False, mixer_scale=1.0):
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

        scale = 0.5 * np.pi * gain

        for step in range(k_steps):
            s = (step + 0.5) / k_steps
            gamma = s * np.pi * dt
            beta = (1.0 - s) * np.pi * dt
            beta2 = beta * mixer_scale

            # 1. Degree-1 drift
            for i in range(n):
                qc.crz(g1[i] * gamma * scale, anc[0], param[i])

            # 2. Degree-2 drift
            if use_deg2_drift:
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(g2[i, j]) < 1e-12:
                            continue
                        qc.cx(param[i], param[j])
                        qc.crz(2.0 * g2[i, j] * gamma * scale, anc[0], param[j])
                        qc.cx(param[i], param[j])

            # 3. Product mixer
            for i in range(n):
                qc.crx(beta, anc[0], param[i])

            # 4. Correlated mixer (Controlled-XX on all adjacent pairs)
            if use_corr_mixer:
                for i in range(n):
                    for j in range(i + 1, n):
                        # Controlled-RXX(beta2) via H + CX + CRZ + CX + H
                        qc.h(param[i]); qc.h(param[j])
                        qc.cx(param[i], param[j])
                        qc.crz(beta2, anc[0], param[j])
                        qc.cx(param[i], param[j])
                        qc.h(param[j]); qc.h(param[i])

        qc.h(anc)
        qc.measure(param, cp)
        qc.measure(anc, ca)
        return self._run(qc)


R, DT, KS, SHOTS = 0.6, 0.5, 15, 65536
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]

print("=" * 100)
print("LEVEL 2 WALK EXPERIMENT — Correlated Drift + Correlated Mixer (2x2 Factorial)")
print("=" * 100)
print(f"  R={R}, dt={DT}, k={KS}, {SHOTS} shots.")
print(f"  Arm A: deg1 drift + product mixer       (Level 1 control)")
print(f"  Arm B: deg1+2 drift + product mixer     (Reproduces v42)")
print(f"  Arm C: deg1 drift + correlated mixer    (Isolates mixer)")
print(f"  Arm D: deg1+2 drift + correlated mixer  (Full Level 2)")
print()

MIXER_SCALES = [0.25, 0.5, 1.0]

for name, H in PROBLEMS:
    print("=" * 100)
    print(f"  PROBLEM: {name}")
    print("=" * 100)
    print(f"  {'blk':>4}{'d1ok':>6}{'arm':>18}{'scale':>7}{'corr(P,-E)':>12}"
          f"{'enh_true':>10}{'enh_deg1':>10}{'mode=x*':>9}{'depth':>7}")
    print("  " + "-" * 86)

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

        # First, sense once for this block
        with contextlib.redirect_stdout(io.StringIO()):
            q = Level2Walk(ansatz, H, shot_budget=SHOTS, sim_seed=17, merged_walk=False)
        g1, g2 = sense_deg12(q, centre, R, act)

        # Runs for Arms A and B (scale=1.0)
        for arm_name, deg2_d, corr_m in [
            ("A: deg1+prod", False, False),
            ("B: deg1+2+prod", True, False),
        ]:
            with contextlib.redirect_stdout(io.StringIO()):
                q_exec = Level2Walk(ansatz, H, shot_budget=SHOTS, sim_seed=17, merged_walk=False)
            q_exec.reset_shot_stream()
            counts = q_exec.walk(centre, KS, DT, R, act, g1, g2,
                                 use_deg2_drift=deg2_d, use_corr_mixer=corr_m)

            sel = np.zeros(2 ** n)
            for bs, c in counts.items():
                parts = bs.split()
                if len(parts) == 2 and parts[0][-1] == '1':
                    sel[int(parts[1].replace(" ", ""), 2)] += c
            sel = sel / max(sel.sum(), 1)
            cc = float(np.corrcoef(sel, -e_by_idx)[0, 1])

            print(f"  {bi:>4}{'Y' if d1ok else 'n':>6}{arm_name:>18}{1.0:>7.2f}"
                  f"{cc:>12.4f}{sel[i_true] * 2 ** n:>10.3f}"
                  f"{sel[i_d1] * 2 ** n:>10.3f}"
                  f"{str(int(np.argmax(sel)) == i_true):>9}"
                  f"{q_exec.last_circuit_depth:>7}", flush=True)

        # Runs for Arms C and D with scale sweep
        for m_scale in MIXER_SCALES:
            for arm_name, deg2_d, corr_m in [
                ("C: deg1+corr", False, True),
                ("D: deg1+2+corr", True, True),
            ]:
                with contextlib.redirect_stdout(io.StringIO()):
                    q_exec = Level2Walk(ansatz, H, shot_budget=SHOTS, sim_seed=17, merged_walk=False)
                q_exec.reset_shot_stream()
                counts = q_exec.walk(centre, KS, DT, R, act, g1, g2,
                                     use_deg2_drift=deg2_d, use_corr_mixer=corr_m,
                                     mixer_scale=m_scale)

                sel = np.zeros(2 ** n)
                for bs, c in counts.items():
                    parts = bs.split()
                    if len(parts) == 2 and parts[0][-1] == '1':
                        sel[int(parts[1].replace(" ", ""), 2)] += c
                sel = sel / max(sel.sum(), 1)
                cc = float(np.corrcoef(sel, -e_by_idx)[0, 1])

                print(f"  {'':>4}{'':>6}{arm_name:>18}{m_scale:>7.2f}"
                      f"{cc:>12.4f}{sel[i_true] * 2 ** n:>10.3f}"
                      f"{sel[i_d1] * 2 ** n:>10.3f}"
                      f"{str(int(np.argmax(sel)) == i_true):>9}"
                      f"{q_exec.last_circuit_depth:>7}", flush=True)

        print("  " + "." * 86)
