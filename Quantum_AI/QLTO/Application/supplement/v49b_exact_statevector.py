"""Localise the remaining model gap by exact statevector, not by sampling.

v49 fixed a kron-ordering bug (Qiskit puts qubit 0 as the LSB, so the walk
unitary must be built V_{n-1} (x) ... (x) V_0) and the complete model then matched
the simulator EXACTLY on some blocks - TVD 0.0091 against a shot floor of 0.0092,
corr 0.9998, versus TVD-from-uniform of 0.31. So the structure is right: the Gram
overlaps <psi_x'|psi_x> were the missing piece.

What remains is block-dependent and SHAPE-PRESERVING: Heisenberg block 0 has
corr 0.9935 with ratio 8.43. Both distributions are normalised, so a high
correlation with a large TVD means the model's modulation AMPLITUDE is wrong, not
its pattern. Guessing which factor is responsible is exactly the mode that failed
nine times today, so don't.

Instead compare against the circuit's EXACT statevector. The walk circuit without
measurements is a unitary on 1 + n + N qubits - 512 amplitudes at n = N = 4 - so
the anc=1 branch can be extracted exactly and compared to the model term by term.
No shot noise, no post-selection statistics, and any difference is a modelling
error with a definite location.

THREE COMPARISONS, each isolating one factor:

    walk only       V applied, imprint set to t = 0. Tests the walk unitary and
                    the kron ordering alone.
    imprint only    imprint applied, drift and mixer set to zero. Tests U_t, the
                    Gram structure and the branch algebra alone.
    full            both, which is the shipped circuit.

Whichever comparison fails is the factor that is wrong, and the ratio of the
model's branch norm to the circuit's gives the size of the error directly rather
than through a TVD.
"""
import sys, os, contextlib, io, itertools
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import efficient_su2, PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Statevector, Operator
import nisq_v3
sys.path.insert(0, os.path.join(APP, 'supplement'))
with contextlib.redirect_stdout(io.StringIO()):
    from v41_oracle_balance import heis, maxcut, E
    from v42_degree2_drift import sense_deg12
    from v49_complete_model import walk_unitary, tvd

R, DT, KS = 0.6, 0.5, 15
PROBLEMS = [("MaxCut N=4", maxcut(4)), ("Heisenberg N=4", heis(4))]


def build_circuit(q, centre, act, gvec, k, dt, R, n, N, use_imprint):
    """The walk circuit with NO measurements, so its unitary can be read off."""
    anc = AncillaRegister(1, 'anc')
    param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(N, 'sys')
    qc = QuantumCircuit(anc, param, sysr)
    qc.h(anc)
    qc.h(param)
    qc.append(q.build_w_gate(param, sysr, centre, R, act),
              list(param) + list(sysr))
    if use_imprint:
        qc.append(PauliEvolutionGate(q.H_sense, time=dt * np.pi,
                                     synthesis=LieTrotter(reps=1)).control(1),
                  [anc[0]] + list(sysr))
    gain = 1.0 / np.sqrt(max(R, 1e-9))
    for step in range(k):
        s = (step + 0.5) / k
        gamma = s * np.pi * dt
        beta = (1.0 - s) * np.pi * dt
        for i in range(n):
            qc.crz(gvec[i] * gamma * 0.5 * np.pi * gain, anc[0], param[i])
        for i in range(n):
            qc.crx(beta, anc[0], param[i])
    qc.h(anc)
    return qc


def circuit_branch(qc, n, N):
    """P(param) conditioned on anc=1, exactly, from the statevector."""
    psi = Statevector(qc).data
    P = np.zeros(2 ** n)
    for i, a in enumerate(psi):
        if i & 1:                                  # anc is qubit 0, the LSB
            y = (i >> 1) & (2 ** n - 1)
            P[y] += abs(a) ** 2
    return P


print("=" * 92)
print("EXACT STATEVECTOR — localising the model gap without sampling")
print("=" * 92)
print(f"  R={R}, dt={DT}, k={KS}. No measurements, no shots: the walk circuit's")
print(f"  unitary is read directly and its anc=1 branch compared to the model.")
print()
print(f"  {'problem':>15}{'blk':>4}{'arm':>14}{'TVD':>10}{'corr':>9}"
      f"{'anc=1 wt':>10}{'model wt':>10}")
print("  " + "-" * 72)

for name, H in PROBLEMS:
    N = H.num_qubits
    ansatz = efficient_su2(N, reps=1)
    M = ansatz.num_parameters
    with contextlib.redirect_stdout(io.StringIO()):
        q = nisq_v3.QLTOv3(ansatz, H, shot_budget=65536, sim_seed=17,
                           merged_walk=False)
    BLK = [b['params'] for b in q.layers if b['params']]
    centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

    for bi, act in enumerate(BLK):
        n = len(act)
        sig = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        idx = np.array([int(''.join('1' if s[i] > 0 else '0'
                                    for i in range(n))[::-1], 2) for s in sig])
        q.reset_shot_stream()
        g1, _ = sense_deg12(q, centre, R, act)

        # |psi_x> TAKEN FROM THE W GATE ITSELF, not from assign_parameters.
        # A controlled Z-rotation can differ from the assigned one by a global
        # phase e^{-i theta/2}, and a global phase that depends on x becomes a
        # RELATIVE phase inside sum_x |x>|psi_x>. RY is real so it has no such
        # ambiguity, which is exactly why the RY blocks (0, 2) matched to
        # machine precision while the RZ blocks (1, 3) did not.
        wq = QuantumCircuit(QuantumRegister(n, 'param'), QuantumRegister(N, 'sys'))
        wq.h(range(n))
        wq.append(q.build_w_gate(wq.qregs[0], wq.qregs[1], centre, R, act),
                  list(range(n + N)))
        wpsi = Statevector(wq).data
        psis = np.zeros((2 ** n, 2 ** N), dtype=complex)
        for i, a in enumerate(wpsi):
            y = i & (2 ** n - 1)                   # param occupies the low bits
            s = i >> n
            psis[y, s] = a
        psis *= np.sqrt(2 ** n)                    # undo the 1/sqrt(2^n) from H

        for tag, gv, use_imp in (('walk only', g1, False),
                                 ('imprint only', np.zeros(n), True),
                                 ('full', g1, True)):
            qc = build_circuit(q, centre, act, gv, KS, DT, R, n, N, use_imp)
            P_exact = circuit_branch(qc, n, N)
            w_exact = float(P_exact.sum())
            P_exact = P_exact / max(w_exact, 1e-18)

            if use_imp:
                ev = PauliEvolutionGate(q.H_sense, time=DT * np.pi,
                                        synthesis=LieTrotter(reps=1))
                Ut = np.asarray(Operator(ev).data)
            else:
                Ut = np.eye(2 ** N, dtype=complex)
            V = walk_unitary(gv, KS, DT, R, n)
            branch = psis - V @ (psis @ Ut.T)
            Pm = np.sum(np.abs(branch) ** 2, axis=1)
            w_model = float(Pm.sum()) / (4.0 * 2 ** n)   # (I-U)/2 and the 1/2^n
            Pm = Pm / max(Pm.sum(), 1e-18)

            cc = float(np.corrcoef(Pm, P_exact)[0, 1]) \
                if P_exact.std() > 1e-14 and Pm.std() > 1e-14 else float('nan')
            print(f"  {name if tag == 'walk only' else '':>15}"
                  f"{bi if tag == 'walk only' else '':>4}{tag:>14}"
                  f"{tvd(Pm, P_exact):>10.5f}{cc:>9.4f}"
                  f"{w_exact:>10.5f}{w_model:>10.5f}", flush=True)
        print("  " + "." * 72)

print()
print("  'walk only' failing isolates the walk unitary or the kron ordering.")
print("  'imprint only' failing isolates U_t, the Gram structure or the branch")
print("  algebra. Both passing while 'full' fails would mean the two factors do")
print("  not compose the way the model assumes - which would be the interesting")
print("  case, since the model applies them as V (x) U_t on a product state and")
print("  the circuit applies them in sequence to an ENTANGLED one.")
