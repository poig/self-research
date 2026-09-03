"""Is P(good) = Theta(1) for the direct-readout path? The roadmap's last open check.

research_roadmap_final.md, action plan item 3: "Check AE signal amplitude for
direct-readout path. 1 day. Analytic calculation: is P(good) = Theta(1) for
non-scrambling ansaetze?" It was never run, and it is the last unresolved item in
any of the planning documents.

WHY IT WAS ASKED. The roadmap's Porter-Thomas caveat: "If C_fixed is deeply
scrambling (as needed for BQP-hardness in any theory application), output
probabilities concentrate as O(1/2^N), which could make specific joint
probabilities P(x_i=1 AND P_l=-1) exponentially small. This undercuts the
Theta(1) claim in exactly the regime where hardness is interesting."

Amplitude estimation needs Theta(1/sqrt(a)) iterations to resolve an amplitude a,
so if P(good) collapsed as 2^-N the AE route would be dead on arrival - the
quadratic in eps would be eaten by an exponential in the signal.

WHAT IS BEING MEASURED. From the W-gate state, the joint probability

    P(good) = P(x_i = 1 AND the l-th measured Pauli reads -1)

exactly, by statevector, as ansatz depth grows (reps = 1..5). Reported alongside
the max single-bitstring probability, which IS the quantity Porter-Thomas
concentrates, so the two can be told apart.

THE PREDICTION WORTH STATING FIRST. P(x_i=1) = 1/2 exactly - W is controlled ON
param and cannot move its populations - and P(P_l=-1) = (1 - <P_l>)/2. Scrambling
drives <P_l> toward 0, which sends P(P_l=-1) toward 1/2, NOT toward 0. So the
good event is a TWO-BIT MARGINAL and should stay Theta(1) however deep the
circuit gets, while individual 2^N-outcome bitstring probabilities do collapse.
If that holds, the Porter-Thomas caveat is aimed at the wrong object and item 3
resolves in AE's favour - which does not revive AE, since v50/v60 and the
T2-forfeiture argument already demoted it, but it does close the question.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp, Statevector
import nisq_v5


def heis(N):
    ops = []
    for i in range(N - 1):
        for p in "XYZ":
            s = ["I"] * N
            s[i] = s[i + 1] = p
            ops.append(("".join(s), 1.0))
    return SparsePauliOp.from_list(ops)


R = 0.6

print("=" * 96)
print("AE SIGNAL AMPLITUDE — is P(good) Theta(1), or does Porter-Thomas kill it?")
print("=" * 96)
print(f"  R={R}, exact statevector. P(good) = P(x_i=1 AND Pauli_l = -1) on the")
print(f"  W-gate state, block 0, averaged over qubits and over the group's terms.")
print(f"  'max bitstring' is the largest single joint outcome probability - the")
print(f"  quantity Porter-Thomas actually concentrates, shown for contrast.")
print()
print(f"  {'N':>3}{'reps':>6}{'M':>5}{'n':>4}{'P(x_i=1)':>11}{'P(good) min':>13}"
      f"{'P(good) mean':>14}{'max bitstring':>15}{'unif 2^-(n+N)':>15}")
print("  " + "-" * 86)

for N in (4, 6):
    H = heis(N)
    for reps in (1, 2, 3, 4, 5):
        a = efficient_su2(N, reps=reps)
        M = a.num_parameters
        with contextlib.redirect_stdout(io.StringIO()):
            q = nisq_v5.QLTOv5(a, H, shot_budget=256, gradient_mode='direct')
        act = q.layers[0]['params']
        n = len(act)
        centre = np.random.RandomState(11).uniform(-np.pi, np.pi, M)

        group = q.groups[0]
        qc = QuantumCircuit(QuantumRegister(n, 'param'), QuantumRegister(N, 'sys'))
        qc.h(range(n))
        q._build_w(qc, qc.qregs[0], qc.qregs[1], centre, R, act)
        q._basis(qc, qc.qregs[1], group)
        psi = Statevector(qc).data
        p = np.abs(psi) ** 2                      # index = param + 2^n * sys

        idx = np.arange(len(p))
        pm = idx & (2 ** n - 1)
        sy = idx >> n

        px1 = float(p[(pm >> 0) & 1 == 1].sum())  # P(x_0 = 1)

        # P(good) for every (param bit i, Pauli term l) pair in this group
        goods = []
        labels = group.paulis.to_labels()
        for i in range(n):
            for lbl in labels:
                sup = [k for k, ch in enumerate(reversed(lbl)) if ch != 'I']
                if not sup:
                    continue
                par = np.zeros(len(idx), dtype=int)
                for k in sup:
                    par ^= (sy >> k) & 1
                sel = (((pm >> i) & 1) == 1) & (par == 1)   # x_i=1 AND P_l=-1
                goods.append(float(p[sel].sum()))
        goods = np.array(goods)
        print(f"  {N:>3}{reps:>6}{M:>5}{n:>4}{px1:>11.6f}{goods.min():>13.6f}"
              f"{goods.mean():>14.6f}{p.max():>15.3e}"
              f"{1.0 / 2 ** (n + N):>15.3e}", flush=True)
    print("  " + "." * 86)

print()
print("  P(good) holding near a constant as reps grows means AE's signal does NOT")
print("  collapse and the Porter-Thomas caveat was aimed at the wrong object -")
print("  bitstring probabilities concentrate, two-bit marginals do not. That")
print("  closes action-plan item 3. It does NOT revive AE: the T2 forfeiture")
print("  (Theta(M/eps) against Theta(1/eps^2)) is what demoted it, and that")
print("  argument is untouched by this result.")
