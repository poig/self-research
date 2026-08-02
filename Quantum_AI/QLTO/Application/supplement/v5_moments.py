"""Are ALL moments of H free from the QPE sensing shots?

The docs route the second moment through the X basis, Re<U> ~ 1 - tau^2<H^2>/2.
Two problems with that now: it is a HADAMARD-path readout, and the k=1 path
measured 1.91x worse than parameter-shift on shots while carrying an irreducible
sin() bias; and the signal is tau^2-suppressed, so its relative precision is poor.

QPE should make it free instead. Phase estimation samples eigenvalues with
probability |<E_k|psi>|^2, so over shots

    E[e] = <H>,   E[e^2] = <H^2>,   E[e^m] = <H^m>

and e^m is a PER-SHOT quantity, so its Walsh coefficients are empirical means -
LINEAR functionals, hence unbiased at any shots-per-vertex by T2, exactly like the
first moment. If that holds, the folded-spectrum objective <(H-omega)^2>, the
variance preconditioner, and the metrology QFI all come out of shots already being
taken, with no new circuit element.

THE RISK IS WRAP. QPE folds the phase into one turn, so any eigenvalue outside
+-margin*||H0|| aliases. The second moment weights the tails harder than the
first, so wrap error could bias <H^2> even where <H> is fine. That is what this
measures: both moments against exact, at several qpe_margin values.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import QFT, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)


def qpe_samples(q, c, R, act):
    """Per-shot decoded energies and their param bits, from one sensing circuit."""
    n, k = len(act), q.num_ancillas
    anc = AncillaRegister(k, 'anc'); param = QuantumRegister(n, 'param')
    sysr = QuantumRegister(q.ansatz.num_qubits, 'sys')
    qc = QuantumCircuit(anc, param, sysr,
                        ClassicalRegister(n, 'c'), ClassicalRegister(k, 'a'))
    qc.h(anc); qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    for a in range(k):
        qc.append(PauliEvolutionGate(
            q.H_sense, time=(2 ** a) * q.tau0,
            synthesis=SuzukiTrotter(order=2, reps=max(1, (2 ** a) // 2))
        ).control(1), [anc[a]] + list(sysr))
    qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
    qc.measure(param, qc.cregs[0]); qc.measure(anc, qc.cregs[1])
    out = []
    for bs, cnt in q._run(qc).items():
        parts = bs.split()
        if len(parts) != 2:
            continue
        m = int(parts[0], 2); phi = m / (2 ** k)
        if phi >= 0.5:
            phi -= 1.0
        e = -2.0 * np.pi * phi / (q.tau0 + 1e-12)
        sg = np.array([1.0 if (i < len(parts[1][::-1]) and parts[1][::-1][i] == '1')
                       else -1.0 for i in range(n)])
        out.append((e, sg, cnt))
    return out


ansatz, H, _ = B.get_heisenberg_problem(4)
H2 = (H @ H).simplify()
c = np.random.RandomState(3).uniform(-np.pi, np.pi, ansatz.num_parameters)
R = 0.6

print("=" * 86)
print("Moment recovery from QPE sensing shots, vs exact, averaged over the cube")
print("=" * 86)
print("  <H> and <H^2> here are means over the SAME uniform superposition of")
print("  vertices the sensing circuit prepares, so the exact reference is the")
print("  corner average - computed by enumeration.")
print()
q0 = Q(ansatz, H, shot_budget=8192, num_ancillas=4)
BLK = [b['params'] for b in q0.layers]
act = BLK[0]
n = len(act)
sig = np.array([[1.0 if (v >> i) & 1 else -1.0 for i in range(n)]
                for v in range(2 ** n)])
ex1 = ex2 = 0.0
w1 = np.zeros(n); w2 = np.zeros(n)
for s in sig:
    p = c.copy(); p[act] = c[act] + R * s
    sv = Statevector(ansatz.assign_parameters(p))
    e1 = float(np.real(sv.expectation_value(H)))
    e2 = float(np.real(sv.expectation_value(H2)))
    ex1 += e1 / len(sig); ex2 += e2 / len(sig)
    w1 += e1 * s / len(sig); w2 += e2 * s / len(sig)

print(f"  {'k':>3}{'margin':>8}{'<H> meas':>11}{'<H> exact':>11}"
      f"{'<H2> meas':>12}{'<H2> exact':>12}{'H2 err %':>10}")
print("  " + "-" * 67)
for k, margin in ((4, 2.0), (5, 2.0), (6, 2.0), (6, 1.2), (6, 4.0), (7, 2.0)):
    q = Q(ansatz, H, shot_budget=65536, num_ancillas=k, qpe_margin=margin)
    S = qpe_samples(q, c, R, act)
    tot = sum(cnt for _, _, cnt in S)
    m1 = sum(e * cnt for e, _, cnt in S) / tot
    m2 = sum(e * e * cnt for e, _, cnt in S) / tot
    print(f"  {k:>3}{margin:>8.1f}{m1:>11.4f}{ex1:>11.4f}{m2:>12.3f}"
          f"{ex2:>12.3f}{100*(m2-ex2)/abs(ex2):>9.1f}%", flush=True)

print()
print("  Now the thing that actually matters: are the degree-1 Walsh coefficients")
print("  of e^2 recovered? Those are what a folded-spectrum gradient needs.")
print()
q = Q(ansatz, H, shot_budget=65536, num_ancillas=6, qpe_margin=2.0)
acc1, acc2 = [], []
for _ in range(5):
    S = qpe_samples(q, c, R, act)
    tot = sum(cnt for _, _, cnt in S)
    a1 = sum(e * sg * cnt for e, sg, cnt in S) / tot
    a2 = sum(e * e * sg * cnt for e, sg, cnt in S) / tot
    acc1.append(a1); acc2.append(a2)
m1 = np.mean(acc1, axis=0); m2 = np.mean(acc2, axis=0)
print(f"  deg1 of <H>   exact {np.array2string(w1, precision=4)}")
print(f"                meas  {np.array2string(m1, precision=4)}")
print(f"    cos = {float(w1@m1/(np.linalg.norm(w1)*np.linalg.norm(m1))):.5f}"
      f"   norm ratio = {np.linalg.norm(m1)/np.linalg.norm(w1):.4f}")
print(f"  deg1 of <H^2> exact {np.array2string(w2, precision=3)}")
print(f"                meas  {np.array2string(m2, precision=3)}")
print(f"    cos = {float(w2@m2/(np.linalg.norm(w2)*np.linalg.norm(m2))):.5f}"
      f"   norm ratio = {np.linalg.norm(m2)/np.linalg.norm(w2):.4f}")
print()
print("  cos ~ 1 on the SECOND row is the claim: a folded-spectrum gradient is")
print("  available from shots already taken, with no X basis and no tau^2 penalty.")
