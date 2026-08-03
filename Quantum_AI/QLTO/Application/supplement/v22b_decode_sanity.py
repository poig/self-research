"""Falsification test for the V4 decode, independent of any gradient comparison.

A cosine of 0.97 can hide a systematic error - a wrong bit order or a flipped
sign still correlates with the truth if the error is small or structured. So test
the decode against something it must match EXACTLY, not approximately.

AT R = 0 the W-gate maps every vertex to the same point theta_c, so the per-shot
energy is an unbiased sample of <H>(theta_c) with no smearing, no bias, no O(R^2)
term. The measured mean must therefore equal the exact expectation to shot noise
and nothing else. Any register-order or sign bug shows up here as a gross
mismatch rather than a small degradation.

Second check: the gradient must FLIP SIGN with the perturbation. Reading the
marginal with the x_i=1 and x_i=0 bins swapped is exactly the bit-order failure
mode, and it is invisible to |cos| but flips its sign.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit.quantum_info import Statevector
import benchmark as B
import nisq_v3
from importlib import util as _u

_spec = _u.spec_from_file_location(
    "_v22", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v22_v4_accuracy.py"))

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

# reimplemented here rather than imported, so this file tests the LOGIC and not
# a shared helper that could carry the same bug
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def build(q, c, R, act, group):
    nq = q.ansatz.num_qubits
    param = QuantumRegister(len(act), 'param'); sysr = QuantumRegister(nq, 'sys')
    qc = QuantumCircuit(param, sysr, ClassicalRegister(len(act), 'cp'),
                        ClassicalRegister(nq, 'cs'))
    qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    basis = ['I'] * nq
    for p in group.paulis:
        lbl = str(p)[::-1]
        for qi in range(nq):
            if lbl[qi] != 'I':
                basis[qi] = lbl[qi]
    for qi, b in enumerate(basis):
        if b == 'X':
            qc.h(sysr[qi])
        elif b == 'Y':
            qc.sdg(sysr[qi]); qc.h(sysr[qi])
    qc.measure(param, qc.cregs[0]); qc.measure(sysr, qc.cregs[1])
    return qc


def shot_energy(grp, sysbits):
    e = 0.0
    for coeff, p in zip(grp.coeffs, grp.paulis):
        lbl = str(p)[::-1]
        par = sum(int(sysbits[qq]) for qq in range(len(lbl)) if lbl[qq] != 'I')
        e += float(np.real(coeff)) * (1 if par % 2 == 0 else -1)
    return e


def mean_energy_and_grad(q, c, R, act, groups):
    tot_e, tot_n = 0.0, 0
    acc = np.zeros(len(act))
    for grp in groups:
        counts = q._run(build(q, c, R, act, grp))
        num = np.zeros((2, len(act))); den = np.zeros((2, len(act)))
        ge, gn = 0.0, 0
        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            sysbits, xbits = parts[0][::-1], parts[1][::-1]
            e = shot_energy(grp, sysbits)
            ge += e * cnt; gn += cnt
            for i in range(len(act)):
                b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                num[b, i] += e * cnt; den[b, i] += cnt
        tot_e += ge / max(gn, 1); tot_n = 1
        m1 = np.divide(num[1], den[1], out=np.zeros(len(act)), where=den[1] > 0)
        m0 = np.divide(num[0], den[0], out=np.zeros(len(act)), where=den[0] > 0)
        acc += (m1 - m0) / (2.0 * R + 1e-12) if R > 0 else 0.0
    return tot_e, acc


def exact_grad(ansatz, H, c, act):
    g = np.zeros(len(act))
    for j, i in enumerate(act):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[j] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp)).expectation_value(H)))
                      - float(np.real(Statevector(ansatz.assign_parameters(pm)).expectation_value(H))))
    return g


PROBLEMS = [("H2", B.get_h2_problem),
            ("MaxCut N=4", lambda: B.get_maxcut_problem(4)),
            ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4))]
SHOTS = 65536

print("=" * 92)
print("V4 DECODE SANITY — two tests the decode must pass EXACTLY, not approximately")
print("=" * 92)
print(f"  {SHOTS} shots/circuit. H_sense is traceless, so compare against <H_sense>,")
print("  not <H>: the identity term is stripped before sensing.")
print()
print(f"  {'problem':<18}{'R':>5}{'measured <H>':>14}{'exact <H_sense>':>17}"
      f"{'abs err':>10}{'shot sigma':>12}")
print("  " + "-" * 76)

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=SHOTS)
    groups = q.H_sense.group_commuting(qubit_wise=True)
    act = [b['params'] for b in q.layers if b['params']][0]
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)

    sv = Statevector(ansatz.assign_parameters(c))
    ex = float(np.real(sv.expectation_value(q.H_sense)))
    var = float(np.real(sv.expectation_value((q.H_sense @ q.H_sense).simplify()))) - ex ** 2
    sig = np.sqrt(max(var, 0) / SHOTS) * len(groups)

    meas, _ = mean_energy_and_grad(q, c, 0.0, act, groups)
    print(f"  {name:<18}{0.0:>5.1f}{meas:>14.5f}{ex:>17.5f}"
          f"{abs(meas-ex):>10.5f}{sig:>12.5f}")

print()
print("  A register-order or sign bug does NOT survive this: at R=0 there is no")
print("  smearing and no O(R^2) term, so the only permitted discrepancy is shot noise.")

print()
print(f"  {'problem':<18}{'R':>5}{'cos(g,gx)':>11}{'SIGNED cos':>12}"
      f"{'|g|/|gx|':>10}{'sinc(R)':>9}")
print("  " + "-" * 65)
for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=SHOTS)
    groups = q.H_sense.group_commuting(qubit_wise=True)
    act = [b['params'] for b in q.layers if b['params']][0]
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, ansatz.num_parameters)
    gx = exact_grad(ansatz, H, c, act)
    R = 0.6
    _, g = mean_energy_and_grad(q, c, R, act, groups)
    cs = float(g @ gx / (np.linalg.norm(g) * np.linalg.norm(gx) + 1e-15))
    print(f"  {name:<18}{R:>5.1f}{abs(cs):>11.4f}{cs:>12.4f}"
          f"{np.linalg.norm(g)/np.linalg.norm(gx):>10.4f}{np.sin(R)/R:>9.4f}")

print()
print("  SIGNED cos must be POSITIVE. Swapping the x_i=1 and x_i=0 bins - the")
print("  bit-order failure mode - leaves |cos| untouched and flips the sign, so the")
print("  signed column is what actually tests the ordering.")
