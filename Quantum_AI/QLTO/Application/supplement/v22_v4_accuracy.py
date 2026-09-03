"""Does the direct-readout variant actually OPTIMISE, or only look good on cosines?

v18 measured V4's gradient quality (cos 0.974-0.991, equal or better than QPE) and
I have been quoting cost projections off the back of it ever since. But a gradient
cosine is not an optimisation result, and there is a specific reason to doubt the
step from one to the other.

THE OBJECTION. V3 reads the energy from a k-bit ANCILLA, so the measured register
spans 2^(n+k) outcomes with k=4. V4 reads it from the SYSTEM register, so the
outcome space is 2^(n+N) and grows exponentially in system size. If the estimator
needed to resolve that distribution, V4 would fall over exactly where it matters.

WHY IT SHOULD NOT, and this is T2 once more: conditioned on vertex x,

    E[E_g(s) | x] = sum_s |<s|B_g U(theta_x)|0>|^2 E_g(s) = <psi(theta_x)|H_g|psi(theta_x)>

so the per-shot number is an unbiased sample of the vertex energy whatever 2^N is.
The estimator takes a MEAN of a bounded quantity - it never estimates the
distribution over s - so support size does not enter, and Var stays Var(H_g). Both
variants also collapse to ONE vertex per shot, so neither gets 2^n evaluations
free; the hypercube buys coverage, not parallel evaluation.

That is an argument. This file is the measurement. Identical walk, identical
schedule, identical k_steps, PAIRED seeds - the ONLY difference is where the
energy comes from:

    v3    QPE sensing  (1 circuit/block, k=4 ancillas, ~1976 depth at N=6)
    v4    direct Pauli (G circuits/block, ~21 depth at N=6)

If V4's final energies match V3's, the cost projections stand. If they do not,
they are worthless and the cheap variant is cheap because it does less.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorEstimator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

EST = StatevectorEstimator()
def energy_at(ansatz, H, p):
    return float(EST.run([(ansatz, H, np.asarray([p]))]).result()[0].data.evs.ravel()[0])


def group_basis(group, nq):
    basis = ['I'] * nq
    for p in group.paulis:
        lbl = str(p)[::-1]
        for q in range(nq):
            if lbl[q] != 'I':
                basis[q] = lbl[q]
    return basis


def build_direct(q, c, R, act, group):
    """W gate, rotate into the group's basis, measure BOTH registers."""
    nq = q.ansatz.num_qubits
    param = QuantumRegister(len(act), 'param')
    sysr = QuantumRegister(nq, 'sys')
    qc = QuantumCircuit(param, sysr,
                        ClassicalRegister(len(act), 'c_param'),
                        ClassicalRegister(nq, 'c_sys'))
    qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    for qi, b in enumerate(group_basis(group, nq)):
        if b == 'X':
            qc.h(sysr[qi])
        elif b == 'Y':
            qc.sdg(sysr[qi]); qc.h(sysr[qi])
    qc.measure(param, qc.cregs[0])
    qc.measure(sysr, qc.cregs[1])
    return qc


def direct_gradient(q, c, R, act, groups):
    """Sum of per-group marginals. <H> = sum_g <H_g>, so gradients add."""
    g_out = np.zeros(len(c))
    acc = np.zeros(len(act))
    for grp in groups:
        counts = q._run(build_direct(q, c, R, act, grp))
        num = np.zeros((2, len(act))); den = np.zeros((2, len(act)))
        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            sysbits = parts[0][::-1]          # c_sys registered last -> printed first
            xbits = parts[1][::-1]
            e = 0.0
            for coeff, p in zip(grp.coeffs, grp.paulis):
                lbl = str(p)[::-1]
                par = sum(int(sysbits[qq]) for qq in range(len(lbl)) if lbl[qq] != 'I')
                e += float(np.real(coeff)) * (1 if par % 2 == 0 else -1)
            for i in range(len(act)):
                b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                num[b, i] += e * cnt; den[b, i] += cnt
        m1 = np.divide(num[1], den[1], out=np.zeros(len(act)), where=den[1] > 0)
        m0 = np.divide(num[0], den[0], out=np.zeros(len(act)), where=den[0] > 0)
        acc += (m1 - m0) / (2.0 * R + 1e-12)
    g_out[act] = acc
    return g_out


def run(prob, arm, seed, shots, epochs=20, k=15):
    ansatz, H = prob()
    q = Q(ansatz, H, shot_budget=shots, num_ancillas=4)
    groups = H.group_commuting(qubit_wise=True)
    BLK = [b['params'] for b in q.layers if b['params']]
    p = np.random.RandomState(seed).uniform(-np.pi, np.pi, ansatz.num_parameters)
    ncirc = 0
    for ep in range(epochs):
        R = max(0.6 * (0.9 ** ep), 1e-4)
        dt = max(0.5 * (0.95 ** (ep + 1)), 0.01)
        for act in BLK:
            if arm == 'v3':
                grad = q.sense_gradient(p, R, act); ncirc += 1
            else:
                grad = direct_gradient(q, p, R, act, groups); ncirc += len(groups)
            p = q._execute_walk(p, k, dt, R, act, grad); ncirc += 1
    return energy_at(ansatz, H, p), ncirc


PROBLEMS = [
    ("H2",             B.get_h2_problem),
    ("MaxCut N=4",     lambda: B.get_maxcut_problem(4)),
    ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
    ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
]
SEEDS = (42, 43, 44, 45, 46, 47)
SHOTS = 8192

print("=" * 96)
print("V4 DIRECT READOUT — END-TO-END ACCURACY against V3's QPE sensing")
print("=" * 96)
print(f"  {len(SEEDS)} PAIRED seeds, 20 epochs, k_steps=15, {SHOTS} shots/circuit.")
print("  Identical walk and schedule; the ONLY difference is the energy readout.")
print()
print(f"  {'problem':<18}{'exact':>10}{'V3 QPE':>10}{'V4 direct':>11}{'diff':>9}"
      f"{'sigma':>7}{'V3 circ':>9}{'V4 circ':>9}")
print("  " + "-" * 83)

for name, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    prob = (lambda a, h: (lambda: (a, h)))(ansatz, H)
    exact = float(np.min(np.linalg.eigvalsh(H.to_matrix())))

    e3, e4, c3, c4 = [], [], 0, 0
    for s in SEEDS:
        v3, n3 = run(prob, 'v3', s, SHOTS)
        v4, n4 = run(prob, 'v4', s, SHOTS)
        e3.append(v3); e4.append(v4); c3, c4 = n3, n4
    e3 = np.array(e3); e4 = np.array(e4)
    d = e4 - e3                                   # paired: >0 means V4 WORSE
    sem = d.std(ddof=1) / np.sqrt(len(SEEDS))
    print(f"  {name:<18}{exact:>10.4f}{e3.mean():>10.4f}{e4.mean():>11.4f}"
          f"{d.mean():>+9.4f}{abs(d.mean())/max(sem,1e-9):>7.1f}{c3:>9}{c4:>9}",
          flush=True)

print()
print("  diff = V4 - V3 on PAIRED seeds, so POSITIVE means V4 reached a higher")
print("  (worse) energy. Sigma is on the paired difference. These notes record two")
print("  sub-2-sigma results that reversed on replication, so read <2 sigma as a")
print("  tie, not as a small effect.")
print()
print("  If V4 ties V3 here, the 2^N objection is answered empirically as well as")
print("  by T2, and the billing projections in v21 stand. If V4 loses, the cheap")
print("  variant is cheap because it does less, and v21's flat-in-N column is void.")
