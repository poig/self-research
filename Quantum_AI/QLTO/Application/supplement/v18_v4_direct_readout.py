"""V4 hypothesis: the circuit-count win does not need QPE at all.

v16/v17 showed the QPE ladder costs 19-141x the depth and 5-40x the two-qubit
gates of parameter-shift, which is what made the "cheaper" claim collapse. But
QPE was never the source of the advantage. Reread T1: the marginal over param bit
i is the degree-1 Walsh coefficient of the ENERGY, and it is unbiased at any
shots-per-vertex because it is LINEAR in whatever energy number each shot
carries. The estimator does not care where that number came from.

The advantage therefore comes from the PARAMETER SUPERPOSITION - one circuit
serving all M components - and QPE was only ever the energy readout. It bought
exactly one thing: G-independence, reading <H> in a single measurement setting
whatever H looks like. That is worth an exponential depth ladder only if G is
huge; on these problems G is 1-3.

So replace it with the boring readout: rotate into each qubit-wise-commuting
basis, measure the param register AND the system register in the same shot, and
compute the per-shot energy classically from the system bits. Same decode, same
theorem, no ancilla, no time evolution, no Trotter error.

  circuits   B*G          against parameter-shift's 2*M*G  -> ratio B/(2M),
                          and the G CANCELS, so the win is Hamiltonian-independent
  depth      ansatz + W   instead of ansatz + W + (2^k - 1)*tau0 of evolution

Predicted: same circuit-count win, a small multiple of the ansatz depth instead
of 19-141x, and FEWER total two-qubit gates than parameter-shift rather than more.
This script checks all three, and checks the gradient still points the right way.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']
BACKEND = AerSimulator()


def group_basis(group, nq):
    """Per-qubit measurement basis for a qubit-wise-commuting group."""
    basis = ['I'] * nq
    for p in group.paulis:
        lbl = str(p)[::-1]
        for q in range(nq):
            if lbl[q] != 'I':
                basis[q] = lbl[q]
    return basis


def build_direct(q, c, R, act, group):
    """W gate, rotate to the group's basis, measure BOTH registers."""
    nq = q.ansatz.num_qubits
    param = QuantumRegister(len(act), 'param')
    sysr = QuantumRegister(nq, 'sys')
    cp = ClassicalRegister(len(act), 'c_param')
    cs = ClassicalRegister(nq, 'c_sys')
    qc = QuantumCircuit(param, sysr, cp, cs)

    qc.h(param)
    qc.append(q.build_w_gate(param, sysr, c, R, act), list(param) + list(sysr))
    for qi, b in enumerate(group_basis(group, nq)):
        if b == 'X':
            qc.h(sysr[qi])
        elif b == 'Y':
            qc.sdg(sysr[qi]); qc.h(sysr[qi])
    qc.measure(param, cp)
    qc.measure(sysr, cs)
    return qc


def direct_gradient(q, c, R, shots, seed=None):
    """One circuit per (block, group). Marginal decode, identical to the QPE one."""
    M = len(c)
    grad = np.zeros(M)
    groups = q.H_sense.group_commuting(qubit_wise=True)
    ncirc, depths, cxs = 0, [], []

    for blk in q.layers:
        act = blk['params']
        if not act:
            continue
        # <H> = sum_g <H_g>, so the gradient is the SUM of the per-group
        # marginals, not their average. Accumulating num/den across groups
        # would divide by G.
        gblk = np.zeros(len(act))

        for g in groups:
            num = np.zeros((2, len(act)))
            den = np.zeros((2, len(act)))
            qc = build_direct(q, c, R, act, g)
            t = transpile(qc, basis_gates=BASIS, optimization_level=1)
            depths.append(t.depth()); cxs.append(t.count_ops().get('cx', 0))
            counts = BACKEND.run(t, shots=shots, seed_simulator=seed).result().get_counts()
            ncirc += 1

            for bitstr, cnt in counts.items():
                parts = bitstr.split()
                sysbits = parts[0][::-1]      # c_sys registered last -> printed first
                xbits = parts[1][::-1]
                e = 0.0
                for coeff, p in zip(g.coeffs, g.paulis):
                    lbl = str(p)[::-1]
                    par = sum(int(sysbits[qq]) for qq in range(len(lbl))
                              if lbl[qq] != 'I')
                    e += float(np.real(coeff)) * (1 if par % 2 == 0 else -1)
                for i in range(len(act)):
                    b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                    num[b, i] += e * cnt
                    den[b, i] += cnt

            m1 = np.divide(num[1], den[1], out=np.zeros(len(act)), where=den[1] > 0)
            m0 = np.divide(num[0], den[0], out=np.zeros(len(act)), where=den[0] > 0)
            gblk += (m1 - m0) / (2.0 * R + 1e-12)

        grad[act] = gblk

    return grad, ncirc, float(np.mean(depths)), int(sum(cxs))


def exact_gradient(ansatz, H, c):
    g = np.zeros(len(c))
    for i in range(len(c)):
        pp = c.copy(); pp[i] += np.pi / 2
        pm = c.copy(); pm[i] -= np.pi / 2
        g[i] = 0.5 * (float(np.real(Statevector(ansatz.assign_parameters(pp))
                                    .expectation_value(H)))
                      - float(np.real(Statevector(ansatz.assign_parameters(pm))
                                      .expectation_value(H))))
    return g


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (nu * nv)) if nu > 1e-15 and nv > 1e-15 else 0.0


PROBLEMS = [
    ("H2",             B.get_h2_problem),
    ("MaxCut N=4",     lambda: B.get_maxcut_problem(4)),
    ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
    ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
]
SHOTS, REP, R = 4096, 5, 0.6

print("=" * 98)
print("V4 DIRECT READOUT — does the circuit-count win survive without QPE?")
print("=" * 98)
print(f"  R={R}, {SHOTS} shots/circuit, {REP} repeats. QPE numbers are k=4, the shipping default.")

for pname, fn in PROBLEMS:
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz, H, _ = fn()
    q = Q(ansatz, H, shot_budget=SHOTS)
    M = ansatz.num_parameters
    c = np.random.RandomState(7).uniform(-np.pi, np.pi, M)
    gx = exact_gradient(ansatz, H, c)
    G = len(H.group_commuting(qubit_wise=True))

    # baseline: parameter-shift circuit cost
    a = ansatz.assign_parameters(c); a.measure_all()
    ta = transpile(a, basis_gates=BASIS, optimization_level=1)
    d_ps, cx_ps, n_ps = ta.depth(), ta.count_ops().get('cx', 0), 2 * M * G

    # QPE k=4 cost, for the comparison
    dq, cq, nq_ = [], [], 0
    for blk in q.layers:
        if not blk['params']:
            continue
        tq = transpile(q._build_qpe_sensing_circuit(c, R, blk['params']),
                       basis_gates=BASIS, optimization_level=1)
        dq.append(tq.depth()); cq.append(tq.count_ops().get('cx', 0)); nq_ += 1

    runs = [direct_gradient(q, c, R, SHOTS, seed=1000 + r) for r in range(REP)]
    cs = [cos(g, gx) for g, _, _, _ in runs]
    gmean = np.mean([g for g, _, _, _ in runs], axis=0)
    _, ncirc, dep, cxtot = runs[0]

    # QPE gradient quality at the same shots, for the comparison
    qcs = []
    for _ in range(REP):
        gq = np.zeros(M)
        for blk in q.layers:
            if blk['params']:
                gq += q.sense_gradient(c, R, blk['params'])
        qcs.append(cos(gq, gx))

    print(f"\n  ===== {pname} | M={M} | G={G} | qubits {ansatz.num_qubits} =====")
    print(f"  {'method':<20}{'circuits':>9}{'depth':>8}{'total cx':>10}"
          f"{'cos':>9}{'x PS cx':>10}")
    print("  " + "-" * 66)
    print(f"  {'parameter-shift':<20}{n_ps:>9}{d_ps:>8}{n_ps*cx_ps:>10}"
          f"{'1.0000':>9}{1.0:>10.2f}")
    print(f"  {'QLTO V3 (QPE k=4)':<20}{nq_:>9}{int(np.mean(dq)):>8}{sum(cq):>10}"
          f"{np.mean(qcs):>9.4f}{sum(cq)/max(n_ps*cx_ps,1):>10.2f}")
    print(f"  {'QLTO V4 (direct)':<20}{ncirc:>9}{int(dep):>8}{cxtot:>10}"
          f"{np.mean(cs):>9.4f}{cxtot/max(n_ps*cx_ps,1):>10.2f}")
    print(f"    V4 norm ratio |g|/|gx| = "
          f"{np.linalg.norm(gmean)/np.linalg.norm(gx):.4f}"
          f"   (sinc(R) = {np.sin(R)/R:.4f} is the predicted attenuation)")

print()
print("  The test: does V4 keep circuits << parameter-shift while bringing total")
print("  cx BELOW 1.0x and depth back to a small multiple of the ansatz? If so the")
print("  QPE ladder was a cost with no matching benefit at these G, and V4 is the")
print("  version of this method that can actually run on hardware.")
