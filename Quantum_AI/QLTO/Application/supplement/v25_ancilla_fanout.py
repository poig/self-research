"""Can V3's controlled evolution be made FLAT in N by fanning out the control?

v24b left V3's sensing depth at N^0.64 after term sorting. The residual is the
ancilla: within each QPE stage every controlled Pauli term is controlled on the
SAME qubit, and gates sharing a qubit cannot run in parallel, so a set of terms
that act on disjoint system qubits still serialises on the control.

THE FIX. A control in the computational basis can be COPIED - |c>|0> -> |c>|c>
by CNOT is not cloning, it is fan-out, and it is exact for c in {0,1} and
therefore exact on superpositions of basis states too (it entangles, and the
uncompute disentangles). So:

    CNOT tree: anc -> F copies                     depth O(log F)
    each disjoint term controlled on its own copy  depth O(layers), flat in N
    CNOT tree reversed                             depth O(log F)

THE COST, stated up front because it is the thing that decides whether this is
worth it: F extra qubits, and 2(F-1) extra CNOTs that were not there before.
Gate count is what fidelity is charged on, so this trades FIDELITY for DEPTH -
the opposite direction from what V3 needs. Measure both.

controlled-exp(-i theta P) = V ; CRZ(2 theta) ; V^dag with V UNCONTROLLED, the
identity already used for the degree-2 CRZZ in T7, so each term needs exactly one
controlled gate and the rest is free Clifford.

CORRECTNESS is checked against the un-fanned circuit by direct unitary
comparison at sizes where the matrix is computable. A depth win on a circuit that
computes something else is worthless.
"""
import sys, os, contextlib, io
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
from qiskit import transpile, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Operator
import benchmark as B
import nisq_v3

_R = nisq_v3.QLTOv3
def Q(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return _R(*a, **k)

BASIS = ['rz', 'sx', 'x', 'cx']


def support(p):
    lbl = str(p)[::-1]
    return [i for i, ch in enumerate(lbl) if ch != 'I']


def layers_of(op):
    """Partition terms into layers of mutually disjoint support."""
    lays = []
    for p, c in zip(op.paulis, op.coeffs):
        s = set(support(p))
        for L in lays:
            if not (s & L['u']):
                L['t'].append((p, c)); L['u'] |= s
                break
        else:
            lays.append({'t': [(p, c)], 'u': set(s)})
    return [L['t'] for L in lays]


def ctrl_term(qc, ctrl, sysr, pauli, theta):
    """controlled-exp(-i theta P) = V ; CRZ(2 theta) ; V^dag, V uncontrolled."""
    lbl = str(pauli)[::-1]
    sup = [i for i, ch in enumerate(lbl) if ch != 'I']
    if not sup:
        return
    for q in sup:                       # diagonalise P
        if lbl[q] == 'X':
            qc.h(sysr[q])
        elif lbl[q] == 'Y':
            qc.sdg(sysr[q]); qc.h(sysr[q])
    for a, b in zip(sup[:-1], sup[1:]):  # collect parity onto sup[-1]
        qc.cx(sysr[a], sysr[b])
    qc.crz(2.0 * theta, ctrl, sysr[sup[-1]])
    for a, b in reversed(list(zip(sup[:-1], sup[1:]))):
        qc.cx(sysr[a], sysr[b])
    for q in sup:
        if lbl[q] == 'X':
            qc.h(sysr[q])
        elif lbl[q] == 'Y':
            qc.h(sysr[q]); qc.s(sysr[q])


def build(op, t, N, fanout, reps=1):
    """One controlled Trotter evolution. fanout=1 reproduces the serial version."""
    lays = layers_of(op)
    width = max(len(L) for L in lays)
    F = width if fanout > 1 else 1
    anc = QuantumRegister(1, 'anc')
    hlp = QuantumRegister(max(F - 1, 0), 'hlp') if F > 1 else None
    sysr = QuantumRegister(N, 'sys')
    regs = [anc] + ([hlp] if hlp is not None else []) + [sysr]
    qc = QuantumCircuit(*regs)

    copies = [anc[0]] + ([hlp[i] for i in range(F - 1)] if F > 1 else [])
    if F > 1:                                # CNOT tree, depth O(log F)
        done = 1
        while done < F:
            for i in range(min(done, F - done)):
                qc.cx(copies[i], copies[done + i])
            done *= 2
    for _ in range(reps):
        for L in lays:
            for j, (p, c) in enumerate(L):
                ctrl_term(qc, copies[j % F], sysr,
                          p, float(np.real(c)) * t / reps)
    if F > 1:                                # uncompute
        done = 1 << (int(np.ceil(np.log2(F))) - 1)
        while done >= 1:
            for i in range(min(done, F - done)):
                qc.cx(copies[i], copies[done + i])
            done //= 2
    return qc


def stats(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=1)
    return t.depth(), t.count_ops().get('cx', 0)


print("=" * 92)
print("ANCILLA FAN-OUT — trading qubits and gates for depth")
print("=" * 92)
print("  One controlled Trotter step of H_sense, terms layer-sorted in both arms.")
print("  serial = every term controlled on one ancilla; fanout = one copy per")
print("  term within a layer, CNOT tree in and out.")
print()
print(f"  {'N':>4}{'layers':>8}{'F':>4}{'depth serial':>14}{'depth fanout':>14}"
      f"{'speedup':>9}{'cx serial':>11}{'cx fanout':>11}{'unitary err':>13}")
print("  " + "-" * 88)

rows = []
for N in (4, 6, 8, 10, 12):
    with contextlib.redirect_stdout(io.StringIO()):
        ans, H, _ = B.get_heisenberg_problem(N)
    q = Q(ans, H, shot_budget=1024)
    Hs = q.H_sense
    t_evo = q.tau0
    lays = layers_of(Hs)
    width = max(len(L) for L in lays)

    qa = build(Hs, t_evo, N, fanout=1)
    qb = build(Hs, t_evo, N, fanout=2)
    d1, x1 = stats(qa); d2, x2 = stats(qb)

    err = float('nan')
    if N <= 8:
        # Qiskit puts register 0 in the LEAST significant bit, so with
        # anc=q0, hlp=q1..q(F-1), sys=q(F).. the basis index is
        #     a + 2*h + 2^F * s      (fanout)        a + 2*s   (serial)
        # My first pass had the ancilla as the MOST significant bit, which
        # compared two different orderings and produced a spurious 1.35.
        Ua = Operator(qa).data
        Ub = Operator(qb).data
        dsys = 2 ** N
        ia, ib = [], []
        for a_ in range(2):
            for s_ in range(dsys):
                ia.append(a_ + 2 * s_)              # serial: no helpers
                ib.append(a_ + (1 << width) * s_)   # fanout: helpers all |0>
        ia, ib = np.array(ia), np.array(ib)
        err = float(np.linalg.norm(Ua[np.ix_(ia, ia)] - Ub[np.ix_(ib, ib)], 2))

    rows.append((N, d1, d2, x1, x2))
    print(f"  {N:>4}{len(lays):>8}{width:>4}{d1:>14}{d2:>14}{d1/max(d2,1):>9.1f}"
          f"{x1:>11}{x2:>11}{err:>13.2e}", flush=True)

ns = np.array([r[0] for r in rows], float)
a1 = np.polyfit(np.log(ns), np.log([r[1] for r in rows]), 1)[0]
a2 = np.polyfit(np.log(ns), np.log([r[2] for r in rows]), 1)[0]
g1 = np.polyfit(np.log(ns), np.log([r[3] for r in rows]), 1)[0]
g2 = np.polyfit(np.log(ns), np.log([r[4] for r in rows]), 1)[0]
print()
print(f"  depth  serial N^{a1:.2f}   fanout N^{a2:.2f}")
print(f"  cx     serial N^{g1:.2f}   fanout N^{g2:.2f}")
print()
print("  A fanout depth exponent near 0 means V3's duration can be held under")
print("  rep_delay at every N, and its 3 circuits/epoch then beat V4's 5 outright.")
print("  But read the cx columns first: if fan-out RAISES gate count, it makes")
print("  V3's fidelity problem worse while fixing a cost problem that sorting had")
print("  already mostly solved - a bad trade for the thing V3 actually needs.")
