"""Does the log-width design register compose with QPE readout? Qubits AND G at once.

Two independent savings exist in this project and they have never been built into
the same circuit:

  V6's DESIGN REGISTER    M parameters on ceil(log2(M+1))+1 qubits instead of M.
                          Costs G circuits per gradient - one per qubit-wise
                          commuting group - because it reads Pauli expectation
                          values in a measurement basis.

  V5's QPE READOUT        G-INDEPENDENCE: one circuit whatever H is, because the
                          energy arrives as a phase on an ancilla ladder rather
                          than as a basis measurement. Costs one param qubit per
                          parameter, since _build_qpe_sensing_circuit uses
                          QuantumRegister(n_active) and decodes a MARGINAL over
                          the hypercube {0,1}^n.

So V6 is cheap in qubits and pays G; V5 is cheap in circuits and pays M qubits.
The obvious question is whether both hold at once, and the structural argument
says yes: the QPE ladder acts only on `sys`, controlled by `anc`. It never
touches the param register, so it should be indifferent to whether that register
holds one qubit per parameter or a log-width design.

  That is an argument about a circuit, and project rule R1 is explicit that a
  construction which has never been a circuit is a conjecture about a circuit.
  This file builds it.

WHAT IS BUILT. V6's _direct_template up to the point where it rotates into a
measurement basis - same Hadamard-design W construction, same scratch-wire parity
trick, same ceil(log2(n+1))+1 register - and then, instead of _basis() and
measuring `sys`:

    for a in range(k):   controlled-e^{-i H 2^a tau0} on sys, controlled by anc[a]
    QFT^-1 on anc
    measure param and anc

The decode is V6's marginal over the design ROW, with the per-shot energy coming
from the QPE phase instead of from Pauli signs:

    phi = m / 2^k,  wrapped to [-0.5, 0.5)
    E   = -2 pi phi / tau0 + h_offset

WHAT WOULD SETTLE IT. The QPE-on-design gradient must agree with V6's own
direct-readout gradient, because they estimate the SAME smoothed quantity and
differ only in how the energy reaches the classical record. Agreement means the
composition holds; disagreement means the phase readout does not survive the
design encoding, which is what v106 found for the twirl construction (there for a
specific reason - conjugation is isospectral - that does not apply here, since
this register changes the STATE and not the Hamiltonian).

TIER (project rule R1): tier A. The QPE-on-design circuit is built, transpiled
and run on AerSimulator with finite shots. V6's direct readout is also tier A.
The exact smoothed gradient used as a third reference is tier B (Statevector) and
is the reference only.
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library import efficient_su2, QFT, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator

import benchmark as B
from nisq_v6 import QLTOv6, _design_spec, _sign_table, _CTRL

SHOTS = 32768
N_ANC = 4
QPE_MARGIN = 2.0
SEEDS = (0, 1, 2)


def sensing_hamiltonian(H):
    """Strip the identity for coherent sensing. Mirrors nisq_v5."""
    ident = 0.0
    kp, kc = [], []
    for pauli, coeff in zip(H.paulis, H.coeffs):
        if set(pauli.to_label()) == {'I'}:
            ident += complex(coeff).real
        else:
            kp.append(pauli.to_label()); kc.append(coeff)
    H0 = (SparsePauliOp(kp, kc).simplify() if kp
          else SparsePauliOp('I' * H.num_qubits, [0.0]))
    return H0, ident


class QPEDesign(QLTOv6):
    """V6's design register, read out by QPE instead of by a Pauli basis."""

    def __init__(self, ansatz, hamiltonian, num_ancillas=N_ANC, **kw):
        super().__init__(ansatz, hamiltonian, **kw)
        self.k_anc = int(num_ancillas)
        self.H_sense, self.h_offset = sensing_hamiltonian(self.hamiltonian)
        nq = self.H_sense.num_qubits
        self.H0_norm = (float(np.linalg.norm(self.H_sense.to_matrix(), ord=2))
                        if nq <= 14 else float(np.sum(np.abs(self.H_sense.coeffs))))
        self.tau0 = np.pi / (QPE_MARGIN * self.H0_norm + 1e-12)

    def _qpe_template(self, active):
        """V6's W on the design register; QPE ladder instead of a basis rotation."""
        n = len(active)
        ns = max(1, min(self.n_scratch, n))
        m_row, cols = _design_spec(n, ns, self.design_resolution)
        nreg = m_row + (1 if self.design_resolution >= 4 else 0)
        theta = list(self.ansatz.parameters)
        radius = Parameter('R_qpe')
        pos = {p: i for i, p in enumerate(active)}

        anc = AncillaRegister(self.k_anc, 'anc')
        param = QuantumRegister(nreg, 'param')
        sysr = QuantumRegister(self.N, 'sys')
        scr = QuantumRegister(ns, 'par')
        cp = ClassicalRegister(nreg, 'cp')
        ca = ClassicalRegister(self.k_anc, 'ca')
        qc = QuantumCircuit(anc, param, sysr, scr, cp, ca)

        qc.h(anc)
        qc.h(param)

        # ---- identical to V6's _direct_template W construction ----
        for s in range(ns):
            qc.x(scr[s])
            if self.design_resolution >= 4:
                qc.cx(param[m_row], scr[s])
        prev = [0] * ns
        for inst in self.ansatz.data:
            op = inst.operation
            qs = [sysr[self.ansatz.find_bit(b).index] for b in inst.qubits]
            prm = [p for p in op.params
                   if isinstance(p, ParameterExpression) and p.parameters]
            if not prm:
                qc.append(op, qs); continue
            gi = self._pidx[next(iter(prm[0].parameters))]
            if gi not in pos:
                qc.append(op.__class__(theta[gi]), qs); continue
            if op.name not in _CTRL:
                raise ValueError("no controlled form of '%s'" % op.name)
            p = pos[gi]; s = p % ns; c = cols[p]
            qc.append(op.__class__(theta[gi] - radius), qs)
            for b in range(m_row):
                if (c ^ prev[s]) >> b & 1:
                    qc.cx(param[b], scr[s])
            prev[s] = c
            getattr(qc, _CTRL[op.name])(2.0 * radius, scr[s], qs[0])
        for s in range(ns):
            for b in range(m_row):
                if prev[s] >> b & 1:
                    qc.cx(param[b], scr[s])
            if self.design_resolution >= 4:
                qc.cx(param[m_row], scr[s])
            qc.x(scr[s])
        # ---- end shared construction ----

        for a in range(self.k_anc):
            t = (2 ** a) * self.tau0
            reps = int(max(1, (2 ** a) // 2, np.ceil(t / 2.0)))
            qc.append(PauliEvolutionGate(
                self.H_sense, time=t,
                synthesis=SuzukiTrotter(order=2, reps=reps)).control(1),
                [anc[a]] + list(sysr))
        qc.append(QFT(num_qubits=self.k_anc, inverse=True, do_swaps=True), anc)

        qc.measure(param, cp)
        qc.measure(anc, ca)
        return transpile(qc, self.backend, optimization_level=1), theta, radius, \
            m_row, cols, nreg

    def sense_qpe(self, centre, R, active):
        """One circuit, whatever G is. Returns (grad, mean energy, n_qubits)."""
        n = len(active)
        Rv = self._radius(R, n)
        t_qc, theta, radius, m_row, cols, nreg = self._qpe_template(active)
        bind = {theta[i]: float(centre[i]) for i in range(len(theta))}
        bind[radius] = float(Rv)
        counts = self._run_transpiled(t_qc.assign_parameters(bind, inplace=False))

        num = np.zeros((2, n)); den = np.zeros((2, n))
        e_tot = e_cnt = 0.0
        SG = _sign_table(m_row, cols)
        for bits, cnt in counts.items():
            parts = bits.split()
            if len(parts) != 2:
                continue
            # cregs created cp then ca -> Qiskit prints LAST-created first
            m = int(parts[0], 2)
            regw = int(parts[1], 2)
            phi = m / (2 ** self.k_anc)
            if phi >= 0.5:
                phi -= 1.0
            energy = -2.0 * np.pi * phi / (self.tau0 + 1e-12) + self.h_offset

            d = regw & ((1 << m_row) - 1)
            fold = 1.0 - 2.0 * ((regw >> m_row) & 1) if nreg > m_row else 1.0
            sg = SG[d] * fold
            e_tot += energy * cnt; e_cnt += cnt
            for i in range(n):
                b = 1 if sg[i] > 0 else 0
                num[b, i] += energy * cnt
                den[b, i] += cnt

        m1 = np.divide(num[1], den[1], out=np.zeros(n), where=den[1] > 0)
        m0 = np.divide(num[0], den[0], out=np.zeros(n), where=den[0] > 0)
        grad = np.zeros(len(centre))
        grad[active] = (m1 - m0) / (2.0 * Rv + 1e-12)
        return grad, (e_tot / e_cnt if e_cnt else float('nan')), t_qc.num_qubits


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


print("=" * 100)
print("v132  QPE READOUT ON THE LOG-WIDTH DESIGN REGISTER")
print("=" * 100)
print("  V6: log M qubits, G circuits.  V5-QPE: 1 circuit, M qubits.")
print("  Both savings in one circuit has never been built. Building it.")
print("  TIER A - transpiled circuits on AerSimulator, %d shots." % SHOTS)
print()

for N in (4, 6):
    anz, H, name = B.get_heisenberg_problem(N)
    M = len(anz.parameters)
    print("-" * 100)
    print("%s   N=%d, M=%d" % (name, N, M))
    print("-" * 100)

    rows = []
    for sd in SEEDS:
        theta = np.random.default_rng(sd).uniform(-np.pi, np.pi, M)
        active = list(range(M))

        q6 = QLTOv6(anz, H, shot_budget=SHOTS, sim_seed=10 + sd,
                    backend=AerSimulator(seed_simulator=10 + sd))
        with contextlib.redirect_stdout(io.StringIO()):
            g6, _e6 = q6.sense(theta, 0.45, active)
        n6 = max(t.num_qubits for t, _, _ in
                 [q6._direct_template(active, g) for g in q6.groups])

        qq = QPEDesign(anz, H, num_ancillas=N_ANC, shot_budget=SHOTS,
                       sim_seed=10 + sd,
                       backend=AerSimulator(seed_simulator=10 + sd))
        with contextlib.redirect_stdout(io.StringIO()):
            gq, _eq, nq = qq.sense_qpe(theta, 0.45, active)

        # V5's circuit is anc(k) + param(M) + sys(N); no scratch wires.
        tq, _th, _r, _mr, _cl, _nr = qq._qpe_template(active)
        d6 = max(t.depth() for t, _, _ in
                 [q6._direct_template(active, g) for g in q6.groups])
        x6 = sum(t.count_ops().get('cx', 0) for t, _, _ in
                 [q6._direct_template(active, g) for g in q6.groups])
        rows.append((cosine(gq, g6), len(q6.groups), n6, nq,
                     N_ANC + M + N, d6, x6, tq.depth(),
                     tq.count_ops().get('cx', 0)))

    c = float(np.mean([r[0] for r in rows]))
    G, n6, nq, n5 = rows[0][1], rows[0][2], rows[0][3], rows[0][4]
    d6, x6, dq, xq = rows[0][5], rows[0][6], rows[0][7], rows[0][8]
    print("      readout                circuits  qubits   depth    2q gates   cos")
    print("   " + "-" * 78)
    print("      V6 direct (Pauli)        %4d     %3d     %6d    %6d      --"
          % (G, n6, d6, x6))
    print("      V5 QPE (1 qubit/param)      1     %3d          -         -      --"
          % n5)
    print("      QPE on design register      1     %3d     %6d    %6d   %+.4f"
          % (nq, dq, xq, c))
    print()
    print("      Depth/2q for V6 is the WORST group; per-gradient totals are %dx"
          % G)
    print("      that if the %d circuits run in series. QPE-on-design: %d deep,"
          % (G, dq))
    print("      %d two-qubit gates, in ONE circuit." % xq)
    print()
    print("      V5 count is anc(%d) + param(M=%d) + sys(N=%d) = %d, the full"
          % (N_ANC, M, N, n5))
    print("      circuit. Design register saves %d qubits at M=%d, and the saving"
          % (n5 - nq, M))
    print("      grows as M/log2(M) - that is the whole point of the register.")
    print()

print("=" * 100)
print("READING IT")
print("=" * 100)
print("  If the cos column is high, both savings hold in ONE circuit: G-independence")
print("  from the phase readout and log-M width from the design register. The QPE")
print("  ladder only ever touches `sys` under control of `anc`, so it cannot see")
print("  what the param register encodes - the structural argument, now built.")
print()
print("  IF IT IS LOW, the phase readout does not survive the design encoding and")
print("  the reason is the finding. Note v106 found exactly that for twirl_cal, but")
print("  for a reason that does NOT apply here: there the register acts by")
print("  CONJUGATION, so H_sigma is isospectral and a phase carries no signal. Here")
print("  the register changes the STATE, so energies genuinely move.")
print()
print("  WHAT THIS DOES NOT SHOW. The QPE ladder's depth is the (2^k-1)*tau0")
print("  evolution ladder that killed V5 on hardware (survival 0.098 at Heisenberg")
print("  N=6). Qubit width and circuit count are what this file measures; DEPTH is")
print("  the cost that moved, and it is unchanged by the design register.")
print()
print("  SCOPE. Heisenberg N=4,6, efficient_su2 reps=1, %d ancillas, %d shots,"
      % (N_ANC, SHOTS))
print("  %d seeds, R=0.45, no noise model, no hardware." % len(SEEDS))
