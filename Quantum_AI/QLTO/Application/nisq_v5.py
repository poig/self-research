"""QLTO V5 - the one-circuit gradient, with everything that did not earn its place removed.

Standalone: numpy and qiskit only. It imports nothing from nisq_v2 or nisq_v3.

WHAT V5 IS. One measurement circuit per commuting group per block gives EVERY
parameter's gradient component in that block, from the same shot record. A
bounded classical step consumes it. That is the whole method.

    for each block:
        prepare  |+>^n on param, apply W: |x>|0> -> |x>|psi(theta_c + Rx)>
        rotate   sys into commuting group g's measurement basis
        measure  param AND sys together
        decode   g_i = [<E | x_i=1> - <E | x_i=0>] / 2R     (a MARGINAL, per T1)
        step     p_i -= alpha * R * g_i / max_j|g_j|

WHY THE MARGINAL IS THE POINT (T1/T2). Every shot carries a value for every bit,
so all M components come from the same shots - ONE circuit family, not 2M. And
the estimator is LINEAR in the shot record, so it is unbiased at any
shots-per-vertex, including fewer than one per vertex. That is what lets the
blocks be wide, which is where the cost advantage lives (T10: cost-optimal block
width n* ~ 0.65 M, circuits/gradient ~ 1.5, constant in M).

WHAT WAS REMOVED FROM V3, AND WHY. Each of these was measured, not assumed.

  the quantum walk        REMOVED. It costs 2 circuits per block-epoch against
                          this step's 1 and was never measured ahead of it: at
                          Heisenberg N=4 across four independent runs spanning
                          two implementations and both merged_walk settings, and
                          at wide R (R0=pi/2, where v9_globalgrid puts the box at
                          1.7 -> 3.3 minima - the walk's own claim), gradstep won
                          7 of 8. See supplement v20, v53, v53b, v53c.
                          The mechanism explains it: the drift phase accumulates
                          ~23.9*g and WRAPS, so the walk keeps only direction
                          (atan2, bounded) and discards magnitude (hypot,
                          unbounded). It therefore PLATEAUS - 4x the shots moved
                          it by -0.04 while gradstep gained +0.24.

  the QPE ancilla ladder  REMOVED, and with it the only estimator-level Trotter
                          error in the method. QPE bought G-INDEPENDENCE (one
                          circuit whatever H is) at the cost of the (2^k - 1)*tau0
                          evolution ladder, which is where V3's depth came from:
                          19-141x parameter-shift, growth exponent 1.26, and
                          survival 0.098 at Heisenberg N=6 even after kappa 4->3
                          halved the gates. Direct readout costs G circuits per
                          block instead of 1, and measures depth 1.5x with
                          exponent 1.06 and 2q gates 0.28-0.46x of parameter-shift
                          - BETTER than the baseline it is competing with
                          (v18, v19). v22 found it ties V3 on optimisation
                          (3 ties, 1 loss withdrawn once v27 showed the harness
                          returns 3.3 sigma on a null).
                          The trade is real and it is a hardware judgement: v21
                          measured that vendors bill circuits and shots with NO
                          depth term, which favours QPE, but a circuit whose
                          survival is 0.098 returns noise whatever it costs.

  the Boltzmann decoder   NOT CARRIED OVER. It ties the marginal path on small
                          blocks at half the circuits, but it is NONLINEAR - it
                          must resolve each vertex's energy before weighting it,
                          so it needs shots >~ 2^n and dies exactly at the wide
                          blocks where T10's advantage lives.

  merged_walk, fanout,    all walk-specific or measured negative.
  uncompute_w, boltzmann,
  moments, folded spectrum

WHAT IS KEPT AND WHY.

  R-smearing              NOT a defect. E_hat({i})/R is the exact gradient of the
                          SMOOTHED objective E_R = E_x[E(theta+Rx)], the
                          Nesterov-Spokoiny/ES smoothing. Signal is 2R*dE so
                          large R buys SNR; bias is O(R^2) so small R buys
                          accuracy; the decaying schedule anneals between them.
                          At Heisenberg N=6 this reaches cos 0.9678 on 16k shots
                          where parameter-shift needs 37k for a worse 0.9544
                          (v14) - and the advantage grows with N.

  max-normalised step     p_i -= alpha*R*g_i/max|g_j|. Scale-free WITHIN a block,
                          which makes it immune to the per-block scale bias
                          anomaly_c measured at up to 2.4x, 80 sigma from unity.
                          A step using raw magnitudes would inherit that error.

  free energy log         The degree-0 Walsh coefficient is the plain mean of the
                          per-vertex energies, so the epoch energy costs NOTHING.
                          It is biased: E_hat(empty) = E(theta_c) + (R^2/2) Tr H
                          + O(R^4), and that bias is NOT removable from these
                          shots - every vertex has x_i^2 = 1, so the diagonal
                          curvature is perfectly degenerate with the constant and
                          no linear functional separates them. Use it for
                          MONITORING and call energy_exact() once at the end; that
                          is 1 circuit instead of one per epoch.

COST, at Heisenberg N=4, 20 epochs, 4 blocks:
    V3 walk + per-epoch log   160 + 20 = 180 circuits
    V5 gradstep + free log     G*80 + 1
so V5 is cheaper whenever G < 2.2, and G=3 on Heisenberg - i.e. V5 trades circuit
count for depth, deliberately. Choose V3 when the hardware is good enough to run
its ladder and the bill is what hurts; choose V5 when depth is what hurts.
"""
import numpy as np
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import QFT, PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

# Controlled forms of the single-parameter rotations a hardware-efficient ansatz
# is built from. Anything outside this raises rather than being silently skipped:
# a parameter that fails to become controlled would sit at its centre value and
# report a zero gradient, which is indistinguishable from a flat direction.
_CTRL = {'rx': 'crx', 'ry': 'cry', 'rz': 'crz', 'p': 'cp', 'u1': 'cp'}


class QLTOv5:
    """One-circuit-per-group gradient + bounded classical step.

    gradient_mode='direct' keeps the current grouped readout. gradient_mode='qpe'
    switches to the V3-style QPE sensing path, which removes the G factor at the
    cost of the ancilla ladder and deeper coherent evolution.
    """

    def __init__(self, ansatz, hamiltonian, shot_budget=8192, alpha=0.9,
                 sim_seed=None, backend=None, gradient_mode='direct',
                 num_ancillas=3, qpe_margin=2.0):
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.shot_budget = int(shot_budget)
        self.alpha = float(alpha)
        self.gradient_mode = str(gradient_mode).lower()
        if self.gradient_mode not in ('direct', 'qpe'):
            raise ValueError("gradient_mode must be 'direct' or 'qpe'")
        self.N = ansatz.num_qubits
        self.M = ansatz.num_parameters
        self.sim_seed = sim_seed
        self._shot_index = 0
        self.nefv = 0
        self.max_circuit_depth = 0
        self.last_energy = None

        self.backend = backend or AerSimulator()
        self.groups = self._group(hamiltonian)
        # ParameterView has no .index(), so map parameter -> global index once.
        self._pidx = {p: i for i, p in enumerate(ansatz.parameters)}
        self.layers = self._layers()
        self.num_ancillas = max(1, int(num_ancillas))
        if self.gradient_mode == 'qpe' and self.num_ancillas < 2:
            raise ValueError('QPE mode requires num_ancillas >= 2')
        self.qpe_margin = float(qpe_margin)
        self.H_sense, self.h_offset, self.H_range = self._sensing_hamiltonian(
            hamiltonian)
        self.tau0 = np.pi / (self.qpe_margin * self.H_range + 1e-12)

    # ── setup ────────────────────────────────────────────────────────────────

    @staticmethod
    def _group(H):
        """Qubit-wise commuting groups: one measurement setting each."""
        return list(H.group_commuting(qubit_wise=True))

    def _layers(self):
        """Parameters that act on disjoint qubits and can share a W gate.

        One layer per parameterised instruction depth, matching how V3 blocks the
        ansatz. A parameter's block is determined by the qubits it touches.
        """
        blocks, seen, cur, used = [], set(), [], set()
        for inst in self.ansatz.data:
            prm = [p for p in inst.operation.params
                   if isinstance(p, ParameterExpression) and p.parameters]
            if not prm:
                continue
            idx = sorted(self._pidx[q] for p in prm for q in p.parameters)
            qs = {self.ansatz.find_bit(b).index for b in inst.qubits}
            if qs & used:
                blocks.append(cur)
                cur, used = [], set()
            cur.extend(i for i in idx if i not in seen)
            seen.update(idx)
            used |= qs
        if cur:
            blocks.append(cur)
        return [{'params': b} for b in blocks if b]

    # ── the W gate ───────────────────────────────────────────────────────────

    def _build_w(self, qc, param, sysr, centre, R, active):
        """|x>|0> -> |x>|psi(theta_c + Rx)>, x in {-1,+1}^n.

        Each active parameter is emitted at its LOW corner theta_c - R, then a
        controlled rotation of 2R on the corresponding param qubit lifts it to the
        HIGH corner when that bit is 1. Two corners per axis, one control each.
        """
        pos = {p: i for i, p in enumerate(active)}
        for inst in self.ansatz.data:
            op = inst.operation
            qs = [sysr[self.ansatz.find_bit(b).index] for b in inst.qubits]
            prm = [p for p in op.params
                   if isinstance(p, ParameterExpression) and p.parameters]
            if not prm:
                qc.append(op, qs)
                continue
            gi = self._pidx[next(iter(prm[0].parameters))]
            if gi not in pos:
                qc.append(op.__class__(float(centre[gi])), qs)
                continue
            if op.name not in _CTRL:
                raise ValueError(
                    f"V5 cannot build a controlled form of '{op.name}'. Add it to "
                    f"_CTRL, or the parameter would silently stay at its centre "
                    f"value and report a zero gradient.")
            qc.append(op.__class__(float(centre[gi]) - R), qs)
            getattr(qc, _CTRL[op.name])(2.0 * R, param[pos[gi]], qs[0])

    @staticmethod
    def _basis(qc, sysr, group):
        """Rotate each qubit into the group's shared measurement basis."""
        axis = {}
        for lbl in (p[0] if isinstance(p, tuple) else str(p)
                    for p in group.paulis.to_labels()):
            for q, ch in enumerate(reversed(lbl)):
                if ch != 'I':
                    axis[q] = ch
        for q, ch in axis.items():
            if ch == 'X':
                qc.h(sysr[q])
            elif ch == 'Y':
                qc.sdg(sysr[q])
                qc.h(sysr[q])
        return axis

    @staticmethod
    def _sensing_hamiltonian(H):
        """Strip the identity term for coherent sensing and estimate the range."""
        ident = 0.0
        keep_p, keep_c = [], []
        for pauli, coeff in zip(H.paulis, H.coeffs):
            if set(pauli.to_label()) == {'I'}:
                ident += complex(coeff).real
            else:
                keep_p.append(pauli.to_label())
                keep_c.append(coeff)

        H0 = (SparsePauliOp(keep_p, keep_c).simplify() if keep_p
              else SparsePauliOp('I' * H.num_qubits, [0.0]))

        if H0.num_qubits <= 14:
            ev = np.linalg.eigvalsh(H0.to_matrix())
            rng = float(ev[-1] - ev[0])
        else:
            rng = 2.0 * float(np.sum(np.abs(H0.coeffs)))
        return H0, ident, max(rng, 1e-12)

    def _build_qpe_sensing_circuit(self, centre, R, active):
        """V3-style QPE sensing circuit shared by all QPE decoders."""
        n_active = len(active)
        k = self.num_ancillas

        anc = AncillaRegister(k, 'anc')
        param = QuantumRegister(n_active, 'param')
        sysr = QuantumRegister(self.N, 'sys')
        c_param = ClassicalRegister(n_active, 'c_param')
        c_anc = ClassicalRegister(k, 'c_anc')

        qc = QuantumCircuit(anc, param, sysr, c_param, c_anc)
        qc.h(anc)
        qc.h(param)
        self._build_w(qc, param, sysr, centre, R, active)

        for a in range(k):
            t = (2 ** a) * self.tau0
            reps = int(max(1, (2 ** a) // 2, np.ceil(t / 2.0)))
            qc.append(PauliEvolutionGate(
                self.H_sense, time=t,
                synthesis=SuzukiTrotter(order=2, reps=reps)).control(1),
                [anc[a]] + list(sysr))

        qc.append(QFT(num_qubits=k, inverse=True, do_swaps=True), anc)
        qc.measure(param, c_param)
        qc.measure(anc, c_anc)
        return qc

    def _decode_gradient_qpe(self, counts, centre, active, R):
        """Decode the QPE energy samples into the same marginal gradient form."""
        n_active = len(active)
        k = self.num_ancillas
        num = np.zeros((2, n_active))
        den = np.zeros((2, n_active))
        e_tot = e_cnt = 0.0

        for bitstr, cnt in counts.items():
            parts = bitstr.split()
            if len(parts) != 2:
                continue
            m = int(parts[0], 2)
            phi = m / (2 ** k)
            if phi >= 0.5:
                phi -= 1.0
            energy = -2.0 * np.pi * phi / (self.tau0 + 1e-12)
            energy += self.h_offset

            xbits = parts[1][::-1]
            e_tot += energy * cnt
            e_cnt += cnt
            for i in range(n_active):
                b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                num[b, i] += energy * cnt
                den[b, i] += cnt

        m1 = np.divide(num[1], den[1], out=np.zeros(n_active), where=den[1] > 0)
        m0 = np.divide(num[0], den[0], out=np.zeros(n_active), where=den[0] > 0)
        grad = np.zeros(len(centre))
        grad[active] = (m1 - m0) / (2.0 * R + 1e-12)
        return grad, (e_tot / e_cnt if e_cnt else float('nan'))

    # ── sensing ──────────────────────────────────────────────────────────────

    def sense(self, centre, R, active):
        """Marginal gradient over the block, plus the free degree-0 energy.

        Returns (grad, e_mean). One circuit per commuting group; every shot
        contributes to every parameter's marginal, which is T1/T2.
        """
        if self.gradient_mode == 'qpe':
            qc = self._build_qpe_sensing_circuit(centre, R, active)
            counts = self._run(qc)
            return self._decode_gradient_qpe(counts, centre, active, R)

        n = len(active)
        num = np.zeros((2, n))
        den = np.zeros((2, n))
        e_tot = e_cnt = 0.0

        for group in self.groups:
            qc = QuantumCircuit(QuantumRegister(n, 'param'),
                                QuantumRegister(self.N, 'sys'),
                                ClassicalRegister(n, 'cp'),
                                ClassicalRegister(self.N, 'cs'))
            param, sysr = qc.qregs[0], qc.qregs[1]
            qc.h(param)
            self._build_w(qc, param, sysr, centre, R, active)
            self._basis(qc, sysr, group)
            qc.measure(param, qc.cregs[0])
            qc.measure(sysr, qc.cregs[1])
            counts = self._run(qc)

            labels = group.paulis.to_labels()
            coeffs = np.real(group.coeffs)
            for bitstr, cnt in counts.items():
                parts = bitstr.split()
                if len(parts) != 2:
                    continue
                sbits, xbits = parts[0][::-1], parts[1][::-1]
                e = 0.0
                for lbl, c in zip(labels, coeffs):
                    s = 1
                    for q, ch in enumerate(reversed(lbl)):
                        if ch != 'I' and q < len(sbits) and sbits[q] == '1':
                            s = -s
                    e += c * s
                e_tot += e * cnt
                e_cnt += cnt
                for i in range(n):
                    b = 1 if (i < len(xbits) and xbits[i] == '1') else 0
                    num[b, i] += e * cnt
                    den[b, i] += cnt

        m1 = np.divide(num[1], den[1], out=np.zeros(n), where=den[1] > 0)
        m0 = np.divide(num[0], den[0], out=np.zeros(n), where=den[0] > 0)
        grad = np.zeros(len(centre))
        grad[active] = (m1 - m0) / (2.0 * R + 1e-12)
        # degree-0 Walsh coefficient: E(theta_c) + (R^2/2) Tr H + O(R^4). Biased,
        # and provably not correctable from these shots - see the module docstring.
        return grad, (e_tot / e_cnt if e_cnt else float('nan')) / len(self.groups)

    # ── step ─────────────────────────────────────────────────────────────────

    def grad_step(self, centre, R, active, grad):
        """Bounded, max-normalised, scale-free within the block."""
        p = np.asarray(centre, dtype=float).copy()
        g = np.asarray(grad)[active]
        mx = float(np.max(np.abs(g)))
        if mx < 1e-12:
            return p
        p[active] = p[active] - self.alpha * R * g / mx
        return p

    # ── driver ───────────────────────────────────────────────────────────────

    def _run(self, qc):
        t_qc = transpile(qc, self.backend, optimization_level=1)
        self.max_circuit_depth = max(self.max_circuit_depth, t_qc.depth())
        self.nefv += 1
        kw = {}
        if self.sim_seed is not None:
            kw['seed_simulator'] = int(self.sim_seed) + self._shot_index
            self._shot_index += 1
        return self.backend.run(t_qc, shots=self.shot_budget,
                                **kw).result().get_counts()

    def run_epoch(self, params, search_radius):
        """One sweep over all blocks. Energy comes free from the same shots."""
        p = np.asarray(params, dtype=float).copy()
        es = []
        for blk in self.layers:
            act = blk['params']
            if not act:
                continue
            grad, e = self.sense(p, search_radius, act)
            es.append(e)
            p = self.grad_step(p, search_radius, act, grad)
        self.last_energy = float(np.mean(es)) if es else None
        return p, self.last_energy

    def energy_exact(self, params):
        """Exact energy, 0 circuits on a simulator. Call ONCE at the end.

        The per-epoch log is free but biased by the R-smearing; this is the number
        to report. On hardware, replace with one measurement per commuting group.
        """
        sv = Statevector(self.ansatz.assign_parameters(np.asarray(params)))
        return float(np.real(sv.expectation_value(self.hamiltonian)))

    def minimize(self, params, epochs=20, r0=0.6, r_decay=0.9):
        p = np.asarray(params, dtype=float).copy()
        trace = []
        for ep in range(epochs):
            R = max(r0 * (r_decay ** ep), 1e-4)
            p, e = self.run_epoch(p, R)
            trace.append(e)
        return p, trace
