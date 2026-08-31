"""QLTO V6 - the log-register gradient: one block, one circuit per commuting group.

Standalone: numpy and qiskit only. It imports nothing from nisq_v5. V5 is a frozen
published artifact and must not move; V6 is the line that develops.

WHAT CHANGED FROM V5, and it is one thing with three consequences.

V5 spends ONE REGISTER QUBIT PER PARAMETER. That caps block width n, and since the
advantage over parameter-shift is exactly 2n while circuits per gradient are
G*M/n, the register is what held the method at 2N. V6 indexes rows of a
resolution-IV Hadamard design instead, on ceil(log2(n+1)) + 1 qubits. The sign for
parameter j at register value (d, f) is

    sigma_j = (-1)^(popcount(d AND c_j) + f)

Because the register no longer scales with M, ONE GLOBAL BLOCK becomes reachable:
L = 1, and circuits per gradient fall from G*L to G. At N=6, M=36 that is 3
circuits rather than 18, on 10 register qubits rather than 36.

MEASURED, Heisenberg, matched total shots T = 294912, basis rz/sx/x/cx:

    N=6, M=36          circuits  shots/circ  reg q  depth  2q/grad     cos
    V5                       18       16384      6     34      396  0.9796
    V6                        3       98304     10    102      402  0.9751
    parameter-shift         216        1365      0     22     2160  0.9877

Equal accuracy to V5 on a sixth of the circuits; 72x fewer circuits and 5.4x fewer
two-qubit gates than parameter-shift. Parameter-shift wins the accuracy column at
THIS budget because 294912 is above the N=6 crossover of about 131k measured in
v69; below it the column flips. Depth is 3x V5's and is the axis V6 loses.

THREE THINGS THAT HAD TO BE RIGHT, each of which was wrong first.

  GRAY-ORDERED COLUMNS. With the obvious c_j = j+1 every parameter rebuilds its
  parity from scratch, costing O(M log M) two-qubit gates, and the encoding
  measured 2.5x WORSE than V5 despite using 6x fewer circuits. Gray ordering makes
  consecutive columns differ in one bit so a running parity advances by a single
  CNOT: O(M), 334 -> 124 gates per circuit at M=36.

  PER-WIRE GRAY SLICES. Dealing parities round-robin over several scratch wires
  breaks the serialisation that made depth 229 against V5's 34. But parameters
  sharing a wire are k apart in the Gray sequence, so plain ordering put the gate
  count back up (372 -> 522). Each wire therefore gets its own slice of the column
  space: high bits name the wire, low bits run a private Gray sequence.

  RADIUS RESCALED BY BLOCK WIDTH. A block of n parameters displaces the state by
  about sqrt(n)*R, so a radius chosen for an N-wide block over-displaces an M-wide
  one and the linearisation the estimator rests on degrades. Handing V5's R = 0.45
  straight to a 36-parameter block gives cos 0.886 instead of 0.975 - a silent
  twenty percent regression that reads as a bug elsewhere. V6 divides by
  sqrt(n/N) internally so the SAME R a caller would give V5 is correct here.

WHAT V6 DOES NOT FIX. The error still falls as T^(-1/3) against parameter-shift's
T^(-1/2), because the finite-radius bias cR^2 is untouched; parameter-shift still
wins at large budgets and the crossover still recedes as M^3. Richardson
extrapolation on R would change that exponent and is not implemented. And G is
untouched: at global block width G IS the circuit count, so it is now the entire
quantum cost, and v30 measured G ~ N^4.24 for molecular Hamiltonians.

WHAT WAS DROPPED FROM V5. The QPE sensing path, and with it num_ancillas,
qpe_margin, the sensing-Hamiltonian rescaling and the tau0 calibration. V5 keeps
them; V6 is direct-only. QPE bought G-independence at the cost of a
(2^k - 1)*tau0 evolution ladder whose measured survival was 0.098 at Heisenberg
N=6. V6 reinstates it as readout='qpe' (v132); tau0 is calibrated from the exact
spectral norm only up to 14 qubits and from the free coefficient 1-norm above
that, so the 2^N matrix is NOT a dependency of the shipped path. Depth is.

USAGE:

    q = QLTOv6(ansatz, hamiltonian, shot_budget=8192)
    theta, trace = q.minimize(theta0, epochs=20)

Cross-checked against V5 on the same landscape in supplement/v81_v5_vs_v6.py.
"""
import numpy as np
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    AncillaRegister, transpile)
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library import QFTGate, PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator

# Controlled forms of the single-parameter rotations a hardware-efficient ansatz
# is built from. Anything outside this raises rather than being silently skipped:
# a parameter that fails to become controlled would sit at its centre value and
# report a zero gradient, which is indistinguishable from a flat direction.
_CTRL = {'rx': 'crx', 'ry': 'cry', 'rz': 'crz', 'p': 'cp', 'u1': 'cp'}


def _resv_cols(n, m):
    """Greedy resolution-V columns in GF(2)^m, or None if m is too small.

    Resolution IV (what the foldover alone buys) clears main effects of
    TWO-factor interactions but leaves them confounded with THREE-factor ones.
    v90 measured that directly: the shipped Gray columns have min|S| = 3 at every
    size tested - three parameters whose columns XOR to zero - and on a loss with
    real degree-3 content the gradient cosine falls to 0.714 at M=16.

    Two conditions lift it to resolution V:
      (a) no column equals the XOR of two others   -> no 3-term relation
      (b) all pairwise XORs are distinct           -> no 4-term relation
    """
    cols, pair = [], set()
    for c in range(1, 1 << m):
        if c in pair or c in cols:
            continue
        new, ok = set(), True
        for d in cols:
            x = c ^ d
            if x == 0 or x in pair or x in new or x in cols:
                ok = False
                break
            new.add(x)
        if ok:
            cols.append(c)
            pair |= new
            if len(cols) == n:
                return cols
    return None


def _design_spec(n, k=1, resolution=4):
    """Column assignment for a block of n parameters over k scratch wires.

    RESOLUTION selects the width/fidelity trade, and neither setting is
    universally right - which is why this is a knob and not a new default.

      4 (default)  minimum width, m_row = ceil(log2(n+1)), Gray-ordered. What
                   every result up to v89 was measured on. Correct whenever the
                   caller is WIDTH-bound.
      5            no 3-term and no 4-term confounding. Costs m ~ 2 log2(n)
                   against the minimum log2(n) - measured m_row 6, 8, 8 at
                   n = 8, 12, 16 - and recovers the cosine to 0.979 / 0.991 /
                   0.927 against 0.902 / 0.940 / 0.714 (v90). Correct whenever
                   the caller is FIDELITY-bound and the loss is non-linear over
                   the sensing radius.

    The crossover against a LINEAR register moves with this choice: minimum
    width pays from about n=8, resolution V only from about n=12.

    Parameter j takes column c_j; column 0 is all-ones and carries the intercept,
    so it is skipped. Row indices need m_row = ceil(log2(n+1)) bits and one further
    qubit carries the FOLDOVER, whose presence in every parity flips all signs
    together and makes the design resolution IV, clearing main effects of
    two-factor interactions.

    Columns are GRAY-ORDERED so a running parity advances by one CNOT. With k > 1
    the parameters are dealt round-robin, so wire s carries j = s, s+k, s+2k, ...
    which are k apart in the Gray sequence and would differ in several bits. Each
    wire therefore gets its own slice: high bits name the wire, low bits run a
    private Gray sequence, so every wire walks one-bit transitions. This costs
    nothing in register width, since ceil(log2(k)) + ceil(log2(n/k + 1)) is about
    log2(n) either way. When the slices do not fit, fall back to the plain
    sequence, which is correct and merely more gates.

    Correctness never depends on the ordering: the circuit XORs whatever bits
    differ from the previous column, so any assignment gives the right signs.
    """
    m_row = max(1, int(np.ceil(np.log2(n + 1))))
    gray = lambda t: t ^ (t >> 1)
    if resolution >= 5:
        # Grow the register until a resolution-V set exists. Bounded: the
        # pairwise-XOR condition needs C(n,2) <~ 2^m, so m ~ 2 log2(n) suffices
        # and the loop terminates well before m_row + 8.
        for m in range(m_row, m_row + 9):
            cols = _resv_cols(n, m)
            if cols is not None:
                return m, cols
        # No set found in range: fall through to the resolution-IV design rather
        # than fail, since a correct-but-aliased gradient beats no gradient.
    if k > 1:
        per = -(-n // k)
        m_lo = max(1, int(np.ceil(np.log2(per + 1))))
        m_hi = int(np.ceil(np.log2(k)))
        if m_lo + m_hi <= m_row:
            cols = [((p % k) << m_lo) ^ gray(p // k + 1) for p in range(n)]
            if len(set(cols)) == n and 0 not in cols:
                return m_row, cols
    return m_row, [gray(j + 1) for j in range(n)]


def _design_sign(d, f, c):
    """sigma for one parameter at one measured register value.

    Scalar reference. The decode uses _sign_table, which is this function
    tabulated; kept because it is the readable statement of the rule.
    """
    return -1.0 if ((bin(d & c).count('1') + f) & 1) else 1.0


def _parity(v):
    """Bitwise parity of an int64 array, XOR-folded.

    Not np.bitwise_count: that needs numpy >= 2.0 and this module claims only
    numpy. Folding is branch-free and vectorises.
    """
    v = v ^ (v >> 32)
    v = v ^ (v >> 16)
    v = v ^ (v >> 8)
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return v & 1


def _sign_table(m_row, cols):
    """All sigma at once: SIGN[d, i] = _design_sign(d, 0, cols[i]).

    _design_sign depends only on (d, f, c), and d takes 2^m_row values with
    m_row = ceil(log2(n+1)), so the entire decode is a lookup of at most
    2^m_row * n entries built once per block. Evaluating it per (shot, parameter)
    instead cost 43% of an epoch - 167,496 calls at N=6 M=24, against 19% for the
    simulation itself. The foldover bit f flips every sign together, so it is
    applied as a scalar rather than tabulated.
    """
    d = np.arange(1 << m_row, dtype=np.int64)[:, None]
    c = np.asarray(cols, dtype=np.int64)[None, :]
    return 1.0 - 2.0 * _parity(d & c).astype(float)


def _support_masks(labels, nbits):
    """Bit mask of the non-identity support of each Pauli label.

    The energy of a shot is c . (-1)^popcount(sysbits AND supp), so the label
    string is only ever needed as this integer.
    """
    return np.array([sum(1 << q for q, ch in enumerate(reversed(lbl))
                         if ch != 'I' and q < nbits)
                     for lbl in labels], dtype=np.int64)


class QLTOv6:
    """Log-width design register, one global block, bounded classical step."""

    def __init__(self, ansatz, hamiltonian, shot_budget=8192, alpha=0.9,
                 sim_seed=None, backend=None, block_mode='global',
                 n_scratch=3, scale_radius=True, design_resolution=4,
                 radius_exponent=0.5, readout='direct', num_ancillas=4,
                 qpe_margin=2.0):
        # Decompose until every parameter-bearing gate is one _CTRL knows how to
        # control. This MUST check for progress: decompose() reaches a fixed point
        # on gates it cannot reduce further, and an unbounded `while` here spins
        # forever rather than falling through, hanging the constructor silently on
        # any circuit carrying a parameterised gate outside _CTRL - including
        # anything with a data-encoding prefix such as `cry`.
        try:
            _prev_ops = None
            for _ in range(16):
                if not any(inst.operation.name not in _CTRL
                           and inst.operation.params for inst in ansatz.data):
                    break
                ansatz = ansatz.decompose()
                _ops = ansatz.count_ops()
                if _ops == _prev_ops:
                    break
                _prev_ops = _ops
        except Exception:
            pass
        self.ansatz = ansatz
        self.hamiltonian = hamiltonian
        self.shot_budget = int(shot_budget)
        self.alpha = float(alpha)
        self.block_mode = str(block_mode).lower()
        if self.block_mode not in ('layered', 'global'):
            raise ValueError("block_mode must be 'layered' or 'global'")
        self.N = ansatz.num_qubits
        self.M = ansatz.num_parameters
        self.sim_seed = sim_seed
        self._shot_index = 0
        self.nefv = 0
        self.max_circuit_depth = 0
        self.last_energy = None
        self.n_scratch = max(1, int(n_scratch))
        # 4 = minimum-width Gray columns, every result up to v89. 5 = no 3-term
        # or 4-term confounding, costing m ~ 2 log2(n) and recovering the cosine
        # v90 measured collapsing to 0.714 at n=16. See _design_spec.
        self.design_resolution = int(design_resolution)
        self.scale_radius = bool(scale_radius)
        # Exponent p in R_eff = R (N/n)^p. 0.5 is shipped and unswept; see
        # _radius for why it may be too aggressive at large M.
        self.radius_exponent = float(radius_exponent)

        # readout='direct' reads Pauli expectation values in a measurement basis
        # and costs G circuits per gradient. readout='qpe' reads the energy as a
        # PHASE on an ancilla ladder, which is G-INDEPENDENT - one circuit
        # whatever H is - at the cost of the (2^k-1)*tau0 evolution depth.
        #
        # Both use the SAME log-width design register, which is the point: V5's
        # QPE path spent one param qubit per parameter, so it bought
        # G-independence and paid M qubits for it. supplement/v132 built the
        # composition and measured it at Heisenberg N=4 and N=6:
        #
        #   readout                circuits  qubits  depth  2q   cos vs direct
        #   V6 direct (worst grp)     3        15      45   201      --
        #   V5 QPE (1 qubit/param)    1        34       -     -      --
        #   QPE on design register    1        19     955   517   +0.9658
        #
        # So both savings hold at once - 1 circuit AND ceil(log2(M+1))+1 register
        # qubits - and DEPTH is what pays for it: 21x one direct circuit at N=6.
        # That is why 'direct' is the default: the ladder is what killed V5 on
        # hardware (survival 0.098 at Heisenberg N=6). Use 'qpe' where depth is
        # cheap and G is large.
        self.readout = str(readout).lower()
        if self.readout not in ('direct', 'qpe'):
            raise ValueError("readout must be 'direct' or 'qpe'")
        self.k_anc = max(2, int(num_ancillas))
        self.qpe_margin = float(qpe_margin)
        self._qpe_template_cache = {}

        self.backend = backend or AerSimulator()
        self.groups = self._group(hamiltonian)
        if self.readout == 'qpe':
            self.H_sense, self.h_offset = self._sensing_hamiltonian(hamiltonian)
            _nq = self.H_sense.num_qubits
            self.H0_norm = (
                float(np.linalg.norm(self.H_sense.to_matrix(), ord=2))
                if _nq <= 14 else float(np.sum(np.abs(self.H_sense.coeffs))))
            # Aliasing bound: phi = -E*tau0/2pi must stay inside one turn, so
            # |E|*tau0 <= pi. H_sense is traceless (the identity went to
            # h_offset), so ||H_sense||_2 bounds |E|.
            self.tau0 = np.pi / (self.qpe_margin * self.H0_norm + 1e-12)
        # ParameterView has no .index(), so map parameter -> global index once.
        self._pidx = {p: i for i, p in enumerate(ansatz.parameters)}
        self._direct_template_cache = {}
        # The natural layer partition is kept EVEN IN GLOBAL MODE, because the
        # measurement block and the step block are different objects. The gradient
        # is measured over one global block, which is where the G-circuit cost
        # comes from; the step is applied layer by layer, which is where the
        # scaling has to be local. See grad_step.
        self.step_layers = self._layers()
        if self.block_mode == 'global':
            self.layers = [{'params': list(range(self.M))}]
        else:
            self.layers = self.step_layers

    # ── setup ────────────────────────────────────────────────────────────────

    @staticmethod
    def _sensing_hamiltonian(H):
        """Strip the identity term for coherent sensing; return (H0, offset).

        QPE reads a phase proportional to an EIGENVALUE, and a constant shift in
        H is a global phase the ladder cannot see. Pulling the identity out and
        adding it back classically is what makes the decoded energy comparable
        with the direct readout's.
        """
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
        return H0, ident

    @staticmethod
    def _group(H):
        """Qubit-wise commuting groups: one measurement setting each.

        DO NOT SIMPLY FLIP THIS TO qubit_wise=False. It would be a large win on
        chemistry and it would silently produce WRONG ENERGIES, because _basis()
        below assumes qubit-wise groups: it builds one axis per qubit and lets the
        last label win. For a general-commuting group such as {XXI, YYI}, qubit 0
        is assigned 'X' then overwritten to 'Y', and the circuit measures neither.
        General commuting needs a full Clifford diagonaliser per group, which is
        not implemented.

        WHAT IT WOULD BE WORTH, measured in v119/v120 on v30's JW Hamiltonians:

            raw Pauli terms                4170 at N=12   ~ N^4.61
            G qubit-wise (this line)       1420           ~ N^4.24
            G general commuting (qiskit)    227           ~ N^3.79
            G general commuting (DSATUR)    175           ~ N^3.43

        8.1x fewer measurement settings at N=12, and the exponent falls by 0.81 -
        so the gap widens with size. Since V6's remaining quantum cost IS G, that
        is a direct cut to V6's cost.

        THREE REASONS IT IS STILL NOT DONE.
          1. The Clifford diagonaliser above, without which it is incorrect.
          2. It buys nothing on the benchmarked families: Heisenberg 3->2,
             MaxCut 1->1, H2 2->2, LiH 3->2. The 8.1x is chemistry-only and V6
             has never been benchmarked on chemistry.
          3. The Clifford depth is unpaid, and depth is the axis V6 already loses.

        AND IT DOES NOT WIDEN V6's ADVANTAGE. C_PS/C_V6 = 2MG/G = 2M, so G
        cancels and parameter-shift gets the same 8.1x. This lowers the floor for
        both rather than changing the ratio.

        Also measured (v120): Qiskit's grouper IS largest-first, and on the
        GENERAL-commuting graph DSATUR beats it by 20-23% while on the qubit-wise
        graph it beats it by only 1-4%. The heuristic ordering inverts between the
        two relations, so a grouper should try several and keep the best rather
        than trusting one. Minimum clique cover is NP-hard - arXiv:1907.03358.
        """
        return list(H.group_commuting(qubit_wise=True))

    def _layers(self):
        """Parameters acting on disjoint qubits, which can share one W gate."""
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

    # ── radius ───────────────────────────────────────────────────────────────

    def _radius(self, R, n):
        """Radius for a block of width n, from one quoted for an N-wide block.

        THE EXPONENT IS A KNOB AND WAS NEVER SWEPT. p = 0.5 is the shipped value
        and comes from one measurement at the V5->V6 transition: a block of n
        parameters displaces the state by about sqrt(n)*R, and handing V5's R
        straight to a 36-parameter block gave cos 0.886 instead of 0.975. That
        fixed a real regression and it fixed it at one M.

        SWEPT IN v111, AND p = 0.5 SURVIVED. The hypothesis was that the rescale
        is too aggressive at large M: it shrinks the SIGNAL, which goes as
        sin(R_eff), to buy down a bias going as R_eff^2, and by M=64 the bias has
        stopped being the binding constraint. Measured, that is wrong and the
        failure is not close:

            best p    M=16   M=32   M=64
            16384     0.25   0.25   0.50
            262144    0.50   0.50   0.50

        A smaller p wins only at low budget AND small M, by +0.012 to +0.015 cos.
        At M=64 it loses badly and p=0 collapses to cos 0.095 - no better than
        random. The reason is the displacement, not the signal: sqrt(M) R_eff is
        what the linearisation sees, and p = 0.5 is exactly the exponent that
        holds it fixed at sqrt(N) R. At M=64 with p=0 that displacement is about
        3.6 radians, far outside any linear regime.

        So the exponent is not a schedule in disguise, and V6's growing large-M
        deficit (v110: cos gap to parameter-shift +0.019 -> +0.025 -> +0.144 at
        16384 shots) is NOT self-inflicted by the rescale. Holding the
        displacement fixed forces R_eff ~ 1/sqrt(M), the signal falls with it,
        and no choice of p escapes that - it is the price of reading M components
        from G circuits.

        p stays exposed because it is now a knob with a measured answer rather
        than an unexamined constant. p = 0 disables the rescale entirely (V5's
        behaviour, the trap v81 keeps visible).
        """
        if not self.scale_radius or n <= self.N:
            return float(R)
        return float(R) * float((self.N / float(n)) ** self.radius_exponent)

    # ── circuit ──────────────────────────────────────────────────────────────

    def _direct_template(self, active, group):
        """Cached, parameterized design-encoded sensing circuit for one block."""
        key = (tuple(active), tuple(group.paulis.to_labels()))
        cached = self._direct_template_cache.get(key)
        if cached is not None:
            return cached

        n = len(active)
        ns = max(1, min(self.n_scratch, n))
        m_row, cols = _design_spec(n, ns, self.design_resolution)
        nreg = m_row + (1 if self.design_resolution >= 4 else 0)
        theta = list(self.ansatz.parameters)
        radius = Parameter(f'R_{n}_{len(self._direct_template_cache)}')
        pos = {p: i for i, p in enumerate(active)}

        qc = QuantumCircuit(QuantumRegister(nreg, 'param'),
                            QuantumRegister(self.N, 'sys'),
                            QuantumRegister(ns, 'par'),
                            ClassicalRegister(nreg, 'cp'),
                            ClassicalRegister(self.N, 'cs'))
        param, sysr, scr = qc.qregs[0], qc.qregs[1], qc.qregs[2]
        qc.h(param)

        # Seed each scratch wire once. The X makes it read NOT(parity) so the
        # controlled rotation fires on sigma = +1 without an X around every
        # rotation; the foldover bit contributes identically to every parameter
        # and so is folded in once per wire rather than twice per parameter.
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
                qc.append(op, qs)
                continue
            gi = self._pidx[next(iter(prm[0].parameters))]
            if gi not in pos:
                qc.append(op.__class__(theta[gi]), qs)
                continue
            if op.name not in _CTRL:
                raise ValueError(
                    f"V6 cannot build a controlled form of '{op.name}'. Add it to "
                    f"_CTRL, or the parameter would silently stay at its centre "
                    f"value and report a zero gradient.")
            p = pos[gi]
            s = p % ns
            c = cols[p]
            qc.append(op.__class__(theta[gi] - radius), qs)
            for b in range(m_row):
                if (c ^ prev[s]) >> b & 1:
                    qc.cx(param[b], scr[s])
            prev[s] = c
            getattr(qc, _CTRL[op.name])(2.0 * radius, scr[s], qs[0])

        # Return every scratch wire to |0>. Leaving them set would not change the
        # measured statistics, since each is a function of the register that is
        # itself measured, but clean ancillas keep the template composable.
        for s in range(ns):
            for b in range(m_row):
                if prev[s] >> b & 1:
                    qc.cx(param[b], scr[s])
            if self.design_resolution >= 4:
                qc.cx(param[m_row], scr[s])
            qc.x(scr[s])

        self._basis(qc, sysr, group)
        qc.measure(param, qc.cregs[0])
        qc.measure(sysr, qc.cregs[1])
        template = transpile(qc, self.backend, optimization_level=1)
        cached = (template, theta, radius)
        self._direct_template_cache[key] = cached
        return cached

    def _qpe_template(self, active):
        """Design-encoded W, then a QPE ladder instead of a measurement basis.

        Identical to _direct_template up to the point where that method calls
        _basis() and measures `sys`. Here the energy leaves as a PHASE instead,
        so the circuit is the same whatever H is - that is where G-independence
        comes from - and the design register is unchanged, which is where the
        log-width saving comes from. supplement/v132 built and measured it.

        The ladder acts only on `sys` under control of `anc`, so it never touches
        the design register. That is why the two savings compose at all.
        """
        key = tuple(active)
        cached = self._qpe_template_cache.get(key)
        if cached is not None:
            return cached

        n = len(active)
        ns = max(1, min(self.n_scratch, n))
        m_row, cols = _design_spec(n, ns, self.design_resolution)
        nreg = m_row + (1 if self.design_resolution >= 4 else 0)
        theta = list(self.ansatz.parameters)
        radius = Parameter(f'R_qpe_{n}')
        pos = {p: i for i, p in enumerate(active)}

        anc = AncillaRegister(self.k_anc, 'anc')
        qc = QuantumCircuit(anc,
                            QuantumRegister(nreg, 'param'),
                            QuantumRegister(self.N, 'sys'),
                            QuantumRegister(ns, 'par'),
                            ClassicalRegister(nreg, 'cp'),
                            ClassicalRegister(self.k_anc, 'ca'))
        param, sysr, scr = qc.qregs[1], qc.qregs[2], qc.qregs[3]
        qc.h(anc)
        qc.h(param)

        for s_ in range(ns):
            qc.x(scr[s_])
            if self.design_resolution >= 4:
                qc.cx(param[m_row], scr[s_])
        prev = [0] * ns
        for inst in self.ansatz.data:
            op = inst.operation
            qs = [sysr[self.ansatz.find_bit(b).index] for b in inst.qubits]
            prm = [pp for pp in op.params
                   if isinstance(pp, ParameterExpression) and pp.parameters]
            if not prm:
                qc.append(op, qs)
                continue
            gi = self._pidx[next(iter(prm[0].parameters))]
            if gi not in pos:
                qc.append(op.__class__(theta[gi]), qs)
                continue
            if op.name not in _CTRL:
                raise ValueError(
                    f"V6 cannot build a controlled form of '{op.name}'.")
            pp_ = pos[gi]
            s_ = pp_ % ns
            c = cols[pp_]
            qc.append(op.__class__(theta[gi] - radius), qs)
            for b in range(m_row):
                if (c ^ prev[s_]) >> b & 1:
                    qc.cx(param[b], scr[s_])
            prev[s_] = c
            getattr(qc, _CTRL[op.name])(2.0 * radius, scr[s_], qs[0])
        for s_ in range(ns):
            for b in range(m_row):
                if prev[s_] >> b & 1:
                    qc.cx(param[b], scr[s_])
            if self.design_resolution >= 4:
                qc.cx(param[m_row], scr[s_])
            qc.x(scr[s_])

        for a in range(self.k_anc):
            t = (2 ** a) * self.tau0
            reps = int(max(1, (2 ** a) // 2, np.ceil(t / 2.0)))
            qc.append(PauliEvolutionGate(
                self.H_sense, time=t,
                synthesis=SuzukiTrotter(order=2, reps=reps)).control(1),
                [anc[a]] + list(sysr))
        qc.append(QFTGate(self.k_anc).inverse(), anc)

        qc.measure(param, qc.cregs[0])
        qc.measure(anc, qc.cregs[1])
        template = transpile(qc, self.backend, optimization_level=1)
        cached = (template, theta, radius, m_row, cols, nreg)
        self._qpe_template_cache[key] = cached
        return cached

    def _sense_qpe(self, centre, R, active):
        """ONE circuit for the whole gradient, whatever G is.

        The decode is _sense's marginal over the design ROW; only the per-shot
        energy differs, arriving as a QPE phase rather than as Pauli signs.
        """
        n = len(active)
        Rv = self._radius(R, n)
        t_qc, theta, radius, m_row, cols, nreg = self._qpe_template(active)
        bind = {theta[i]: float(centre[i]) for i in range(len(theta))}
        bind[radius] = float(Rv)
        counts = self._run_transpiled(
            t_qc.assign_parameters(bind, inplace=False))

        num = np.zeros((2, n))
        den = np.zeros((2, n))
        e_tot = e_cnt = 0.0
        SG = _sign_table(m_row, cols)
        # cregs are created cp then ca, and Qiskit prints the LAST-created first,
        # so parts[0] is the phase word and parts[1] the design row.
        items = [(k, v) for k, v in counts.items() if len(k.split()) == 2]
        if items:
            mw = np.array([int(k.split()[0], 2) for k, _ in items], dtype=np.int64)
            regw = np.array([int(k.split()[1], 2) for k, _ in items], dtype=np.int64)
            cnts = np.array([v for _, v in items], dtype=float)
            phi = mw / float(1 << self.k_anc)
            phi = np.where(phi >= 0.5, phi - 1.0, phi)
            e = -2.0 * np.pi * phi / (self.tau0 + 1e-12) + self.h_offset

            d = regw & ((1 << m_row) - 1)
            if self.design_resolution >= 4 and nreg > m_row:
                fold = 1.0 - 2.0 * ((regw >> m_row) & 1)
            else:
                fold = np.ones(len(items))
            sg = SG[d] * fold[:, None]

            pos_ = sg > 0
            ec = e * cnts
            num[1] = (ec[:, None] * pos_).sum(0)
            num[0] = (ec[:, None] * ~pos_).sum(0)
            den[1] = (cnts[:, None] * pos_).sum(0)
            den[0] = (cnts[:, None] * ~pos_).sum(0)
            e_tot = float(ec.sum())
            e_cnt = float(cnts.sum())

        m1 = np.divide(num[1], den[1], out=np.zeros(n), where=den[1] > 0)
        m0 = np.divide(num[0], den[0], out=np.zeros(n), where=den[0] > 0)
        grad = np.zeros(len(centre))
        grad[active] = (m1 - m0) / (2.0 * Rv + 1e-12)
        return grad, (e_tot / e_cnt if e_cnt else 0.0)

    # ── decode ───────────────────────────────────────────────────────────────

    def sense(self, centre, R, active):
        """Marginal gradient over the block, plus the free degree-0 energy.

        One circuit per commuting group; every shot contributes to every
        parameter's marginal. sigma is reconstructed from the measured design ROW
        rather than read off one bit per parameter.

        With readout='qpe' this dispatches to _sense_qpe, which costs ONE circuit
        instead of G by reading the energy as a phase. See __init__.
        """
        if self.readout == 'qpe':
            return self._sense_qpe(centre, R, active)
        n = len(active)
        Rv = self._radius(R, n)
        ns = max(1, min(self.n_scratch, n))
        m_row, cols = _design_spec(n, ns, self.design_resolution)

        # PER-GROUP accumulation, then SUM. Accumulating across groups before
        # dividing computes a MEAN where the energy is a SUM (E = sum_g E_g),
        # which makes the gradient G times too small.
        m_sum = np.zeros(n)
        e_sum = 0.0

        for group in self.groups:
            num = np.zeros((2, n))
            den = np.zeros((2, n))
            e_tot = e_cnt = 0.0
            t_qc, theta, radius = self._direct_template(active, group)
            bind = {theta[i]: float(centre[i]) for i in range(len(theta))}
            bind[radius] = float(Rv)
            counts = self._run_transpiled(
                t_qc.assign_parameters(bind, inplace=False))

            labels = group.paulis.to_labels()
            coeffs = np.real(group.coeffs)
            # VECTORISED DECODE, bit-exact against the scalar loop it replaced.
            # Both registers are read as integers rather than reversed strings:
            # bit q of the system word is qubit q, which is what the Pauli label
            # indexes after reversal, so the reversal cancels and only the masks
            # below are needed. Verified num/den/e_tot/e_cnt identical by
            # np.array_equal on real counts.
            items = [(k, v) for k, v in counts.items() if len(k.split()) == 2]
            if items:
                sysw = np.array([int(k.split()[0], 2) for k, _ in items],
                                dtype=np.int64)
                regw = np.array([int(k.split()[1], 2) for k, _ in items],
                                dtype=np.int64)
                cnts = np.array([v for _, v in items], dtype=float)
                nsys = len(items[0][0].split()[0])
                nreg_m = len(items[0][0].split()[1])

                # energy per distinct outcome: c . (-1)^popcount(sys AND supp)
                supp = _support_masks(labels, nsys)
                signs = 1.0 - 2.0 * _parity(sysw[:, None] & supp[None, :])
                e = signs @ coeffs

                d = regw & ((1 << m_row) - 1)
                if self.design_resolution >= 4 and m_row < nreg_m:
                    fold = 1.0 - 2.0 * ((regw >> m_row) & 1)
                else:
                    fold = np.ones(len(items))
                sg = _sign_table(m_row, cols)[d] * fold[:, None]

                pos = sg > 0
                ec = e * cnts
                num[1] = (ec[:, None] * pos).sum(0)
                num[0] = (ec[:, None] * ~pos).sum(0)
                den[1] = (cnts[:, None] * pos).sum(0)
                den[0] = (cnts[:, None] * ~pos).sum(0)
                e_tot = float(ec.sum())
                e_cnt = float(cnts.sum())

            m1 = np.divide(num[1], den[1], out=np.zeros(n), where=den[1] > 0)
            m0 = np.divide(num[0], den[0], out=np.zeros(n), where=den[0] > 0)
            m_sum += m1 - m0                      # sum the GROUPS' contributions
            e_sum += (e_tot / e_cnt) if e_cnt else 0.0

        grad = np.zeros(len(centre))
        grad[active] = m_sum / (2.0 * Rv + 1e-12)
        # degree-0 Walsh coefficient: E(theta_c) + (R^2/2) Tr H + O(R^4). Biased,
        # and not correctable from these shots, since every vertex has x_i^2 = 1
        # so the diagonal curvature is degenerate with the constant. Monitoring
        # only; call energy_exact() once at the end.
        return grad, e_sum

    # ── step ─────────────────────────────────────────────────────────────────

    def grad_step(self, centre, R, active, grad):
        """Bounded step, max-normalised PER LAYER rather than per block.

        V5 divides by max|g| over the active block and calls it scale-free within
        the block, which is right when the block IS a layer: each layer then gets
        its own scale and advances at full step size whatever the other layers'
        gradient magnitudes are. That is what made it immune to the per-block
        scale bias measured at up to 2.4x, 80 sigma from unity.

        Under a GLOBAL measurement block the same line means something else. One
        max|g| taken over all M parameters divides every coordinate, so any
        parameter whose gradient is small next to the single largest component
        anywhere in the circuit barely moves. The normalisation that removed
        cross-block scale bias in V5 imposes it in V6. Measured consequence at
        Heisenberg N=4, 180 NEFV: -5.62 +/- 0.55 against V3's -6.00 +/- 0.03,
        with the spread being stalled runs rather than estimator noise.

        So the scale is taken per LAYER while the gradient is still measured over
        the whole block. Costs nothing: the gradient is already in hand, this only
        changes how it is consumed. Uses the same rescaled radius the gradient was
        measured at, or the step size and the smoothing scale disagree.
        """
        p = np.asarray(centre, dtype=float).copy()
        g_all = np.asarray(grad)
        # THE SCHEDULED RADIUS, NOT THE RESCALED ONE. _radius() shrinks R by
        # sqrt(n/N) so the LINEARISATION stays valid across a wider block; that is
        # a property of the estimator, not a statement about how far the optimiser
        # should walk. Using the reduced value here made a global block step at
        # sqrt(N/M) of V5's size on top of taking one step per epoch instead of L,
        # i.e. about an eighth of the movement. Measured at Heisenberg N=4, 60
        # NEFV, per-trial best: reduced gave -5.97 -4.59 -5.44 -5.92 -5.98,
        # scheduled gives -5.99 -4.67 -6.06 -6.09 -5.97, so two seeds cross V3's
        # -6.0030 at a third of its circuits.
        Rv = float(R)
        aset = set(int(i) for i in active)

        parts = [[i for i in lay['params'] if i in aset]
                 for lay in self.step_layers]
        parts = [q for q in parts if q]
        if not parts:                      # no layer structure: fall back to one
            parts = [list(aset)]

        for q in parts:
            g = g_all[q]
            mx = float(np.max(np.abs(g)))
            if mx < 1e-12:
                continue                   # inert layer, e.g. a commuting RZ block
            p[q] = p[q] - self.alpha * Rv * g / mx
        return p

    # ── driver ───────────────────────────────────────────────────────────────

    def _run_transpiled(self, t_qc):
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
        """Exact energy, 0 circuits on a simulator. Call ONCE at the end."""
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
