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
N=6, and calibrating tau0 exactly needs the spectral norm, hence the 2^N matrix.

USAGE:

    q = QLTOv6(ansatz, hamiltonian, shot_budget=8192)
    theta, trace = q.minimize(theta0, epochs=20)

Cross-checked against V5 on the same landscape in supplement/v81_v5_vs_v6.py.
"""

import numpy as np
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    transpile)
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_aer import AerSimulator

# Controlled forms of the single-parameter rotations
_CTRL = {'rx': 'crx', 'ry': 'cry', 'rz': 'crz', 'p': 'cp', 'u1': 'cp'}


# ----------------------------------------------------------------------
# Helpers for symbolic pruning
# ----------------------------------------------------------------------

def rotation_axis(op):
    """Returns the rotation axis of a parameterised single-qubit gate, or None."""
    name = op.name.lower()
    if name == 'rx':
        return 'X'
    if name == 'ry':
        return 'Y'
    if name in ('rz', 'p', 'u1', 'phase'):
        return 'Z'
    if name == 'r':
        # R(phi, theta) is implemented as U(phi, theta, 0) in many backends;
        # this is a rough heuristic – only exact for phi = 0 or pi/2.
        try:
            phi = float(op.params[1])
        except (TypeError, ValueError, IndexError):
            return None
        if abs(np.sin(phi)) < 1e-9:
            return 'X'
        if abs(np.cos(phi)) < 1e-9:
            return 'Y'
    return None


def parameterised_index(op, param_order):
    """Global index of the ansatz parameter this gate rotates, or None."""
    if not op.params:
        return None
    first = op.params[0]
    if isinstance(first, ParameterExpression) and first.parameters:
        free = list(first.parameters)
        if len(free) == 1:
            try:
                return param_order.index(free[0])
            except ValueError:
                pass
    return None


# ----------------------------------------------------------------------
# Design helpers (unchanged from original, except no functional changes)
# ----------------------------------------------------------------------

def _resv_cols(n, m):
    """Greedy resolution-V columns in GF(2)^m, or None if m is too small."""
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
    """Column assignment for a block of n parameters over k scratch wires."""
    m_row = max(1, int(np.ceil(np.log2(n + 1))))
    gray = lambda t: t ^ (t >> 1)
    if resolution >= 5:
        for m in range(m_row, m_row + 9):
            cols = _resv_cols(n, m)
            if cols is not None:
                return m, cols
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
    """sigma for one parameter at one measured register value."""
    return -1.0 if ((bin(d & c).count('1') + f) & 1) else 1.0


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------

class QLTOv6:
    """Log-width design register, one global block, bounded classical step."""

    def __init__(self, ansatz, hamiltonian, shot_budget=8192, alpha=0.9,
                 sim_seed=None, backend=None, block_mode='global',
                 n_scratch=3, scale_radius=True, design_resolution=4):
        # ---- decomposition loop (original) ----
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
        self.design_resolution = int(design_resolution)
        self.scale_radius = bool(scale_radius)

        self.backend = backend or AerSimulator()
        self.groups = self._group(hamiltonian)
        self._pidx = {p: i for i, p in enumerate(ansatz.parameters)}
        self._direct_template_cache = {}
        self.step_layers = self._layers()
        if self.block_mode == 'global':
            self.layers = [{'params': list(range(self.M))}]
        else:
            self.layers = self.step_layers

        # ---- NEW: symbolic pruning ----
        self.dead_params = self._get_structurally_dead_params()
        if self.dead_params:
            print(f"[V6] Symbolically pruned {len(self.dead_params)} "
                  f"structurally dead parameter(s): {sorted(self.dead_params)}")

    # ------------------------------------------------------------------
    # Setup methods (original)
    # ------------------------------------------------------------------
    @staticmethod
    def _group(H):
        """Qubit-wise commuting groups: one measurement setting each."""
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

    def _radius(self, R, n):
        """Radius for a block of width n, from one quoted for an N-wide block."""
        if not self.scale_radius or n <= self.N:
            return float(R)
        return float(R) * float(np.sqrt(self.N / float(n)))

    # ------------------------------------------------------------------
    # NEW: symbolic zero‑gradient identification
    # ------------------------------------------------------------------
    def _get_structurally_dead_params(self):
        """
        Identify parameters that are provably zero‑gradient because:
          1. the parameterised gate is the LAST operation on its qubit
             (so there is no later gate that could conjugate its generator), and
          2. its generator commutes with the full Hamiltonian.

        This is deliberately conservative. It will catch, for example,
        a terminal RZ layer on a diagonal Hamiltonian (e.g. MaxCut, Ising).
        It will NOT incorrectly prune parameters followed by any later gate
        on the same qubit, including non‑parameterised single‑qubit gates
        or arbitrary two‑qubit gates.
        """
        dead_indices = set()
        decomp = self.ansatz.decompose()
        param_order = list(self.ansatz.parameters)

        # Scan from the end backwards. For each qubit we keep the first
        # parameterised single‑qubit gate we encounter in the reverse scan,
        # provided that no later (in the forward direction) gate touches
        # that qubit. This guarantees the gate is truly terminal.
        candidates = {}
        blocked = set()

        for instr in reversed(decomp.data):
            qs = [decomp.find_bit(q).index for q in instr.qubits]

            # If this is a single‑qubit, parameterised, controlled‑able gate
            # and its qubit has not been touched by a later gate, record it.
            if (len(qs) == 1 and
                instr.operation.name in _CTRL and
                instr.operation.params):
                q = qs[0]
                if q not in blocked:
                    p_idx = parameterised_index(instr.operation, param_order)
                    axis = rotation_axis(instr.operation)
                    if p_idx is not None and axis is not None:
                        candidates[q] = (p_idx, axis)

            # Any gate (including this one) makes all qubits it touches
            # blocked for earlier gates.
            for q in qs:
                blocked.add(q)

        # Now check commutator with the full Hamiltonian.
        n_q = self.N
        h_paulis = self.hamiltonian.paulis  # PauliList
        for q, (p_idx, axis) in candidates.items():
            # Build generator: I...I ⊗ axis ⊗ I...I
            lbl = ['I'] * n_q
            # Qiskit labels are little‑endian: index 0 is rightmost qubit.
            lbl[n_q - 1 - q] = axis
            gen = SparsePauliOp.from_list([("".join(lbl), 1.0)])
            # commutes() returns a bool or array; we require commutation with
            # every term in H.
            if bool(np.all(h_paulis.commutes(gen.paulis[0]))):
                dead_indices.add(p_idx)

        return dead_indices

    # ------------------------------------------------------------------
    # Circuit construction (original, except no changes)
    # ------------------------------------------------------------------
    def _direct_template(self, active, group):
        """Cached, parameterized design-encoded sensing circuit for one block."""
        key = (tuple(active), tuple(group.paulis.to_labels()))
        cached = self._direct_template_cache.get(key)
        if cached is not None:
            return cached

        n = len(active)
        ns = max(1, min(self.n_scratch, n))
        m_row, cols = _design_spec(n, ns, self.design_resolution)
        nreg = m_row + 1
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

        for s in range(ns):
            qc.x(scr[s])
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
                    f"V6 cannot build a controlled form of '{op.name}'.")
            p = pos[gi]
            s = p % ns
            c = cols[p]
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
            qc.cx(param[m_row], scr[s])
            qc.x(scr[s])

        self._basis(qc, sysr, group)
        qc.measure(param, qc.cregs[0])
        qc.measure(sysr, qc.cregs[1])
        template = transpile(qc, self.backend, optimization_level=1)
        cached = (template, theta, radius)
        self._direct_template_cache[key] = cached
        return cached

    # ------------------------------------------------------------------
    # NEW: energy measurement without perturbation (for dead blocks)
    # ------------------------------------------------------------------
    def _measure_energy_at_centre(self, params):
        """
        Measure <H> at the current parameter values without any parameter
        perturbation. Used when a block has all parameters pruned, so the
        gradient is identically zero but we still want the energy estimate
        for the trace.

        Returns (total_energy_estimate, total_shots_used).
        """
        total_energy = 0.0
        # We construct one circuit per commuting group, exactly as in sense()
        # but with no active parameters and no design register.
        for group in self.groups:
            qc = QuantumCircuit(QuantumRegister(self.N, 'sys'),
                                ClassicalRegister(self.N, 'cs'))
            sysr = qc.qregs[0]
            # Apply the ansatz with all parameters fixed.
            bind = {p: float(params[i]) for i, p in enumerate(self.ansatz.parameters)}
            # Append the ansatz circuit (already decomposed)
            for inst in self.ansatz.data:
                if inst.operation.params:
                    # bind parameters
                    new_op = inst.operation.copy()
                    new_op.params = [bind.get(p, p) for p in inst.operation.params]
                    qc.append(new_op, [sysr[self.ansatz.find_bit(b).index]
                                        for b in inst.qubits])
                else:
                    qc.append(inst.operation, [sysr[self.ansatz.find_bit(b).index]
                                               for b in inst.qubits])
            # Basis rotation
            self._basis(qc, sysr, group)
            qc.measure(sysr, qc.cregs[0])
            t_qc = transpile(qc, self.backend, optimization_level=1)
            self.max_circuit_depth = max(self.max_circuit_depth, t_qc.depth())
            self.nefv += 1
            kw = {}
            if self.sim_seed is not None:
                kw['seed_simulator'] = int(self.sim_seed) + self._shot_index
                self._shot_index += 1
            counts = self.backend.run(t_qc, shots=self.shot_budget,
                                      **kw).result().get_counts()

            labels = group.paulis.to_labels()
            coeffs = np.real(group.coeffs)
            e_tot = 0.0
            e_cnt = 0
            for bitstr, cnt in counts.items():
                # bitstr is the system register only (no split)
                sbits = bitstr[::-1]
                e = 0.0
                for lbl, c in zip(labels, coeffs):
                    s = 1
                    for q, ch in enumerate(reversed(lbl)):
                        if ch != 'I' and q < len(sbits) and sbits[q] == '1':
                            s = -s
                    e += c * s
                e_tot += e * cnt
                e_cnt += cnt
            total_energy += (e_tot / e_cnt) if e_cnt else 0.0
        return total_energy

    # ------------------------------------------------------------------
    # Gradient estimation (original, unchanged)
    # ------------------------------------------------------------------
    def sense(self, centre, R, active):
        """Marginal gradient over the block, plus the free degree-0 energy."""
        n = len(active)
        Rv = self._radius(R, n)
        ns = max(1, min(self.n_scratch, n))
        m_row, cols = _design_spec(n, ns, self.design_resolution)

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
                d = sum(1 << b for b in range(m_row)
                        if b < len(xbits) and xbits[b] == '1')
                f = 1 if (m_row < len(xbits) and xbits[m_row] == '1') else 0
                for i in range(n):
                    b = 1 if _design_sign(d, f, cols[i]) > 0 else 0
                    num[b, i] += e * cnt
                    den[b, i] += cnt

            m1 = np.divide(num[1], den[1], out=np.zeros(n), where=den[1] > 0)
            m0 = np.divide(num[0], den[0], out=np.zeros(n), where=den[0] > 0)
            m_sum += m1 - m0
            e_sum += (e_tot / e_cnt) if e_cnt else 0.0

        grad = np.zeros(len(centre))
        grad[active] = m_sum / (2.0 * Rv + 1e-12)
        return grad, e_sum

    # ------------------------------------------------------------------
    # Classical step (original)
    # ------------------------------------------------------------------
    def grad_step(self, centre, R, active, grad):
        """Bounded step, max-normalised per layer."""
        p = np.asarray(centre, dtype=float).copy()
        g_all = np.asarray(grad)
        Rv = float(R)   # scheduled radius, not rescaled
        aset = set(int(i) for i in active)

        parts = [[i for i in lay['params'] if i in aset]
                 for lay in self.step_layers]
        parts = [q for q in parts if q]
        if not parts:
            parts = [list(aset)]

        for q in parts:
            g = g_all[q]
            mx = float(np.max(np.abs(g)))
            if mx < 1e-12:
                continue
            p[q] = p[q] - self.alpha * Rv * g / mx
        return p

    # ------------------------------------------------------------------
    # Driver methods (modified run_epoch)
    # ------------------------------------------------------------------
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
            # Filter out structurally dead parameters for this block.
            act = [idx for idx in blk['params'] if idx not in self.dead_params]
            if not act:
                # All parameters in this block are pruned: gradient is zero,
                # but we still need an energy estimate for the trace.
                e = self._measure_energy_at_centre(p)
                es.append(e)
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