"""
commute_gradient.py

Exact implementation of Bowles et al. (2024) Theorem 3:
"Backpropagation scaling in parameterised quantum circuits"
arXiv:2306.14962v4

─────────────────────────────────────────────────────────────────────────────
THEORY SUMMARY
─────────────────────────────────────────────────────────────────────────────

For commuting-block circuits with B blocks, the full gradient ∇C(θ) can be
obtained from 2B − 1 circuits (Appendix B, Fig 7):

  Last block (b=B):  1 circuit using the commutator observable (Theorem 1)
                     ∂C/∂θⱼᴮ = ⟨ψ_B | i[Gⱼ, H] | ψ_B⟩

  Each intermediate block b:  2 circuits (one for O₀, one for O₁):
                     ∂C/∂θⱼᵇ = 2⟨ Z_anc ⊗ Oⱼ ⟩_{|ϕ_b⟩}

  where the LCU state is:
    |ϕ_b⟩ = ½[|0⟩(W̃+W')|ψ_b⟩ + |1⟩(W̃−W')|ψ_b⟩]

  prepared by the Fig-7 circuit:
    |0⟩_anc |ψ_b⟩_sys
        ↓  H_anc
        ↓  X_anc → ctrl-W̃ (control on |1⟩) → X_anc  (= ctrl on |0⟩)
        ↓  ctrl-W' (control on |1⟩)
        ↓  H_anc
        → measure  Z_anc ⊗ Oⱼ_sys

  with:
    W  = U_{b+1} ⋯ U_B          (future blocks as a unitary)
    W̃  = Gⱼ · W · Gⱼ           (W conjugated by generator — exact via Operator)
    W' = i^{1−gⱼ} · W           (W scaled by phase)
    Oⱼ = iᵍʲ · Gⱼ · H          (observable; Hermitian when computed correctly)
    gⱼ = 1 if {Gⱼ, H_eff} = 0  (anticommute)
    gⱼ = 0 if [Gⱼ, H_eff] = 0  (commute)
    H_eff = W†HW                 (effective observable after future blocks)

─────────────────────────────────────────────────────────────────────────────
IMPORTANT NOTE ON W̃ AND NEFV SCALING
─────────────────────────────────────────────────────────────────────────────

The true 2B−1 NEFV (Sampler shots) requires that W̃ is THE SAME for all
generators Gⱼ in block b.  This holds when generators differ only in qubit
index and the future circuit W contains no CNOT gates that couple generator
qubits to other qubits (pure commuting-generator circuits, Sec 3 of Bowles).

For EfficientSU2 with linear CNOT entanglement, CNOTs couple qubit k to k+1,
so W̃_j = Gⱼ·W·Gⱼ is QUBIT-SPECIFIC.  We therefore compute W̃_j exactly per
generator via Operator conjugation (O(4^n), practical for n ≤ 10 qubits).

NEFV accounting:
  • Pure commuting circuits (no CNOTs in W): 2B−1  (one circuit per block)
  • EfficientSU2 / CNOT-entangled:           O(M)  (one circuit per param)
    but each circuit is CORRECT (vs the wrong Hadamard test in the old code).

The get_nefv_cost() method returns both values for transparency.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Pauli anticommutation
# ─────────────────────────────────────────────────────────────────────────────

def _pauli_anticommutes(p1: str, p2: str) -> bool:
    """Return True iff single-qubit Paulis p1 and p2 anticommute."""
    if p1 == 'I' or p2 == 'I':
        return False
    return p1 != p2  # X,Y,Z: same type commutes, different types anticommute


# ─────────────────────────────────────────────────────────────────────────────
# Helper: split SparsePauliOp relative to a single-qubit generator
# ─────────────────────────────────────────────────────────────────────────────

def _split_H_by_commutation(cost_op: SparsePauliOp, gen_pauli_type: str,
                             gen_qubit_idx: int):
    """
    Split cost_op into H_anticomm and H_comm relative to generator
    Gⱼ = <gen_pauli_type> on qubit <gen_qubit_idx>.

    A term Q in cost_op anticommutes with Gⱼ iff the Pauli on gen_qubit_idx
    in Q anticommutes with gen_pauli_type.

    Returns (H_anticomm, H_comm) as SparsePauliOp or None if empty.
    """
    n = cost_op.num_qubits
    ac_labels, ac_coeffs = [], []
    co_labels, co_coeffs = [], []

    for pauli, coeff in zip(cost_op.paulis, cost_op.coeffs):
        label = pauli.to_label()          # big-endian string, len = n
        # qubit 0 in Qiskit is the RIGHTMOST character of the label
        char_idx = n - 1 - gen_qubit_idx
        local_pauli = label[char_idx]

        if _pauli_anticommutes(gen_pauli_type, local_pauli):
            ac_labels.append(label)
            ac_coeffs.append(coeff)
        else:
            co_labels.append(label)
            co_coeffs.append(coeff)

    H_ac = (SparsePauliOp(ac_labels, ac_coeffs).simplify()
            if ac_labels else None)
    H_co = (SparsePauliOp(co_labels, co_coeffs).simplify()
            if co_labels else None)
    return H_ac, H_co


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build Observable Oⱼ for a single generator
# ─────────────────────────────────────────────────────────────────────────────

def _build_Oj(gen_op: SparsePauliOp, H_anticomm, H_comm):
    """
    Compute Oⱼ as a Hermitian SparsePauliOp:

      O₁ contribution (gⱼ=1, anticommuting terms):  2i·Gⱼ·H_anticomm
      O₀ contribution (gⱼ=0,  commuting terms):     2·Gⱼ·H_comm

    Returns the combined observable as a real-coefficient SparsePauliOp
    (imaginary parts should cancel for Hermitian Oⱼ).
    """
    terms = []
    if H_anticomm is not None:
        raw = (2j * (gen_op @ H_anticomm)).simplify()
        # iGH where {G,H}=0 is Hermitian → coefficients should be real
        terms.append(SparsePauliOp(raw.paulis, np.real(raw.coeffs)))
    if H_comm is not None:
        raw = (2.0 * (gen_op @ H_comm)).simplify()
        terms.append(SparsePauliOp(raw.paulis, np.real(raw.coeffs)))

    if not terms:
        return None
    combined = sum(terms).simplify()
    # Sanitize residual floating-point imaginary parts
    return SparsePauliOp(combined.paulis, np.real(combined.coeffs))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: determine W' phase
# ─────────────────────────────────────────────────────────────────────────────

def _get_W_prime_phase(gen_pauli_type: str, gen_qubit: int,
                       cost_op: SparsePauliOp) -> complex:
    """
    Determine the phase factor i^{1-gⱼ} for W'.

    gⱼ is defined by the DOMINANT commutation relation of Gⱼ with H:
    If anticommuting terms exist → gⱼ=1, W'=W  (phase = i^0 = 1)
    If only commuting terms      → gⱼ=0, W'=iW  (phase = i^1 = i)

    When both parts exist, gⱼ=1 takes priority (standard convention).
    """
    H_ac, _ = _split_H_by_commutation(cost_op, gen_pauli_type, gen_qubit)
    gj = 1 if H_ac is not None else 0
    return (1j) ** (1 - gj)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class CommutingBlockGradient:
    """
    Exact Bowles et al. (2024) Theorem 3 gradient estimator.

    Improvements over the previous (incorrect) implementation:
    ──────────────────────────────────────────────────────────
    OLD  Hadamard test approximation:
         ∂C/∂θⱼᵇ ≈ ⟨Z_anc ⊗ (H·Gⱼ)⟩   ← drops W̃ conjugation, wrong for
                                            non-last layers with CNOTs

    NEW  Exact LCU construction (Appendix B, Eq 59–63):
         ∂C/∂θⱼᵇ = 2⟨Z_anc ⊗ Oⱼ⟩_{|ϕ_b⟩}
         |ϕ_b⟩ prepared by H → ctrl-W̃ → ctrl-W' → H circuit (Fig 7)
         W̃ = Gⱼ·W·Gⱼ computed exactly via Qiskit Operator conjugation
    """

    def __init__(self, ansatz, cost_op: SparsePauliOp):
        self.cost_op = cost_op

        # Preserve the *external* parameter order from the input circuit.
        # Callers (e.g. optimizers) typically pass params_values aligned to this.
        self._input_ansatz = ansatz
        self._input_params = list(ansatz.parameters)
        self.num_params = len(self._input_params)
        self.param_map = {p: i for i, p in enumerate(self._input_params)}

        # Normalize the ansatz to single-parameter Pauli rotations so the LCU
        # construction (which assumes exp(-i θ P/2) gates) applies. This also
        # makes EfficientSU2(su2_gates=['u3']) usable by turning multi-parameter
        # 'u'/'u3' gates into rx/ry/rz (+ cx) sequences.
        self.ansatz = self._normalize_ansatz(ansatz)

        # Map external parameter vectors (input order) into the normalized
        # circuit's parameter order.
        self._work_params = list(self.ansatz.parameters)
        self._work_index = {p: i for i, p in enumerate(self._work_params)}
        # If normalization dropped parameters (shouldn't happen for symbolic
        # circuits), fall back to the unnormalized input circuit.
        try:
            self._orig_to_work = [self._work_index[p] for p in self._input_params]
        except KeyError:
            self.ansatz = ansatz
            self._work_params = list(self.ansatz.parameters)
            self._work_index = {p: i for i, p in enumerate(self._work_params)}
            self._orig_to_work = [self._work_index[p] for p in self._input_params]

        self.layers = self._detect_layers()
        self.param_to_layer = {}
        for l_idx, layer in enumerate(self.layers):
            for p_idx in layer['params']:
                self.param_to_layer[p_idx] = l_idx

    def _normalize_ansatz(self, ansatz: QuantumCircuit) -> QuantumCircuit:
        """Best-effort conversion to rx/ry/rz/cx basis for 1-parameter gates."""
        try:
            from qiskit import transpile

            normalized = transpile(
                ansatz,
                basis_gates=['rx', 'ry', 'rz', 'cx'],
                optimization_level=0,
            )

            # Ensure we didn't lose symbolic parameters.
            if set(ansatz.parameters) <= set(normalized.parameters):
                return normalized
        except Exception:
            pass
        return ansatz

    def _to_work_params(self, params_values) -> np.ndarray:
        """Reorder an external parameter vector into self.ansatz order."""
        params_arr = np.asarray(params_values, dtype=float)
        if params_arr.shape != (self.num_params,):
            raise ValueError(
                f"params_values must be shape ({self.num_params},), got {params_arr.shape}"
            )
        work = np.zeros(len(self._work_params), dtype=float)
        work[self._orig_to_work] = params_arr
        return work

    # ── Layer detection ────────────────────────────────────────────────────

    def _detect_layers(self):
        """
        Group decomposed ansatz gates into commuting blocks by Pauli generator
        type (X / Y / Z).  A CNOT or other non-parameterised gate ends the
        current block.  Each layer dict contains:
          'params'    : list of global parameter indices
          'generators': list of SparsePauliOp (one per param)
          'gen_qubits': list of qubit indices for each generator
          'end_index' : instruction index of the last gate in this block
          'type'      : 'X', 'Y', or 'Z'
        """
        layers = []
        cur = {'params': [], 'generators': [], 'gen_qubits': [],
               'end_index': -1, 'type': None}
        decomposed = self.ansatz.decompose()

        for idx, instr in enumerate(decomposed.data):
            p_idx = -1
            if instr.operation.params:
                param = instr.operation.params[0]
                target = None
                if isinstance(param, ParameterExpression):
                    ps = list(param.parameters)
                    if ps:
                        target = ps[0]
                elif isinstance(param, Parameter):
                    target = param
                if target in self.param_map:
                    p_idx = self.param_map[target]

            if p_idx != -1:
                name = instr.operation.name.lower()
                # Qiskit decompositions often use:
                # - "r(theta, phi)" for Ry(theta) when phi≈pi/2
                # - "p(theta)" for phase/virtual Rz(theta) (global phase ignored)
                if name == 'rx':
                    gen_type = 'X'
                elif name == 'ry':
                    gen_type = 'Y'
                elif name in {'rz', 'p', 'u1'}:
                    gen_type = 'Z'
                elif name == 'r':
                    # RGate(theta, phi) rotates around axis in the xy-plane.
                    # EfficientSU2(… su2_gates=['ry','rz']) decomposes Ry into R(theta, pi/2).
                    # We treat phi≈pi/2 (mod 2pi) as a Y generator, phi≈0 as X.
                    try:
                        phi = float(instr.operation.params[1])
                    except Exception:
                        phi = np.pi / 2.0
                    two_pi = 2.0 * np.pi
                    phi_mod = ((phi + np.pi) % two_pi) - np.pi
                    if np.isclose(abs(phi_mod), np.pi / 2.0, atol=1e-6):
                        gen_type = 'Y'
                    elif np.isclose(phi_mod, 0.0, atol=1e-6) or np.isclose(abs(phi_mod), np.pi, atol=1e-6):
                        gen_type = 'X'
                    else:
                        # Fallback: unknown axis → treat as Y (matches common Ry decomposition)
                        gen_type = 'Y'
                else:
                    # Default: treat as Z-like phase rotation
                    gen_type = 'Z'

                # Build single-qubit generator as n-qubit SparsePauliOp
                q_idx   = instr.qubits[0]._index
                op_list = ['I'] * self.ansatz.num_qubits
                op_list[self.ansatz.num_qubits - 1 - q_idx] = gen_type
                gen_op  = SparsePauliOp("".join(op_list))

                # New block when generator type changes
                if cur['type'] is not None and cur['type'] != gen_type:
                    layers.append(cur)
                    cur = {'params': [], 'generators': [], 'gen_qubits': [],
                           'end_index': -1, 'type': None}

                cur['type'] = gen_type
                cur['params'].append(p_idx)
                cur['generators'].append(gen_op)
                cur['gen_qubits'].append(q_idx)
                cur['end_index'] = idx

            elif cur['params']:
                # Fixed gate (CNOT, barrier …) ends the current block
                layers.append(cur)
                cur = {'params': [], 'generators': [], 'gen_qubits': [],
                       'end_index': -1, 'type': None}

        if cur['params']:
            layers.append(cur)
        return layers

    # ── Circuit slicing ────────────────────────────────────────────────────

    def _slice_ansatz(self, start_idx: int, end_idx: int,
                      params_values) -> QuantumCircuit:
        """Return a sub-circuit for instruction indices [start_idx, end_idx]."""
        work_params = self._to_work_params(params_values)
        bound = self.ansatz.assign_parameters(work_params)
        dec   = bound.decompose()
        qc    = QuantumCircuit(*bound.qregs)
        for reg in bound.cregs:
            qc.add_register(reg)
        for i in range(start_idx, min(end_idx + 1, len(dec.data))):
            inst = dec.data[i]
            if inst.operation.name == 'barrier':
                continue
            qc.append(inst.operation, inst.qubits, inst.clbits)
        return qc

    # ── Exact W̃ construction ──────────────────────────────────────────────

    def _build_W_tilde_gate(self, gen_op: SparsePauliOp,
                            u_future: QuantumCircuit) -> UnitaryGate:
        """
        Compute W̃ = Gⱼ · W · Gⱼ exactly using Qiskit Operator arithmetic.

        Gⱼ is Hermitian and unitary so Gⱼ⁻¹ = Gⱼ.
        Complexity: O(4^n) — practical for n ≤ 10 qubits.

        This is EXACT regardless of CNOT structure in W, unlike the
        approximate "negate angles" approach used in the old code.
        """
        Gj_op = Operator(gen_op)
        W_op  = Operator(u_future)
        W_tilde_data = (Gj_op @ W_op @ Gj_op).data
        return UnitaryGate(W_tilde_data, label='W~')

    # ── LCU circuit (Bowles Fig 7) ─────────────────────────────────────────

    def _build_lcu_circuit(self, u_past: QuantumCircuit,
                           W_gate: UnitaryGate,
                           W_tilde_gate: UnitaryGate,
                           apply_i_phase_on_ancilla: bool = False) -> QuantumCircuit:
        """
        Build the linear-combination-of-unitaries (LCU) circuit that prepares

          |ϕ_b⟩ = ½[|0⟩(W̃ + W')|ψ_b⟩  +  |1⟩(W̃ − W')|ψ_b⟩]

        Circuit (Bowles Fig 7, Appendix B):
          1. Apply U_past to sys register  → |ψ_b⟩
          2. H on ancilla                 → ½(|0⟩+|1⟩)|ψ_b⟩
          3. ctrl-W̃ controlled on |0⟩    → ½(|0⟩W̃|ψ_b⟩ + |1⟩|ψ_b⟩)
          4. ctrl-W' controlled on |1⟩   → ½(|0⟩W̃|ψ_b⟩ + |1⟩W'|ψ_b⟩)
          5. H on ancilla (basis change)  → |ϕ_b⟩

        Measuring Z_anc ⊗ Oⱼ on |ϕ_b⟩ yields ½ ∂C/∂θⱼᵇ  (× 2 in caller).
        """
        n   = self.ansatz.num_qubits
        qr  = QuantumRegister(n,  'sys')
        anc = AncillaRegister(1,  'anc')
        qc  = QuantumCircuit(anc, qr)

        # 1. Prepare |ψ_b⟩
        qc.compose(u_past, qubits=qr, inplace=True)

        # 2. Superposition on ancilla
        qc.h(anc[0])

        # 3. ctrl-W̃ on |0⟩  (X → ctrl-W̃ on |1⟩ → X  =  ctrl on |0⟩)
        qc.x(anc[0])
        qc.append(W_tilde_gate.control(1), [anc[0]] + list(qr))
        qc.x(anc[0])

        # 4. ctrl-W' on |1⟩
        # For the commuting case (g=0) the theory requires W' = iW.
        # Implement the global phase i as an explicit S gate on the ancilla:
        #   diag(I, iW) = (S ⊗ I) · diag(I, W)
        # This avoids relying on UnitaryGate(global_phase) behavior under control.
        if apply_i_phase_on_ancilla:
            qc.s(anc[0])
        qc.append(W_gate.control(1), [anc[0]] + list(qr))

        # 5. Basis change on ancilla  (so Z_anc measures the interference)
        qc.h(anc[0])

        return qc

    # ── Build the Z_anc ⊗ Oⱼ observable ──────────────────────────────────

    def _build_combined_observable(self, Oj: SparsePauliOp) -> SparsePauliOp:
        """
                Construct  (Oⱼ_sys ⊗ Z_anc)  as a SparsePauliOp on (n+1) qubits.

                Important Qiskit ordering note:
                - In a circuit built as QuantumCircuit(anc, sys), the ancilla is qubit 0
                    (least significant), so Z_on_ancilla is the RIGHTMOST Pauli.
                - SparsePauliOp.tensor follows this convention: A.tensor(B) acts as
                    (A ⊗ B) where B is on the least-significant qubits.
        """
        # n      = self.ansatz.num_qubits
        # Z_anc  = SparsePauliOp("Z" + "I" * n)   # ancilla is qubit 0 in circuit
        # return Z_anc.tensor(Oj)                   # tensor: (anc) ⊗ (sys)
        Z_anc = SparsePauliOp("Z")
        return Oj.tensor(Z_anc)

    # ── Main gradient computation ──────────────────────────────────────────

    def compute_gradient(self, estimator, params_values):
        """
        Exact gradient via Bowles Theorem 3 LCU construction.

        Returns gradient vector ∇C(θ) of length num_params.

        NEFV:
          Last block       : 1 circuit  (commutator observable, parallel)
          Each intermediate: 1 circuit per generator in that block
                             (exact LCU; would be 2 circuits total if no CNOTs)
          Total            : 1 + Σ_{b<B} M_b  unique circuits

        Falls back to parameter-shift if LCU fails (hardware without
        arbitrary controlled unitaries, or transpilation failures).
        """
        gradients = np.zeros(self.num_params)

        work_params = self._to_work_params(params_values)
        bound     = self.ansatz.assign_parameters(work_params)
        dec       = bound.decompose()
        total_ins = len(dec.data)

        pubs      = []
        meta      = []   # list of (scale_factor, global_param_idx)

        for l_idx, layer in enumerate(self.layers):
            idxs = layer['params']
            if not idxs:
                continue

            end_idx      = layer['end_index']
            is_last      = (end_idx >= total_ins - 1) or (l_idx == len(self.layers) - 1)
            gen_type     = layer['type']

            # ── LAST BLOCK: direct commutator measurement (Theorem 1) ────────
            if is_last:
                qc_last = self._slice_ansatz(0, end_idx, params_values)

                for p_local, p_global in enumerate(idxs):
                    gen = layer['generators'][p_local]

                    # ∂C/∂θⱼᴮ = ⟨i[Gⱼ, H]⟩ = ½i⟨[Gⱼ, H]⟩
                    comm = (0.5j * (gen @ self.cost_op
                                    - self.cost_op @ gen)).simplify()
                    comm_real = SparsePauliOp(comm.paulis,
                                             np.real(comm.coeffs))

                    if not np.isclose(np.sum(np.abs(comm_real.coeffs)), 0):
                        pubs.append((qc_last, comm_real))
                        meta.append((1.0, p_global))

            # ── INTERMEDIATE BLOCKS: exact LCU construction (Theorem 3) ─────
            else:
                u_past   = self._slice_ansatz(0, end_idx, params_values)
                u_future = self._slice_ansatz(end_idx + 1, total_ins - 1,
                                              params_values)

                for p_local, p_global in enumerate(idxs):
                    gen       = layer['generators'][p_local]
                    gen_qubit = layer['gen_qubits'][p_local]

                    # ── Split H into commuting / anticommuting parts relative to Gⱼ ──
                    # For general SparsePauliOp sums, both parts may be non-empty.
                    # The Appendix-B circuit uses W' = iW for commuting terms (g=0)
                    # and W' = W for anticommuting terms (g=1), so we evaluate them
                    # as separate PUBs and sum their contributions.
                    H_ac, H_co = _split_H_by_commutation(self.cost_op, gen_type, gen_qubit)
                    if H_ac is None and H_co is None:
                        continue

                    # ── Build W̃ = Gⱼ · W · Gⱼ (exact) ─────────────────────
                    try:
                        W_tilde_gate = self._build_W_tilde_gate(gen, u_future)
                    except Exception:
                        # Fallback: param-shift for this parameter only
                        pubs.append(('param_shift', p_global))
                        meta.append((None, p_global))
                        continue

                    W_op  = Operator(u_future)
                    W_gate = UnitaryGate(W_op.data, label='W')

                    # Helper: add one LCU PUB for a specific (H_part, g)
                    def _add_lcu_pub(H_part: SparsePauliOp, g: int):
                        if H_part is None:
                            return
                        # O_j = i^g * (G/2) * H_part.
                        # We implement this by building Oj = i^g * G * H_part (no 1/2),
                        # and then applying an overall scale 0.5 to match (G/2).
                        phase_oj = (1j) ** g
                        raw = (phase_oj * (gen @ H_part)).simplify()
                        Oj = SparsePauliOp(raw.paulis, np.real(raw.coeffs))
                        if Oj is None or np.isclose(np.sum(np.abs(Oj.coeffs)), 0):
                            return

                        # W' = i^{1-g} W.
                        # For g=0 this is iW which we implement via an S gate on the ancilla.
                        apply_i_phase = (g == 0)
                        qc_lcu = self._build_lcu_circuit(
                            u_past,
                            W_gate,
                            W_tilde_gate,
                            apply_i_phase_on_ancilla=apply_i_phase,
                        )

                        # Synthesis: controlled dense unitaries must be decomposed for Aer primitives.
                        from qiskit import transpile
                        qc_lcu = transpile(qc_lcu, basis_gates=['u', 'cx'], optimization_level=1)

                        obs = self._build_combined_observable(Oj)

                        # Gradient contribution = ⟨ 2(Z ⊗ O_j) ⟩, with O_j using (G/2).
                        # Since we built Oj with G (not G/2), multiply by 0.5.
                        pubs.append((qc_lcu, obs))
                        meta.append((0.5 * 2.0, p_global))

                    # Add contributions for commuting (g=0) and anticommuting (g=1) parts.
                    _add_lcu_pub(H_co, g=0)
                    _add_lcu_pub(H_ac, g=1)

        # ── Execute all PUBs ─────────────────────────────────────────────────
        if not pubs:
            return self.compute_gradient_param_shift(estimator, params_values)

        # Separate any fallback param-shift markers
        estimator_pubs  = [(i, p, m) for i, (p, m) in enumerate(zip(pubs, meta))
                           if not (isinstance(p, str) and p == 'param_shift')]
        paramshift_idxs = [m[1] for p, m in zip(pubs, meta)
                           if isinstance(p, str) and p == 'param_shift']

        # Run Estimator batch
        if estimator_pubs:
            real_pubs = [p for _, p, _ in estimator_pubs]
            real_meta = [(s, g) for _, _, (s, g) in estimator_pubs]
            try:
                results = estimator.run(real_pubs).result()
                for i, (scale, p_idx) in enumerate(real_meta):
                    gradients[p_idx] += scale * float(results[i].data.evs)
            except Exception as e:
                print(f"  [CommGrad] Estimator failed ({e}), using param-shift.")
                return self.compute_gradient_param_shift(estimator, params_values)

        # Run param-shift for any failed individual parameters
        if paramshift_idxs:
            ps_grad = self.compute_gradient_param_shift(estimator, params_values)
            for idx in paramshift_idxs:
                gradients[idx] = ps_grad[idx]

        # Sanity: fall back entirely if all zeros (dead gradient)
        if np.allclose(gradients, 0.0):
            return self.compute_gradient_param_shift(estimator, params_values)

        return gradients

    # ── Parameter-shift fallback ───────────────────────────────────────────

    def compute_gradient_param_shift(self, estimator, params_values):
        """
        Standard parameter-shift rule — robust fallback.
        Cost: 2M circuits where M = num_params.
        """
        gradients = np.zeros(self.num_params)
        shift = np.pi / 2.0
        pubs  = []

        base_work = self._to_work_params(params_values)

        for i in range(self.num_params):
            w_idx = self._orig_to_work[i]
            p_p = np.array(base_work, dtype=float); p_p[w_idx] += shift
            p_m = np.array(base_work, dtype=float); p_m[w_idx] -= shift
            pubs.append((self.ansatz, self.cost_op, p_p))
            pubs.append((self.ansatz, self.cost_op, p_m))

        try:
            results = estimator.run(pubs).result()
            for i in range(self.num_params):
                val_p = float(results[2 * i    ].data.evs)
                val_m = float(results[2 * i + 1].data.evs)
                gradients[i] = 0.5 * (val_p - val_m)
        except Exception:
            pass

        return gradients

    # ── NEFV cost reporting ────────────────────────────────────────────────

    def get_nefv_cost(self):
        """
        Return a dict with theoretical and actual NEFV circuit counts.

        theoretical_2B_minus_1:
          Applies to pure commuting-generator circuits (no CNOT in W).
          One LCU circuit per block (2 variants for O₀/O₁), minus 1 for
          last block which needs only 1 circuit.  Total = 2B − 1.

        actual_with_cnot:
          For EfficientSU2 / CNOT-entangled circuits, W̃ is qubit-specific
          so we build one circuit per parameter (but each is correct).
          Total = 1 (last block) + Σ_{b<B} M_b.
        """
        B = len(self.layers)
        theoretical = 2 * B - 1

        actual = 1  # last block: 1 commutator circuit
        for l_idx, layer in enumerate(self.layers[:-1]):
            actual += len(layer['params'])

        return {
            'num_blocks': B,
            'theoretical_2B_minus_1': theoretical,
            'actual_with_cnot': actual,
            'param_shift_cost': 2 * self.num_params,
            'speedup_vs_param_shift': self.num_params / max(actual, 1),
        }

    def describe(self):
        """Print a human-readable summary of the detected blocks."""
        print(f"CommutingBlockGradient: {len(self.layers)} blocks, "
              f"{self.num_params} parameters")
        cost = self.get_nefv_cost()
        for i, layer in enumerate(self.layers):
            print(f"  Block {i:2d} | type={layer['type']} | "
                  f"params={len(layer['params'])} | "
                  f"qubits={layer['gen_qubits']}")
        print(f"  NEFV (theoretical 2B-1)  : {cost['theoretical_2B_minus_1']}")
        print(f"  NEFV (actual, CNOT-aware): {cost['actual_with_cnot']}")
        print(f"  NEFV (param-shift ref)   : {cost['param_shift_cost']}")
        print(f"  Speedup vs param-shift   : "
              f"{cost['speedup_vs_param_shift']:.1f}×")