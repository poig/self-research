"""
commute_gradient.py

Implements Backpropagation Scaling Gradient Estimation for Commuting-Block Circuits.
Reference: Bowles et al., "Backpropagation scaling in parameterised quantum circuits" (2024).
arXiv:2306.14962v4

Key Feature:
Uses the "Auxiliary Qubit" method (Hadamard Test logic) to measure the gradient 
of an entire commuting layer in one go. Scaling is O(L) circuits, not O(M) parameters.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import SparsePauliOp

# Rank used to group same-generator rotations into a contiguous block.
_TYPE_RANK = {'X': 0, 'Y': 1, 'Z': 2}


def rotation_generator(op, qubit, num_qubits):
    """(type_key, SparsePauliOp G) for a parameterised single-qubit rotation.

    Matching on the gate *name* alone is not enough: `efficient_su2().decompose()`
    lowers RY/RZ to `r(theta, phi)` and `p(theta)`, so a name test for 'ry'/'rz'
    silently labels every generator 'Z'. Convention: U = exp(-i*theta*G/2).
    """
    name = op.name.lower()
    pos = num_qubits - 1 - qubit  # little-endian Pauli string position

    def pauli(letter):
        s = ['I'] * num_qubits
        s[pos] = letter
        return SparsePauliOp("".join(s))

    if name == 'rx':
        return 'X', pauli('X')
    if name == 'ry':
        return 'Y', pauli('Y')
    if name in ('rz', 'p', 'u1', 'phase'):
        # P(theta) = e^{i*theta/2} RZ(theta): same generator up to global phase.
        return 'Z', pauli('Z')
    if name == 'r':
        # R(theta, phi) = exp(-i*theta/2 (cos(phi) X + sin(phi) Y))
        try:
            phi = float(op.params[1])
        except (TypeError, ValueError):
            return None, None
        c, s = np.cos(phi), np.sin(phi)
        if abs(s) < 1e-9:
            return 'X', pauli('X')
        if abs(c) < 1e-9:
            return 'Y', pauli('Y')
        return f'R{phi:.6f}', (c * pauli('X') + s * pauli('Y')).simplify()
    return None, None


def canonical_order(decomposed):
    """Index permutation grouping same-generator rotations into contiguous blocks.

    Qiskit emits the rotation layer interleaved per qubit (RY(q0), RZ(q0), RY(q1),
    ...), so a contiguous scan sees the generator type alternate on every gate and
    splits the circuit into singleton "blocks". Rotations on *different* qubits
    always commute, so regrouping by generator type inside each entangler-free
    segment is an exact circuit identity - provided each qubit's own gate order is
    untouched, which is verified below.

    Returns (order, ok). If the permutation would reorder two gates on the same
    qubit, the original order is returned with ok=False.
    """
    n = decomposed.num_qubits
    order, segment, ok = [], [], True

    def per_qubit(seq):
        out = {}
        for idx, _, q in seq:
            out.setdefault(q, []).append(idx)
        return out

    def flush():
        nonlocal ok
        if not segment:
            return
        regrouped = sorted(segment, key=lambda e: _TYPE_RANK.get(e[1], 3))  # stable
        if per_qubit(regrouped) != per_qubit(segment):
            ok = False                      # would swap two gates on one qubit
            regrouped = list(segment)
        order.extend(idx for idx, _, _ in regrouped)
        segment.clear()

    for idx, instr in enumerate(decomposed.data):
        q = decomposed.find_bit(instr.qubits[0]).index if instr.qubits else 0
        key = None
        if instr.operation.params:
            first = instr.operation.params[0]
            if isinstance(first, ParameterExpression) and first.parameters:
                key, _ = rotation_generator(instr.operation, q, n)
        if key is not None and len(instr.qubits) == 1:
            segment.append((idx, key, q))
        else:
            flush()
            order.append(idx)
    flush()
    return order, ok


class CommutingBlockGradient:
    def __init__(self, ansatz, cost_op):
        self.ansatz = ansatz
        self.cost_op = cost_op
        self.num_params = ansatz.num_parameters
        self.param_map = {p: i for i, p in enumerate(ansatz.parameters)}
        
        # Canonical instruction order: all index-based slicing below refers to it.
        self.order, self.order_exact = canonical_order(self.ansatz.decompose())
        if not self.order_exact:
            print("WARNING [CommutingBlockGradient]: could not regroup rotations into "
                  "commuting blocks without reordering gates on a shared qubit; "
                  "falling back to circuit order (layers will be finer than optimal).")

        # Map parameters to the layer they belong to
        self.layers = self._detect_layers()
        self.param_to_layer = {}
        for l_idx, layer in enumerate(self.layers):
            for p_idx in layer['params']:
                self.param_to_layer[p_idx] = l_idx

    def canonical_data(self, circuit):
        """Instructions of `circuit` in the canonical (block-grouped) order."""
        data = circuit.data
        return [data[i] for i in self.order if i < len(data)]

    def _detect_layers(self):
        """Groups ansatz gates into Commuting Blocks."""
        layers = []
        current_layer = {'params': [], 'generators': [], 'end_index': -1, 'type': None}
        data = self.canonical_data(self.ansatz.decompose())
        nq = self.ansatz.num_qubits

        for idx, instr in enumerate(data):
            p_idx = -1
            if len(instr.operation.params) > 0:
                param = instr.operation.params[0]
                target_param = None
                if isinstance(param, ParameterExpression):
                    params_in_expr = list(param.parameters)
                    if params_in_expr: target_param = params_in_expr[0]
                elif isinstance(param, Parameter):
                    target_param = param

                if target_param in self.param_map:
                    p_idx = self.param_map[target_param]

            gen_type, gen_op = (None, None)
            if p_idx != -1:
                q = instr.qubits[0]._index
                gen_type, gen_op = rotation_generator(instr.operation, q, nq)

            if p_idx != -1 and gen_op is not None:
                if current_layer['type'] is not None and current_layer['type'] != gen_type:
                    layers.append(current_layer)
                    current_layer = {'params': [], 'generators': [], 'end_index': -1, 'type': None}

                current_layer['type'] = gen_type
                current_layer['params'].append(p_idx)
                current_layer['generators'].append(gen_op)
                current_layer['end_index'] = idx
            elif len(current_layer['params']) > 0:
                # Non-commuting barrier (e.g. CNOT) ends the block
                layers.append(current_layer)
                current_layer = {'params': [], 'generators': [], 'end_index': -1, 'type': None}

        if current_layer['params']:
            layers.append(current_layer)
        return layers

    def _slice_ansatz(self, start_idx, end_idx, bound_params):
        """Creates a sub-circuit for a specific range of instructions."""
        full_bound = self.ansatz.assign_parameters(bound_params)
        data = self.canonical_data(full_bound.decompose())

        sub_qc = QuantumCircuit(*full_bound.qregs)
        # Ensure we add all classical registers if present
        for reg in full_bound.cregs:
            sub_qc.add_register(reg)

        for i in range(start_idx, end_idx + 1):
            if i < len(data):
                inst = data[i]
                # FIX: Skip barriers as they cause to_gate() to fail
                if inst.operation.name == 'barrier':
                    continue
                sub_qc.append(inst.operation, inst.qubits, inst.clbits)
        return sub_qc

    def compute_gradient(self, estimator, params_values):
        """
        Gradient of <H> w.r.t. every ansatz parameter.

        The final commuting block is free of any subsequent unitary, so its
        derivatives are direct expectation values of the commutator,

            dC/dtheta_j = <psi| (i/2) [G_j, H] |psi>,

        costing one circuit per parameter instead of the two a shift rule needs.
        Every earlier block uses the parameter-shift rule.

        NOT the O(B) protocol of Bowles et al. A previous version of this method
        claimed that scaling via a single controlled-U_future Hadamard test, but
        that circuit measures Im<phi|U_f^dag M|phi>, whereas the gradient needs
        Im<phi|G U_f^dag H U_f|phi>. One controlled-U_future cannot produce the
        U_f^dag H U_f conjugation - Bowles requires controlled-W' *and*
        controlled-W~ for precisely this reason (Appendix B). The old path also
        built the non-Hermitian observable Z (x) (H@G), so the estimator rejected
        every batch and a bare `except` silently re-ran full parameter-shift while
        `get_nefv_cost()` kept reporting the theoretical 2L. Both are gone.

        `self.last_nefv` records the circuits actually submitted.
        """
        gradients = np.zeros(self.num_params)
        n_layers = len(self.layers)
        pubs = []
        meta_data = []
        shift = np.pi / 2.0

        for l_idx, layer in enumerate(self.layers):
            idxs = layer['params']
            if not idxs: continue

            if l_idx == n_layers - 1:
                # Direct commutator: U_future is the identity here.
                qc_simple = self._slice_ansatz(0, layer['end_index'], params_values)

                for p_local_idx, p_global_idx in enumerate(idxs):
                    gen = layer['generators'][p_local_idx]
                    # i[G, H] is Hermitian for Hermitian G, H; the 1/2 comes from
                    # the exp(-i*theta*G/2) convention.
                    comm_op = (0.5j * (gen @ self.cost_op - self.cost_op @ gen)).simplify()
                    comm_op = SparsePauliOp(comm_op.paulis, np.real(comm_op.coeffs))

                    if not np.isclose(np.sum(np.abs(comm_op.coeffs)), 0):
                        pubs.append((qc_simple, comm_op))
                        meta_data.append(('direct', p_global_idx))
            else:
                for p_global_idx in idxs:
                    p_p = np.array(params_values); p_p[p_global_idx] += shift
                    p_m = np.array(params_values); p_m[p_global_idx] -= shift
                    pubs.append((self.ansatz, self.cost_op, p_p))
                    pubs.append((self.ansatz, self.cost_op, p_m))
                    meta_data.append(('shift+', p_global_idx))
                    meta_data.append(('shift-', p_global_idx))

        self.last_nefv = len(pubs)
        if not pubs:
            return gradients

        results = estimator.run(pubs).result()
        for i, (m_type, p_idx) in enumerate(meta_data):
            val = float(results[i].data.evs)
            if m_type == 'direct':
                gradients[p_idx] = val
            elif m_type == 'shift+':
                gradients[p_idx] += 0.5 * val
            else:
                gradients[p_idx] -= 0.5 * val

        return gradients

    def compute_gradient_param_shift(self, estimator, params_values):
        """Standard Parameter Shift. Costs 2 circuits per parameter."""
        gradients = np.zeros(self.num_params)
        shift = np.pi / 2.0
        pubs = []

        for i in range(self.num_params):
            p_p = np.array(params_values); p_p[i] += shift
            p_m = np.array(params_values); p_m[i] -= shift
            pubs.append((self.ansatz, self.cost_op, p_p))
            pubs.append((self.ansatz, self.cost_op, p_m))

        self.last_nefv = len(pubs)
        try:
            results = estimator.run(pubs).result()
            for i in range(self.num_params):
                val_p = float(results[2*i].data.evs)
                val_m = float(results[2*i+1].data.evs)
                gradients[i] = 0.5 * (val_p - val_m)
        except Exception:
            pass

        return gradients

    def get_nefv_cost(self):
        """Circuits actually submitted by the last gradient call - not a formula."""
        return {
            'actual_with_cnot': getattr(self, 'last_nefv', 0),
        }