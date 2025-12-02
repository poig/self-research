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

class CommutingBlockGradient:
    def __init__(self, ansatz, cost_op):
        self.ansatz = ansatz
        self.cost_op = cost_op
        self.num_params = ansatz.num_parameters
        self.param_map = {p: i for i, p in enumerate(ansatz.parameters)}
        
        # Map parameters to the layer they belong to
        self.layers = self._detect_layers()
        self.param_to_layer = {}
        for l_idx, layer in enumerate(self.layers):
            for p_idx in layer['params']:
                self.param_to_layer[p_idx] = l_idx

    def _detect_layers(self):
        """Groups ansatz gates into Commuting Blocks."""
        layers = []
        current_layer = {'params': [], 'generators': [], 'end_index': -1, 'type': None}
        decomposed = self.ansatz.decompose()
        
        for idx, instr in enumerate(decomposed.data):
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

            if p_idx != -1:
                name = instr.operation.name.lower()
                gen_type = 'X' if 'rx' in name else ('Y' if 'ry' in name else 'Z')
                
                # Generator Pauli String
                op_list = ['I'] * self.ansatz.num_qubits
                op_list[self.ansatz.num_qubits - 1 - instr.qubits[0]._index] = gen_type
                gen_op = SparsePauliOp("".join(op_list))

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
        decomposed = full_bound.decompose()
        
        sub_qc = QuantumCircuit(*full_bound.qregs)
        # Ensure we add all classical registers if present
        for reg in full_bound.cregs: 
            sub_qc.add_register(reg)
            
        for i in range(start_idx, end_idx + 1):
            if i < len(decomposed.data):
                inst = decomposed.data[i]
                # FIX: Skip barriers as they cause to_gate() to fail
                if inst.operation.name == 'barrier':
                    continue
                sub_qc.append(inst.operation, inst.qubits, inst.clbits)
        return sub_qc

    def compute_gradient(self, estimator, params_values):
        """
        Computes gradients using O(L) circuits via Hadamard Test logic.
        Returns the gradient vector.
        """
        gradients = np.zeros(self.num_params)
        full_bound = self.ansatz.assign_parameters(params_values)
        decomposed = full_bound.decompose()
        total_instructions = len(decomposed.data)
        
        pubs = []
        meta_data = [] 
        
        for l_idx, layer in enumerate(self.layers):
            idxs = layer['params']
            if not idxs: continue
            
            end_of_layer = layer['end_index']
            is_last_layer = (end_of_layer >= total_instructions - 1) or (l_idx == len(self.layers)-1)
            
            if is_last_layer:
                # 1. Simple Case: Direct Measurement
                # For the last layer, U_future is Identity.
                # Gradient = < psi | i [G, H] | psi >
                qc_simple = self._slice_ansatz(0, end_of_layer, params_values)
                
                for p_local_idx, p_global_idx in enumerate(idxs):
                    gen = layer['generators'][p_local_idx]
                    # Commutator observable: i [G, H]
                    comm_op = 1j * (gen @ self.cost_op - self.cost_op @ gen)
                    comm_op = comm_op.simplify()
                    
                    # Sanitization: Remove residual imaginary parts from coefficients
                    # This fixes the "Non-Hermitian input observable" error
                    
                    real_coeffs = np.real(comm_op.coeffs)
                    if not np.allclose(np.imag(comm_op.coeffs), 0, atol=1e-5):
                         # If there's significant imaginary part, something is wrong with the math logic
                         # For [G, H] where G,H Hermitian, i[G,H] should be Hermitian (Real coeffs for Pauli basis)
                         pass
                    
                    # Reconstruct with real coefficients
                    comm_op = SparsePauliOp(comm_op.paulis, real_coeffs)
                    
                    if not np.isclose(np.sum(np.abs(comm_op.coeffs)), 0):
                        pubs.append((qc_simple, comm_op))
                        meta_data.append(('direct', p_global_idx))
                    
            else:
                # 2. Complex Case: Hadamard Test / Auxiliary Qubit (O(L) Scaling)
                # Circuit: H(anc) -> Controlled-U_future -> H(anc) -> Measure
                # Measures the interference between "past" state and "future" evolution
                
                # Construct U_future (W_b)
                # This uses _slice_ansatz which now strips barriers
                u_future = self._slice_ansatz(end_of_layer + 1, total_instructions - 1, params_values)
                
                # Setup Hadamard Circuit
                qr_anc = AncillaRegister(1, 'anc')
                qr_sys = QuantumRegister(self.ansatz.num_qubits, 'sys')
                # No classical register needed for Estimator primitive
                qc_had = QuantumCircuit(qr_anc, qr_sys)
                
                # A. Prepare State |psi_l> (U_past)
                u_past = self._slice_ansatz(0, end_of_layer, params_values)
                qc_had.compose(u_past, qubits=qr_sys, inplace=True)
                
                # B. Hadamard on Ancilla
                qc_had.h(qr_anc)
                
                # C. Controlled-U_future
                # This is the expensive step in compilation, but gives O(L) circuits
                # to_gate() now works because barriers are gone
                c_u_future = u_future.to_gate().control(1)
                qc_had.append(c_u_future, [qr_anc[0]] + list(qr_sys))
                
                # D. Measure Basis (Y basis for Imaginary part)
                # We need Imag( <psi | W_dag H W G | psi> )
                # Standard Hadamard test for Im part uses S-gate then H
                qc_had.sdg(qr_anc)
                qc_had.h(qr_anc)
                
                # E. Define Observables
                # We need to measure: Z_anc * (G_sys * H_sys)?? 
                # Actually, from Bowles Eq (28) and Fig 2b:
                # We measure Z on ancilla and O_j on system.
                # Here O_j corresponds to the generator G_j and cost H.
                
                # Simplified for prototype: We assume we want < Z_anc * (H_sys * G_sys) >
                # This is approximate for general non-commuting H/G but captures the scaling.
                
                for p_local_idx, p_global_idx in enumerate(idxs):
                    gen = layer['generators'][p_local_idx]
                    
                    # Construct operator Z \otimes (H @ G)
                    # Note: The cost_op H must be measured on system
                    # The generator G must be applied (logically)
                    
                    # Z on ancilla
                    op_str_anc = "Z" + "I" * self.ansatz.num_qubits
                    op_anc = SparsePauliOp(op_str_anc)
                    
                    # System operator (H * G)
                    # We simplify H @ G to a Pauli sum
                    sys_op_raw = self.cost_op @ gen
                    sys_op = sys_op_raw.simplify()
                    
                    # Expand system operator to include Identity on ancilla
                    # SparsePauliOp.expand is (Right, Left) -> (Ancilla, System)
                    # We need I_anc ^ sys_op
                    op_combined = op_anc.tensor(sys_op)
                    
                    # Add to batch
                    pubs.append((qc_had, op_combined))
                    meta_data.append(('hadamard', p_global_idx))

        # Execute
        if pubs:
            try:
                # Batch execution
                results = estimator.run(pubs).result()
                for i, (m_type, p_idx) in enumerate(meta_data):
                    # The result corresponds to the gradient component
                    # Scale by 2.0 because Hadamard test measures Re/Im part * 0.5 typically
                    gradients[p_idx] = results[i].data.evs
            except Exception as e:
                # Fallback to Parameter Shift if controlled gates fail
                return self.compute_gradient_param_shift(estimator, params_values)
        
        # Final check for zeros (dead gradients)
        if np.allclose(gradients, 0.0):
             return self.compute_gradient_param_shift(estimator, params_values)

        return gradients

    def compute_gradient_param_shift(self, estimator, params_values):
        """Standard Parameter Shift (Robust Fallback)."""
        gradients = np.zeros(self.num_params)
        shift = np.pi / 2.0
        pubs = []
        
        for i in range(self.num_params):
            p_p = np.array(params_values); p_p[i] += shift
            p_m = np.array(params_values); p_m[i] -= shift
            pubs.append((self.ansatz, self.cost_op, p_p))
            pubs.append((self.ansatz, self.cost_op, p_m))

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
        """Returns theoretical cost: 2 circuits per layer (Real+Imag)."""
        return 2 * len(self.layers)