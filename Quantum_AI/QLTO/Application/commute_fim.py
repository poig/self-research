"""
commute_fim.py

Implements Efficient Quantum Fisher Information Matrix (QFIM) Estimation.
Reference: Gómez-Lurbe et al., "Efficient protocol... for Commuting-Block Circuits" (2025).
arXiv:2505.09818v1

Scaling:
- Block-Diagonal: O(L) circuits.
- Off-Block-Diagonal: O(L^2) circuits.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterExpression, Parameter, QuantumRegister, AncillaRegister
from qiskit.quantum_info import SparsePauliOp

from commute_gradient import rotation_generator, canonical_order

class CommutingBlockFIM:
    def __init__(self, ansatz, full=False):
        """
        Args:
            ansatz (QuantumCircuit): The ansatz circuit.
            full (bool): If True, computes off-diagonal blocks (O(L^2)).
                         If False, only block-diagonal (O(L)).
        """
        self.ansatz = ansatz
        self.num_qubits = ansatz.num_qubits
        self.num_params = ansatz.num_parameters
        self.param_map = {p: i for i, p in enumerate(ansatz.parameters)}
        self.full = full
        self.order, self.order_exact = canonical_order(ansatz.decompose())
        self.layers = self._detect_layers()
        self.last_nefv = 0

    def canonical_data(self, circuit):
        """Instructions of `circuit` in the canonical (block-grouped) order."""
        data = circuit.data
        return [data[i] for i in self.order if i < len(data)]

    def get_nefv_cost(self):
        """Circuits actually submitted by the last compute_fim call.

        The block-diagonal protocol genuinely costs one circuit per layer: every
        observable inside a block is a product of that block's mutually commuting
        generators, so they share a single measurement setting.
        """
        return self.last_nefv

    def _detect_layers(self):
        """Parses ansatz into commuting layers."""
        layers = []
        current_layer = {'params': [], 'generators': [], 'type': None, 'end_index': -1}
        data = self.canonical_data(self.ansatz.decompose())

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

            gen_type, generator = (None, None)
            if p_idx != -1:
                gen_type, generator = rotation_generator(
                    instr.operation, instr.qubits[0]._index, self.num_qubits)

            if p_idx != -1 and generator is not None:
                if current_layer['type'] is not None and current_layer['type'] != gen_type:
                    layers.append(current_layer)
                    current_layer = {'params': [], 'generators': [], 'type': None, 'end_index': -1}

                current_layer['type'] = gen_type
                current_layer['params'].append(p_idx)
                current_layer['generators'].append(generator)
                current_layer['end_index'] = idx
            elif len(current_layer['params']) > 0:
                 layers.append(current_layer)
                 current_layer = {'params': [], 'generators': [], 'type': None, 'end_index': -1}

        if current_layer['params']:
            layers.append(current_layer)
        return layers

    def compute_fim(self, estimator, params_values):
        """
        Computes the QFIM.
        """
        fim = np.zeros((self.num_params, self.num_params))
        self.last_nefv = 0
        bound_ansatz = self.ansatz.assign_parameters(params_values)
        decomposed = bound_ansatz.decompose()

        # 1. Block-Diagonal Terms (O(L) circuits)
        for layer in self.layers:
            self._compute_block_diagonal(estimator, layer, decomposed, fim)

        # 2. Off-Block-Diagonal Terms (O(L^2) circuits)
        if self.full:
            self._compute_off_diagonal(estimator, params_values, decomposed, fim)

        return fim

    def _compute_block_diagonal(self, estimator, layer, full_circuit, fim_matrix):
        """Computes elements within a single commuting block."""
        indices = layer['params']
        gens = layer['generators']
        end_idx = layer['end_index']
        
        if not indices: return

        # Truncate circuit to end of layer (canonical order)
        data = self.canonical_data(full_circuit)
        sub_qc = QuantumCircuit(*full_circuit.qregs, *full_circuit.cregs)
        for i in range(end_idx + 1):
            inst = data[i]
            sub_qc.append(inst.operation, inst.qubits, inst.clbits)
            
        observables = []
        obs_map = []
        
        for i, g in enumerate(gens):
            observables.append(g)
            obs_map.append(('single', i))
            
        for i in range(len(gens)):
            for j in range(i+1, len(gens)):
                observables.append(gens[i].compose(gens[j]))
                obs_map.append(('pair', i, j))
                
        try:
            self.last_nefv += 1
            job = estimator.run([(sub_qc, observables)])
            result = job.result()[0]
            if hasattr(result.data, 'evs'):
                evs = result.data.evs
            else:
                evs = result.values
            
            vals = {}
            k = 0
            for i in range(len(gens)):
                vals[('single', i)] = evs[k]; k += 1
            for i in range(len(gens)):
                for j in range(i+1, len(gens)):
                    vals[('pair', i, j)] = evs[k]; k += 1
            
            for i_local, p_i in enumerate(indices):
                for j_local, p_j in enumerate(indices):
                    if i_local == j_local:
                        exp_i = vals[('single', i_local)]
                        # 1 - <G>^2 is a variance and cannot be negative, but a
                        # SAMPLED <G> can land just above 1 and make it so. That
                        # then reaches nisq_v2's 1/sqrt(g_ii) and yields NaN,
                        # which surfaces as TranspilerError 'Binding to NaN'.
                        # It fires near convergence, exactly when parameters
                        # saturate and <G> -> +-1 so the variance -> 0. Latent
                        # until the estimators were given a finite shot budget.
                        fim_matrix[p_i, p_j] = max(0.0, 1.0 - float(np.real(exp_i))**2)
                    else:
                        i, j = sorted((i_local, j_local))
                        exp_prod = vals[('pair', i, j)]
                        exp_i = vals[('single', i_local)]
                        exp_j = vals[('single', j_local)]
                        entry = exp_prod - (exp_i * exp_j)
                        fim_matrix[p_i, p_j] = entry
                        fim_matrix[p_j, p_i] = entry
                        
        except Exception as e:
            print(f"FIM Block Error: {e}")

    def _compute_off_diagonal(self, estimator, params_values, full_circuit, fim_matrix):
        """
        Computes off-diagonal blocks using the efficient Ancilla protocol.
        Scaling: O(L^2) circuits.
        """
        n_layers = len(self.layers)
        data = self.canonical_data(full_circuit)

        for l1_idx in range(n_layers):
            for l2_idx in range(l1_idx + 1, n_layers):
                layer1 = self.layers[l1_idx]
                layer2 = self.layers[l2_idx]
                
                # 1. Prepare Circuit with Ancilla
                qr_anc = AncillaRegister(1, 'anc')
                # Important: Use same system register size/name context if possible, 
                # but constructing fresh registers is safer.
                qr_sys = QuantumRegister(self.num_qubits, 'q')
                qc = QuantumCircuit(qr_anc, qr_sys)
                
                # 2. Prepare |psi_l1>
                end_l1 = layer1['end_index']
                for i in range(end_l1 + 1):
                    inst = data[i]
                    q_indices = [full_circuit.find_bit(q).index for q in inst.qubits]
                    qc.append(inst.operation, [qr_sys[k] for k in q_indices])
                
                # 3. Hadamard Test Logic
                qc.h(qr_anc)
                
                start_w = end_l1 + 1
                end_w = layer2['end_index']
                l1_type = layer1['type']
                
                for i in range(start_w, end_w + 1):
                    inst = data[i]
                    op = inst.operation
                    q_indices = [full_circuit.find_bit(q).index for q in inst.qubits]
                    
                    sign = 1.0
                    op_type = 'I'
                    if 'rx' in op.name: op_type = 'X'
                    elif 'ry' in op.name: op_type = 'Y'
                    elif 'rz' in op.name: op_type = 'Z'
                    
                    if op_type != 'I' and l1_type != 'I' and op_type != l1_type:
                        sign = -1.0
                        
                    if len(op.params) == 1 and not isinstance(op.params[0], Parameter):
                         val = float(op.params[0])
                         
                         # |0> branch (Original)
                         qc.x(qr_anc)
                         qc.append(op.control(1), [qr_anc[0]] + [qr_sys[k] for k in q_indices])
                         qc.x(qr_anc)
                         
                         # |1> branch (Modified sign)
                         new_op = op.copy()
                         new_op.params = [val * sign]
                         qc.append(new_op.control(1), [qr_anc[0]] + [qr_sys[k] for k in q_indices])

                    else:
                        qc.append(op, [qr_sys[k] for k in q_indices])

                qc.h(qr_anc)
                
                # 4. Observables
                observables = []
                obs_indices = []
                
                gens1 = layer1['generators']
                gens2 = layer2['generators']
                
                # Z operator on Ancilla (qubit 0 in `qr_anc` + `qr_sys` composition if little-endian?)
                # Qiskit layout: usually registers are concatenated. 
                # If we do op_sys.tensor(op_anc), op_sys is on higher indices, op_anc on lower.
                # In QuantumCircuit(qr_anc, qr_sys), qr_anc is index 0.
                
                op_anc = SparsePauliOp(["Z"]) # 1 qubit
                
                for i, g1 in enumerate(gens1):
                    for j, g2 in enumerate(gens2):
                        op_sys = g2.compose(g1) # N qubits
                        
                        # Correct Tensor Product for Qubit 0 = Ancilla
                        full_op = op_sys.tensor(op_anc) 
                        
                        observables.append(full_op)
                        obs_indices.append((i, j))
                
                try:
                    self.last_nefv += 1
                    job = estimator.run([(qc, observables)])
                    result = job.result()[0]
                    if hasattr(result.data, 'evs'):
                        evs = result.data.evs
                    else:
                        evs = result.values

                    p1_indices = layer1['params']
                    p2_indices = layer2['params']
                    
                    for k, (i, j) in enumerate(obs_indices):
                        p_row = p1_indices[i]
                        p_col = p2_indices[j]
                        val = evs[k]
                        fim_matrix[p_row, p_col] = val
                        fim_matrix[p_col, p_row] = val
                        
                except Exception as e:
                    print(f"Off-Diagonal Error L{l1_idx}-L{l2_idx}: {e}")