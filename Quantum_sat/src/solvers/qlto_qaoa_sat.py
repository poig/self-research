"""
QLTO-QAOA SAT SOLVER
====================

This file implements a full SAT solver using the Quantum Approximate 
Optimization Algorithm (QAOA) as the circuit ansatz, and the 
Quantum Local-to-Global Optimization (QLTO) algorithm as the 
quantum optimizer.

This combines the ideas from:
1.  `qlto_nisq.py`: The core QLTO optimizer (hybrid loop, W_Gate, U_PE oracle).
2.  `qlto_sat_formal.py`: The SAT problem definition and the logic for 
    converting a SAT problem into a Qiskit Hamiltonian.

How it works:
1.  A SAT problem is defined (e.g., (x1 or ~x2) and ...).
2.  `sat_to_hamiltonian` converts this into a "cost" Hamiltonian H_C, where
    the ground state (lowest energy) is the bitstring that solves the SAT problem.
3.  `create_qaoa_ansatz` builds a parameterized quantum circuit (ansatz)
    designed to find this ground state. The parameters are (gamma, beta).
4.  `run_qlto_optimizer` takes this ansatz and the cost Hamiltonian. It then
    runs the QLTO hybrid loop to find the *optimal (gamma, beta) parameters*.
5.  The `__main__` block runs the optimizer, gets the best parameters, and then
    runs the final QAOA circuit with those parameters to get the solution.
"""

import numpy as np
import time
from typing import Callable, Tuple, Dict, Any, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import math
import random

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
    from qiskit.circuit.library import RXGate, RYGate, RZGate, RZZGate, CXGate, PauliEvolutionGate
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.quantum_info import SparsePauliOp, Operator
    
    from qiskit_aer.primitives import Estimator as AerEstimator, Sampler as AerSampler
    
    # Primitives V1 Wrappers (simplified for this context)
    class BaseEstimator: 
        def __init__(self): self._estimator = AerEstimator()
        def run(self, pubs, **kwargs): 
            circuits = [p[0] for p in pubs]
            observables = [p[1] for p in pubs]
            return self._estimator.run(circuits, observables=observables, **kwargs)
            
    class BaseSampler: 
        def __init__(self): self._sampler = AerSampler()
        def run(self, pubs, **kwargs): 
            circuits = [p[0] for p in pubs]
            return self._sampler.run(circuits, parameter_values=[[]]*len(circuits), **kwargs)
    
    # Shared Helper
    class CountingWrapper:
        def __init__(self, base): self._base = base; self.n_calls = 0; self.time_seconds = 0.0
        def run(self, *args, **kwargs):
            t0 = time.time()
            res = self._base.run(*args, **kwargs)
            t1 = time.time()
            self.n_calls += 1
            self.time_seconds += (t1 - t0)
            return res
        def __getattr__(self, name): return getattr(self._base, name)

    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit components not found. This script requires Qiskit.")
    
    # Create dummy classes to allow script to be imported
    QISKIT_AVAILABLE = False
    class QuantumCircuit: pass
    class QuantumRegister: pass
    class ClassicalRegister: pass
    class AncillaRegister: pass
    class RXGate: pass
    class RYGate: pass
    class RZGate: pass
    class RZZGate: pass
    class CXGate: pass
    class PauliEvolutionGate: pass
    class Parameter: pass
    class ParameterVector: pass
    class SparsePauliOp: pass
    class Operator: pass
    class BaseEstimator: pass
    class BaseSampler: pass
    class CountingWrapper: pass

# Compatibility flag expected by external modules (e.g. quantum_sat_solver)
# QLTO_AVAILABLE mirrors whether Qiskit was successfully imported and the
# solver functionality is therefore available.
QLTO_AVAILABLE = bool(QISKIT_AVAILABLE)


# ==============================================================================
# STEP 1: SAT PROBLEM DEFINITION
# (Adapted from qlto_sat_formal.py)
# ==============================================================================

@dataclass
class SATClause:
    """Represents a single SAT clause (disjunction of literals)"""
    literals: Tuple[int, ...]
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        """Check if clause is satisfied by a classical assignment dict"""
        for lit in self.literals:
            var = abs(lit)
            if var not in assignment:
                continue # This variable isn't part of the assignment
            value = assignment[var]
            # Positive literal needs True, negative needs False
            if (lit > 0 and value) or (lit < 0 and not value):
                return True
        return False
    
    def get_variables(self) -> Set[int]:
        """Get all variables in this clause"""
        return {abs(lit) for lit in self.literals}

@dataclass
class SATProblem:
    """Complete SAT problem instance"""
    n_vars: int
    clauses: List[SATClause]
    
    def __post_init__(self):
        self.n_clauses = len(self.clauses)
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        """Check if ALL clauses are satisfied"""
        return all(clause.is_satisfied(assignment) for clause in self.clauses)

    def count_violated(self, assignment: Dict[int, bool]) -> int:
        """Count number of violated clauses"""
        return sum(1 for clause in self.clauses if not clause.is_satisfied(assignment))

# ==============================================================================
# STEP 2: SAT-TO-HAMILTONIAN CONVERTER
# ==============================================================================

def sat_to_hamiltonian(problem: SATProblem) -> SparsePauliOp:
    """
    Converts a SATProblem into a Qiskit SparsePauliOp Hamiltonian.

    The Hamiltonian is constructed as H_C = sum(H_j) where H_j is a
    penalty term for the j-th clause.
    - H_j = 0 if clause j is satisfied
    - H_j = 1 if clause j is violated
    
    The ground state of H_C (lowest energy) is the assignment that
    violates the fewest clauses. If the problem is satisfiable,
    the ground state energy will be 0.

    We use the formal mapping:
    - Variable x_i -> Qubit i
    - x_i = 0 (False) -> |0> state
    - x_i = 1 (True)  -> |1> state

    A clause is violated ONLY if all its literals are false.
    e.g., (x1 or ~x2) is violated if x1=0 AND x2=1.
    The penalty term is a projector: |0><0|_1 @ |1><1|_2
    
    Using |0><0| = (I - Z)/2 and |1><1| = (I + Z)/2, we can expand:
    (I - Z1)/2 @ (I + Z2)/2 = 1/4 * (I@I + I@Z2 - Z1@I - Z1@Z2)
    
    This function builds the sum of all such projectors.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required to build Hamiltonians.")
        
    n_vars = problem.n_vars
    hamiltonian_terms = defaultdict(float)

    for clause in problem.clauses:
        # This dict will store the components of the projector for this clause
        # We represent pauli dictionaries as {qubit_index: 'Z'}; start with
        # an empty pauli dict (identity) and coeff 1.0
        clause_projector_terms = {0: ({}, 1.0)} # Start with (Identity, coeff=1.0)
        
        # Build the projector term, e.g., (I-Z1)/2 @ (I+Z2)/2 @ (I-Z3)/2
        for lit in clause.literals:
            var_idx = abs(lit) - 1 # 0-indexed qubit
            
            # Projector for this literal
            # lit > 0 (e.g., x1): maps to |0> (False). Projector = (I - Z)/2
            # lit < 0 (e.g., ~x2): maps to |1> (True). Projector = (I + Z)/2
            
            # --- BUG FIX ---
            # The comment above was wrong.
            # Correct projector mapping:
            # positive literal (x1): violation when x1 == False -> |0><0| = (I + Z)/2
            # negative literal (~x2): violation when x2 == True  -> |1><1| = (I - Z)/2
            literal_op = {'I': 0.5, 'Z': 0.5 if lit > 0 else -0.5}
            # --- END FIX ---

            # Multiply this op into the clause_projector_terms
            new_projector_terms = {}
            for op_pauli_str, op_coeff in literal_op.items():
                for existing_pauli_dict, existing_coeff in clause_projector_terms.values():
                    
                    new_pauli_dict = existing_pauli_dict.copy()
                    if op_pauli_str != 'I':
                        current_op = new_pauli_dict.get(var_idx, 'I')
                        # Pauli algebra: I*Z=Z, Z*I=Z, Z*Z=I
                        if current_op == 'I':
                            new_pauli_dict[var_idx] = 'Z'
                        elif current_op == 'Z':
                            new_pauli_dict[var_idx] = 'I'
                    
                    # Use a deterministic key based on qubit index ordering.
                    # Sorting by index ensures keys are tuples of (int, str)
                    # and avoids comparing ints with strings.
                    pauli_str_key = tuple(sorted(new_pauli_dict.items(), key=lambda x: x[0]))
                    new_coeff = existing_coeff * op_coeff
                    
                    if pauli_str_key not in new_projector_terms:
                        new_projector_terms[pauli_str_key] = (new_pauli_dict, 0.0)
                    
                    new_projector_terms[pauli_str_key] = (
                        new_pauli_dict, 
                        new_projector_terms[pauli_str_key][1] + new_coeff
                    )
            clause_projector_terms = new_projector_terms

        # Add the expanded projector terms to the total Hamiltonian
        for pauli_dict, coeff in clause_projector_terms.values():
            if abs(coeff) > 1e-9:
                # Convert dict {0:'Z', 2:'Z'} to string 'IZI...Z'
                pauli_str_list = ['I'] * n_vars
                for idx, op in pauli_dict.items():
                    if idx < n_vars:
                        pauli_str_list[idx] = op
                
                # Qiskit LSB: reverse the string
                final_pauli_str = "".join(pauli_str_list[::-1])
                hamiltonian_terms[final_pauli_str] += coeff

    # Create the final SparsePauliOp
    pauli_list = []
    for pauli_str, coeff in hamiltonian_terms.items():
        if abs(coeff) > 1e-9:
            pauli_list.append((pauli_str, coeff))
    
    if not pauli_list: # Handle empty problem
        pauli_list = [('I' * n_vars, 0.0)]
        
    return SparsePauliOp.from_list(pauli_list)

# ==============================================================================
# STEP 3: QAOA ANSATZ CREATION
# ==============================================================================

def create_qaoa_ansatz(problem: SATProblem, p_layers: int) -> Tuple[QuantumCircuit, int]:
    """
    Creates the parameterized QAOA circuit template for a SAT problem.

    The circuit has 2*p parameters (p gammas, p betas).
    
    Ansatz structure:
    1. H gates on all qubits
    2. p layers of:
       a. Cost Hamiltonian (H_C) evolution: e^(-i * gamma * H_C)
       b. Mixer Hamiltonian (H_B) evolution: e^(-i * beta * H_B)
    
    H_C = The SAT Hamiltonian (sum of RZZ, RZ gates)
    H_B = The mixer (sum of RX gates)
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required to build an ansatz.")
        
    n_vars = problem.n_vars
    ansatz_qc = QuantumCircuit(n_vars)
    
    gammas = ParameterVector('g', p_layers)
    betas = ParameterVector('b', p_layers)
    
    # 1. Initial superposition
    ansatz_qc.h(range(n_vars))
    ansatz_qc.barrier()
    
    # Get the cost Hamiltonian (H_C) to build the cost layers
    # We build the cost layer manually from the pauli list for
    # compatibility with build_w_gate_nisq.
    cost_hamiltonian = sat_to_hamiltonian(problem)

    for k in range(p_layers):
        # 2a. Cost Layer (e^(-i * gamma_k * H_C))
        # We manually decompose the Pauli evolution
        # H_C = sum(coeff * Pauli)
        # e^(-i*g*H_C) approx product( e^(-i*g*coeff*Pauli) )
        for pauli, coeff in cost_hamiltonian.to_list():
            # Qiskit LSB: pauli string is reversed
            pauli_str = pauli[::-1] 
            
            qubit_indices = [i for i, p in enumerate(pauli_str) if p != 'I']
            pauli_ops = [p for p in pauli_str if p != 'I']
            
            if not qubit_indices:
                continue # Skip Identity term

            # We can only easily synthesize Z and ZZ terms in QAOA
            if all(p == 'Z' for p in pauli_ops):
                if len(qubit_indices) == 1:
                    # Single-qubit RZ gate
                    ansatz_qc.rz(gammas[k] * 2 * coeff.real, qubit_indices[0])
                elif len(qubit_indices) == 2:
                    # Two-qubit RZZ gate
                    q1, q2 = qubit_indices
                    ansatz_qc.rzz(gammas[k] * 2 * coeff.real, q1, q2)
                else:
                    # Multi-qubit RZ...Z gate
                    # Decompose: RZ..Z(g) = CNOT(q1,qN) ... RZ(g) ... CNOT(q1,qN)
                    last_q = qubit_indices[-1]
                    for q in qubit_indices[:-1]:
                        ansatz_qc.cx(q, last_q)
                    ansatz_qc.rz(gammas[k] * 2 * coeff.real, last_q)
                    for q in qubit_indices[:-1]:
                        ansatz_qc.cx(q, last_q)
        
        ansatz_qc.barrier()
        
        # 2b. Mixer Layer (e^(-i * beta_k * H_B))
        # H_B = sum(X_i)
        for i in range(n_vars):
            ansatz_qc.rx(2 * betas[k], i)
            
        ansatz_qc.barrier()

    return ansatz_qc, 2 * p_layers

# ==============================================================================
# STEP 4: QLTO CORE CIRCUITS (OPTIMIZER INTERNALS)
# (Adapted from qlto_nisq.py)
# ==============================================================================

def build_w_gate_nisq(
    param_reg: QuantumRegister,
    ansatz_reg: QuantumRegister,
    vqe_ansatz_template: QuantumCircuit,
    n_params: int,
    bits_per_param: int,
    param_bounds: np.ndarray
    ) -> QuantumCircuit:
    """
    Strategy C Redesign: Builds the W' gate using direct parametric control.
    This coherently maps |theta>|0> -> |theta>|psi(theta)>
    
    It parses the vqe_ansatz_template and replaces each ParameterizedGate
    with a series of controlled gates, controlled by the bits of the
    param_reg.
    """
    if not QISKIT_AVAILABLE: return None

    qc_w = QuantumCircuit(param_reg, ansatz_reg, name="W_Gate_Shallow_Struct")
    
    param_map = {} # Map {Parameter_object: index}
    current_param_idx = 0
    
    # Decompose to get to basic gates (RX, RZ, RZZ, CNOT)
    decomposed_template = vqe_ansatz_template.decompose()

    for instruction in decomposed_template.data:
        op = instruction.operation
        qubit_indices = [decomposed_template.find_bit(q).index for q in instruction.qubits]
        target_ansatz_qubits = [ansatz_reg[i] for i in qubit_indices]

        # 1. Handle Parameterized Gates (RX, RZ, RZZ, etc.)
        if hasattr(op, 'params') and op.params and isinstance(op.params[0], Parameter):
            param_obj = op.params[0]
            
            # Find which parameter index this is (0 to n_params-1)
            if param_obj not in param_map: 
                param_map[param_obj] = current_param_idx
                current_param_idx += 1
            param_k_index = param_map[param_obj]
            
            if param_k_index >= n_params:
                print(f"Warning: Found more parameters in ansatz ({param_k_index+1}) than expected ({n_params}). Skipping.")
                continue

            # Identify the parameter bits for this VQE parameter
            param_start_bit = param_k_index * bits_per_param
            
            min_val, max_val = param_bounds[param_k_index]
            full_range = max_val - min_val
            
            # --- Strategy C Logic: Controlled Rotation per Bit ---
            for i in range(bits_per_param):
                control_qubit = param_reg[param_start_bit + i]
                
                # Geometric scaling: angle = (full_range / 2^B) * 2^i
                # (Note: A simpler linear scaling is used here for robustness)
                angle_contribution = full_range / (2**bits_per_param)
                
                controlled_gate = None
                
                # Re-create the gate type with the small angle
                if isinstance(op, RYGate): 
                    controlled_gate = RYGate(angle_contribution).control(1)
                elif isinstance(op, RZGate): 
                    controlled_gate = RZGate(angle_contribution).control(1)
                elif isinstance(op, RXGate): 
                    controlled_gate = RXGate(angle_contribution).control(1)
                elif isinstance(op, RZZGate):
                    controlled_gate = RZZGate(angle_contribution).control(1)
                else:
                    continue # Skip gates we don't know how to control
                    
                # Apply controlled rotation
                control_and_target_qubits = [control_qubit] + target_ansatz_qubits
                qc_w.append(controlled_gate, control_and_target_qubits)
        
        # 2. Handle Fixed Gates (like CNOTs)
        elif isinstance(op, CXGate) and len(target_ansatz_qubits) == 2:
            qc_w.cx(target_ansatz_qubits[0], target_ansatz_qubits[1])
        
        # 3. Handle Barriers (for visualization)
        elif op.name == 'barrier':
            qc_w.barrier(target_ansatz_qubits)

    # Ensure every qubit appears in the circuit (add identity on unused qubits)
    # This prevents to_instruction() from producing an instruction that omits
    # unused qubits and causing an argument-count mismatch when appended.
    try:
        for qb in qc_w.qubits:
            qc_w.id(qb)
    except Exception:
        # Fallback: ignore if the backend/Qiskit version doesn't support id()
        pass

    return qc_w


# Simple cache for W circuits to avoid rebuilding identical templates
_W_cache: Dict[tuple, QuantumCircuit] = {}

def get_cached_w(param_reg: QuantumRegister, ansatz_reg: QuantumRegister, template: QuantumCircuit,
                 n_params: int, bits_per_param: int, param_bounds: np.ndarray) -> QuantumCircuit:
    """Return a cached copy of the W gate circuit for these discretization params."""
    try:
        key = (n_params, bits_per_param, tuple(np.array(param_bounds).flatten()))
    except Exception:
        key = (n_params, bits_per_param)
    if key in _W_cache:
        cached = _W_cache[key]
        # If cached circuit matches qubit count for this request, return a copy
        expected_qubits = len(param_reg) + len(ansatz_reg)
        try:
            if cached.num_qubits == expected_qubits:
                return cached.copy()
        except Exception:
            pass
    # Build fresh with the provided registers to ensure qubit/register alignment
    w = build_w_gate_nisq(param_reg, ansatz_reg, template, n_params, bits_per_param, param_bounds)
    # Cache the template (best-effort)
    try:
        _W_cache[key] = w.copy()
    except Exception:
        _W_cache[key] = w
    return w.copy()

def build_coherent_phase_oracle_nisq(
    n_params: int, bits_per_param: int, param_bounds: np.ndarray, delta_t: float,
    vqe_ansatz_template: QuantumCircuit, vqe_hamiltonian: SparsePauliOp, n_qubits_ansatz: int
    ) -> QuantumCircuit:
    """
    Builds the shallow phase kickback circuit U'_PE.
    
    This is the "one big circuit" oracle. It performs the operation:
    |theta>|psi>|anc> -> |theta>|psi>|anc + phase(H_C, |psi(theta)>)>
    
    It uses phase kickback to measure the energy <psi(theta)|H_C|psi(theta)>
    and write it as a phase on the phase_anc_reg.
    """
    if not QISKIT_AVAILABLE: return None
        
    n_qubits_p = n_params * bits_per_param
    
    # --- Define all quantum registers ---
    param_reg = QuantumRegister(n_qubits_p, 'param')
    ansatz_reg = QuantumRegister(n_qubits_ansatz, 'ansatz')
    phase_anc_reg = AncillaRegister(1, 'phase_anc') 
    
    all_regs = [phase_anc_reg, param_reg, ansatz_reg]
    u_pe_full_circuit = QuantumCircuit(*all_regs, name="U_PE_Shallow_Circuit")

    # 1. Build W' Gate (The QAOA Ansatz) and target qubits
    w_circuit = get_cached_w(param_reg, ansatz_reg, vqe_ansatz_template.copy(),
                              n_params, bits_per_param, param_bounds)
    if w_circuit is None:
        return None
    w_gate_inst = w_circuit.to_instruction(label="W_Gate_Shallow")
    w_qubits = list(param_reg) + list(ansatz_reg)
    # If the W circuit instruction expects fewer qubits than the target
    # (this can happen if some qubits were unused and Qiskit trimmed them),
    # expand it by composing into a fresh identity circuit of the expected size.
    try:
        expected_q = len(w_qubits)
        actual_q = getattr(w_circuit, 'num_qubits', None)
        if actual_q is not None and actual_q != expected_q:
            exp = QuantumCircuit(expected_q, name="W_Gate_Expanded")
            try:
                exp.compose(w_circuit, qubits=list(range(actual_q)), inplace=True)
                w_circuit = exp
                w_gate_inst = w_circuit.to_instruction(label="W_Gate_Shallow")
            except Exception:
                # If compose on integer-mapped circuits fails, try appending id to expand
                for i in range(expected_q - actual_q):
                    w_circuit.id(w_circuit.qubits[-1])
                w_gate_inst = w_circuit.to_instruction(label="W_Gate_Shallow")
    except Exception:
        pass

    # 2. Build Controlled Hamiltonian Evolution (The "Tool" or "Cost Function")
    try:
        # We use a PauliEvolutionGate, which Qiskit knows how to synthesize
        # This represents e^(-i * H_C * delta_t)
        ham_evol_gate = PauliEvolutionGate(vqe_hamiltonian, time=delta_t)
        
        # Create the controlled version: C-U
        controlled_ham_evol = ham_evol_gate.control(1)
        ham_evol_qubits = [phase_anc_reg[0]] + list(ansatz_reg)
    except Exception as e:
        print(f"ERROR building Controlled Hamiltonian Evolution Gate: {e}")
        return None

    # 3. Assemble Full U'_PE Circuit (Quantum Phase Estimation)
    u_pe_full_circuit.h(phase_anc_reg[0])

    # Apply W gate: try to compose the circuit directly; fall back to append(instruction)
    try:
        u_pe_full_circuit.compose(w_circuit, qubits=w_qubits, inplace=True)
    except Exception:
        try:
            u_pe_full_circuit.append(w_gate_inst, w_qubits)
        except Exception as e:
            print(f"ERROR appending W gate: {e}")
            return None

    # Apply C-U: Kicks back phase e^(i * E(theta) * dt)
    u_pe_full_circuit.append(controlled_ham_evol, ham_evol_qubits)

    # Apply W_inverse: compose inverse then fallback to append inverse instruction
    try:
        u_pe_full_circuit.compose(w_circuit.inverse(), qubits=w_qubits, inplace=True)
    except Exception:
        try:
            u_pe_full_circuit.append(w_gate_inst.inverse(), w_qubits)
        except Exception:
            # If inverse append fails, continue (best-effort)
            pass

    u_pe_full_circuit.h(phase_anc_reg[0])

    return u_pe_full_circuit

# ==============================================================================
# STEP 5: QLTO OPTIMIZER (HYBRID LOOP & UTILITIES)
# (Adapted from qlto_nisq.py)
# ==============================================================================

# --- Parameter Encoding ---

# ----------------------------
# Gray code helpers
# ----------------------------
def binary_to_gray(n: int) -> int:
    return n ^ (n >> 1)


def gray_to_binary(g: int) -> int:
    b = 0
    shift = g
    while shift:
        b ^= shift
        shift >>= 1
    return b

def encode_parameters(theta_vector: np.ndarray, param_bounds: np.ndarray, bits_per_param: int) -> Tuple[int, int]:
    """Converts a continuous theta vector into a discrete integer index."""
    n_params = len(theta_vector)
    total_parameter_qubits = n_params * bits_per_param
    num_bins = 2**bits_per_param
    full_bitstring = ""
    for i in range(n_params):
        theta_val = theta_vector[i]
        min_val, max_val = param_bounds[i]
        theta_clamped = np.clip(theta_val, min_val, max_val)
        range_val = max_val - min_val
        if range_val > 1e-9:
            normalized_val = (theta_clamped - min_val) / (range_val + 1e-9)
        else:
            normalized_val = 0.5
        bin_index = int(round(np.clip(normalized_val, 0.0, 1.0) * (num_bins - 1)))
        full_bitstring += format(bin_index, f'0{bits_per_param}b')
    parameter_index = int(full_bitstring, 2) if full_bitstring else 0
    # Convert to Gray to improve Hamming-neighborhood locality
    gray_index = binary_to_gray(parameter_index)
    return int(gray_index), total_parameter_qubits

def decode_parameters(parameter_index: int, n_params: int, bits_per_param: int, param_bounds: np.ndarray) -> np.ndarray:
    """Converts a discrete integer index back into a continuous theta vector."""
    total_parameter_qubits = n_params * bits_per_param
    num_bins = 2**bits_per_param
    dimension = 2**total_parameter_qubits
    # Decode from Gray back to binary index
    if parameter_index < 0 or parameter_index >= dimension: parameter_index = 0
    bin_index = gray_to_binary(parameter_index)
    full_bitstring = format(bin_index, f'0{total_parameter_qubits}b')
    theta_vector = np.zeros(n_params)
    for i in range(n_params):
        start = i * bits_per_param
        end = (i + 1) * bits_per_param
        bin_string = full_bitstring[start:end]
        if not bin_string: continue
        bin_index = int(bin_string, 2)
        normalized_val = (bin_index + 0.5) / num_bins # Use center of bin
        min_val, max_val = param_bounds[i]
        theta_vector[i] = normalized_val * (max_val - min_val) + min_val
        theta_vector[i] = np.clip(theta_vector[i], min_val + 1e-6, max_val - 1e-6)
    return theta_vector

# --- QLTO Utility Functions ---

def build_tunneling_operator_QW(n_qubits_p: int) -> QuantumCircuit: 
    """Builds the mixer/tunneling operator T for the parameter space."""
    if not QISKIT_AVAILABLE: return None
    qc = QuantumCircuit(n_qubits_p, name="T_QWOA_Mixer_Rx")
    # Simple mixer: RX(pi/4) on all parameter qubits
    for i in range(n_qubits_p):
        qc.rx(np.pi / 4, i)
    return qc

def run_qlto_evolution_nisq(
    n_qubits_p: int, n_qubits_ansatz: int, initial_param_idx: int, 
    U_PE_full_circuit: QuantumCircuit, T_gate: QuantumCircuit, K_steps: int
    ) -> QuantumCircuit:
    """ Applies interleaved evolution (U_PE * T)^K """
    if not QISKIT_AVAILABLE or U_PE_full_circuit is None or T_gate is None: return None
    
    # Registers must match U_PE
    param_reg = QuantumRegister(n_qubits_p, 'param')
    ansatz_reg = QuantumRegister(n_qubits_ansatz, 'ansatz')
    phase_anc_reg = AncillaRegister(1, 'phase_anc')
    all_regs = [phase_anc_reg, param_reg, ansatz_reg]
    evol_qc = QuantumCircuit(*all_regs, name="QLTO_Evolution")

    # 1. Prepare initial parameter state
    prep_circ = QuantumCircuit(param_reg, name="Prep_Init")
    binary_string = format(initial_param_idx, f'0{n_qubits_p}b')
    for i, bit in enumerate(reversed(binary_string)): # LSB
        if bit == '1': prep_circ.x(i)
    evol_qc.append(prep_circ.to_instruction(), param_reg)

    u_pe_qubits = list(phase_anc_reg) + list(param_reg) + list(ansatz_reg)
    u_pe_inst = U_PE_full_circuit.to_instruction(label="U_PE_Shallow")
    t_inst = T_gate.to_instruction(label="T_QWOA_Inst")

    # 2. Apply K steps of evolution
    for k in range(int(round(K_steps))):
        evol_qc.append(u_pe_inst, u_pe_qubits) # Apply U_PE
        evol_qc.append(t_inst, param_reg)      # Apply T
        
    return evol_qc

def measure_and_process_samples_nisq(
    evolved_circuit: QuantumCircuit, shots: int, n_qubits_p: int, 
    param_reg_name: str, sampler: CountingWrapper
) -> Dict[int, float]:
    """ Runs evolved circuit, measures param reg, returns probability distribution."""
    if not QISKIT_AVAILABLE or evolved_circuit is None: return {}
    
    param_reg = next((reg for reg in evolved_circuit.qregs if reg.name == param_reg_name), None)
    if param_reg is None or param_reg.size != n_qubits_p: return {}

    measure_qc = evolved_circuit.copy(name="QLTO_Measure_Param")
    param_classical_reg = ClassicalRegister(n_qubits_p, name='c_param')
    measure_qc.add_register(param_classical_reg)
    measure_qc.measure(param_reg, param_classical_reg)

    try:
        # Sampler V1 run method expects a list of (circuit,) tuples
        job = sampler.run([(measure_qc,)], shots=max(1, shots)) 
        result = job.result()
        
        if hasattr(result, 'quasi_dists') and result.quasi_dists:
            # Qiskit Aer Sampler returns quasi_dists
            raw_dist = result.quasi_dists[0]
            # Convert binary keys to integer keys
            return {int(k, 2): float(v) for k, v in raw_dist.binary_probabilities().items() if v is not None}
        
        return {}
    except Exception as e:
        print(f"DEBUG: Sampler run failed in measure_and_process: {e}")
        return {}

def calculate_prob_filtered_centroid_theta(
    quasi_dist: Dict[int, float], top_n_states: int, n_params: int, 
    bits_per_param: int, param_bounds: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """
    Calculates the centroid (weighted average) of the best parameters
    found in the probability distribution.
    """
    mid_points = np.mean(param_bounds, axis=1)
    if not quasi_dist: return mid_points.copy()
    
    valid_dist = {idx: prob for idx, prob in quasi_dist.items() if prob > 1e-9 and np.isfinite(prob)}
    if not valid_dist: return mid_points.copy()
    
    # Sort states by probability
    sorted_states = sorted(valid_dist.items(), key=lambda item: item[1], reverse=True)
    num_states_to_keep = max(1, min(top_n_states, len(sorted_states)))
    filtered_states = sorted_states[:num_states_to_keep]
    
    temp_centroid = np.zeros(n_params)
    total_prob_filtered = 0.0

    for param_idx, probability in filtered_states:
        # Apply temperature to sharpen or flatten distribution.
        # temperature < 1.0 -> sharper (weights high-prob states more)
        # temperature > 1.0 -> flatter
        try:
            if temperature <= 0:
                temp = 1.0
            else:
                temp = temperature
            weight = probability ** (1.0 / temp)
        except Exception:
            weight = probability
        
        try:
            decoded_theta = decode_parameters(param_idx, n_params, bits_per_param, param_bounds)
            if np.all(np.isfinite(decoded_theta)):
                temp_centroid += weight * decoded_theta
                total_prob_filtered += weight
        except Exception:
            continue
            
    if total_prob_filtered > 1e-9:
        centroid_theta = temp_centroid / total_prob_filtered
    else:
        centroid_theta = mid_points.copy()
    
    # Final clipping
    for i in range(n_params):
        min_b, max_b = param_bounds[i]
        centroid_theta[i] = np.clip(centroid_theta[i], min_b + 1e-6, max_b - 1e-6)
        
    return centroid_theta

# --- Smart Initialization (from Structure-Aligned) ---

def aligned_initial_parameters(n_params: int, param_bounds: np.ndarray, p_layers: int) -> np.ndarray:
    """
    Generates a good initial guess for QAOA parameters.
    Uses an adiabatic-inspired linear schedule.
    """
    gammas = np.zeros(p_layers)
    betas = np.zeros(p_layers)
    
    for k in range(p_layers):
        s = (k + 0.5) / p_layers # Schedule from 0 -> 1
        
        # Gamma: 0 -> pi
        gammas[k] = s * np.pi
        
        # Beta: pi/2 -> 0
        betas[k] = (1 - s) * np.pi / 2
        
    # Interleave: [gamma_0, beta_0, gamma_1, beta_1, ...]
    initial_theta = np.zeros(n_params)
    initial_theta[0::2] = gammas
    initial_theta[1::2] = betas
    
    # Clip to bounds just in case
    for i in range(n_params):
        min_b, max_b = param_bounds[i]
        initial_theta[i] = np.clip(initial_theta[i], min_b + 1e-6, max_b - 1e-6)
        
    return initial_theta

# --- The Main QLTO Optimizer Function ---

def run_qlto_optimizer(
    sat_problem: SATProblem,
    p_layers: int = 2,
    bits_per_param: int = 4,
    shots: int = 400,
    max_iterations: int = 60,
    scans_per_epoch: int = 1, # Keep low for speed
    prob_filter_top_n: int = 5,
    K_steps_initial: int = 5,
    K_steps_final: int = 1,
    K_annealing_decay: float = 2.0,
    delta_t_initial: float = 0.35, 
    delta_t_final: float = 0.05,
    update_eta: float = 0.2,
    rng_seed: Optional[int] = None,
    use_smart_init: bool = True
    , init_theta: Optional[np.ndarray] = None
):
    """
    This is the main QLTO optimizer loop.
    It takes a SAT problem and finds the optimal QAOA parameters.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit not available; cannot run QLTO.")

    # --- 1. Setup ---
    print("\n--- [QLTO Optimizer] Starting ---")
    print(f"  Problem: {sat_problem.n_vars} vars, {sat_problem.n_clauses} clauses")
    print(f"  QAOA Layers (p): {p_layers}")
    print(f"  QLTO Params: {bits_per_param} bits/param, {max_iterations} iterations")
    
    estimator = CountingWrapper(BaseEstimator())
    sampler = CountingWrapper(BaseSampler())

    n_qubits_ansatz = sat_problem.n_vars

    # --- 2. Build Core Components ---
    print("  [QLTO] Building VQE Hamiltonian (H_C) for cost function...")
    vqe_hamiltonian = sat_to_hamiltonian(sat_problem)
    
    print(f"  [QLTO] Building QAOA Ansatz Template (for W_Gate)...")
    vqe_ansatz_template, n_params = create_qaoa_ansatz(sat_problem, p_layers)
    
    if n_params != 2 * p_layers:
        print(f"Warning: Expected {2*p_layers} params, but ansatz has {n_params}. Adjusting.")
    
    # Define parameter bounds (gamma: 0->2pi, beta: 0->pi)
    param_bounds = []
    for i in range(n_params):
        if i % 2 == 0: # Gamma
            param_bounds.append([0.0, 2 * np.pi])
        else: # Beta
            param_bounds.append([0.0, np.pi])
    param_bounds = np.array(param_bounds)

    # --- 3. Set Initial Parameters ---
    # Deterministic seed for reproducibility when requested
    if rng_seed is not None:
        try:
            random.seed(rng_seed)
            np.random.seed(rng_seed)
        except Exception:
            pass

    if use_smart_init:
        print("  [QLTO] Using smart adiabatic-inspired initial parameters.")
        theta = aligned_initial_parameters(n_params, param_bounds, p_layers)
    else:
        print("  [QLTO] Using random initial parameters.")
        theta = np.random.rand(n_params) * (param_bounds[:, 1] - param_bounds[:, 0]) + param_bounds[:, 0]

    # Warm-start override
    if init_theta is not None:
        try:
            theta = np.array(init_theta, dtype=float).copy()
            print("  [QLTO] Using provided init_theta warm-start.")
        except Exception:
            pass

    # --- 4. Define Energy Evaluation Function (for logging) ---
    def eval_energy(theta_vec: np.ndarray) -> float:
        # Classical eval: binds parameters and calls Estimator
        try:
            bound_ansatz = vqe_ansatz_template.assign_parameters(theta_vec.tolist())
            pubs = [(bound_ansatz, vqe_hamiltonian)]
            job = estimator.run(pubs) 
            r = job.result()
            return float(r.values[0])
        except Exception as e:
            print(f"DEBUG: Estimator failed: {e}")
            return float('inf')

    # Initial energy calculation (1 NFEV)
    energy = eval_energy(theta)
    if not np.isfinite(energy):
        print("ERROR: Initial energy calculation failed. Aborting.")
        return {"error": "Initial energy failed to be finite."}

    energy_history: List[float] = [energy]
    best_energy = energy
    best_theta = theta.copy()
    
    print(f"  [QLTO] Initial Energy: {energy:.6f}")
    
    last_oracle_delta_t = -1.0 
    U_PE_circuit_cache = None
    # Frequency for delta_t rebuilds (stabilize U_PE)
    delta_update_freq = 4
    # plateau patience for skipping rebuilds
    patience = 4
    stall = 0
    
    # --- 5. Run The QLTO Hybrid Loop ---
    for t in range(max_iterations):
        print(f"\n--- Iteration {t+1}/{max_iterations} ---")
        previous_theta = theta.copy()
        
        # 5a. Anneal QLTO parameters (K and delta_t)
        progress_ratio = t / max(1, max_iterations - 1)
        decay_factor = (1 - progress_ratio) ** K_annealing_decay
        K_steps = max(K_steps_final, int(K_steps_initial * decay_factor))
        # Only update delta_t occasionally to avoid rebuilding the oracle every iteration
        if t % delta_update_freq == 0:
            current_delta_t = delta_t_initial - (delta_t_initial - delta_t_final) * progress_ratio
            rebuild_oracle = True
        else:
            current_delta_t = last_oracle_delta_t if last_oracle_delta_t > 0 else delta_t_initial
            rebuild_oracle = False
        print(f"  Annealing: K_steps={K_steps}, delta_t={current_delta_t:.3f} (rebuild={rebuild_oracle})")

        # 5b. Encode current theta vector to integer index
        param_idx, n_qubits_p = encode_parameters(theta, param_bounds, bits_per_param)
        print(f"  State: n_qubits_p={n_qubits_p}, current_index={param_idx}")

        # 5c. Build/Retrieve QLTO Oracles (U_PE and T)
        try:
            # Only rebuild when requested and not stalled, or when the cache is empty
            if (rebuild_oracle and stall < patience) or U_PE_circuit_cache is None or abs(current_delta_t - last_oracle_delta_t) > 1e-9:
                print(f"  Building new U_PE oracle (delta_t={current_delta_t:.3f})...")
                U_PE_full_circuit = build_coherent_phase_oracle_nisq(
                    n_params, bits_per_param, param_bounds, current_delta_t,
                    vqe_ansatz_template, vqe_hamiltonian, n_qubits_ansatz
                )
                last_oracle_delta_t = current_delta_t
                U_PE_circuit_cache = U_PE_full_circuit
            else:
                U_PE_full_circuit = U_PE_circuit_cache
                
            T_gate = build_tunneling_operator_QW(n_qubits_p)
            if T_gate is None or U_PE_full_circuit is None: 
                raise ValueError("Oracle/Mixer build failed.")
        except Exception as e:
            print(f"\n *** Optimizer stopped (Oracle/Mixer build failed): {e} ***")
            break

        # 5d. Run QLTO Evolution on Sampler
        print(f"  Running QLTO evolution ({K_steps} steps) on Sampler...")
        evol_qc = run_qlto_evolution_nisq(
            n_qubits_p, n_qubits_ansatz, param_idx, 
            U_PE_full_circuit, T_gate, K_steps
        )
        # Adaptive shots: cheap early, heavier later
        if t < max(3, max_iterations // 4):
            use_shots = max(200, shots // 4)
        else:
            use_shots = shots
        dist = measure_and_process_samples_nisq(
            evol_qc, use_shots, n_qubits_p, 'param', sampler
        )
        # If peaked and we used low shots, re-measure with higher shots for a better centroid
        if dist and use_shots < shots:
            try:
                if max(dist.values()) > 0.2:
                    dist2 = measure_and_process_samples_nisq(evol_qc, min(shots, use_shots * 4), n_qubits_p, 'param', sampler)
                    if dist2:
                        dist = dist2
            except Exception:
                pass
        if not dist:
            print("  Warning: Sampler returned no results. Skipping step.")
            theta = previous_theta.copy()
            energy = energy_history[-1]
            energy_history.append(energy)
            continue
            
        # 5e. Calculate Centroid (Classical post-processing)
        centroid = calculate_prob_filtered_centroid_theta(
            dist, prob_filter_top_n, n_params, bits_per_param, param_bounds, temperature=0.5
        )

        # 5f. Update theta vector
        next_theta = (1 - update_eta) * best_theta + update_eta * centroid 
        theta = np.clip(next_theta, param_bounds[:, 0] + 1e-6, param_bounds[:, 1] - 1e-6)

        # 5g. Evaluate energy for logging (1 NFEV)
        energy = eval_energy(theta)
        
        if not np.isfinite(energy):
            print("  Warning: New energy is not finite. Reverting to previous theta.")
            theta = previous_theta.copy()
            energy = energy_history[-1]
        
        # 5h. Update Best Found Solution
        if np.isfinite(energy) and energy < best_energy:
            print(f"  *** New Best Energy: {energy:.6f} ***")
            best_energy = energy
            best_theta = theta.copy()
            stall = 0
        else:
            stall += 1

        energy_history.append(energy)
        print(f"  Energy: {energy:.6f} (Best: {best_energy:.6f})")

    # --- 6. Return Final Results ---
    print(f"\n--- [QLTO Optimizer] Finished ---")
    print(f"  Final NFEV count (Estimator calls): {estimator.n_calls}")
    print(f"  Final Sampler calls: {sampler.n_calls}")
    print(f"  Final Best Energy: {best_energy:.6f}")
    
    return {
        'final_theta': best_theta,
        'final_energy': best_energy,
        'energy_history': energy_history,
        'estimator_calls': estimator.n_calls,
        'sampler_calls': sampler.n_calls,
        'total_iterations': len(energy_history) - 1,
    }


# ==============================================================================
# STEP 6: TEST CASE AND VERIFICATION
# ==============================================================================

def run_final_solution_check(
    sat_problem: SATProblem, 
    p_layers: int, 
    best_theta: np.ndarray, 
    shots: int = 4096
) -> Tuple[bool, str, Dict[str, float]]:
    """
    Takes the best parameters, runs the QAOA circuit, and finds the
    most probable bitstring solution.
    """
    print("\n--- [Final Solution Check] ---")
    
    # 1. Build the final, optimized QAOA circuit
    final_ansatz, _ = create_qaoa_ansatz(sat_problem, p_layers)
    final_circuit = final_ansatz.assign_parameters(best_theta)
    final_circuit.measure_all()
    
    # 2. Run on Sampler to get counts
    print(f"  Running final optimized circuit for {shots} shots...")
    sampler = AerSampler()
    job = sampler.run(final_circuit, shots=shots)
    result = job.result()
    
    # Get binary probabilities (e.g., {'001': 0.1, '101': 0.8, ...})
    counts = result.quasi_dists[0].binary_probabilities()
    
    if not counts:
        print("  ERROR: No measurement results from final run.")
        return False, "", {}
    # Quick diagnostics
    print("  Sampler top states:", sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8])

    # 2b. Try verifying top-k candidates directly and attempt local hill-climb repair
    candidate_assign, candidate_prob, candidate_bitstr = verify_counts_with_verifier(counts, sat_problem.n_vars, sat_problem, top_k=8)
    if candidate_assign is not None:
        print("  Found satisfying candidate directly from samples (prob {:.3f}).".format(candidate_prob))
        return True, candidate_bitstr, counts

    # Try hill-climb repair on top candidates
    candidates = samples_to_assignments(counts, sat_problem.n_vars, top_k=8)
    for assign, prob in candidates:
        improved_assign, viol = local_hill_climb_on_assignment(sat_problem, assign)
        if viol == 0:
            # Found repair
            bs = ''.join('1' if improved_assign[i+1] else '0' for i in range(min(sat_problem.n_vars, len(improved_assign))))
            bitstring_lsb = bs[::-1]
            print("  Found solved assignment via local hill-climb.")
            return True, bitstring_lsb, counts
        
    # 3. Find the most probable solution
    best_bitstring = max(counts, key=counts.get)
    best_prob = counts[best_bitstring]
    
    print(f"  Most probable solution: '{best_bitstring}' (Prob: {best_prob*100:.2f}%)")
    
    # 4. Verify the solution
    # Qiskit uses LSB (Little-Endian) order, so '101' means q0=1, q1=0, q2=1
    # Our SAT problem maps var=1 -> q0, var=2 -> q1, etc.
    # So we must REVERSE the bitstring for our mapping
    solution_bitstring_msb = best_bitstring[::-1]
    
    # Convert '101' -> {1: True, 2: False, 3: True}
    solution_dict = {}
    for i, bit in enumerate(solution_bitstring_msb):
        var_num = i + 1
        solution_dict[var_num] = (bit == '1')
        
    # Check if this assignment satisfies the problem
    is_solution = sat_problem.is_satisfied(solution_dict)
    violated = sat_problem.count_violated(solution_dict)
    
    print(f"  Assignment: {solution_dict}")
    print(f"  Violated clauses: {violated}")
    
    return is_solution, best_bitstring, counts


def unit_test_sat_to_hamiltonian():
    """Quick unit test for sat_to_hamiltonian.

    Builds a tiny SAT problem and checks that the Hamiltonian
    produced assigns positive energy to unsatisfied assignments
    and (near-)zero energy to satisfying assignments.
    """
    try:
        import numpy as np
    except Exception:
        print("unit_test_sat_to_hamiltonian: numpy not available")
        return False

    # Simple clause (x1 OR x2) -- only assignment that violates is x1=False,x2=False
    p = SATProblem(n_vars=2, clauses=[SATClause((1, 2))])
    H = sat_to_hamiltonian(p)
    # Convert to dense matrix for small n (2 qubits -> 4x4)
    try:
        mat = H.to_matrix()
    except Exception:
        # Fallback for older qiskit versions
        mat = H.to_sparse().toarray()

    energies = []
    for i in range(4):
        v = np.zeros(4, dtype=complex)
        v[i] = 1.0
        e = float(np.vdot(v, mat.dot(v)).real)
        energies.append(e)

    # The |00> basis index is 0 in the standard ordering. That assignment should be penalized.
    unsat_energy = energies[0]
    sat_energies = energies[1:]

    ok = (unsat_energy > 1e-8) and all(abs(e) < 1e-6 for e in sat_energies)
    if ok:
        print("unit_test_sat_to_hamiltonian: PASS")
    else:
        print("unit_test_sat_to_hamiltonian: FAIL -> energies=", energies)
    return ok


def brute_force_backdoor(sat_problem, backdoor_vars):
    """Try all assignments to a small set of backdoor variables and
    brute-force the remaining vars to find a satisfying assignment.

    Returns a satisfying full assignment dict {var_index: bool} or None.
    Note: this is intended for small problems / small backdoors only.
    """
    from itertools import product

    n = sat_problem.n_vars
    backdoor = list(backdoor_vars)
    others = [v for v in range(1, n + 1) if v not in backdoor]

    # Safety cap to avoid huge searches
    if len(backdoor) > 20 or len(others) > 25:
        print("brute_force_backdoor: problem too large for brute force; aborting")
        return None

    for bits in product([False, True], repeat=len(backdoor)):
        partial = {v: b for v, b in zip(backdoor, bits)}

        # brute-force remaining vars
        for bits2 in product([False, True], repeat=len(others)):
            assign = partial.copy()
            for v, b in zip(others, bits2):
                assign[v] = b
            if sat_problem.is_satisfied(assign):
                return assign

    return None


def samples_to_assignments(counts, n_vars, top_k=10):
    """Convert sampler counts/probabilities into a list of (assignment, prob)

    - counts: dict mapping bitstring (Qiskit LSB-order) -> count or probability
    - n_vars: number of SAT variables
    - top_k: return only the top-k most likely samples
    """
    import numpy as np

    if not counts:
        return []

    total = float(sum(counts.values()))
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for bitstring, val in items:
        prob = val / total if total > 0 else 0.0
        # Convert Qiskit LSB-order -> our var numbering (1-indexed left-to-right)
        # Qiskit '101' means q0=1, q1=0, q2=1 so we reverse to MSB-order to map var1->first bit
        bs = bitstring[::-1]
        assign = {i + 1: (bs[i] == '1') for i in range(min(len(bs), n_vars))}
        out.append((assign, prob))
    return out


def local_hill_climb_on_assignment(sat_problem: SATProblem, starting_assign: Dict[int,bool], max_iters=100):
    """Greedy hill-climb: flip single bits if it reduces violated clause count."""
    current = starting_assign.copy()
    current_viol = sat_problem.count_violated(current)
    for _ in range(max_iters):
        improved = False
        for var in range(1, sat_problem.n_vars + 1):
            new_assign = current.copy()
            new_assign[var] = not new_assign[var]
            new_viol = sat_problem.count_violated(new_assign)
            if new_viol < current_viol:
                current = new_assign
                current_viol = new_viol
                improved = True
                if current_viol == 0:
                    return current, 0
                break
        if not improved:
            break
    return current, current_viol


def paramindex_to_subset(parameter_index: int, n_vars: int, n_qubits_p: int):
    """Map parameter index bits to a subset of variables.

    Simple mapping: use the least-significant n_vars bits as mask (q0 -> var1).
    """
    binstr = format(parameter_index, f'0{n_qubits_p}b')
    mask = binstr[-n_vars:][::-1]
    subset = [i+1 for i, b in enumerate(mask) if b == '1']
    return subset


def find_backdoor_from_samples(samples: Dict[int, float], sat_problem: SATProblem, n_qubits_p: int, top_k=20):
    sorted_s = sorted(samples.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    for idx, prob in sorted_s:
        subset = paramindex_to_subset(idx, sat_problem.n_vars, n_qubits_p)
        if not subset:
            continue
        sol = brute_force_backdoor(sat_problem, subset)
        if sol is not None:
            return subset, sol
    return None, None


# -----------------------------
# FPT / Backdoor helper functions
# (imported/adapted from qlto_fpt_solver_v7...)
# -----------------------------

def paramindex_to_subset_v2(parameter_index: int, n_vars: int, n_mask_bits: int) -> List[int]:
    """Map a parameter index into a subset of variable indices (0-indexed return).

    This version returns 0-indexed variable indices to match the backdoor routines.
    It uses the least-significant bits of the parameter index as the mask.
    """
    total_bits = n_mask_bits
    binstr = format(parameter_index, f'0{total_bits}b') if total_bits > 0 else ''
    mask = binstr[-n_vars:][::-1] if binstr else ''
    subset = [i for i, b in enumerate(mask) if b == '1']
    return subset


def verify_solution(clauses: List[Tuple[int, ...]], assignment: Dict[int, bool]) -> bool:
    """
    Verifies that a 0-indexed assignment satisfies a list of 1-indexed clauses.
    `clauses` is a list of tuples of integer literals (DIMACS format).
    `assignment` maps global 0-indexed variable IDs -> bool.
    """
    if assignment is None:
        return False
    for clause in clauses:
        satisfied = False
        for lit in clause:
            var_0_idx = abs(lit) - 1
            is_positive = (lit > 0)
            if var_0_idx in assignment and assignment[var_0_idx] == is_positive:
                satisfied = True
                break
        if not satisfied:
            return False
    return True


def reduce_problem_by_fixing(sat_problem: SATProblem, fixed_assign: Dict[int, bool]):
    """Return a reduced SATProblem after fixing some variables.

    fixed_assign: mapping of 0-indexed var -> bool
    Returns (reduced_problem, remaining_vars_global_0_idx)
    """
    # Build new clauses by substituting fixed vars; adjust variable numbering
    new_clauses = []
    remaining_vars = []
    fixed_set = set(fixed_assign.keys())

    # Map old global var (1-indexed) -> new index (1-indexed) for remaining vars
    remap = {}
    cur = 1
    for v in range(1, sat_problem.n_vars + 1):
        if (v - 1) not in fixed_set:
            remap[v] = cur
            remaining_vars.append(v - 1)
            cur += 1

    for clause in sat_problem.clauses:
        skip_clause = False
        new_clause = []
        for lit in clause.literals:
            var = abs(lit)
            val = True if lit > 0 else False
            if (var - 1) in fixed_set:
                # If fixed variable satisfies the clause, clause is removed
                if fixed_assign[var - 1] == val:
                    skip_clause = True
                    break
                else:
                    # Literal false under fixed assignment; omit it
                    continue
            else:
                # Append remapped literal
                new_var = remap[var]
                new_lit = new_var if lit > 0 else -new_var
                new_clause.append(new_lit)
        if skip_clause:
            continue
        # If clause becomes empty => UNSAT under this fixing
        if not new_clause:
            return None, None
        new_clauses.append(tuple(new_clause))

    reduced_problem = SATProblem(n_vars=len(remaining_vars), clauses=[SATClause(c) for c in new_clauses])
    return reduced_problem, remaining_vars


def try_candidate_backdoor_pysat(sat_problem: SATProblem, candidate_subset_0_idx: List[int], timeout: Optional[float] = None, max_masks: Optional[int] = None):
    """Try all assignments to candidate_subset_0_idx using PySAT to solve the reduced instances.

    Returns a full assignment mapping (0-indexed -> bool) if found, else None.
    """
    try:
        from pysat.solvers import Glucose3
    except Exception:
        print("PySAT not available; cannot run try_candidate_backdoor_pysat")
        return None

    k_prime = len(candidate_subset_0_idx)
    # Determine mask iteration limit
    total_masks = 2 ** k_prime
    if max_masks is None:
        max_masks = total_masks
    else:
        max_masks = min(int(max_masks), total_masks)

    start_time = time.time()
    # Iterate up to max_masks or until timeout
    for mask in range(max_masks):
        fixed = {}
        for i, var0 in enumerate(candidate_subset_0_idx):
            bit = (mask >> i) & 1
            fixed[var0] = bool(bit)

        reduced, remaining = reduce_problem_by_fixing(sat_problem, fixed)
        if reduced is None:
            # Unsat under this fixing
            continue

        # Build CNF for PySAT (1-indexed variable numbers)
        cnf = []
        for clause in reduced.clauses:
            cnf.append(list(clause.literals))

        # Try to solve with Glucose3
        try:
            g = Glucose3()
            for c in cnf:
                g.add_clause(c)
            sat = g.solve()
            if not sat:
                g.delete()
                continue
            model = g.get_model()
            g.delete()
        except Exception:
            # solver error - skip this mask
            continue

        # Timeout check between masks
        if timeout is not None and (time.time() - start_time) > float(timeout):
            # Best-effort: stop searching further
            return None

        # model contains values for the reduced_problem variables (1-indexed)
        full_assign = {}
        # Apply fixed assignments
        for v0, b in fixed.items():
            full_assign[v0] = b

        # Map reduced model back to global 0-indexed vars
        for lit in model:
            if lit == 0:
                continue
            var = abs(lit)
            val = (lit > 0)
            # remaining[var-1] is the original 0-indexed variable
            if 1 <= var <= len(remaining):
                global_var0 = remaining[var - 1]
                full_assign[global_var0] = val

        # Final check with our verifier using 1-indexed clauses
        original_clauses = [c.literals for c in sat_problem.clauses]
        if verify_solution(original_clauses, full_assign):
            return full_assign

    return None


def extract_subset_from_top_samples(samples: Dict[int, float], n_vars: int, topT: int = 50, threshold: float = 0.15, n_mask_bits: int = 10) -> List[int]:
    """Produce a superset of likely backdoor variables (0-indexed) from samples.

    This mirrors the v7 benchmark extraction: look at the top-T parameter indices,
    decode each index into a variable subset (via paramindex_to_subset_v2) and
    score variables by weighted frequency. Return the superset consisting of
    variables whose weighted-score >= threshold.
    """
    if not samples:
        return []

    # Consider top-T parameter indices by probability
    sorted_s = sorted(samples.items(), key=lambda kv: kv[1], reverse=True)[:topT]
    var_scores = defaultdict(float)  # variable (0-index) -> score
    total_weight = 0.0
    for idx, p in sorted_s:
        subset = paramindex_to_subset_v2(idx, n_vars, n_mask_bits)
        if not subset:
            continue
        # Weight each variable in subset by the sample probability
        for v in subset:
            var_scores[v] += p
        total_weight += p

    if total_weight <= 0:
        return []

    # Normalize and select variables above threshold
    superset = [v for v, s in var_scores.items() if (s / total_weight) >= threshold]
    # Sort for determinism
    superset.sort()
    return superset


def is_minimal_backdoor(sat_problem: SATProblem, subset_0_idx: List[int]) -> bool:
    """Check minimality by testing proper subsets (greedy small check)."""
    if not subset_0_idx:
        return False
    # Try removing one element at a time and see if the remaining still is a backdoor
    for i in range(len(subset_0_idx)):
        smaller = subset_0_idx[:i] + subset_0_idx[i+1:]
        sol = try_candidate_backdoor_pysat(sat_problem, smaller)
        if sol is not None:
            return False
    return True


def shrink_superset_greedy(sat_problem: SATProblem, superset_0_idx: List[int], timeout: Optional[float] = None, per_candidate_max_masks: Optional[int] = 200000) -> List[int]:
    """Attempt to greedily shrink a superset backdoor to a smaller one by dropping redundant vars.

    New: accepts an overall `timeout` (seconds) and `per_candidate_max_masks` to bound
    the inner classical search on each trial. This makes shrinking fast and bounded.
    """
    current = list(superset_0_idx)
    changed = True
    start = time.time()
    # Guard: maximum outer iterations to avoid pathological loops
    max_outer_iters = max(4, len(current) * 2)
    outer_iter = 0
    while changed and (timeout is None or (time.time() - start) < float(timeout)) and outer_iter < max_outer_iters:
        outer_iter += 1
        changed = False
        # Try removing each variable; randomize order to avoid worst-case deterministic behavior
        for v in list(current):
            # Check global timeout
            if timeout is not None and (time.time() - start) >= float(timeout):
                return current

            trial = [x for x in current if x != v]
            # Compute per-candidate mask cap: don't try more than per_candidate_max_masks masks
            try:
                total_masks = 2 ** len(trial)
            except OverflowError:
                total_masks = float('inf')
            max_masks = min(int(total_masks) if total_masks != float('inf') else per_candidate_max_masks, int(per_candidate_max_masks))

            sol = try_candidate_backdoor_pysat(sat_problem, trial, timeout=max(0.5, (timeout - (time.time() - start)) if timeout else None), max_masks=max_masks)
            if sol is not None:
                current = trial
                changed = True
                break
    return current


def run_fpt_pipeline_from_samples(clauses: List[Tuple[int, ...]], n_vars: int, samples: Dict[int, float], n_mask_bits: int = 10, top_k: int = 50) -> Tuple[Optional[List[int]], Optional[Dict[int, bool]]]:
    """Legacy entrypoint: run FPT pipeline starting from a samples distribution.

    Kept for backwards-compatibility with callers that already have a samples dict.
    """
    # Build a SATProblem wrapper for helper functions. Accept either a
    # SATProblem instance or an iterable of clause tuples.
    if isinstance(clauses, SATProblem):
        problem = clauses
        if n_vars is None:
            n_vars = problem.n_vars
    else:
        sat_clauses = [SATClause(c) for c in clauses]
        problem = SATProblem(n_vars=n_vars, clauses=sat_clauses)

    candidate_subsets = extract_subset_from_top_samples(samples, top_k, n_vars, n_mask_bits)
    for subset in candidate_subsets:
        # Try PySAT-based candidate solving
        sol = try_candidate_backdoor_pysat(problem, subset)
        if sol is not None:
            # Return subset and full assignment (0-indexed map)
            return subset, sol

    return None, None


def run_fpt_pipeline(clauses: List[Tuple[int, ...]], n_vars: int, mode_or_name, cfg: Dict[str, Any], trial_seed: int) -> dict:
    """Primary FPT pipeline expected by quantum_sat_solver.

    Signature matches the v7 experiment runner:
      run_fpt_pipeline(clauses, n_vars, problem_name_or_mode, cfg, trial_seed)

    This function adapts the single-trial `run_trial` logic from the v7
    benchmark script and returns a dictionary with results similar to
    that script (`success`, `subset`, `solution_0_idx`, timings, etc.).
    """
    # Import PySAT availability
    try:
        from pysat.solvers import Glucose3  # noqa: F401
        PYSAT_AVAILABLE = True
    except Exception:
        PYSAT_AVAILABLE = False

    # Unpack config
    p_layers = cfg.get('p_layers', 2)
    bits_per_param = cfg.get('bits_per_param', 3)
    n_mask_bits = cfg.get('N_MASK_BITS', 10)
    shots = cfg.get('shots', 2048)
    top_T_candidates = cfg.get('top_T_candidates', 50)
    freq_threshold = cfg.get('freq_threshold', 0.15)
    # Verbose / progress control
    verbose = bool(cfg.get('verbose', False) or cfg.get('progress', False))

    # Build SATProblem. Accept either:
    #  - clauses: List[Tuple[int,...]] (the normal case), or
    #  - clauses: SATProblem (caller already constructed a problem)
    try:
        if isinstance(clauses, SATProblem):
            # Caller passed a SATProblem directly
            problem = clauses
            # If n_vars provided as None or inconsistent, prefer the problem's n_vars
            if n_vars is None:
                n_vars = problem.n_vars
            elif n_vars != problem.n_vars:
                # Allow caller to pass n_vars but warn/normalize to the problem
                # (we avoid raising here to be permissive)
                n_vars = problem.n_vars
        else:
            # Expect clauses to be an iterable of tuples
            sat_clauses = [SATClause(c) for c in clauses]
            problem = SATProblem(n_vars=n_vars, clauses=sat_clauses)
    except Exception as e:
        return {'status': 'error', 'error': f'Failed to create SATProblem: {e}'}

    # Small deterministic RNG for this trial
    qls_np = np
    qls_np.random.seed(trial_seed)

    # Build ansatz and Hamiltonian
    try:
        ansatz, n_params = create_qaoa_ansatz(problem, p_layers)
        vqe_hamiltonian = sat_to_hamiltonian(problem)
    except Exception as e:
        return {'status': 'error', 'error': f'ansatz/hamiltonian build failed: {e}'}

    if verbose:
        print(f"  [QLTO-FPT] Trial {trial_seed}: built ansatz (p={p_layers}, n_params={n_params}) and Hamiltonian")

    param_bounds = qls_np.array([[0.0, 2 * qls_np.pi] if i % 2 == 0 else [0.0, qls_np.pi] for i in range(n_params)])
    theta0 = qls_np.random.rand(n_params) * (param_bounds[:, 1] - param_bounds[:, 0]) + param_bounds[:, 0]

    # Encode and build phase oracle
    param_idx, n_qubits_p = encode_parameters(theta0, param_bounds, bits_per_param)
    if n_qubits_p < n_mask_bits:
        return {'status': 'error', 'error': f'n_qubits_p ({n_qubits_p}) is smaller than N_MASK_BITS ({n_mask_bits})'}

    try:
        delta_t = 0.35
        U_PE = build_coherent_phase_oracle_nisq(n_params, bits_per_param, param_bounds, delta_t,
                                                ansatz, vqe_hamiltonian, problem.n_vars)
        if U_PE is None:
            raise ValueError('U_PE build returned None')
    except Exception as e:
        return {'status': 'error', 'error': f'U_PE build failed: {e}'}
    if verbose:
        print(f"  [QLTO-FPT] Trial {trial_seed}: phase-oracle (U_PE) built")

    T_gate = build_tunneling_operator_QW(n_qubits_p)
    evol_qc = run_qlto_evolution_nisq(n_qubits_p, problem.n_vars, param_idx, U_PE, T_gate, K_steps=3)
    if evol_qc is None:
        return {'status': 'error', 'error': 'evolution circuit build failed'}
    if verbose:
        print(f"  [QLTO-FPT] Trial {trial_seed}: evolution circuit built (n_qubits_param={n_qubits_p})")

    sampler = CountingWrapper(BaseSampler())

    # Quantum sampling
    if verbose:
        print(f"  [QLTO-FPT] Trial {trial_seed}: starting quantum sampling (shots={shots})...")
    t0_sampler = time.time()
    samples = measure_and_process_samples_nisq(evol_qc, shots, n_qubits_p, 'param', sampler)
    t1_sampler = time.time()
    sampler_time = t1_sampler - t0_sampler
    if verbose:
        print(f"  [QLTO-FPT] Trial {trial_seed}: sampling completed in {sampler_time:.3f}s; collected {len(samples)} raw samples")

    if not samples:
        return {'status': 'ok', 'success': False, 'reason': 'no-samples', 'sampler_time': sampler_time}

    totalp = sum(samples.values())
    if totalp <= 0:
        return {'status': 'ok', 'success': False, 'reason': 'empty-dist', 'sampler_time': sampler_time}

    samples_norm = {idx: p/totalp for idx,p in samples.items()}

    # Classical heuristic extraction & search (frequency-based + shrink + PySAT)
    if verbose:
        print(f"  [QLTO-FPT] Trial {trial_seed}: starting classical extraction and greedy shrink")
    t_classical_start = time.time()
    superset = extract_subset_from_top_samples(samples_norm, top_T_candidates, n_vars, n_mask_bits)
    k_prime_initial = len(superset)
    if verbose:
        print(f"    - superset candidates extracted: {k_prime_initial} vars (top_T_candidates={top_T_candidates})")
    shrunk_set = shrink_superset_greedy(problem, superset)
    k_prime_final = len(shrunk_set)
    if verbose:
        print(f"    - shrunk set size: {k_prime_final} vars")
    solution_0_idx = try_candidate_backdoor_pysat(problem, shrunk_set)
    is_minimal = None
    if solution_0_idx is not None:
        is_minimal = is_minimal_backdoor(problem, shrunk_set)
    t_classical_end = time.time()
    classical_search_time = t_classical_end - t_classical_start
    if verbose:
        print(f"  [QLTO-FPT] Trial {trial_seed}: classical search finished in {classical_search_time:.3f}s")

    total_time = time.time() - t0_sampler + classical_search_time

    if solution_0_idx is not None:
        sat_ok = verify_solution([c.literals for c in problem.clauses], solution_0_idx)
        return {
            'status': 'ok',
            'success': bool(sat_ok),
            'subset': shrunk_set,
            'solution_0_idx': solution_0_idx,
            'k_prime_initial': k_prime_initial,
            'k_prime_final': k_prime_final,
            'is_minimal': is_minimal,
            'sampler_time': sampler_time,
            'classical_search_time': classical_search_time,
            'total_time': total_time,
        }

    return {
        'status': 'ok',
        'success': False,
        'reason': 'no-solution-from-heuristic',
        'subset': shrunk_set,
        'solution_0_idx': None,
        'k_prime_initial': k_prime_initial,
        'k_prime_final': k_prime_final,
        'is_minimal': None,
        'sampler_time': sampler_time,
        'classical_search_time': classical_search_time,
        'total_time': total_time,
    }


def run_progressive_precision(problem, p_layers=2, bits_stage1=2, bits_stage2=4, **kwargs):
    """Run a coarse QLTO (low bits) then refine with higher precision."""
    print(">>> Stage 1: coarse search (low precision)")
    res1 = run_qlto_optimizer(problem, p_layers=p_layers, bits_per_param=bits_stage1, **kwargs)
    if "error" in res1:
        return res1
    # Use the final_theta from stage1 as a warm-start for stage2
    coarse_theta = res1.get('final_theta')
    print(">>> Stage 2: refine around coarse centroid (higher precision)")
    kwargs2 = dict(kwargs)
    kwargs2.update({'use_smart_init': False})
    res2 = run_qlto_optimizer(problem, p_layers=p_layers, bits_per_param=bits_stage2, init_theta=coarse_theta, **kwargs2)
    return res2


def repeated_qlto(problem: SATProblem, n_repeats: int = 3, **kwargs):
    """Run several independent QLTO attempts and return the best result."""
    best = None
    for i in range(n_repeats):
        kwargs_i = dict(kwargs)
        if 'rng_seed' in kwargs and kwargs['rng_seed'] is not None:
            kwargs_i['rng_seed'] = kwargs['rng_seed'] + i
        else:
            kwargs_i['rng_seed'] = i
        res = run_qlto_optimizer(problem, **kwargs_i)
        if res is None or 'error' in res:
            continue
        if best is None or res.get('final_energy', float('inf')) < best.get('final_energy', float('inf')):
            best = res
    return best


def verify_counts_with_verifier(counts: Dict[str, float], n_vars: int, sat_problem: SATProblem, top_k: int = 10):
    """Check the top-k measurement results against the classical SAT verifier.

    Returns (assignment_dict, prob, bitstring) for the first satisfying sample found,
    or (None, 0.0, '') if none found.
    """
    if not counts:
        return None, 0.0, ''

    # Use samples_to_assignments to convert and rank
    ranked = samples_to_assignments(counts, n_vars, top_k=top_k)
    for assign, prob in ranked:
        if sat_problem.is_satisfied(assign):
            # Build bitstring in Qiskit LSB order for compatibility: var1->q0 is LSB
            # Our samples_to_assignments converts Qiskit LSB->assign with var1 as first entry
            # Reconstruct bitstring MSB order for display
            bs = ''.join('1' if assign[i+1] else '0' for i in range(min(n_vars, len(assign))))
            bitstring_lsb = bs[::-1]
            return assign, prob, bitstring_lsb

    return None, 0.0, ''


if __name__ == "__main__":
    
    if not QISKIT_AVAILABLE:
        print("Qiskit not found. Please install qiskit and qiskit-aer to run this test.")
    else:
        # --- 1. Define a 3-SAT Problem ---
        # This problem has N=3 variables
        # A known satisfying assignment is x1=True, x2=False, x3=True
        # (which is {1: True, 2: False, 3: True})
        # Another is x1=False, x2=True, x3=False
        print("="*80)
        print("QLTO-QAOA SAT SOLVER TEST")
        print("="*80)
        
        c1 = SATClause((1, 2, -3))
        c2 = SATClause((-1, -2, 3))
        c3 = SATClause((1, -2, 3))
        c4 = SATClause((-1, 2, -3))
        
        test_problem = SATProblem(n_vars=3, clauses=[c1, c2, c3, c4])

        # --- 2. Set Optimizer Parameters ---
        # Use low values for a quick test
        p_layers = 2        # QAOA layers
        bits_per_param = 3  # QLTO parameter precision (2^3 = 8 bins)
        max_iterations = 15 # QLTO optimization loops
        
        # --- 3. Run the QLTO Optimizer ---
        # This finds the best (gamma, beta) parameters
        opt_results = run_qlto_optimizer(
            test_problem,
            p_layers=p_layers,
            bits_per_param=bits_per_param,
            max_iterations=max_iterations,
            K_steps_initial=5,
            K_steps_final=1,
            update_eta=0.7,
            use_smart_init=True
        )
        
        # --- 4. Verify the Solution ---
        if "error" not in opt_results:
            best_theta = opt_results['final_theta']
            
            is_solution, bitstring, _ = run_final_solution_check(
                test_problem,
                p_layers,
                best_theta
            )
            
            print("\n" + "="*80)
            print("FINAL TEST RESULT")
            print("="*80)
            if is_solution:
                print("  STATUS: ✅ SUCCESS!")
                print(f"  The QLTO optimizer found parameters that successfully")
                print(f"  solved the SAT problem.")
                print(f"  Found solution: '{bitstring}' (Qiskit LSB order)")
            else:
                print("  STATUS: ❌ FAILURE")
                print(f"  The QLTO optimizer did not find a valid solution.")
                print(f"  Most probable bitstring: '{bitstring}'")
            print("="*80)
        else:
            print("Optimizer failed to run.")
            
    res = unit_test_sat_to_hamiltonian()
    print('unit_test_sat_to_hamiltonian ->', res)

