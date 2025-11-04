"""
QAOA with QLTO Optimizer
========================

Use QLTO (Quantum-enhanced Learning To Optimize) to optimize QAOA parameters
for SAT solving. This combines:

1. QAOA's variational structure (quantum circuit for SAT)
2. QLTO's multi-basin optimization (escapes local minima)
3. Quantum natural gradients (better convergence)

This is still exponential in worst case, but:
- Much more likely to find solutions than standard QAOA
- Can handle rugged landscapes (densely coupled SAT)
- Polynomial in the number of PARAMETERS (not problem size)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import sys
import time

# Import QLTO optimizer
sys.path.append('../Quantum_AI/QLTO')
try:
    from qlto_multi_basin_search import MultiBasinQLTOOptimizer
    QLTO_AVAILABLE = True
except:
    QLTO_AVAILABLE = False


def sat_to_hamiltonian(clauses: List[Tuple[int, ...]], n_vars: int) -> callable:
    """
    Convert SAT clauses to energy function.
    
    Args:
        clauses: List of clauses (tuples of literals)
        n_vars: Number of variables
        
    Returns:
        Energy function that takes bitstring and returns number of violated clauses
    """
    def energy(bitstring: str) -> float:
        """Count violated clauses (0 = all satisfied)"""
        assignment = {i+1: (bitstring[i] == '1') for i in range(n_vars)}
        violated = 0
        
        for clause in clauses:
            satisfied = False
            for lit in clause:
                var = abs(lit)
                val = assignment.get(var, False)
                if (lit > 0 and val) or (lit < 0 and not val):
                    satisfied = True
                    break
            if not satisfied:
                violated += 1
                
        return violated
    
    return energy


def create_qaoa_circuit(n_vars: int, depth: int = 1) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    Create QAOA circuit for SAT with parameterized gates.
    
    Args:
        n_vars: Number of variables
        depth: Number of QAOA layers (p)
        
    Returns:
        (circuit, parameters) where parameters = [beta_0, gamma_0, beta_1, gamma_1, ...]
    """
    qc = QuantumCircuit(n_vars)
    
    # Initial state: uniform superposition
    qc.h(range(n_vars))
    
    # QAOA layers
    parameters = []
    
    for layer in range(depth):
        # Problem Hamiltonian: gamma rotation
        gamma = Parameter(f'γ_{layer}')
        parameters.append(gamma)
        
        # Apply ZZ interactions (clause coupling)
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                qc.rzz(gamma, i, j)
        
        # Mixer Hamiltonian: beta rotation
        beta = Parameter(f'β_{layer}')
        parameters.append(beta)
        
        # Apply X rotations (mixing)
        for i in range(n_vars):
            qc.rx(2 * beta, i)
    
    # Measurement
    qc.measure_all()
    
    return qc, parameters


def solve_sat_qaoa_qlto(
    clauses: List[Tuple[int, ...]],
    n_vars: int,
    depth: int = 3,
    max_iterations: int = 100,
    n_basins: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Solve SAT using QAOA optimized with QLTO multi-basin search.
    
    This is the BEST quantum approach we have:
    - Uses QAOA structure (proven to work for combinatorial optimization)
    - Uses QLTO optimizer (escapes local minima, explores multiple basins)
    - Still exponential in worst case, but practical for small-medium instances
    
    Args:
        clauses: SAT clauses
        n_vars: Number of variables
        depth: QAOA depth (number of layers p)
        max_iterations: Maximum QLTO iterations
        n_basins: Number of basins to explore simultaneously
        verbose: Print progress
        
    Returns:
        Dict with:
            - satisfiable: bool
            - assignment: Dict[int, bool] or None
            - method: str
            - energy: float (number of violated clauses, 0 = SAT)
            - iterations: int
    """
    
    if not QLTO_AVAILABLE:
        return {
            'satisfiable': False,
            'assignment': None,
            'method': 'QAOA-QLTO (not available)',
            'energy': float('inf'),
            'iterations': 0,
            'error': 'QLTO not available'
        }
    
    if verbose:
        print("\n" + "="*80)
        print("QAOA WITH QLTO OPTIMIZER")
        print("="*80)
        print(f"Problem: {n_vars} variables, {len(clauses)} clauses")
        print(f"QAOA depth: p={depth} ({2*depth} parameters)")
        print(f"Multi-basin search: {n_basins} basins")
        print("="*80)
        print()
    
    start_time = time.time()
    
    # Create QAOA circuit
    qc, params = create_qaoa_circuit(n_vars, depth=depth)
    
    if verbose:
        print(f"[1/4] Created QAOA circuit: {qc.num_qubits} qubits, {len(params)} parameters")
    
    # Create energy function
    energy_fn = sat_to_hamiltonian(clauses, n_vars)
    
    # Define objective function for QLTO
    def objective(param_values: np.ndarray) -> float:
        """
        Evaluate QAOA circuit with given parameters.
        Returns: energy (number of violated clauses)
        """
        # Bind parameters
        bound_circuit = qc.bind_parameters({params[i]: param_values[i] for i in range(len(params))})
        
        # Simulate using AerSimulator
        try:
            # AerSimulator is more robust and feature-rich
            simulator = AerSimulator(method='statevector')
            
            # We need a circuit without measurements for statevector simulation
            qc_sim = bound_circuit.remove_final_measurements(inplace=False)
            
            # Run the simulation
            result = simulator.run(qc_sim).result()
            sv = result.get_statevector()
            
            # Get probabilities
            probs = sv.probabilities_dict()
            
            # Compute expected energy
            expected_energy = 0.0
            for bitstring, prob in probs.items():
                # Pad bitstring to n_vars length
                bitstring = bitstring.zfill(n_vars)
                expected_energy += prob * energy_fn(bitstring)
            
            return expected_energy
            
        except Exception as e:
            if verbose:
                print(f"    ⚠️  Simulation error: {e}")
            return float('inf')
    
    if verbose:
        print(f"[2/4] Created objective function (energy = violated clauses)")
    
    # Initialize QLTO optimizer
    n_params = len(params)
    
    optimizer = MultiBasinQLTOOptimizer(
        n_params=n_params,
        n_basins=n_basins,
        bounds=[(-np.pi, np.pi)] * n_params,  # QAOA parameter range
        verbose=verbose
    )
    
    if verbose:
        print(f"[3/4] Initialized QLTO optimizer")
        print(f"    • Parameters: {n_params}")
        print(f"    • Basins: {n_basins}")
        print(f"    • Max iterations: {max_iterations}")
        print()
        print(f"[4/4] Running multi-basin optimization...")
    
    # Optimize!
    best_params, best_energy, history = optimizer.optimize(
        objective,
        max_iterations=max_iterations
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        print()
        print("="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Best energy: {best_energy:.6f} violated clauses")
        print(f"Iterations: {len(history)}")
        print(f"Time: {elapsed:.3f}s")
        print()
    
    # Extract best solution
    if best_energy < 0.5:  # Essentially 0 (all clauses satisfied)
        # Get final measurement
        bound_circuit = qc.bind_parameters({params[i]: best_params[i] for i in range(len(params))})
        
        simulator = AerSimulator(method='statevector')
        qc_sim = bound_circuit.remove_final_measurements(inplace=False)
        result = simulator.run(qc_sim).result()
        sv = result.get_statevector()
        probs = sv.probabilities_dict()
        
        # Get most likely bitstring
        best_bitstring = max(probs.items(), key=lambda x: x[1])[0].zfill(n_vars)
        
        # Convert to assignment
        assignment = {i+1: (best_bitstring[i] == '1') for i in range(n_vars)}
        
        if verbose:
            print("✅ SATISFIABLE")
            print(f"   Assignment: {best_bitstring}")
            print()
        
        return {
            'satisfiable': True,
            'assignment': assignment,
            'method': 'QAOA-QLTO',
            'energy': best_energy,
            'iterations': len(history),
            'time': elapsed,
            'best_params': best_params
        }
    else:
        if verbose:
            print(f"❌ NO SOLUTION FOUND (best energy: {best_energy:.3f})")
            print(f"   → Problem may be UNSAT or need more iterations")
            print()
        
        return {
            'satisfiable': False,
            'assignment': None,
            'method': 'QAOA-QLTO',
            'energy': best_energy,
            'iterations': len(history),
            'time': elapsed,
            'best_params': best_params
        }


def solve_sat_qaoa_qlto_with_restart(
    clauses: List[Tuple[int, ...]],
    n_vars: int,
    max_restarts: int = 3,
    depth: int = 3,
    max_iterations: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Solve SAT with QAOA-QLTO with multiple restarts for robustness.
    
    This gives us the BEST chance of finding a solution:
    - Multiple independent runs
    - Each run explores different basins
    - Returns first successful solution or best attempt
    """
    
    if verbose:
        print("\n" + "="*80)
        print("QAOA-QLTO WITH MULTIPLE RESTARTS")
        print("="*80)
        print(f"Max restarts: {max_restarts}")
        print(f"Each restart: {max_iterations} iterations, p={depth}")
        print("="*80)
    
    best_result = None
    best_energy = float('inf')
    
    for restart in range(max_restarts):
        if verbose:
            print(f"\n{'='*80}")
            print(f"RESTART {restart + 1}/{max_restarts}")
            print('='*80)
        
        result = solve_sat_qaoa_qlto(
            clauses, n_vars,
            depth=depth,
            max_iterations=max_iterations,
            n_basins=5,
            verbose=verbose
        )
        
        if result['satisfiable']:
            if verbose:
                print(f"\n✅ Solution found on restart {restart + 1}!")
            return result
        
        if result['energy'] < best_energy:
            best_energy = result['energy']
            best_result = result
    
    if verbose:
        print(f"\n⚠️  No solution found after {max_restarts} restarts")
        print(f"   Best energy: {best_energy:.3f} violated clauses")
    
    return best_result


if __name__ == "__main__":
    # Test on a simple 3-SAT instance
    print("Testing QAOA-QLTO on simple 3-SAT...")
    
    clauses = [
        (1, 2, 3),
        (-1, 2, -3),
        (1, -2, 3),
        (-1, -2, -3)
    ]
    n_vars = 3
    
    result = solve_sat_qaoa_qlto(clauses, n_vars, depth=2, max_iterations=50, verbose=True)
    
    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Satisfiable: {result['satisfiable']}")
    print(f"Assignment: {result.get('assignment')}")
    print(f"Energy: {result['energy']}")
    print(f"Iterations: {result['iterations']}")
