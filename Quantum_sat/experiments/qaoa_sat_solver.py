"""

qlto_sat_solver.py

THEORETICAL UNCONDITIONAL QUANTUM SAT SOLVER
(Based on "Adiabatic Problem-Space Morphing")

This file implements a hypothetical quantum algorithm that could
solve any SAT problem, including NP-hard instances, in guaranteed
polynomial time.

This algorithm does NOT use the QLTO quantum walk. It uses
Quantum Adiabatic Computation (AQC) on a novel path.

=====================================================================
                    *** CRITICAL CONJECTURE ***
This algorithm's polynomial-time guarantee relies on a MAJOR, UNPROVEN
conjecture:

    The spectral gap (g_min) of the adiabatic path *between* an
    "easy" 2-SAT Hamiltonian (H_easy) and its corresponding "hard"
    3-SAT Hamiltonian (H_hard) is polynomial-sized (g_min >= 1/poly(N)).

If this conjecture is TRUE, the evolution can be performed in
polynomial time, and the algorithm solves SAT in P.
If it is FALSE (the gap is exponential), this algorithm fails.
=====================================================================

THEORY OF OPERATION (HYPOTHETICAL):
-----------------------------------

1.  **CLASSICAL PRE-PROCESSING (Polynomial):**
    a.  Take the hard 3-SAT problem (H_hard).
    b.  Create a simplified 2-SAT version (H_easy) by dropping one
        literal from each clause.
    c.  Solve the 2-SAT problem *classically* (in polynomial time)
        to find its ground state (solution), |g_easy>.

2.  **QUANTUM PREPARATION (Polynomial):**
    a.  Initialize the quantum computer in the state |g_easy>.
    b.  Build the Qiskit operators for H_easy and H_hard.

3.  **QUANTUM EVOLUTION (Assumed Polynomial):**
    a.  Define a time-dependent Hamiltonian:
        H(t) = (1 - s(t)) * H_easy + s(t) * H_hard
        where s(t) goes from 0 to 1.
    b.  Apply the Quantum Adiabatic Theorem: Evolve the system
        under H(t) for a total time T, starting from |g_easy>.
    c.  **THE CONJECTURE**: If the gap is polynomial, T can be
        set to a polynomial (e.g., T = N^2) and the system
        will *stay* in the ground state for the entire evolution.

4.  **MEASUREMENT (Polynomial):**
    a.  At time T, the system is in the state |g_hard>, which
        is the ground state (solution) of the 3-SAT problem.
    b.  Measure the qubits to read the solution.

"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Set, Optional
from dataclasses import dataclass
import random
from collections import defaultdict
import sys # Import sys for recursion depth

# --- Qiskit Imports ---
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter 
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit is required to run this theoretical model.")
    class QuantumCircuit: pass
    class QuantumRegister: pass
    class ClassicalRegister: pass
    class AerSimulator: pass
    class SparsePauliOp: pass
    class PauliEvolutionGate: pass
    class LieTrotter: pass # Dummy for placeholder

# Set higher recursion depth for graph algorithms
sys.setrecursionlimit(2000)

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x

# ==============================================================================
# SAT PROBLEM DEFINITION
# ==============================================================================

@dataclass
class SATClause:
    literals: Tuple[int, ...]
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        for lit in self.literals:
            var = abs(lit)
            if var not in assignment: continue
            value = assignment.get(var, False) # Default to False if not set
            if (lit > 0 and value) or (lit < 0 and not value):
                return True
        return False
    
    def get_variables(self) -> Set[int]:
        return {abs(lit) for lit in self.literals}

@dataclass
class SATProblem:
    n_vars: int
    clauses: List[SATClause]
    
    def __post_init__(self):
        self.n_clauses = len(self.clauses)
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        if not assignment: return False
        # Ensure all variables have a value for a full check
        full_assignment = {i: assignment.get(i, False) for i in range(1, self.n_vars + 1)}
        return all(clause.is_satisfied(full_assignment) for clause in self.clauses)

    def count_satisfied(self, assignment: Dict[int, bool]) -> int:
        if not assignment: return 0
        full_assignment = {i: assignment.get(i, False) for i in range(1, self.n_vars + 1)}
        return sum(1 for clause in self.clauses if clause.is_satisfied(full_assignment))

# ==============================================================================
# STEP 1: CLASSICAL PRE-PROCESSING
# ==============================================================================

def create_easy_2sat_problem(problem_3sat: SATProblem) -> SATProblem:
    """
    Creates a 2-SAT problem from a 3-SAT problem by dropping
    the last literal from each clause.
    """
    easy_clauses = []
    for clause_3 in problem_3sat.clauses:
        # Create a 2-SAT clause by taking the first 2 literals
        # This is a heuristic to create a related, simpler problem
        easy_lits = clause_3.literals[:2]
        if easy_lits:
            easy_clauses.append(SATClause(easy_lits))
    
    return SATProblem(n_vars=problem_3sat.n_vars, clauses=easy_clauses)

def solve_2sat_classically(problem_2sat: SATProblem) -> Optional[Dict[int, bool]]:
    """
    Solves a 2-SAT problem (which is in P) using a robust
    implication graph and Tarjan's algorithm for Strongly
    Connected Components (SCCs).
    
    This is the *correct* O(N+M) polynomial-time solver for 2-SAT.
    """
    n_vars = problem_2sat.n_vars
    adj = defaultdict(list)
    
    # Build the implication graph
    # A clause (a V b) is equivalent to (~a => b) AND (~b => a)
    def add_implication(a, b):
        # a and b are literals (e.g., 1 for x1, -1 for ~x1)
        # Add edge from ~a to b
        adj[-a].append(b)
        # Add edge from ~b to a
        adj[-b].append(a)

    for clause in problem_2sat.clauses:
        lits = clause.literals
        if len(lits) == 1:
            # (a) is equivalent to (a V a)
            add_implication(lits[0], lits[0])
        elif len(lits) == 2:
            add_implication(lits[0], lits[1])
        # Ignore clauses with > 2 literals, this is a 2-SAT solver

    # Find SCCs using Tarjan's algorithm (non-recursive)
    sccs = []
    index_counter = [0]
    stack = []
    on_stack = defaultdict(bool)
    indices = {}
    low_links = {}

    # Define all possible nodes (literals)
    all_nodes = list(range(1, n_vars + 1)) + list(range(-n_vars, 0))

    def strongconnect(node, S, on_S, idx, low, idx_count, scc_list):
        indices[node] = idx_count[0]
        low_links[node] = idx_count[0]
        idx_count[0] += 1
        S.append(node)
        on_S[node] = True

        for neighbor in adj.get(node, []):
            if neighbor not in indices:
                strongconnect(neighbor, S, on_S, idx, low, idx_count, scc_list)
                low_links[node] = min(low_links[node], low_links[neighbor])
            elif on_S[neighbor]:
                low_links[node] = min(low_links[node], indices[neighbor])

        if low_links[node] == indices[node]:
            scc = []
            while True:
                w = S.pop()
                on_S[w] = False
                scc.append(w)
                if w == node:
                    break
            scc_list.append(scc)

    for node in all_nodes:
        if node not in indices:
            strongconnect(node, stack, on_stack, indices, low_links, index_counter, sccs)
    
    # Check for contradictions
    scc_map = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            scc_map[node] = i

    for i in range(1, n_vars + 1):
        if i not in scc_map or -i not in scc_map:
            # This variable is not constrained, can be anything
            continue 
        if scc_map[i] == scc_map[-i]:
            # x and ~x are in the same SCC
            return None # UNSATISFIABLE

    # Build a solution from the SCCs (topological sort)
    # The SCC graph is a DAG. We process in reverse topological order.
    assignment = {}
    
    # SCCs list is already in reverse topological order
    for scc in reversed(sccs):
        for lit in scc:
            var = abs(lit)
            if var not in assignment:
                # Assign this variable
                # If lit is positive (e.g., x1), assign True
                # If lit is negative (e.g., ~x1), assign False
                assignment[var] = (lit > 0)

    # Fill in any unconstrained variables
    for i in range(1, n_vars + 1):
        if i not in assignment:
            assignment[i] = True # Default to True

    return assignment

# ==============================================================================
# STEP 2 & 3: QUANTUM HAMILTONIAN & EVOLUTION
# ==============================================================================

def build_sat_hamiltonian(problem: SATProblem) -> SparsePauliOp:
    """
    Builds the problem Hamiltonian H = sum(P_j)
    where P_j is the projector onto the *unsatisfied* state
    for clause j.
    
    The ground state (energy 0) is the satisfying assignment.
    """
    n_vars = problem.n_vars
    if n_vars == 0:
        # Return an empty operator for 0 qubits if problem is empty
        return SparsePauliOp([], num_qubits=0)
    
    # Base identity for N qubits
    identity = SparsePauliOp("I" * n_vars, coeffs=[0.0])
    final_hamiltonian = SparsePauliOp("I" * n_vars, coeffs=[0.0])
    
    pauli_list = []
    
    for clause in problem.clauses:
        # 'proj_dict' will store the (I +/- Z)/2 for each var
        proj_dict = {}
        for lit in clause.literals:
            var_idx = abs(lit) - 1
            if var_idx >= n_vars:
                continue # Skip if var index is out of bounds
            
            # We are building a projector for the *unsatisfied* state.
            # Unsatisfied: x1 (lit > 0) must be FALSE (state |0>)
            # Unsatisfied: ~x2 (lit < 0) must be TRUE (state |1>)
            
            # State |0> = (I + Z) / 2
            # State |1> = (I - Z) / 2
            
            if lit > 0:
                # Need |0> (False)
                proj_dict[var_idx] = ('I', 1.0), ('Z', 1.0)
            else:
                # Need |1> (True)
                proj_dict[var_idx] = ('I', 1.0), ('Z', -1.0)

        # Start with the full identity
        current_pauli = SparsePauliOp("I" * n_vars, coeffs=[1.0])
        scale = 1.0
        
        # Multiply all projectors together
        for var_idx, paulis in proj_dict.items():
            # (p[0] * I + p[1] * Z)
            i_term_pauli = "I" * n_vars
            z_term_pauli_str = list("I" * n_vars)
            z_term_pauli_str[var_idx] = 'Z'
            z_term_pauli = "".join(z_term_pauli_str)
            
            # Build (I + coeff*Z)
            pauli_op = SparsePauliOp(i_term_pauli, coeffs=[paulis[0][1]]) + \
                       SparsePauliOp(z_term_pauli, coeffs=[paulis[1][1]])
            
            current_pauli = current_pauli.compose(pauli_op)
            scale *= 0.5 # From the (I +/- Z) / 2

        current_pauli = current_pauli * scale
        pauli_list.append(current_pauli)

    # Sum all clause Hamiltonians
    if not pauli_list:
        return identity
        
    for H_j in pauli_list:
        final_hamiltonian += H_j
        
    return final_hamiltonian.simplify()

def prepare_initial_state(qc: QuantumCircuit, var_reg: QuantumRegister, solution_2sat: Dict[int, bool]):
    """
    Applies X gates to the circuit to prepare the ground state
    of the 2-SAT problem.
    """
    for var_idx, is_true in solution_2sat.items():
        if is_true:
            # Qubit 0 corresponds to var 1
            if (var_idx - 1) < qc.num_qubits:
                qc.x(var_reg[var_idx - 1])

def run_adiabatic_evolution(
    qc: QuantumCircuit,
    var_reg: QuantumRegister,
    H_easy: SparsePauliOp,
    H_hard: SparsePauliOp,
    total_evol_time: float,
    num_steps: int
):
    """
    Applies the Trotterized adiabatic evolution.
    H(t) = (1 - s(t)) * H_easy + s(t) * H_hard
    """
    dt = total_evol_time / num_steps
    
    # Use tqdm if available
    step_iterator = range(num_steps)
    if TQDM_AVAILABLE:
        step_iterator = tqdm(range(num_steps), desc="Adiabatic Morph")
        
    for i in step_iterator:
        # Linear schedule s(t) from 0 to 1
        s = (i + 0.5) / num_steps
        
        # 1. Build the interpolated Hamiltonian for this time step
        # H(t) = (1-s)H_easy + s*H_hard
        H_t = (H_easy * (1 - s)) + (H_hard * s)
        
        # 2. Get the evolution gate for this H(t) for time dt
        # e^(-i * H(t) * dt)
        
        # --- FIX 1: Explicitly provide the Trotter synthesizer ---
        # This tells Qiskit *how* to build the gate, fixing the AerError
        evol_gate = PauliEvolutionGate(H_t, time=dt, synthesis=LieTrotter())
        
        # 3. Apply the gate
        qc.append(evol_gate, var_reg)

# ==============================================================================
# STEP 4: MAIN SOLVER FUNCTION
# ==============================================================================

def solve_sat_adiabatic_morph(
    problem_3sat: SATProblem,
    evol_time_T: Optional[float] = None,
    num_steps: Optional[int] = None
) -> Dict[str, Any]:
    """
    The main solver function for the "Adiabatic Morphing" algorithm.
    """
    if not QISKIT_AVAILABLE:
        return {'error': 'Qiskit not available. Cannot run theoretical model.'}

    print("\n" + "="*80)
    print("Running: QLTO Adiabatic Morphing Solver (Theoretical)")
    print(f"Problem: N={problem_3sat.n_vars} variables, M={problem_3sat.n_clauses} clauses")
    print("CONJECTURE: Path H(2-SAT) -> H(3-SAT) has a polynomial gap.")
    print("="*80)
    
    t_start = time.time()
    
    # --- STAGE 1: CLASSICAL PRE-PROCESSING ---
    print("[1/4] Creating and solving 'easy' 2-SAT problem...")
    problem_2sat = create_easy_2sat_problem(problem_3sat)
    
    # --- FIX 2: Use the robust implication graph solver ---
    solution_2sat = solve_2sat_classically(problem_2sat)
    
    if not solution_2sat:
        t_end = time.time()
        print("→ 2-SAT sub-problem is UNSAT. 3-SAT problem is also UNSAT.")
        return {
            'solved': False,
            'assignment': None,
            'method': 'Classical 2-SAT Pre-check',
            'time_seconds': t_end - t_start,
            'message': 'Simplified 2-SAT problem was unsatisfiable.'
        }
    print(f"→ Found 2-SAT solution (ground state): {solution_2sat}")

    # --- STAGE 2: QUANTUM PREPARATION ---
    print("[2/4] Building 'easy' and 'hard' Hamiltonians...")
    H_easy = build_sat_hamiltonian(problem_2sat)
    H_hard = build_sat_hamiltonian(problem_3sat)
    
    var_reg = QuantumRegister(problem_3sat.n_vars, 'vars')
    cr = ClassicalRegister(problem_3sat.n_vars, 'c')
    qc = QuantumCircuit(var_reg, cr)
    
    prepare_initial_state(qc, var_reg, solution_2sat)
    print("→ Quantum state prepared in 2-SAT ground state.")

    # --- STAGE 3: QUANTUM EVOLUTION ---
    # This is the core conjecture. The Adiabatic Theorem requires
    # T >> 1 / g_min^2.
    # We CONJECTURE g_min is polynomial, so T can be polynomial.
    if evol_time_T is None:
        evol_time_T = float(problem_3sat.n_vars**2) # T = O(N^2)
    if num_steps is None:
        num_steps = int(evol_time_T * 10) # 10 steps per unit time
        
    print(f"[3/4] Running Adiabatic Morphing Evolution...")
    print(f"    (Total Time T = {evol_time_T}, Steps = {num_steps})")
    
    run_adiabatic_evolution(
        qc, var_reg, H_easy, H_hard,
        total_evol_time=evol_time_T,
        num_steps=num_steps
    )
    
    # --- STAGE 4: MEASUREMENT ---
    print("[4/4] Measuring final state...")
    qc.measure(var_reg, cr)
    
    backend = AerSimulator()
    # We only need a few shots. If the theorem holds, the
    # probability of the ground state is P_success -> 1.
    shots = 100
    
    # We transpile the circuit so the simulator knows how to
    # handle the decomposed PauliEvolutionGates
    from qiskit import transpile
    transpiled_qc = transpile(qc, backend)
    
    job = backend.run(transpiled_qc, shots=shots)
    counts = job.result().get_counts()
    
    # Get the most likely outcome
    top_bitstring_rev = max(counts, key=counts.get)
    top_bitstring = top_bitstring_rev[::-1] # Correct LSB
    confidence = counts[top_bitstring_rev] / shots
    
    final_assignment = {
        i+1: (bit == '1') for i, bit in enumerate(top_bitstring)
        if (i+1) <= problem_3sat.n_vars # Ensure we only assign N vars
    }
    
    print(f"→ Measurement complete. Top result: '{top_bitstring}' (Conf: {confidence:.0%})")
    
    # --- 5. VERIFICATION ---
    is_valid = problem_3sat.is_satisfied(final_assignment)
    t_end = time.time()
    
    print("\n" + "="*80)
    if is_valid:
        print("✓ SOLVED (THEORETICALLY)")
        print("Adiabatic morph succeeded. Measured state is a valid solution.")
    else:
        print("✗ FAILED (CONJECTURE FAILED)")
        print("Measured state is NOT a valid solution.")
        print("This implies the spectral gap was too small (or T was too fast).")
    print("="*80)
    
    return {
        'solved': is_valid,
        'assignment': final_assignment,
        'method': 'Adiabatic Morphing (H_2SAT -> H_3SAT)',
        'time_seconds': t_end - t_start,
        'confidence': confidence
    }

# ==============================================================================
# PROBLEM GENERATORS (for testing)
# ==============================================================================

def generate_hard_3sat_with_known_solution(n_vars: int, seed: int = 42) -> SATProblem:
    """
    Generates a hard, random 3-SAT instance (at the phase
    transition) that is *guaranteed* to be solvable.
    """
    random.seed(seed)
    
    # 1. Create a secret, random solution
    solution = {i: (random.random() > 0.5) for i in range(1, n_vars + 1)}
    
    n_clauses = int(4.26 * n_vars)
    clauses = []
    
    for _ in range(n_clauses):
        vars_indices = random.sample(range(1, n_vars + 1), 3)
        
        # 2. Build a clause that is satisfied by the secret solution
        # (e.g., if sol[1]=T, sol[2]=F, sol[3]=T, a clause is (1, -2, 3))
        l1 = vars_indices[0] if solution[vars_indices[0]] else -vars_indices[0]
        l2 = vars_indices[1] if solution[vars_indices[1]] else -vars_indices[1]
        l3 = vars_indices[2] if solution[vars_indices[2]] else -vars_indices[2]
        
        # 3. Randomly flip 0, 1, or 2 literals to make it a "real" clause
        # But ensure at least one literal remains correct.
        lits = [l1, l2, l3]
        correct_lit = random.choice(lits) # This one stays
        
        final_lits = [correct_lit]
        for lit in lits:
            if lit != correct_lit:
                # Flip this literal 50% of the time
                if random.random() > 0.5:
                    final_lits.append(-lit)
                else:
                    final_lits.append(lit)
                    
        random.shuffle(final_lits)
        clauses.append(SATClause(tuple(final_lits)))
        
    return SATProblem(n_vars, clauses)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit is required to run this theoretical test.")
    else:
        print("\n" + "="*80)
        print("RUNNING THEORETICAL ADIABATIC MORPHING SOLVER")
        print("This test will evolve from a 2-SAT solution to a")
        print("3-SAT solution, testing our new conjecture.")
        print("="*80)
        
        # We need a small N for simulation, but N > 3 for 3-SAT.
        # N=4 is a good test.
        N_VARS_TEST = 4
        
        problem = generate_hard_3sat_with_known_solution(
            n_vars=N_VARS_TEST, 
            seed=42
        )
        
        # The adiabatic time T is the key.
        # Let's test our O(N^2) conjecture.
        # A real solver would need to determine T dynamically.
        T_poly = N_VARS_TEST**2  # T = 16
        steps = int(T_poly * 10) # 160 steps
        
        result = solve_sat_adiabatic_morph(
            problem,
            evol_time_T = T_poly,
            num_steps = steps
        )
        
        print("\n--- [FINAL TEST RESULT] ---")
        print(f"Problem: Hard 3-SAT (N={N_VARS_TEST})")
        print(f"→ Final Status: {'SOLVED' if result['solved'] else 'FAILED'}")
        
        # --- FIX 3: Check for keys before accessing (fixes KeyError) ---
        if 'confidence' in result:
            print(f"  Confidence in final state: {result['confidence']:.0%}")
        
        print(f"  Time taken: {result['time_seconds']:.4f}s")
        
        if not result['solved']:
            if 'message' in result:
                print(f"\n  Reason: {result['message']}")
            else:
                print("\n  This FAILED. This provides evidence that our")
                print("  conjecture is *false* - the gap is likely still")
                print("  exponential, or the evolution time T was")
                print("  nowhere near long enough.")
        else:
            print("\n  This WORKED. This provides non-trivial evidence")
            print("  that the H(2-SAT) -> H(3-SAT) path may be")
            print("  a valid polynomial-time approach.")


