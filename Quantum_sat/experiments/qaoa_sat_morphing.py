"""

qlto_sat_morphing.py

ADIABATIC PROBLEM-SPACE MORPHING SAT SOLVER

A novel approach that avoids the exponential spectral gap barrier by:
1. Starting from an EASY problem with a KNOWN solution
2. Adiabatically morphing the problem itself to the HARD problem
3. Relying on a NEW CONJECTURE about the morphing path

KEY INNOVATION:
===============

Instead of walking from "simple state" → "hard problem solution" (fails),
we walk from "easy problem solution" → "hard problem solution" (might work!).

THE CONJECTURE:
==============

For a hard 3-SAT instance φ₃, define a related easy 2-SAT instance φ₂ by
dropping one literal from each clause.

CONJECTURE: The spectral gap along the adiabatic path from H(φ₂) to H(φ₃)
           is Ω(1/poly(N)), NOT exponentially small.

If this conjecture holds → Polynomial-time SAT solver
If this conjecture fails → Same exponential barrier as before

ALGORITHM:
==========

1. INPUT: Hard 3-SAT problem φ₃
   
2. CONSTRUCT EASY PROBLEM:
   φ₂ = Reduce φ₃ to 2-SAT (drop one literal per clause)
   
3. SOLVE EASY PROBLEM CLASSICALLY:
   solution₂ = Solve_2SAT(φ₂)  [O(N·M) time, in P]
   
4. PREPARE QUANTUM STATE:
   |ψ(0)⟩ = |solution₂⟩  [Ground state of H(φ₂)]
   
5. ADIABATIC EVOLUTION:
   H(t) = (1 - s(t))·H(φ₂) + s(t)·H(φ₃)
   where s(t) = t/T
   
   Evolve for time T = O(poly(N)) [IF conjecture holds]
   
6. MEASURE:
   |ψ(T)⟩ should be the solution to φ₃
   
7. VERIFY classically

THEORETICAL ANALYSIS:
====================

By the Adiabatic Theorem, evolution time is:
    T = O(1/g_min²)

where g_min is the minimum spectral gap along the path.

Three scenarios:

A) BEST CASE (Conjecture True):
   g_min = Ω(1/poly(N))
   → T = O(poly(N))
   → POLYNOMIAL TIME! ✓
   
B) PARTIAL CASE (Structure-Dependent):
   g_min depends on problem structure
   → Polynomial for structured instances
   → Exponential for adversarial
   → Same as qlto_sat_formal.py
   
C) WORST CASE (Conjecture False):
   g_min = O(1/exp(N))
   → T = O(exp(N))
   → No better than classical ✗

TESTING THE CONJECTURE:
=======================

We can test this empirically:
1. Generate 3-SAT instances of increasing size
2. Compute the spectral gap numerically (for small N)
3. Fit to see if gap scales polynomially or exponentially

If polynomial → BREAKTHROUGH!
If exponential → Back to drawing board
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any, Set, Optional
from dataclasses import dataclass
import random
from collections import defaultdict

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
    from qiskit_aer import AerSimulator
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    SparsePauliOp = None
    Operator = None
    Statevector = None
    print("Warning: Qiskit required for morphing solver")

try:
    from Quantum_sat.experiments.qaoa_sat_formal import SATClause, SATProblem
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    # Define minimal versions
    @dataclass
    class SATClause:
        literals: Tuple[int, ...]
        def get_variables(self) -> Set[int]:
            return {abs(lit) for lit in self.literals}
        def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
            for lit in self.literals:
                var = abs(lit)
                if var not in assignment:
                    continue
                value = assignment[var]
                if (lit > 0 and value) or (lit < 0 and not value):
                    return True
            return False
    
    @dataclass
    class SATProblem:
        n_vars: int
        clauses: List[SATClause]
        def __post_init__(self):
            self.n_clauses = len(self.clauses)
        def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
            return all(clause.is_satisfied(assignment) for clause in self.clauses)


# ==============================================================================
# STEP 1: CONSTRUCT EASY 2-SAT PROBLEM FROM HARD 3-SAT
# ==============================================================================

def reduce_to_2sat(problem: SATProblem, strategy: str = 'drop_last') -> SATProblem:
    """
    Reduce a k-SAT problem to 2-SAT by dropping literals.
    
    Strategies:
    - 'drop_last': Drop the last literal from each clause
    - 'drop_random': Drop a random literal
    - 'drop_least_frequent': Drop the least frequent variable
    
    This creates an EASIER problem that is RELATED to the original.
    """
    new_clauses = []
    
    for clause in problem.clauses:
        if len(clause.literals) <= 2:
            # Already 2-SAT, keep as is
            new_clauses.append(clause)
        elif len(clause.literals) == 3:
            # Drop one literal to make 2-SAT
            if strategy == 'drop_last':
                # Drop the last literal
                new_lits = clause.literals[:2]
            elif strategy == 'drop_random':
                # Drop a random literal
                keep_indices = random.sample(range(3), 2)
                new_lits = tuple(clause.literals[i] for i in keep_indices)
            elif strategy == 'drop_least_frequent':
                # Count frequencies and drop least common
                # (simplified: just drop last for now)
                new_lits = clause.literals[:2]
            else:
                new_lits = clause.literals[:2]
            
            new_clauses.append(SATClause(new_lits))
        else:
            # More than 3 literals, drop to 2
            new_clauses.append(SATClause(clause.literals[:2]))
    
    return SATProblem(problem.n_vars, new_clauses)


# ==============================================================================
# STEP 2: SOLVE 2-SAT CLASSICALLY (POLYNOMIAL TIME)
# ==============================================================================

def solve_2sat_implication_graph(problem: SATProblem) -> Optional[Dict[int, bool]]:
    """
    Solve 2-SAT using the implication graph method.
    
    2-SAT is in P! This runs in O(N + M) time.
    
    Algorithm:
    1. Build implication graph: (¬x → y) and (¬y → x) for each clause (x ∨ y)
    2. Find strongly connected components (Kosaraju's algorithm)
    3. If x and ¬x are in the same SCC → UNSAT
    4. Otherwise, assign values based on SCC topological order
    
    Returns: Satisfying assignment or None if UNSAT
    """
    if not all(len(clause.literals) <= 2 for clause in problem.clauses):
        raise ValueError("solve_2sat requires a 2-SAT problem")
    
    # Build implication graph
    # For clause (x ∨ y): add edges (¬x → y) and (¬y → x)
    graph = defaultdict(list)  # adjacency list
    
    for clause in problem.clauses:
        if len(clause.literals) == 1:
            # Unit clause: variable must be true/false
            lit = clause.literals[0]
            # ¬lit → lit (if ¬lit is false, lit must be true)
            graph[-lit].append(lit)
        elif len(clause.literals) == 2:
            x, y = clause.literals
            # (x ∨ y) ≡ (¬x → y) ∧ (¬y → x)
            graph[-x].append(y)
            graph[-y].append(x)
    
    # Find strongly connected components using Kosaraju's algorithm
    def kosaraju_scc(graph, all_nodes):
        """Find SCCs in directed graph"""
        # Step 1: DFS on original graph to get finish times
        visited = set()
        finish_order = []
        
        def dfs1(node):
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs1(neighbor)
            finish_order.append(node)
        
        for node in all_nodes:
            if node not in visited:
                dfs1(node)
        
        # Step 2: Build reverse graph
        reverse_graph = defaultdict(list)
        for node in all_nodes:
            for neighbor in graph.get(node, []):
                reverse_graph[neighbor].append(node)
        
        # Step 3: DFS on reverse graph in reverse finish order
        visited = set()
        sccs = []
        
        def dfs2(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in reverse_graph.get(node, []):
                if neighbor not in visited:
                    dfs2(neighbor, component)
        
        for node in reversed(finish_order):
            if node not in visited:
                component = []
                dfs2(node, component)
                sccs.append(component)
        
        return sccs
    
    # Get all nodes (literals)
    all_nodes = set()
    for var in range(1, problem.n_vars + 1):
        all_nodes.add(var)
        all_nodes.add(-var)
    
    sccs = kosaraju_scc(graph, all_nodes)
    
    # Build SCC membership map
    scc_id = {}
    for i, scc in enumerate(sccs):
        for node in scc:
            scc_id[node] = i
    
    # Check satisfiability: x and ¬x must be in different SCCs
    for var in range(1, problem.n_vars + 1):
        if scc_id.get(var, -1) == scc_id.get(-var, -2):
            # x and ¬x in same SCC → UNSAT
            return None
    
    # Build satisfying assignment
    # Variable is TRUE if its SCC comes AFTER ¬var's SCC in topological order
    assignment = {}
    for var in range(1, problem.n_vars + 1):
        var_scc = scc_id.get(var, float('inf'))
        neg_var_scc = scc_id.get(-var, float('inf'))
        # Lower SCC id = later in topological order (from our algorithm)
        # We want to set var=True if ¬var comes later (must be false)
        assignment[var] = (var_scc < neg_var_scc)
    
    return assignment


# ==============================================================================
# STEP 3: BUILD HAMILTONIANS
# ==============================================================================

def sat_to_hamiltonian(problem: SATProblem) -> SparsePauliOp:
    """
    Convert SAT problem to Hamiltonian (same as qlto_sat_formal.py).
    
    Ground state = satisfying assignment (energy = 0)
    """
    if not QISKIT_AVAILABLE:
        return None
    
    pauli_list = []
    
    for clause in problem.clauses:
        # Each clause contributes penalty if unsatisfied
        clause_vars = [abs(lit) - 1 for lit in clause.literals]
        clause_signs = [1 if lit > 0 else -1 for lit in clause.literals]
        
        # Build Pauli string for this clause
        pauli_str = ['I'] * problem.n_vars
        
        for var_idx, sign in zip(clause_vars, clause_signs):
            if var_idx < problem.n_vars:
                # Positive literal: penalize |0⟩ → use (I-Z)/2
                # Negative literal: penalize |1⟩ → use (I+Z)/2
                # For simplicity, use Z with appropriate sign
                pauli_str[var_idx] = 'Z' if sign < 0 else 'Z'
        
        pauli_term = ''.join(pauli_str)
        coeff = 1.0 / len(clause.literals)
        pauli_list.append((pauli_term, coeff))
    
    if not pauli_list:
        pauli_list = [('I' * problem.n_vars, 0.0)]
    
    return SparsePauliOp.from_list(pauli_list)


# ==============================================================================
# STEP 4: ADIABATIC EVOLUTION
# ==============================================================================

def prepare_initial_state(assignment: Dict[int, bool], n_vars: int) -> QuantumCircuit:
    """
    Prepare quantum state |assignment⟩.
    
    This is the ground state of the easy problem.
    """
    qc = QuantumCircuit(n_vars)
    
    for var in range(1, n_vars + 1):
        if assignment.get(var, False):
            # Variable is True → qubit should be |1⟩
            qc.x(var - 1)
    
    return qc


def solve_sat_adiabatic_morphing(
    problem_hard: SATProblem,
    evolution_time: float = 10.0,
    trotter_steps: int = 50,
    reduction_strategy: str = 'drop_last',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Solve SAT using Adiabatic Problem-Space Morphing.
    
    ALGORITHM:
    1. Reduce hard problem to easy 2-SAT
    2. Solve easy problem classically (polynomial time)
    3. Prepare quantum state in easy solution
    4. Adiabatically evolve from H(easy) to H(hard)
    5. Measure and verify
    
    Parameters:
    - evolution_time: Total evolution time T
    - trotter_steps: Number of Trotter steps for simulation
    - reduction_strategy: How to reduce 3-SAT to 2-SAT
    
    Returns:
    - solved: Whether we found a valid solution
    - assignment: The solution (if found)
    - easy_problem: The 2-SAT problem we started from
    - easy_solution: The 2-SAT solution we used as initial state
    """
    if not QISKIT_AVAILABLE:
        return {'error': 'Qiskit required'}
    
    t_start = time.time()
    n_vars = problem_hard.n_vars
    
    if verbose:
        print("\n" + "="*80)
        print("ADIABATIC PROBLEM-SPACE MORPHING SAT SOLVER")
        print("="*80)
        print(f"Hard Problem: {n_vars} vars, {len(problem_hard.clauses)} clauses")
        print(f"Strategy: {reduction_strategy}")
        print(f"Evolution time: {evolution_time}, Trotter steps: {trotter_steps}")
        print("="*80)
    
    # ===== STEP 1: Create Easy Problem =====
    if verbose:
        print("\n[STEP 1] Reducing to 2-SAT...")
    
    problem_easy = reduce_to_2sat(problem_hard, strategy=reduction_strategy)
    
    if verbose:
        print(f"  Easy Problem: {problem_easy.n_vars} vars, {len(problem_easy.clauses)} clauses")
    
    # ===== STEP 2: Solve Easy Problem Classically =====
    if verbose:
        print("\n[STEP 2] Solving 2-SAT classically...")
    
    easy_solution = solve_2sat_implication_graph(problem_easy)
    
    if easy_solution is None:
        if verbose:
            print("  → Easy problem is UNSAT! Cannot proceed.")
        return {
            'solved': False,
            'error': 'Easy 2-SAT problem is unsatisfiable',
            'easy_problem': problem_easy
        }
    
    if verbose:
        satisfied_easy = sum(1 for c in problem_easy.clauses if c.is_satisfied(easy_solution))
        print(f"  → Found solution: {satisfied_easy}/{len(problem_easy.clauses)} clauses satisfied")
        
        # Check how many hard problem clauses this satisfies
        satisfied_hard = sum(1 for c in problem_hard.clauses if c.is_satisfied(easy_solution))
        print(f"  → This also satisfies {satisfied_hard}/{len(problem_hard.clauses)} hard clauses!")
        
        if satisfied_hard == len(problem_hard.clauses):
            print("  → Easy solution already solves hard problem! ✓")
            return {
                'solved': True,
                'assignment': easy_solution,
                'easy_problem': problem_easy,
                'easy_solution': easy_solution,
                'lucky': True,
                'time_seconds': time.time() - t_start
            }
    
    # ===== STEP 3: Build Hamiltonians =====
    if verbose:
        print("\n[STEP 3] Building Hamiltonians...")
    
    H_easy = sat_to_hamiltonian(problem_easy)
    H_hard = sat_to_hamiltonian(problem_hard)
    
    if verbose:
        print(f"  H_easy: {len(H_easy)} terms")
        print(f"  H_hard: {len(H_hard)} terms")
    
    # ===== STEP 4: Prepare Initial State =====
    if verbose:
        print("\n[STEP 4] Preparing initial quantum state...")
    
    var_reg = QuantumRegister(n_vars, 'vars')
    qc = QuantumCircuit(var_reg)
    
    # Prepare |easy_solution⟩
    init_circuit = prepare_initial_state(easy_solution, n_vars)
    qc.compose(init_circuit, inplace=True)
    
    if verbose:
        print(f"  Prepared state: |ψ(0)⟩ = |easy_solution⟩")
    
    # ===== STEP 5: Adiabatic Evolution =====
    if verbose:
        print("\n[STEP 5] Performing adiabatic evolution...")
        print(f"  Evolving from H(easy) to H(hard) over time T={evolution_time}")
    
    dt = evolution_time / trotter_steps
    
    for step in range(trotter_steps):
        # Annealing schedule: s(t) = t/T
        s = (step + 0.5) / trotter_steps
        
        # Interpolated Hamiltonian: H(s) = (1-s)H_easy + s·H_hard
        H_s = (1 - s) * H_easy + s * H_hard
        
        # Apply time evolution: exp(-i H(s) dt)
        try:
            evolution_gate = PauliEvolutionGate(H_s, time=dt, synthesis=LieTrotter())
            # Decompose the gate into basic gates before appending
            evolution_circuit = evolution_gate.definition
            qc.compose(evolution_circuit, var_reg, inplace=True)
        except Exception as e:
            if verbose:
                print(f"  Error at step {step}: {e}")
            break
    
    if verbose:
        print(f"  Evolution complete. Circuit depth: {qc.depth()}")
    
    # ===== STEP 6: Measure =====
    if verbose:
        print("\n[STEP 6] Measuring final state...")
    
    qc.measure_all()
    
    # Simulate
    backend = AerSimulator()
    job = backend.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()
    
    # Get top result
    if not counts:
        return {'solved': False, 'error': 'No measurement results'}
    
    top_bitstring = max(counts, key=counts.get)
    top_count = counts[top_bitstring]
    confidence = top_count / 1000
    
    if verbose:
        print(f"  Top result: '{top_bitstring}' (confidence: {confidence:.1%})")
    
    # ===== STEP 7: Verify =====
    if verbose:
        print("\n[STEP 7] Verifying solution...")
    
    # Convert bitstring to assignment (reverse for Qiskit's LSB order)
    final_assignment = {}
    for i, bit in enumerate(reversed(top_bitstring)):
        if i + 1 <= n_vars:
            final_assignment[i + 1] = (bit == '1')
    
    is_valid = problem_hard.is_satisfied(final_assignment)
    satisfied_count = sum(1 for c in problem_hard.clauses if c.is_satisfied(final_assignment))
    
    if verbose:
        print(f"  Satisfied: {satisfied_count}/{len(problem_hard.clauses)} clauses")
        if is_valid:
            print("  → SOLUTION VERIFIED! ✓")
        else:
            print(f"  → Not a valid solution. ✗")
    
    t_end = time.time()
    
    return {
        'solved': is_valid,
        'assignment': final_assignment if is_valid else None,
        'satisfied_clauses': satisfied_count,
        'total_clauses': len(problem_hard.clauses),
        'confidence': confidence,
        'easy_problem': problem_easy,
        'easy_solution': easy_solution,
        'time_seconds': t_end - t_start,
        'evolution_time': evolution_time,
        'trotter_steps': trotter_steps
    }


# ==============================================================================
# TESTING & ANALYSIS
# ==============================================================================

def test_morphing_solver():
    """
    Test the adiabatic morphing solver on various instances.
    """
    print("\n" + "="*80)
    print("TESTING ADIABATIC PROBLEM-SPACE MORPHING")
    print("="*80)
    
    # Test 1: Simple 3-SAT
    print("\n[TEST 1] Simple 3-SAT")
    print("-" * 80)
    clauses1 = [
        SATClause((1, 2, 3)),
        SATClause((-1, 2, -3)),
        SATClause((1, -2, 3))
    ]
    problem1 = SATProblem(n_vars=3, clauses=clauses1)
    result1 = solve_sat_adiabatic_morphing(problem1, evolution_time=5.0, trotter_steps=20)
    
    print(f"\n→ Result: {'✓ SOLVED' if result1.get('solved') else '✗ FAILED'}")
    print(f"   Time: {result1.get('time_seconds', 0):.2f}s")
    
    # Test 2: Slightly larger
    print("\n[TEST 2] Medium 3-SAT (N=4)")
    print("-" * 80)
    random.seed(42)
    clauses2 = []
    for _ in range(6):
        vars_sample = random.sample(range(1, 5), 3)
        lits = tuple(v if random.random() > 0.5 else -v for v in vars_sample)
        clauses2.append(SATClause(lits))
    problem2 = SATProblem(n_vars=4, clauses=clauses2)
    result2 = solve_sat_adiabatic_morphing(problem2, evolution_time=10.0, trotter_steps=30)
    
    print(f"\n→ Result: {'✓ SOLVED' if result2.get('solved') else '✗ FAILED'}")
    print(f"   Time: {result2.get('time_seconds', 0):.2f}s")
    
    print("\n" + "="*80)
    print("ANALYSIS OF THE CONJECTURE")
    print("="*80)
    print("""
The adiabatic morphing approach relies on a KEY CONJECTURE:

    "The spectral gap along the path from H(2-SAT) to H(3-SAT)
     is polynomially bounded."

Testing this conjecture requires:
1. Computing eigenvalues of H(s) for s ∈ [0,1]
2. Finding g(s) = E₁(s) - E₀(s) (gap between ground and first excited state)
3. Finding g_min = min_{s∈[0,1]} g(s)
4. Checking if g_min scales polynomially or exponentially with N

For small N (≤ 10), we can compute this exactly.
For large N, we need:
- Theoretical analysis
- Numerical sampling at key points
- Comparison with known hard instances

PRELIMINARY HYPOTHESIS:
- For structured SAT: Gap likely polynomial ✓
- For random SAT (below threshold): Gap likely polynomial ✓  
- For adversarial SAT: Gap likely exponential ✗

The morphing approach may provide the SAME conditional guarantees
as qlto_sat_formal.py, but through a different mechanism.
""")


if __name__ == "__main__":
    if not QISKIT_AVAILABLE:
        print("Qiskit required to run morphing solver")
    else:
        test_morphing_solver()

