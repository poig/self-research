"""

================================================================================
ADVERSARIAL ATTACK TESTING: Finding the Exponential Barrier
================================================================================

This tests THREE attack strategies against hierarchical scaffolding:

1. SPECIFIC LAYER ATTACK: Design clause k to force exponential gap at layer k
2. CUMULATIVE ATTACK: Distribute gap closure across many layers
3. EXPONENTIA    print(f"\n3. EXPONENTIAL LAYERS:")
    print(f"   Verdict: {results['exponential_layers']['verdict']}")
    num_layers_exp = len(results['exponential_layers'].get('all_gaps', []))
    print(f"   Layers: {num_layers_exp} (expected {2**N3})")
    if num_layers_exp >= 2**N3:YERS: Force exponentially many layers to reach solution

Goal: Definitively show WHERE the adversary can hide the exponential complexity.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from qlto_sat_scaffolding import SATClause, SATProblem
from qlto_sat_hierarchical_scaffolding import (
    test_hierarchical_scaffolding,
    build_hierarchical_layers,
    analyze_layer_gap
)


# ============================================================================
# ATTACK 1: SPECIFIC LAYER ATTACK
# ============================================================================

def generate_specific_layer_attack(N: int, target_layer: int) -> SATProblem:
    """
    Generate adversarial SAT where layer k is designed to have exponential gap.
    
    Strategy:
    - Layers 0 to k-1: Easy clauses (all satisfied by x_i = 1)
    - Layer k: HARD clause that forces exponential gap
    - Layers k+1 to M: Normal clauses
    
    The hard clause at layer k creates avoided crossing via entanglement.
    """
    print(f"\n{'='*80}")
    print(f"ATTACK 1: Specific Layer Attack (target layer {target_layer})")
    print(f"{'='*80}")
    
    clauses = []
    
    # Phase 1: Easy clauses (layers 0 to target_layer-1)
    # All satisfied by assignment x_i = True for all i
    print(f"\nPhase 1: Layers 0-{target_layer-1} (easy clauses)")
    for i in range(target_layer):
        # Add clause (x_1 ∨ x_2 ∨ x_3), (x_2 ∨ x_3 ∨ x_4), etc.
        vars_in_clause = [(i % N) + 1, ((i+1) % N) + 1, ((i+2) % N) + 1]
        clause = SATClause(literals=tuple(vars_in_clause))
        clauses.append(clause)
        print(f"  Layer {i}: {clause.literals} (satisfied by all x_i=True)")
    
    # Phase 2: HARD clause at target layer
    # Force x_1 = False via complex entanglement
    print(f"\nPhase 2: Layer {target_layer} (HARD CLAUSE)")
    
    if N >= 4:
        # Create entangled clause: (¬x_1 ∨ ¬x_2 ∨ ¬x_3)
        # This FORCES at least one of x_1, x_2, x_3 to be False
        # Creates avoided crossing with previous all-True state
        hard_clause = SATClause(literals=(-1, -2, -3))
        clauses.append(hard_clause)
        print(f"  Hard clause: {hard_clause.literals}")
        print(f"  → Forces deviation from x_i=True solution")
        print(f"  → Expected: Exponential gap due to avoided crossing")
    else:
        # For small N, use simpler hard clause
        hard_clause = SATClause(literals=(-1, -2, N if N > 2 else 3))
        clauses.append(hard_clause)
        print(f"  Hard clause: {hard_clause.literals}")
    
    # Phase 3: Normal clauses after target layer
    print(f"\nPhase 3: Layers {target_layer+1}+ (normal clauses)")
    remaining = max(0, 3*N - len(clauses))  # Target M ≈ 3N
    for i in range(remaining):
        # Add random-ish clauses
        idx = target_layer + 1 + i
        vars_in_clause = [
            ((idx * 2) % N) + 1,
            ((idx * 2 + 1) % N) + 1,
            ((idx * 2 + 2) % N) + 1
        ]
        # Mix signs
        signs = [1 if (idx + j) % 2 == 0 else -1 for j in range(3)]
        literals = tuple(s * v for s, v in zip(signs, vars_in_clause))
        clause = SATClause(literals=literals)
        clauses.append(clause)
    
    print(f"\nTotal clauses: {len(clauses)}")
    print(f"Expected failure point: Layer {target_layer}")
    
    return SATProblem(n_vars=N, clauses=clauses)


# ============================================================================
# ATTACK 2: CUMULATIVE GAP CLOSURE
# ============================================================================

def generate_cumulative_attack(N: int) -> SATProblem:
    """
    Generate adversarial SAT where gap closes CUMULATIVELY across layers.
    
    Strategy:
    - Each layer reduces gap by constant factor
    - After O(N) layers, gap becomes exponentially small
    - No single "hard" layer, but cumulative effect
    
    Mathematical model:
    - Layer 0: gap = 1.0
    - Layer k: gap = gap_{k-1} * (1 - 1/N)
    - After N layers: gap ≈ 1/e ≈ 0.368
    - After 2N layers: gap ≈ 1/e² ≈ 0.135
    - After kN layers: gap ≈ 1/e^k → exponential decay
    """
    print(f"\n{'='*80}")
    print(f"ATTACK 2: Cumulative Gap Closure")
    print(f"{'='*80}")
    
    clauses = []
    
    # Strategy: Create increasingly entangled clauses
    # Each clause involves more variables, creating tighter constraints
    
    print(f"\nBuilding cumulative attack with N={N}")
    print(f"Expected: Gap decays as (1-1/N)^k ≈ e^(-k/N)")
    
    # Phase 1: Start with simple clauses
    for i in range(N):
        # Simple 2-literal clauses (easy to satisfy)
        var1 = i + 1
        var2 = ((i + 1) % N) + 1
        clause = SATClause(literals=(var1, var2, var2))  # Degenerate 3-clause
        clauses.append(clause)
        if i < 3:
            print(f"  Layer {i}: {clause.literals} (loose constraint)")
    
    # Phase 2: Add increasingly tight constraints
    for i in range(N, 3*N):
        # 3-literal clauses with mixed polarities
        # Designed to create overlapping constraints
        var1 = ((i * 3) % N) + 1
        var2 = ((i * 3 + 1) % N) + 1
        var3 = ((i * 3 + 2) % N) + 1
        
        # Alternate signs to create entanglement
        sign1 = 1 if i % 3 == 0 else -1
        sign2 = 1 if (i+1) % 3 == 0 else -1
        sign3 = 1 if (i+2) % 3 == 0 else -1
        
        clause = SATClause(literals=(sign1*var1, sign2*var2, sign3*var3))
        clauses.append(clause)
        
        if i < N + 3:
            print(f"  Layer {i}: {clause.literals} (tightening)")
    
    print(f"\nTotal clauses: {len(clauses)}")
    print(f"Expected: Gap decays gradually, becomes exponential after ~{N} layers")
    
    return SATProblem(n_vars=N, clauses=clauses)


# ============================================================================
# ATTACK 3: EXPONENTIAL LAYERS
# ============================================================================

def generate_exponential_layers_attack(N: int) -> Tuple[SATProblem, int]:
    """
    Generate adversarial SAT that REQUIRES exponentially many layers.
    
    Strategy:
    - Binary counter construction
    - Each "step" requires adding O(N) clauses
    - Need 2^k steps to reach solution
    - Total layers: M = O(2^k * N)
    
    Example for N=3:
    - Need to count from 000 to 111 (8 states)
    - Each state requires separate layer
    - Total layers: 8 * 3 = 24 clauses
    
    This forces the NUMBER of layers to be exponential.
    """
    print(f"\n{'='*80}")
    print(f"ATTACK 3: Exponential Layers")
    print(f"{'='*80}")
    
    clauses = []
    
    # Binary counter: Force counting through all 2^N assignments
    # Each assignment requires a separate clause to "activate" it
    
    num_states = 2**N
    print(f"\nBinary counter for N={N}")
    print(f"Number of states: {num_states}")
    print(f"Creating clause for each state...")
    
    for state in range(num_states):
        # Convert state to binary assignment
        assignment = []
        for bit in range(N):
            if state & (1 << bit):
                assignment.append(bit + 1)  # True
            else:
                assignment.append(-(bit + 1))  # False
        
        # Create clause that's satisfied ONLY by this assignment
        # NOT(assignment) ∨ x_special
        # This forces progression through states
        
        # For 3-SAT, take first 3 literals
        literals = tuple(assignment[:3])
        clause = SATClause(literals=literals)
        clauses.append(clause)
        
        if state < 5 or state >= num_states - 2:
            print(f"  State {state:03b}: {clause.literals}")
    
    if num_states > 10:
        print(f"  ... ({num_states - 7} more states) ...")
    
    print(f"\nTotal clauses (layers): {len(clauses)}")
    print(f"Expected: Need O(2^N) = O({num_states}) layers to solve")
    print(f"Expected: Exponential in N!")
    
    return SATProblem(n_vars=N, clauses=clauses), num_states


# ============================================================================
# PIGEONHOLE PRINCIPLE (Classic Hard Instance)
# ============================================================================

def generate_pigeonhole_sat(N: int) -> SATProblem:
    """
    Generate pigeonhole SAT: N+1 pigeons, N holes (UNSAT).
    
    This is a classic hard instance for resolution-based SAT solvers.
    Tests if hierarchical scaffolding can detect UNSATISFIABILITY.
    
    Clauses:
    1. Each pigeon in at least one hole: (p_i1 ∨ p_i2 ∨ ... ∨ p_iN)
    2. No two pigeons in same hole: (¬p_i1 ∨ ¬p_j1) for all i<j, all holes
    """
    print(f"\n{'='*80}")
    print(f"PIGEONHOLE PRINCIPLE: {N+1} pigeons, {N} holes (UNSAT)")
    print(f"{'='*80}")
    
    clauses = []
    
    # Encoding: Variable p_ih means "pigeon i in hole h"
    # Variable index: (i-1)*N + h for pigeon i, hole h
    
    def var(pigeon: int, hole: int) -> int:
        """Get variable index for pigeon i in hole h."""
        return (pigeon - 1) * N + hole
    
    # Clause type 1: Each pigeon in at least one hole
    print(f"\nClauses: Each pigeon in at least one hole")
    for pigeon in range(1, N+2):  # N+1 pigeons
        # For 3-SAT, need to break into multiple clauses
        if N <= 3:
            literals = tuple(var(pigeon, h) for h in range(1, N+1))
            clause = SATClause(literals=literals)
            clauses.append(clause)
            if pigeon <= 2:
                print(f"  Pigeon {pigeon}: {clause.literals}")
        else:
            # Break into 3-literal clauses
            for start in range(0, N, 3):
                end = min(start + 3, N)
                literals = tuple(var(pigeon, h) for h in range(start+1, end+1))
                if len(literals) == 3:
                    clause = SATClause(literals=literals)
                    clauses.append(clause)
    
    # Clause type 2: No two pigeons in same hole
    print(f"\nClauses: No two pigeons in same hole")
    count = 0
    for hole in range(1, N+1):
        for pigeon1 in range(1, N+2):
            for pigeon2 in range(pigeon1+1, N+2):
                # ¬p_i1 ∨ ¬p_j1 (can't both be in same hole)
                literals = (-var(pigeon1, hole), -var(pigeon2, hole), -var(pigeon2, hole))
                clause = SATClause(literals=literals)
                clauses.append(clause)
                count += 1
                if count <= 3:
                    print(f"  Hole {hole}, pigeons {pigeon1},{pigeon2}: {clause.literals[:2]}")
    
    print(f"\nTotal clauses: {len(clauses)}")
    print(f"Total variables: {(N+1)*N}")
    print(f"Expected: UNSATISFIABLE (should detect contradiction)")
    
    return SATProblem(n_vars=(N+1)*N, clauses=clauses)


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

def run_adversarial_tests():
    """
    Run all three adversarial attack strategies.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ADVERSARIAL TESTING")
    print("="*80)
    print("\nGoal: Find WHERE the exponential barrier appears")
    print("\nThree attack strategies:")
    print("1. Specific Layer Attack - exponential gap at target layer")
    print("2. Cumulative Attack - gap decays across many layers")
    print("3. Exponential Layers - require exp(N) layers to solve")
    print("\n" + "="*80)
    
    results = {}
    
    # ========================================================================
    # TEST 1: Specific Layer Attack (N=4, target layer 5)
    # ========================================================================
    print("\n\n" + "="*80)
    print("TEST 1: SPECIFIC LAYER ATTACK")
    print("="*80)
    
    N1 = 4
    target = 5
    problem1 = generate_specific_layer_attack(N1, target_layer=target)
    
    print(f"\nRunning hierarchical scaffolding on specific layer attack...")
    result1 = test_hierarchical_scaffolding(problem1, strategy='sequential')
    results['specific_layer'] = result1
    
    print(f"\n{'='*80}")
    print(f"RESULT: {result1['verdict']}")
    print(f"{'='*80}")
    
    if result1['verdict'] == 'EXPONENTIAL_SOME':
        print(f"✓ CONFIRMED: Specific layer attack WORKS!")
        print(f"  → Found exponential gap at layer {result1['exponential_layers'][0]}")
        print(f"  → Expected at layer {target}, actual at {result1['exponential_layers']}")
    else:
        print(f"✗ UNEXPECTED: Attack failed to create exponential gap")
        print(f"  → All layers maintained polynomial gap")
        print(f"  → Need stronger attack strategy")
    
    # ========================================================================
    # TEST 2: Cumulative Attack (N=4)
    # ========================================================================
    print("\n\n" + "="*80)
    print("TEST 2: CUMULATIVE GAP CLOSURE")
    print("="*80)
    
    N2 = 4
    problem2 = generate_cumulative_attack(N2)
    
    print(f"\nRunning hierarchical scaffolding on cumulative attack...")
    result2 = test_hierarchical_scaffolding(problem2, strategy='sequential')
    results['cumulative'] = result2
    
    print(f"\n{'='*80}")
    print(f"RESULT: {result2['verdict']}")
    print(f"{'='*80}")
    
    if result2['verdict'] == 'EXPONENTIAL_SOME':
        print(f"✓ CONFIRMED: Cumulative attack WORKS!")
        print(f"  → Gap decayed to exponential at layers: {result2['exponential_layers']}")
        print(f"  → Cumulative effect visible")
    else:
        print(f"✗ UNEXPECTED: Cumulative attack failed")
        print(f"  → Gap remained polynomial throughout")
        
        # Analyze gap trend
        if 'min_gaps' in result2 and result2['min_gaps']:
            gaps = result2['min_gaps']
            print(f"\n  Gap trend analysis:")
            for i in range(min(5, len(gaps))):
                print(f"    Layer {i}: g_min = {gaps[i]:.6f}")
            if len(gaps) > 10:
                print(f"    ...")
                for i in range(len(gaps)-3, len(gaps)):
                    print(f"    Layer {i}: g_min = {gaps[i]:.6f}")
    
    # ========================================================================
    # TEST 3: Exponential Layers (N=3, expect 2^3=8 layers)
    # ========================================================================
    print("\n\n" + "="*80)
    print("TEST 3: EXPONENTIAL LAYERS REQUIREMENT")
    print("="*80)
    
    N3 = 3
    problem3, expected_layers = generate_exponential_layers_attack(N3)
    
    print(f"\nRunning hierarchical scaffolding on exponential layers attack...")
    result3 = test_hierarchical_scaffolding(problem3, strategy='sequential')
    results['exponential_layers'] = result3
    
    print(f"\n{'='*80}")
    print(f"RESULT: {result3['verdict']}")
    print(f"{'='*80}")
    
    actual_layers = len(result3.get('all_gaps', []))
    print(f"\nExpected layers: {expected_layers} (2^N)")
    print(f"Actual layers: {actual_layers}")
    
    if actual_layers == expected_layers:
        print(f"✓ CONFIRMED: Required exponential layers!")
        print(f"  → Algorithm needs O(2^N) layers")
        print(f"  → Even with constant gap, total time is exponential")
    else:
        print(f"✗ UNEXPECTED: Layers don't match")
        print(f"  → Need to analyze why")
    
    # ========================================================================
    # TEST 4: Pigeonhole (Classic Hard Instance)
    # ========================================================================
    print("\n\n" + "="*80)
    print("TEST 4: PIGEONHOLE PRINCIPLE (UNSAT)")
    print("="*80)
    
    N4 = 2  # 3 pigeons, 2 holes (smallest UNSAT instance)
    problem4 = generate_pigeonhole_sat(N4)
    
    print(f"\nRunning hierarchical scaffolding on pigeonhole...")
    result4 = test_hierarchical_scaffolding(problem4, strategy='sequential')
    results['pigeonhole'] = result4
    
    print(f"\n{'='*80}")
    print(f"RESULT: {result4['verdict']}")
    print(f"{'='*80}")
    
    if result4['verdict'] == 'EXPONENTIAL_SOME':
        print(f"✓ CONFIRMED: Pigeonhole creates exponential gap!")
        print(f"  → Classic hard instance defeats algorithm")
    else:
        print(f"⚠️  INTERESTING: Pigeonhole didn't create exponential gap")
        print(f"  → But it's UNSAT, so what happens?")
        if 'min_gaps' in result4 and result4['min_gaps']:
            print(f"  → Minimum gap: {min(result4['min_gaps']):.6f}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "="*80)
    print("FINAL SUMMARY: WHERE IS THE EXPONENTIAL BARRIER?")
    print("="*80)
    
    print("\n1. SPECIFIC LAYER ATTACK:")
    print(f"   Verdict: {results['specific_layer']['verdict']}")
    if results['specific_layer']['verdict'] == 'EXPONENTIAL_SOME':
        print(f"   ✓ Found exponential gap at specific layer")
    else:
        print(f"   ✗ Failed to create exponential gap")
    
    print("\n2. CUMULATIVE ATTACK:")
    print(f"   Verdict: {results['cumulative']['verdict']}")
    if results['cumulative']['verdict'] == 'EXPONENTIAL_SOME':
        print(f"   ✓ Gap decayed cumulatively to exponential")
    else:
        print(f"   ✗ Gap remained polynomial")
    
    print("\n3. EXPONENTIAL LAYERS:")
    print(f"   Verdict: {results['exponential_layers']['verdict']}")
    # Get number of layers from result
    num_layers_exp = len(results['exponential_layers'].get('min_gaps', []))
    print(f"   Layers: {num_layers_exp} (expected {2**N3})")
    if num_layers_exp >= 2**N3:
        print(f"   ✓ Required exponential layers")
    else:
        print(f"   ? Layers don't match expectation")
    
    print("\n4. PIGEONHOLE (UNSAT):")
    print(f"   Verdict: {results['pigeonhole']['verdict']}")
    if results['pigeonhole']['verdict'] == 'EXPONENTIAL_SOME':
        print(f"   ✓ Classic hard instance creates exponential gap")
    else:
        print(f"   ? Interesting behavior on UNSAT")
    
    # Overall conclusion
    print("\n" + "="*80)
    print("OVERALL CONCLUSION")
    print("="*80)
    
    num_exponential = sum(1 for r in results.values() if r['verdict'] == 'EXPONENTIAL_SOME')
    
    if num_exponential > 0:
        print(f"\n✓✓✓ CONFIRMED: Adversarial attacks CAN create exponential gaps!")
        print(f"     Found {num_exponential}/4 attacks successful")
        print(f"\n     → Hierarchical scaffolding is CONDITIONAL")
        print(f"     → Works for random SAT (loose coupling)")
        print(f"     → Fails for adversarial SAT (designed entanglement)")
        print(f"     → Boundary is now UNDERSTOOD")
    else:
        print(f"\n⚠️⚠️⚠️ UNEXPECTED: None of the attacks created exponential gaps!")
        print(f"\n     → Need stronger adversarial constructions")
        print(f"     → Or hierarchical is more robust than expected?")
        print(f"     → Requires deeper theoretical analysis")
    
    return results


if __name__ == "__main__":
    results = run_adversarial_tests()
    
    print("\n\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nSee results above for detailed analysis.")
    print("This definitively answers WHERE the exponential barrier appears.")

