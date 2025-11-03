"""
Create truly HARD SAT instances (UNDECOMPOSABLE)

Current problem: All test cases have k*=0 (too easy!)
Goal: Generate SAT instances with large minimal separator (k* > 0)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from sat_decompose import create_test_sat_instance
import random

def create_densely_coupled_sat(n, k_target, clause_density=4.0):
    """
    Create SAT with DENSE coupling → large minimal separator
    
    Strategy:
    1. All variables appear in MANY clauses together
    2. High connectivity → hard to partition
    3. No obvious community structure
    
    Args:
        n: Number of variables
        k_target: Target backdoor size (we want k* ≈ k_target)
        clause_density: Clauses per variable (higher = more coupling)
    
    Returns:
        clauses, backdoor_vars, planted_solution
    """
    m = int(n * clause_density)  # Total clauses
    
    clauses = []
    planted_solution = [random.choice([True, False]) for _ in range(n)]
    
    # Strategy: Create clauses that couple ALL variables together
    # This forces large separator!
    
    for i in range(m):
        # Pick 3 random variables (high chance of overlap)
        vars_in_clause = random.sample(range(1, n+1), 3)
        
        # Make clause satisfiable by planted solution
        literals = []
        for v in vars_in_clause:
            if random.random() < 0.5:
                # Positive literal
                literals.append(v if planted_solution[v-1] else -v)
            else:
                # Negative literal
                literals.append(-v if planted_solution[v-1] else v)
        
        clauses.append(literals)
    
    # Backdoor: pick k_target variables that appear most frequently
    var_counts = [0] * (n + 1)
    for clause in clauses:
        for lit in clause:
            var_counts[abs(lit)] += 1
    
    # Sort by frequency
    var_freq = [(v, var_counts[v]) for v in range(1, n+1)]
    var_freq.sort(key=lambda x: x[1], reverse=True)
    
    backdoor_vars = [v for v, _ in var_freq[:k_target]]
    
    return clauses, backdoor_vars, planted_solution


def create_complete_graph_sat(n, k):
    """
    Create SAT where ALL variables are coupled (complete graph)
    → Minimal separator = k (can't do better!)
    
    This is the HARDEST type of SAT!
    """
    clauses = []
    planted_solution = [random.choice([True, False]) for _ in range(n)]
    
    # Create clauses connecting every pair of variables
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            # Clause coupling variable i and j
            # Make it satisfiable
            lit_i = i if planted_solution[i-1] else -i
            lit_j = j if planted_solution[j-1] else -j
            
            # Add some noise variable
            k_noise = random.randint(1, n)
            lit_k = k_noise if planted_solution[k_noise-1] else -k_noise
            
            clauses.append([lit_i, lit_j, lit_k])
    
    # Backdoor = first k variables
    backdoor_vars = list(range(1, k+1))
    
    return clauses, backdoor_vars, planted_solution


def create_chain_sat(n, k, chain_length=5):
    """
    Create SAT with CHAIN structure: 1-2-3-4-5-...
    → Must cut chain → separator size ≈ 1
    
    But if chain is dense, separator can be larger!
    """
    clauses = []
    planted_solution = [random.choice([True, False]) for _ in range(n)]
    
    # Create chain of constraints
    for i in range(1, n):
        # Link variable i to i+1
        lit_i = i if planted_solution[i-1] else -i
        lit_next = (i+1) if planted_solution[i] else -(i+1)
        
        # Add random third variable
        k_rand = random.randint(1, n)
        lit_k = k_rand if planted_solution[k_rand-1] else -k_rand
        
        clauses.append([lit_i, lit_next, lit_k])
    
    # Add more clauses to make it harder
    for _ in range(n):
        vars_in_clause = random.sample(range(1, n+1), 3)
        literals = [v if planted_solution[v-1] else -v for v in vars_in_clause]
        clauses.append(literals)
    
    backdoor_vars = list(range(1, k+1))
    
    return clauses, backdoor_vars, planted_solution


def create_clique_sat(n, k, clique_size=6):
    """
    Create SAT with CLIQUE structure: subset of variables all coupled
    → Must separate clique → separator size ≈ clique_size/2
    """
    clauses = []
    planted_solution = [random.choice([True, False]) for _ in range(n)]
    
    # Create a clique (complete subgraph)
    clique_vars = list(range(1, min(clique_size+1, n+1)))
    
    # All pairs in clique are coupled
    for i in clique_vars:
        for j in clique_vars:
            if i < j:
                lit_i = i if planted_solution[i-1] else -i
                lit_j = j if planted_solution[j-1] else -j
                
                # Add noise
                k_noise = random.randint(1, n)
                lit_k = k_noise if planted_solution[k_noise-1] else -k_noise
                
                clauses.append([lit_i, lit_j, lit_k])
    
    # Add random clauses
    for _ in range(n):
        vars_in_clause = random.sample(range(1, n+1), 3)
        literals = [v if planted_solution[v-1] else -v for v in vars_in_clause]
        clauses.append(literals)
    
    backdoor_vars = list(range(1, k+1))
    
    return clauses, backdoor_vars, planted_solution


if __name__ == '__main__':
    print("=" * 70)
    print("Creating HARD SAT instances (UNDECOMPOSABLE)")
    print("=" * 70)
    
    test_cases = [
        # (name, n, k, structure, expected_k_star)
        ("Dense Coupling", 12, 8, 'dense', 6),
        ("Complete Graph", 10, 8, 'complete', 8),
        ("Chain Structure", 12, 6, 'chain', 2),
        ("Clique Structure", 12, 8, 'clique', 4),
    ]
    
    for name, n, k, structure, expected_k_star in test_cases:
        print(f"\n{name} (N={n}, k={k}, structure={structure})")
        print(f"   Expected k* ≈ {expected_k_star}")
        
        if structure == 'dense':
            clauses, backdoor, sol = create_densely_coupled_sat(n, k)
        elif structure == 'complete':
            clauses, backdoor, sol = create_complete_graph_sat(n, k)
        elif structure == 'chain':
            clauses, backdoor, sol = create_chain_sat(n, k)
        elif structure == 'clique':
            clauses, backdoor, sol = create_clique_sat(n, k)
        
        print(f"   Generated: {len(clauses)} clauses, {len(backdoor)} backdoor vars")
        print(f"   Clause density: {len(clauses)/n:.1f} clauses/var")
        
        # Calculate coupling
        var_occurrences = {}
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                if v not in var_occurrences:
                    var_occurrences[v] = set()
                var_occurrences[v].update([abs(l) for l in clause if abs(l) != v])
        
        avg_coupling = sum(len(neighbors) for neighbors in var_occurrences.values()) / len(var_occurrences)
        print(f"   Average coupling: {avg_coupling:.1f} neighbors/var")
        
        if avg_coupling < 3:
            print(f"   ⚠️  Too sparse! May still be decomposable")
        elif avg_coupling > 8:
            print(f"   ✅ Dense! Likely UNDECOMPOSABLE with k* ≈ {expected_k_star}")
        else:
            print(f"   ⚙️  Moderate coupling, k* may vary")
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print("""
Why previous tests all had k*=0:
- 'modular' structure: Variables naturally partition into modules
- 'hierarchical' structure: Tree-like, easy to separate
- Low clause density: Sparse coupling

How to get k* > 0 (UNDECOMPOSABLE):
- Dense coupling: Every var connected to many others
- Complete graph: ALL variables coupled together
- High clause density: 4+ clauses per variable
- Clique structures: Subsets with complete connections

Test these new instances with:
    python experiments/sat_undecomposable_quantum.py --hard
""")
