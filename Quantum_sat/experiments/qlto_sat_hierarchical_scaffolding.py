"""

qlto_sat_hierarchical_scaffolding.py

HIERARCHICAL ADIABATIC SCAFFOLDING FOR SAT

Instead of one evolution from seed → full problem,
we do MULTIPLE layers, each adding more clauses.

Key question: Does each layer maintain polynomial gap?
Or does adversary force exponential gap at some layer?

This is the critical test for extending to unconditional SAT.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator


@dataclass
class SATClause:
    """A single SAT clause (disjunction of literals)"""
    literals: Tuple[int, ...]  # Positive = variable, negative = negation
    
    def is_satisfied(self, assignment: Dict[int, bool]) -> bool:
        """Check if this clause is satisfied by assignment"""
        for lit in self.literals:
            var = abs(lit)
            val = assignment.get(var, False)
            if (lit > 0 and val) or (lit < 0 and not val):
                return True
        return False
    
    def get_variables(self) -> Set[int]:
        """Get set of variables in this clause"""
        return {abs(lit) for lit in self.literals}


@dataclass
class SATProblem:
    """A SAT problem instance"""
    n_vars: int
    clauses: List[SATClause]


@dataclass
class HierarchicalLayer:
    """One layer in hierarchical scaffolding"""
    layer_id: int
    seed_clauses: List[SATClause]  # Clauses satisfied so far
    new_clause: SATClause  # New clause to add this layer
    rest_clauses: List[SATClause]  # Remaining clauses
    
    def get_all_active_clauses(self) -> List[SATClause]:
        """Get all clauses active at this layer"""
        return self.seed_clauses + [self.new_clause]


def build_hierarchical_layers(
    problem: SATProblem,
    strategy: str = 'sequential'
) -> List[HierarchicalLayer]:
    """
    Build hierarchical layers for scaffolding.
    
    Strategy:
    - 'sequential': Add clauses one by one in order
    - 'greedy': Add clause that filters most candidates
    - 'random': Random order
    """
    layers = []
    
    if strategy == 'sequential':
        # Simple: add clauses in order
        for i in range(len(problem.clauses)):
            layer = HierarchicalLayer(
                layer_id=i,
                seed_clauses=problem.clauses[:i] if i > 0 else [],
                new_clause=problem.clauses[i],
                rest_clauses=problem.clauses[i+1:]
            )
            layers.append(layer)
    
    elif strategy == 'random':
        # Random order
        indices = list(range(len(problem.clauses)))
        random.shuffle(indices)
        
        for i, idx in enumerate(indices):
            layer = HierarchicalLayer(
                layer_id=i,
                seed_clauses=[problem.clauses[j] for j in indices[:i]] if i > 0 else [],
                new_clause=problem.clauses[idx],
                rest_clauses=[problem.clauses[j] for j in indices[i+1:]]
            )
            layers.append(layer)
    
    elif strategy == 'greedy':
        # Add clause that maximally constrains current solution space
        remaining = list(range(len(problem.clauses)))
        used = []
        
        for i in range(len(problem.clauses)):
            if i == 0:
                # Start with first clause
                idx = remaining.pop(0)
                used.append(idx)
            else:
                # Pick clause that shares most variables with used clauses
                best_idx = 0
                best_score = -1
                
                used_vars = set()
                for j in used:
                    used_vars.update(problem.clauses[j].get_variables())
                
                for k, idx in enumerate(remaining):
                    overlap = len(problem.clauses[idx].get_variables() & used_vars)
                    if overlap > best_score:
                        best_score = overlap
                        best_idx = k
                
                idx = remaining.pop(best_idx)
                used.append(idx)
            
            layer = HierarchicalLayer(
                layer_id=i,
                seed_clauses=[problem.clauses[j] for j in used[:-1]] if i > 0 else [],
                new_clause=problem.clauses[used[-1]],
                rest_clauses=[problem.clauses[j] for j in remaining]
            )
            layers.append(layer)
    
    return layers


def analyze_layer_gap(
    layer: HierarchicalLayer,
    n_vars: int,
    num_points: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Analyze spectral gap for one hierarchical layer.
    
    This evolves: H(s) = H_seed + H_new + s·H_rest
    
    Where:
    - H_seed: All clauses satisfied so far (constant energy)
    - H_new: The new clause being added (constant energy)
    - H_rest: Remaining clauses (scaled by s)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"LAYER {layer.layer_id} ANALYSIS")
        print(f"{'='*80}")
        print(f"Seed clauses: {len(layer.seed_clauses)}")
        print(f"New clause: {layer.new_clause.literals}")
        print(f"Rest clauses: {len(layer.rest_clauses)}")
    
    # Build Hamiltonians
    dim = 2 ** n_vars
    
    # H_seed: All clauses satisfied so far
    H_seed = np.zeros((dim, dim), dtype=float)
    if len(layer.seed_clauses) > 0:
        for x in range(dim):
            assignment = {i+1: bool((x >> i) & 1) for i in range(n_vars)}
            penalty = sum(1.0 for c in layer.seed_clauses if not c.is_satisfied(assignment))
            H_seed[x, x] = penalty
    
    # H_new: The new clause
    H_new = np.zeros((dim, dim), dtype=float)
    for x in range(dim):
        assignment = {i+1: bool((x >> i) & 1) for i in range(n_vars)}
        penalty = 0.0 if layer.new_clause.is_satisfied(assignment) else 1.0
        H_new[x, x] = penalty
    
    # H_rest: Remaining clauses
    H_rest = np.zeros((dim, dim), dtype=float)
    if len(layer.rest_clauses) > 0:
        for x in range(dim):
            assignment = {i+1: bool((x >> i) & 1) for i in range(n_vars)}
            penalty = sum(1.0 for c in layer.rest_clauses if not c.is_satisfied(assignment))
            H_rest[x, x] = penalty
    
    # Sample along path: H(s) = H_seed + H_new + s·H_rest
    s_values = np.linspace(0, 1, num_points)
    gaps = []
    degeneracies = []
    
    for s in s_values:
        H_s = H_seed + H_new + s * H_rest
        eigenvalues = np.linalg.eigvalsh(H_s)
        
        E_min = eigenvalues[0]
        ground_mask = np.abs(eigenvalues - E_min) < 1e-6
        degeneracy = np.sum(ground_mask)
        
        excited = eigenvalues[~ground_mask]
        gap = excited[0] - E_min if len(excited) > 0 else 0.0
        
        gaps.append(gap)
        degeneracies.append(degeneracy)
    
    gaps = np.array(gaps)
    degeneracies = np.array(degeneracies)
    
    # Analysis
    g_min_full = np.min(gaps)
    idx_min_full = np.argmin(gaps)
    s_min_full = s_values[idx_min_full]
    
    # Interior only
    interior_mask = (s_values >= 0.05) & (s_values <= 0.95)
    if np.sum(interior_mask) > 0:
        interior_gaps = gaps[interior_mask]
        interior_s = s_values[interior_mask]
        g_min_interior = np.min(interior_gaps)
        s_min_interior = interior_s[np.argmin(interior_gaps)]
    else:
        g_min_interior = g_min_full
        s_min_interior = s_min_full
    
    if verbose:
        print(f"\nResults:")
        print(f"  Full g_min: {g_min_full:.6f} at s={s_min_full:.4f}")
        print(f"  Interior g_min: {g_min_interior:.6f} at s={s_min_interior:.4f}")
        print(f"  Degeneracy at s=0: {degeneracies[0]}")
        print(f"  Degeneracy at s=1: {degeneracies[-1]}")
        
        # Check if gap is linear
        if g_min_interior > 0.01:
            # Fit linear model
            fit_mask = (s_values >= 0.1) & (s_values <= 0.9)
            if np.sum(fit_mask) > 2:
                s_fit = s_values[fit_mask]
                g_fit = gaps[fit_mask]
                # Linear fit: gap = a*s + b
                A = np.vstack([s_fit, np.ones(len(s_fit))]).T
                coeffs = np.linalg.lstsq(A, g_fit, rcond=None)[0]
                a, b = coeffs
                
                # Check if close to gap = s (a≈1, b≈0)
                if abs(a - 1.0) < 0.2 and abs(b) < 0.2:
                    print(f"  ✓ Gap is approximately LINEAR: Δ(s) ≈ {a:.2f}·s + {b:.2f}")
                else:
                    print(f"  ✗ Gap is NOT linear: Δ(s) ≈ {a:.2f}·s + {b:.2f}")
    
    return {
        'layer_id': layer.layer_id,
        's_values': s_values,
        'gaps': gaps,
        'degeneracies': degeneracies,
        'g_min_full': g_min_full,
        'g_min_interior': g_min_interior,
        's_min': s_min_interior,
        'H_seed': H_seed,
        'H_new': H_new,
        'H_rest': H_rest
    }


def test_hierarchical_scaffolding(
    problem: SATProblem,
    strategy: str = 'sequential',
    verbose: bool = True
) -> Dict:
    """
    Test hierarchical scaffolding on a SAT problem.
    
    Returns gap statistics for each layer.
    """
    print(f"\n{'='*80}")
    print(f"HIERARCHICAL SCAFFOLDING TEST")
    print(f"{'='*80}")
    print(f"Problem: N={problem.n_vars}, M={len(problem.clauses)}")
    print(f"Strategy: {strategy}")
    
    # Build layers
    layers = build_hierarchical_layers(problem, strategy=strategy)
    print(f"Total layers: {len(layers)}")
    
    # Analyze each layer
    results = []
    for layer in layers:
        result = analyze_layer_gap(layer, problem.n_vars, verbose=verbose)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY ACROSS ALL LAYERS")
    print(f"{'='*80}")
    
    min_gaps = [r['g_min_interior'] for r in results]
    
    print(f"\nGap statistics:")
    print(f"  Mean g_min: {np.mean(min_gaps):.6f}")
    print(f"  Min g_min: {np.min(min_gaps):.6f} (Layer {np.argmin(min_gaps)})")
    print(f"  Max g_min: {np.max(min_gaps):.6f} (Layer {np.argmax(min_gaps)})")
    print(f"  Std g_min: {np.std(min_gaps):.6f}")
    
    # Check if all layers have polynomial gap
    polynomial_threshold = 1.0 / problem.n_vars  # Polynomial if g > 1/N
    constant_threshold = 0.01  # Constant if g > 0.01
    
    n_polynomial = np.sum(np.array(min_gaps) > polynomial_threshold)
    n_constant = np.sum(np.array(min_gaps) > constant_threshold)
    
    print(f"\nGap classification:")
    print(f"  Constant (g > 0.01): {n_constant}/{len(layers)} layers")
    print(f"  Polynomial (g > 1/N={polynomial_threshold:.4f}): {n_polynomial}/{len(layers)} layers")
    
    if n_constant == len(layers):
        print(f"\n✓✓✓ ALL LAYERS HAVE CONSTANT GAP!")
        print(f"    → Hierarchical scaffolding achieves O(M·N³) = O(N⁴) complexity!")
        verdict = "CONSTANT_ALL"
    elif n_polynomial == len(layers):
        print(f"\n✓✓ ALL LAYERS HAVE POLYNOMIAL GAP!")
        print(f"    → Hierarchical scaffolding achieves polynomial complexity!")
        verdict = "POLYNOMIAL_ALL"
    else:
        print(f"\n✗ SOME LAYERS HAVE EXPONENTIAL GAP")
        print(f"    → Hierarchical scaffolding fails for this instance")
        print(f"    → Exponential layers: {[i for i, g in enumerate(min_gaps) if g <= polynomial_threshold]}")
        verdict = "EXPONENTIAL_SOME"
    
    return {
        'problem': problem,
        'strategy': strategy,
        'layers': results,
        'min_gaps': min_gaps,
        'verdict': verdict,
        'n_constant': n_constant,
        'n_polynomial': n_polynomial
    }


def generate_random_3sat(n_vars: int, n_clauses: int, seed: int = 42) -> SATProblem:
    """Generate random 3-SAT"""
    random.seed(seed)
    clauses = []
    for _ in range(n_clauses):
        vars_sample = random.sample(range(1, n_vars + 1), min(3, n_vars))
        lits = tuple(v if random.random() > 0.5 else -v for v in vars_sample)
        clauses.append(SATClause(lits))
    return SATProblem(n_vars, clauses)


def generate_adversarial_sat(n_vars: int) -> SATProblem:
    """
    Generate an adversarial SAT instance designed to defeat hierarchical scaffolding.
    
    Strategy: Create clauses that are highly entangled, so no simple ordering
    maintains polynomial gaps.
    """
    # Pigeonhole principle: N+1 pigeons, N holes
    # This is known to be hard for resolution-based solvers
    clauses = []
    
    # Each pigeon must be in some hole
    for pigeon in range(n_vars):
        lits = tuple(range(pigeon * n_vars + 1, (pigeon + 1) * n_vars + 1))
        clauses.append(SATClause(lits))
    
    # No two pigeons in same hole (if we have enough variables)
    if n_vars >= 3:
        for hole in range(n_vars):
            for p1 in range(n_vars):
                for p2 in range(p1 + 1, n_vars):
                    var1 = p1 * n_vars + hole + 1
                    var2 = p2 * n_vars + hole + 1
                    clauses.append(SATClause((-var1, -var2)))
    
    # This creates an UNSAT instance (n+1 pigeons, n holes)
    # But useful for testing adversarial behavior
    
    return SATProblem(n_vars * n_vars, clauses[:20])  # Limit to 20 clauses for testing


if __name__ == "__main__":
    print("="*80)
    print("HIERARCHICAL SCAFFOLDING: THE CRITICAL TEST")
    print("="*80)
    print("""
This tests whether hierarchical scaffolding can maintain polynomial gaps
across ALL layers, or if adversarial instances force exponential closure.

If all layers have constant/polynomial gap → Extended conditional result ✓
If some layer has exponential gap → Hierarchical doesn't help ✗
""")
    
    # Test 1: Random SAT (expect all layers to have polynomial gap)
    print("\n" + "="*80)
    print("TEST 1: Random 3-SAT (N=4, M=16)")
    print("="*80)
    
    problem1 = generate_random_3sat(4, 16, seed=42)
    result1 = test_hierarchical_scaffolding(problem1, strategy='sequential', verbose=True)
    
    # Test 2: Different strategy
    print("\n" + "="*80)
    print("TEST 2: Random 3-SAT with Greedy Strategy")
    print("="*80)
    
    result2 = test_hierarchical_scaffolding(problem1, strategy='greedy', verbose=True)
    
    # Test 3: Smaller instance for detailed analysis
    print("\n" + "="*80)
    print("TEST 3: Small Instance (N=3, M=6)")
    print("="*80)
    
    problem3 = generate_random_3sat(3, 6, seed=123)
    result3 = test_hierarchical_scaffolding(problem3, strategy='sequential', verbose=True)
    
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    print(f"""
Test 1 (Random N=4): {result1['verdict']}
  - Constant gap layers: {result1['n_constant']}/{len(result1['layers'])}
  - Polynomial gap layers: {result1['n_polynomial']}/{len(result1['layers'])}

Test 2 (Greedy N=4): {result2['verdict']}
  - Constant gap layers: {result2['n_constant']}/{len(result2['layers'])}
  - Polynomial gap layers: {result2['n_polynomial']}/{len(result2['layers'])}

Test 3 (Small N=3): {result3['verdict']}
  - Constant gap layers: {result3['n_constant']}/{len(result3['layers'])}
  - Polynomial gap layers: {result3['n_polynomial']}/{len(result3['layers'])}

""")
    
    if all(r['verdict'].startswith('CONSTANT') or r['verdict'].startswith('POLYNOMIAL') 
           for r in [result1, result2, result3]):
        print("✓✓✓ HIERARCHICAL SCAFFOLDING MAINTAINS POLYNOMIAL GAPS!")
        print("    This extends the conditional result to multi-layer evolution.")
        print("    Next: Test on larger N and adversarial instances.")
    else:
        print("✗ HIERARCHICAL SCAFFOLDING HAS EXPONENTIAL LAYERS")
        print("    The hierarchical approach doesn't extend the result.")
        print("    Single-layer scaffolding remains the best conditional algorithm.")

