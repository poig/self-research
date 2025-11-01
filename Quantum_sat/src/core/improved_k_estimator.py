"""
Improved k Estimation with Multiple Heuristics
==============================================

Better methods for estimating backdoor size k:
1. Satisfaction-based: k ~ log2(N) × (1 - best_satisfaction_rate)
2. Variable degree: k ~ number of high-degree variables
3. Energy landscape: k ~ depth of local minimum basin
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


def estimate_k_improved(clauses: List[Tuple[int, ...]], 
                       n_vars: int, 
                       samples: int = 300) -> Tuple[float, float, Dict]:
    """
    Improved k estimation combining multiple heuristics.
    
    Returns: (k_estimate, confidence, diagnostics)
    """
    
    # =========================================================================
    # Method 1: Satisfaction-based heuristic (most reliable)
    # =========================================================================
    energies = []
    best_state = None
    best_energy = float('inf')
    
    for _ in range(samples):
        state = np.random.randint(0, 2, n_vars)
        
        # Count violations
        violations = 0
        for clause in clauses:
            satisfied = any(
                (lit > 0 and state[abs(lit)-1] == 1) or 
                (lit < 0 and state[abs(lit)-1] == 0) 
                for lit in clause
            )
            if not satisfied:
                violations += 1
        
        energies.append(violations)
        
        if violations < best_energy:
            best_energy = violations
            best_state = state.copy()
    
    energies = np.array(energies)
    m = len(clauses)
    
    # Best satisfaction rate
    satisfaction_rate = 1.0 - (best_energy / m)
    
    # Heuristic: k ≈ log2(N) × unsatisfied_fraction
    # Intuition: If 90% clauses satisfied, only ~10% variables are critical
    if satisfaction_rate >= 0.99:
        # Nearly all satisfied → very structured
        k_satisfaction = max(1.0, 0.5 * np.log2(n_vars))
    elif satisfaction_rate >= 0.90:
        # Most satisfied → small backdoor
        k_satisfaction = np.log2(n_vars) * (1 - satisfaction_rate) * 5
    elif satisfaction_rate >= 0.75:
        # Moderate → medium backdoor  
        k_satisfaction = np.log2(n_vars) + (1 - satisfaction_rate) * n_vars / 4
    else:
        # Hard → large backdoor
        k_satisfaction = n_vars / 3
    
    # =========================================================================
    # Method 2: Variable degree analysis (graph structure)
    # =========================================================================
    var_degrees = Counter()
    for clause in clauses:
        for lit in clause:
            var_degrees[abs(lit)] += 1
    
    # High-degree variables are more likely to be critical
    # But for random SAT, most variables have similar degree
    degrees = np.array(list(var_degrees.values()))
    if len(degrees) > 0:
        avg_degree = np.mean(degrees)
        degree_std = np.std(degrees)
        
        # If degrees are concentrated → random structure (k ~ log N)
        # If degrees vary widely → some hubs exist (k smaller)
        if degree_std / avg_degree < 0.3:
            # Uniform degrees → random structure
            k_degree = np.log2(n_vars)
        else:
            # Some hubs exist
            median_degree = np.median(degrees)
            high_degree_vars = np.sum(degrees > 1.5 * median_degree)
            k_degree = min(n_vars / 4, 0.5 + np.log2(max(1, high_degree_vars)))
    else:
        k_degree = np.log2(n_vars)
    
    # =========================================================================
    # Method 3: Energy distribution (landscape shape)
    # =========================================================================
    energy_std = np.std(energies)
    energy_range = np.max(energies) - np.min(energies)
    
    # Count how many states are near-optimal
    near_optimal = np.sum(energies <= best_energy + 2)
    frac_near_optimal = near_optimal / len(energies)
    
    # If many near-optimal states → structured (small k)
    # If few near-optimal states → random/hard (medium k)
    if frac_near_optimal > 0.1:
        # Many good states → small backdoor
        k_landscape = max(1.0, np.log2(n_vars) * (1 - frac_near_optimal))
    elif frac_near_optimal > 0.01:
        # Some good states → medium backdoor
        k_landscape = np.log2(n_vars) * 1.2
    else:
        # Few good states → harder instance
        k_landscape = np.log2(n_vars) * 1.5
    
    # =========================================================================
    # Combine methods with weights
    # =========================================================================
    # Satisfaction method is most reliable
    k_estimate = (
        0.6 * k_satisfaction +
        0.25 * k_degree +
        0.15 * k_landscape
    )
    
    # Clamp to reasonable range
    k_estimate = np.clip(k_estimate, 1.0, n_vars * 0.6)
    
    # Confidence based on method agreement
    estimates = [k_satisfaction, k_degree, k_landscape]
    estimate_std = np.std(estimates)
    estimate_mean = np.mean(estimates)
    
    # If methods agree → high confidence
    if estimate_std < 0.3 * estimate_mean:
        confidence = 0.85
    elif estimate_std < 0.5 * estimate_mean:
        confidence = 0.75
    else:
        confidence = 0.60
    
    # Diagnostics
    diagnostics = {
        'k_satisfaction': k_satisfaction,
        'k_degree': k_degree,
        'k_landscape': k_landscape,
        'best_energy': best_energy,
        'satisfaction_rate': satisfaction_rate,
        'samples_used': samples,
        'energy_std': energy_std,
        'estimate_agreement': 1.0 - (estimate_std / estimate_mean) if estimate_mean > 0 else 0.0
    }
    
    return k_estimate, confidence, diagnostics


def smart_routing_thresholds(n_vars: int) -> Dict[str, float]:
    """
    Data-driven routing thresholds (calibrated heuristically).
    
    Based on complexity theory:
    - Quantum advantage when √(2^k) << 2^k → k << N
    - Practical cutoff: k ≤ log2(N) for strong advantage
    """
    log_n = np.log2(n_vars)
    
    return {
        'quantum_threshold': log_n,           # k ≤ log N → O(√2^k) tractable
        'hybrid_threshold': n_vars / 4,       # k ≤ N/4 → O(2^k) tractable  
        'scaffolding_threshold': n_vars / 2,  # k ≤ N/2 → guided search helps
        'min_confidence': 0.55                # Below this, don't trust estimate (lowered for heuristic)
    }


# Quick test
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'benchmarks'))
    
    # Simple generator (avoid imports)
    def generate_random_3sat(n_vars, n_clauses, seed=None):
        if seed is not None:
            np.random.seed(seed)
        clauses = []
        for _ in range(n_clauses):
            vars = np.random.choice(n_vars, size=3, replace=False) + 1
            signs = np.random.choice([-1, 1], size=3)
            clause = tuple(int(s * v) for s, v in zip(signs, vars))
            clauses.append(clause)
        return clauses
    
    print("Testing improved k estimation...\n")
    
    test_cases = [
        ("Easy (under-constrained)", 10, 25),
        ("Medium (phase transition)", 12, 50),
        ("Hard (over-constrained)", 14, 60),
    ]
    
    for name, n, m in test_cases:
        clauses = generate_random_3sat(n, m, seed=42)
        k_est, conf, diag = estimate_k_improved(clauses, n, samples=300)
        
        thresholds = smart_routing_thresholds(n)
        
        print(f"{name}: N={n}, M={m}")
        print(f"  k_estimate: {k_est:.2f}")
        print(f"  Confidence: {conf:.1%}")
        print(f"  Breakdown: satisfaction={diag['k_satisfaction']:.2f}, "
              f"degree={diag['k_degree']:.2f}, landscape={diag['k_landscape']:.2f}")
        print(f"  Best satisfaction: {diag['satisfaction_rate']:.1%}")
        print(f"  Quantum threshold: {thresholds['quantum_threshold']:.2f}")
        
        if k_est <= thresholds['quantum_threshold'] and conf >= thresholds['min_confidence']:
            print(f"  → Recommend: QUANTUM ✅")
        elif k_est <= thresholds['hybrid_threshold']:
            print(f"  → Recommend: HYBRID")
        else:
            print(f"  → Recommend: CLASSICAL")
        print()
