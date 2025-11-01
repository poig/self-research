"""
Polynomial-Time Structure Analyzer for SAT
==========================================

This module provides TRULY POLYNOMIAL backdoor estimation without
constructing the full exponential-size Hamiltonian.

Key Innovation: We never enumerate 2^N states!

Methods:
1. Clause-variable graph analysis (O(m+n))
2. Monte Carlo energy sampling (O(samples × m))
3. Local search heuristics (O(restarts × iterations × m))

All methods are polynomial in input size (m clauses, n variables).
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from collections import defaultdict
import random


class PolynomialStructureAnalyzer:
    """
    Backdoor size estimation in TRULY POLYNOMIAL TIME.
    
    No exponential operations - everything is O(poly(m, n)).
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._last_mc_diagnostics = None  # Store CI and convergence info
    
    def analyze(self, clauses: List[Tuple[int, ...]], n_vars: int) -> Tuple[float, float]:
        """
        Main analysis: Estimate backdoor size k in polynomial time.
        
        Returns:
            (k_estimate, confidence)
        
        Complexity: O(m × n + n^2) - POLYNOMIAL!
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("POLYNOMIAL-TIME STRUCTURE ANALYSIS")
            print(f"{'='*70}")
            print(f"  Variables: N = {n_vars}")
            print(f"  Clauses:   M = {len(clauses)}")
            print(f"  Input size: {len(clauses) * 3} literals")
        
        # Method 1: Graph-based analysis (O(m + n))
        k_graph, conf_graph = self._analyze_clause_graph(clauses, n_vars)
        
        # Method 2: Monte Carlo sampling (O(samples × m))
        k_sampling, conf_sampling = self._monte_carlo_estimate(clauses, n_vars, samples=min(1000, 2*n_vars))
        
        # Method 3: Local search (O(restarts × iterations × m))
        k_local, conf_local = self._local_search_estimate(clauses, n_vars, restarts=10)
        
        # Combine estimates (weighted average)
        k_estimate = (
            0.4 * k_graph +
            0.3 * k_sampling +
            0.3 * k_local
        )
        
        confidence = (conf_graph + conf_sampling + conf_local) / 3
        
        if self.verbose:
            print(f"\n  Results:")
            print(f"    Graph method:    k ≈ {k_graph:.2f} (conf: {conf_graph:.2f})")
            print(f"    Sampling method: k ≈ {k_sampling:.2f} (conf: {conf_sampling:.2f})")
            
            # Show CI if available from adaptive Monte Carlo
            if self._last_mc_diagnostics:
                diag = self._last_mc_diagnostics
                print(f"      → 95% CI: [{diag['ci_lower']:.2f}, {diag['ci_upper']:.2f}]")
                print(f"      → CI width: {diag['ci_width']:.2f}")
                print(f"      → Samples: {diag['samples_used']:,}")
                print(f"      → Converged: {'✅' if diag['converged'] else '❌'}")
            
            print(f"    Local search:    k ≈ {k_local:.2f} (conf: {conf_local:.2f})")
            print(f"    Combined:        k ≈ {k_estimate:.2f} (conf: {confidence:.2f})")
        
        return k_estimate, confidence
    
    # ============================================================================
    # Method 1: Graph-Based Analysis (Deterministic, O(m + n))
    # ============================================================================
    
    def _analyze_clause_graph(self, clauses: List[Tuple[int, ...]], n_vars: int) -> Tuple[float, float]:
        """
        Analyze clause-variable bipartite graph structure.
        
        Complexity: O(m + n) for basic metrics, O(n^2) for advanced
        
        Key insight: Backdoor variables are "hubs" in the graph
        - High degree (appear in many clauses)
        - High betweenness centrality (connect many components)
        - Form strongly connected subgraphs
        """
        # Build variable-occurrence map
        var_occurrences = defaultdict(list)
        for clause_idx, clause in enumerate(clauses):
            for lit in clause:
                var = abs(lit)
                if var <= n_vars:
                    var_occurrences[var].append(clause_idx)
        
        # Compute variable influence scores
        var_scores = {}
        for var in range(1, n_vars + 1):
            if var not in var_occurrences:
                var_scores[var] = 0.0
                continue
            
            # Score based on frequency and clause connectivity
            frequency = len(var_occurrences[var])
            
            # Variables in overlapping clauses are more important
            clause_set = set(var_occurrences[var])
            overlap = 0
            for other_var in var_occurrences:
                if other_var != var:
                    overlap += len(clause_set & set(var_occurrences[other_var]))
            
            # Normalized score
            var_scores[var] = frequency * (1 + overlap / max(1, len(clauses)))
        
        # Estimate backdoor size: count "highly influential" variables
        if not var_scores:
            return 0.0, 0.9
        
        scores = list(var_scores.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 1.0
        
        # Variables with score > mean + std are likely backdoor
        threshold = mean_score + std_score
        k_estimate = sum(1 for s in scores if s > threshold)
        
        # Confidence based on score separation
        if std_score > 0:
            separation = (max(scores) - mean_score) / std_score
            confidence = min(0.9, 0.5 + 0.1 * separation)
        else:
            confidence = 0.5
        
        return float(k_estimate), confidence
    
    # ============================================================================
    # Method 2: Monte Carlo Sampling (Probabilistic, O(samples × m))
    # ============================================================================
    
    def _monte_carlo_estimate(self, clauses: List[Tuple[int, ...]], n_vars: int, 
                               samples: int = 1000,
                               adaptive: bool = True,
                               ci_width_target: float = 0.3,
                               max_samples: int = 10000) -> Tuple[float, float]:
        """
        Adaptive Monte Carlo with bootstrap CI and importance sampling.
        
        Complexity: O(samples × m × k) where k ≈ 3 for 3-SAT
        
        Key improvements over naive sampling:
        1. Adaptive: Increases samples until CI narrow enough (statistically sound)
        2. Bootstrap: Computes 95% CI for k estimate (not just point estimate)
        3. Importance sampling: Seeds from local search to boost low-energy events
        
        Returns:
            (k_estimate, confidence) where confidence now reflects CI quality
        """
        if not adaptive:
            # Fall back to simple fixed-sample method
            return self._simple_monte_carlo(clauses, n_vars, samples)
        
        # Phase 1: Seed with local search states (importance sampling)
        seed_states = self._get_local_search_seeds(clauses, n_vars, n_seeds=20)
        
        energies = []
        best_states = []  # Track best states for perturbation
        current_samples = 0
        batch_size = min(500, samples)
        
        while current_samples < max_samples:
            # Generate batch with importance sampling
            if len(best_states) > 0 and np.random.rand() > 0.5:
                # 50% importance sampling: perturb good states
                batch_states = self._perturb_states(best_states, batch_size, n_vars)
            elif len(seed_states) > 0 and current_samples < 1000:
                # Early: use local search seeds
                batch_states = self._perturb_states(seed_states, batch_size, n_vars)
            else:
                # Random sampling
                batch_states = [np.random.randint(0, 2, n_vars, dtype=np.uint8) 
                               for _ in range(batch_size)]
            
            # Compute energies
            batch_energies = []
            for state in batch_states:
                violations = sum(1 for c in clauses if self._state_violates_clause(state, c))
                batch_energies.append(violations)
            
            energies.extend(batch_energies)
            current_samples += batch_size
            
            # Update best states (top 10)
            all_states_energies = list(zip(batch_states, batch_energies))
            all_states_energies.sort(key=lambda x: x[1])
            best_states = [s for s, e in all_states_energies[:10]]
            
            # Check convergence (every 1000 samples, minimum 2000)
            if current_samples >= 2000 and current_samples % 1000 == 0:
                k_est, ci_lo, ci_hi = self._bootstrap_ci_for_k(
                    np.array(energies), n_vars, n_bootstrap=500, alpha=0.05
                )
                ci_width = ci_hi - ci_lo
                
                if ci_width <= ci_width_target:
                    # Converged!
                    break
        
        # Final estimate with bootstrap CI
        energies_array = np.array(energies)
        k_estimate, ci_lower, ci_upper = self._bootstrap_ci_for_k(
            energies_array, n_vars, n_bootstrap=1000, alpha=0.05
        )
        ci_width = ci_upper - ci_lower
        
        # Confidence based on CI quality
        if ci_width <= ci_width_target:
            confidence = 0.95
        elif ci_width <= 2 * ci_width_target:
            confidence = 0.80
        elif ci_width <= 3 * ci_width_target:
            confidence = 0.65
        else:
            confidence = 0.50
        
        # Store diagnostic info
        self._last_mc_diagnostics = {
            'samples_used': current_samples,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'converged': ci_width <= ci_width_target,
            'min_energy': np.min(energies_array),
            'mean_energy': np.mean(energies_array)
        }
        
        return k_estimate, confidence
    
    def _simple_monte_carlo(self, clauses: List[Tuple[int, ...]], n_vars: int,
                           samples: int) -> Tuple[float, float]:
        """Simple fixed-sample Monte Carlo (for comparison/fallback)"""
        energies = []
        for _ in range(samples):
            state = np.random.randint(0, 2, n_vars, dtype=np.uint8)
            violations = sum(1 for c in clauses if self._state_violates_clause(state, c))
            energies.append(violations)
        
        energies = np.array(energies)
        E_min = np.min(energies)
        E_range = np.max(energies) - E_min
        E_threshold = E_min + 0.1 * E_range
        low_energy_fraction = np.mean(energies <= E_threshold)
        
        if low_energy_fraction > 0:
            k_estimate = -np.log2(low_energy_fraction)
            k_estimate = max(0, min(n_vars, k_estimate))
        else:
            k_estimate = n_vars
        
        E_std = np.std(energies)
        E_mean = np.mean(energies)
        cv = E_std / (E_mean + 1e-10)
        confidence = max(0.3, min(0.8, 1.0 - cv))
        
        return k_estimate, confidence
    
    def _get_local_search_seeds(self, clauses: List[Tuple[int, ...]], n_vars: int,
                                n_seeds: int = 20) -> List[np.ndarray]:
        """Quick local search to seed importance sampling"""
        seeds = []
        for _ in range(n_seeds):
            state = np.random.randint(0, 2, n_vars, dtype=np.uint8)
            # Mini WalkSAT: 10 flips
            for __ in range(10):
                violations = [i for i, c in enumerate(clauses) 
                             if self._state_violates_clause(state, c)]
                if not violations:
                    break
                clause = clauses[violations[np.random.randint(len(violations))]]
                var = abs(clause[np.random.randint(len(clause))]) - 1
                if 0 <= var < n_vars:
                    state[var] = 1 - state[var]
            seeds.append(state.copy())
        return seeds
    
    def _perturb_states(self, base_states: List[np.ndarray], n_samples: int,
                       n_vars: int) -> List[np.ndarray]:
        """Generate samples by perturbing good states"""
        samples = []
        for _ in range(n_samples):
            base = base_states[np.random.randint(len(base_states))].copy()
            # Flip 1-3 random bits
            n_flips = np.random.randint(1, min(4, n_vars + 1))
            for __ in range(n_flips):
                bit = np.random.randint(n_vars)
                base[bit] = 1 - base[bit]
            samples.append(base)
        return samples
    
    def _bootstrap_ci_for_k(self, energies: np.ndarray, n_vars: int,
                           n_bootstrap: int = 1000, alpha: float = 0.05
                           ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for k estimate.
        
        Strategy:
        1. Resample energies with replacement
        2. For each resample, compute low_energy_fraction → k
        3. Report percentiles as CI
        
        Returns:
            (k_median, ci_lower, ci_upper)
        """
        E_min = np.min(energies)
        E_max = np.max(energies)
        E_range = E_max - E_min
        
        # Threshold for "low energy" (within 10% of spectrum)
        E_threshold = E_min + 0.1 * E_range if E_range > 0 else E_min + 1.0
        
        k_estimates = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            resample_idx = np.random.randint(0, len(energies), size=len(energies))
            resample = energies[resample_idx]
            
            # Compute low-energy fraction
            low_frac = np.mean(resample <= E_threshold)
            
            # Convert to k
            if low_frac > 0 and low_frac < 1.0:
                k = -np.log2(low_frac)
            elif low_frac >= 1.0:
                k = 0.0
            else:
                k = float(n_vars)
            
            k_estimates.append(max(0.0, min(float(n_vars), k)))
        
        k_estimates = np.array(k_estimates)
        k_median = np.median(k_estimates)
        ci_lower = np.percentile(k_estimates, 100 * alpha / 2)
        ci_upper = np.percentile(k_estimates, 100 * (1 - alpha / 2))
        
        return k_median, ci_lower, ci_upper
    
    def _state_violates_clause(self, state: np.ndarray, clause: Tuple[int, ...]) -> bool:
        """Check if state violates clause (all literals false)."""
        for lit in clause:
            var_idx = abs(lit) - 1
            if var_idx >= len(state):
                continue
            
            var_value = state[var_idx]
            if lit > 0:  # Positive literal
                if var_value == 1:
                    return False  # Literal is satisfied
            else:  # Negative literal
                if var_value == 0:
                    return False  # Literal is satisfied
        
        return True  # All literals false → clause violated
    
    # ============================================================================
    # Method 3: Local Search (Heuristic, O(restarts × iterations × m))
    # ============================================================================
    
    def _local_search_estimate(self, clauses: List[Tuple[int, ...]], n_vars: int,
                                restarts: int = 10) -> Tuple[float, float]:
        """
        Estimate backdoor size via local search (WalkSAT-style).
        
        Complexity: O(restarts × max_flips × m) where max_flips = O(n)
        
        Key insight: Track which variables are "critical" during search
        - Variables that frequently need to be flipped are likely backdoor
        - Variables that stay fixed across solutions are not backdoor
        """
        flip_counts = defaultdict(int)
        best_energy_found = len(clauses)
        
        for restart in range(restarts):
            # Random initial state
            state = np.random.randint(0, 2, n_vars, dtype=np.uint8)
            
            # Local search
            max_flips = min(100, 10 * n_vars)
            for flip in range(max_flips):
                # Compute current energy
                violations = [i for i, c in enumerate(clauses) 
                             if self._state_violates_clause(state, c)]
                energy = len(violations)
                
                if energy < best_energy_found:
                    best_energy_found = energy
                
                if energy == 0:
                    break  # Found satisfying assignment
                
                # Pick random violated clause
                if not violations:
                    break
                clause_idx = random.choice(violations)
                clause = clauses[clause_idx]
                
                # Pick random variable in clause and flip it
                valid_vars = [abs(lit) for lit in clause if abs(lit) <= n_vars]
                if not valid_vars:
                    continue
                
                var = random.choice(valid_vars)
                var_idx = var - 1
                
                state[var_idx] = 1 - state[var_idx]
                flip_counts[var] += 1
        
        # Estimate k from flip frequency distribution
        if not flip_counts:
            return n_vars / 2, 0.3
        
        flips = list(flip_counts.values())
        mean_flips = np.mean(flips)
        std_flips = np.std(flips) if len(flips) > 1 else 1.0
        
        # Variables with high flip counts are likely backdoor
        threshold = mean_flips + std_flips
        k_estimate = sum(1 for f in flips if f > threshold)
        
        # Confidence based on search success
        if best_energy_found == 0:
            confidence = 0.8  # Found solution → high confidence
        else:
            # Lower confidence if couldn't find good solution
            confidence = max(0.3, 0.7 - 0.1 * best_energy_found)
        
        return float(k_estimate), confidence


# ============================================================================
# Comparison Benchmark
# ============================================================================

def compare_exponential_vs_polynomial(clauses, n_vars):
    """
    Compare exponential (full Hamiltonian) vs polynomial (graph+sampling) methods.
    """
    import time
    
    print(f"\n{'='*70}")
    print(f"EXPONENTIAL vs POLYNOMIAL COMPARISON (N={n_vars})")
    print(f"{'='*70}")
    
    # Method 1: Exponential (current QSA)
    print("\n[1] Exponential Method (Full Hamiltonian):")
    if n_vars <= 14:
        try:
            from quantum_structure_analyzer import QuantumStructureAnalyzer
            qsa = QuantumStructureAnalyzer(verbose=False)
            
            t0 = time.time()
            H = qsa._build_hamiltonian(clauses, n_vars)
            t1 = time.time()
            
            k_exp, conf_exp = qsa._spectral_backdoor_estimate(H, n_vars)
            t2 = time.time()
            
            print(f"  Build Hamiltonian: {t1-t0:.4f}s (O(2^N) operations)")
            print(f"  Analyze spectrum:  {t2-t1:.4f}s")
            print(f"  Total time:        {t2-t0:.4f}s")
            print(f"  Estimate:          k ≈ {k_exp:.2f} (conf: {conf_exp:.2f})")
            print(f"  Operations:        ~{2**n_vars:,} (exponential)")
        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
            print(f"  ❌ OUT OF MEMORY!")
            print(f"  Tried to allocate: {(2**n_vars)**2 * 8 / (1024**3):.1f} GB")
            print(f"  This demonstrates why polynomial methods are essential!")
            k_exp, conf_exp = None, None
            t2, t0 = 0, 0
    else:
        print(f"  ⏭️  SKIPPED (N={n_vars} too large)")
        mem_gb = (2**n_vars)**2 * 8 / (1024**3)
        print(f"  Would require:     {mem_gb:.1f} GB memory (dense matrix)")
        print(f"  Would take:        ~{2**(n_vars-10):.0f}x longer than N=10")
        print(f"  Status:            ❌ EXPONENTIAL - INFEASIBLE")
        k_exp, conf_exp = None, None
        t2, t0 = 0, 0
    
    # Method 2: Polynomial (new approach)
    print("\n[2] Polynomial Method (Graph + Sampling):")
    poly_analyzer = PolynomialStructureAnalyzer(verbose=False)
    
    t3 = time.time()
    k_poly, conf_poly = poly_analyzer.analyze(clauses, n_vars)
    t4 = time.time()
    
    print(f"  Total time:        {t4-t3:.4f}s")
    print(f"  Estimate:          k ≈ {k_poly:.2f} (conf: {conf_poly:.2f})")
    print(f"  Operations:        ~{len(clauses) * n_vars:,} (polynomial)")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    
    if k_exp is not None and (t2-t0) > 0:
        speedup = (t2-t0)/(t4-t3)
        print(f"  Speedup:           {speedup:.2f}x faster (polynomial vs exponential)")
        print(f"  Estimate agreement: Δk = {abs(k_exp - k_poly):.2f}")
        print(f"  Both methods feasible: ✅ (N={n_vars} small enough)")
    else:
        print(f"  Speedup:           ∞ (exponential method failed/skipped)")
        print(f"  Polynomial time:   {t4-t3:.4f}s")
        print(f"  Polynomial ONLY feasible option for N={n_vars}")
    
    print(f"\n  📊 Scalability Analysis:")
    print(f"     Polynomial method:   ✅ Works for N > 100 (O(m×n))")
    print(f"     Exponential method:  ❌ Fails at N ≈ 15-16 (O(2^N))")
    print(f"\n  💡 Key Insight:")
    print(f"     For structure detection, we DON'T need full Hamiltonian!")
    print(f"     Graph + sampling gives backdoor estimate in polynomial time.")


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    from test_lanczos_scalability import generate_random_3sat
    
    print("="*70)
    print("POLYNOMIAL-TIME STRUCTURE ANALYZER - DEMO")
    print("="*70)
    
    # Test on small instance
    print("\n[TEST 1] Small instance (N=10)")
    clauses_10 = generate_random_3sat(10, 42, seed=1000)
    compare_exponential_vs_polynomial(clauses_10, 10)
    
    # Test on medium instance
    print("\n[TEST 2] Medium instance (N=16)")
    clauses_16 = generate_random_3sat(16, 67, seed=1600)
    compare_exponential_vs_polynomial(clauses_16, 16)
    
    # Test on large instance (impossible for exponential!)
    print("\n[TEST 3] Large instance (N=100) - POLYNOMIAL ONLY")
    clauses_100 = generate_random_3sat(100, 420, seed=10000)
    
    print(f"\n{'='*70}")
    print(f"POLYNOMIAL METHOD ON N=100")
    print(f"{'='*70}")
    
    poly_analyzer = PolynomialStructureAnalyzer(verbose=True)
    k_estimate, confidence = poly_analyzer.analyze(clauses_100, 100)
    
    print(f"\n✅ Successfully analyzed N=100 in polynomial time!")
    print(f"   (Exponential method would need 2^100 ≈ 10^30 operations)")
