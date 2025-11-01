"""
Fast CDCL Probe for Structure Detection
========================================

Extremely cheap probe (1s budget) that provides strong signals about problem structure.

Key metrics extracted:
- Conflict growth rate (exponential = hard, linear = structured)
- Learned clause rate (high = structure)
- Unit propagation progress (high = easy)
- Backbone variables found (structure indicator)

This is MUCH cheaper than heavy sampling and often sufficient for decision-making.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Set
from collections import deque


class CDCLProbe:
    """
    Lightweight CDCL-inspired probe that runs for ~1 second.
    
    Extracts structural features without full solving:
    - Conflict analysis patterns
    - Unit propagation effectiveness
    - Clause learning patterns
    - Backbone detection
    
    Cost: O(conflicts × m × n) ≈ O(1000 × m × n) for 1s budget
    """
    
    def __init__(self, time_budget: float = 1.0, max_conflicts: int = 1000):
        self.time_budget = time_budget
        self.max_conflicts = max_conflicts
    
    def probe(self, clauses: List[Tuple[int, ...]], n_vars: int) -> Dict:
        """
        Run cheap CDCL probe and extract structural features.
        
        Returns:
            dict with keys:
                - conflict_rate: conflicts per second
                - learned_clause_rate: clauses learned per conflict
                - unit_prop_efficiency: literals propagated per decision
                - backbone_size: estimated backbone variables
                - branching_factor: average choices per decision
                - early_termination: whether solved/UNSAT in budget
                - decision_quality: how good random decisions were
                - predicted_difficulty: "easy" / "medium" / "hard"
        """
        start_time = time.time()
        
        # State: current assignment
        assignment = {}  # var -> value (True/False)
        decision_stack = []  # Stack of decisions
        
        # Metrics
        conflicts = 0
        decisions = 0
        unit_props = 0
        learned_clauses = []
        backbone_candidates = set(range(n_vars))
        
        # Preprocessing: unit propagation
        initial_units = self._find_unit_clauses(clauses, assignment)
        for var, val in initial_units:
            assignment[var] = val
            unit_props += 1
        
        # Check if already SAT/UNSAT
        status = self._check_status(clauses, assignment)
        if status in ['SAT', 'UNSAT']:
            elapsed = time.time() - start_time
            return {
                'conflict_rate': 0.0,
                'learned_clause_rate': 0.0,
                'unit_prop_efficiency': len(initial_units),
                'backbone_size': len(assignment),
                'branching_factor': 1.0,
                'early_termination': True,
                'decision_quality': 1.0,
                'predicted_difficulty': 'easy',
                'probe_time': elapsed,
                'conflicts': 0,
                'decisions': 0
            }
        
        # Main CDCL-like loop
        while (time.time() - start_time) < self.time_budget and conflicts < self.max_conflicts:
            # Make a random decision (simplified heuristic)
            free_vars = [v for v in range(n_vars) if v not in assignment]
            if not free_vars:
                break
            
            # Pick high-degree variable (VSIDS-like)
            var_degrees = self._compute_var_degrees(clauses, assignment)
            var = max(free_vars, key=lambda v: var_degrees.get(v, 0))
            val = np.random.choice([True, False])
            
            assignment[var] = val
            decision_stack.append((var, val))
            decisions += 1
            
            # Unit propagation
            props_before = len(assignment)
            units = self._find_unit_clauses(clauses, assignment)
            for u_var, u_val in units:
                if u_var in assignment and assignment[u_var] != u_val:
                    # Conflict!
                    conflicts += 1
                    
                    # Analyze conflict (simplified)
                    conflict_clause = self._analyze_conflict(clauses, assignment, u_var)
                    learned_clauses.append(conflict_clause)
                    
                    # Backtrack (simplified: just remove last decision)
                    if decision_stack:
                        last_var, last_val = decision_stack.pop()
                        del assignment[last_var]
                        # Remove propagated literals
                        assignment = {v: val for v, val in assignment.items() 
                                     if (v, val) in decision_stack or v in [u[0] for u in initial_units]}
                    break
                else:
                    assignment[u_var] = u_val
                    unit_props += 1
            
            props_after = len(assignment)
            
            # Update backbone candidates
            backbone_candidates &= set(assignment.keys())
        
        elapsed = time.time() - start_time
        
        # Compute metrics
        conflict_rate = conflicts / elapsed if elapsed > 0 else 0
        learned_rate = len(learned_clauses) / conflicts if conflicts > 0 else 0
        unit_efficiency = unit_props / decisions if decisions > 0 else 0
        branching_factor = len([v for v in range(n_vars) if v not in assignment]) / max(1, decisions)
        decision_quality = 1.0 - (conflicts / max(1, decisions))
        
        # Predict difficulty
        if conflict_rate < 50 and unit_efficiency > 2:
            difficulty = "easy"
        elif conflict_rate < 200 and learned_rate > 0.5:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        return {
            'conflict_rate': conflict_rate,
            'learned_clause_rate': learned_rate,
            'unit_prop_efficiency': unit_efficiency,
            'backbone_size': len(backbone_candidates),
            'branching_factor': branching_factor,
            'early_termination': False,
            'decision_quality': decision_quality,
            'predicted_difficulty': difficulty,
            'probe_time': elapsed,
            'conflicts': conflicts,
            'decisions': decisions,
            'learned_clauses_count': len(learned_clauses)
        }
    
    def _find_unit_clauses(self, clauses: List[Tuple[int, ...]], 
                          assignment: Dict[int, bool]) -> List[Tuple[int, bool]]:
        """Find unit clauses given current assignment"""
        units = []
        for clause in clauses:
            unassigned = []
            satisfied = False
            
            for lit in clause:
                var = abs(lit) - 1
                val_needed = lit > 0
                
                if var in assignment:
                    if assignment[var] == val_needed:
                        satisfied = True
                        break
                else:
                    unassigned.append((var, val_needed))
            
            if not satisfied and len(unassigned) == 1:
                units.append(unassigned[0])
        
        return units
    
    def _check_status(self, clauses: List[Tuple[int, ...]], 
                     assignment: Dict[int, bool]) -> str:
        """Check if current assignment satisfies formula or creates conflict"""
        all_satisfied = True
        has_conflict = False
        
        for clause in clauses:
            satisfied = False
            all_false = True
            
            for lit in clause:
                var = abs(lit) - 1
                val_needed = lit > 0
                
                if var in assignment:
                    if assignment[var] == val_needed:
                        satisfied = True
                        break
                    # else: literal is false
                else:
                    all_false = False
            
            if all_false:
                has_conflict = True
            if not satisfied:
                all_satisfied = False
        
        if has_conflict:
            return 'UNSAT'
        if all_satisfied:
            return 'SAT'
        return 'UNKNOWN'
    
    def _compute_var_degrees(self, clauses: List[Tuple[int, ...]], 
                            assignment: Dict[int, bool]) -> Dict[int, int]:
        """Compute variable degrees in unassigned clauses"""
        degrees = {}
        for clause in clauses:
            # Check if clause already satisfied
            satisfied = False
            for lit in clause:
                var = abs(lit) - 1
                val_needed = lit > 0
                if var in assignment and assignment[var] == val_needed:
                    satisfied = True
                    break
            
            if not satisfied:
                for lit in clause:
                    var = abs(lit) - 1
                    if var not in assignment:
                        degrees[var] = degrees.get(var, 0) + 1
        
        return degrees
    
    def _analyze_conflict(self, clauses: List[Tuple[int, ...]], 
                         assignment: Dict[int, bool], conflict_var: int) -> Tuple[int, ...]:
        """Simplified conflict analysis - return a learned clause"""
        # In real CDCL, this would do resolution
        # Here we just return a simple clause involving recent decisions
        recent_vars = list(assignment.keys())[-3:]  # Last 3 decisions
        learned = tuple(-(v+1) if assignment[v] else (v+1) for v in recent_vars)
        return learned


def should_skip_heavy_analysis(probe_results: Dict, 
                              fast_threshold: float = 0.3) -> Tuple[bool, str]:
    """
    Decide whether to skip heavy sampling based on probe.
    
    Args:
        probe_results: Output from CDCLProbe.probe()
        fast_threshold: Threshold for "fast enough to skip analysis"
    
    Returns:
        (should_skip, reason)
    """
    # Case 1: Solved in probe
    if probe_results['early_termination']:
        return True, f"Solved in probe ({probe_results['probe_time']:.3f}s)"
    
    # Case 2: Very easy (high unit propagation, low conflicts)
    if (probe_results['predicted_difficulty'] == 'easy' and 
        probe_results['unit_prop_efficiency'] > 3):
        return True, f"Easy instance (unit_prop={probe_results['unit_prop_efficiency']:.1f})"
    
    # Case 3: Extremely hard (high conflict rate, low learning)
    if (probe_results['predicted_difficulty'] == 'hard' and
        probe_results['conflict_rate'] > 500):
        return True, f"Adversarial instance (conflict_rate={probe_results['conflict_rate']:.0f})"
    
    # Case 4: Probe time already exceeded fast threshold
    if probe_results['probe_time'] > fast_threshold:
        # If probe took this long, classical solver is probably best
        return True, f"Probe took {probe_results['probe_time']:.2f}s (>threshold)"
    
    return False, "Probe inconclusive, proceeding with analysis"


# ============================================================================
# Integration Example
# ============================================================================

def enhanced_pipeline_with_probe(clauses: List[Tuple[int, ...]], n_vars: int):
    """
    Example of integrated pipeline with CDCL probe early exit.
    
    Flow:
    1. Run 1-second CDCL probe
    2. If probe says "easy" or "hard" → skip heavy analysis
    3. Otherwise → run adaptive Monte Carlo
    4. Dispatch based on combined information
    """
    from src.core.polynomial_structure_analyzer import PolynomialStructureAnalyzer
    from src.core.safe_dispatcher import SafeDispatcher
    
    print(f"Analyzing instance: N={n_vars}, M={len(clauses)}")
    
    # Phase 0: Cheap CDCL probe
    print("\n[0/3] Running CDCL probe (1s budget)...")
    probe = CDCLProbe(time_budget=1.0, max_conflicts=1000)
    probe_results = probe.probe(clauses, n_vars)
    
    print(f"  Probe results:")
    print(f"    Difficulty: {probe_results['predicted_difficulty']}")
    print(f"    Conflict rate: {probe_results['conflict_rate']:.0f}/s")
    print(f"    Unit prop efficiency: {probe_results['unit_prop_efficiency']:.1f}")
    print(f"    Backbone size: {probe_results['backbone_size']}")
    print(f"    Decision quality: {probe_results['decision_quality']:.2%}")
    print(f"    Time: {probe_results['probe_time']:.3f}s")
    
    # Check if we can skip heavy analysis
    skip, reason = should_skip_heavy_analysis(probe_results)
    
    if skip:
        print(f"\n  ⚡ FAST PATH: {reason}")
        print(f"  → Skipping heavy analysis, dispatching directly")
        
        # Make quick decision based on probe
        if probe_results['early_termination']:
            return "solved_in_probe", probe_results
        elif probe_results['predicted_difficulty'] == 'easy':
            return "use_classical", probe_results
        else:
            return "use_robust_classical", probe_results
    
    # Phase 1: Heavy analysis (only if probe was inconclusive)
    print(f"\n[1/3] Probe inconclusive, running full analysis...")
    analyzer = PolynomialStructureAnalyzer(verbose=False)
    k_estimate, confidence = analyzer.analyze(clauses, n_vars)
    
    diag = analyzer._last_mc_diagnostics
    
    print(f"  k ≈ {k_estimate:.2f} ± {diag['ci_width']/2:.2f}")
    print(f"  Confidence: {confidence:.2%}")
    
    # Phase 2: Dispatch
    print(f"\n[2/3] Dispatching...")
    dispatcher = SafeDispatcher(confidence_threshold=0.70, enable_verification=True)
    
    decision = dispatcher.dispatch(
        k_estimate=k_estimate,
        confidence=confidence,
        ci_lower=diag['ci_lower'],
        ci_upper=diag['ci_upper'],
        n_vars=n_vars,
        clauses=clauses,
        estimator_diagnostics=diag
    )
    
    print(f"  Solver: {decision.solver.value}")
    print(f"  Reason: {decision.reason}")
    
    return decision, probe_results


if __name__ == "__main__":
    from test_lanczos_scalability import generate_random_3sat
    
    print("="*70)
    print("CDCL PROBE DEMONSTRATION")
    print("="*70)
    
    # Test on various instances
    test_cases = [
        (10, 30, "easy"),
        (12, 50, "medium"),
        (14, 58, "hard"),
    ]
    
    for n, m, name in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {name} (N={n}, M={m})")
        print(f"{'='*70}")
        
        clauses = generate_random_3sat(n, m, seed=n*100)
        result, probe_results = enhanced_pipeline_with_probe(clauses, n)
        
        print(f"\nResult: {result}")
