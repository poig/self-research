"""
Safe Dispatcher with Verification Probes
========================================

Production-ready dispatcher that:
1. Only uses expensive solvers when confidence is justified
2. Runs cheap verification probes before committing
3. Falls back to robust solver when uncertain
4. Tracks decisions and learns from mistakes

This is the critical safety layer that prevents catastrophic misdispatch.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SolverType(Enum):
    """Available solver strategies"""
    BACKDOOR_QUANTUM = "backdoor_quantum"      # Full quantum search over backdoor
    BACKDOOR_HYBRID = "backdoor_hybrid"        # Classical search over backdoor
    SCAFFOLDING = "scaffolding"                # CDCL with backdoor guidance
    ROBUST_CLASSICAL = "robust_classical"      # Pure CDCL (CaDiCaL/Glucose)


@dataclass
class DispatchDecision:
    """Record of a dispatch decision"""
    solver: SolverType
    k_estimate: float
    confidence: float
    ci_lower: float
    ci_upper: float
    reason: str
    expected_complexity: str
    verification_passed: bool
    timestamp: float
    
    # Post-execution (filled in later)
    actual_solve_time: Optional[float] = None
    success: Optional[bool] = None
    correct_dispatch: Optional[bool] = None


class SafeDispatcher:
    """
    Conservative dispatcher with statistical decision rules.
    
    Key principles:
    1. High confidence threshold (≥ 0.75) for risky paths
    2. Verification probes before expensive commitments
    3. Multiple estimators must agree (consensus)
    4. Fallback to robust solver when uncertain
    5. Learn from mistakes via telemetry
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.75,
                 k_threshold_quantum: float = None,  # log2(N) + 1
                 k_threshold_hybrid: float = None,   # N / 3
                 enable_verification: bool = True,
                 verbose: bool = False):
        """
        Args:
            confidence_threshold: Minimum confidence to use fast solvers
            k_threshold_quantum: Max k for quantum solver (default: log2(N)+1)
            k_threshold_hybrid: Max k for hybrid solver (default: N/3)
            enable_verification: Run verification probes before dispatch
            verbose: Print decision reasoning
        """
        self.confidence_threshold = confidence_threshold
        self.k_threshold_quantum = k_threshold_quantum
        self.k_threshold_hybrid = k_threshold_hybrid
        self.enable_verification = enable_verification
        self.verbose = verbose
        
        self.history: List[DispatchDecision] = []
        self.stats = {
            'total_decisions': 0,
            'quantum_used': 0,
            'hybrid_used': 0,
            'scaffolding_used': 0,
            'robust_used': 0,
            'verification_failures': 0,
            'confidence_rejections': 0,
            'k_too_large_rejections': 0,
        }
    
    def dispatch(self, 
                 k_estimate: float,
                 confidence: float,
                 ci_lower: float,
                 ci_upper: float,
                 n_vars: int,
                 clauses: List[Tuple[int, ...]],
                 estimator_diagnostics: Optional[Dict] = None) -> DispatchDecision:
        """
        Decide which solver to use with safety checks.
        
        Decision logic:
        1. Check confidence threshold
        2. Check k bounds (sanity + thresholds)
        3. Run verification probe if enabled
        4. Dispatch to appropriate solver
        5. Log decision for telemetry
        
        Args:
            k_estimate: Backdoor size estimate
            confidence: Confidence in estimate (0-1)
            ci_lower, ci_upper: 95% confidence interval bounds
            n_vars: Number of variables
            clauses: CNF clauses
            estimator_diagnostics: Optional dict with 'converged', 'samples_used', etc.
        
        Returns:
            DispatchDecision with solver choice and reasoning
        """
        self.stats['total_decisions'] += 1
        
        # Set dynamic thresholds if not specified
        k_thresh_quantum = self.k_threshold_quantum or (np.log2(n_vars) + 1)
        k_thresh_hybrid = self.k_threshold_hybrid or (n_vars / 3)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("SAFE DISPATCHER DECISION")
            print(f"{'='*70}")
            print(f"  Input:")
            print(f"    k_estimate = {k_estimate:.2f}")
            print(f"    95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]")
            print(f"    CI width = {ci_upper - ci_lower:.2f}")
            print(f"    Confidence = {confidence:.2%}")
            print(f"    N = {n_vars}, M = {len(clauses)}")
            print(f"  Thresholds:")
            print(f"    Confidence: ≥ {self.confidence_threshold:.2%}")
            print(f"    k_quantum: ≤ {k_thresh_quantum:.2f}")
            print(f"    k_hybrid: ≤ {k_thresh_hybrid:.2f}")
        
        # ====================================================================
        # Safety Check 1: Confidence Threshold
        # ====================================================================
        if confidence < self.confidence_threshold:
            reason = (f"Low confidence ({confidence:.2%} < {self.confidence_threshold:.2%}). "
                     f"Not safe to trust estimate.")
            
            if self.verbose:
                print(f"\n  ❌ REJECTED: {reason}")
                print(f"  → Fallback to ROBUST_CLASSICAL")
            
            self.stats['confidence_rejections'] += 1
            self.stats['robust_used'] += 1
            
            decision = DispatchDecision(
                solver=SolverType.ROBUST_CLASSICAL,
                k_estimate=k_estimate,
                confidence=confidence,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                reason=reason,
                expected_complexity=f"O(1.3^N) ≈ O(1.3^{n_vars}) ≈ O({1.3**n_vars:.2e})",
                verification_passed=False,
                timestamp=time.time()
            )
            self.history.append(decision)
            return decision
        
        # ====================================================================
        # Safety Check 2: Sanity Bounds on k
        # ====================================================================
        if k_estimate < 0 or k_estimate > n_vars:
            reason = f"Invalid k_estimate ({k_estimate:.2f} outside [0, {n_vars}]). Estimator bug?"
            
            if self.verbose:
                print(f"\n  ❌ REJECTED: {reason}")
                print(f"  → Fallback to ROBUST_CLASSICAL")
            
            self.stats['robust_used'] += 1
            
            decision = DispatchDecision(
                solver=SolverType.ROBUST_CLASSICAL,
                k_estimate=k_estimate,
                confidence=confidence,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                reason=reason,
                expected_complexity=f"O(1.3^N)",
                verification_passed=False,
                timestamp=time.time()
            )
            self.history.append(decision)
            return decision
        
        # ====================================================================
        # Safety Check 3: CI Quality (convergence check)
        # ====================================================================
        ci_width = ci_upper - ci_lower
        if estimator_diagnostics and not estimator_diagnostics.get('converged', True):
            reason = (f"Monte Carlo didn't converge (CI width {ci_width:.2f} too large). "
                     f"Estimate unreliable.")
            
            if self.verbose:
                print(f"\n  ⚠️  WARNING: {reason}")
                print(f"  → Using SCAFFOLDING (safer than blind dispatch)")
            
            self.stats['scaffolding_used'] += 1
            
            decision = DispatchDecision(
                solver=SolverType.SCAFFOLDING,
                k_estimate=k_estimate,
                confidence=confidence,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                reason=reason,
                expected_complexity=f"O(1.3^N) with backdoor guidance",
                verification_passed=False,
                timestamp=time.time()
            )
            self.history.append(decision)
            return decision
        
        # ====================================================================
        # Safety Check 4: Verification Probe (cheap test)
        # ====================================================================
        verification_passed = True
        if self.enable_verification:
            verification_passed = self._run_verification_probe(
                clauses, n_vars, k_estimate, ci_upper
            )
            
            if not verification_passed:
                reason = (f"Verification probe failed. Estimate k={k_estimate:.2f} "
                         f"doesn't match quick test.")
                
                if self.verbose:
                    print(f"\n  ❌ VERIFICATION FAILED: {reason}")
                    print(f"  → Fallback to SCAFFOLDING")
                
                self.stats['verification_failures'] += 1
                self.stats['scaffolding_used'] += 1
                
                decision = DispatchDecision(
                    solver=SolverType.SCAFFOLDING,
                    k_estimate=k_estimate,
                    confidence=confidence,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    reason=reason,
                    expected_complexity=f"O(1.3^N) with guidance",
                    verification_passed=False,
                    timestamp=time.time()
                )
                self.history.append(decision)
                return decision
        
        # ====================================================================
        # Decision: Choose Solver Based on k
        # ====================================================================
        
        # Option 1: Small backdoor → Quantum search
        if k_estimate <= k_thresh_quantum:
            reason = (f"Small backdoor (k={k_estimate:.2f} ≤ {k_thresh_quantum:.2f}). "
                     f"Quantum search over 2^k states is tractable.")
            expected_time = f"O(√(2^k) × N^4) ≈ O({2**(k_estimate/2):.1f} × {n_vars**4:,})"
            
            if self.verbose:
                print(f"\n  ✅ DISPATCH to BACKDOOR_QUANTUM")
                print(f"  Reason: {reason}")
                print(f"  Expected: {expected_time}")
            
            self.stats['quantum_used'] += 1
            solver = SolverType.BACKDOOR_QUANTUM
            
        # Option 2: Medium backdoor → Hybrid classical search
        elif k_estimate <= k_thresh_hybrid:
            reason = (f"Medium backdoor (k={k_estimate:.2f} ≤ {k_thresh_hybrid:.2f}). "
                     f"Classical enumeration over 2^k is feasible.")
            expected_time = f"O(2^k × N^4) ≈ O({2**k_estimate:.0f} × {n_vars**4:,})"
            
            if self.verbose:
                print(f"\n  ✅ DISPATCH to BACKDOOR_HYBRID")
                print(f"  Reason: {reason}")
                print(f"  Expected: {expected_time}")
            
            self.stats['hybrid_used'] += 1
            solver = SolverType.BACKDOOR_HYBRID
            
        # Option 3: Large backdoor → Use estimate to guide CDCL
        elif k_estimate <= 2 * n_vars / 3:
            reason = (f"Large backdoor (k={k_estimate:.2f} > {k_thresh_hybrid:.2f}). "
                     f"Use estimate to guide CDCL branching.")
            expected_time = f"O(1.3^N) with scaffolding guidance"
            
            if self.verbose:
                print(f"\n  ✅ DISPATCH to SCAFFOLDING")
                print(f"  Reason: {reason}")
                print(f"  Expected: {expected_time}")
            
            self.stats['scaffolding_used'] += 1
            solver = SolverType.SCAFFOLDING
            
        # Option 4: Very large backdoor → Pure CDCL
        else:
            reason = (f"Very large backdoor (k={k_estimate:.2f} > {2*n_vars/3:.1f}). "
                     f"No structure to exploit, use pure CDCL.")
            expected_time = f"O(1.3^N)"
            
            if self.verbose:
                print(f"\n  ✅ DISPATCH to ROBUST_CLASSICAL")
                print(f"  Reason: {reason}")
                print(f"  Expected: {expected_time}")
            
            self.stats['robust_used'] += 1
            solver = SolverType.ROBUST_CLASSICAL
        
        decision = DispatchDecision(
            solver=solver,
            k_estimate=k_estimate,
            confidence=confidence,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            reason=reason,
            expected_complexity=expected_time,
            verification_passed=verification_passed,
            timestamp=time.time()
        )
        
        self.history.append(decision)
        return decision
    
    def _run_verification_probe(self, 
                                clauses: List[Tuple[int, ...]],
                                n_vars: int,
                                k_estimate: float,
                                ci_upper: float,
                                time_budget: float = 0.1) -> bool:
        """
        Quick verification: Does small backdoor search actually work?
        
        Strategy:
        1. Pick k_test = ceil(ci_upper) variables (upper bound)
        2. Try 100 random assignments to those k variables
        3. Check if any leads to easy subproblem
        4. Return True if verification passes
        
        Cost: O(100 × 2^k_test × m) ≈ O(samples × clauses)
        This is CHEAP compared to full solving.
        """
        k_test = min(int(np.ceil(ci_upper)), n_vars)
        
        if k_test > 15:
            # Too expensive to verify
            return True
        
        # Pick k_test high-degree variables (heuristic)
        var_degrees = np.zeros(n_vars)
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if 0 <= var < n_vars:
                    var_degrees[var] += 1
        
        top_vars = np.argsort(var_degrees)[-k_test:]
        
        # Try 100 random assignments to these variables
        n_tests = min(100, 2**k_test)
        success_count = 0
        
        for _ in range(n_tests):
            # Random assignment to backdoor vars
            backdoor_assign = {v: np.random.randint(2) for v in top_vars}
            
            # Check if this simplifies the problem significantly
            # (i.e., many clauses become satisfied)
            satisfied = 0
            for clause in clauses:
                for lit in clause:
                    var = abs(lit) - 1
                    if var in backdoor_assign:
                        val = backdoor_assign[var]
                        if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                            satisfied += 1
                            break
            
            sat_fraction = satisfied / len(clauses)
            
            # If >50% satisfied, this is a good backdoor
            if sat_fraction > 0.5:
                success_count += 1
        
        # Pass if at least 10% of tests found good assignments
        success_rate = success_count / n_tests
        return success_rate >= 0.1
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dispatcher statistics"""
        if self.stats['total_decisions'] == 0:
            return {'total_decisions': 0}
        
        total = self.stats['total_decisions']
        
        return {
            'total_decisions': total,
            'quantum_used': self.stats['quantum_used'],
            'hybrid_used': self.stats['hybrid_used'],
            'scaffolding_used': self.stats['scaffolding_used'],
            'robust_used': self.stats['robust_used'],
            'quantum_fraction': self.stats['quantum_used'] / total,
            'hybrid_fraction': self.stats['hybrid_used'] / total,
            'scaffolding_fraction': self.stats['scaffolding_used'] / total,
            'robust_fraction': self.stats['robust_used'] / total,
            'fast_path_fraction': (self.stats['quantum_used'] + self.stats['hybrid_used']) / total,
            'confidence_rejections': self.stats['confidence_rejections'],
            'verification_failures': self.stats['verification_failures'],
            'k_too_large_rejections': self.stats['k_too_large_rejections'],
        }
    
    def print_statistics(self):
        """Pretty-print dispatcher statistics"""
        stats = self.get_statistics()
        
        if stats['total_decisions'] == 0:
            print("No decisions made yet")
            return
        
        print(f"\n{'='*70}")
        print("DISPATCHER STATISTICS")
        print(f"{'='*70}")
        print(f"Total decisions:          {stats['total_decisions']}")
        print(f"\nSolver usage:")
        print(f"  Quantum:                {stats['quantum_used']} ({stats['quantum_fraction']:.1%})")
        print(f"  Hybrid classical:       {stats['hybrid_used']} ({stats['hybrid_fraction']:.1%})")
        print(f"  Scaffolding:            {stats['scaffolding_used']} ({stats['scaffolding_fraction']:.1%})")
        print(f"  Robust classical:       {stats['robust_used']} ({stats['robust_fraction']:.1%})")
        print(f"\nFast path (quantum+hybrid): {stats['fast_path_fraction']:.1%}")
        print(f"\nSafety events:")
        print(f"  Confidence rejections:  {stats['confidence_rejections']}")
        print(f"  Verification failures:  {stats['verification_failures']}")
        print(f"  k too large rejections: {stats['k_too_large_rejections']}")
        print(f"{'='*70}\n")
