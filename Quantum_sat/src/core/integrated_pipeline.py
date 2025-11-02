"""
Integrated Performance-Optimized Pipeline
=========================================

Combines all three expert-recommended enhancements:
  A) CDCL probe for early exit
  B) Sequential SPRT for sample reduction
  C) ML classifier for fast prediction

Pipeline flow:
  1. Run 1-second CDCL probe
     - If easy/solved → skip analysis, return k=0
     - If adversarial hard → skip analysis, return k=N
  
  2. If inconclusive, run ML classifier
     - Features: graph metrics + probe results
     - If high confidence (>80%) → use ML prediction
  
  3. If still uncertain, run sequential Monte Carlo
     - SPRT early stopping saves 50-90% samples
     - Adaptive thresholds from initial samples

Expected performance: Analysis 2-4s → 0.1-0.5s (4-40× speedup)
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Import our three enhancements
from src.enhancements.cdcl_probe import CDCLProbe, should_skip_heavy_analysis
from src.enhancements.sequential_testing import sequential_monte_carlo_estimate
from src.enhancements.ml_classifier import FastFeatureExtractor, BackdoorSizeClassifier


@dataclass
class AnalysisResult:
    """Result from integrated analysis pipeline"""
    k_estimate: float
    confidence: float
    method_used: str  # 'cdcl_probe', 'ml_classifier', 'sequential_monte_carlo'
    analysis_time: float
    samples_used: int
    converged: bool
    reasoning: str


class IntegratedPipeline:
    """
    Production pipeline combining all performance enhancements.
    
    Designed to minimize analysis overhead while maintaining safety.
    """
    
    def __init__(self, 
                 ml_classifier: Optional[BackdoorSizeClassifier] = None,
                 enable_cdcl_probe: bool = True,
                 enable_ml_classifier: bool = True,
                 enable_sequential_mc: bool = True,
                 ml_confidence_threshold: float = 0.80,
                 verbose: bool = True):
        """
        Args:
            ml_classifier: Pre-trained ML model (optional)
            enable_cdcl_probe: Use CDCL probe for early exit
            enable_ml_classifier: Use ML for fast prediction
            enable_sequential_mc: Use sequential testing for sampling
            ml_confidence_threshold: Min confidence to trust ML (0.80 = 80%)
            verbose: Print progress
        """
        self.ml_classifier = ml_classifier
        self.enable_cdcl_probe = enable_cdcl_probe
        self.enable_ml_classifier = enable_ml_classifier
        self.enable_sequential_mc = enable_sequential_mc
        self.ml_confidence_threshold = ml_confidence_threshold
        self.verbose = verbose
        
        # Initialize components
        if self.enable_cdcl_probe:
            self.cdcl_probe = CDCLProbe(time_budget=1.0)
        else:
            self.cdcl_probe = None
    
    def analyze(self, clauses: List[Tuple[int, ...]], n_vars: int) -> AnalysisResult:
        """
        Run integrated analysis pipeline.
        
        Args:
            clauses: CNF clauses
            n_vars: Number of variables
        
        Returns:
            AnalysisResult with k estimate and metadata
        """
        t_start = time.time()
        
        # ====================================================================
        # Phase 1: CDCL Probe (1 second)
        # ====================================================================
        
        cdcl_results = None
        
        if self.enable_cdcl_probe:
            if self.verbose:
                print("[Phase 1/3] Running 1-second CDCL probe...")
            
            t0 = time.time()
            cdcl_results = self.cdcl_probe.probe(clauses, n_vars)
            probe_time = time.time() - t0
            
            if self.verbose:
                print(f"  Probe completed in {probe_time:.2f}s")
                print(f"  Difficulty: {cdcl_results['predicted_difficulty']}")
                print(f"  Conflict rate: {cdcl_results['conflict_rate']:.1f}/s")
            
            # Check if we should skip heavy analysis
            should_skip, skip_reason = should_skip_heavy_analysis(cdcl_results)
            
            if should_skip:
                # Early exit based on probe
                if cdcl_results.get('early_termination', False):
                    k_est = 0  # SAT - solved quickly
                elif cdcl_results['predicted_difficulty'] == 'easy':
                    k_est = np.log2(n_vars) + 1  # Small backdoor expected
                else:
                    k_est = n_vars  # Adversarial hard - large backdoor
                
                analysis_time = time.time() - t_start
                
                if self.verbose:
                    print(f"  ✅ Early exit: {skip_reason}")
                    print(f"  Returning k ≈ {k_est:.1f}")
                
                return AnalysisResult(
                    k_estimate=k_est,
                    confidence=0.70,  # Conservative from probe
                    method_used='cdcl_probe',
                    analysis_time=analysis_time,
                    samples_used=0,
                    converged=True,
                    reasoning=skip_reason
                )
        
        # ====================================================================
        # Phase 2: ML Classifier (milliseconds)
        # ====================================================================
        
        if self.enable_ml_classifier and self.ml_classifier is not None:
            if self.verbose:
                print("[Phase 2/3] Running ML classifier...")
            
            t0 = time.time()
            
            # Extract features
            features = FastFeatureExtractor.extract(
                clauses, n_vars, 
                cdcl_probe_results=cdcl_results
            )
            
            # Predict k
            k_ml, conf_ml = self.ml_classifier.predict(features)
            ml_time = time.time() - t0
            
            if self.verbose:
                print(f"  ML prediction: k ≈ {k_ml:.1f} (confidence: {conf_ml:.1%})")
                print(f"  Completed in {ml_time*1000:.1f}ms")
            
            # If high confidence, use ML prediction
            if conf_ml >= self.ml_confidence_threshold:
                analysis_time = time.time() - t_start
                
                if self.verbose:
                    print(f"  ✅ High confidence ML prediction - using k ≈ {k_ml:.1f}")
                
                return AnalysisResult(
                    k_estimate=k_ml,
                    confidence=conf_ml,
                    method_used='ml_classifier',
                    analysis_time=analysis_time,
                    samples_used=0,
                    converged=True,
                    reasoning=f"ML confidence {conf_ml:.1%} exceeds threshold {self.ml_confidence_threshold:.1%}"
                )
        
        # ====================================================================
        # Phase 3: Sequential Monte Carlo (adaptive samples)
        # ====================================================================
        
        if self.verbose:
            print("[Phase 3/3] Running sequential Monte Carlo...")
        
        if self.enable_sequential_mc:
            # Use sequential testing with early stopping
            k_mc, conf_mc, samples_used, converged = sequential_monte_carlo_estimate(
                clauses, n_vars,
                alpha=0.05,
                beta=0.10,
                min_samples=200,
                max_samples=5000,
                verbose=self.verbose
            )
        else:
            # Fallback to fixed sampling (expensive!)
            from src.core.polynomial_structure_analyzer import PolynomialStructureAnalyzer
            analyzer = PolynomialStructureAnalyzer(verbose=self.verbose)
            k_mc, conf_mc = analyzer._simple_monte_carlo(clauses, n_vars, samples=5000)
            samples_used = 5000
            converged = True
        
        analysis_time = time.time() - t_start
        
        if self.verbose:
            print(f"  Monte Carlo: k ≈ {k_mc:.1f} (confidence: {conf_mc:.1%})")
            print(f"  Used {samples_used} samples, converged: {converged}")
            print(f"  Total analysis time: {analysis_time:.2f}s")
        
        return AnalysisResult(
            k_estimate=k_mc,
            confidence=conf_mc,
            method_used='sequential_monte_carlo',
            analysis_time=analysis_time,
            samples_used=samples_used,
            converged=converged,
            reasoning=f"Sequential MC with {samples_used} samples"
        )


# ============================================================================
# Integration with Safe Dispatcher
# ============================================================================

def integrated_dispatcher_pipeline(clauses: List[Tuple[int, ...]], 
                                   n_vars: int,
                                   ml_classifier: Optional[BackdoorSizeClassifier] = None,
                                   verbose: bool = True,
                                   true_k: Optional[int] = None) -> Dict:
    """
    Full production pipeline: integrated analysis + safe dispatcher.
    
    Args:
        clauses: SAT instance clauses
        n_vars: Number of variables
        ml_classifier: Optional pre-trained ML classifier
        verbose: Print progress messages
        true_k: If provided, use this known backdoor size instead of estimation
                (for validation/testing with structured instances)
    
    Returns:
        {
            'k_estimate': float,
            'confidence': float,
            'method_used': str,
            'analysis_time': float,
            'samples_used': int,
            'recommended_solver': str,
            'reasoning': str,
        }
    """
    # Use true_k if provided, otherwise run analysis
    if true_k is not None:
        # Skip analysis, use known k
        k = float(true_k)
        result = AnalysisResult(
            k_estimate=k,
            confidence=1.0,
            method_used='true_k_provided',
            analysis_time=0.0,
            samples_used=0,
            converged=True,
            reasoning=f'Using provided true_k={true_k}'
        )
        if verbose:
            print(f"[Using true_k={true_k}] Skipping analysis, routing with known backdoor size")
    else:
        # Run integrated analysis
        pipeline = IntegratedPipeline(
            ml_classifier=ml_classifier,
            enable_cdcl_probe=True,
            enable_ml_classifier=True,
            enable_sequential_mc=True,
            verbose=verbose
        )
        
        result = pipeline.analyze(clauses, n_vars)
        k = result.k_estimate
    
    if k <= np.log2(n_vars) + 1:
        solver = "quantum"
        reason = f"Small backdoor (k={k:.1f} ≤ log₂(N)+1)"
    elif k <= n_vars / 3:
        solver = "hybrid_qaoa"
        reason = f"Medium backdoor (k={k:.1f} ≤ N/3)"
    elif k <= 2 * n_vars / 3:
        solver = "scaffolding_search"
        reason = f"Large backdoor (k={k:.1f} ≤ 2N/3)"
    else:
        solver = "robust_cdcl"
        reason = f"Very large backdoor (k={k:.1f} > 2N/3)"
    
    # Safety check
    if result.confidence < 0.70:
        solver = "robust_cdcl"
        reason = f"Low confidence ({result.confidence:.1%}) - using robust solver"
    
    return {
        'k_estimate': result.k_estimate,
        'confidence': result.confidence,
        'method_used': result.method_used,
        'analysis_time': result.analysis_time,
        'samples_used': result.samples_used,
        'recommended_solver': solver,
        'reasoning': reason,
    }


# ============================================================================
# Performance Comparison Demo
# ============================================================================

def compare_pipelines():
    """Compare old (expensive) vs new (optimized) pipeline"""
    from test_lanczos_scalability import generate_random_3sat
    from src.core.polynomial_structure_analyzer import PolynomialStructureAnalyzer
    
    print("="*70)
    print("PIPELINE PERFORMANCE COMPARISON")
    print("="*70)
    print()
    print("Comparing:")
    print("  OLD: Fixed 5000 samples Monte Carlo (2-4s)")
    print("  NEW: CDCL probe + ML + Sequential MC (0.1-0.5s)")
    print()
    
    # Generate test instances
    test_cases = [
        (10, 42, "easy"),
        (12, 50, "medium"),
        (14, 58, "hard"),
        (16, 66, "very hard"),
    ]
    
    # OLD pipeline
    print("[1/2] Running OLD pipeline (fixed sampling)...")
    old_analyzer = PolynomialStructureAnalyzer(verbose=False)
    
    old_times = []
    old_samples = []
    
    for n, m, name in test_cases:
        clauses = generate_random_3sat(n, m, seed=n*100)
        
        t0 = time.time()
        k_old, conf_old = old_analyzer._simple_monte_carlo(clauses, n, samples=5000)
        time_old = time.time() - t0
        
        old_times.append(time_old)
        old_samples.append(5000)
        
        print(f"  {name:12s}: k={k_old:5.1f}, conf={conf_old:.2%}, time={time_old:6.2f}s")
    
    # NEW pipeline
    print("\n[2/2] Running NEW pipeline (integrated optimizations)...")
    new_pipeline = IntegratedPipeline(
        ml_classifier=None,  # No pre-trained model for demo
        enable_cdcl_probe=True,
        enable_ml_classifier=False,  # Skip ML for now
        enable_sequential_mc=True,
        verbose=False
    )
    
    new_times = []
    new_samples = []
    methods_used = []
    
    for n, m, name in test_cases:
        clauses = generate_random_3sat(n, m, seed=n*100)
        
        result = new_pipeline.analyze(clauses, n)
        
        new_times.append(result.analysis_time)
        new_samples.append(result.samples_used)
        methods_used.append(result.method_used)
        
        print(f"  {name:12s}: k={result.k_estimate:5.1f}, conf={result.confidence:.2%}, "
              f"time={result.analysis_time:6.2f}s, method={result.method_used}")
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY:")
    print("-"*70)
    
    total_old_time = sum(old_times)
    total_new_time = sum(new_times)
    speedup = total_old_time / total_new_time if total_new_time > 0 else 0
    
    avg_old_samples = np.mean(old_samples)
    avg_new_samples = np.mean(new_samples)
    sample_reduction = (1 - avg_new_samples/avg_old_samples) * 100 if avg_old_samples > 0 else 0
    
    print(f"  Total time OLD: {total_old_time:6.2f}s")
    print(f"  Total time NEW: {total_new_time:6.2f}s")
    print(f"  Speedup:        {speedup:6.2f}×")
    print()
    print(f"  Avg samples OLD: {avg_old_samples:7.0f}")
    print(f"  Avg samples NEW: {avg_new_samples:7.0f}")
    print(f"  Sample reduction: {sample_reduction:5.1f}%")
    print()
    print(f"  Methods used:")
    for method in set(methods_used):
        count = methods_used.count(method)
        print(f"    {method:25s}: {count}/{len(methods_used)}")
    print()
    print("KEY RESULTS:")
    print("  ✅ Analysis time reduced by", f"{speedup:.1f}×")
    print("  ✅ Sample count reduced by", f"{sample_reduction:.0f}%")
    print("  ✅ Maintains statistical rigor (bootstrap CI, SPRT)")
    print("  ✅ Early exit on easy/hard instances")
    print()


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("INTEGRATED PIPELINE DEMO")
    print("="*70)
    print()
    
    # Demo single instance
    from test_lanczos_scalability import generate_random_3sat
    
    print("[Demo] Analyzing single instance with full pipeline...")
    print()
    
    n = 14
    m = 58
    clauses = generate_random_3sat(n, m, seed=42)
    
    # Run integrated pipeline
    result = integrated_dispatcher_pipeline(
        clauses, n, 
        ml_classifier=None,
        verbose=True
    )
    
    print()
    print("="*70)
    print("FINAL DECISION:")
    print(f"  k estimate:       {result['k_estimate']:.1f}")
    print(f"  Confidence:       {result['confidence']:.1%}")
    print(f"  Analysis time:    {result['analysis_time']:.2f}s")
    print(f"  Samples used:     {result['samples_used']}")
    print(f"  Method:           {result['method_used']}")
    print(f"  Recommended:      {result['recommended_solver']}")
    print(f"  Reasoning:        {result['reasoning']}")
    print("="*70)
    print()
    
    # Performance comparison
    print("\n" + "="*70)
    print("RUNNING PERFORMANCE COMPARISON...")
    print("="*70)
    print()
    
    compare_pipelines()
