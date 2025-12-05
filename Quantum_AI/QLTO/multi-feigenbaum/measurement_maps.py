"""
measurement_maps.py

Library of measurement probability functions for Paper 5:
"Beyond Phase Expression: Universality Classes of Measurement-Induced Chaos"

This module provides different measurement maps M(θ) and their associated
bifurcation dynamics. The key insight is that ALL smooth unimodal maps
with quadratic maximum should give Feigenbaum δ = 4.669...

Maps Implemented:
1. sin²(θ/2) - Standard Hadamard test (Papers 1-4)
2. cos²(θ/2) - Equivalent (phase shifted)
3. (1 + cos θ)/2 - Ramsey interferometry
4. exp(-θ²/σ²) - Gaussian weak measurement
5. |sin θ| - Cusp map (different universality class!)
6. Weak measurement family - Parameterized by strength g

"""

import numpy as np
from typing import Callable, Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MeasurementMap:
    """Container for a measurement map with metadata."""
    name: str
    func: Callable[[float], float]
    description: str
    maximum_type: str  # 'quadratic', 'cusp', 'flat'
    expected_delta: float  # Predicted Feigenbaum constant


# =============================================================================
# CORE MEASUREMENT MAPS
# =============================================================================

def sin2_map(x: float) -> float:
    """
    Standard sin² map for bifurcation analysis: f(x) = sin²(πx)
    
    This is equivalent to the Hadamard test map:
    - P(|1⟩) = sin²(φ/2) with φ = 2πx
    - Quadratic maximum at x = 0.5
    - Expected δ = 4.669...
    
    For iteration: x_{n+1} = r · sin²(π·x_n)
    """
    return np.sin(np.pi * x) ** 2


def cos2_map(x: float) -> float:
    """
    Complementary map: f(x) = cos²(πx)
    
    Phase-shifted version of sin²:
    - Maximum at x = 0, minimum at x = 0.5
    - Same universality class as sin²
    """
    return np.cos(np.pi * x) ** 2


def ramsey_map(x: float) -> float:
    """
    Ramsey-style map: f(x) = (1 + cos(2πx))/2 = cos²(πx)
    
    Mathematically equivalent to cos²:
    - Included for completeness
    """
    return (1 + np.cos(2 * np.pi * x)) / 2


def tent_map(x: float) -> float:
    """
    Tent map: f(x) = 1 - |2x - 1|
    
    This has a CUSP (non-smooth) maximum at x = 0.5:
    - Different universality class!
    - Period-doubling still occurs but with different constant
    """
    return 1 - np.abs(2 * x - 1)


def gaussian_map(x: float, sigma: float = 0.3) -> float:
    """
    Gaussian map: f(x) = exp(-((x-0.5)/σ)²)
    
    Has a FLAT maximum (not quadratic) when σ is large:
    - For small σ: Approximate quadratic behavior
    - For large σ: Flat plateau → different dynamics
    """
    return np.exp(-((x - 0.5) / sigma) ** 2)


def cubic_map(x: float) -> float:
    """
    Cubic map: f(x) = 4x(1-x)(1-2x+2x²) normalized
    
    Has cubic behavior near maximum:
    - Different universality class than quadratic
    """
    # Simple quadratic for comparison (same as logistic just rescaled)
    return 4 * x * (1 - x) / 1.0  # Max at x=0.5, f(0.5)=1


def weak_measurement_map(theta: float, g: float) -> float:
    """
    Weak measurement with strength g ∈ [0, 1].
    
    P(|1⟩) = (1/2) + (g/2)·sin(θ) + O(g²)
    
    As g → 1: Approaches projective (sin²-like)
    As g → 0: Flat (no information, no chaos)
    
    This allows exploring the measurement-strength phase transition!
    """
    # Linear approximation for weak measurement
    # Full formula would need quantum state evolution
    if g < 0.01:
        return 0.5  # No information extracted
    
    # Interpolate between flat (g=0) and sin² (g=1)
    flat_part = 0.5
    projective_part = np.sin(theta / 2) ** 2
    
    return (1 - g) * flat_part + g * projective_part


# =============================================================================
# BIFURCATION DYNAMICS
# =============================================================================

def iterate_map(
    M: Callable[[float], float],
    x0: float,
    r: float,
    n_transient: int = 5000,
    n_sample: int = 500
) -> np.ndarray:
    """
    Iterate the feedback map x_{n+1} = r · M(x_n)
    
    Args:
        M: Measurement probability function M(x) mapping [0,1] → [0,1]
        x0: Initial condition x ∈ [0, 1]
        r: Control parameter (like learning rate)
        n_transient: Iterations to discard (wait for attractor)
        n_sample: Iterations to record
    
    Returns:
        Array of attractor samples
    """
    x = x0
    
    # Transient
    for _ in range(n_transient):
        x = r * M(x)
        # Keep x in valid range
        if x < 0 or x > 1:
            x = np.clip(x, 0.001, 0.999)
    
    # Sample attractor
    samples = np.zeros(n_sample)
    for i in range(n_sample):
        x = r * M(x)
        if x < 0 or x > 1:
            x = np.clip(x, 0.001, 0.999)
        samples[i] = x
    
    return samples


def detect_period(attractor: np.ndarray, tol: float = 1e-6) -> int:
    """
    Detect the period of an attractor from samples.
    
    Returns:
        Period (1, 2, 4, 8, ...) or 0 if chaotic/undetermined
    """
    for p in [1, 2, 4, 8, 16, 32, 64]:
        if len(attractor) >= 2 * p:
            # Check if all samples p apart are the same
            is_periodic = True
            for i in range(p):
                subset = attractor[i::p]
                if np.std(subset) > tol:
                    is_periodic = False
                    break
            if is_periodic:
                return p
    return 0  # Chaotic or high period


def find_bifurcation_points(
    M: Callable[[float], float],
    r_min: float = 0.5,
    r_max: float = 1.0,
    n_points: int = 10000
) -> List[Dict]:
    """
    Find period-doubling bifurcation points for a given map.
    
    Returns list of dicts with:
        - r: bifurcation parameter value
        - from_period: period before bifurcation
        - to_period: period after bifurcation
    """
    rs = np.linspace(r_min, r_max, n_points)
    
    bifurcations = []
    prev_period = 0
    seen_from = set()
    
    for r in rs:
        attractor = iterate_map(M, 0.4, r, n_transient=2000, n_sample=200)
        period = detect_period(attractor)
        
        if period > prev_period > 0 and period == 2 * prev_period:
            if prev_period not in seen_from:
                bifurcations.append({
                    'r': r,
                    'from_period': prev_period,
                    'to_period': period
                })
                seen_from.add(prev_period)
        
        if period > 0:
            prev_period = period
    
    return bifurcations


def compute_feigenbaum_delta(bifurcations: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    Compute Feigenbaum δ ratios from bifurcation points.
    
    δ_n = (r_{n} - r_{n-1}) / (r_{n+1} - r_n)
    
    Should converge to 4.669... for quadratic maximum maps.
    
    Returns:
        (deltas, ratios) - interval widths and their ratios
    """
    if len(bifurcations) < 3:
        return [], []
    
    rs = [b['r'] for b in bifurcations]
    deltas = [rs[i+1] - rs[i] for i in range(len(rs) - 1)]
    
    ratios = []
    for i in range(len(deltas) - 1):
        if deltas[i+1] > 1e-10:
            ratios.append(deltas[i] / deltas[i+1])
    
    return deltas, ratios


def compute_lyapunov_exponent(
    M: Callable[[float], float],
    r: float,
    x0: float = 0.4,
    n_iter: int = 10000
) -> float:
    """
    Compute the Lyapunov exponent for the map x_{n+1} = r·M(x_n).
    
    λ = lim (1/N) Σ log|f'(x_n)|
    
    λ > 0: Chaotic
    λ < 0: Stable periodic
    λ = 0: Edge of chaos
    """
    x = x0
    lyapunov_sum = 0.0
    
    # Small perturbation for numerical derivative
    epsilon = 1e-8
    
    for _ in range(n_iter):
        # Numerical derivative of f(x) = r·M(x)
        f_x = r * M(x)
        f_x_eps = r * M(x + epsilon)
        derivative = (f_x_eps - f_x) / epsilon
        
        if abs(derivative) > 1e-12:
            lyapunov_sum += np.log(abs(derivative))
        
        x = np.clip(f_x, 1e-10, 1 - 1e-10)
    
    return lyapunov_sum / n_iter


# =============================================================================
# MAP REGISTRY
# =============================================================================

def get_measurement_maps() -> Dict[str, MeasurementMap]:
    """
    Get dictionary of all available measurement maps.
    """
    return {
        'sin2': MeasurementMap(
            name='sin²(πx)',
            func=sin2_map,
            description='Standard Hadamard test (Papers 1-4)',
            maximum_type='quadratic',
            expected_delta=4.669
        ),
        'cos2': MeasurementMap(
            name='cos²(πx)',
            func=cos2_map,
            description='Complementary probability',
            maximum_type='quadratic',
            expected_delta=4.669
        ),
        'logistic': MeasurementMap(
            name='4x(1-x)',
            func=cubic_map,
            description='Logistic map (classic chaos)',
            maximum_type='quadratic',
            expected_delta=4.669
        ),
        'gaussian': MeasurementMap(
            name='exp(-(x-0.5)²/σ²)',
            func=lambda x: gaussian_map(x, sigma=0.3),
            description='Gaussian weak measurement',
            maximum_type='quadratic',  # with small sigma, approximately quadratic
            expected_delta=4.669
        ),
        'tent': MeasurementMap(
            name='1-|2x-1|',
            func=tent_map,
            description='Tent map (cusp maximum)',
            maximum_type='cusp',
            expected_delta=0.0  # Different behavior - immediate chaos
        ),
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MEASUREMENT MAPS LIBRARY")
    print("Paper 5: Beyond Phase Expression")
    print("=" * 70)
    
    FEIGENBAUM = 4.669201609
    maps = get_measurement_maps()
    
    for key, mmap in maps.items():
        print(f"\n--- {mmap.name} ({mmap.maximum_type} maximum) ---")
        print(f"Description: {mmap.description}")
        
        # Find bifurcations
        bifurcations = find_bifurcation_points(mmap.func, r_min=0.5, r_max=1.0)
        
        if len(bifurcations) >= 3:
            print(f"Found {len(bifurcations)} bifurcations:")
            for b in bifurcations[:4]:
                print(f"  Period {b['from_period']} → {b['to_period']} at r = {b['r']:.6f}")
            
            deltas, ratios = compute_feigenbaum_delta(bifurcations)
            if ratios:
                best_ratio = ratios[-1] if ratios else 0
                error = abs(best_ratio - FEIGENBAUM) / FEIGENBAUM * 100
                print(f"Best δ estimate: {best_ratio:.4f} (error: {error:.1f}%)")
                print(f"Expected δ: {mmap.expected_delta}")
        else:
            print(f"Not enough bifurcations found (need ≥3, got {len(bifurcations)})")
            print("This may indicate no chaos (flat map) or different dynamics")
    
    print("\n" + "=" * 70)
    print("Library loaded successfully!")
    print("=" * 70)
