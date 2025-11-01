"""
Benchmark Harness for SAT Instance Analysis
===========================================

Process a folder of DIMACS CNF files and perform comprehensive spectral analysis.

**Input**: Directory containing .cnf files (DIMACS format)
**Output**: 
  - CSV file with analysis results for each instance
  - Histograms of estimated backdoor sizes
  - Coverage plots (fraction with k ≤ log N, k ≤ N/4, etc.)
  - Runtime and scalability statistics

**Usage**:
  python sat_benchmark_harness.py --input <dimacs_folder> --output <results_folder>

**DIMACS Format Example**:
  c Comment line
  p cnf 10 20  (10 variables, 20 clauses)
  1 -2 3 0     (clause: x1 ∨ ¬x2 ∨ x3)
  -1 2 -3 0
  ...

**Analysis Pipeline**:
  1. Parse DIMACS file
  2. Build Hamiltonian H = Σ_c Π_c
  3. Compute spectral measures (gap, PR, spacing)
  4. Estimate backdoor size k
  5. Classify instance hardness
  6. Log results and timing
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

sys.path.insert(0, '.')
from quantum_structure_analyzer import QuantumStructureAnalyzer


def parse_dimacs(filepath: str) -> Tuple[List[Tuple[int, ...]], int]:
    """
    Parse DIMACS CNF file
    
    Args:
        filepath: Path to .cnf file
    
    Returns:
        clauses: List of clauses (tuples of literals)
        n_vars: Number of variables
    """
    clauses = []
    n_vars = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments
            if line.startswith('c'):
                continue
            
            # Parse problem line
            if line.startswith('p cnf'):
                parts = line.split()
                n_vars = int(parts[2])
                continue
            
            # Parse clause
            if line and not line.startswith('c') and not line.startswith('p'):
                literals = [int(x) for x in line.split() if int(x) != 0]
                if literals:
                    clauses.append(tuple(literals))
    
    return clauses, n_vars


def analyze_instance(filepath: str, qsa: QuantumStructureAnalyzer, 
                     timeout: float = 300.0) -> Dict:
    """
    Analyze a single SAT instance
    
    Args:
        filepath: Path to DIMACS file
        qsa: QuantumStructureAnalyzer instance
        timeout: Maximum time (seconds) for analysis
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Parse instance
        clauses, n_vars = parse_dimacs(filepath)
        n_clauses = len(clauses)
        
        # Skip if too large
        if n_vars > 28:
            return {
                'filename': os.path.basename(filepath),
                'status': 'SKIPPED',
                'reason': 'N too large (> 28)',
                'n_vars': n_vars,
                'n_clauses': n_clauses
            }
        
        # Run analysis with timeout
        start_time = time.time()
        
        # Store clauses
        qsa._current_clauses = clauses
        qsa._current_n_vars = n_vars
        
        # Build Hamiltonian
        H = qsa._build_hamiltonian(clauses, n_vars)
        
        # Estimate backdoor size using spectral method
        if n_vars <= 10:
            # Small: exact diagonalization
            H_matrix = H.to_matrix()
            evals = np.linalg.eigvalsh(H_matrix)
            evals = np.sort(evals)
            
            gap = evals[1] - evals[0] if len(evals) > 1 else 0.0
            ground_energy = evals[0]
            
            # Compute spectral measures
            PR = len(evals) / np.sum(evals**2) if np.sum(evals**2) > 0 else 1.0
            
            spacings = np.diff(evals)
            spacings = spacings[spacings > 1e-10]
            if len(spacings) > 1:
                r_vals = [min(spacings[i], spacings[i+1]) / max(spacings[i], spacings[i+1]) 
                         for i in range(len(spacings)-1) if max(spacings[i], spacings[i+1]) > 0]
                r_mean = np.mean(r_vals) if r_vals else 0.0
            else:
                r_mean = 0.0
            
            method = "exact"
            
        else:
            # Large: Lanczos
            k_estimate, confidence = qsa._lanczos_spectral_estimate(H, n_vars)
            
            # Get gap from Lanczos
            from scipy.sparse.linalg import eigsh, LinearOperator
            diag = qsa._compute_diagonal_vectorized(clauses, n_vars)
            dim = 2 ** n_vars
            
            def matvec(v):
                return diag * v
            
            linop = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
            k_compute = min(20, dim - 2)
            
            evals, _ = eigsh(linop, k=k_compute, which='SA')
            evals = np.sort(evals)
            
            gap = evals[1] - evals[0] if len(evals) > 1 else 0.0
            ground_energy = evals[0]
            
            # Spectral measures from sampled spectrum
            PR = k_compute / np.sum(evals**2) if np.sum(evals**2) > 0 else 1.0
            
            spacings = np.diff(evals)
            spacings = spacings[spacings > 1e-10]
            if len(spacings) > 1:
                r_vals = [min(spacings[i], spacings[i+1]) / max(spacings[i], spacings[i+1]) 
                         for i in range(len(spacings)-1) if max(spacings[i], spacings[i+1]) > 0]
                r_mean = np.mean(r_vals) if r_vals else 0.0
            else:
                r_mean = 0.0
            
            method = "lanczos"
        
        # Estimate k from gap
        if gap > 1e-10:
            k_from_gap = -np.log2(gap)
            k_from_gap = max(0, min(n_vars, k_from_gap))
        else:
            k_from_gap = n_vars
        
        # Classify instance
        if k_from_gap <= np.log2(n_vars + 1):
            hardness = "quasi-polynomial"
        elif k_from_gap <= n_vars / 4:
            hardness = "exponential-tractable"
        else:
            hardness = "exponential-hard"
        
        analysis_time = time.time() - start_time
        
        # Check timeout
        if analysis_time > timeout:
            return {
                'filename': os.path.basename(filepath),
                'status': 'TIMEOUT',
                'n_vars': n_vars,
                'n_clauses': n_clauses,
                'time': analysis_time
            }
        
        return {
            'filename': os.path.basename(filepath),
            'status': 'SUCCESS',
            'n_vars': n_vars,
            'n_clauses': n_clauses,
            'ratio': n_clauses / n_vars if n_vars > 0 else 0,
            'ground_energy': float(ground_energy),
            'spectral_gap': float(gap),
            'k_estimate': float(k_from_gap),
            'participation_ratio': float(PR),
            'level_spacing_r': float(r_mean),
            'hardness': hardness,
            'method': method,
            'time': analysis_time
        }
        
    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'status': 'ERROR',
            'error': str(e),
            'n_vars': n_vars if 'n_vars' in locals() else 0,
            'n_clauses': n_clauses if 'n_clauses' in locals() else 0
        }


def run_benchmark(input_dir: str, output_dir: str, max_instances: int = None):
    """
    Run benchmark on all DIMACS files in directory
    
    Args:
        input_dir: Directory containing .cnf files
        output_dir: Directory for results
        max_instances: Maximum number of instances to process (None = all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .cnf files
    cnf_files = list(Path(input_dir).glob('**/*.cnf'))
    
    if max_instances:
        cnf_files = cnf_files[:max_instances]
    
    print(f"Found {len(cnf_files)} DIMACS files in {input_dir}")
    
    if len(cnf_files) == 0:
        print("No .cnf files found!")
        return
    
    # Initialize QSA
    qsa = QuantumStructureAnalyzer(use_ml=False)
    
    # Process instances
    results = []
    
    for filepath in tqdm(cnf_files, desc="Analyzing instances"):
        result = analyze_instance(str(filepath), qsa)
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Generate plots
    generate_plots(df, output_dir)
    
    # Print summary
    print_summary(df)


def generate_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate visualization plots
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    # Filter successful results
    df_success = df[df['status'] == 'SUCCESS'].copy()
    
    if len(df_success) == 0:
        print("⚠️  No successful analyses to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Backdoor size distribution
    axes[0, 0].hist(df_success['k_estimate'], bins=30, alpha=0.7, 
                    color='blue', edgecolor='black')
    axes[0, 0].axvline(df_success['k_estimate'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f"Mean = {df_success['k_estimate'].mean():.2f}")
    axes[0, 0].set_xlabel('Estimated Backdoor Size k', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Distribution of Backdoor Sizes', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: k vs N scatter
    axes[0, 1].scatter(df_success['n_vars'], df_success['k_estimate'], alpha=0.6)
    axes[0, 1].plot(df_success['n_vars'], df_success['n_vars'], '--', 
                    color='red', alpha=0.5, label='k = N')
    axes[0, 1].plot(df_success['n_vars'], 
                    df_success['n_vars'].apply(lambda x: np.log2(x+1)), 
                    '--', color='green', alpha=0.5, label='k = log₂(N)')
    axes[0, 1].set_xlabel('Number of Variables (N)', fontsize=12)
    axes[0, 1].set_ylabel('Estimated k', fontsize=12)
    axes[0, 1].set_title('Backdoor Size vs Problem Size', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Hardness distribution
    hardness_counts = df_success['hardness'].value_counts()
    axes[0, 2].pie(hardness_counts.values, labels=hardness_counts.index, 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Instance Hardness Classification', fontsize=14)
    
    # Plot 4: Spectral gap distribution
    axes[1, 0].hist(np.log10(df_success['spectral_gap'] + 1e-10), bins=30, 
                    alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_xlabel('log₁₀(Spectral Gap)', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title('Distribution of Spectral Gaps', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Level spacing r-statistic
    axes[1, 1].hist(df_success['level_spacing_r'], bins=30, alpha=0.7, 
                    color='purple', edgecolor='black')
    axes[1, 1].axvline(0.39, color='blue', linestyle='--', linewidth=2, 
                       label='Poisson (integrable)')
    axes[1, 1].axvline(0.53, color='red', linestyle='--', linewidth=2, 
                       label='GOE (chaotic)')
    axes[1, 1].set_xlabel('Level Spacing <r>', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Level Spacing Statistics', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Analysis time vs N
    axes[1, 2].scatter(df_success['n_vars'], df_success['time'], alpha=0.6, color='orange')
    axes[1, 2].set_xlabel('Number of Variables (N)', fontsize=12)
    axes[1, 2].set_ylabel('Analysis Time (s)', fontsize=12)
    axes[1, 2].set_title('Computational Cost', fontsize=14)
    axes[1, 2].set_yscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'benchmark_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plots saved to: {plot_path}")
    plt.close()
    
    # Coverage plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_bins = 20
    n_max = df_success['n_vars'].max()
    bins = np.linspace(df_success['n_vars'].min(), n_max, n_bins)
    
    fractions_log = []
    fractions_quarter = []
    fractions_half = []
    bin_centers = []
    
    for i in range(len(bins) - 1):
        mask = (df_success['n_vars'] >= bins[i]) & (df_success['n_vars'] < bins[i+1])
        subset = df_success[mask]
        
        if len(subset) > 0:
            n_avg = subset['n_vars'].mean()
            frac_log = (subset['k_estimate'] <= np.log2(n_avg + 1)).mean()
            frac_quarter = (subset['k_estimate'] <= n_avg / 4).mean()
            frac_half = (subset['k_estimate'] <= n_avg / 2).mean()
            
            bin_centers.append(n_avg)
            fractions_log.append(frac_log)
            fractions_quarter.append(frac_quarter)
            fractions_half.append(frac_half)
    
    if bin_centers:
        ax.plot(bin_centers, fractions_log, 'o-', linewidth=2, markersize=8, 
                label='k ≤ log₂(N) [Quasi-poly]')
        ax.plot(bin_centers, fractions_quarter, 's-', linewidth=2, markersize=8, 
                label='k ≤ N/4 [Tractable]')
        ax.plot(bin_centers, fractions_half, '^-', linewidth=2, markersize=8, 
                label='k ≤ N/2')
        
        ax.set_xlabel('Problem Size (N)', fontsize=13)
        ax.set_ylabel('Fraction of Instances', fontsize=13)
        ax.set_title('Backdoor Size Coverage Analysis', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    coverage_path = os.path.join(output_dir, 'coverage_analysis.png')
    plt.savefig(coverage_path, dpi=150, bbox_inches='tight')
    print(f"✅ Coverage plot saved to: {coverage_path}")
    plt.close()


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics
    
    Args:
        df: DataFrame with results
    """
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\nTotal instances: {len(df)}")
    print(f"Successful: {len(df[df['status'] == 'SUCCESS'])}")
    print(f"Failed: {len(df[df['status'] == 'ERROR'])}")
    print(f"Timeout: {len(df[df['status'] == 'TIMEOUT'])}")
    print(f"Skipped: {len(df[df['status'] == 'SKIPPED'])}")
    
    df_success = df[df['status'] == 'SUCCESS']
    
    if len(df_success) > 0:
        print(f"\n--- Problem Sizes ---")
        print(f"N range: [{df_success['n_vars'].min()}, {df_success['n_vars'].max()}]")
        print(f"M range: [{df_success['n_clauses'].min()}, {df_success['n_clauses'].max()}]")
        print(f"Ratio (M/N) mean: {df_success['ratio'].mean():.2f}")
        
        print(f"\n--- Backdoor Size Estimates ---")
        print(f"k mean: {df_success['k_estimate'].mean():.2f}")
        print(f"k median: {df_success['k_estimate'].median():.2f}")
        print(f"k range: [{df_success['k_estimate'].min():.2f}, {df_success['k_estimate'].max():.2f}]")
        
        # Coverage statistics
        k_log = (df_success['k_estimate'] <= df_success['n_vars'].apply(lambda x: np.log2(x+1)))
        k_quarter = (df_success['k_estimate'] <= df_success['n_vars'] / 4)
        k_half = (df_success['k_estimate'] <= df_success['n_vars'] / 2)
        
        print(f"\n--- Coverage (Fraction of Instances) ---")
        print(f"k ≤ log₂(N) [quasi-polynomial]: {k_log.mean():.1%}")
        print(f"k ≤ N/4 [exponential-tractable]: {k_quarter.mean():.1%}")
        print(f"k ≤ N/2: {k_half.mean():.1%}")
        
        print(f"\n--- Hardness Classification ---")
        for hardness, count in df_success['hardness'].value_counts().items():
            print(f"{hardness}: {count} ({100*count/len(df_success):.1f}%)")
        
        print(f"\n--- Performance ---")
        print(f"Mean analysis time: {df_success['time'].mean():.4f}s")
        print(f"Median analysis time: {df_success['time'].median():.4f}s")
        print(f"Total time: {df_success['time'].sum():.2f}s ({df_success['time'].sum()/60:.2f} min)")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='SAT Instance Benchmark Harness')
    parser.add_argument('--input', type=str, required=True, 
                       help='Directory containing DIMACS .cnf files')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of instances to process')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SAT BENCHMARK HARNESS")
    print("="*70)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    if args.max:
        print(f"Max instances: {args.max}")
    print("="*70)
    
    run_benchmark(args.input, args.output, args.max)


if __name__ == "__main__":
    main()
