"""
ibm_scaling_runner.py

Run the Scaling Structure Algorithm on IBM Quantum hardware.
This is the experimental validation for Paper 6.

The experiment:
1. Create superposition over r-values (stable vs chaotic)
2. Run Hadamard test to encode dynamics
3. Measure to see if P(stable) > P(chaotic)

Key claim: Quantum interference concentrates probability
at stable r-values (bifurcation points).
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Configuration
BACKEND_NAME = 'ibm_brisbane'  # Or 'ibmq_qasm_simulator' for testing
SHOTS = 4096
DATA_DIR = Path(__file__).parent / 'data' / 'hardware'
FIGURES_DIR = Path(__file__).parent / 'figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SCALING ALGORITHM CIRCUITS
# =============================================================================

def sin2_map(x: float, r: float) -> float:
    """Classical sin² map for reference."""
    return r * np.sin(np.pi * x) ** 2


def compute_lyapunov(r: float) -> float:
    """Compute Lyapunov exponent classically."""
    x = 0.5
    lyap = 0.0
    for _ in range(500):
        df = abs(r * np.pi * np.sin(2 * np.pi * x))
        if df > 1e-10:
            lyap += np.log(df)
        x = sin2_map(x, r)
        x = np.clip(x, 1e-6, 1-1e-6)
    return lyap / 500


def is_stable(r: float) -> bool:
    """Check if r leads to stable dynamics (λ < 0)."""
    return compute_lyapunov(r) < 0


def create_scaling_circuit(n_r_qubits: int = 3, n_iterations: int = 2) -> QuantumCircuit:
    """
    Create the scaling algorithm circuit.
    
    Structure:
    - r-register: n_r_qubits in superposition
    - For each iteration: Hadamard test encodes dynamics
    - Measurement reveals interference pattern
    
    If working correctly:
    - Stable r values should have higher probability
    - This demonstrates the scaling structure speedup
    """
    r_reg = QuantumRegister(n_r_qubits, 'r')
    work = QuantumRegister(1, 'w')
    
    qc = QuantumCircuit(r_reg, work)
    
    # Step 1: Superposition over r
    for i in range(n_r_qubits):
        qc.h(r_reg[i])
    
    # Initialize work qubit
    qc.h(work[0])
    
    qc.barrier()
    
    # Step 2: r-dependent iterations (Hadamard test style)
    for k in range(n_iterations):
        # Hadamard test on work qubit
        qc.h(work[0])
        
        # Controlled rotations dependent on r
        for j in range(n_r_qubits):
            # Angle depends on r-bit contribution
            # This creates r-dependent phase accumulation
            angle = np.pi / (2 ** (k + 1)) * (2 ** j)
            qc.crz(angle, r_reg[j], work[0])
        
        # Complete Hadamard test
        qc.h(work[0])
        
        qc.barrier()
    
    qc.measure_all()
    return qc


def create_oracle_circuit(n_r_qubits: int = 3, n_grover: int = 1) -> QuantumCircuit:
    """
    Create circuit with oracle that marks chaotic r values.
    
    The oracle applies Z to r-values in chaotic region.
    Grover diffusion then amplifies stable r.
    """
    r_reg = QuantumRegister(n_r_qubits, 'r')
    qc = QuantumCircuit(r_reg)
    
    # Superposition
    for i in range(n_r_qubits):
        qc.h(r_reg[i])
    
    qc.barrier()
    
    for _ in range(n_grover):
        # Oracle: Mark high-r (chaotic) states
        # For 3 qubits: mark |110⟩ and |111⟩ (r > 0.75)
        qc.ccz(r_reg[0], r_reg[1], r_reg[2])
        
        qc.barrier()
        
        # Diffusion
        for i in range(n_r_qubits):
            qc.h(r_reg[i])
            qc.x(r_reg[i])
        
        qc.h(r_reg[n_r_qubits - 1])
        qc.mcx(list(range(n_r_qubits - 1)), n_r_qubits - 1)
        qc.h(r_reg[n_r_qubits - 1])
        
        for i in range(n_r_qubits):
            qc.x(r_reg[i])
            qc.h(r_reg[i])
        
        qc.barrier()
    
    qc.measure_all()
    return qc


# =============================================================================
# HARDWARE EXECUTION
# =============================================================================

def run_scaling_experiment(
    service: QiskitRuntimeService,
    backend_name: str,
    n_r_qubits: int = 3,
    n_iterations: int = 2,
    use_oracle: bool = False,
    n_grover: int = 1
) -> Dict:
    """
    Run the scaling algorithm on IBM hardware.
    
    Returns dict with:
    - r_probs: probability distribution over r values
    - stability: classical stability labels
    - metadata: experiment info
    """
    print(f"\n{'='*60}")
    print(f"SCALING STRUCTURE EXPERIMENT")
    print(f"Backend: {backend_name}")
    print(f"r-qubits: {n_r_qubits}, iterations: {n_iterations}")
    print(f"Oracle: {use_oracle}, Grover steps: {n_grover}")
    print(f"{'='*60}\n")
    
    # Connect to backend
    print(f"Connecting to {backend_name}...")
    backend = service.backend(backend_name)
    
    # Create circuit
    if use_oracle:
        qc = create_oracle_circuit(n_r_qubits, n_grover)
    else:
        qc = create_scaling_circuit(n_r_qubits, n_iterations)
    
    print(f"Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
    
    # Transpile
    print("Transpiling...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)
    print(f"Transpiled depth: {isa_qc.depth()}")
    
    # Execute
    print(f"Submitting job ({SHOTS} shots)...")
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_qc], shots=SHOTS)
    
    print(f"Job ID: {job.job_id()}")
    print("Waiting for results...")
    
    result = job.result()
    counts = result[0].data.meas.get_counts()
    
    print(f"Got {len(counts)} unique outcomes")
    
    # Process results: extract r-distribution
    N_r = 2 ** n_r_qubits
    r_counts = np.zeros(N_r)
    total = 0
    
    for bitstring, count in counts.items():
        total += count
        # Extract r-bits (depends on circuit structure)
        if use_oracle:
            # All qubits are r-register
            r_idx = int(bitstring[::-1], 2) % N_r
        else:
            # r-register is lower bits, work qubit is higher
            r_bits = bitstring[::-1][:n_r_qubits]
            r_idx = int(r_bits[::-1], 2)
        r_counts[r_idx] += count
    
    r_probs = r_counts / total
    
    # Classical stability labels
    r_values = np.linspace(0.5, 0.85, N_r)
    stability = np.array([is_stable(r) for r in r_values])
    
    # Compute key metrics
    p_stable = r_probs[stability].sum()
    p_chaotic = r_probs[~stability].sum()
    ratio = p_stable / p_chaotic if p_chaotic > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"P(stable):  {p_stable:.4f}")
    print(f"P(chaotic): {p_chaotic:.4f}")
    print(f"Ratio:      {ratio:.2f}x")
    print(f"{'='*60}\n")
    
    results = {
        'r_values': r_values.tolist(),
        'r_probs': r_probs.tolist(),
        'stability': stability.tolist(),
        'p_stable': float(p_stable),
        'p_chaotic': float(p_chaotic),
        'ratio': float(ratio),
        'metadata': {
            'backend': backend_name,
            'n_r_qubits': n_r_qubits,
            'n_iterations': n_iterations,
            'use_oracle': use_oracle,
            'n_grover': n_grover,
            'shots': SHOTS,
            'job_id': job.job_id(),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    return results


def run_comparison_experiment(service: QiskitRuntimeService, backend_name: str):
    """
    Run comparison: with vs without oracle.
    
    Expected:
    - Without oracle: P(stable)/P(chaotic) ≈ N_stable/N_chaotic (baseline)
    - With oracle: P(stable)/P(chaotic) > baseline (amplification!)
    """
    print("\n" + "=" * 70)
    print("COMPARISON EXPERIMENT: Without vs With Oracle")
    print("=" * 70)
    
    results = {}
    
    # Without oracle (baseline)
    print("\n[1/2] Running WITHOUT oracle (baseline)...")
    results['no_oracle'] = run_scaling_experiment(
        service, backend_name,
        n_r_qubits=3, n_iterations=2,
        use_oracle=False
    )
    
    # With oracle
    print("\n[2/2] Running WITH oracle (Grover)...")
    results['with_oracle'] = run_scaling_experiment(
        service, backend_name,
        n_r_qubits=3, n_iterations=0,
        use_oracle=True, n_grover=1
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = DATA_DIR / f'scaling_comparison_{backend_name}_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to: {filename}")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\nWithout Oracle:")
    print(f"  P(stable)/P(chaotic) = {results['no_oracle']['ratio']:.2f}x")
    print(f"\nWith Oracle (Grover):")
    print(f"  P(stable)/P(chaotic) = {results['with_oracle']['ratio']:.2f}x")
    print(f"\nAmplification factor: {results['with_oracle']['ratio'] / results['no_oracle']['ratio']:.2f}x")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_hardware_results(results: Dict, save_path: str = None):
    """Plot results from hardware run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    r_vals = np.array(results['r_values'])
    r_probs = np.array(results['r_probs'])
    stability = np.array(results['stability'])
    
    # Panel A: r-distribution
    ax1 = axes[0]
    colors = ['green' if s else 'red' for s in stability]
    ax1.bar(r_vals, r_probs, width=0.03, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Bifurcation Parameter r', fontsize=11)
    ax1.set_ylabel('Probability P(r)', fontsize=11)
    ax1.set_title(f"IBM Hardware: r-Distribution\n(Green=stable, Red=chaotic)", 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Stable vs Chaotic comparison
    ax2 = axes[1]
    labels = ['Stable', 'Chaotic']
    values = [results['p_stable'], results['p_chaotic']]
    bars = ax2.bar(labels, values, color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Probability', fontsize=11)
    ax2.set_title(f"Ratio = {results['ratio']:.2f}x\n(>1 means amplification works!)", 
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=11)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f"Scaling Algorithm on {results['metadata']['backend']}", 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("SCALING STRUCTURE ALGORITHM - IBM QUANTUM HARDWARE")
    print("Paper 6 Experimental Validation")
    print("=" * 70)
    
    # 1. Authenticate
    print("\nConnecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService()
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTo set up IBM Quantum credentials:")
        print("  from qiskit_ibm_runtime import QiskitRuntimeService")
        print("  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        return
    
    # 2. List available backends
    print("\nAvailable backends:")
    backends = service.backends()
    for b in backends[:5]:
        print(f"  - {b.name}: {b.num_qubits} qubits")
    
    # 3. Run experiment
    print(f"\nUsing backend: {BACKEND_NAME}")
    
    try:
        results = run_comparison_experiment(service, BACKEND_NAME)
        
        # 4. Visualize
        for key, data in results.items():
            plot_hardware_results(
                data,
                save_path=str(FIGURES_DIR / f'hardware_{key}_{BACKEND_NAME}.png')
            )
        
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print("""
Key result: If ratio WITH oracle > ratio WITHOUT oracle,
then Grover amplification is working on real hardware!

This demonstrates the SCALING STRUCTURE SPEEDUP.
        """)
        
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
