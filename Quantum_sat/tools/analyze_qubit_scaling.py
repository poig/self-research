"""
Qubit Requirements for Backdoor-Based Quantum SAT Solving
==========================================================

Analysis of qubit growth with problem size and backdoor size
"""
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("QUBIT REQUIREMENTS FOR QUANTUM SAT SOLVING")
print("="*80)
print()

print("BASIC QUBIT REQUIREMENTS:")
print("-" * 80)
print()
print("For a backdoor-based quantum approach:")
print("  1. Backdoor register: k qubits (to represent 2^k assignments)")
print("  2. Ancilla for unit propagation: O(N-k) qubits")
print("  3. Clause checking: O(M) ancilla qubits")
print("  4. Grover's algorithm overhead: O(1) qubits")
print()
print("Total qubits ≈ k + (N-k) + M = N + M qubits")
print()

print("="*80)
print("QUBIT SCALING ANALYSIS:")
print("="*80)
print()

# Different problem sizes
test_cases = [
    # (N, k, M, description)
    (20, 3, 88, "Tiny problem, small backdoor"),
    (20, 10, 88, "Tiny problem, medium backdoor"),
    (50, 5, 220, "Small problem, small backdoor"),
    (50, 15, 220, "Small problem, medium backdoor"),
    (50, 35, 220, "Small problem, large backdoor"),
    (100, 10, 440, "Medium problem, small backdoor"),
    (100, 33, 440, "Medium problem, medium backdoor"),
    (100, 70, 440, "Medium problem, large backdoor"),
    (200, 15, 880, "Large problem, small backdoor"),
    (200, 67, 880, "Large problem, medium backdoor"),
    (200, 140, 880, "Large problem, large backdoor"),
]

print(f"{'N':>4} {'k':>4} {'M':>5} {'k/N':>6} | {'Backdoor':>10} {'Ancilla':>10} {'Total':>10} | {'2^k Search':>15} | {'Feasible?':>10}")
print("-" * 80)

for N, k, M, desc in test_cases:
    ratio = k / N
    
    # Qubit requirements
    backdoor_qubits = k
    ancilla_qubits = N - k + M  # Unit prop + clause checking
    total_qubits = backdoor_qubits + ancilla_qubits
    
    search_space = 2**k
    
    # Feasibility analysis
    if k <= 20:
        feasible = "YES ✓"
    elif k <= 30:
        feasible = "Maybe"
    else:
        feasible = "NO ✗"
    
    print(f"{N:>4} {k:>4} {M:>5} {ratio:>6.1%} | {backdoor_qubits:>10} {ancilla_qubits:>10} {total_qubits:>10} | {search_space:>15,} | {feasible:>10}")

print()
print("="*80)
print("KEY INSIGHTS:")
print("="*80)
print()

print("1. QUBIT COUNT SCALES AS: Q ≈ N + M")
print("   - Linearly with problem size!")
print("   - NOT exponential in N")
print("   - For clause-to-variable ratio M/N ≈ 4.4:")
print("     Q ≈ N + 4.4N = 5.4N qubits")
print()

print("2. BACKDOOR SIZE k DETERMINES:")
print("   - Search space: 2^k (exponential!)")
print("   - NOT total qubit count (that's N+M)")
print()

print("3. PRACTICAL LIMITS:")
print("   Current quantum hardware: ~100-1000 qubits")
print("   - N=50:  Q ≈ 270 qubits → Feasible on today's hardware ✓")
print("   - N=100: Q ≈ 540 qubits → Feasible on today's hardware ✓")
print("   - N=200: Q ≈ 1080 qubits → Requires near-future hardware")
print()

print("4. THE REAL BOTTLENECK: CIRCUIT DEPTH, NOT QUBITS!")
print("   - Grover needs O(√(2^k)) iterations")
print("   - Each iteration: O(N+M) gate depth for oracle")
print("   - Total depth: O(√(2^k) × (N+M))")
print()

print("="*80)
print("DETAILED SCALING ANALYSIS:")
print("="*80)
print()

print("For fixed N=50, varying k:")
print()
print(f"{'k':>4} {'k/N':>6} | {'Qubits':>8} | {'Grover Iters':>15} | {'Total Depth':>15} | {'Quantum Adv?':>12}")
print("-" * 80)

N_fixed = 50
M_fixed = 220
for k in [3, 5, 10, 15, 20, 25, 30, 35]:
    ratio = k / N_fixed
    qubits = N_fixed + M_fixed
    grover_iters = int(2**(k/2))
    depth_per_iter = N_fixed + M_fixed  # Simplified
    total_depth = grover_iters * depth_per_iter
    
    if k <= 10:
        advantage = "YES ✓"
    elif k <= 20:
        advantage = "Limited"
    else:
        advantage = "NO ✗"
    
    print(f"{k:>4} {ratio:>6.1%} | {qubits:>8} | {grover_iters:>15,} | {total_depth:>15,} | {advantage:>12}")

print()
print("Notice: Qubits stay constant (270), but depth grows exponentially with k!")
print()

print("="*80)
print("WHY LARGE BACKDOORS FAIL (k > 2N/3):")
print("="*80)
print()

k_large = 35
N_example = 50
M_example = 220

qubits_needed = N_example + M_example
grover_iterations = int(2**(k_large/2))
depth_per_iteration = N_example + M_example
total_depth = grover_iterations * depth_per_iteration

print(f"Example: N={N_example}, k={k_large} (70% backdoor)")
print()
print(f"  Qubits needed: {qubits_needed} → Feasible ✓")
print(f"  Grover iterations: 2^{k_large/2:.1f} = {grover_iterations:,}")
print(f"  Depth per iteration: ~{depth_per_iteration}")
print(f"  Total circuit depth: {total_depth:,}")
print()
print(f"  With T1/T2 ≈ 100μs, gate time ≈ 100ns:")
print(f"    Max gates before decoherence: ~1,000")
print(f"    We need: {total_depth:,} gates")
print(f"    Ratio: {total_depth/1000:.0f}x over limit!")
print()
print("  → INTRACTABLE due to circuit depth, not qubit count!")
print()

print("="*80)
print("OPTIMAL REGIME FOR QUANTUM:")
print("="*80)
print()

print("Sweet spot: k ≤ log₂(N) + 1")
print()
print(f"{'N':>4} | {'Optimal k':>10} | {'Qubits':>8} | {'Circuit Depth':>15} | {'Classical Ops':>15}")
print("-" * 80)

for N in [20, 50, 100, 200]:
    k_optimal = int(np.log2(N)) + 1
    M = int(N * 4.4)
    qubits = N + M
    grover_iters = int(2**(k_optimal/2))
    depth = grover_iters * (N + M)
    classical_ops = 2**k_optimal
    
    print(f"{N:>4} | {k_optimal:>10} | {qubits:>8} | {depth:>15,} | {classical_ops:>15,}")

print()
print("In this regime:")
print("  ✓ Qubits scale linearly: O(N)")
print("  ✓ Circuit depth manageable: O(√N × N) = O(N^1.5)")
print("  ✓ Exponential quantum speedup over classical")
print()

print("="*80)
print("SUMMARY:")
print("="*80)
print()
print("Qubit growth: LINEAR in N (Q ≈ 5.4N for M/N=4.4)")
print("  - N=50:  ~270 qubits")
print("  - N=100: ~540 qubits")
print("  - N=200: ~1080 qubits")
print()
print("Circuit depth growth: EXPONENTIAL in k (D ~ 2^(k/2) × N)")
print("  - k=5:  ~270 depth (feasible)")
print("  - k=10: ~8,600 depth (borderline)")
print("  - k=35: ~50M depth (impossible)")
print()
print("The bottleneck is NOT qubits, it's CIRCUIT DEPTH!")
print("This is why k > 2N/3 fails - the circuit becomes too deep.")
print()
print("Quantum advantage requires k ≤ log₂(N)+1 where:")
print("  - Qubits: manageable (linear)")
print("  - Depth: tractable (polynomial in N)")
print("  - Speedup: exponential (2^k → 2^(k/2))")
print()
