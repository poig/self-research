"""
Why Large Backdoors (k > 2N/3) Don't Help Quantum Solvers
==========================================================

The key insight: Backdoor size k determines the SEARCH SPACE, not solution probability!
"""
import numpy as np

print("="*80)
print("WHY 70% BACKDOOR DOESN'T ENABLE QUANTUM SPEEDUP")
print("="*80)
print()

# Example: N=50, k=35 (70% backdoor)
N = 50
k_small = 5
k_medium = 15
k_large = 35

print("BACKDOOR DEFINITION:")
print("-" * 80)
print("A backdoor of size k means:")
print("  - There exist k 'critical' variables")
print("  - Once you set these k variables correctly, the remaining (N-k) variables")
print("    can be solved in polynomial time (e.g., by unit propagation)")
print()
print("THE CATCH: You have to TRY all 2^k assignments to the backdoor variables!")
print()

print("="*80)
print("SEARCH SPACE COMPARISON:")
print("="*80)
print()

cases = [
    (N, k_small, "Small backdoor (quantum advantage)"),
    (N, k_medium, "Medium backdoor (limited quantum help)"),
    (N, k_large, "Large backdoor (no quantum advantage)"),
]

print(f"{'Case':>40} | {'k':>4} | {'2^k Search Space':>20} | {'Quantum Time':>15}")
print("-" * 80)

for n, k, desc in cases:
    search_space = 2**k
    quantum_time = 2**(k/2)  # Grover's algorithm gives sqrt speedup
    
    print(f"{desc:>40} | {k:>4} | {search_space:>20,} | {quantum_time:>15,.0f}")

print()
print("="*80)
print("THE PROBLEM WITH LARGE BACKDOORS:")
print("="*80)
print()

print(f"For N={N}, k={k_large} (70% backdoor):")
print()
print(f"1. Search space = 2^{k_large} = {2**k_large:,} assignments")
print(f"   Even with Grover: √(2^{k_large}) = 2^{k_large/2:.1f} = {2**(k_large/2):,.0f} quantum queries")
print()
print(f"2. Compare to small backdoor k={k_small}:")
print(f"   Search space = 2^{k_small} = {2**k_small:,} assignments")
print(f"   With Grover: 2^{k_small/2:.1f} = {2**(k_small/2):.0f} quantum queries")
print()
print(f"3. Speedup comparison:")
print(f"   Classical ratio: 2^{k_large} / 2^{k_small} = 2^{k_large-k_small} = {2**(k_large-k_small):,}x")
print(f"   Quantum ratio:   2^{k_large/2:.1f} / 2^{k_small/2:.1f} = {2**(k_large/2-k_small/2):,.0f}x")
print()
print("   → Large backdoor is STILL exponentially harder!")
print()

print("="*80)
print("WHY THIS MATTERS:")
print("="*80)
print()

print("Quantum advantage comes from:")
print("  ✓ Grover's algorithm: √(2^k) speedup on unstructured search")
print("  ✓ But the base 2^k is STILL exponential!")
print()
print("For k=35:")
print(f"  - Classical: 2^35 = {2**35:,} operations")
print(f"  - Quantum:   2^17.5 ≈ {int(2**17.5):,} operations")
print(f"  - Still INTRACTABLE even with quantum!")
print()

print("For k=5:")
print(f"  - Classical: 2^5 = {2**5} operations")
print(f"  - Quantum:   2^2.5 ≈ {int(2**2.5)} operations")
print(f"  - TRACTABLE! This is where quantum helps.")
print()

print("="*80)
print("THE COUNTERINTUITIVE TRUTH:")
print("="*80)
print()
print("❌ WRONG: 'Larger backdoor = more structure = better for quantum'")
print()
print("✅ RIGHT: 'Smaller backdoor = less search = exponential quantum speedup'")
print()
print("A 70% backdoor means:")
print("  - 70% of variables MUST be searched (2^0.7N assignments)")
print("  - Only 30% can be solved efficiently")
print("  - The problem is MOSTLY unstructured!")
print()
print("A 10% backdoor means:")
print("  - 10% of variables need search (2^0.1N assignments)")
print("  - 90% can be solved efficiently")
print("  - The problem is HIGHLY structured!")
print()

print("="*80)
print("PRACTICAL THRESHOLDS:")
print("="*80)
print()
print("For N=50:")
print()
print(f"  k ≤ 6:   2^k ≤ 64        → Quantum can enumerate all in superposition")
print(f"  k ≤ 16:  2^k ≤ 65,536    → Grover helps, but borderline tractable")
print(f"  k ≤ 33:  2^k ≤ 8 billion → Heuristics might find sub-structure")
print(f"  k > 33:  2^k > 8 billion → Essentially random, no quantum advantage")
print()
print("This is why the thresholds are:")
print("  - Quantum:     k ≤ log₂(N)+1  (exponential advantage)")
print("  - Hybrid QAOA: k ≤ N/3        (polynomial advantage)")
print("  - Scaffolding: k ≤ 2N/3       (heuristic advantage)")
print("  - CDCL:        k > 2N/3       (no advantage, use classical)")
print()

print("="*80)
print("SUMMARY:")
print("="*80)
print()
print("Backdoor size k = SIZE OF THE HARD PART you must search")
print()
print("Smaller k = More structure = Better for quantum")
print("Larger k  = Less structure = No quantum advantage")
print()
print("N=50, k=35 → 70% backdoor → 2^35 search space → INTRACTABLE")
print("N=50, k=5  → 10% backdoor → 2^5 search space  → TRACTABLE with quantum")
print()
