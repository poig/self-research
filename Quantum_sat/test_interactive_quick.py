"""
Quick test of the interactive AES cracker with 1-round AES
"""

import sys
sys.path.insert(0, 'src/core')
sys.path.insert(0, 'src/solvers')

from quantum_sat_solver import ComprehensiveQuantumSATSolver

# Test case
plaintext_hex = "3243f6a8885a308d313198a2e0370734"
ciphertext_hex = "3925841d02dc09fbdc118597196a0b32"

plaintext = bytes.fromhex(plaintext_hex)
ciphertext = bytes.fromhex(ciphertext_hex)

print("="*80)
print("QUICK TEST: 1-Round AES Cracking")
print("="*80)
print()
print(f"Plaintext:  {plaintext_hex}")
print(f"Ciphertext: {ciphertext_hex}")
print()

# Test with 1-round first
try:
    from test_1round_aes import encode_1_round_aes
    
    print("Encoding 1-round AES...")
    clauses, n_vars, key_vars = encode_1_round_aes(plaintext, ciphertext)
    print(f"✅ {len(clauses):,} clauses, {n_vars:,} variables")
    print()
    
    print("Creating solver (fast mode, 4 cores)...")
    solver = ComprehensiveQuantumSATSolver(
        verbose=True,
        prefer_quantum=True,
        enable_quantum_certification=False,
        decompose_methods=["Louvain", "Treewidth"],
        n_jobs=4
    )
    print()
    
    print("Solving...")
    result = solver.solve(clauses, n_vars, timeout=120.0, check_final=False)
    print()
    
    if result.satisfiable:
        print("="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print(f"Method: {result.method_used}")
        if result.k_star:
            print(f"k*: {result.k_star}")
        if "Decomposed" in result.method_used:
            print("✅ Decomposition succeeded - AES is crackable!")
    else:
        print("❌ Failed to solve")
        
except ImportError as e:
    print(f"❌ Error: {e}")
    print()
    print("Make sure you're in the Quantum_sat directory and have all dependencies!")
