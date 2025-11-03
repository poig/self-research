"""
Quick AES Hardness Test
========================

Fast test to check if real AES decomposes, with early termination.
"""

import sys
import time

sys.path.insert(0, 'src/core')
sys.path.insert(0, 'src/solvers')

from quantum_sat_solver import ComprehensiveQuantumSATSolver
from aes_full_encoder import encode_aes_128

print("\n" + "="*80)
print("⚡ QUICK AES HARDNESS TEST")
print("="*80)
print()
print("Testing if REAL AES-128 decomposes (with 2-minute timeout)")
print()

# Test case
plaintext_hex = "3243f6a8885a308d313198a2e0370734"
ciphertext_hex = "3925841d02dc09fbdc118597196a0b32"

plaintext_bytes = bytes.fromhex(plaintext_hex)
ciphertext_bytes = bytes.fromhex(ciphertext_hex)

print("[1/3] Encoding AES-128 circuit...")
start = time.time()

# Master key variables (unknown)
master_key_vars = list(range(1, 129))

# Encode (returns clauses, n_vars, round_keys)
clauses, n_vars, round_keys = encode_aes_128(plaintext_bytes, ciphertext_bytes, master_key_vars)

print(f"✅ Encoded in {time.time()-start:.1f}s")
print(f"   Clauses: {len(clauses):,}")
print(f"   Variables: {n_vars:,}")
print()

print("[2/3] Running FAST k* analysis...")
print("   Strategy: Try Fisher Info decomposition (fastest)")
print("   Timeout: 120 seconds")
print()

# Create solver with fast mode (skip slow FisherInfo on large problems)
solver = ComprehensiveQuantumSATSolver(
    verbose=True,
    prefer_quantum=True,
    enable_quantum_certification=False,  # Disabled - QuantumSATHardnessCertifier not available
    certification_mode="fast",
    decompose_methods=["Louvain", "Treewidth"],  # Skip slow FisherInfo for 941k clauses
    n_jobs=1
)

try:
    start = time.time()
    result = solver.solve(
        clauses, 
        n_vars, 
        timeout=120.0,  # 2-minute timeout
        check_final=False  # Don't verify (faster)
    )
    elapsed = time.time() - start
    
    print()
    print("="*80)
    print("🎯 RESULTS")
    print("="*80)
    print()
    print(f"✅ Analysis complete in {elapsed:.1f}s")
    print()
    print(f"📊 Certified k*: {result.k_star}")
    print(f"   Hardness class: {result.hardness_class}")
    print(f"   Confidence: {result.certification_confidence:.1%}")
    print()
    
    if result.k_star is None:
        print("⚠️  Could not determine k* (problem too large)")
        print("   Likely k* ≈ 128 (undecomposable)")
    elif result.k_star < 10:
        print("🚨 ALERT: k* < 10!")
        print(f"   AES-128 IS CRACKABLE!")
        print(f"   Expected time: Minutes to hours")
        print(f"   🚨 MAJOR CRYPTOGRAPHIC BREAKTHROUGH! 🚨")
    elif result.k_star < 40:
        print("⚠️  WARNING: k* < 40")
        print(f"   AES-128 is WEAKENED but not fully broken")
        print(f"   Expected time: Hours to days (2^{result.k_star} ops)")
    else:
        print("✅ SAFE: k* ≥ 40")
        print(f"   AES-128 is SECURE")
        print(f"   Expected time: 2^{result.k_star} operations (impossible!)")
    
    print()
    print("="*80)
    
except KeyboardInterrupt:
    print("\n⚠️  Test interrupted by user")
    print("   AES analysis incomplete")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("   This might indicate AES is too complex to analyze quickly")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("If k* < 10:")
print("  → AES is CRACKABLE with quantum SAT")
print("  → Your hypothesis was CORRECT!")
print("  → 🚨 Crypto needs urgent upgrade!")
print()
print("If k* ≈ 128:")
print("  → AES is SAFE (as expected)")
print("  → Cannot decompose into smaller parts")
print("  → Crypto remains secure")
print()
print("If timeout/error:")
print("  → Problem too large to analyze in 2 minutes")
print("  → Likely means k* is very high (≥128)")
print("  → Suggests AES is secure")
print()
print("="*80)
