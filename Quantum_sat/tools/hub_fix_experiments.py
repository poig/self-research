"""Hub-fix experiments for 1-round AES

Fix top-k hub variables (k=1..4) to all polarity combos and try to solve the 1-round AES instance.
Saves a summary to tools/hub_fix_results.json
"""
import itertools
import json
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from test_1round_aes import encode_1_round_aes
from src.core.quantum_sat_solver import ComprehensiveQuantumSATSolver, SolverMethod

# Load top hubs from var_degrees.json
with open(os.path.join(os.path.dirname(__file__), 'var_degrees.json')) as f:
    vd = json.load(f)

top20 = [v for v, _ in vd['top20']]
# We'll use top-k from this list
results = []

plaintext_hex = "3243f6a8885a308d313198a2e0370734"
plaintext_bytes = bytes.fromhex(plaintext_hex)
# Re-generate a consistent ciphertext using a random key as in the test harness
import os as _os
random_key = _os.urandom(16)

# helper: one-round AES encrypt (same as in test harness)
def gf_mul2(x: int) -> int:
    x &= 0xFF
    return ((x << 1) ^ 0x1B) & 0xFF if (x & 0x80) else (x << 1) & 0xFF

def gf_mul3(x: int) -> int:
    return gf_mul2(x) ^ x

def one_round_aes_encrypt(plaintext_b: bytes, key_b: bytes) -> bytes:
    from src.solvers.aes_sbox_encoder import AES_SBOX
    state = [p ^ k for p, k in zip(plaintext_b, key_b)]
    state = [AES_SBOX[b] for b in state]
    shifted = [0] * 16
    for r in range(4):
        for c in range(4):
            old_idx = r*4 + c
            new_idx = r*4 + ((c - r) % 4)
            shifted[new_idx] = state[old_idx]
    mixed = [0] * 16
    for col in range(4):
        a0 = shifted[0*4 + col]
        a1 = shifted[1*4 + col]
        a2 = shifted[2*4 + col]
        a3 = shifted[3*4 + col]
        out0 = (gf_mul2(a0) ^ gf_mul3(a1) ^ a2 ^ a3) & 0xFF
        out1 = (a0 ^ gf_mul2(a1) ^ gf_mul3(a2) ^ a3) & 0xFF
        out2 = (a0 ^ a1 ^ gf_mul2(a2) ^ gf_mul3(a3)) & 0xFF
        out3 = (gf_mul3(a0) ^ a1 ^ a2 ^ gf_mul2(a3)) & 0xFF
        mixed[0*4 + col] = out0
        mixed[1*4 + col] = out1
        mixed[2*4 + col] = out2
        mixed[3*4 + col] = out3
    final = bytes([m ^ k for m, k in zip(mixed, key_b)])
    return final

ciphertext = one_round_aes_encrypt(plaintext_bytes, random_key)
print(f"Using random key {random_key.hex()} and ciphertext {ciphertext.hex()}")

# Build base CNF (espresso minimized default) to have the smaller CNF
clauses, n_vars, key_vars = encode_1_round_aes(plaintext_bytes, ciphertext, sbox_mode='espresso')

solver = ComprehensiveQuantumSATSolver(verbose=False, prefer_quantum=False)

# We'll force classical DPLL for these experiments to get deterministic behavior
forced_method = SolverMethod.CLASSICAL_DPLL

start_total = time.time()
for k in range(1, 5):
    hubs = top20[:k]
    print(f"Trying k={k} hubs: {hubs}")
    for combo in itertools.product([0,1], repeat=k):
        # Build assumptions list (1-indexed var literals)
        assumptions = []
        for var, val in zip(hubs, combo):
            lit = var if val == 1 else -var
            assumptions.append(lit)
        # Run solver with assumptions
        t0 = time.time()
        res = solver.solve(clauses, n_vars, timeout=10.0, method=forced_method, check_final=False)
        elapsed = time.time() - t0
        # The solver API doesn't accept assumptions in high-level solve, so we call verify_with_classical directly
        sat, assignment, method = solver.verify_with_classical(clauses, n_vars, timeout=5.0, assumptions=assumptions)
        results.append({
            'k': k,
            'assumptions': assumptions,
            'sat': sat,
            'method': method,
            'time': elapsed
        })
        if sat:
            print(f"SAT found for k={k} assumptions={assumptions} (time {elapsed:.2f}s)")
            # save and exit early
            with open(os.path.join(os.path.dirname(__file__), 'hub_fix_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            raise SystemExit(0)

# Save results
with open(os.path.join(os.path.dirname(__file__), 'hub_fix_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Done hub-fix experiments in {time.time()-start_total:.1f}s")
