"""
Test 1-Round AES
================

Instead of full 10-round AES, test just 1 round.
This will:
1. Run MUCH faster (90k clauses vs 941k)
2. Give us k* estimate for single round
3. Show if AES rounds decompose

If 1-round has k* < 10: Full AES might be crackable!
If 1-round has k* ≈ 16-32: Full AES is definitely secure (10× worse)
"""

import sys
import time
sys.path.append('..')
# sys.path.insert(0, 'src/core')
# sys.path.insert(0, 'src/solvers')

try:
    from src.core.quantum_sat_solver import ComprehensiveQuantumSATSolver
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.core.quantum_sat_solver import ComprehensiveQuantumSATSolver
try:
    from src.solvers.aes_sbox_encoder import encode_sbox_naive
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.solvers.aes_sbox_encoder import encode_sbox_naive
try:
    from src.solvers.aes_mixcolumns_encoder import encode_mixcolumns_column
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.solvers.aes_mixcolumns_encoder import encode_mixcolumns_column

def encode_1_round_aes(plaintext_bytes, ciphertext_bytes):
    """
    Encode just 1 round of AES (much smaller problem).
    
    Structure:
      1. AddRoundKey (plaintext XOR key)
      2. SubBytes (16 S-boxes)
      3. ShiftRows
      4. MixColumns
      5. AddRoundKey (round key)
      6. Assert output == ciphertext
    
    Expected: ~94,000 clauses (10× smaller than full AES)
    """
    
    clauses = []
    next_var_id = 1
    
    # Master key variables (unknown)
    master_key_vars = list(range(next_var_id, next_var_id + 128))
    next_var_id += 128
    
    # Plaintext variables (known, will be fixed)
    plaintext_vars = list(range(next_var_id, next_var_id + 128))
    next_var_id += 128
    
    # Fix plaintext
    print("  Fixing plaintext...")
    for byte_idx in range(16):
        byte_val = plaintext_bytes[byte_idx]
        for bit_idx in range(8):
            var = plaintext_vars[byte_idx * 8 + bit_idx]
            bit = (byte_val >> bit_idx) & 1
            if bit == 1:
                clauses.append((var,))
            else:
                clauses.append((-var,))
    
    # AddRoundKey (initial)
    print("  Encoding AddRoundKey...")
    state_vars = []
    for i in range(128):
        out_var = next_var_id
        next_var_id += 1
        # out = plaintext XOR key
        p, k, out = plaintext_vars[i], master_key_vars[i], out_var
        clauses.append((-p, -k, -out))
        clauses.append((-p, k, out))
        clauses.append((p, -k, out))
        clauses.append((p, k, -out))
        state_vars.append(out_var)
    
    # SubBytes (16 S-boxes)
    print("  Encoding 16 S-boxes...")
    new_state = []
    for byte_idx in range(16):
        input_vars = state_vars[byte_idx*8:(byte_idx+1)*8]
        output_vars = list(range(next_var_id, next_var_id + 8))
        next_var_id += 8
        
        sbox_clauses = encode_sbox_naive(input_vars, output_vars)
        clauses.extend(sbox_clauses)
        new_state.extend(output_vars)
    
    state_vars = new_state
    
    # ShiftRows (just permutation, no clauses needed)
    print("  Applying ShiftRows...")
    shifted = [0] * 128
    for row in range(4):
        for col in range(4):
            old_pos = row * 32 + col * 8
            new_col = (col - row) % 4
            new_pos = row * 32 + new_col * 8
            shifted[new_pos:new_pos+8] = state_vars[old_pos:old_pos+8]
    state_vars = shifted
    
    # MixColumns (4 columns)
    print("  Encoding 4 MixColumns...")
    new_state = []
    for col in range(4):
        # Prepare input_vars as List[List[int]] (4 bytes × 8 bits)
        input_vars = []
        for row in range(4):
            byte_vars = state_vars[row*32 + col*8 : row*32 + col*8 + 8]
            input_vars.append(byte_vars)
        
        # Prepare output_vars as List[List[int]]
        output_vars = []
        for row in range(4):
            output_vars.append(list(range(next_var_id + row*8, next_var_id + row*8 + 8)))
        
        # Call encode_mixcolumns_column with correct signature
        col_clauses, col_next_id = encode_mixcolumns_column(input_vars, output_vars, next_var_id)
        clauses.extend(col_clauses)
        
        # Flatten output vars to state
        for row in range(4):
            new_state.extend(output_vars[row])
        
        next_var_id = col_next_id
    
    state_vars = new_state
    
    # Final AddRoundKey (with round key = master key for simplicity)
    print("  Encoding final AddRoundKey...")
    final_state = []
    for i in range(128):
        out_var = next_var_id
        next_var_id += 1
        s, k, out = state_vars[i], master_key_vars[i], out_var
        clauses.append((-s, -k, -out))
        clauses.append((-s, k, out))
        clauses.append((s, -k, out))
        clauses.append((s, k, -out))
        final_state.append(out_var)
    
    # Fix ciphertext
    print("  Fixing ciphertext...")
    for byte_idx in range(16):
        byte_val = ciphertext_bytes[byte_idx]
        for bit_idx in range(8):
            var = final_state[byte_idx * 8 + bit_idx]
            bit = (byte_val >> bit_idx) & 1
            if bit == 1:
                clauses.append((var,))
            else:
                clauses.append((-var,))
    
    print(f"\n✅ 1-round AES encoded!")
    print(f"   Clauses: {len(clauses):,}")
    print(f"   Variables: {next_var_id:,}")
    print(f"   Master key: vars 1-128")
    
    return clauses, next_var_id, master_key_vars


# Only run test if this file is executed directly
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔬 TESTING 1-ROUND AES (FAST TEST)")
    print("="*80)
    print()

    plaintext_hex = "3243f6a8885a308d313198a2e0370734"
    ciphertext_hex = "3925841d02dc09fbdc118597196a0b32"

    plaintext_bytes = bytes.fromhex(plaintext_hex)
    ciphertext_bytes = bytes.fromhex(ciphertext_hex)

    print("[1/3] Encoding 1-round AES...")
    start = time.time()
    clauses, n_vars, key_vars = encode_1_round_aes(plaintext_bytes, ciphertext_bytes)
    print(f"   Time: {time.time()-start:.1f}s")
    print()

    print("[2/3] Running k* certification (should be fast!)...")
    solver = ComprehensiveQuantumSATSolver(
        verbose=True,
        prefer_quantum=True,
        enable_quantum_certification=False,  # Disabled - not needed for k* estimation
        certification_mode="fast"
    )

    start = time.time()
    result = solver.solve(clauses, n_vars, timeout=60.0, check_final=False)
    elapsed = time.time() - start

    print()
    print("="*80)
    print("📊 RESULTS")
    print("="*80)
    print()
    print(f"k* = {result.k_star}")
    print(f"Hardness: {result.hardness_class}")
    print(f"Time: {elapsed:.1f}s")
    print()
    # Interpret and display results
    if result.k_star is not None:
        if result.k_star < 10:
            print("🚨 1-round AES has k* < 10!")
            print("   This suggests full 10-round AES might also decompose!")
            print("   Need to test full AES to confirm.")
        elif result.k_star < 32:
            print("⚠️  1-round AES has k* < 32")
            print(f"   Full 10-round AES likely has k* ≈ {result.k_star * 10}")
            print("   Full AES may be weakened; further testing recommended.")
        else:
            print("✅ 1-round AES has k* ≥ 32")
            print("   Full 10-round AES likely has k* ≈ 128")
            print("   AES is SECURE under this analysis.")
    else:
        print("⚠️  Could not determine k* for 1-round AES")
        print("   The structure analysis did not return a backdoor size.")
        print("   Consider rerunning with different decomposition methods or enabling certification.")

    print()
    print("="*80)
