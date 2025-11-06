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
    from src.solvers.aes_sbox_encoder import AES_SBOX
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.solvers.aes_sbox_encoder import encode_sbox_naive, AES_SBOX, encode_sbox_tseitin
try:
    from src.solvers.aes_mixcolumns_encoder import encode_mixcolumns_column
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.solvers.aes_mixcolumns_encoder import encode_mixcolumns_column

def encode_1_round_aes(plaintext_bytes, ciphertext_bytes, sbox_mode: str = 'naive'):
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

        if sbox_mode == 'naive':
            sbox_clauses = encode_sbox_naive(input_vars, output_vars)
            clauses.extend(sbox_clauses)
        elif sbox_mode == 'tseitin':
            # encode_sbox_tseitin may allocate indicator/auxiliary vars starting at next_var_id
            tseit_clauses, next_var_id = encode_sbox_tseitin(input_vars, output_vars, next_var_id)
            clauses.extend(tseit_clauses)
        elif sbox_mode == 'espresso':
            # Use logic-minimized per-output encoding
            from src.solvers.aes_sbox_encoder import encode_sbox_espresso
            sbox_clauses = encode_sbox_espresso(input_vars, output_vars)
            clauses.extend(sbox_clauses)
        else:
            raise ValueError(f"Unknown sbox_mode: {sbox_mode}")
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
        
        # Do NOT pre-allocate output_vars here. Let the encoder allocate outputs
        # and return them to us to avoid variable-index collisions with internal temps.
        col_clauses, col_next_id, allocated_outputs = encode_mixcolumns_column(input_vars, None, next_var_id)
        clauses.extend(col_clauses)

        # Flatten allocated output vars to state
        for row in range(4):
            new_state.extend(allocated_outputs[row])

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

    print("[2/3] Running k* certification (fast-mode: classical by default)...")
    # Allow user to opt into quantum pipeline via --quantum or env var USE_QUANTUM=1
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantum', action='store_true', help='Use quantum pipeline (slower)')
    parser.add_argument('--decompose', action='store_true', help='Attempt polynomial decomposition path')
    parser.add_argument('--force-method', type=str, default=None, help='Force solver method (e.g. qaoa_qlto, qsvt, structure_aligned_qaoa, classical_dpll)')
    args, _ = parser.parse_known_args()
    use_quantum = args.quantum or os.getenv('USE_QUANTUM', '') == '1'

    # If the user supplied --force-method, map it to SolverMethod here so it's
    # available both for the initial run and any retry that follows.
    forced_method = None
    if args.force_method:
        try:
            from src.core.quantum_sat_solver import SolverMethod
            # Try to convert to the enum; if that fails, keep the raw string
            try:
                forced_method = SolverMethod(args.force_method)
            except Exception:
                # Accept raw string as fallback so solver can interpret it
                forced_method = args.force_method
        except Exception:
            # If import fails for any reason, keep the raw string
            forced_method = args.force_method

    solver = ComprehensiveQuantumSATSolver(
        verbose=True,
        prefer_quantum=use_quantum,
        enable_quantum_certification=False,  # Disabled - not needed for k* estimation
        certification_mode="fast"
    )

    if args.decompose:
        print('\n⚙️  Decomposition requested: letting solver attempt internal polynomial decomposition and routing')
        # Force decomposition inside the solver even when k_est == 0
        start = time.time()
        result = solver.solve(clauses, n_vars, timeout=60.0, check_final=False, force_decompose=True, method=forced_method)
        elapsed = time.time() - start
    else:
        start = time.time()
        result = solver.solve(clauses, n_vars, timeout=60.0, check_final=False, method=forced_method)
        elapsed = time.time() - start

    # If the encoding is UNSAT (likely because provided ciphertext isn't a single-round
    # encryption of the plaintext under any key), auto-generate a consistent key/cipher
    # and re-run the test to validate the encoder and solver pipeline.
    if not getattr(result, 'satisfiable', False):
        print("\n⚠️  Instance appears UNSAT. Generating a consistent one-round ciphertext and retrying...")
        # AES helper functions
        import os as _os

        def gf_mul2(x: int) -> int:
            x &= 0xFF
            return ((x << 1) ^ 0x1B) & 0xFF if (x & 0x80) else (x << 1) & 0xFF

        def gf_mul3(x: int) -> int:
            return gf_mul2(x) ^ x

        def one_round_aes_encrypt(plaintext_b: bytes, key_b: bytes) -> bytes:
            if len(plaintext_b) != 16 or len(key_b) != 16:
                raise ValueError("Need 16-byte plaintext and key")
            # AddRoundKey
            state = [p ^ k for p, k in zip(plaintext_b, key_b)]
            # SubBytes (use AES_SBOX)
            state = [AES_SBOX[b] for b in state]
            # ShiftRows (row-major ordering: row*4 + col)
            shifted = [0] * 16
            for r in range(4):
                for c in range(4):
                    old_idx = r*4 + c
                    new_idx = r*4 + ((c - r) % 4)
                    shifted[new_idx] = state[old_idx]
            # MixColumns
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
            # Final AddRoundKey (use master key as round key)
            final = bytes([m ^ k for m, k in zip(mixed, key_b)])
            return final

        # Pick a random key and derive a ciphertext consistent with one-round AES
        random_key = _os.urandom(16)
        derived_cipher = one_round_aes_encrypt(plaintext_bytes, random_key)
        print(f"   Generated random key: {random_key.hex()}")
        print(f"   Derived consistent ciphertext: {derived_cipher.hex()}")

        # Re-encode with the derived ciphertext and re-run solver
        clauses, n_vars, key_vars = encode_1_round_aes(plaintext_bytes, derived_cipher)
        print("[retry] Running solver on generated consistent instance...")
        start = time.time()
        # Preserve the forced method if the user passed --force-method earlier
        try:
            result = solver.solve(clauses, n_vars, timeout=60.0, check_final=False, method=forced_method)
        except Exception:
            # Fallback to default signature if solver doesn't accept method kwarg
            result = solver.solve(clauses, n_vars, timeout=60.0, check_final=False)
        elapsed = time.time() - start
        # Attempt to expose found key in model (if SAT)
        if getattr(result, 'satisfiable', False):
            model = getattr(result, 'model', getattr(result, 'assignment', None))
            print("   Solver returned a model. Extracting key bits (first 128 vars)...")
            try:
                # model may be iterable of ints or dict
                if isinstance(model, dict):
                    key_bits = [int(bool(model.get(i+1, False))) for i in range(128)]
                else:
                    true_vars = set(v for v in model if isinstance(v, int) and v > 0)
                    key_bits = [1 if (i+1) in true_vars else 0 for i in range(128)]
                # pack first 128 bits into 16 bytes (LSB-first per byte)
                recovered_key = bytearray(16)
                for byte_i in range(16):
                    kb = 0
                    for bit_i in range(8):
                        kb |= (key_bits[byte_i*8 + bit_i] & 1) << bit_i
                    recovered_key[byte_i] = kb
                print(f"   Recovered key (LSB-bit ordering): {recovered_key.hex()}")
            except Exception:
                pass

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

    # --- NEW: Try to extract and display the recovered master key if available ---
    try:
        model = None
        # result may be a dict-like or object with attributes
        if isinstance(result, dict):
            model = result.get('assignment') or result.get('model')
        else:
            model = getattr(result, 'assignment', None) or getattr(result, 'model', None)

        if model:
            # Normalize model into a mapping of 0-indexed var -> bool
            key_bits = None
            if isinstance(model, dict):
                # keys may be 0-indexed or 1-indexed
                # prefer 0-indexed mapping; if 1-indexed, convert
                if all(isinstance(k, int) and k >= 0 for k in model.keys()):
                    key_bits = [int(bool(model.get(i, False))) for i in range(128)]
                else:
                    # assume 1-indexed
                    key_bits = [1 if model.get(i+1, False) else 0 for i in range(128)]
            else:
                # model may be iterable of integers (positive = true var)
                try:
                    true_vars = set(int(v) for v in model if isinstance(v, int) and v > 0)
                    key_bits = [1 if (i+1) in true_vars else 0 for i in range(128)]
                except Exception:
                    key_bits = None

            if key_bits is not None:
                # pack into 16 bytes (LSB-first per byte)
                recovered_key = bytearray(16)
                for byte_i in range(16):
                    kb = 0
                    for bit_i in range(8):
                        kb |= (key_bits[byte_i*8 + bit_i] & 1) << bit_i
                    recovered_key[byte_i] = kb
                print(f"\n🔑 Recovered master key (LSB bit ordering): {recovered_key.hex()}")
            else:
                print("\n🔑 Could not normalize model into key bits; model type unexpected.")
        else:
            print("\n🔑 No assignment/model available in result to extract key.")
    except Exception as e:
        print(f"\n⚠️  Failed to extract recovered key: {e}")
