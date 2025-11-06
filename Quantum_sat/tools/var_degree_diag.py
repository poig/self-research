"""Variable-degree diagnostic for 1-round AES CNF

Builds the CNF using the existing encoders in tools/test_1round_aes.py
and reports the top variables by adjacency degree and a histogram.

Usage: python tools/var_degree_diag.py
"""
import json
import os
import sys
import time
from collections import defaultdict, Counter

# Ensure repo src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.solvers.aes_sbox_encoder import encode_sbox_naive, AES_SBOX, encode_sbox_tseitin
from src.solvers.aes_mixcolumns_encoder import encode_mixcolumns_column


def encode_1_round_aes_clauses(plaintext_bytes, ciphertext_bytes, sbox_mode: str = 'naive'):
    clauses = []
    next_var_id = 1
    var_labels = {}

    # Master key variables (unknown)
    master_key_vars = list(range(next_var_id, next_var_id + 128))
    next_var_id += 128
    for i, v in enumerate(master_key_vars):
        var_labels[v] = f"master_key_bit_{i}"

    # Plaintext variables (known, will be fixed)
    plaintext_vars = list(range(next_var_id, next_var_id + 128))
    next_var_id += 128
    for i, v in enumerate(plaintext_vars):
        var_labels[v] = f"plaintext_bit_{i}"

    # Fix plaintext
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
    state_vars = []
    for i in range(128):
        out_var = next_var_id
        next_var_id += 1
        p, k, out = plaintext_vars[i], master_key_vars[i], out_var
        var_labels[out_var] = f"ark1_out_bit_{i}"
        clauses.append((-p, -k, -out))
        clauses.append((-p, k, out))
        clauses.append((p, -k, out))
        clauses.append((p, k, -out))
        state_vars.append(out_var)

    # SubBytes (16 S-boxes)
    new_state = []
    for byte_idx in range(16):
        input_vars = state_vars[byte_idx*8:(byte_idx+1)*8]
        output_vars = list(range(next_var_id, next_var_id + 8))
        for b in range(8):
            var_labels[output_vars[b]] = f"sbox_{byte_idx}_out_bit_{b}"
        # advance next_var_id for output bits; additional aux/indicators may be allocated by tseitin
        next_var_id += 8
        if sbox_mode == 'naive':
            sbox_clauses = encode_sbox_naive(input_vars, output_vars)
            clauses.extend(sbox_clauses)
        elif sbox_mode == 'tseitin':
            tseit_clauses, next_var_id = encode_sbox_tseitin(input_vars, output_vars, next_var_id)
            clauses.extend(tseit_clauses)
        elif sbox_mode == 'espresso':
            from src.solvers.aes_sbox_encoder import encode_sbox_espresso
            sbox_clauses = encode_sbox_espresso(input_vars, output_vars)
            clauses.extend(sbox_clauses)
        else:
            raise ValueError(f"Unknown sbox_mode: {sbox_mode}")
        new_state.extend(output_vars)
    state_vars = new_state

    # ShiftRows
    shifted = [0] * 128
    for row in range(4):
        for col in range(4):
            old_pos = row * 32 + col * 8
            new_col = (col - row) % 4
            new_pos = row * 32 + new_col * 8
            shifted[new_pos:new_pos+8] = state_vars[old_pos:old_pos+8]
    state_vars = shifted

    # MixColumns (4 columns)
    new_state = []
    for col in range(4):
        input_vars = []
        for row in range(4):
            byte_vars = state_vars[row*32 + col*8 : row*32 + col*8 + 8]
            input_vars.append(byte_vars)
        col_clauses, col_next_id, allocated_outputs = encode_mixcolumns_column(input_vars, None, next_var_id)
        clauses.extend(col_clauses)
        for row in range(4):
            out_byte = allocated_outputs[row]
            for b, v in enumerate(out_byte):
                # label by column,row within the AES state
                # compute global byte index: row*4 + col
                byte_idx = row*4 + col
                var_labels[v] = f"mixcol_col{col}_byte{byte_idx}_out_bit_{b}"
                new_state.extend([v])
        next_var_id = col_next_id
    state_vars = new_state

    # Final AddRoundKey
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
    for byte_idx in range(16):
        byte_val = ciphertext_bytes[byte_idx]
        for bit_idx in range(8):
            var = final_state[byte_idx * 8 + bit_idx]
            bit = (byte_val >> bit_idx) & 1
            if bit == 1:
                clauses.append((var,))
            else:
                clauses.append((-var,))

    return clauses, next_var_id - 1
    # unreachable


def build_var_adjacency(clauses):
    adj = defaultdict(set)
    for cl in clauses:
        vars_in_clause = [abs(lit) for lit in cl]
        for i, v in enumerate(vars_in_clause):
            for u in vars_in_clause[i+1:]:
                adj[v].add(u)
                adj[u].add(v)
    return adj


def main():
    # pick a random key so ciphertext is consistent
    import os
    import random
    plaintext_hex = "3243f6a8885a308d313198a2e0370734"
    plaintext_bytes = bytes.fromhex(plaintext_hex)
    random_key = os.urandom(16)

    # derive a consistent ciphertext
    def gf_mul2(x: int) -> int:
        x &= 0xFF
        return ((x << 1) ^ 0x1B) & 0xFF if (x & 0x80) else (x << 1) & 0xFF
    def gf_mul3(x: int) -> int:
        return gf_mul2(x) ^ x
    def one_round_aes_encrypt(plaintext_b: bytes, key_b: bytes) -> bytes:
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sbox-mode', choices=['naive', 'tseitin', 'espresso'], default='naive', help='S-box encoding mode')
    args, _ = parser.parse_known_args()

    sbox_mode = args.sbox_mode

    print(f"Encoding CNF (sbox_mode={sbox_mode})...")
    start = time.time()
    clauses, n_vars = encode_1_round_aes_clauses(plaintext_bytes, ciphertext, sbox_mode=sbox_mode)
    print(f"Clauses: {len(clauses):,} Vars: {n_vars:,} Time: {time.time()-start:.2f}s")

    print("Building variable adjacency graph...")
    start = time.time()
    adj = build_var_adjacency(clauses)
    print(f"Adjacency built in {time.time()-start:.2f}s")

    degrees = {v: len(neigh) for v, neigh in adj.items()}
    deg_vals = list(degrees.values())
    deg_counter = Counter(deg_vals)

    top20 = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
    print("Top 20 variables by degree:")
    for v, d in top20:
        print(f"  Var {v}: degree {d}")

    avg_deg = sum(deg_vals)/len(deg_vals)
    max_deg = max(deg_vals) if deg_vals else 0
    print(f"Avg degree: {avg_deg:.2f} Max degree: {max_deg}")

    # simple histogram buckets
    buckets = [(0,1),(2,4),(5,9),(10,19),(20,49),(50,99),(100,9999)]
    hist = {f"{a}-{b}": 0 for a,b in buckets}
    for dv in deg_vals:
        for a,b in buckets:
            if a <= dv <= b:
                hist[f"{a}-{b}"] += 1
                break
    print("Degree histogram:")
    for k in hist:
        print(f"  {k}: {hist[k]}")

    # Save raw degrees
    out = {
        'n_vars': n_vars,
        'n_clauses': len(clauses),
        'top20': top20,
        'avg_degree': avg_deg,
        'max_degree': max_deg,
        'histogram': hist
    }
    with open(os.path.join(os.path.dirname(__file__), 'var_degrees.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved var_degrees.json")

if __name__ == '__main__':
    main()
