# Small SAT-checker for 1-round AES instances
# Encodes the CNF using encode_1_round_aes and solves with PySAT (Glucose3)
import sys
sys.path.append('..')
try:
    from tools.test_1round_aes import encode_1_round_aes
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from tools.test_1round_aes import encode_1_round_aes

try:
    from src.solvers.aes_sbox_encoder import AES_SBOX
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.solvers.aes_sbox_encoder import AES_SBOX

import os
import random

try:
    from pysat.solvers import Glucose3
except Exception as e:
    print("PySAT not available:", e)
    raise


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


plaintext_hex = "3243f6a8885a308d313198a2e0370734"
ciphertext_hex = "3925841d02dc09fbdc118597196a0b32"
plaintext_bytes = bytes.fromhex(plaintext_hex)
ciphertext_bytes = bytes.fromhex(ciphertext_hex)

print('Encoding original provided instance...')
clauses, n_vars, key_vars = encode_1_round_aes(plaintext_bytes, ciphertext_bytes)
print(f'Clauses: {len(clauses)}, Vars: {n_vars}')

s = Glucose3()
for c in clauses:
    s.add_clause(list(c))
print('Solving...')
ok = s.solve()
print('SAT?', ok)
if ok:
    print('Model length:', len(s.get_model()))

# Now generate random key and derived cipher
max_attempts = 20
found = False
for attempt in range(1, max_attempts + 1):
    random_key = os.urandom(16)
    derived = one_round_aes_encrypt(plaintext_bytes, random_key)
    print(f'Attempt #{attempt}: Generated key: {random_key.hex()}')
    print(f'Attempt #{attempt}: Derived cipher: {derived.hex()}')

    clauses2, n_vars2, key_vars2 = encode_1_round_aes(plaintext_bytes, derived)
    print(f'Clauses: {len(clauses2)}, Vars: {n_vars2}')

    s2 = Glucose3()
    for c in clauses2:
        s2.add_clause(list(c))
    print('Solving derived (no key fixed)...')
    ok2 = s2.solve()
    print('SAT (derived, no key fixed)?', ok2)
    if ok2:
        m = s2.get_model()
        true_vars = set(v for v in m if isinstance(v, int) and v>0)
        key_bits = [1 if i+1 in true_vars else 0 for i in range(128)]
        recovered = bytearray(16)
        for byte_i in range(16):
            kb = 0
            for bit_i in range(8):
                kb |= (key_bits[byte_i*8 + bit_i] & 1) << bit_i
            recovered[byte_i] = kb
        print('Recovered key (no key fixed):', recovered.hex())
        print('Matches?', recovered.hex() == random_key.hex())
        found = True
        break

    # Try solving with master key as assumptions (so we can extract an UNSAT core)
    print('Attempting solve with master key provided as assumptions...')
    # Build assumptions: LSB-first per byte for variables 1..128
    key_bits = []
    for b in random_key:
        for bit in range(8):
            key_bits.append((b >> bit) & 1)
    assumptions = []
    for i, bit in enumerate(key_bits):
        lit = (i+1) if bit == 1 else -(i+1)
        assumptions.append(lit)

    # Use the same solver object but call solve with assumptions
    ok_assump = s2.solve(assumptions=assumptions)
    print('SAT (derived with assumptions)?', ok_assump)
    if ok_assump:
        m = s2.get_model()
        true_vars = set(v for v in m if isinstance(v, int) and v>0)
        key_bits = [1 if i+1 in true_vars else 0 for i in range(128)]
        recovered = bytearray(16)
        for byte_i in range(16):
            kb = 0
            for bit_i in range(8):
                kb |= (key_bits[byte_i*8 + bit_i] & 1) << bit_i
            recovered[byte_i] = kb
        print('Recovered key (assumptions):', recovered.hex())
        print('Matches?', recovered.hex() == random_key.hex())
        found = True
        break
    else:
        # Extract UNSAT core if available
        try:
            core = s2.get_core()
            print(f'UNSAT core (assumptions subset): {core}')
            # Map core literals back to key bit indices for debugging
            core_bits = [lit for lit in core if abs(lit) <= 128]
            print(f'Conflicting key bit literals (<=128): {core_bits}')
        except Exception:
            print('Could not extract UNSAT core from solver')

print('Done')
