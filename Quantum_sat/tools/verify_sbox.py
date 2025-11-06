"""Verify AES S-box CNF encoding on a few samples.

Creates a small CNF for one S-box instance using `encode_sbox_naive` and
checks satisfiability when input is fixed to a value and output fixed to
AES_SBOX[input].
"""
import sys
sys.path.append('..')
try:
    from src.solvers.aes_sbox_encoder import encode_sbox_naive, AES_SBOX
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.solvers.aes_sbox_encoder import encode_sbox_naive, AES_SBOX

try:
    from pysat.solvers import Glucose3
except Exception as e:
    print('PySAT not available:', e)
    raise


def int_to_bits_lsb(value, n=8):
    return [(value >> i) & 1 for i in range(n)]


def check_sbox(value):
    # variables 1..8 input, 9..16 output
    input_vars = list(range(1, 9))
    output_vars = list(range(9, 17))
    clauses = encode_sbox_naive(input_vars, output_vars)
    # fix input bits
    input_bits = int_to_bits_lsb(value)
    for i, b in enumerate(input_bits):
        lit = input_vars[i] if b == 1 else -input_vars[i]
        clauses.append((lit,))
    # fix output bits to AES_SBOX[value]
    out_val = AES_SBOX[value]
    out_bits = int_to_bits_lsb(out_val)
    for i, b in enumerate(out_bits):
        lit = output_vars[i] if b == 1 else -output_vars[i]
        clauses.append((lit,))

    s = Glucose3()
    for c in clauses:
        s.add_clause(list(c))
    ok = s.solve()
    return ok


if __name__ == '__main__':
    test_values = [0x00, 0x53, 0xFF, 0x6F, 0xDB]
    for v in test_values:
        ok = check_sbox(v)
        print(f'S-box check for input 0x{v:02X}:', 'SAT' if ok else 'UNSAT')
