# Check MixColumns encoding for a single 4-byte column
# Uses encode_mixcolumns_column from src.solvers.aes_mixcolumns_encoder

import sys
sys.path.append('..')
try:
    from src.solvers.aes_mixcolumns_encoder import encode_mixcolumns_column
except ModuleNotFoundError:
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.solvers.aes_mixcolumns_encoder import encode_mixcolumns_column

try:
    from pysat.solvers import Glucose3
except Exception as e:
    print('PySAT not available:', e)
    raise

# Test vector from AES spec (from many references):
# input column = [0xDB, 0x13, 0x53, 0x45]
# expected output = [0x8E, 0x4D, 0xA1, 0xBC]
input_bytes = [0xDB, 0x13, 0x53, 0x45]
expected = [0x8E, 0x4D, 0xA1, 0xBC]

# Allocate input and output bit variables (LSB-first per byte)
var_counter = 1
input_vars = []
for b in input_bytes:
    byte_vars = list(range(var_counter, var_counter + 8))
    var_counter += 8
    input_vars.append(byte_vars)

# Do NOT pre-allocate output_vars here. Pass None and let the encoder allocate
# its output variables (it returns them as allocated_outputs). This avoids
# collisions with the encoder's internal temporary variables.
clauses, next_var, allocated_outputs = encode_mixcolumns_column(input_vars, None, var_counter)
print('Encoded MixColumns column: clauses', len(clauses), 'next_var', next_var)

# Fix inputs and outputs
fixed_clauses = list(clauses)
for i, b in enumerate(input_bytes):
    for bit_idx in range(8):
        var = input_vars[i][bit_idx]
        bit = (b >> bit_idx) & 1
        fixed_clauses.append((var,) if bit == 1 else (-var,))

for i, b in enumerate(expected):
    for bit_idx in range(8):
        var = allocated_outputs[i][bit_idx]
        bit = (b >> bit_idx) & 1
        fixed_clauses.append((var,) if bit == 1 else (-var,))

s = Glucose3()
for c in fixed_clauses:
    s.add_clause(list(c))
print('Solving MixColumns unit test...')
ok = s.solve()
print('SAT?', ok)
if ok:
    m = s.get_model()
    true_vars = set(v for v in m if isinstance(v, int) and v>0)
    print('True vars (sample):', sorted([v for v in true_vars if v < 100]))

print('Done')
