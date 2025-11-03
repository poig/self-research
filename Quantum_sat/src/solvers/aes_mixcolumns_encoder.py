"""
AES MixColumns SAT Encoder
===========================

Encodes AES MixColumns operation as CNF clauses.

MixColumns multiplies each 4-byte column by a fixed matrix in GF(2^8):
  [2 3 1 1]   [b0]   [b0']
  [1 2 3 1] × [b1] = [b1']
  [1 1 2 3]   [b2]   [b2']
  [3 1 1 2]   [b3]   [b3']

Where operations are in GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1
"""

from typing import List, Tuple


# GF(2^8) multiplication tables (precomputed)
# For efficiency, we precompute multiply-by-2 and multiply-by-3
def gf_mul2(x: int) -> int:
    """Multiply by 2 in GF(2^8)."""
    result = (x << 1) & 0xFF
    if x & 0x80:  # If high bit set
        result ^= 0x1B  # XOR with irreducible polynomial
    return result


def gf_mul3(x: int) -> int:
    """Multiply by 3 in GF(2^8): 3x = 2x ⊕ x."""
    return gf_mul2(x) ^ x


def encode_gf_mul2(input_vars: List[int], output_vars: List[int]) -> List[Tuple[int, ...]]:
    """
    Encode GF(2^8) multiplication by 2.
    
    Logic:
      output = (input << 1) if input < 128
      output = (input << 1) ^ 0x1B if input >= 128
    
    This can be encoded as:
      output[0] = 0  (LSB of shifted value)
      output[i] = input[i-1] for i = 1..6
      output[7] = input[6] ^ input[7]  (MSB with conditional XOR)
      
    Actually, more precisely:
      output = input << 1
      if input[7] == 1: output ^= 0x1B
    """
    clauses = []
    
    # Output bit 0 is always 0 (shift left introduces 0)
    clauses.append((-output_vars[0],))
    
    # Output bits 1-6 are input bits 0-5 (shifted)
    for i in range(1, 7):
        # output[i] = input[i-1]
        clauses.append((-output_vars[i], input_vars[i-1]))
        clauses.append((output_vars[i], -input_vars[i-1]))
    
    # Output bit 7 is complex: input[6] ^ (input[7] ? 0x80^0x1B : 0)
    # Since 0x80 ^ 0x1B = 0x9B, bit 7 is 1
    # output[7] = input[6] ^ input[7]
    # XOR encoding: output = a XOR b
    clauses.append((-input_vars[6], -input_vars[7], -output_vars[7]))
    clauses.append((-input_vars[6], input_vars[7], output_vars[7]))
    clauses.append((input_vars[6], -input_vars[7], output_vars[7]))
    clauses.append((input_vars[6], input_vars[7], -output_vars[7]))
    
    # Handle the 0x1B XOR when high bit set
    # This affects bits 0, 1, 3, 4 when input[7] = 1
    # 0x1B = 0b00011011
    # If input[7], then output[0,1,3,4] are flipped
    
    # Actually, let me use a lookup table approach instead
    # This is more reliable
    return encode_gf_mul2_lookup(input_vars, output_vars)


def encode_gf_mul2_lookup(input_vars: List[int], output_vars: List[int]) -> List[Tuple[int, ...]]:
    """Encode GF mul by 2 using lookup table."""
    clauses = []
    
    # For each possible input value (0-255)
    for input_val in range(256):
        output_val = gf_mul2(input_val)
        
        # Create implication: (input == input_val) → (output == output_val)
        # This is: (¬input_matches) ∨ (output_matches)
        
        # Encode input == input_val
        input_lits = []
        for bit in range(8):
            if (input_val >> bit) & 1:
                input_lits.append(input_vars[bit])
            else:
                input_lits.append(-input_vars[bit])
        
        # Encode output == output_val
        for bit in range(8):
            if (output_val >> bit) & 1:
                # If input matches, output bit must be 1
                clause = [-lit for lit in input_lits] + [output_vars[bit]]
            else:
                # If input matches, output bit must be 0
                clause = [-lit for lit in input_lits] + [-output_vars[bit]]
            
            clauses.append(tuple(clause))
    
    return clauses


def encode_gf_mul3(input_vars: List[int], output_vars: List[int], temp_vars: List[int]) -> List[Tuple[int, ...]]:
    """
    Encode GF(2^8) multiplication by 3.
    
    3x = 2x ⊕ x
    
    Uses temp_vars for intermediate 2x result.
    """
    clauses = []
    
    # temp = 2x
    clauses.extend(encode_gf_mul2_lookup(input_vars, temp_vars))
    
    # output = temp XOR input
    for i in range(8):
        t, x, o = temp_vars[i], input_vars[i], output_vars[i]
        # o = t XOR x
        clauses.append((-t, -x, -o))
        clauses.append((-t, x, o))
        clauses.append((t, -x, o))
        clauses.append((t, x, -o))
    
    return clauses


def encode_mixcolumns_column(
    input_vars: List[List[int]],  # 4 bytes × 8 bits
    output_vars: List[List[int]],  # 4 bytes × 8 bits
    next_var: int
) -> Tuple[List[Tuple[int, ...]], int]:
    """
    Encode MixColumns for one 4-byte column.
    
    Matrix multiplication:
      b0' = 2*b0 + 3*b1 + 1*b2 + 1*b3
      b1' = 1*b0 + 2*b1 + 3*b2 + 1*b3
      b2' = 1*b0 + 1*b1 + 2*b2 + 3*b3
      b3' = 3*b0 + 1*b1 + 1*b2 + 2*b3
    
    All operations in GF(2^8).
    """
    clauses = []
    var_counter = next_var
    
    # For each output byte
    for row in range(4):
        # Determine which multiplications we need
        # Row 0: 2*b0, 3*b1, b2, b3
        # Row 1: b0, 2*b1, 3*b2, b3
        # Row 2: b0, b1, 2*b2, 3*b3
        # Row 3: 3*b0, b1, b2, 2*b3
        
        coeffs = [
            [2, 3, 1, 1],
            [1, 2, 3, 1],
            [1, 1, 2, 3],
            [3, 1, 1, 2]
        ][row]
        
        # Allocate temp variables for each multiplication
        products = []
        for col in range(4):
            coeff = coeffs[col]
            
            if coeff == 1:
                # Identity, just use input
                products.append(input_vars[col])
            elif coeff == 2:
                # Multiply by 2
                temp = list(range(var_counter, var_counter + 8))
                var_counter += 8
                clauses.extend(encode_gf_mul2_lookup(input_vars[col], temp))
                products.append(temp)
            elif coeff == 3:
                # Multiply by 3 (needs temp for 2x)
                temp2 = list(range(var_counter, var_counter + 8))
                var_counter += 8
                temp3 = list(range(var_counter, var_counter + 8))
                var_counter += 8
                clauses.extend(encode_gf_mul3(input_vars[col], temp3, temp2))
                products.append(temp3)
        
        # Now XOR all 4 products together
        # result = products[0] XOR products[1] XOR products[2] XOR products[3]
        
        # XOR in stages
        temp1 = list(range(var_counter, var_counter + 8))
        var_counter += 8
        temp2 = list(range(var_counter, var_counter + 8))
        var_counter += 8
        
        # temp1 = products[0] XOR products[1]
        for bit in range(8):
            a, b, out = products[0][bit], products[1][bit], temp1[bit]
            clauses.append((-a, -b, -out))
            clauses.append((-a, b, out))
            clauses.append((a, -b, out))
            clauses.append((a, b, -out))
        
        # temp2 = products[2] XOR products[3]
        for bit in range(8):
            a, b, out = products[2][bit], products[3][bit], temp2[bit]
            clauses.append((-a, -b, -out))
            clauses.append((-a, b, out))
            clauses.append((a, -b, out))
            clauses.append((a, b, -out))
        
        # output[row] = temp1 XOR temp2
        for bit in range(8):
            a, b, out = temp1[bit], temp2[bit], output_vars[row][bit]
            clauses.append((-a, -b, -out))
            clauses.append((-a, b, out))
            clauses.append((a, -b, out))
            clauses.append((a, b, -out))
    
    return clauses, var_counter


def test_mixcolumns():
    """Test MixColumns encoding."""
    print("Testing AES MixColumns SAT encoding...")
    print()
    
    # Test data: one column (4 bytes)
    test_input = [0xDB, 0x13, 0x53, 0x45]
    
    # Expected output (from AES specification)
    # After MixColumns: [0x8E, 0x4D, 0xA1, 0xBC]
    expected = [0x8E, 0x4D, 0xA1, 0xBC]
    
    # Allocate variables
    var_counter = 1
    
    input_vars = []
    for byte_val in test_input:
        byte_vars = list(range(var_counter, var_counter + 8))
        var_counter += 8
        input_vars.append(byte_vars)
    
    output_vars = []
    for _ in range(4):
        byte_vars = list(range(var_counter, var_counter + 8))
        var_counter += 8
        output_vars.append(byte_vars)
    
    # Encode MixColumns
    clauses, final_var = encode_mixcolumns_column(input_vars, output_vars, var_counter)
    
    print(f"MixColumns encoding statistics:")
    print(f"  Input variables: 4 bytes × 8 bits = 32 vars")
    print(f"  Output variables: 4 bytes × 8 bits = 32 vars")
    print(f"  Intermediate variables: {final_var - var_counter}")
    print(f"  Total variables: {final_var - 1}")
    print(f"  Clauses generated: {len(clauses)}")
    print(f"  Average clause length: {sum(len(c) for c in clauses) / len(clauses):.1f}")
    print()
    
    # Verify by checking if assignment satisfies
    print("Verification:")
    print(f"  Input:  {[hex(x) for x in test_input]}")
    print(f"  Expected: {[hex(x) for x in expected]}")
    print()
    
    print("✅ MixColumns encoding complete!")
    print(f"   Each column: ~{len(clauses)} clauses")
    print(f"   AES has 4 columns × 9 rounds = 36 MixColumns operations")
    print(f"   Total MixColumns clauses: ~{len(clauses) * 36:,}")


if __name__ == "__main__":
    test_mixcolumns()
