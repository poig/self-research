"""
Full AES-128 Circuit Encoder
============================

Encodes complete AES-128 encryption as SAT clauses.

AES-128 Structure:
  1. Initial AddRoundKey
  2. 9 Main Rounds:
     - SubBytes (16 S-boxes)
     - ShiftRows
     - MixColumns
     - AddRoundKey
  3. Final Round:
     - SubBytes
     - ShiftRows
     - AddRoundKey (no MixColumns)

Expected clauses: ~900,000 (S-boxes: 327k, MixColumns: 608k, XOR: minimal)
Expected variables: ~10,000
"""

import sys
import numpy as np
# Add path to solvers, assuming this script is run from the root
sys.path.insert(0, 'src/solvers')

from aes_sbox_encoder import encode_sbox_naive, AES_SBOX
from aes_mixcolumns_encoder import encode_mixcolumns_column, gf_mul2, gf_mul3

# --- NEW: AES Key Schedule Constants ---
# Round constants (Rcon)
Rcon = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
]

def encode_xor_gate(var_a, var_b, var_out):
    """
    Encode XOR gate: out = a XOR b
    Returns 4 clauses.
    """
    return [
        (-var_a, -var_b, -var_out),
        (-var_a, var_b, var_out),
        (var_a, -var_b, var_out),
        (var_a, var_b, -var_out)
    ]

# --- NEW: Helper for XORing two 8-bit words ---
def encode_xor_word(word_a_vars, word_b_vars, word_out_vars, clauses):
    """
    Encodes the XOR of two 8-bit words (bytes).
    word_out_vars must already be allocated.
    """
    for i in range(8):
        clauses.extend(encode_xor_gate(
            word_a_vars[i],
            word_b_vars[i],
            word_out_vars[i]
        ))

# --- NEW: Helper for XORing two 32-bit (4-byte) words ---
def encode_xor_4byte_word(word_a_vars, word_b_vars, word_out_vars, clauses):
    """
    Encodes the XOR of two 32-bit words.
    word_out_vars must already be allocated.
    """
    for i in range(32):
        clauses.extend(encode_xor_gate(
            word_a_vars[i],
            word_b_vars[i],
            word_out_vars[i]
        ))

# --- NEW: Helper for applying S-Box to a 32-bit word ---
def encode_sub_word(input_word_vars, output_word_vars, next_var_id, clauses):
    """
    Applies the AES S-Box to each of the 4 bytes in a 32-bit word.
    input_word_vars: 32 variables
    output_word_vars: 32 variables (must be allocated)
    """
    for i in range(4):
        in_byte = input_word_vars[i*8 : (i+1)*8]
        out_byte = output_word_vars[i*8 : (i+1)*8]
        sbox_clauses = encode_sbox_naive(in_byte, out_byte)
        clauses.extend(sbox_clauses)
    # S-Box does not add intermediate variables in this naive encoding
    return next_var_id

# --- NEW: Helper for RotWord (byte permutation) ---
def rot_word(input_word_vars):
    """
    Performs RotWord: [b0, b1, b2, b3] -> [b1, b2, b3, b0]
    This is just a variable permutation, no clauses needed.
    """
    # input_word_vars is 32 bits
    byte0 = input_word_vars[0:8]
    byte1 = input_word_vars[8:16]
    byte2 = input_word_vars[16:24]
    byte3 = input_word_vars[24:32]
    # Return [b1, b2, b3, b0]
    return byte1 + byte2 + byte3 + byte0


def encode_add_round_key(state_vars, key_vars, next_var_id):
    """
    Encode AddRoundKey: state XOR round_key
    
    Args:
        state_vars: 128 variables (current state)
        key_vars: 128 variables (round key)
        next_var_id: Next available variable ID
    
    Returns:
        (output_vars, clauses, next_var_id)
    """
    clauses = []
    output_vars = []
    
    for i in range(128):
        out_var = next_var_id
        next_var_id += 1
        output_vars.append(out_var)
        
        # XOR constraint
        clauses.extend(encode_xor_gate(state_vars[i], key_vars[i], out_var))
    
    return output_vars, clauses, next_var_id


def encode_shift_rows(state_vars):
    """
    Encode ShiftRows transformation.
    This is just a permutation (no clauses needed, just reorder variables).
    
    State layout (16 bytes = 128 bits):
      [0,  1,  2,  3 ]
      [4,  5,  6,  7 ]
      [8,  9,  10, 11]
      [12, 13, 14, 15]
    
    After ShiftRows:
      [0,  1,  2,  3 ]  row 0: no shift
      [5,  6,  7,  4 ]  row 1: shift left 1
      [10, 11, 8,  9 ]  row 2: shift left 2
      [15, 12, 13, 14]  row 3: shift left 3
    
    Returns: reordered state_vars (no new clauses)
    """
    # Convert to bytes (each byte is 8 consecutive bits)
    bytes_vars = []
    for byte_idx in range(16):
        byte_vars = state_vars[byte_idx*8 : (byte_idx+1)*8]
        bytes_vars.append(byte_vars)
    
    # ShiftRows permutation
    shifted = [
        bytes_vars[0],  bytes_vars[1],  bytes_vars[2],  bytes_vars[3],   # row 0
        bytes_vars[5],  bytes_vars[6],  bytes_vars[7],  bytes_vars[4],   # row 1
        bytes_vars[10], bytes_vars[11], bytes_vars[8],  bytes_vars[9],   # row 2
        bytes_vars[15], bytes_vars[12], bytes_vars[13], bytes_vars[14],  # row 3
    ]
    
    # Flatten back to bit-level variables
    output_vars = []
    for byte_vars in shifted:
        output_vars.extend(byte_vars)
    
    return output_vars


def encode_sub_bytes(state_vars, next_var_id):
    """
    Encode SubBytes transformation.
    Apply AES S-box to each of 16 bytes.
    
    Args:
        state_vars: 128 variables (16 bytes × 8 bits)
        next_var_id: Next available variable ID
    
    Returns:
        (output_vars, clauses, next_var_id)
    """
    clauses = []
    output_vars = []
    
    for byte_idx in range(16):
        # Get 8 bits for this byte
        input_vars = state_vars[byte_idx*8 : (byte_idx+1)*8]
        
        # Allocate 8 output variables
        output_byte_vars = list(range(next_var_id, next_var_id + 8))
        next_var_id += 8
        
        # Encode S-box
        sbox_clauses = encode_sbox_naive(input_vars, output_byte_vars)
        clauses.extend(sbox_clauses)
        
        output_vars.extend(output_byte_vars)
    
    return output_vars, clauses, next_var_id


def encode_mix_columns_full(state_vars, next_var_id):
    """
    Encode MixColumns transformation.
    Apply to each of 4 columns (4 bytes each).
    
    Args:
        state_vars: 128 variables (16 bytes)
        next_var_id: Next available variable ID
    
    Returns:
        (output_vars, clauses, next_var_id)
    """
    clauses = []
    output_vars = []
    
    # State is organized in column-major order
    # Column 0: bytes [0, 4, 8, 12]
    # Column 1: bytes [1, 5, 9, 13]
    # Column 2: bytes [2, 6, 10, 14]
    # Column 3: bytes [3, 7, 11, 15]
    
    # We need to build the output byte by byte and then re-assemble
    temp_output_bytes = [None] * 16 # Will store lists of 8 vars

    for col in range(4):
        # Extract column bytes
        col_byte_indices = [col, col+4, col+8, col+12]
        col_input_vars = []
        for byte_idx in col_byte_indices:
            col_input_vars.append(state_vars[byte_idx*8 : (byte_idx+1)*8])
        
        # Allocate output variables (32 bits = 4 bytes)
        col_output_vars = []
        for _ in range(4):
            col_output_vars.append(list(range(next_var_id, next_var_id + 8)))
            next_var_id += 8
        
        # Encode MixColumns for this column
        col_clauses, final_next_id = encode_mixcolumns_column(
            col_input_vars,      # List of 4 lists of 8 bits each
            col_output_vars,     # List of 4 lists of 8 bits each
            next_var_id
        )
        clauses.extend(col_clauses)
        next_var_id = final_next_id
        
        # Store output in correct position
        for i, byte_idx in enumerate(col_byte_indices):
            temp_output_bytes[byte_idx] = col_output_vars[i]

    # Flatten the collected output bytes
    output_vars = [var for byte_vars in temp_output_bytes for var in byte_vars]
    
    return output_vars, clauses, next_var_id


def encode_key_schedule(master_key_vars, next_var_id):
    """
    --- FULLY IMPLEMENTED AES-128 KEY SCHEDULE ---
    Generate 11 round keys (176 bytes total) from the 16-byte master key.
    
    Args:
        master_key_vars: 128 variables (master key, vars 1-128)
        next_var_id: Next available variable ID (should be 129)
    
    Returns:
        (round_keys, clauses, next_var_id)
        round_keys[i] = 128 variables for round i
    """
    clauses = []
    
    # The key schedule works with 44 32-bit words, w[0]...w[43]
    w = [None] * 44 # Each element will be a list of 32 variables
    
    # First 4 words (16 bytes / 128 bits) are the master key
    w[0] = master_key_vars[0:32]
    w[1] = master_key_vars[32:64]
    w[2] = master_key_vars[64:96]
    w[3] = master_key_vars[96:128]
    
    for i in range(4, 44):
        # Allocate 32 new variables for this word
        w[i] = list(range(next_var_id, next_var_id + 32))
        next_var_id += 32
        
        # Get predecessor words
        temp = w[i-1]
        w_i_minus_4 = w[i-4]
        
        if i % 4 == 0:
            # --- Special transformation for the first word of each round key ---
            
            # 1. RotWord
            temp_rot = rot_word(temp) # [b1, b2, b3, b0]
            
            # 2. SubWord
            # Allocate 32 vars for SubWord output
            temp_sub = list(range(next_var_id, next_var_id + 32))
            next_var_id += 32
            next_var_id = encode_sub_word(temp_rot, temp_sub, next_var_id, clauses)
            
            # 3. XOR with Rcon[i/4]
            # Create Rcon word [Rcon[i/4], 0, 0, 0]
            rcon_byte_val = Rcon[i // 4]
            
            # Allocate 32 vars for the Rcon word
            rcon_word_vars = list(range(next_var_id, next_var_id + 32))
            next_var_id += 32
            
            # Constrain these variables to the Rcon value
            for j in range(32):
                if j < 8: # First byte
                    bit_val = (rcon_byte_val >> j) & 1
                    clauses.append((rcon_word_vars[j],) if bit_val else (-rcon_word_vars[j],))
                else: # Other 3 bytes are 0
                    clauses.append((-rcon_word_vars[j],))
            
            # Allocate 32 vars for the XOR output
            temp_xor_rcon = list(range(next_var_id, next_var_id + 32))
            next_var_id += 32
            
            encode_xor_4byte_word(temp_sub, rcon_word_vars, temp_xor_rcon, clauses)
            
            # The final 'temp' for the main XOR is this transformed word
            temp = temp_xor_rcon
        
        # --- Main XOR step ---
        # w[i] = w[i-4] XOR temp
        encode_xor_4byte_word(w_i_minus_4, temp, w[i], clauses)

    # Group the 44 words into 11 round keys (16 bytes / 128 bits each)
    round_keys = []
    for i in range(11):
        round_key_vars = w[i*4] + w[i*4+1] + w[i*4+2] + w[i*4+3]
        round_keys.append(round_key_vars)
    
    return round_keys, clauses, next_var_id


def encode_aes_128(plaintext_bytes, ciphertext_bytes, master_key_vars):
    """
    Encode full AES-128 encryption as SAT.
    
    Args:
        plaintext_bytes: 16 bytes (known plaintext)
        ciphertext_bytes: 16 bytes (known ciphertext)
        master_key_vars: 128 variables (unknown key to recover)
    
    Returns:
        (clauses, total_vars, round_key_vars)
        
    Structure:
        plaintext -> AddRoundKey(round_key[0])
        -> 9 rounds of [SubBytes -> ShiftRows -> MixColumns -> AddRoundKey]
        -> final round [SubBytes -> ShiftRows -> AddRoundKey]
        -> ciphertext
    """
    clauses = []
    next_var_id = max(master_key_vars) + 1
    
    print("Encoding AES-128 circuit...")
    
    # Convert plaintext bytes to bit variables (fixed values)
    print("  [1/13] Encoding plaintext constraints...")
    plaintext_vars = []
    for byte in plaintext_bytes:
        for bit_idx in range(8):
            var = next_var_id
            next_var_id += 1
            plaintext_vars.append(var)
            
            # Fix this bit to plaintext value
            bit_val = (byte >> bit_idx) & 1
            if bit_val == 1:
                clauses.append((var,))
            else:
                clauses.append((-var,))
    
    # --- THIS IS THE CRITICAL FIX ---
    # Generate round keys *from the master key*
    print("  [2/13] Generating round keys (full key schedule)...")
    schedule_start_vars = next_var_id
    round_keys, key_sched_clauses, next_var_id = encode_key_schedule(
        master_key_vars, next_var_id
    )
    clauses.extend(key_sched_clauses)
    schedule_vars = next_var_id - schedule_start_vars
    schedule_clauses = len(key_sched_clauses)
    print(f"     Key schedule: {schedule_vars:,} variables, {schedule_clauses:,} clauses")
    
    # Initial AddRoundKey
    print("  [3/13] Encoding initial AddRoundKey...")
    state_vars, ark_clauses, next_var_id = encode_add_round_key(
        plaintext_vars, round_keys[0], next_var_id
    )
    clauses.extend(ark_clauses)
    
    # Main rounds (1-9)
    for round_num in range(1, 10):
        print(f"  [{3+round_num}/13] Encoding round {round_num}...")
        
        # SubBytes
        state_vars, sb_clauses, next_var_id = encode_sub_bytes(state_vars, next_var_id)
        clauses.extend(sb_clauses)
        
        # ShiftRows (permutation, no clauses)
        state_vars = encode_shift_rows(state_vars)
        
        # MixColumns
        state_vars, mc_clauses, next_var_id = encode_mix_columns_full(state_vars, next_var_id)
        clauses.extend(mc_clauses)
        
        # AddRoundKey
        state_vars, ark_clauses, next_var_id = encode_add_round_key(
            state_vars, round_keys[round_num], next_var_id
        )
        clauses.extend(ark_clauses)
    
    # Final round (no MixColumns)
    print(f"  [13/13] Encoding final round...")
    
    # SubBytes
    state_vars, sb_clauses, next_var_id = encode_sub_bytes(state_vars, next_var_id)
    clauses.extend(sb_clauses)
    
    # ShiftRows
    state_vars = encode_shift_rows(state_vars)
    
    # AddRoundKey
    state_vars, ark_clauses, next_var_id = encode_add_round_key(
        state_vars, round_keys[10], next_var_id
    )
    clauses.extend(ark_clauses)
    
    # Assert output equals ciphertext
    print("  [FINAL] Encoding ciphertext constraints...")
    for byte_idx, byte in enumerate(ciphertext_bytes):
        for bit_idx in range(8):
            var = state_vars[byte_idx*8 + bit_idx]
            bit_val = (byte >> bit_idx) & 1
            
            if bit_val == 1:
                clauses.append((var,))
            else:
                clauses.append((-var,))
    
    print(f"\n✅ AES-128 circuit encoding complete!")
    print(f"   Total clauses: {len(clauses):,}")
    print(f"   Total variables: {next_var_id - 1:,}") # -1 because next_var_id is 1-based
    print(f"   Master key variables: {master_key_vars[0]} to {master_key_vars[-1]}")
    
    return clauses, next_var_id - 1, round_keys


if __name__ == "__main__":
    print("="*80)
    print("TESTING FULL AES-128 CIRCUIT ENCODING")
    print("="*80)
    print()
    
    # Test vectors
    plaintext = bytes([
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
        0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34
    ])
    
    ciphertext = bytes([
        0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
        0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32
    ])
    
    # Allocate master key variables (unknown - to be recovered!)
    master_key_vars = list(range(1, 129))  # Variables 1-128
    
    print("Test case:")
    print(f"  Plaintext:  {plaintext.hex()}")
    print(f"  Ciphertext: {ciphertext.hex()}")
    print(f"  Master key: {'??' * 16} (to be recovered!)")
    print()
    
    # Encode AES
    clauses, total_vars, round_keys = encode_aes_128(
        plaintext, ciphertext, master_key_vars
    )
    
    print()
    print("="*80)
    print("ENCODING STATISTICS")
    print("="*80)
    print(f"Total clauses:   {len(clauses):,}")
    print(f"Total variables: {total_vars:,}")
    print(f"Average clause length: {sum(len(c) for c in clauses) / len(clauses):.1f}")
    print()
    
    # Breakdown
    # print("Expected breakdown:")
    # print(f"  S-boxes:     16 × 10 rounds × ~2048 clauses = {16*10*2048:,}")
    # print(f"  MixColumns:  4 × 9 rounds × ~16896 clauses  = {4*9*16896:,}")
    # print(f"  AddRoundKey: 11 × 128 × 4 clauses          = {11*128*4:,}")
    # print(f"  Key Schedule: ...")
    # print(f"  Fixed bits:  256 clauses (plaintext + ciphertext)")
    # print()
    
    print("🎯 Ready to test on quantum SAT solver!")
    print("   This will determine if AES-128 is crackable (k* < 10)")
    print("   or secure (k* ≈ 128)")
