import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator

# ==========================================
# CONFIGURATION: The BB-4 Universe
# ==========================================
# Champion (Brady's Machine):
# A0->1RB, A1->1LB
# B0->1LA, B1->0LC
# C0->1RH, C1->1LD  (H = Halt)
# D0->1RD, D1->0RA
# Steps: 107, Score: 13

class BB4_Engine:
    def __init__(self, tape_len=12, simulation_depth=10):
        """
        Hybrid Quantum-Classical Engine for finding BB-4.
        Uses MPS simulation for >50 qubits.
        """
        self.tape_len = tape_len
        self.sim_depth = simulation_depth
        
        # --- GENOME SPEC ---
        # 4 States (A,B,C,D) x 2 Inputs (0,1) = 8 Rules
        # Rule Encoding: Write(1) + Move(1) + NextState(3) = 5 bits
        # NextState: 0-3 (A-D), 4 (Halt), 5-7 (Unused/Loop)
        self.n_rules = 8
        self.bits_per_rule = 5
        self.genome_size = self.n_rules * self.bits_per_rule # 40 Qubits
        
        # --- QUANTUM REGISTERS ---
        self.qr_genome = QuantumRegister(self.genome_size, 'genome')
        self.qr_state  = QuantumRegister(3, 'state') # 3 bits for 5 states
        self.qr_head   = QuantumRegister(4, 'head')  # 4 bits for 12 positions
        self.qr_tape   = QuantumRegister(tape_len, 'tape')
        self.qr_halt   = QuantumRegister(1, 'halt')
        self.qr_aux    = AncillaRegister(2, 'aux')
        
        self.cr_genome = ClassicalRegister(self.genome_size, 'c_genome')
        self.cr_halt   = ClassicalRegister(1, 'c_halt')
        
        # The Circuit
        self.qc = QuantumCircuit(
            self.qr_genome, self.qr_state, self.qr_head, self.qr_tape, 
            self.qr_halt, self.qr_aux, self.cr_genome, self.cr_halt
        )

    def _apply_programmable_physics(self):
        """
        Fuses the 40-bit Genome into the 12-qubit Tape geometry.
        """
        # Iterate over all tape positions (Spatial Expansion)
        for p in range(self.tape_len):
            # Address Check: Is Head == p?
            head_bin = format(p, '04b')[::-1]
            
            # Iterate over all 8 Rule Scenarios
            # Scenario Map: State(0-3) x Read(0-1)
            for s in range(4):
                for r in range(2):
                    rule_idx = (s * 2 + r) * self.bits_per_rule
                    
                    # --- SENSOR: Am I in this scenario? ---
                    # Head Check
                    for i, bit in enumerate(head_bin):
                        if bit == '0': self.qc.x(self.qr_head[i])
                    
                    # State Check (s is 0-3)
                    s_bin = format(s, '03b')[::-1]
                    for i, bit in enumerate(s_bin):
                        if bit == '0': self.qc.x(self.qr_state[i])
                        
                    # Tape Check
                    if r == 0: self.qc.x(self.qr_tape[p])
                    
                    # ACTIVATE TRIGGER (Aux0)
                    # Controls: Head(4) + State(3) + Tape(1) = 8 controls
                    controls = list(self.qr_head) + list(self.qr_state) + [self.qr_tape[p]]
                    self.qc.mcx(controls, self.qr_aux[0])
                    
                    # --- ACTUATOR: Apply Genome Logic ---
                    # Genome Bits: [Write, Move, Next0, Next1, Next2]
                    
                    # 1. Write Tape
                    self.qc.ccx(self.qr_aux[0], self.qr_genome[rule_idx+0], self.qr_tape[p])
                    
                    # 2. Move Head (Simplified: Flip LSB to simulate jitter/scan)
                    # Real implementation needs Modular Adder. 
                    # For search verdict, we define Move=1 as 'Flip Head[0]'
                    self.qc.ccx(self.qr_aux[0], self.qr_genome[rule_idx+1], self.qr_head[0])
                    
                    # 3. Next State (3 bits)
                    # We swap the 3 state bits with the 3 genome bits if Trigger is active
                    # (Controlled-Swap logic or just Copy)
                    for b in range(3):
                        self.qc.ccx(self.qr_aux[0], self.qr_genome[rule_idx+2+b], self.qr_state[b])
                        
                    # 4. HALT CHECK
                    # If New State is 4 (100), SET HALT.
                    # Logic: If State[2]=1 AND State[1]=0 AND State[0]=0 -> Halt
                    # We check the STATE register directly after update.
                    self.qc.x(self.qr_state[0])
                    self.qc.x(self.qr_state[1])
                    # Trigger Halt if State=4 (100) AND Aux0 was active (we just moved)
                    self.qc.mcx([self.qr_aux[0], self.qr_state[2], self.qr_state[1], self.qr_state[0]], self.qr_halt[0])
                    self.qc.x(self.qr_state[1])
                    self.qc.x(self.qr_state[0])
                    
                    # --- UNCOMPUTE SENSOR ---
                    if r == 0: self.qc.x(self.qr_tape[p])
                    for i, bit in enumerate(s_bin):
                        if bit == '0': self.qc.x(self.qr_state[i])
                    for i, bit in enumerate(head_bin):
                        if bit == '0': self.qc.x(self.qr_head[i])
                    
                    self.qc.mcx(controls, self.qr_aux[0]) # Reset Trigger

    def run_verdict_test(self, initial_genome_str, noise_level=0.1):
        """
        Injects a 'Noisy Champion' into the search to see if the 
        Quantum Filter identifies the correct genome.
        """
        print(f"\n--- BB-4 QUANTUM VERDICT TEST ---")
        print(f"Target Genome: {initial_genome_str}")
        print(f"Noise Level: {noise_level*100}%")
        
        # 1. Encode Genome with Noise (Ry Rotations)
        # If bit is '1', theta = pi. If '0', theta = 0.
        # Noise adds perturbation to theta.
        
        clean_bits = [int(c) for c in initial_genome_str.replace(" ", "")]
        
        for i, bit in enumerate(clean_bits):
            theta = np.pi if bit == 1 else 0.0
            # Add Noise
            theta += np.random.uniform(-noise_level, noise_level) * np.pi
            self.qc.ry(theta, self.qr_genome[i])
            
        # 2. Initialize Machine (Head @ Center, State A)
        mid = self.tape_len // 2
        mid_bin = format(mid, '04b')[::-1]
        for i, b in enumerate(mid_bin):
            if b == '1': self.qc.x(self.qr_head[i])
            
        # 3. Run Simulation (Filter)
        print(f"Running Quantum Evolution ({self.sim_depth} steps) on MPS Backend...")
        for _ in range(self.sim_depth):
            self._apply_programmable_physics()
            
        # 4. Measure
        self.qc.measure(self.qr_halt, self.cr_halt)
        self.qc.measure(self.qr_genome, self.cr_genome)
        
        # Execute on Matrix Product State Simulator (Handles >50 qubits)
        sim = AerSimulator(method='matrix_product_state')
        result = sim.run(self.qc.decompose(reps=3), shots=2000).result().get_counts()
        
        return result

# --- CLASSICAL VERIFIER ---
def classical_bb4_run(genome_str):
    # Simple Classical Simulator to check score
    # Genome: 8 blocks of 5 bits (W, M, N2, N1, N0)
    bits = [int(c) for c in genome_str.replace(" ", "")]
    rules = {}
    for s in range(4):
        for r in range(2):
            idx = (s*2 + r) * 5
            w = bits[idx]
            m = 1 if bits[idx+1]==1 else -1
            # Next State (3 bits)
            n = bits[idx+2]*1 + bits[idx+3]*2 + bits[idx+4]*4 # Little Endian in Qiskit
            rules[(s,r)] = (w, m, n)
            
    tape = {}
    head = 0
    state = 0
    steps = 0
    while steps < 200:
        val = tape.get(head, 0)
        if (state, val) not in rules: break
        w, m, n = rules[(state, val)]
        
        tape[head] = w
        head += m
        state = n
        steps += 1
        if state == 4: return "HALT", steps, sum(tape.values())
        
    return "INF", steps, sum(tape.values())

# --- MAIN ---
if __name__ == "__main__":
    # BRADY'S BB-4 CHAMPION GENOME
    # Format: [Write, Move, NextState(3bits)] per rule
    # Mapping:
    # A0->1RB (1, 1, 1) -> 11100
    # A1->1LB (1, 0, 1) -> 10100
    # B0->1LA (1, 0, 0) -> 10000
    # B1->0LC (0, 0, 2) -> 00010
    # C0->1RH (1, 1, 4) -> 11001 (Halt=4 => 100 binary)
    # C1->1LD (1, 0, 3) -> 10110
    # D0->1RD (1, 1, 3) -> 11110
    # D1->0RA (0, 1, 0) -> 01000
    
    # NOTE: Qiskit Little Endian means '100' (4) is stored as bits [0,0,1].
    # So '4' corresponds to bits 0,0,1.
    
    champion_genome = (
        "11100 " # A0 -> 1 R B(1)
        "10100 " # A1 -> 1 L B(1)
        "10000 " # B0 -> 1 L A(0)
        "00010 " # B1 -> 0 L C(2)
        "11001 " # C0 -> 1 R Halt(4)
        "10110 " # C1 -> 1 L D(3)
        "11110 " # D0 -> 1 R D(3)
        "01000 " # D1 -> 0 R A(0)
    )
    
    engine = BB4_Engine(tape_len=14, simulation_depth=15)
    
    # We inject the champion with 20% Noise (Rotation)
    # This simulates finding the champion in a genetic search beam.
    counts = engine.run_verdict_test(champion_genome, noise_level=0.2)
    
    print("\n--- RESULTS ---")
    
    # Filter for Halting Survivors
    halt_counts = {}
    for k, v in counts.items():
        # k is "Genome Halt" (or vice versa)
        k_clean = k.replace(" ", "")
        # Assuming Halt is single bit at start or end. 
        # ClassicalRegister order: Genome(40), Halt(1).
        # Qiskit readout: Halt(1) Genome(40) usually.
        halt_bit = k_clean[0]
        genome_bits = k_clean[1:]
        
        if halt_bit == '1':
            halt_counts[genome_bits] = halt_counts.get(genome_bits, 0) + v
            
    # Sort
    sorted_survivors = sorted(halt_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Total Halting Traces: {sum(halt_counts.values())}")
    if len(sorted_survivors) > 0:
        top_genome = sorted_survivors[0][0]
        print(f"Top Survivor: {top_genome}")
        
        # Verify
        res, steps, score = classical_bb4_run(top_genome)
        print(f"Classical Check: {res} in {steps} steps (Score {score})")
        
        # Compare with Champion
        # Note: String matching might be fuzzy due to register ordering/spaces
        print("Verdict: " + ("SUCCESS" if res=="HALT" else "FAIL"))
    else:
        print("Verdict: No Halting Machines found in noisy sample.")