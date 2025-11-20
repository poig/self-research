import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator

# ==========================================
# PART 1: THE QUANTUM MULTIVERSE FILTER
# ==========================================

class QuantumGenesis:
    def __init__(self, n_steps=6):
        """
        The Quantum Engine that searches the Hilbert Space for Halting Machines.
        n_steps: Number of time evolution steps (BB-2 champion halts at step 6).
        """
        self.n_steps = n_steps
        self.n_tape = 5 # Small finite universe for the quantum simulation
        
        # --- GENOME REGISTERS ---
        # BB-2 Genome: 4 Transition Rules (A0, A1, B0, B1)
        # Each Rule: Write(1), Move(1), NextState(1) -> 12 Qubits
        self.qr_genome = QuantumRegister(12, 'genome')
        
        # --- PHYSICAL REGISTERS ---
        self.qr_state = QuantumRegister(1, 'state') # 0=A, 1=B
        self.qr_tape  = QuantumRegister(self.n_tape, 'tape')
        self.qr_head  = QuantumRegister(3, 'head')  # 3 bits for 5 positions
        
        # --- CONTROL REGISTERS ---
        self.qr_halt  = QuantumRegister(1, 'halt_flag')
        self.qr_aux   = AncillaRegister(2, 'sensor')
        
        # --- READOUT ---
        self.cr_genome = ClassicalRegister(12, 'genome_readout')
        self.cr_halt   = ClassicalRegister(1, 'halt_readout')
        
        self.qc = QuantumCircuit(
            self.qr_genome, self.qr_state, self.qr_tape, 
            self.qr_head, self.qr_halt, self.qr_aux, 
            self.cr_genome, self.cr_halt
        )

    def _fuse_programmable_physics(self):
        """
        The 'Landscape Logic'. This function fuses the Genome (Rules) 
        into the Geometry of the Tape/State.
        """
        # Map of Scenarios (State, Read) -> Genome Index
        # Genome Layout: [A0_W, A0_M, A0_N, A1_W...]
        scenarios = [
            (0, 0, 0), # State A, Read 0 -> Bit 0
            (0, 1, 3), # State A, Read 1 -> Bit 3
            (1, 0, 6), # State B, Read 0 -> Bit 6
            (1, 1, 9)  # State B, Read 1 -> Bit 9
        ]
        
        # Iterate over every cell in the tape (Spatial Expansion)
        for p in range(self.n_tape):
            # Head Position Binary (for control)
            head_bin = format(p, '03b')[::-1]
            
            for (curr_s, read_v, rule_idx) in scenarios:
                
                # --- 1. SENSOR ACTIVATION ---
                # Detect: Head==p AND State==curr_s AND Tape[p]==read_v
                
                # Apply X gates to match '0' conditions
                for i, bit in enumerate(head_bin):
                    if bit == '0': self.qc.x(self.qr_head[i])
                if curr_s == 0: self.qc.x(self.qr_state[0])
                if read_v == 0: self.qc.x(self.qr_tape[p])
                
                # Activate Sensor (Aux 0)
                controls = list(self.qr_head) + list(self.qr_state) + [self.qr_tape[p]]
                self.qc.mcx(controls, self.qr_aux[0])
                
                # --- 2. ACTUATOR (Controlled by Sensor AND Genome) ---
                # Apply laws of physics only if Sensor is active.
                # The physics itself is determined by the Genome Qubits.
                
                # A. WRITE TAPE (Rule Bit 0)
                self.qc.ccx(self.qr_aux[0], self.qr_genome[rule_idx+0], self.qr_tape[p])
                
                # B. MOVE HEAD (Rule Bit 1)
                # Simplified Logic: If Move=1, Flip Head LSB (Local jitter for small tapes)
                # In a full sim, this would be a Modular Adder.
                self.qc.ccx(self.qr_aux[0], self.qr_genome[rule_idx+1], self.qr_head[0])
                
                # C. UPDATE STATE (Rule Bit 2)
                self.qc.ccx(self.qr_aux[0], self.qr_genome[rule_idx+2], self.qr_state[0])
                
                # D. HALT DETECTION
                # Definition: Halt if State B (1) -> State B (1)
                # If Current=1 AND Next(Genome)=1 AND Sensor=1 -> Halt=1
                if curr_s == 1:
                     self.qc.mcx([self.qr_aux[0], self.qr_genome[rule_idx+2]], self.qr_halt[0])
                
                # --- 3. UNCOMPUTE SENSOR ---
                if read_v == 0: self.qc.x(self.qr_tape[p])
                if curr_s == 0: self.qc.x(self.qr_state[0])
                for i, bit in enumerate(head_bin):
                    if bit == '0': self.qc.x(self.qr_head[i])
                
                # Reset Aux
                self.qc.mcx(controls, self.qr_aux[0])

    def run_evolution(self):
        print(f"Initializing Quantum Multiverse (Steps={self.n_steps})...")
        
        # 1. Superposition of All Physical Laws (Genomes)
        self.qc.h(self.qr_genome)
        
        # 2. Initialize Head at center
        start_pos = self.n_tape // 2
        bin_start = format(start_pos, '03b')[::-1]
        for i, bit in enumerate(bin_start):
            if bit == '1': self.qc.x(self.qr_head[i])
            
        # 3. Evolve Time
        # We apply the physics layer N times
        for t in range(self.n_steps):
            self._fuse_programmable_physics()
            
        # 4. Collapse the Wavefunction
        # We measure Halt first (Post-Selection)
        self.qc.measure(self.qr_halt, self.cr_halt)
        # We measure the Genome to see WHO halted
        self.qc.measure(self.qr_genome, self.cr_genome)
        
        sim = AerSimulator()
        # High shots to find rare events
        return sim.run(self.qc, shots=10000).result().get_counts()

# ==========================================
# PART 2: THE CLASSICAL REFEREE
# ==========================================

class ClassicalReferee:
    def __init__(self, genome_str):
        self.genome_str = genome_str.replace(" ", "")
        self.rules = self._parse_genome(self.genome_str)
        self.tape = {}
        self.head = 0
        self.state = 0 # A=0, B=1
        self.steps = 0
        self.max_steps = 100 
        
    def _parse_genome(self, s):
        # 12 bits -> 4 rules (A0, A1, B0, B1)
        # Block: Write, Move(0=L,1=R), NextState
        rules = {}
        scenarios = [(0,0), (0,1), (1,0), (1,1)]
        for i, sc in enumerate(scenarios):
            block = s[i*3 : (i+1)*3]
            w = int(block[0])
            m = 1 if block[1]=='1' else -1
            n = int(block[2])
            rules[sc] = (w, m, n)
        return rules

    def compete(self):
        # Run the machine
        history = []
        while self.steps < self.max_steps:
            val = self.tape.get(self.head, 0)
            
            # Check Halt (B->B transition)
            rule = self.rules[(self.state, val)]
            w, m, n = rule
            
            if self.state == 1 and n == 1:
                self.tape[self.head] = w
                self.steps += 1
                return "HALT", self.steps, sum(self.tape.values())
            
            self.tape[self.head] = w
            self.head += m
            self.state = n
            self.steps += 1
            
        return "INF", self.steps, sum(self.tape.values())

# ==========================================
# PART 3: THE TOURNAMENT (MAIN)
# ==========================================

if __name__ == "__main__":
    # 1. Run Quantum Filter
    q_eng = QuantumGenesis(n_steps=6) # BB-2 Champion needs 6 steps
    q_results = q_eng.run_evolution()
    
    # 2. Parse Survivors (Halt=1)
    survivors = []
    print("\n--- QUANTUM SURVIVORS (Halt Bit = 1) ---")
    
    for k, count in q_results.items():
        # k is "Halt Genome" (e.g., "1 101010...")
        # Qiskit formatting depends on reg order. 
        # We defined Halt, then Genome.
        # Likely: "Genome Halt" or "Halt Genome"
        # Let's assume standard space split:
        parts = k.split()
        if len(parts) == 2:
            halt_bit = parts[0]
            genome = parts[1]
        else:
            # Fallback parsing
            k_clean = k.replace(" ", "")
            halt_bit = k_clean[0]
            genome = k_clean[1:]
            
        if halt_bit == '1':
            survivors.append((genome, count))
            
    # Sort by Quantum Probability (Count)
    # Higher count = More robust halting (or halted earlier/multiple paths)
    survivors.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(survivors)} unique genomes that halted in the simulation.")
    print("Top 5 Quantum Probabilities:")
    for g, c in survivors[:5]:
        print(f"Genome: {g} | Count: {c}")
        
    # 3. Classical Finals
    print("\n--- CLASSICAL FINALS (Infinite Tape) ---")
    print(f"{'GENOME':<15} | {'STATUS':<6} | {'STEPS':<5} | {'ONES (Score)'}")
    print("-" * 55)
    
    leaderboard = []
    
    # We only test the survivors found by the quantum computer
    for genome, q_conf in survivors:
        referee = ClassicalReferee(genome)
        status, steps, score = referee.compete()
        
        # Only print interesting ones (HALT or High Score)
        if status == "HALT":
            leaderboard.append((genome, steps, score))
            if len(leaderboard) <= 10: # Print first 10 live
                print(f"{genome:<15} | {status:<6} | {steps:<5} | {score}")
                
    # 4. Crowning the Champion
    if leaderboard:
        # Sort by Steps (Busy Beaver metric)
        champion_bb = max(leaderboard, key=lambda x: x[1])
        # Sort by Ones (Sigma metric)
        champion_sig = max(leaderboard, key=lambda x: x[2])
        
        print("-" * 55)
        print(f"BUSY BEAVER CHAMPION (Max Steps):")
        print(f"Genome: {champion_bb[0]}")
        print(f"Steps:  {champion_bb[1]}")
        print(f"Score:  {champion_bb[2]}")
        
        print(f"\nSIGMA CHAMPION (Max Ones):")
        print(f"Genome: {champion_sig[0]}")
        print(f"Steps:  {champion_sig[1]}")
        print(f"Score:  {champion_sig[2]}")
        
        # KNOWN BB-2 CHAMPION (Rado's Sigma(2,2))
        # 1 1 1 (A0->1RB) | 1 1 1 (A1->1RB) | 1 1 0 (B0->1LA) | 1 1 1 (B1->Halt)
        # Genome: 111 111 110 111
        # Note: Our encoding might be slightly permuted (L/R bit, State bit)
        # But we should see 6 steps, 4 ones.
    else:
        print("No valid Halting Busy Beavers found in the sample.")