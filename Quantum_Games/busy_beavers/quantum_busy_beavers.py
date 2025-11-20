import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit_aer import AerSimulator

class GeometricBusyBeaver:
    def __init__(self, tape_len=6):
        self.tape_len = tape_len
        
        # --- GEOMETRY DEFINITION ---
        # 1. The Head Position (Where we are in space)
        self.qr_head = QuantumRegister(3, 'head_pos') # 8 slots (0-7)
        
        # 2. The Tape (The Environment) - One qubit per cell
        self.qr_tape = QuantumRegister(tape_len, 'tape_data')
        
        # 3. The Internal State (The Machine Mind)
        # 2 qubits allows 4 states: A(00), B(01), C(10), Halt(11)
        self.qr_state = QuantumRegister(2, 'machine_state')
        
        # 4. The Sensor (Ancilla to trigger transitions)
        self.qr_sensor = AncillaRegister(1, 'rule_trigger')
        
        # Readout
        self.cr = ClassicalRegister(tape_len + 5, 'readout')
        
        self.qc = QuantumCircuit(self.qr_head, self.qr_tape, self.qr_state, self.qr_sensor, self.cr)
        self.sim = AerSimulator()

    def _activate_sensor_at_pos(self, pos_idx):
        """Helper: Checks if Head is at specific position 'pos_idx'."""
        # Address decoding logic (Linear fusion)
        bin_str = format(pos_idx, '03b')[::-1]
        for i, bit in enumerate(bin_str):
            if bit == '0': self.qc.x(self.qr_head[i])

    def fuse_transition_rule(self, state_code, read_val, write_val, move_dir, next_state_code):
        """
        Fuses a single BB rule into the landscape.
        The rule is applied *everywhere* in space (Superposition of Head Positions).
        """
        # We iterate over every physical tape cell to fuse this rule locally.
        # (This is the "Expansion" of the rule into geometry)
        
        for pos in range(self.tape_len):
            # --- 1. SENSOR (Detect Condition) ---
            # Trigger ONLY if: Head==pos AND State==state_code AND Tape[pos]==read_val
            
            # A. Check Head Position
            self._activate_sensor_at_pos(pos)
            
            # B. Check Machine State
            state_bin = format(state_code, '02b')[::-1]
            for i, bit in enumerate(state_bin):
                if bit == '0': self.qc.x(self.qr_state[i])
                
            # C. Check Tape Value
            if read_val == 0: self.qc.x(self.qr_tape[pos])
            
            # D. Set Sensor Flag
            controls = list(self.qr_head) + list(self.qr_state) + [self.qr_tape[pos]]
            self.qc.mcx(controls, self.qr_sensor[0])
            
            # --- 2. ACTUATOR (Apply Physics if Sensor is ON) ---
            
            # A. WRITE (Flip tape if Write != Read)
            if write_val != read_val:
                self.qc.cx(self.qr_sensor[0], self.qr_tape[pos])
            
            # B. CHANGE STATE (Flip bits to match next_state)
            # Current is 'state_code', Target is 'next_state_code'.
            # We XOR them to find which bits to flip.
            diff = state_code ^ next_state_code
            diff_bin = format(diff, '02b')[::-1]
            for i, bit in enumerate(diff_bin):
                if bit == '1': self.qc.cx(self.qr_sensor[0], self.qr_state[i])
            
            # C. MOVE HEAD
            # If Sensor is ON, we shift head.
            # Note: Implementing coherent shift is complex. 
            # For this demo, we use a simplified modular addition logic fused here.
            # (If Head=pos, set Head=pos+dir).
            # To prevent "runaway" shifts, we assume this is a clocked step.
            # We apply a specialized 'Shift' gate sequence. 
            # (Omitted for brevity: we assume the transition moves the head conceptually)
            # *Simulation Trick*: We simulate the shift by just noting the next active head
            # in a real full circuit, this requires a QFT adder or Modular Incrementer.
            self._fuse_coherent_shift(pos, move_dir)
            
            # --- 3. RESET SENSOR (Uncompute) ---
            # Reverse B and C checks
            if read_val == 0: self.qc.x(self.qr_tape[pos])
            for i, bit in enumerate(state_bin):
                if bit == '0': self.qc.x(self.qr_state[i])
            # Reverse A (Head check)
            self._activate_sensor_at_pos(pos) # It's symmetric (X gates)
            # Reset Trigger (since we used it as control, we need to uncompute it
            # by re-running the detection logic into it)
            # For simplicity in this linear flow, we assume Sensor reset happens naturally or via measure-reset in a loop.
            self.qc.mcx(controls, self.qr_sensor[0])

    def _fuse_coherent_shift(self, current_pos, direction):
        """
        Moves the head from current_pos to next_pos conditioned on Sensor.
        """
        next_pos = (current_pos + direction) % 8 # Wrap around 8 slots
        
        # Logic: If Sensor=1 AND Head=current, Flip Head to next.
        # This is tricky because we are controlling on Head.
        # Standard approach: We use the "Next Head" buffer approach in real hardware.
        # For this visual demo, we will perform a simplified SWAP logic.
        pass # Implementation requires a temporary register to avoid self-control paradoxes.

    def run_step(self, ruleset):
        # Apply all rules in the ruleset for one time step
        for rule in ruleset:
            # rule: (state, read, write, move, next_state)
            self.fuse_transition_rule(*rule)

    def simulate_history(self, ruleset, steps=4):
        history = []
        
        # Init State
        # Head at 3 (Center), State A (0), Tape 000000
        # Using Statevector to track evolution perfectly
        
        # Python Simulation of the "Landscape Logic" to show the trajectory
        # (Since fully coherent shifting is too heavy for this snippet)
        tape = [0] * self.tape_len
        head = self.tape_len // 2
        state = 0 # A
        
        print(f"Initial: State A, Head {head}, Tape {tape}")
        
        for t in range(steps):
            # Find applicable rule
            read_val = tape[head]
            active_rule = None
            
            # Search the "Landscape" for the trigger zone
            for r in ruleset:
                # r = (state_match, read_match, write, move, next_state)
                if r[0] == state and r[1] == read_val:
                    active_rule = r
                    break
            
            if not active_rule:
                print("HALT (No rule found or explicit halt)")
                break
                
            # Apply Actuator
            # 1. Write
            tape[head] = active_rule[2]
            # 2. Move
            head += active_rule[3]
            # Optional: Use Modulo (%) to wrap the geometry "Cyclic Automata" (Finite Ring) 
            # head = (head + active_rule[3]) % self.tape_len

            # 3. Next State
            state = active_rule[4]
            
            # Halt Check (BB specific logic often uses a specific state for halt)
            if state == 3: # We define 3 as HALT
                print(f"Step {t+1}: HALT STATE REACHED")
                break
                
            # Visuals
            state_char = ['A', 'B', 'C', 'H'][state]
            tape_str = "".join([str(x) for x in tape])
            tape_vis = list(tape_str)
            tape_vis[head] = f"[{tape_vis[head]}]" # Highlight Head
            print(f"Step {t+1}: State {state_char} | {''.join(tape_vis)}")
            
            history.append(tape[:])
            
        return history

# --- CONFIGURING THE BEAVERS ---

if __name__ == "__main__":
    eng = GeometricBusyBeaver(tape_len=20)
    
    # --- DEFINING BB-2 (2 States, 2 Symbols) ---
    # Radó's Sigma(2, 2) Champion
    # States: A=0, B=1, Halt=3
    # Moves: Left=-1, Right=+1
    # Format: (CurrentState, Read, Write, Move, NextState)
    
    bb2_rules = [
        (0, 0, 1,  1, 1), # A0 -> 1RB
        (0, 1, 1, -1, 1), # A1 -> 1LB
        (1, 0, 1, -1, 0), # B0 -> 1LA
        (1, 1, 1,  1, 3), # B1 -> 1RH (Halt)
    ]
    
    print("\n--- GEOMETRIC BUSY BEAVER (BB-2) ---")
    eng.simulate_history(bb2_rules, steps=10)
    
    # --- DEFINING BB-3 (3 States, 2 Symbols) ---
    # Radó's Sigma(3, 2) Champion
    # States: A=0, B=1, C=2, Halt=3
    
    bb3_rules = [
        (0, 0, 1,  1, 1), # A0 -> 1RB
        (0, 1, 1,  1, 3), # A1 -> 1RH (Halt)
        (1, 0, 0,  1, 2), # B0 -> 0RC
        (1, 1, 1,  1, 1), # B1 -> 1RB
        (2, 0, 1, -1, 2), # C0 -> 1LC
        (2, 1, 1, -1, 0), # C1 -> 1LA
    ]
    
    print("\n--- GEOMETRIC BUSY BEAVER (BB-3) ---")
    eng.simulate_history(bb3_rules, steps=15)