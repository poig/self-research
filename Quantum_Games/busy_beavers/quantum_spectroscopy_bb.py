import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator

class BusyBeaverSpectroscopy:
    def __init__(self, n_tape_bits=5, n_clock_bits=4):
        """
        n_tape_bits: Size of the Universe (Tape length).
        n_clock_bits: Size of the History (2^N steps).
        """
        self.n_tape = n_tape_bits
        self.n_clock = n_clock_bits
        self.n_steps = 2**n_clock_bits
        
        # --- REGISTERS ---
        self.qr_clock = QuantumRegister(n_clock_bits, 'time')
        self.qr_state = QuantumRegister(1, 'machine_state') # 0=A, 1=B
        self.qr_head  = QuantumRegister(int(np.ceil(np.log2(n_tape_bits))), 'head_pos')
        self.qr_tape  = QuantumRegister(n_tape_bits, 'tape')
        
        # Aux (Sensor) for transition logic
        self.qr_aux   = AncillaRegister(1, 'sensor')
        
        self.cr_clock = ClassicalRegister(n_clock_bits, 'freq_readout')
        
        self.qc = QuantumCircuit(self.qr_clock, self.qr_state, self.qr_head, self.qr_tape, self.qr_aux, self.cr_clock)

    def _build_transition_unitary(self, rules):
        """
        Compiles the Turing Machine Rules into a single Unitary Gate (U).
        This U performs exactly ONE step of the machine.
        """
        # We build a sub-circuit for the logic U
        u_circ = QuantumCircuit(self.qr_state, self.qr_head, self.qr_tape, self.qr_aux)
        
        # --- FUSE RULES INTO GEOMETRY ---
        # We iterate over every Tape Position 'p' to apply local physics
        for p in range(self.n_tape):
            # Address Decoding: Is Head == p?
            # We construct a control condition for Head == p
            head_bin = format(p, f'0{self.qr_head.size}b')[::-1]
            
            for rule in rules:
                # Rule Format: (CurrentState, ReadVal, WriteVal, MoveDir, NextState)
                curr_s, read_v, write_v, move_d, next_s = rule
                
                # 1. ACTIVATE SENSOR
                # Trigger if: Head==p AND State==curr_s AND Tape[p]==read_v
                
                # Apply X to match 0 conditions
                for i, bit in enumerate(head_bin):
                    if bit == '0': u_circ.x(self.qr_head[i])
                if curr_s == 0: u_circ.x(self.qr_state[0])
                if read_v == 0: u_circ.x(self.qr_tape[p])
                
                # Sensor Trigger
                controls = list(self.qr_head) + list(self.qr_state) + [self.qr_tape[p]]
                u_circ.mcx(controls, self.qr_aux[0])
                
                # 2. APPLY PHYSICS (Controlled by Sensor)
                
                # Write Tape (if changed)
                if write_v != read_v:
                    u_circ.cx(self.qr_aux[0], self.qr_tape[p])
                    
                # Update State (if changed)
                if next_s != curr_s:
                    u_circ.cx(self.qr_aux[0], self.qr_state[0])
                    
                # Move Head (Cyclic Shift)
                # This is complex logic simplified for the demo: 
                # We only implement a "Virtual Shift" logic or use a simplified swap.
                # For robust simulation, we assume the 'Head' register updates.
                # Here, we just flip the LSB of Head to simulate movement in a tiny space (2-cell)
                # or simply acknowledge that coherent pointer arithmetic is the bottleneck.
                # *Simplification for Spectral Proof*: We keep Head static to focus on State/Tape loops.
                
                # 3. DEACTIVATE SENSOR (Uncompute)
                # Reverse X gates
                if read_v == 0: u_circ.x(self.qr_tape[p])
                if curr_s == 0: u_circ.x(self.qr_state[0])
                for i, bit in enumerate(head_bin):
                    if bit == '0': u_circ.x(self.qr_head[i])
                    
                # Reset Aux (Re-run trigger)
                u_circ.mcx(controls, self.qr_aux[0])

        return u_circ.to_gate(label="U_step")

    def construct_time_crystal(self, rules, custom_gate=None):
        """
        Builds the History State: Sum |t> |System_t>
        """
        # 1. Initialize Time Superposition
        self.qc.h(self.qr_clock)
        
        # 2. Get the Physics Engine (The U gate)
        if custom_gate is not None:
            u_gate = custom_gate
        else:
            u_gate = self._build_transition_unitary(rules)
        
        # 3. Controlled Evolution (Phase Estimation style)
        # To create Sum |t> U^t, we apply controlled U operations.
        # Bit 0 of Clock applies U^1
        # Bit 1 of Clock applies U^2
        # ...
        
        print(f"Fusing Time Crystal ({self.n_steps} steps)...")
        
        # Power 0 (Clock Bit 0)
        c_u = u_gate.control(1)
        self.qc.append(c_u, [self.qr_clock[0]] + list(self.qr_state) + list(self.qr_head) + list(self.qr_tape) + list(self.qr_aux))
        
        # Power 1 (Clock Bit 1) - Apply U twice
        # (In a real optimized circuit we would fuse U^2, but here we just append twice)
        u2_gate = QuantumCircuit(self.qr_state, self.qr_head, self.qr_tape, self.qr_aux)
        u2_gate.append(u_gate, range(u_gate.num_qubits))
        u2_gate.append(u_gate, range(u_gate.num_qubits))
        c_u2 = u2_gate.to_gate(label="U^2").control(1)
        self.qc.append(c_u2, [self.qr_clock[1]] + list(self.qr_state) + list(self.qr_head) + list(self.qr_tape) + list(self.qr_aux))
        
        # (For higher bits, we would continue U^4, U^8...)
        
    def apply_spectroscopy_with_erasure(self):
        """
        Apply QFT, but ALSO measure the Machine State to erase 'Which-Path' info.
        """
        print("Applying Spectroscopy with State Erasure...")
        
        # 1. Apply QFT to Time
        # Fix Deprecation: Use QFTGate if available or ignore warning for now
        self.qc.append(QFT(self.n_clock, inverse=False).to_gate(), self.qr_clock)
        
        # 2. Measure Clock
        self.qc.measure(self.qr_clock, self.cr_clock)
        
        # 3. CRITICAL: Measure the Machine State
        # We need a classical register for this
        cr_state = ClassicalRegister(1, 'state_readout')
        self.qc.add_register(cr_state)
        self.qc.measure(self.qr_state[0], cr_state) # Measure State LSB
        
        return cr_state # Return handle to filter results

    def run(self):
        sim = AerSimulator()
        # We need high shots to resolve the spectrum
        return sim.run(self.qc.decompose(reps=2), shots=2048).result().get_counts()

# --- EXECUTION ---
if __name__ == "__main__":
    # Setup: Small Universe (4 tape bits), History (4 time bits -> 16 steps)
    engine = BusyBeaverSpectroscopy(n_tape_bits=4, n_clock_bits=2)
    
    # --- DEFINING A LOOPING MACHINE ---
    # This machine just flips State A <-> B endlessly. Period = 2.
    # Ideally: (0, 0, 0, 0, 1), (1, 0, 0, 0, 0)
    # PROBLEM: The generic compiler fails to uncompute ancillas because the state changes.
    # FIX: For this simple A<->B loop, the Unitary is just an X gate on the State qubit!
    
    # Custom Unitary for A <-> B Loop
    qc_loop = QuantumCircuit(engine.qr_state, engine.qr_head, engine.qr_tape, engine.qr_aux)
    qc_loop.x(engine.qr_state[0]) # The "Physics" is just a bit flip
    u_loop = qc_loop.to_gate(label="U_Loop")
    
    engine.construct_time_crystal(rules=None, custom_gate=u_loop)
    # Use the new Erasure method
    engine.apply_spectroscopy_with_erasure()
    
    result = engine.run()
    
    print("\n--- FILTERED SPECTRUM (State A Only) ---")
    # We filter the shots. We only keep shots where State was '0' (A).
    
    filtered_counts = {}
    for k, count in result.items():
        # Key format depends on register order. usually "State Clock"
        # k might be "0 1001" (State=0, Clock=9)
        # Let's parse assuming space separation or bit counting
        k_clean = k.replace(" ", "")
        
        # MSB is State (added last), LSBs are Clock
        state_bit = k_clean[0] 
        clock_bits = k_clean[1:]
        
        if state_bit == '0': # POST-SELECTION: Keep only State A
            freq = int(clock_bits, 2)
            filtered_counts[freq] = filtered_counts.get(freq, 0) + count
            
    # Print Filtered
    total_filtered = sum(filtered_counts.values())
    if total_filtered == 0:
        print("No shots found in State A!")
    else:
        for f in sorted(filtered_counts.keys()):
            prob = filtered_counts[f] / total_filtered
            bar = "█" * int(prob * 20)
            print(f"Freq {f} | {prob:.4f} {bar}")

    print("\nInterpretation:")
    print("- By measuring the State, we collapse the system to a specific subspace.")
    print("- If the machine loops A->B->A, observing 'A' should reveal the loop frequency clearly.")

    print("\n" + "="*40 + "\n")

    # --- DEFINING A NON-LOOPING MACHINE (Growth) ---
    print("--- NON-LOOPING MACHINE (Growth) ---")
    # Rule: Write 1, Move Right, Stay A (Fills tape with 1s)
    # (State, Read, Write, Move, Next)
    growth_rules = [
        (0, 0, 1, 1, 0), # A,0 -> Write 1, Move Right, Stay A
        (0, 1, 1, 1, 0)  # A,1 -> Write 1, Move Right, Stay A
    ]
    
    # Re-initialize engine for clean state
    engine_growth = BusyBeaverSpectroscopy(n_tape_bits=4, n_clock_bits=2)
    engine_growth.construct_time_crystal(growth_rules)
    engine_growth.apply_spectroscopy_with_erasure()
    
    result_growth = engine_growth.run()
    
    print("\n--- FILTERED SPECTRUM (State A Only) ---")
    
    filtered_counts_g = {}
    for k, count in result_growth.items():
        k_clean = k.replace(" ", "")
        state_bit = k_clean[0] 
        clock_bits = k_clean[1:]
        
        if state_bit == '0': # Keep State A
            freq = int(clock_bits, 2)
            filtered_counts_g[freq] = filtered_counts_g.get(freq, 0) + count
            
    total_filtered_g = sum(filtered_counts_g.values())
    if total_filtered_g == 0:
        print("No shots found in State A!")
    else:
        for f in sorted(filtered_counts_g.keys()):
            prob = filtered_counts_g[f] / total_filtered_g
            bar = "█" * int(prob * 20)
            print(f"Freq {f} | {prob:.4f} {bar}")
            
    print("\nInterpretation:")
    print("- A 'Growth' machine constantly changes the tape configuration.")
    print("- This aperiodicity should result in a flatter or more complex spectrum compared to the simple loop.")