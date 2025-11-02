

direct inspire from this video: https://youtu.be/bO5nvE289ec?si=whOaCAf5QJqaYu7d

Block Component,Function,Status/Role in O(N) Classical Loop
1. QNN Circuit (UQNN​),The parameterized trial state ansatz.,Quantum: No change.
2. Dynamic Gradient/Metric,Measures the gradient (g) and QFIM (F) components.,Quantum-Aware: Use Parameter Shift Rule or Hadamard tests with mid-circuit measurement (MCM) to efficiently gather the data as classical bits.
3. Quantum Linear Solver Circuit (UHHL​),Solves the N×N linear system Fg~​=g for the natural gradient g~​.,"Quantum: Eliminates the classical O(N3) bottleneck, achieving O(logN) computation time."
4. Classical O(N) Control Unit,"Manages I/O, convergence, and final update.",Classical (The Minimum):
4a. Update Extraction (I/O),Takes the output state $,\tilde{g}\rangle$ from UHHL​ (which has been measured and averaged) and extracts the final N continuous values Δθ to memory.
4b. Convergence Logic,Checks the scalar cost function and applies classical update logic θnew​=θold​+η⋅Δθ.,
4c. Reprogramming,Sends the N new floating-point values θnew​ to the quantum device controllers for the next iteration.,