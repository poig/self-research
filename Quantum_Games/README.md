# Quantum Games: Approaches to Non-Unitary Cellular Automata

This directory contains research code exploring different methods to simulate **Conway's Game of Life** (a non-linear, irreversible cellular automaton) and **Busy Beaver Turing Machines** on **Quantum Computers** (which are strictly linear and reversible).

Each approach represents a different trade-off between **Qubit Efficiency**, **Physical Correctness**, and **Quantum Coherence**.

## 1. Conway's Game of Life (`conway_GoL/`)

This subdirectory contains various implementations of Conway's Game of Life, ranging from amplitude encoding to reversible logic.

### Implementations

#### 1. Amplitude Conway (`conway_GoL/amplitude_conway.py`)
*   **Concept:** **Quantum Walk**.
*   **Encoding:** **Amplitude Encoding**. The grid is encoded in the probability amplitudes of a superposition state.
*   **Resources:** **$O(\log N)$ Qubits**. (Exponential Reduction).
*   **Physics:** Linear Wave Propagation.
*   **Verdict:** Excellent for demonstrating quantum advantage in space complexity, but does not replicate the non-linear "Soliton" behavior of classical Life (Gliders diffuse into waves).

#### 2. Reversible Conway (`conway_GoL/quantum_conway.py` / `conway_GoL/compare_conway.py`)
*   **Concept:** **Ping-Pong Buffer**.
*   **Encoding:** **Direct Encoding** ($|0\rangle$ or $|1\rangle$ per cell).
*   **Resources:** **$2N$ Qubits** (Current Grid + Next Grid).
*   **Physics:** Exact Classical Match (via Reset).
*   **Verdict:** The standard way to embed classical logic. Uses `reset` operations, breaking global quantum coherence to reuse qubits. Effectively a classical simulation on quantum hardware.

#### 3. Hybrid Kernel Conway (`conway_GoL/kernel_conway.py`)
*   **Concept:** **Quantum GPU (Batch Processing)**.
*   **Encoding:** **Local Encoding**.
*   **Resources:** **$O(1)$ Qubits** (Fixed 14-qubit Kernel).
*   **Physics:** Exact Classical Match.
*   **Verdict:** The most scalable approach for current NISQ devices. It processes the grid cell-by-cell (or in batches) using a small, fixed quantum circuit. Trades time (sequential execution) for space (constant qubits).

#### 4. Super Conway (`conway_GoL/super_conway.py`)
*   **Concept:** **Unitary Time Travel**.
*   **Encoding:** **Direct Encoding**.
*   **Resources:** **$N$ Qubits**.
*   **Physics:** **Strictly Reversible CA** (Parity Life).
*   **Verdict:** A theoretical model that implements a naturally reversible variant of Life. Allows for "Hyper-Kernel" operations where $U^{1000}$ (1000 steps) is applied as a single gate, enabling instant "Time Jumps".

#### 5. Sliding Window Conway (`conway_GoL/sliding_window_conway.py`)
*   **Concept:** **Memory Optimization**.
*   **Encoding:** **Row-wise Encoding**.
*   **Resources:** **$O(W)$ Qubits** (Linear in Width, Constant in Height).
*   **Physics:** Exact Classical Match (within the window).
*   **Verdict:** The most qubit-efficient way to simulate coherent quantum life on large grids. It processes the grid as a rolling window of 3 rows.

> [!IMPORTANT]
> **Note on Grid Size & Padding:**
> When simulating small patterns (like a 3x3 Blinker) on small toroidal grids (like 3x3), the pattern will wrap around and interact with itself, causing artifacts (e.g., the whole grid filling up).
> **Solution:** Always use a grid size at least 2 cells larger than the pattern (e.g., 5x5 for a 3x3 Blinker) to provide "padding" and prevent self-interaction.

### QLTO: Quantum Life & Topology Optimization (`conway_GoL/qlto_conway.py`)

**Goal:** To build a **Quantum Cellular Automaton** that runs autonomously on a quantum processor without classical intervention.

**Solution: "Ping-Pong" Architecture**
We successfully stabilized the automaton using a **Dissipative Quantum Circuit**.
*   **Two-Buffer System:** Registers $A$ and $B$ of $N$ qubits.
*   **Logic Gates:** Explicit **Toffoli (CCX)** gates.
*   **Qubit Reuse:** Compute $A \to B$, Measure $B$, Reset $A$, Compute $B \to A$.

This allows infinite simulation time using constant ($2N$) qubits.

## 2. Quantum Busy Beavers (`busy_beavers/`)

This subdirectory explores the simulation and search for Busy Beaver Turing Machines using quantum mechanics.

### Files
*   `busy_beavers/universal_bb_search.py`: A quantum search algorithm that superposes all possible transition rules to find Halting Machines.
*   `busy_beavers/quantum_spectroscopy_bb.py`: Uses Quantum Phase Estimation (Spectroscopy) to detect infinite loops (non-halting behavior) by analyzing the energy spectrum of the evolution operator.
*   `busy_beavers/quantum_busy_beavers.py`: Core implementation of the Quantum Turing Machine landscape.

### Theoretical Insights

#### 1. Qubit Scale: Logarithmic Compression
Your architecture achieves an **exponential compression** of the state space.
*   **Classical State Space:** $N \times L \times 2^L$.
*   **Quantum Landscape Qubits:** $L + \log_2 L + \log_2 N$.
*   **Scaling Law:** Linear with Tape Length ($L$), Logarithmic with States ($N$).

#### 2. Circuit Depth: The "Time" Bottleneck
*   **Simulation:** To simulate $T$ steps, you apply $U$ exactly $T$ times.
*   **Depth:** $O(T \times \text{GateComplexity}(U))$.
*   **The Busy Beaver Problem:** Since $T$ grows uncomputably fast, the circuit depth explodes.

#### 3. The "Quantum Shortcut" (Spectroscopy)
Instead of running $T$ steps linearly, you create a superposition of time. If the machine enters a loop early, the **Spectral Signature** reveals it instantly.
*   **Depth for Spectroscopy:** Scales with $\log_2 T$.
*   **Implication:** Potential to detect "Non-Halting" behavior (loops) using polylogarithmic circuit depth.

## 3. Quantum Fractals (`quantum_fractal/`)

### Fractal Landscape (`quantum_fractal/fractal_landscape.py`)
Generates fractal landscapes using quantum interference patterns. This explores the visual and topological properties of quantum states as they evolve or are measured.

## Usage

To compare the Hybrid Kernel approach (recommended) with Classical Conway:

```bash
cd conway_GoL
python compare_conway.py
```

To run the Universal Busy Beaver Search:

```bash
cd busy_beavers
python universal_bb_search.py
```
