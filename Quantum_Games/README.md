# Quantum Games: Approaches to Non-Unitary Cellular Automata

This directory contains research code exploring different methods to simulate **Conway's Game of Life** (a non-linear, irreversible cellular automaton) on **Quantum Computers** (which are strictly linear and reversible).

Each approach represents a different trade-off between **Qubit Efficiency**, **Physical Correctness**, and **Quantum Coherence**.

## 1. Amplitude Conway (`amplitude_conway.py`)
*   **Concept:** **Quantum Walk**.
*   **Encoding:** **Amplitude Encoding**. The grid is encoded in the probability amplitudes of a superposition state.
*   **Resources:** **$O(\log N)$ Qubits**. (Exponential Reduction).
*   **Physics:** Linear Wave Propagation.
*   **Verdict:** Excellent for demonstrating quantum advantage in space complexity, but does not replicate the non-linear "Soliton" behavior of classical Life (Gliders diffuse into waves).

## 2. Reversible Conway (`quantum_conway.py` / `compare_conway.py`)
*   **Concept:** **Ping-Pong Buffer**.
*   **Encoding:** **Direct Encoding** ($|0\rangle$ or $|1\rangle$ per cell).
*   **Resources:** **$2N$ Qubits** (Current Grid + Next Grid).
*   **Physics:** Exact Classical Match (via Reset).
*   **Verdict:** The standard way to embed classical logic. Uses `reset` operations, breaking global quantum coherence to reuse qubits. Effectively a classical simulation on quantum hardware.

## 3. Hybrid Kernel Conway (`kernel_conway.py`)
*   **Concept:** **Quantum GPU (Batch Processing)**.
*   **Encoding:** **Local Encoding**.
*   **Resources:** **$O(1)$ Qubits** (Fixed 14-qubit Kernel).
*   **Physics:** Exact Classical Match.
*   **Verdict:** The most scalable approach for current NISQ devices. It processes the grid cell-by-cell (or in batches) using a small, fixed quantum circuit. Trades time (sequential execution) for space (constant qubits).

## 4. Super Conway (`super_conway.py`)
*   **Concept:** **Unitary Time Travel**.
*   **Encoding:** **Direct Encoding**.
*   **Resources:** **$N$ Qubits**.
*   **Physics:** **Strictly Reversible CA** (Parity Life).
*   **Verdict:** A theoretical model that implements a naturally reversible variant of Life. Allows for "Hyper-Kernel" operations where $U^{1000}$ (1000 steps) is applied as a single gate, enabling instant "Time Jumps".

## 5. Sliding Window Conway (`sliding_window_conway.py`)
*   **Concept:** **Memory Optimization**.
*   **Encoding:** **Row-wise Encoding**.
*   **Resources:** **$O(W)$ Qubits** (Linear in Width, Constant in Height).
*   **Physics:** Exact Classical Match (within the window).
*   **Verdict:** The most qubit-efficient way to simulate coherent quantum life on large grids. It processes the grid as a rolling window of 3 rows.

> [!IMPORTANT]
> **Note on Grid Size & Padding:**
> When simulating small patterns (like a 3x3 Blinker) on small toroidal grids (like 3x3), the pattern will wrap around and interact with itself, causing artifacts (e.g., the whole grid filling up).
> **Solution:** Always use a grid size at least 2 cells larger than the pattern (e.g., 5x5 for a 3x3 Blinker) to provide "padding" and prevent self-interaction.

## Usage

To compare the Hybrid Kernel approach (recommended) with Classical Conway:

```bash
python compare_conway.py
```

To run the Amplitude Encoding demo:

```bash
python amplitude_conway.py
```

### summaries

The direct mathematical answer for the **minimum qubit count required to simulate one full, coherent, and reversible step of the entire $N$-cell grid** is:

$$Q_{\text{min}} = \mathbf{2N}$$

Where $N$ is the number of cells ($W \times H$).

***

### Why $2N$ is the Mathematical Minimum

Conway's Game of Life is an **irreversible** process; a given state $S_{t+1}$ can evolve from multiple previous states $S_t$.

1.  **Unitary Constraint:** Quantum mechanics requires all operations (time evolution) to be **unitary** (i.e., reversible).
2.  **Bennett's Principle:** To simulate a classical irreversible function $f(x)$ with a unitary operation $U$, you must preserve the input state $x$ alongside the output $f(x)$ to ensure reversibility:
    $$\left|x\right\rangle\left|0\right\rangle \xrightarrow{U} \left|x\right\rangle\left|f(x)\right\rangle$$
3.  **The $2N$ Requirement:** Since the entire grid ($N$ cells) is the input $x$ and the entire grid ($N$ cells) is the output $f(x)$, you need $N$ qubits to store the previous state ($S_t$) and $N$ qubits to store the new state ($S_{t+1}$).

Your **Ping-Pong Automaton** (`conway.py`) uses this $2N$ architecture precisely because it is the theoretical minimum to maintain the full, coherent superposition of all possible Life trajectories simultaneously.

***

### Mathematical and Practical Optimizations Beyond $2N$

While $2N$ is the theoretical minimum for a single parallel, coherent step of the entire grid, we can reduce the required qubit count by changing the simulation strategy.

| Optimization Technique | Qubit Count | Mathematical Trade-off |
| :--- | :--- | :--- |
| **Sliding Window** | $\mathbf{\approx 3W + 4}$ | **Trades Space for Time/Depth.** Only simulates a small $W \times 3$ slice of the infinite grid at once, moving the window sequentially. |
| **Reversible CA** | $N$ | **Changes the Game Rules.** You must simulate a reversible counterpart (like Parity Life in `super_conway.py`), which has different dynamics than standard GoL. |
| **Mid-Circuit Reset** | $N + \text{Ancilla}$ | **Sacrifices Coherence.** The quantum state must be measured and collapsed (decohered) after computation, and the qubits reset and reused. |

The **Sliding Window** approach is the best way to maintain the full quantum coherence of the simulation while dramatically reducing the total qubit requirement to a small, fixed amount (dependent only on the width $W$ and not the height $H$). For example, a $100 \times 100$ grid requires 20,000 qubits for $2N$, but only $\approx 304$ qubits for the Sliding Window.


# QLTO: Quantum Life & Topology Optimization

## The Goal
To build a **Quantum Cellular Automaton (Game of Life)** that runs autonomously on a quantum processor without classical intervention.

## The Solution: "Ping-Pong" Architecture
We successfully stabilized the automaton using a **Dissipative Quantum Circuit**.

* **Two-Buffer System:** We use two registers ($A$ and $B$) of $N$ qubits.
* **Logic Gates:** Instead of optimizing energy, we use explicit **Toffoli (CCX)** gates to implement the laws of physics.
* **Qubit Reuse:** 1.  $A$ controls $B$.
    2.  Measure $B$.
    3.  **Reset $A$** (Removing Entropy).
    4.  $B$ controls $A$.
    5.  Repeat.

```
[ FRAME 0 ]       [ FRAME 1 ]       [ FRAME 2 ]       [ FRAME 3 ]
          (Init)          (Logic)           (Logic)           (Logic)

Q_A:  ──|State|─[M]─|0>───●───●───●────────[M]─|0>───●───●───●──────── ...
                 │        │   │   │         │        │   │   │
Q_B:  ──|0000|───┼────────X───X───X─[M]─|0>─┼────────X───X───X─[M]─|0> ...
                 │                   │      │                   │
      ┌──────────┴─┐      ┌──────────┼──────┴─┐      ┌──────────┼──────┐
C_0:  │  0-3 (Save)│      │          │        │      │          │      │
      └────────────┘      │          │        │      │          │      │
C_1:                      │  4-7 (Save)       │      │          │      │
                          └──────────┴────────┘      │          │      │
C_2:                                                 │  8-11 (Save)    │
                                                     └──────────┴──────┘
```

This allows infinite simulation time using constant ($2N$) qubits.

## Research & Development Log

We attempted 4 major architectures before solving stability:

### 1. The "Analogue" Failure (Synthesized Hamiltonian)
* **Idea:** Encode the rules into an energy landscape ($H$).
* **Result:** **Drift.** The landscape was too flat ("Golf Course"). The system couldn't find the exact solution and drifted into random noise.

### 2. The "Probabilistic" Failure (QAOA/Grover)
* **Idea:** Use Grover's Algorithm to search for the valid next state.
* **Result:** **Instability.** Quantum Shot Noise accumulated. If Frame 1 had a 1% error, Frame 2 had a 10% error. The universe collapsed.

### 3. The "Resonance" Failure (Spectroscopy)
* **Idea:** Use microwave pulses tuned to "Neighbor Count" frequencies.
* **Result:** **Interference.** Pulses for $N=3$ and $N=4$ interfered with each other, creating "ghost" states.

### 4. The "Holographic" Success (Deterministic Logic)
* **Idea:** Use Coherent Logic Gates to force the state transition.
* **Result:** **Success.** By using `RESET` operations, we physically removed the errors (entropy) from the system at every step, creating a perfectly stable "Digital" Quantum Automaton.

## How to Run
1.  Install Qiskit: `pip install qiskit qiskit-aer`
2.  Run the engine: `python qlto_final_engine.py`


The difference between the **2-Buffer "Ping-Pong"** (A $\leftrightarrow$ B) and the **Multi-Buffer "Ring"** (A $\to$ B $\to$ C $\to$ D...) architectures is fundamentally about **Memory Depth**.

While both allow for infinite simulation time, they support different *kinds* of physics.

### 1. The Physics of Memory (Markovian vs. Non-Markovian)

* **2-Buffer (Ping-Pong):**
    * **Capacity:** Holds only **Current ($T$)** and **Next ($T+1$)**.
    * **Constraint:** As soon as you calculate $T+1$, you *must* wipe $T$ to make room for $T+2$.
    * **Physics:** This restricts you to **Markovian Dynamics**—universes where the future depends *only* on the present moment (like standard Conway's Game of Life). You cannot have "momentum" because the system cannot remember where it came from.

* **Multi-Buffer (Ring, e.g., 4 Buffers):**
    * **Capacity:** Holds **$T$, $T-1$, $T-2$, and $T-3$** simultaneously.
    * **Capability:** When you are calculating Frame $D$, Frames $A$, $B$, and $C$ are still alive in quantum memory.
    * **Physics:** This allows **Non-Markovian Dynamics** (History-Dependent Physics). You can implement rules like:
        * *"A cell dies if it has been alive for 3 consecutive turns"* (Requires checking $T, T-1, T-2$).
        * *"Objects have velocity"* (Requires comparing position at $T$ vs $T-1$).

### 2. Pipelining & Logic Density

* **2-Buffer:**
    * You are limited to a single logical step: **Input $\to$ Logic $\to$ Output**.
    * If your calculation is complex, you have to cram it all into one giant circuit block.

* **Multi-Buffer:**
    * You can break a complex calculation into a **Pipeline**.
    * **Step A $\to$ B:** Calculate Neighbors (Summation).
    * **Step B $\to$ C:** Apply Rules (Birth/Death).
    * **Step C $\to$ D:** Apply Noise Filter (Error Correction).
    * **Step D $\to$ A:** Loop back.
    * This makes the quantum circuit shallower and more stable, as you reset entropy at every sub-stage.

### 3. Temporal Error Correction

* **2-Buffer:** If a bit-flip error happens at Frame $T$, it is immediately baked into Frame $T+1$ and becomes permanent reality.
* **Multi-Buffer:** You can implement **Temporal Majority Voting**.
    * Calculate $A \to B$.
    * Calculate $A \to C$.
    * Calculate $A \to D$.
    * Before resetting $A$, compare $B, C, D$. If they disagree, use a quantum majority vote to fix the error. This creates a stable "Time Crystal" that resists noise.

### Summary Table

| Feature | 2-Buffer (Ping-Pong) | Multi-Buffer (Ring) |
| :--- | :--- | :--- |
| **Qubit Cost** | $2N$ (Minimal) | $k \cdot N$ (Higher) |
| **Physics Type** | Memoryless (Standard CA) | History-Dependent (Momentum, Aging) |
| **Operations** | Compute $\to$ Reset | Compute $\to$ Store $\to$ Compare $\to$ Reset |
| **Best For** | Speed, Simple Automata | Complex Physics, Error Correction |

In short: **2 Buffers are sufficient for *survival* (Conway), but Multi-Buffers are required for *intelligence* (Memory).**