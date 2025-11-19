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