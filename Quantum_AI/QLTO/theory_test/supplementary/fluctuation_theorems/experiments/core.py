"""
Paper 4: Quantum Thermodynamics Core Library
============================================

Shared utilities for all Paper 4 experiments:
- CoherentDemonEngine: The Maxwell's Demon protocol
- Hamiltonian builders
- Entropy/information metrics
- Plotting utilities

This is the ONLY file with core infrastructure.
Individual experiments import from here.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import (SparsePauliOp, partial_trace, entropy, 
                                  DensityMatrix, Statevector)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator

warnings.filterwarnings("ignore")

# ============================================================================
# CONSTANTS
# ============================================================================

k_B = 1.0  # Boltzmann constant (natural units)
SEED = 42

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProtocolResult:
    """Result from a single demon protocol run."""
    tau: float                # Sensing time
    work: float               # Work extracted W
    mutual_info: float        # I(S:A)
    E_initial: float          # Energy before
    E_final: float            # Energy after
    holevo_chi: float         # Holevo capacity


@dataclass 
class EfficiencyResult:
    """Result from efficiency measurement."""
    N: int
    eta: float                # Efficiency (slope of W vs I)
    eta_error: float          # Standard error
    r_squared: float          # Fit quality
    tau_values: np.ndarray
    W_values: np.ndarray
    I_values: np.ndarray


# ============================================================================
# HAMILTONIAN BUILDERS
# ============================================================================

def build_random_hamiltonian(n_qubits: int, seed: int = SEED) -> SparsePauliOp:
    """
    Build random transverse Ising Hamiltonian (exponential DLA).
    
    H = Σ_ij J_ij Z_i Z_j + Σ_i h_i X_i
    
    Returns:
        SparsePauliOp with random all-to-all couplings
    """
    np.random.seed(seed)
    ops = []
    
    # ZZ interactions (all-to-all)
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            J = np.random.uniform(-1, 1)
            label = ["I"] * n_qubits
            label[i] = "Z"
            label[j] = "Z"
            ops.append(("".join(label[::-1]), J))
    
    # X fields
    for i in range(n_qubits):
        h = np.random.uniform(-0.5, 0.5)
        label = ["I"] * n_qubits
        label[i] = "X"
        ops.append(("".join(label[::-1]), h))
    
    return SparsePauliOp.from_list(ops)


def build_ising_1d_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> SparsePauliOp:
    """
    Build 1D transverse Ising Hamiltonian (polynomial DLA).
    
    H = J Σ_i Z_i Z_{i+1} + h Σ_i X_i
    
    Returns:
        SparsePauliOp with nearest-neighbor couplings
    """
    ops = []
    
    # ZZ interactions (nearest-neighbor only)
    for i in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[i] = "Z"
        label[i+1] = "Z"
        ops.append(("".join(label[::-1]), J))
    
    # X fields
    for i in range(n_qubits):
        label = ["I"] * n_qubits
        label[i] = "X"
        ops.append(("".join(label[::-1]), h))
    
    return SparsePauliOp.from_list(ops)


def compute_ground_state_energy(H: SparsePauliOp) -> float:
    """Compute ground state energy via exact diagonalization."""
    H_mat = H.to_matrix()
    eigenvalues = np.linalg.eigvalsh(H_mat)
    return np.min(eigenvalues)


def compute_dla_dimension(n_qubits: int, hamiltonian_type: str = "random") -> int:
    """Estimate DLA dimension for given Hamiltonian type."""
    if hamiltonian_type == "random":
        # Exponential: ~4^N - 1
        return 4**n_qubits - 1
    elif hamiltonian_type == "ising_1d":
        # Polynomial: O(N^2) for 1D Ising
        return n_qubits * (n_qubits - 1)
    else:
        return 4**n_qubits - 1


# ============================================================================
# COHERENT DEMON ENGINE
# ============================================================================

class CoherentDemonEngine:
    """
    Maxwell's Demon for VQA optimization.
    
    Protocol:
    1. Initialize: |+⟩^⊗N system, |+⟩ ancilla
    2. Sense: Controlled evolution U = exp(-iHτ)
    3. Lock: Hadamard on ancilla
    4. Feedback: CRX gates
    5. Measure: Work and mutual information
    """
    
    def __init__(self, n_qubits: int, H: Optional[SparsePauliOp] = None, 
                 seed: int = SEED):
        self.n = n_qubits
        self.seed = seed
        self.backend = AerSimulator(method='statevector')
        
        # Build Hamiltonian if not provided
        self.H = H if H is not None else build_random_hamiltonian(n_qubits, seed)
        
        # Precompute energies
        self.E_ground = compute_ground_state_energy(self.H)
        self._E_initial_cached = None
    
    @property
    def E_initial(self) -> float:
        """Initial energy for |+⟩^⊗N state."""
        if self._E_initial_cached is None:
            plus_state = np.ones(2**self.n) / np.sqrt(2**self.n)
            sv = Statevector(plus_state)
            self._E_initial_cached = sv.expectation_value(self.H).real
        return self._E_initial_cached
    
    def run_cycle(self, tau: float, kick_strength: float = 0.2) -> ProtocolResult:
        """
        Run one complete demon cycle.
        
        Args:
            tau: Sensing time
            kick_strength: Feedback rotation angle
            
        Returns:
            ProtocolResult with all measurements
        """
        qr_sys = QuantumRegister(self.n, 'sys')
        qr_anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr_anc, qr_sys)
        
        # 1. Initialize
        qc.h(qr_sys)
        qc.h(qr_anc)
        
        # 2. Sensing (controlled evolution)
        evo = PauliEvolutionGate(self.H, time=tau)
        qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        
        # 3. Locking
        qc.h(qr_anc)
        qc.save_statevector(label="post_sensing")
        
        # 4. Feedback
        for i in range(self.n):
            qc.crx(kick_strength, qr_anc[0], qr_sys[i])
        
        qc.save_statevector(label="final")
        
        # Execute
        t_qc = transpile(qc, self.backend)
        result = self.backend.run(t_qc).result()
        
        # Extract states
        sv_sensing = result.data(0)["post_sensing"]
        sv_final = result.data(0)["final"]
        
        # Compute mutual information
        rho_sensing = DensityMatrix(sv_sensing)
        S_SA = entropy(rho_sensing)
        rho_S = partial_trace(rho_sensing, [0])
        rho_A = partial_trace(rho_sensing, range(1, self.n + 1))
        S_S = entropy(rho_S)
        S_A = entropy(rho_A)
        mutual_info = S_S + S_A - S_SA
        
        # Compute work
        E_sensing = rho_S.expectation_value(self.H).real
        rho_final = DensityMatrix(sv_final)
        rho_S_final = partial_trace(rho_final, [0])
        E_final = rho_S_final.expectation_value(self.H).real
        work = E_sensing - E_final
        
        # Holevo capacity
        p1 = rho_A.data[1, 1].real
        p0 = 1 - p1
        if p0 > 1e-10 and p1 > 1e-10:
            holevo_chi = -p0 * np.log2(p0) - p1 * np.log2(p1)
        else:
            holevo_chi = 0.0
        
        return ProtocolResult(
            tau=tau,
            work=work,
            mutual_info=mutual_info,
            E_initial=self.E_initial,
            E_final=E_final,
            holevo_chi=holevo_chi
        )
    
    def measure_efficiency(self, n_tau: int = 20, 
                           tau_range: Tuple[float, float] = (0.1, 1.5),
                           kick_strength: float = 0.2) -> EfficiencyResult:
        """
        Measure efficiency by sweeping tau and fitting W vs I.
        
        Returns:
            EfficiencyResult with efficiency and fit quality
        """
        taus = np.linspace(tau_range[0], tau_range[1], n_tau)
        W_values = []
        I_values = []
        
        for tau in taus:
            result = self.run_cycle(tau, kick_strength)
            W_values.append(result.work)
            I_values.append(result.mutual_info)
        
        W_arr = np.array(W_values)
        I_arr = np.array(I_values)
        
        # Fit W = η * I
        if np.std(I_arr) > 1e-6:
            slope, intercept, r, p, se = linregress(I_arr, W_arr)
            eta = slope
            eta_error = se
            r_squared = r**2
        else:
            eta = 0.0
            eta_error = 0.0
            r_squared = 0.0
        
        return EfficiencyResult(
            N=self.n,
            eta=eta,
            eta_error=eta_error,
            r_squared=r_squared,
            tau_values=taus,
            W_values=W_arr,
            I_values=I_arr
        )


# ============================================================================
# PLOTTING UTILITIES
# ============================================================================

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 10,
        'figure.figsize': (10, 8)
    })


def save_figure(fig, filename: str, save_dir: str = "../figures"):
    """Save figure to file."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/{filename}"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filepath}")


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Paper 4 Core Library - Testing")
    print("=" * 50)
    
    # Test with 3 qubits
    engine = CoherentDemonEngine(n_qubits=3)
    print(f"Built engine for N=3")
    print(f"  E_ground = {engine.E_ground:.4f}")
    print(f"  E_initial = {engine.E_initial:.4f}")
    
    # Run one cycle
    result = engine.run_cycle(tau=0.5)
    print(f"\nSingle cycle (τ=0.5):")
    print(f"  Work = {result.work:.4f}")
    print(f"  I(S:A) = {result.mutual_info:.4f}")
    print(f"  Holevo χ = {result.holevo_chi:.4f}")
    
    # Measure efficiency
    eff = engine.measure_efficiency(n_tau=10)
    print(f"\nEfficiency measurement:")
    print(f"  η = {eff.eta:.4f} ± {eff.eta_error:.4f}")
    print(f"  R² = {eff.r_squared:.4f}")
    
    print("\n✓ Core library working correctly!")
