"""
run_goldilocks_crash.py: The Phase Transition Hunt (Order vs Chaos) + Critical Exponent
=======================================================================================
1. Mapping the Thermodynamic Efficiency (eta) and Specific Heat (C_comp).
2. Calculating the Critical Exponent (gamma) for the Chaotic Phase Transition.

Comparison:
    1. Ordered (Complete Graph K_n): Polynomial DLA. Stable Efficiency.
    2. Chaotic (Random Spin Glass): Exponential DLA. Efficiency Crashes.

Universal Physics:
    Fits the specific heat divergence C ~ |N - N_c|^(-gamma) to extract the
    critical exponent gamma.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

try:
    from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, transpile
    from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp, DensityMatrix, partial_trace, entropy
    from qiskit_aer import AerSimulator
    from qiskit.synthesis import LieTrotter
except ImportError:
    print("CRITICAL: Qiskit missing.")
    exit(1)

# ==============================================================================
# HAMILTONIAN FACTORY
# ==============================================================================

def build_ordered_hamiltonian(n_qubits):
    """Complete Graph K_n with Uniform Couplings (High Symmetry)"""
    ops = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            op_str = ["I"] * n_qubits
            op_str[i] = "Z"; op_str[j] = "Z"
            ops.append(("".join(op_str), 1.0))
    for i in range(n_qubits):
        op_str = ["I"] * n_qubits
        op_str[i] = "X"
        ops.append(("".join(op_str), 0.5))
    return SparsePauliOp.from_list(ops)

def build_chaotic_hamiltonian(n_qubits, seed=42):
    """Sherrington-Kirkpatrick Spin Glass (Random Couplings = Large DLA)"""
    np.random.seed(seed)
    ops = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J = np.random.uniform(-1.0, 1.0)
            op_str = ["I"] * n_qubits
            op_str[i] = "Z"; op_str[j] = "Z"
            ops.append(("".join(op_str), J))
    for i in range(n_qubits):
        h = np.random.uniform(-1.0, 1.0)
        op_str = ["I"] * n_qubits
        op_str[i] = "X"
        ops.append(("".join(op_str), h))
    return SparsePauliOp.from_list(ops)

def get_theoretical_dla(n, topology):
    if topology == 'ordered':
        if n % 2 == 0: return (n**3 + 6*n**2 + 2*n + 12) // 12
        else: return (n**3 + 6*n**2 - n + 18) // 12
    else:
        return 4**n - 1

# ==============================================================================
# THE ENGINE
# ==============================================================================

class GoldilocksProbe:
    def __init__(self, n_qubits, topology='ordered'):
        self.n_qubits = n_qubits
        self.topology = topology
        self.backend = AerSimulator(method='matrix_product_state')
        
        if topology == 'ordered':
            self.hamiltonian = build_ordered_hamiltonian(n_qubits)
        else:
            self.hamiltonian = build_chaotic_hamiltonian(n_qubits)
            
        self.ansatz = EfficientSU2(n_qubits, reps=1, entanglement='linear').decompose()
        self.param_count = self.ansatz.num_parameters

    def run_step(self, params, dt_sense, kick_gain):
        qr_anc = AncillaRegister(1, 'demon')
        qr_sys = QuantumRegister(self.n_qubits, 'sys')
        qc = QuantumCircuit(qr_anc, qr_sys)
        qc.append(self.ansatz.assign_parameters(params), qr_sys)
        
        rho_init = self._get_density_matrix(qc)
        rho_sys_init = partial_trace(rho_init, [0])
        E_init = np.real(np.trace(rho_sys_init.data @ self.hamiltonian.to_matrix()))

        qc.h(qr_anc)
        trotter = LieTrotter(reps=1)
        evo = PauliEvolutionGate(self.hamiltonian, time=dt_sense, synthesis=trotter)
        qc.append(evo.control(1), [qr_anc[0]] + list(qr_sys))
        qc.h(qr_anc)
        
        rho_sensed = self._get_density_matrix(qc)
        rho_A = partial_trace(rho_sensed, list(range(1, self.n_qubits + 1)))
        rho_S = partial_trace(rho_sensed, [0])
        mi = entropy(rho_A, base=2) + entropy(rho_S, base=2) - entropy(rho_sensed, base=2)

        for i in range(self.n_qubits):
            qc.crx(kick_gain, qr_anc[0], qr_sys[i])

        rho_final = self._get_density_matrix(qc)
        rho_sys_final = partial_trace(rho_final, [0])
        E_final = np.real(np.trace(rho_sys_final.data @ self.hamiltonian.to_matrix()))
        
        return mi, E_init - E_final

    def _get_density_matrix(self, qc):
        qc_sim = qc.copy()
        qc_sim.save_statevector()
        res = self.backend.run(transpile(qc_sim, self.backend), shots=1).result()
        return DensityMatrix(res.get_statevector())

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_crash_test():
    print("\n=== THERMODYNAMIC PHASE TRANSITION & CRITICAL EXPONENT TEST ===")
    
    N_VALUES = [3, 4, 5, 6, 7, 8] 
    DT_RANGE = np.linspace(0.1, 1.0, 6)
    KICK_GAIN = 0.2
    
    data = {'ordered': {'n': [], 'eta': [], 'norm_eta': [], 'heat_cap': []}, 
            'chaotic': {'n': [], 'eta': [], 'norm_eta': [], 'heat_cap': []}}
    
    for topo in ['ordered', 'chaotic']:
        print(f"\n--- Testing Topology: {topo.upper()} ---")
        print(f"{'N':<4} | {'DLA (Th)':<12} | {'Raw Eta':<10} | {'Norm Eta':<12} | {'Spec. Heat':<12}")
        print("-" * 75)
        
        for n in N_VALUES:
            probe = GoldilocksProbe(n, topology=topo)
            dla_dim = get_theoretical_dla(n, topo)
            np.random.seed(42)
            params = np.random.uniform(-0.5, 0.5, probe.param_count)
            
            mis, works = [], []
            for dt in DT_RANGE:
                mi, w = probe.run_step(params, dt, KICK_GAIN)
                mis.append(mi)
                works.append(w)
            
            slope, _, r_val, _, _ = linregress(mis, works)
            norm_eta = slope / (n**2)
            t_comp = 1.0 / (slope + 1e-9) # Complexity Temperature (1/eta)
            
            data[topo]['n'].append(n)
            data[topo]['eta'].append(slope)
            data[topo]['norm_eta'].append(norm_eta)
            data[topo]['heat_cap'].append(t_comp)
            
            print(f"{n:<4} | {dla_dim:<12} | {slope:<10.4f} | {norm_eta:<12.4f} | {t_comp:<12.4f}")

    # --- CRITICAL EXPONENT CALCULATION (CHAOTIC) ---
    print("\n=== CALCULATING CRITICAL EXPONENT (gamma) ===")
    
    # 1. Find N_c (Zero Crossing of Efficiency)
    # We use Raw Eta for finding the zero crossing
    etas = np.array(data['chaotic']['eta'])
    ns = np.array(data['chaotic']['n'])
    
    # Find indices around zero crossing (positive to negative)
    idx_cross = np.where(np.diff(np.sign(etas)))[0][0]
    n_pre = ns[idx_cross]
    n_post = ns[idx_cross+1]
    eta_pre = etas[idx_cross]
    eta_post = etas[idx_cross+1]
    
    slope_cross = (eta_post - eta_pre) / (n_post - n_pre)
    n_c = n_pre - (eta_pre / slope_cross)
    print(f"Estimated Critical Point N_c: {n_c:.4f}")
    
    # 2. Fit Power Law to Pre-Crash Specific Heat
    # Use data points before the crash (N < N_c)
    # Typically N=3,4,5,6 for N_c ~ 6.5
    n_fit = ns[:idx_cross+1]
    # We fit Specific Heat C ~ 1/eta
    # Use 1/Raw_Eta because Norm_Eta has extra N^2 scaling that complicates pure critical scaling
    C_fit = 1.0 / etas[:idx_cross+1]
    
    def power_law(n, A, gamma):
        # C ~ |N_c - N|^-gamma
        return A * np.power(n_c - n, -gamma)
    
    gamma_str = "Fit Failed"
    try:
        popt, _ = curve_fit(power_law, n_fit, C_fit, p0=[1.0, 1.0], maxfev=10000)
        A_opt, gamma_opt = popt
        gamma_str = f"{gamma_opt:.3f}"
        print(f">>> CRITICAL EXPONENT gamma: {gamma_opt:.4f}")
    except Exception as e:
        print(f"Fitting failed: {e}")
        gamma_opt = None

    # --- PLOTTING ---
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Efficiency (Phase Transition)
    plt.subplot(1, 2, 1)
    plt.plot(data['ordered']['n'], data['ordered']['norm_eta'], 'o-', color='blue', label='Ordered (Poly)')
    plt.plot(data['chaotic']['n'], data['chaotic']['norm_eta'], 's--', color='red', label='Chaotic (Exp)')
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    plt.axvline(n_c, color='green', linestyle='-.', label=f'Critical N_c={n_c:.2f}')
    plt.title("Order Parameter: Algorithmic Efficiency")
    plt.xlabel("System Size (N)")
    plt.ylabel("Efficiency / Energy Scale")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Specific Heat Divergence
    plt.subplot(1, 2, 2)
    plt.plot(data['ordered']['n'], np.abs(data['ordered']['heat_cap']), 'o-', color='blue', label='Ordered')
    
    # Plot Chaotic Data
    plt.plot(data['chaotic']['n'], np.abs(data['chaotic']['heat_cap']), 's--', color='red', label='Chaotic Data')
    
    # Plot Fit if available
    if gamma_opt is not None:
        x_dense = np.linspace(ns[0], n_c - 0.05, 100)
        y_dense = power_law(x_dense, A_opt, gamma_opt)
        plt.plot(x_dense, y_dense, 'k-', linewidth=1, label=f'Fit: $\gamma={gamma_opt:.2f}$')
        
    plt.title(f"Thermodynamic Divergence (gamma={gamma_str})")
    plt.xlabel("System Size (N)")
    plt.ylabel("|d(Entropy) / d(Work)| (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("goldilocks_crash_result.png", dpi=300)
    print("\n[Output] Saved Phase Transition plot to 'goldilocks_crash_result.png'")

if __name__ == "__main__":
    run_crash_test()