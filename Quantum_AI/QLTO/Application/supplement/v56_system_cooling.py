"""Supplement Script v56_system_cooling.py: System-Register Filtered Quantum Cooling (Fixed Filter Quadrature)

Evaluates Alternative 1: Filtered Quantum Cooling (Ding-Chen-Lin / Lin-Lin algorithm)
directly on the physical system register H_sys (no parameter register, no variational loop).

Filter operator: K = sum_{m=1}^M w_m e^{i H t_m} A e^{-i H t_m}
Lowering transition: E_b < E_a  ==> (E_b - E_a) < 0.
Fourier filter f(t) MUST satisfy f_hat(E_b - E_a) > 0 for lowering, 0 for raising.

Problems: Heisenberg N=4, Heisenberg N=6, MaxCut N=4
Metrics evaluated against implementation_plan.md Go/No-Go criteria:
1. Energy error |E - E0| <= 10^-2
2. LCU acceptance rate P_success >= 5%
3. Quadrature terms M <= 8 and Trotter depth
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
import benchmark as B


def get_cooling_filter_quadrature(M, t_max, omega_center, sigma):
    """
    Quadrature for single-sided low-pass filter f_hat(w) = exp(-(w - w0)^2 / 2sigma^2).
    f(t) = (1/2pi) int f_hat(w) e^{-i w t} dw = exp(-i w0 t) * exp(-sigma^2 t^2 / 2)
    For lowering E_b - E_a = -omega_center < 0:
    w0 = -omega_center.
    """
    t_nodes = np.linspace(-t_max, t_max, M)
    dt = (2.0 * t_max) / (M - 1) if M > 1 else 1.0
    # f(t) = exp(+i omega_center t) * exp(- (sigma t)^2 / 2)
    weights = dt * np.exp(1j * omega_center * t_nodes) * np.exp(-0.5 * (sigma * t_nodes) ** 2)
    return weights, t_nodes


def apply_cooling_step_statevector(H_mat, state, A_mat, weights, t_nodes):
    """Applies K = sum_m w_m exp(i H t_m) A exp(-i H t_m) to statevector."""
    eigvals, eigvecs = np.linalg.eigh(H_mat)
    c_eigen = eigvecs.T.conj() @ state
    out_state = np.zeros_like(state, dtype=complex)
    
    for w_m, t_m in zip(weights, t_nodes):
        # exp(-i H t_m) |psi>
        e_minus = eigvecs @ (np.exp(-1j * eigvals * t_m) * c_eigen)
        # A exp(-i H t_m) |psi>
        A_e_minus = A_mat @ e_minus
        # exp(i H t_m) A exp(-i H t_m) |psi>
        c_A_e = eigvecs.T.conj() @ A_e_minus
        e_plus = eigvecs @ (np.exp(1j * eigvals * t_m) * c_A_e)
        out_state += w_m * e_plus
        
    norm_sq = np.vdot(out_state, out_state).real
    # Normalize operator output for probability tracking: K / ||K||_op
    # Raw LCU success probability:
    p_success = norm_sq / (np.sum(np.abs(weights)) ** 2)
    if norm_sq > 1e-15:
        norm_state = out_state / np.sqrt(norm_sq)
    else:
        norm_state = state
    return norm_state, p_success


def run_cooling_experiment(problem_fn, prob_name, M_terms=8, steps=10):
    ansatz, H, name = problem_fn()
    H_mat = H.to_matrix()
    N_qubits = H.num_qubits
    dim = 2 ** N_qubits
    
    # Exact diagonalization
    eigvals, eigvecs = np.linalg.eigh(H_mat)
    E0 = eigvals[0]
    E1 = eigvals[1]
    gap = E1 - E0
    psi0 = eigvecs[:, 0]
    
    # Local perturbation operator A = single qubit X on qubit 0 (breaks symmetry)
    X0 = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    op_list = [I2] * N_qubits
    op_list[0] = X0
    A_mat = op_list[0]
    for k in range(1, N_qubits):
        A_mat = np.kron(A_mat, op_list[k])
        
    # Reference initial state: uniform product state |+>^N
    psi_ref = np.ones(dim, dtype=complex) / np.sqrt(dim)
    E_ref = np.vdot(psi_ref, H_mat @ psi_ref).real
    fid_ref = abs(np.vdot(psi0, psi_ref)) ** 2
    
    # Filter quadrature: target lowering energy shift = gap
    t_max = 3.0 / max(gap, 1e-2)
    sigma = gap / 2.0
    weights, t_nodes = get_cooling_filter_quadrature(M_terms, t_max=t_max, omega_center=gap, sigma=sigma)
    
    print(f"\n=========================================================================")
    print(f"  {prob_name} | Exact E0 = {E0:.4f} | Gap = {gap:.4f} | M={M_terms} terms")
    print(f"  Initial E = {E_ref:.4f} (error = {E_ref - E0:.4f}), Fidelity = {fid_ref:.4f}")
    print(f"=========================================================================")
    print(f"  {'step':>5}{'Energy':>10}{'Error':>10}{'Fidelity':>10}{'P_succ_raw':>12}{'CumP_succ':>12}")
    print("  " + "-" * 62)
    
    curr_state = psi_ref.copy()
    cum_p_succ = 1.0
    go_met = False
    
    for step in range(1, steps + 1):
        curr_state, p_succ = apply_cooling_step_statevector(H_mat, curr_state, A_mat, weights, t_nodes)
        E_curr = np.vdot(curr_state, H_mat @ curr_state).real
        err = E_curr - E0
        fid = abs(np.vdot(psi0, curr_state)) ** 2
        cum_p_succ *= p_succ
        
        print(f"  {step:>5}{E_curr:>10.4f}{err:>10.4f}{fid:>10.4f}{p_succ:>12.4e}{cum_p_succ:>12.4e}")
        if err <= 1e-2 and not go_met:
            print(f"  [GO Met] Target precision 1e-2 reached at step {step}!")
            go_met = True
            
    return E0, err, fid, cum_p_succ


if __name__ == "__main__":
    print("=========================================================================")
    print("SYSTEM-REGISTER FILTERED QUANTUM COOLING (ALTERNATIVE 1 PROTOTYPE)")
    print("=========================================================================")
    
    probs = [
        ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
        ("MaxCut N=4", lambda: B.get_maxcut_problem(4, seed=101)),
        ("Heisenberg N=6", lambda: B.get_heisenberg_problem(6)),
    ]
    
    for name, p_fn in probs:
        for M in [4, 8, 16]:
            run_cooling_experiment(p_fn, name, M_terms=M, steps=8)
