"""Supplement Script v58_qng_geometry.py: Testing Template 4 (Quantum Natural Gradient)

Tests Template 4: Fubini-Study Riemannian Metric Geometry vs Euclidean Gradient Descent
on ill-conditioned / narrow-valley VQE landscapes.

Fubini-Study Metric Tensor:
F_{ij}(theta) = Re[ <\partial_i \psi | \partial_j \psi> - <\partial_i \psi | \psi><\psi | \partial_j \psi> ]

Updates compared (step-norm matched):
- Arm A (Euclidean Gradient / QLTO v5): theta_{t+1} = theta_t - eta * grad_E
- Arm B (Quantum Natural Gradient / QNG):  theta_{t+1} = theta_t - eta * ||grad_E|| * (g_QNG / ||g_QNG||)
  where g_QNG = pinv(F, rcond=1e-2) * grad_E

Problems: Heisenberg N=4, Frustrated Spin Glass N=4, Frustrated Spin Glass N=6
Metrics:
1. Energy convergence E(theta) - E0.
2. Trajectory stability.
"""
import sys, os, contextlib, io, time
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
os.chdir(APP)
import benchmark as B
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector, SparsePauliOp


def compute_state_and_derivatives(ansatz, params, eps=1e-4):
    M = ansatz.num_parameters
    psi_center = Statevector(ansatz.assign_parameters(params)).data
    
    d_psi = []
    for i in range(M):
        p_plus = params.copy(); p_plus[i] += eps
        p_minus = params.copy(); p_minus[i] -= eps
        
        psi_plus = Statevector(ansatz.assign_parameters(p_plus)).data
        psi_minus = Statevector(ansatz.assign_parameters(p_minus)).data
        
        d_i = (psi_plus - psi_minus) / (2.0 * eps)
        d_psi.append(d_i)
        
    return psi_center, d_psi


def compute_fubini_study_qfim(psi_center, d_psi):
    M = len(d_psi)
    F = np.zeros((M, M))
    inner_prod = np.array([np.vdot(psi_center, d_psi[i]) for i in range(M)])
    
    for i in range(M):
        for j in range(i, M):
            term1 = np.vdot(d_psi[i], d_psi[j])
            term2 = np.conj(inner_prod[i]) * inner_prod[j]
            f_ij = np.real(term1 - term2)
            F[i, j] = f_ij
            F[j, i] = f_ij
            
    return F


def compute_energy_and_gradient(ansatz, H_mat, params, eps=1e-4):
    M = ansatz.num_parameters
    psi = Statevector(ansatz.assign_parameters(params)).data
    E_curr = np.real(np.vdot(psi, H_mat @ psi))
    
    grad = np.zeros(M)
    for i in range(M):
        p_plus = params.copy(); p_plus[i] += eps
        p_minus = params.copy(); p_minus[i] -= eps
        
        psi_plus = Statevector(ansatz.assign_parameters(p_plus)).data
        psi_minus = Statevector(ansatz.assign_parameters(p_minus)).data
        
        E_plus = np.real(np.vdot(psi_plus, H_mat @ psi_plus))
        E_minus = np.real(np.vdot(psi_minus, H_mat @ psi_minus))
        grad[i] = (E_plus - E_minus) / (2.0 * eps)
        
    return E_curr, grad


def run_qng_comparison(prob_fn, prob_name, max_epochs=40, seed=42):
    ansatz, H, name = prob_fn()
    H_mat = H.to_matrix()
    N_qubits = H.num_qubits
    M = ansatz.num_parameters
    
    eigvals = np.linalg.eigvalsh(H_mat)
    E0 = eigvals[0]
    
    p_init = np.random.RandomState(seed).uniform(-np.pi, np.pi, M)
    
    print(f"\n=========================================================================")
    print(f"  {prob_name} | Exact E0 = {E0:.4f} | N={N_qubits} qubits, M={M} params")
    print(f"=========================================================================")
    print(f"  {'epoch':>6}{'E_Euclidean':>14}{'Err_Euc':>10}{'E_QNG':>14}{'Err_QNG':>10}{'QNG Advantage':>16}")
    print("  " + "-" * 72)
    
    p_euc = p_init.copy()
    p_qng = p_init.copy()
    
    eta = 0.15
    
    for ep in range(1, max_epochs + 1):
        # 1. Euclidean step
        E_euc, g_euc = compute_energy_and_gradient(ansatz, H_mat, p_euc)
        p_euc = p_euc - eta * g_euc
        err_euc = E_euc - E0
        
        # 2. QNG step (direction from F_pinv @ g_qng, step-norm matched to Euclidean)
        E_qng, g_qng = compute_energy_and_gradient(ansatz, H_mat, p_qng)
        psi_center, d_psi = compute_state_and_derivatives(ansatz, p_qng)
        F = compute_fubini_study_qfim(psi_center, d_psi)
        
        # Moore-Penrose pseudo-inverse to filter null-space
        F_pinv = np.linalg.pinv(F, rcond=1e-2)
        dir_qng = F_pinv @ g_qng
        
        # Norm-matched update direction
        norm_g = np.linalg.norm(g_qng)
        norm_dir = np.linalg.norm(dir_qng)
        if norm_dir > 1e-9:
            step_qng = dir_qng * (norm_g / norm_dir)
        else:
            step_qng = g_qng
            
        p_qng = p_qng - eta * step_qng
        err_qng = E_qng - E0
        
        adv = f"{err_euc / max(err_qng, 1e-6):.2f}x faster" if err_qng < err_euc else "Euclidean faster"
        
        if ep % 4 == 0 or ep == 1 or ep == max_epochs:
            print(f"  {ep:>6}{E_euc:>14.4f}{err_euc:>10.4f}{E_qng:>14.4f}{err_qng:>10.4f}{adv:>16}")
            
    return E0, err_euc, err_qng


if __name__ == "__main__":
    print("=========================================================================")
    print("TEMPLATE 4 TEST: QUANTUM NATURAL GRADIENT (FUBINI-STUDY METRIC)")
    print("=========================================================================")
    
    probs = [
        ("Heisenberg N=4", lambda: B.get_heisenberg_problem(4)),
        ("Frustrated Spin Glass N=4", lambda: B.generate_frustrated_hamiltonian(4, seed=101)),
        ("Frustrated Spin Glass N=6", lambda: B.generate_frustrated_hamiltonian(6, seed=102)),
    ]
    
    for name, p_fn in probs:
        run_qng_comparison(p_fn, name, max_epochs=32, seed=42)
