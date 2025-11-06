#!/usr/bin/env python3
"""
Conservative QLTO scale builder test.
Builds QAOA ansatz and Hamiltonian for increasing N and reports build times.
"""
import time
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.solvers import qlto_qaoa_sat as qls
except Exception:
    try:
        from src.solvers.qlto_qaoa_sat import QLTO_QAOA_SAT_Solver as qls
    except Exception as e:
        raise ImportError('Could not import qlto_qaoa_sat. Run from repo root or adjust import path.')

Ns = [8,16,32,64,96,120,160,200,256,400]

def run_scale():
    results = []
    for N in Ns:
        # Create a trivial SATProblem with N variables and no clauses (we only need ansatz/hamiltonian builders)
        try:
            start = time.time()
            dummy_problem = qls.SATProblem(n_vars=N, clauses=[])
            ansatz, n_params = qls.create_qaoa_ansatz(dummy_problem, p_layers=1)
            t_ansatz = time.time() - start
            start_h = time.time()
            ham = qls.sat_to_hamiltonian(dummy_problem)
            t_ham = time.time() - start_h
            results.append((N, n_params, t_ansatz, t_ham))
            print(f"N={N}: params={n_params}, ansatz_build={t_ansatz:.4f}s, ham_build={t_ham:.4f}s")
        except Exception as e:
            print(f"N={N}: error building: {e}")
            results.append((N, None, None, None))
    return results

if __name__ == '__main__':
    if not qls.QISKIT_AVAILABLE:
        print('Warning: Qiskit not available. This test only needs Qiskit for ansatz builder; ensure Qiskit is installed for full run.')
    run_scale()
