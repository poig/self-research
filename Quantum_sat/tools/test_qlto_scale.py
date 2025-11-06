import time
import traceback
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.solvers.qlto_qaoa_sat import SATProblem, SATClause, create_qaoa_ansatz, sat_to_hamiltonian

# Values of N to test (number of qubits/variables)
Ns = [8, 16, 32, 64, 96, 120, 160, 200, 256, 400]

results = []
for N in Ns:
    print('\n' + '='*60)
    print(f'Testing N={N}')
    try:
        # Build a trivial SATProblem with 1 clause to keep clause-count small
        clause = SATClause(tuple([1])) if N >= 1 else SATClause(tuple())
        problem = SATProblem(n_vars=N, clauses=[clause])

        # Build ansatz
        t0 = time.time()
        ansatz, n_params = create_qaoa_ansatz(problem, p_layers=1)
        t1 = time.time()
        ansatz_time = t1 - t0
        qc_info = None
        try:
            qc_info = (ansatz.num_qubits if hasattr(ansatz, 'num_qubits') else len(ansatz.qubits),
                       len(ansatz.data) if hasattr(ansatz, 'data') else 0)
        except Exception:
            qc_info = ('unknown', 'unknown')

        # Build Hamiltonian
        t2 = time.time()
        h = sat_to_hamiltonian(problem)
        t3 = time.time()
        ham_time = t3 - t2

        print(f'  ✅ Ansatx built: params={n_params}, qc_qubits={qc_info[0]}, qc_ops={qc_info[1]}, time={ansatz_time:.3f}s')
        print(f'  ✅ Hamiltonian built: time={ham_time:.3f}s')
        results.append((N, True, ansatz_time, ham_time, None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f'  ❌ FAILED for N={N}: {e}')
        print(tb)
        results.append((N, False, None, None, str(e)))

print('\n' + '='*60)
print('SUMMARY')
for r in results:
    N, ok, at, ht, err = r
    print(f'N={N:3} -> ok={ok}, ansatz_time={at}, ham_time={ht}, err={err}')

print('\nDone')
