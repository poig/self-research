#!/usr/bin/env python3
"""
Conservative QLTO-FPT benchmark

This is a safer, conservative variant of the `qlto_fpt_solver_v7_hard_benchmark.py`
with defaults that avoid accidental large exponential classical checks or large
quantum simulations.

Defaults:
- p_layers = 1
- bits_per_param = 2
- N_MASK_BITS = 8
- shots = 512
- top_T_candidates = 20
- freq_threshold = 0.2
- classical candidate cap: shrink_superset_greedy disabled by default (use direct try on small k)
- qubit safety limit: only run quantum evolution if n_qubits_p <= 120

Saves CSV to `qlto_fpt_results_conservative.csv`.
"""

import os
import sys
import time
import csv
import random
from typing import List, Tuple, Dict, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.solvers import qlto_qaoa_sat as qls
except Exception:
    try:
        from src.solvers.qlto_qaoa_sat import QLTO_QAOA_SAT_Solver as qls
    except Exception as e:
        raise ImportError("Could not import qlto_qaoa_sat. Run from repo root or adjust import path.")

# Try to import PySAT but tolerate absence (we'll skip classical baseline then)
try:
    from pysat.solvers import Glucose3
    PYSAT_AVAILABLE = True
except Exception:
    PYSAT_AVAILABLE = False

# Conservative configuration
CFG = {
    "p_layers": 1,
    "bits_per_param": 2,
    "N_MASK_BITS": 8,
    "shots": 512,
    "top_T_candidates": 20,
    "freq_threshold": 0.2,
    "trials_per_setting": 3,
    "random_seed": 42,
    "max_simulatable_qubits": 120,
    "max_candidate_size": 10,  # don't try 2^k for k > 10
    "output_csv": "qlto_fpt_results_conservative.csv",
}

# Reuse the SATLIB small problems from the hard benchmark file, but keep a short list
SATLIB_TEST_SUITE = {
    "uf20-01": "uf20-01",  # placeholder, we'll parse from file content if available
}

# If the v7 hard benchmark exists, try to import those contents to reuse problems
hard_bench_path = os.path.join(os.path.dirname(__file__), 'qlto_fpt_solver_v7_hard_benchmark.py')
if os.path.exists(hard_bench_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('v7bench', hard_bench_path)
    v7bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v7bench)
    # copy the test suite if present
    if hasattr(v7bench, 'SATLIB_TEST_SUITE'):
        SATLIB_TEST_SUITE = v7bench.SATLIB_TEST_SUITE

# --- helpers (small, copied/adapted implementations) ---

def parse_dimacs_cnf(file_content: str) -> Tuple[List[Tuple[int, ...]], int]:
    clauses = []
    n_vars = 0
    current_clause = []
    for line in file_content.splitlines():
        line = line.strip()
        if not line or line.startswith('c'):
            continue
        if line.startswith('p cnf'):
            parts = line.split()
            try:
                n_vars = int(parts[2])
            except Exception:
                pass
            continue
        # normalize
        line = line.replace('.', ' ')
        tokens = line.split()
        try:
            lits = [int(x) for x in tokens]
        except Exception:
            continue
        for lit in lits:
            if lit == 0:
                if current_clause:
                    clauses.append(tuple(current_clause))
                current_clause = []
            else:
                current_clause.append(lit)
    if current_clause:
        clauses.append(tuple(current_clause))
    return clauses, n_vars


def paramindex_to_subset_v2(parameter_index: int, n_vars: int, n_mask_bits: int) -> List[int]:
    mask_value = parameter_index & ((1 << n_mask_bits) - 1)
    subset = set()
    for i in range(n_mask_bits):
        if (mask_value >> i) & 1:
            subset.add(i % n_vars)
    return list(subset)


def extract_subset_from_top_samples(samples_norm, n_vars, topT, threshold, n_mask_bits):
    top_samples = sorted(samples_norm.items(), key=lambda kv: kv[1], reverse=True)[:topT]
    var_scores = [0.0] * n_vars
    for idx, prob in top_samples:
        subset = paramindex_to_subset_v2(idx, n_vars, n_mask_bits)
        for v in subset:
            if 0 <= v < n_vars:
                var_scores[v] += prob
    chosen = [i for i, s in enumerate(var_scores) if s >= threshold]
    if not chosen:
        k = min(6, n_vars)
        chosen = sorted(range(n_vars), key=lambda i: var_scores[i], reverse=True)[:k]
    return chosen


def reduce_problem_by_fixing(sat_problem: qls.SATProblem, fixed_assign: Dict[int,bool]):
    all_vars = set(range(sat_problem.n_vars))
    remaining = sorted(list(all_vars - set(fixed_assign.keys())))
    g2l = {g:i+1 for i,g in enumerate(remaining)}
    new_clauses = []
    for c in sat_problem.clauses:
        lits = c.literals
        satisfied = False
        newl = []
        for lit in lits:
            v0 = abs(lit)-1
            pos = lit>0
            if v0 in fixed_assign:
                if fixed_assign[v0] == pos:
                    satisfied = True
                    break
                else:
                    continue
            if v0 in g2l:
                nv = g2l[v0]
                newl.append(nv if pos else -nv)
        if satisfied:
            continue
        if not newl:
            return None, None
        new_clauses.append(qls.SATClause(tuple(newl)))
    return qls.SATProblem(n_vars=len(remaining), clauses=new_clauses), remaining


def try_candidate_backdoor_pysat(sat_problem: qls.SATProblem, candidate_subset_0_idx: List[int], max_k_allowed: int) -> Optional[Dict[int,bool]]:
    if len(candidate_subset_0_idx) > max_k_allowed:
        print(f"  Skipping candidate of size {len(candidate_subset_0_idx)} > {max_k_allowed}")
        return None
    if not PYSAT_AVAILABLE:
        print("  PySAT not available; skipping classical FPT step.")
        return None
    from itertools import product
    for bits in product([False, True], repeat=len(candidate_subset_0_idx)):
        fixed = {v:val for v,val in zip(candidate_subset_0_idx, bits)}
        reduced, remaining = reduce_problem_by_fixing(sat_problem, fixed)
        if reduced is None:
            continue
        if not reduced.clauses:
            full = fixed.copy()
            for r in remaining:
                full[r] = True
            return full
        solver = Glucose3()
        for c in reduced.clauses:
            solver.add_clause(list(c.literals))
        if solver.solve():
            model = solver.get_model()
            solver.delete()
            sol = {}
            for lit in model:
                lv = abs(lit)-1
                val = lit>0
                if lv < len(remaining):
                    sol[remaining[lv]] = val
            full = fixed.copy()
            full.update(sol)
            for r in remaining:
                if r not in full:
                    full[r] = True
            return full
        else:
            solver.delete()
    return None


# Runner

def run_trial(clauses, n_vars, problem_name, cfg, trial_seed):
    try:
        sat_clauses = [qls.SATClause(c) for c in clauses]
        problem = qls.SATProblem(n_vars=n_vars, clauses=sat_clauses)
    except Exception as e:
        return {"status":"error","error":str(e)}

    start_total = time.time()

    p_layers = cfg['p_layers']
    bits_per_param = cfg['bits_per_param']
    try:
        ansatz, n_params = qls.create_qaoa_ansatz(problem, p_layers)
        ham = qls.sat_to_hamiltonian(problem)
    except Exception as e:
        return {"status":"error","error":f"ansatz/ham build failed: {e}"}

    param_bounds = qls.np.array([[0.0, 2*qls.np.pi] if i%2==0 else [0.0, qls.np.pi] for i in range(n_params)])
    qls.np.random.seed(trial_seed)
    theta0 = qls.np.random.rand(n_params) * (param_bounds[:,1]-param_bounds[:,0]) + param_bounds[:,0]

    param_idx, n_qubits_p = qls.encode_parameters(theta0, param_bounds, bits_per_param)

    # safety: only run quantum evolution if n_qubits_p <= cfg['max_simulatable_qubits']
    if n_qubits_p > cfg['max_simulatable_qubits']:
        print(f"  Skipping quantum evolution: n_qubits_p={n_qubits_p} > {cfg['max_simulatable_qubits']}")
        return {"status":"skipped","reason":"too-many-qubits"}

    try:
        U_PE = qls.build_coherent_phase_oracle_nisq(n_params, bits_per_param, param_bounds, 0.35, ansatz, ham, problem.n_vars)
        if U_PE is None:
            return {"status":"error","error":"U_PE build returned None"}
    except Exception as e:
        return {"status":"error","error":str(e)}

    T_gate = qls.build_tunneling_operator_QW(n_qubits_p)
    evol_qc = qls.run_qlto_evolution_nisq(n_qubits_p, problem.n_vars, param_idx, U_PE, T_gate, K_steps=3)
    if evol_qc is None:
        return {"status":"error","error":"evolution circuit build failed"}

    sampler = qls.CountingWrapper(qls.BaseSampler())
    samples = qls.measure_and_process_samples_nisq(evol_qc, cfg['shots'], n_qubits_p, 'param', sampler)
    if not samples:
        return {"status":"ok","success":False,"reason":"no-samples"}
    totalp = sum(samples.values())
    samples_norm = {idx: p/totalp for idx, p in samples.items()}

    superset = extract_subset_from_top_samples(samples_norm, problem.n_vars, cfg['top_T_candidates'], cfg['freq_threshold'], cfg['N_MASK_BITS'])
    if len(superset) > cfg['max_candidate_size']:
        print(f"  Superset size {len(superset)} > max_candidate_size {cfg['max_candidate_size']}; truncating to top {cfg['max_candidate_size']}")
        superset = superset[:cfg['max_candidate_size']]

    # Direct try (no greedy shrink) if small enough
    solution_0_idx = None
    if len(superset) <= cfg['max_candidate_size']:
        solution_0_idx = try_candidate_backdoor_pysat(problem, superset, cfg['max_candidate_size'])

    total_time = time.time() - start_total

    result = {
        "status":"ok",
        "success": solution_0_idx is not None,
        "subset": superset,
        "sampler_time": None,
        "classical_search_time": None,
        "total_time": total_time,
    }
    if solution_0_idx is not None:
        result['solution'] = solution_0_idx

    return result


def run_sweep(cfg):
    out = cfg['output_csv']
    header = ["problem_name","N","M","trial","status","success","subset","total_time","reason"]
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    random_seed = cfg['random_seed']
    run_count = 0
    total_runs = len(SATLIB_TEST_SUITE) * cfg['trials_per_setting']

    print("Running conservative QLTO-FPT sweep")
    for name, cnf in SATLIB_TEST_SUITE.items():
        try:
            clauses, n_vars = parse_dimacs_cnf(cnf)
            m = len(clauses)
        except Exception as e:
            print(f"Failed to parse {name}: {e}")
            continue
        for t in range(cfg['trials_per_setting']):
            run_count += 1
            seed = random_seed + run_count
            print(f"\n[Run {run_count}/{total_runs}] {name} N={n_vars} M={m} trial={t+1}")
            res = run_trial(clauses, n_vars, name, cfg, seed)
            row = [name, n_vars, m, t+1, res.get('status'), res.get('success'), str(res.get('subset')), res.get('total_time'), res.get('reason','')]
            with open(out, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print('->', row)

    print(f"Sweep finished. Results written to {out}")

if __name__ == '__main__':
    if not qls.QISKIT_AVAILABLE:
        print("Error: Qiskit not available. Install qiskit and qiskit-aer to run.")
    else:
        run_sweep(CFG)
