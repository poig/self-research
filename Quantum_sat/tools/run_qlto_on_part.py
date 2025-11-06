"""Run QLTO FPT pipeline on a compact CNF part.

Usage:
  python .\tools\run_qlto_on_part.py --part benchmarks/parts_recursive/part_0_cnf_compact.pkl.gz

This script loads the compact CNF (n_vars, clauses) and calls
`src.solvers.qlto_qaoa_sat.run_fpt_pipeline` with a minimal config.

This is intended for small parts (n_vars <= ~100). It requires Qiskit
if the QLTO path is exercised; otherwise the function may fall back to
PySAT-based heuristics. If Qiskit is not installed, the function may still
run and return a conservative result.
"""
from __future__ import annotations
import argparse
import gzip
import pickle
import os
import sys

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.solvers.qlto_qaoa_sat import run_fpt_pipeline


def read_compact(path: str):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--part', required=True)
    p.add_argument('--bits-per-param', type=int, default=3, help='Bits per VQE parameter (controls n_qubits_p = n_params * bits_per_param)')
    p.add_argument('--n-mask-bits', type=int, default=8, help='N_MASK_BITS used by the FPT pipeline (must be <= n_qubits_p)')
    p.add_argument('--p-layers', type=int, default=1, help='QAOA p layers')
    p.add_argument('--shots', type=int, default=256, help='Shots for sampling')
    args = p.parse_args()

    nvars, clauses = read_compact(args.part)
    print('Loaded part:', args.part, 'nvars=', nvars, 'n_clauses=', len(clauses))

    cfg = {
        'p_layers': args.p_layers,
        'bits_per_param': args.bits_per_param,
        'N_MASK_BITS': args.n_mask_bits,
        'shots': args.shots,
        'top_T_candidates': 20,
        'verbose': True,
    }

    res = run_fpt_pipeline(clauses, nvars, 'tsp_part', cfg, trial_seed=0)
    print('Result:')
    for k, v in res.items():
        print(' ', k, ':', v)

if __name__ == '__main__':
    main()
