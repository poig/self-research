r"""
Convert compact QUBO pickle into a compact CNF-like pickle the estimator can load.

Input QUBO format (gzipped pickle): (n_vars, linear_terms_dict, quad_terms_list)
- n_vars: integer
- linear_terms_dict: {var_index (1-based): coef}
- quad_terms_list: [(i (1-based), j (1-based), coef), ...]

Output compact CNF format (gzipped pickle): (n_vars, clauses)
- clauses: list of tuples of ints (literals) representing 2-literal clauses for each quadratic term

We map each quadratic term a*x_i*x_j into a soft constraint by creating a pairwise clause
representing a penalty. This is a heuristic translation: we create clauses that prefer x_i XOR x_j
or similar depending on sign. This adapter is primarily for resource estimation (pauli-weight
histogram) where quadratic terms become Pauli weight-2 entries.

Usage (PowerShell):
    python .\tools\qubo_to_cnf_adapter.py --in benchmarks/out/tsp_n6_qubo.pkl.gz --out benchmarks/out/tsp_n6_qubo_cnf_compact.pkl.gz

"""
from __future__ import annotations
import argparse
import gzip
import pickle
import os
from typing import List, Tuple, Dict


def read_qubo(path: str):
    with gzip.open(path, 'rb') as f:
        nvars, linear, quad = pickle.load(f)
    return nvars, linear, quad


def qubo_to_cnf_compact(nvars: int, linear: Dict[int,float], quad: List[Tuple[int,int,float]]):
    clauses = []
    # For each quadratic term, create a 2-literal clause approximating the interaction.
    # Heuristic: if coef > 0 -> prefer both 0 or both 1 (x_i==x_j) -> (i or not j) & (not i or j) encoding is 2 CNF
    # But estimator expects a list of clauses (each clause is a disjunction). We'll add both implications as clauses.
    for (i, j, coef) in quad:
        if coef >= 0:
            # add (x_i or not x_j) and (not x_i or x_j) as two clauses
            clauses.append((i, -j))
            clauses.append((-i, j))
        else:
            # negative coupling -> prefer x_i != x_j -> (x_i or x_j) and (not x_i or not x_j)
            clauses.append((i, j))
            clauses.append((-i, -j))
    # Also create unary clauses from linear terms as single-literal clauses (approx)
    for v, coef in (linear or {}).items():
        # model positive linear as (v) and negative as (-v) to bias
        if coef > 0:
            clauses.append((v,))
        elif coef < 0:
            clauses.append((-v,))
    return nvars, clauses


def save_compact(path: str, nvars: int, clauses: List[Tuple[int]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump((nvars, clauses), f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--out', dest='out', required=True)
    args = p.parse_args()
    nvars, linear, quad = read_qubo(args.inp)
    nvars2, clauses = qubo_to_cnf_compact(nvars, linear, quad)
    save_compact(args.out, nvars2, clauses)
    print('WROTE compact CNF ->', args.out, 'nvars=', nvars2, 'clauses=', len(clauses))

if __name__ == '__main__':
    main()
