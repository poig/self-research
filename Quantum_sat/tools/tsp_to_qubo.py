"""
Simple TSP -> QUBO converter.

Creates a compact QUBO pickle suitable for quick experiments. The script supports
- reading a TSPLIB-like matrix from a CSV (rows of distances) or
- generating a small random metric instance for testing.

Output format (gzipped pickle): (n_vars, linear_terms, quad_terms)
- n_vars: integer number of binary variables
- linear_terms: dict {var_index (1-based): coef}
- quad_terms: list of tuples (i (1-based), j (1-based), coef)

By default we build the classical permutation-based QUBO for TSP using variables
x_{i,t} (city i at position t), flattened with var_id = i*n + t + 1 (1-based).

Usage examples (PowerShell):
  # generate small random instance (n=6)
  python .\tools\tsp_to_qubo.py --n 6 --out benchmarks/out/tsp_n6_qubo.pkl.gz --random

  # read distances from CSV
  python .\tools\tsp_to_qubo.py --csv data/tsp6.csv --out benchmarks/out/tsp6_qubo.pkl.gz

"""
from __future__ import annotations
import argparse
import gzip
import pickle
import math
import os
import random
from typing import List, Tuple, Dict


def read_csv_matrix(path: str) -> List[List[float]]:
    mat = []
    with open(path, 'r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            row = [float(x) for x in line.split(',')]
            mat.append(row)
    return mat


def make_qubo_from_distance_matrix(dist: List[List[float]], A: float=None, B: float=None):
    # n cities
    n = len(dist)
    # variables x_{i,t} with i=0..n-1, t=0..n-1
    # flatten index: var = i*n + t  (0-based)
    Nvars = n * n
    # objective: sum_{i,j,t} dist[i][j] * x_{i,t} * x_{j,(t+1) mod n}
    quad_terms = []  # (var1, var2, coef)
    linear_terms = {}
    max_w = max(max(row) for row in dist)
    # choose penalties if not provided
    if A is None:
        A = (n * max_w) * 10.0
    if B is None:
        B = A
    # add travel costs
    for t in range(n):
        tnext = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i == j: continue
                v1 = i * n + t
                v2 = j * n + tnext
                coef = dist[i][j]
                quad_terms.append((v1 + 1, v2 + 1, coef))
    # add row constraints: each city visited exactly once -> (sum_t x_{i,t} - 1)^2
    for i in range(n):
        idxs = [i * n + t for t in range(n)]
        # expand square: sum linear and quadratic
        for v in idxs:
            linear_terms[v+1] = linear_terms.get(v+1, 0.0) + A * (1 - 2*1)  # -2*A*1 inside expansion
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                v1 = idxs[a] + 1
                v2 = idxs[b] + 1
                quad_terms.append((v1, v2, 2.0 * A))
    # add column constraints: each position occupied by exactly one city -> (sum_i x_{i,t} - 1)^2
    for t in range(n):
        idxs = [i * n + t for i in range(n)]
        for v in idxs:
            linear_terms[v+1] = linear_terms.get(v+1, 0.0) + B * (1 - 2*1)
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                v1 = idxs[a] + 1
                v2 = idxs[b] + 1
                quad_terms.append((v1, v2, 2.0 * B))
    # Note: linear_terms currently holds penalty contributions as constants; we keep them as linear coefficients
    return Nvars, linear_terms, quad_terms


def save_qubo(path: str, nvars: int, linear: Dict[int, float], quad: List[Tuple[int,int,float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump((nvars, linear, quad), f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=6, help='Number of cities for random instance')
    p.add_argument('--csv', type=str, default='', help='CSV file with distance matrix (overrides --n)')
    p.add_argument('--out', type=str, default='benchmarks/out/tsp_qubo.pkl.gz')
    p.add_argument('--random', action='store_true', help='Generate a random symmetric metric')
    args = p.parse_args()

    if args.csv:
        mat = read_csv_matrix(args.csv)
    else:
        n = args.n
        # generate random metric (Euclidean in a plane)
        pts = [(random.random(), random.random()) for _ in range(n)]
        mat = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                mat[i][j] = math.hypot(dx, dy)

    nvars, linear, quad = make_qubo_from_distance_matrix(mat)
    save_qubo(args.out, nvars, linear, quad)
    print('WROTE QUBO ->', args.out, 'nvars=', nvars, 'quad_terms=', len(quad))

if __name__ == '__main__':
    main()
