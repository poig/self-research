"""tsp_pipeline.py

Builds a permutation-style TSP QUBO from city coordinates, partitions variables by city
clusters (simple k-means), and writes per-part QUBO pickles for downstream conversion/estimation.

Notes and caveats:
- This is a pragmatic encoder for resource estimation and partition experiments, not an
  optimized production QUBO generator.
- Partitioning is done by clustering city coordinates (geometric), then keeping only quadratic
  terms whose both endpoints are in the same cluster. Inter-cluster terms are written to a
  separate border qubo (if requested) so you can inspect separator sizes.

Output QUBO pickle format: (n_vars, linear_terms_dict, quad_terms_list)
- n_vars: int
- linear_terms_dict: {var_index (1-based): coef}
- quad_terms_list: [(i (1-based), j (1-based), coef), ...]

Example (PowerShell):
  python .\tools\tsp_pipeline.py --n 100 --parts 8 --out-dir benchmarks/out/tsp_demo

"""
from __future__ import annotations
import argparse
import gzip
import math
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np


def generate_random_points(n: int, seed: int | None = None):
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2)
    return pts


def read_csv_coords(path: str) -> np.ndarray:
    pts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(',', ' ').split() if p]
            if len(parts) < 2:
                continue
            pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts)


def pairwise_distance_matrix(pts: np.ndarray) -> np.ndarray:
    n = pts.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        d = np.linalg.norm(pts[i] - pts, axis=1)
        D[i, :] = d
    return D


def make_permutation_qubo_from_dist(dist: np.ndarray, A: float = None, B: float = None):
    # permutation QUBO with variables x_{i,t}, flatten index var = i*n + t (0-based)
    n = dist.shape[0]
    Nvars = n * n
    quad_terms: List[Tuple[int, int, float]] = []
    linear_terms: Dict[int, float] = {}

    max_w = float(dist.max()) if dist.size else 1.0
    if A is None:
        A = n * max_w * 5.0
    if B is None:
        B = A

    # travel costs: sum_{t} sum_{i!=j} dist[i][j] x_{i,t} x_{j,t+1}
    for t in range(n):
        tnext = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                v1 = i * n + t
                v2 = j * n + tnext
                coef = float(dist[i, j])
                quad_terms.append((v1 + 1, v2 + 1, coef))

    # row constraints: each city visited once
    for i in range(n):
        idxs = [i * n + t for t in range(n)]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                v1 = idxs[a] + 1
                v2 = idxs[b] + 1
                quad_terms.append((v1, v2, 2.0 * A))
        for v in idxs:
            linear_terms[v + 1] = linear_terms.get(v + 1, 0.0) + (-2.0 * A)

    # column constraints: each position occupied by exactly one city
    for t in range(n):
        idxs = [i * n + t for i in range(n)]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                v1 = idxs[a] + 1
                v2 = idxs[b] + 1
                quad_terms.append((v1, v2, 2.0 * B))
        for v in idxs:
            linear_terms[v + 1] = linear_terms.get(v + 1, 0.0) + (-2.0 * B)

    return Nvars, linear_terms, quad_terms


def kmeans_cluster(points: np.ndarray, k: int, seed: int | None = None, iters: int = 100):
    rng = np.random.RandomState(seed)
    n = points.shape[0]
    # initialize centers by sampling k distinct points
    indices = rng.choice(n, size=k, replace=False) if k <= n else np.arange(n)
    centers = points[indices].astype(float)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        # assign
        dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for j in range(k):
            mem = points[labels == j]
            if len(mem) == 0:
                centers[j] = points[rng.randint(0, n)]
            else:
                centers[j] = mem.mean(axis=0)
    return labels


def partition_qubo_by_city_clusters(n_cities: int, quad_terms: List[Tuple[int, int, float]], labels: List[int], parts: int):
    # city index is 0..n_cities-1; variable index var = city*n + t (0-based)
    cluster_quads: List[List[Tuple[int, int, float]]] = [[] for _ in range(parts)]
    border_quads: List[Tuple[int, int, float]] = []
    for (i, j, coef) in quad_terms:
        # convert to 0-based
        a = i - 1
        b = j - 1
        # deduce city indices from var indices
        city_a = a // n_cities
        city_b = b // n_cities
        if labels[city_a] == labels[city_b]:
            cluster_quads[labels[city_a]].append((i, j, coef))
        else:
            border_quads.append((i, j, coef))
    return cluster_quads, border_quads


def save_qubo(path: str, nvars: int, linear: Dict[int, float], quad: List[Tuple[int, int, float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump((nvars, linear, quad), f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100, help='Number of cities')
    p.add_argument('--csv', type=str, default='', help='CSV coordinates file (x,y per line)')
    p.add_argument('--parts', type=int, default=8, help='Number of city clusters/parts')
    p.add_argument('--out-dir', type=str, default='benchmarks/out/tsp_pipeline', help='Output directory')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--write-border', action='store_true', help='Write inter-cluster border quad terms separately')
    args = p.parse_args()

    if args.csv:
        pts = read_csv_coords(args.csv)
    else:
        pts = generate_random_points(args.n, seed=args.seed)

    n = pts.shape[0]
    D = pairwise_distance_matrix(pts)
    nvars, linear, quad = make_permutation_qubo_from_dist(D)
    print('Built QUBO: nvars=', nvars, 'quad_terms=', len(quad))

    labels = kmeans_cluster(pts, args.parts, seed=args.seed)
    cluster_quads, border_quads = partition_qubo_by_city_clusters(n, quad, labels, args.parts)

    # write global QUBO
    global_path = os.path.join(args.out_dir, 'tsp_global_qubo.pkl.gz')
    save_qubo(global_path, nvars, linear, quad)
    print('WROTE global qubo ->', global_path)

    # write per-cluster qubos (only intra-cluster quad terms included)
    for i, qlist in enumerate(cluster_quads):
        path = os.path.join(args.out_dir, f'part_{i}_qubo.pkl.gz')
        save_qubo(path, nvars, linear, qlist)
        print('WROTE part', i, '->', path, 'quad_terms=', len(qlist))

    if args.write_border:
        path = os.path.join(args.out_dir, 'border_qubo.pkl.gz')
        save_qubo(path, nvars, {}, border_quads)
        print('WROTE border qubo ->', path, 'terms=', len(border_quads))

    print('Done. Clusters sizes:', [int((labels == i).sum()) for i in range(args.parts)])


if __name__ == '__main__':
    main()
