"""tsp_demo_compare.py

Lightweight demo harness that builds a TSP instance, partitions it using k-means (city clustering),
converts per-part QUBOs to compact CNF-like files using the existing adapter, and invokes the
estimator in batch mode. It also runs OR-Tools TSP solver on the whole instance as a classical baseline
and compares distances.

This script is designed to be runnable locally; it avoids heavy canonical expansions and instead
produces the estimator-friendly compact CNFs and calls `qlto_resource_estimator.py` in batch mode
(if available in the same environment).

Requirements (install locally):
- numpy
- ortools (pip install ortools)  # for the classical TSP baseline

Example (PowerShell):
  # generate, partition into 8 clusters, write per-part QUBOs and convert to CNF compacts
  python .\tools\tsp_demo_compare.py --n 100 --parts 8 --out-dir benchmarks/out/tsp_demo --run-estimator

"""
from __future__ import annotations
import argparse
import gzip
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np

# Ensure project root is on sys.path so `from tools import ...` works when this
# script is executed directly from the repository root. This keeps imports
# stable whether you run `python tools/tsp_demo_compare.py` or `python -m tools.tsp_demo_compare`.
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from tools.tsp_pipeline import generate_random_points, pairwise_distance_matrix, make_permutation_qubo_from_dist, save_qubo


def solve_tsp_ortools(coords: np.ndarray) -> Tuple[float, List[int]]:
    try:
        from ortools.constraint_solver import pywrapcp
        from ortools.constraint_solver import routing_enums_pb2
    except Exception as e:
        print('OR-Tools not installed or failed to import:', e)
        return float('inf'), []

    n = coords.shape[0]
    dist_mat = (np.round(pairwise_distance_matrix(coords) * 100).astype(int)).tolist()

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return dist_mat[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return float('inf'), []

    index = routing.Start(0)
    route = []
    total = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(node)
        next_index = solution.Value(routing.NextVar(index))
        total += dist_mat[node][manager.IndexToNode(next_index)]
        index = next_index
    # convert back to float distance
    return total / 100.0, route


def convert_qubo_parts_to_cnf(out_dir: str):
    # find part_*.pkl.gz and convert
    files = [f for f in os.listdir(out_dir) if f.startswith('part_') and f.endswith('_qubo.pkl.gz')]
    converted = []
    for f in files:
        inpath = os.path.join(out_dir, f)
        outpath = os.path.join(out_dir, f.replace('_qubo.pkl.gz', '_cnf_compact.pkl.gz'))
        cmd = [sys.executable, os.path.join('tools', 'qubo_to_cnf_adapter.py'), '--in', inpath, '--out', outpath]
        print('RUN:', ' '.join(cmd))
        subprocess.check_call(cmd)
        converted.append(outpath)
    return converted


def run_estimator_on_parts(parts_dir: str, out_csv: str):
    cmd = [sys.executable, os.path.join('benchmarks', 'qlto_resource_estimator.py'), '--parts-dir', parts_dir, '--out', out_csv, '--plot']
    print('RUN:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--parts', type=int, default=8)
    p.add_argument('--out-dir', type=str, default='benchmarks/out/tsp_demo')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-estimator', action='store_true')
    args = p.parse_args()

    pts = generate_random_points(args.n, seed=args.seed)
    D = pairwise_distance_matrix(pts)

    total_dist, route = solve_tsp_ortools(pts)
    print('OR-Tools route length (approx):', total_dist)

    nvars, linear, quad = make_permutation_qubo_from_dist(D)
    os.makedirs(args.out_dir, exist_ok=True)
    global_path = os.path.join(args.out_dir, 'tsp_global_qubo.pkl.gz')
    save_qubo(global_path, nvars, linear, quad)
    print('WROTE global qubo ->', global_path)

    # partition by simple kmeans (reuse pipeline) using external script
    cmd = [sys.executable, os.path.join('tools', 'tsp_pipeline.py'), '--n', str(args.n), '--parts', str(args.parts), '--out-dir', args.out_dir, '--seed', str(args.seed)]
    print('RUN:', ' '.join(cmd))
    subprocess.check_call(cmd)

    converted = convert_qubo_parts_to_cnf(args.out_dir)
    print('Converted parts:', converted)

    if args.run_estimator:
        out_csv = os.path.join(args.out_dir, 'tsp_parts_estimates.csv')
        run_estimator_on_parts(args.out_dir, out_csv)
        print('Estimator wrote:', out_csv)


if __name__ == '__main__':
    main()
