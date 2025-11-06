"""
Recursive partitioner

Reads a compact CNF (n_vars, clauses) and recursively partitions any part
whose variable count > --max-vars until all resulting parts have <= max-vars
variables. It can use the hypergraph partitioner (if provided) or fall back to
an in-process graph projection + community detection.

Outputs per-part compact gz files under --out-dir named part_<id>_cnf_compact.pkl.gz

Usage (PowerShell):
  python .\tools\recursive_partitioner.py --cnf .\benchmarks\out\aes10_clauses_compact.pkl.gz --max-vars 1200 --out-dir .\benchmarks\parts_recursive --partitioner C:\path\to\hmetis.exe --parts 4

This script tries to be conservative: when using an external partitioner it will
call it on a sub-CNF HGR file and then map back parts. When no external
partitioner is present it falls back to networkx-based partitioning.
"""
from __future__ import annotations
import argparse
import gzip
import pickle
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple

sys.path.insert(0, os.path.abspath('.'))


def read_compact_cnf(path: str):
    if path.endswith('.gz') or path.endswith('.pkl.gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def write_compact_part(path: str, nvars: int, clauses: List[Tuple[int, ...]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wb') as f:
        pickle.dump((nvars, clauses), f)


def run_hypergraph_partition_local(clauses, nvars, nparts, tmpdir, partitioner):
    # Minimal local HGR writer
    hgr = os.path.join(tmpdir, 'temp.hgr')
    with open(hgr, 'w') as f:
        f.write(f"{len(clauses)} {nvars}\n")
        for cl in clauses:
            vars_in = sorted({abs(int(l)) for l in cl})
            f.write(' '.join(str(v) for v in vars_in) + '\n')
    # call partitioner
    cmd = [partitioner, hgr, str(nparts)]
    try:
        subprocess.check_call(cmd, cwd=tmpdir)
    except Exception as e:
        print('Partitioner failed:', e)
        return []
    # find .part.*
    partf = None
    for fn in os.listdir(tmpdir):
        if '.part' in fn:
            partf = os.path.join(tmpdir, fn)
            break
    if not partf:
        return []
    parts = []
    with open(partf, 'r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts.append(int(line))
    return parts


def in_process_partition(clauses, use_clique=False):
    try:
        import networkx as nx
    except Exception:
        raise RuntimeError('networkx required for in-process partitioning')
    try:
        from community import best_partition as louvain_best
    except Exception:
        louvain_best = None
    G = nx.Graph()
    for cl in clauses:
        vars_in = sorted({abs(int(l)) - 1 for l in cl})
        for v in vars_in:
            if not G.has_node(v):
                G.add_node(v)
        if use_clique:
            for i in range(len(vars_in)):
                for j in range(i+1, len(vars_in)):
                    G.add_edge(vars_in[i], vars_in[j])
        else:
            cnode = ('c', id(cl))
            G.add_node(cnode)
            for v in vars_in:
                G.add_edge(cnode, v)
    # try louvain
    if louvain_best is not None:
        partmap = louvain_best(G)
        communities = {}
        for v, c in partmap.items():
            communities.setdefault(c, []).append(v)
        parts = list(communities.values())
        # return parts as lists of variable indices (1-based)
        parts_out = []
        for p in parts:
            vs = [v for v in p if isinstance(v, int)]
            parts_out.append([v+1 for v in vs])
        return parts_out
    else:
        coms = list(nx.algorithms.community.greedy_modularity_communities(G))
        parts_out = []
        for c in coms:
            vs = [v for v in c if isinstance(v, int)]
            parts_out.append([v+1 for v in vs])
        return parts_out


def clause_subset_for_vars(clauses, vars_set):
    out = []
    for cl in clauses:
        if any(abs(int(l)) in vars_set for l in cl):
            out.append(cl)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cnf', required=True)
    p.add_argument('--max-vars', type=int, default=1200)
    p.add_argument('--out-dir', type=str, default='benchmarks/parts_recursive')
    p.add_argument('--partitioner', type=str, default='')
    p.add_argument('--parts', type=int, default=4, help='Number of parts to split each large block into per iteration')
    args = p.parse_args()

    nvars, clauses = read_compact_cnf(args.cnf)
    work_queue = [(None, list(range(1, nvars+1)), clauses)]  # tuples: (parent_id, var_list, clause_list)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    next_pid = 0
    while work_queue:
        parent, var_list, clist = work_queue.pop(0)
        nvars_part = len(var_list)
        if nvars_part <= args.max_vars:
            # write out direct
            path = os.path.join(out_dir, f'part_{next_pid}_cnf_compact.pkl.gz')
            # build clauses restricted to var_list
            cls = clause_subset_for_vars(clist, set(var_list))
            write_compact_part(path, max(var_list) if var_list else 0, cls)
            print('WROTE part', next_pid, 'n_vars=', len(var_list), 'n_clauses=', len(cls))
            next_pid += 1
            continue
        # else split this block
        print('Recursively partitioning block with', nvars_part, 'vars')
        # build sub-clauses restricted to this var_list
        sub_clauses = clause_subset_for_vars(clist, set(var_list))
        # try hypergraph partitioner if available
        parts_vars = None
        if args.partitioner:
            with tempfile.TemporaryDirectory() as td:
                try:
                    parts_list = run_hypergraph_partition_local(sub_clauses, max(var_list), args.parts, td, args.partitioner)
                    if parts_list:
                        # parts_list is a list of part ids per vertex (1-based)
                        # build mapping var->part
                        var_to_part = {}
                        for idx, pid in enumerate(parts_list):
                            var_to_part[idx+1] = pid
                        parts_vars = {}
                        for v in var_list:
                            pid = var_to_part.get(v, 0)
                            parts_vars.setdefault(pid, []).append(v)
                except Exception as e:
                    print('Hypergraph partition failed, falling back to in-process', e)
        if parts_vars is None:
            # in-process partitioning
            parts_out = in_process_partition(sub_clauses, use_clique=False)
            # parts_out is list of lists of 1-based var indices; filter to our var_list
            parts_vars = {}
            for i, p in enumerate(parts_out):
                # intersect with var_list
                pv = [v for v in p if v in set(var_list)]
                if pv:
                    parts_vars[i] = pv
        # enqueue each child part (keep same clause universe for now)
        for pid, vlist in parts_vars.items():
            work_queue.append((next_pid, vlist, clist))
            next_pid += 1

if __name__ == '__main__':
    main()
