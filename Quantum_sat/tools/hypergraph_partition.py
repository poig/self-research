"""
Hypergraph partition driver for CNF files.

This script reads a compact CNF pickle (n_vars, clauses) and writes an hMetis-style
hypergraph file where each clause is a hyperedge over variable ids (1-based).
Optionally, if `--hmetis-bin` is provided and runnable, the script will call hMetis
and map the resulting part labels back to variables and write per-part compact
CNF files suitable for `qlto_resource_estimator.py`.

Usage:
  python tools/hypergraph_partition.py --cnf benchmarks/out/aes10_clauses_compact.pkl.gz --parts 32 --out-dir benchmarks/parts_hmetis --hmetis-bin C:\path\to\hmetis.exe

If no partitioner binary is provided, the script will only write the `.hgr` file
and exit, so you can partition externally.
"""
from __future__ import annotations
import argparse
import gzip
import pickle
import os
import subprocess
import sys
from typing import List, Tuple


def read_compact_cnf(path: str) -> Tuple[int, List[Tuple[int, ...]]]:
    if path.endswith('.gz') or path.endswith('.pkl.gz'):
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], list):
        return obj[0], obj[1]
    else:
        # assume obj is clauses list
        clauses = obj
        nvars = max(abs(l) for cl in clauses for l in cl)
        return nvars, clauses


def write_hmetis_hgr(nvars: int, clauses: List[Tuple[int, ...]], outpath: str):
    """Write a simple hMetis/hgr format: first line = #hyperedges #vertices
    followed by one hyperedge per line listing vertex ids (1-based) separated by spaces.
    """
    with open(outpath, 'w') as f:
        f.write(f"{len(clauses)} {nvars}\n")
        for cl in clauses:
            # map clause literals to variable indices (1-based)
            vars_in = sorted({abs(int(l)) for l in cl})
            if not vars_in:
                f.write('\n')
            else:
                f.write(' '.join(str(v) for v in vars_in) + '\n')


def parse_hmetis_output(parts_file: str) -> List[int]:
    """Read hMetis partition output: one part label per line for each vertex (1-based indexing).
    Returns list of part ids indexed by vertex_id-1.
    """
    parts = []
    with open(parts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts.append(int(line))
    return parts


def map_vars_to_parts(parts: List[int]) -> dict:
    var2part = {}
    for vidx, pid in enumerate(parts):
        var2part[vidx + 1] = pid
    return var2part


def compute_part_clauses(clauses, var2part):
    from collections import defaultdict
    part_clauses = defaultdict(list)
    separators = set()
    for cl in clauses:
        vars_in = sorted({abs(int(l)) for l in cl})
        touched = set(var2part.get(v, -1) for v in vars_in)
        if len(touched) == 1 and -1 not in touched:
            pid = next(iter(touched))
            part_clauses[pid].append(cl)
        else:
            for v in vars_in:
                if v in var2part:
                    separators.add(v)
    return part_clauses, separators


def write_part_compact_gz(outdir: str, pid: int, clauses: List[Tuple[int, ...]]):
    os.makedirs(outdir, exist_ok=True)
    # compute nvars for this part
    nvars_part = 0
    for cl in clauses:
        for l in cl:
            nvars_part = max(nvars_part, abs(int(l)))
    path = os.path.join(outdir, f'part_{pid}_cnf_compact.pkl.gz')
    with gzip.open(path, 'wb') as f:
        pickle.dump((nvars_part, clauses), f)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cnf', required=True)
    p.add_argument('--parts', type=int, default=32, help='Desired number of parts')
    p.add_argument('--out-dir', type=str, default='benchmarks/parts_hmetis')
    p.add_argument('--hmetis-bin', type=str, default='', help='Path to hMetis/khmetis binary (optional)')
    p.add_argument('--hmetis-opts', type=str, default='', help='Extra options to pass to hMetis')
    args = p.parse_args()

    nvars, clauses = read_compact_cnf(args.cnf)
    hgr_path = os.path.join(args.out_dir if args.out_dir else '.', 'temp.hgr')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Writing hypergraph {hgr_path} (edges={len(clauses)} vars={nvars})')
    write_hmetis_hgr(nvars, clauses, hgr_path)

    if not args.hmetis_bin:
        print('No hMetis binary provided; wrote .hgr file for external partitioning.')
        print('You can run hMetis externally and then re-run this script with --hmetis-bin to process parts.')
        return

    # Attempt to call hMetis; different binaries expect different flags. We'll try common ones.
    parts_out_base = os.path.join(args.out_dir, 'hmetis_parts')
    cmd = [args.hmetis_bin, hgr_path, str(args.parts)]
    if args.hmetis_opts:
        cmd += args.hmetis_opts.split()
    print('Running partitioner:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print('Partitioner call failed:', e)
        return

    # hMetis usually emits a .part.<k> file next to the .hgr (or prints to stdout). Try common names
    candidates = [hgr_path + f'.part.{args.parts}', os.path.join(os.path.dirname(hgr_path), os.path.basename(hgr_path) + f'.part.{args.parts}'), parts_out_base + f'.part.{args.parts}']
    parts_file = None
    for c in candidates:
        if os.path.exists(c):
            parts_file = c
            break
    if parts_file is None:
        # look for any .part.* files in outdir
        for fn in os.listdir(args.out_dir):
            if fn.startswith('temp.hgr.part') or fn.endswith(f'.part.{args.parts}'):
                parts_file = os.path.join(args.out_dir, fn)
                break
    if parts_file is None:
        print('Could not find partition output file. Please run hMetis manually and provide the .part file path.')
        return

    print('Found partition output:', parts_file)
    parts = parse_hmetis_output(parts_file)
    var2part = map_vars_to_parts(parts)
    part_clauses, separators = compute_part_clauses(clauses, var2part)

    written = []
    for pid, cls in part_clauses.items():
        if not cls:
            continue
        path = write_part_compact_gz(args.out_dir, pid, cls)
        print('WROTE part', pid, '->', path, 'clauses=', len(cls))
        written.append(path)

    print('Separator size (vars appearing across parts):', len(separators))
    print('WROTE', len(written), 'parts to', args.out_dir)

if __name__ == '__main__':
    main()
