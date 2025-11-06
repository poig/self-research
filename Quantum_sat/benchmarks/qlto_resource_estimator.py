#!/usr/bin/env python3
"""
QLTO + SAT -> Resource Estimator (AES-128 / AES-256 support + compact comparison CSV)

This enhanced script extends the earlier estimator to:
 - accept AES-128 or AES-256 automatic encoding (if encoder present),
 - load CNF pickles (or gzipped cnf) and run the clause-based estimator (no 2^k expansion),
 - compute a Grover baseline (iterations and oracle-work) for AES-128/256 using user-provided
   oracle-cost parameters (logical width and CNOT/T cost per oracle),
 - produce a compact CSV comparing QLTO estimates vs Grover baseline for AES-128 and AES-256.

Important: this script **does not** "prove" polynomial vs exponential runtime. It computes resource
estimations under explicit assumptions (you must set or measure the oracle cost, phys-per-logical,
optimizer budgets, etc.). Use results as an apples-to-apples quantitative comparison; they are not
a theoretical proof.

Usage examples (from repo root):
  # Estimate using an exported CNF for AES-128 (already created by your encode tool)
  python tools/qlto_resource_estimator.py --cnf benchmarks/out/aes10_clauses.pkl --n-vars 13248 --out benchmarks/compact_aes_compare.csv

  # Attempt to auto-encode AES-256 (if src.solvers.aes_full_encoder exposes encode_aes_256)
  python tools/qlto_resource_estimator.py --cipher aes256 --out benchmarks/compact_aes_compare.csv --force

  # Quick Grover baseline only (no CNF needed)
  python tools/qlto_resource_estimator.py --grover-only --keybits 256 --oracle-logical 300 --oracle-cnot 10000 --out grover_only.csv

Notes:
 - The script uses a conservative gadget cost model for clause exponentials: 2*(k-1) CNOTs per clause of arity k,
   and assumes one ancilla recycled sequentially unless you override.
 - To get trustworthy QLTO resource numbers you should: (A) export canonical CNF for AES-128/256, (B) run this
   estimator on that CNF, and (C) measure exact QLTO ancilla from your module via get_logical_qubit_budget.
 - The Grover baseline requires you to provide an estimate of the per-oracle logical gate cost (CNOTs, T-count).
   Literature values vary — include your chosen oracle-cost numbers in the CSV so results are reproducible.

Output: a CSV file with rows for requested ciphers (AES-128, AES-256) comparing QLTO estimates and Grover baselines.

"""

from __future__ import annotations
import argparse
import pickle
import gzip
import os
import sys
from collections import Counter
import math
from typing import List, Tuple, Dict, Any
import time
import glob

# allow repo imports
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('Quantum_sat/src'))

# ----------------- Utilities (adapted from earlier estimator) -----------------

def _clauses_to_pauli_terms(clauses: List[Tuple[int, ...]]) -> List[Tuple[float, Tuple[int, ...]]]:
    terms = []
    for cl in clauses:
        vars_involved = set()
        for lit in cl:
            v = abs(int(lit)) - 1
            vars_involved.add(v)
        if len(vars_involved) == 0:
            continue
        terms.append((1.0, tuple(sorted(vars_involved))))
    return terms


def pauli_weight_histogram_from_clauses(clauses: List[Tuple[int, ...]]):
    weights = [len(set(abs(int(l)) for l in cl)) for cl in clauses]
    return Counter(weights)


def cnot_cost_per_term(weight: int) -> int:
    if weight <= 1:
        return 0
    return 2 * (weight - 1)


def estimate_from_clauses(n_vars: int,
                          clauses: List[Tuple[int, ...]],
                          qlto_extra_ancilla: int = 128,
                          sequential: bool = True,
                          phys_per_logical: int = 1000,
                          qaoa_layers: int = 2) -> Dict[str, Any]:
    hist = pauli_weight_histogram_from_clauses(clauses)
    cnot_per_layer = sum(cnot_cost_per_term(w) * cnt for w, cnt in hist.items())
    ancilla_for_terms = 1 if any(w > 2 for w in hist) and sequential else (min(len(clauses), 1) if not sequential else 0)
    logical_qubits = n_vars + qlto_extra_ancilla + ancilla_for_terms
    total_cnot_logical = cnot_per_layer * qaoa_layers
    return {
        'n_vars': n_vars,
        'n_clauses': len(clauses),
        'pauli_weight_hist': dict(hist),
        'cnot_per_layer': cnot_per_layer,
        'qaoa_layers': qaoa_layers,
        'total_cnot_logical': total_cnot_logical,
        'ancilla_for_terms': ancilla_for_terms,
        'qlto_extra_ancilla': qlto_extra_ancilla,
        'logical_qubits': logical_qubits,
        'physical_qubits_est': logical_qubits * phys_per_logical
    }


def estimate_qlto_ancilla_count() -> int:
    candidates = ['qlto_qaoa_sat', 'src.solvers.qlto_qaoa_sat', 'Quantum_sat.src.solvers.qlto_qaoa_sat']
    for modname in candidates:
        try:
            mod = __import__(modname, fromlist=['*'])
        except Exception:
            continue
        for fname in ['get_logical_qubit_budget', 'estimate_qlto_ancilla_count', 'build_w_gate_nisq', 'build_w_gate']:
            if hasattr(mod, fname):
                f = getattr(mod, fname)
                try:
                    # try calling defensive
                    if fname == 'get_logical_qubit_budget':
                        res = f(16)
                        if isinstance(res, dict) and 'detected_ancilla' in res:
                            return int(res['detected_ancilla'])
                    else:
                        # attempt small dry-run
                        try:
                            circ = f(n_qubits=8, return_circuit=True)
                        except Exception:
                            try:
                                circ = f(8)
                            except Exception:
                                circ = None
                        if circ is not None and hasattr(circ, 'num_qubits'):
                            nq = int(circ.num_qubits)
                            return max(0, nq - 8)
                except Exception:
                    continue
    return 128

# ----------------- Grover baseline calculation -----------------

def grover_iterations(key_bits: int) -> int:
    """Return the number of Grover oracle calls ~ pi/4 * 2^(key_bits/2)
    Use integer arithmetic where possible (returns Python int with possibly huge magnitude).
    """
    # compute 2^(key_bits/2) as integer if key_bits even, else use sqrt(2)*2^{(key_bits-1)/2}
    if key_bits % 2 == 0:
        base = 1 << (key_bits // 2)  # 2^(key_bits/2)
        # pi/4 factor cannot be exact as integer; return float*int as a float with exponent
        return int((math.pi / 4.0) * base)
    else:
        # odd key bits: use float approximation (rare for AES)
        return int((math.pi / 4.0) * (2 ** (key_bits / 2.0)))


def format_sci(x: Any) -> str:
    try:
        if isinstance(x, int):
            # convert big int to sci notation
            s = f"{x:.6e}"
            return s
        if isinstance(x, float):
            return f"{x:.6e}"
        return str(x)
    except Exception:
        return str(x)

# ----------------- High-level runner & CSV output -----------------

def run_estimate_for_cnf(cnf_path: str, n_vars: int = None, qlto_ancilla: int = None, phys_per_logical: int = 1000, qaoa_layers: int = 2, sequential: bool = True):
    # load CNF (pickle or gzip-pickle)
    knotify = globals().get('knotify', lambda m: print('[knotify]', m))
    knotify(f'Loading CNF from {cnf_path}')
    t0 = time.time()
    if cnf_path.endswith('.gz') or cnf_path.endswith('.pkl.gz'):
        with gzip.open(cnf_path, 'rb') as f:
            obj = pickle.load(f)
    else:
        with open(cnf_path, 'rb') as f:
            obj = pickle.load(f)
    knotify(f'Finished loading CNF in {time.time()-t0:.1f}s')
    # obj can be (n_vars, clauses) if compact, or clauses list
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], list):
        nvars = obj[0]
        clauses = obj[1]
    else:
        clauses = obj
        if n_vars is None:
            # infer
            maxv = 0
            for cl in clauses:
                for lit in cl:
                    maxv = max(maxv, abs(int(lit)))
            nvars = maxv
    qlto_extra = qlto_ancilla if qlto_ancilla is not None else estimate_qlto_ancilla_count()
    report = estimate_from_clauses(n_vars=nvars, clauses=clauses, qlto_extra_ancilla=qlto_extra, sequential=sequential, phys_per_logical=phys_per_logical, qaoa_layers=qaoa_layers)
    return report


def run_auto_encode(cipher: str, plaintext: bytes, ciphertext: bytes, master_key_vars: list, fast: bool = False, max_clauses: int = 2000, force: bool = False):
    # Attempt to import encoder functions
    try:
        enc_mod = __import__('src.solvers.aes_full_encoder', fromlist=['*'])
    except Exception:
        try:
            enc_mod = __import__('src.solvers.aes_full_encoder', fromlist=['*'])
        except Exception:
            enc_mod = None
    if enc_mod is None:
        raise RuntimeError('AES encoder module not found (src.solvers.aes_full_encoder). Provide --cnf instead.')
    if cipher.lower() == 'aes128':
        if not hasattr(enc_mod, 'encode_aes_128'):
            raise RuntimeError('encode_aes_128 not found in aes_full_encoder')
        ret = enc_mod.encode_aes_128(plaintext, ciphertext, master_key_vars)
    elif cipher.lower() == 'aes256':
        if not hasattr(enc_mod, 'encode_aes_256'):
            # try a generic encode with key length parameter
            if hasattr(enc_mod, 'encode_aes'):
                ret = enc_mod.encode_aes(plaintext, ciphertext, master_key_vars, key_bits=256)
            else:
                raise RuntimeError('encode_aes_256 not found; adapt encoder or export CNF manually')
        else:
            ret = enc_mod.encode_aes_256(plaintext, ciphertext, master_key_vars)
    else:
        raise ValueError('Unknown cipher: ' + cipher)
    # normalize
    if isinstance(ret, tuple) and len(ret) == 3:
        clauses, Nvars, round_keys = ret
    else:
        try:
            round_keys, clauses, next_var_id = ret
            Nvars = next_var_id
        except Exception:
            raise RuntimeError('Unexpected return signature from encoder')
    if fast:
        clauses = clauses[:max_clauses]
    return clauses, Nvars


def write_compact_csv(out_path: str, rows: List[Dict[str, Any]]):
    import csv
    keys = list(rows[0].keys()) if rows else []
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            # string-ify big ints
            out = {k: (format_sci(v) if isinstance(v, (int, float)) and abs(v) > 1e9 else v) for k, v in r.items()}
            w.writerow(out)
    print('WROTE CSV ->', out_path)


def plot_parts_summary(rows: List[Dict[str, Any]], out_prefix: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available; skipping plots')
        return
    # simple diagnostics: histogram of clauses per part and cnot_per_layer bar (log10)
    parts = [r for r in rows if r.get('approach', '').startswith('QLTO_cnf_part_') or r.get('part_id') is not None]
    if not parts:
        print('No part rows found to plot')
        return
    n_clauses = [int(r.get('part_n_clauses', r.get('n_clauses', 0))) for r in parts]
    cnot_vals = [float(r.get('cnot_per_layer', r.get('total_cnot_logical', 0))) for r in parts]

    # Histogram of clauses
    plt.figure(figsize=(6,4))
    plt.hist(n_clauses, bins=30)
    plt.xlabel('Clauses per part')
    plt.ylabel('Count')
    plt.title('Per-part clause size distribution')
    p1 = out_prefix + '_parts_clauses_hist.png'
    plt.tight_layout()
    plt.savefig(p1)
    print('WROTE plot ->', p1)

    # Bar plot of cnot (log10)
    import numpy as _np
    x = list(range(len(cnot_vals)))
    y = _np.log10(_np.array(cnot_vals, dtype=float) + 1.0)
    plt.figure(figsize=(8,4))
    plt.bar(x, y)
    plt.xlabel('Part index')
    plt.ylabel('log10(cnot_per_part+1)')
    plt.title('Per-part CNOT (log10)')
    p2 = out_prefix + '_parts_cnot_log10.png'
    plt.tight_layout()
    plt.savefig(p2)
    print('WROTE plot ->', p2)

# ----------------- CLI -----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cnf', type=str, help='Pickle (or gzipped pickle) of clauses or (n_vars, clauses) compact tuple')
    p.add_argument('--cipher', choices=['aes128', 'aes256'], help='Auto-encode AES-128 or AES-256 if encoder available')
    p.add_argument('--force', action='store_true', help='When auto-encoding large AES CNFs, force export (may be heavy)')
    p.add_argument('--fast', action='store_true', help='Use only the first --max-clauses clauses (safe)')
    p.add_argument('--max-clauses', type=int, default=2000)
    p.add_argument('--n-vars', type=int, default=None)
    p.add_argument('--qlto-ancilla', type=int, default=None)
    p.add_argument('--phys-per-logical', type=int, default=1000)
    p.add_argument('--qaoa-layers', type=int, default=2)
    p.add_argument('--sequential', action='store_true')
    p.add_argument('--out', type=str, default='benchmarks/compact_aes_compare.csv')
    p.add_argument('--grover-only', action='store_true', help='Only compute grover baseline for given keybits')
    p.add_argument('--keybits', type=int, default=128, help='Key size for Grover baseline when --grover-only used')
    p.add_argument('--oracle-logical', type=int, default=264, help='Logical qubits assumed for AES oracle (Grover)')
    p.add_argument('--oracle-cnot', type=int, default=20000, help='CNOTs per AES oracle evaluation (for Grover baseline)')
    p.add_argument('--oracle-t', type=int, default=None, help='T-count per AES oracle (optional)')
    p.add_argument('--oracle-phys-per-logical', type=int, default=1000, help='phys-per-logical multiplier for Grover mapping')
    # Partitioning options (in-process, no extra files)
    p.add_argument('--partition', action='store_true', help='Partition CNF into communities and run per-part estimates in-process')
    p.add_argument('--partition-method', choices=['louvain', 'greedy', 'components'], default='louvain', help='Partitioning method to use')
    p.add_argument('--partition-use-clique', action='store_true', help='When building variable graph, use clique projection (denser)')
    p.add_argument('--write-parts', type=str, default='', help='Directory to write per-part compact CNF files (optional)')
    p.add_argument('--part-max-vars', type=int, default=0, help='Only run in-process estimates for parts with <= this many vars (0 = no limit)')
    p.add_argument('--part-only-write', action='store_true', help='Only write per-part compact CNF files and exit (no in-process estimation)')
    p.add_argument('--parts-dir', type=str, default='', help='Directory containing per-part CNF files to process (batch mode)')
    p.add_argument('--parts-glob', type=str, default='part_*_cnf*.pkl*', help='glob pattern for part files inside --parts-dir')
    p.add_argument('--plot', action='store_true', help='Produce simple diagnostic plots for parts when --parts-dir is used')
    args = p.parse_args()

    rows = []

    # If parts-dir is provided as a standalone mode, process it immediately (no --cnf/--cipher required)
    if args.parts_dir:
        pd = args.parts_dir
        pattern = os.path.join(pd, args.parts_glob)
        files = sorted(glob.glob(pattern))
        if not files:
            print('No part files found matching', pattern)
            return
        print('Processing', len(files), 'part files from', pd)
        for fp in files:
            try:
                rep = run_estimate_for_cnf(fp, n_vars=None, qlto_ancilla=args.qlto_ancilla, phys_per_logical=args.phys_per_logical, qaoa_layers=args.qaoa_layers, sequential=args.sequential)
            except Exception as e:
                print('Failed to estimate part', fp, e)
                continue
            base = os.path.basename(fp)
            rows.append({
                'approach': f'QLTO_cnf_part_{base}',
                'part_file': base,
                'n_vars': rep.get('n_vars'),
                'n_clauses': rep.get('n_clauses'),
                'pauli_weight_hist': rep.get('pauli_weight_hist'),
                'cnot_per_layer': rep.get('cnot_per_layer'),
                'total_cnot_logical': rep.get('total_cnot_logical'),
                'logical_qubits': rep.get('logical_qubits'),
                'physical_qubits_est': rep.get('physical_qubits_est'),
            })
        if rows:
            write_compact_csv(args.out, rows)
            if args.plot:
                outpref = os.path.splitext(args.out)[0]
                plot_parts_summary(rows, outpref)
        return

    # Grover-only quick path
    if args.grover_only:
        iters = grover_iterations(args.keybits)
        total_oracle_cnot = iters * args.oracle_cnot
        phys_qubits = args.oracle_logical * args.oracle_phys_per_logical
        rows.append({
            'approach': f'Grover_{args.keybits}',
            'keybits': args.keybits,
            'grover_iters_approx': format_sci(iters),
            'oracle_logical_qubits': args.oracle_logical,
            'oracle_cnot': args.oracle_cnot,
            'total_oracle_cnot': format_sci(total_oracle_cnot),
            'physical_qubits_machine': phys_qubits,
        })
        write_compact_csv(args.out, rows)
        return

    # gather target cipher tasks
    tasks = []
    if args.cipher:
        tasks.append({'kind': 'auto', 'cipher': args.cipher})
    if args.cnf:
        tasks.append({'kind': 'cnf', 'cnf': args.cnf, 'n_vars': args.n_vars})

    if not tasks:
        print('Provide --cipher or --cnf (or use --grover-only).')
        return

    for t in tasks:
        # batch parts-dir mode (process parts files and aggregate rows)
        if args.parts_dir:
            pd = args.parts_dir
            pattern = os.path.join(pd, args.parts_glob)
            files = sorted(glob.glob(pattern))
            if not files:
                print('No part files found matching', pattern)
            else:
                print('Processing', len(files), 'part files from', pd)
                for fp in files:
                    try:
                        # reuse run_estimate_for_cnf which accepts compact gz pickles or pkl
                        rep = run_estimate_for_cnf(fp, n_vars=None, qlto_ancilla=args.qlto_ancilla, phys_per_logical=args.phys_per_logical, qaoa_layers=args.qaoa_layers, sequential=args.sequential)
                    except Exception as e:
                        print('Failed to estimate part', fp, e)
                        continue
                    # attach metadata
                    base = os.path.basename(fp)
                    rows.append({
                        'approach': f'QLTO_cnf_part_{base}',
                        'part_file': base,
                        'n_vars': rep.get('n_vars'),
                        'n_clauses': rep.get('n_clauses'),
                        'pauli_weight_hist': rep.get('pauli_weight_hist'),
                        'cnot_per_layer': rep.get('cnot_per_layer'),
                        'total_cnot_logical': rep.get('total_cnot_logical'),
                        'logical_qubits': rep.get('logical_qubits'),
                        'physical_qubits_est': rep.get('physical_qubits_est'),
                    })
                # write CSV and optionally plot
                if rows:
                    write_compact_csv(args.out, rows)
                    if args.plot:
                        outpref = os.path.splitext(args.out)[0]
                        plot_parts_summary(rows, outpref)
                return
        if t['kind'] == 'auto':
            print('Auto-encoding cipher:', t['cipher'])
            # fixed example plaintext/ciphertext (user can edit script or call encoder themselves)
            plaintext_hex = '00112233445566778899aabbccddeeff'
            ciphertext_hex = '00' * (16 if t['cipher'] == 'aes128' else 32 // 2)
            plaintext = bytes.fromhex(plaintext_hex)
            ciphertext = bytes.fromhex(ciphertext_hex)
            master_key_vars = list(range(1, (128 if t['cipher'] == 'aes128' else 256) + 1))
            try:
                clauses, Nvars = run_auto_encode(t['cipher'], plaintext, ciphertext, master_key_vars, fast=args.fast, max_clauses=args.max_clauses, force=args.force)
            except Exception as e:
                print('Auto-encode failed:', e)
                continue
            report = estimate_from_clauses(Nvars, clauses, qlto_extra_ancilla=(args.qlto_ancilla or estimate_qlto_ancilla_count()), sequential=args.sequential, phys_per_logical=args.phys_per_logical, qaoa_layers=args.qaoa_layers)
            # Grover baseline for that key size
            keybits = 128 if t['cipher'] == 'aes128' else 256
            grover_iters = grover_iterations(keybits)
            grover_total_oracle_cnot = grover_iters * args.oracle_cnot
            grover_phys = args.oracle_logical * args.oracle_phys_per_logical
            rows.append({
                'approach': f"QLTO_{t['cipher']}_auto",
                'keybits': keybits,
                'n_vars': report['n_vars'],
                'n_clauses': report['n_clauses'],
                'pauli_weight_hist': report['pauli_weight_hist'],
                'cnot_per_layer': report['cnot_per_layer'],
                'qaoa_layers': report['qaoa_layers'],
                'total_cnot_logical': report['total_cnot_logical'],
                'logical_qubits': report['logical_qubits'],
                'physical_qubits_est': report['physical_qubits_est'],
                'grover_iters_approx': format_sci(grover_iters),
                'grover_total_oracle_cnot': format_sci(grover_total_oracle_cnot),
                'grover_physical_per_machine': grover_phys,
            })
        elif t['kind'] == 'cnf':
            # lightweight notifier and optional tqdm
            def knotify(msg):
                print('[knotify]', msg)

            print('Loading CNF:', t['cnf'])
            try:
                # If partitioning requested, load full CNF object here and run partitioning logic
                if args.partition:
                    # load CNF object
                    knotify('Starting CNF load...')
                    t_load = time.time()
                    if t['cnf'].endswith('.gz') or t['cnf'].endswith('.pkl.gz'):
                        with gzip.open(t['cnf'], 'rb') as f:
                            obj = pickle.load(f)
                    else:
                        with open(t['cnf'], 'rb') as f:
                            obj = pickle.load(f)
                    knotify(f'CNF loaded in {time.time()-t_load:.1f}s; processing object')
                    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], list):
                        nvars_all = obj[0]
                        clauses_all = obj[1]
                    else:
                        clauses_all = obj
                        nvars_all = t.get('n_vars') or max(abs(l) for cl in clauses_all for l in cl)

                    # build variable graph
                    try:
                        import networkx as nx
                    except Exception:
                        print('Partitioning requires networkx; please install it or run without --partition')
                        continue

                    # try to import tqdm for progress bars; fall back to identity
                    try:
                        from tqdm import tqdm as _tqdm
                        def safe_tqdm(it, **kw):
                            return _tqdm(it, **kw)
                    except Exception:
                        def safe_tqdm(it, **kw):
                            return it

                    def build_var_graph(clauses, use_clique=True):
                        knotify('Building variable graph (this may be slow)')
                        tbg = time.time()
                        G = nx.Graph()
                        for cl in safe_tqdm(clauses, desc='graph:clauses'):
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
                        knotify(f'Built variable graph in {time.time()-tbg:.1f}s; nodes={G.number_of_nodes()} edges={G.number_of_edges()}')
                        return G

                    G = build_var_graph(clauses_all, use_clique=bool(args.partition_use_clique))

                    # partition graph
                    parts = None
                    knotify('Partitioning graph (method=%s)' % args.partition_method)
                    tpart = time.time()
                    if args.partition_method == 'louvain':
                        try:
                            import community as community_louvain
                            knotify('Running python-louvain best_partition (this may take a while)...')
                            t_louv = time.time()
                            partmap = community_louvain.best_partition(G)
                            t_louv_elapsed = time.time() - t_louv
                            # attempt to compute modularity if available
                            modularity = None
                            try:
                                modularity = community_louvain.modularity(partmap, G)
                            except Exception:
                                try:
                                    # some versions expect a dict mapping community->set
                                    modularity = community_louvain.modularity({v: c for v, c in partmap.items()}, G)
                                except Exception:
                                    modularity = None
                            communities = {}
                            for v, c in partmap.items():
                                communities.setdefault(c, []).append(v)
                            parts = list(communities.values())
                            # report verbose stats: number of communities, top sizes
                            sizes = sorted([len(c) for c in parts], reverse=True)
                            knotify(f'Louvain finished in {t_louv_elapsed:.1f}s; communities={len(parts)}; largest_sizes={sizes[:10]}; modularity={modularity}')
                        except Exception:
                            knotify('Louvain partitioning failed or is not available; falling back to greedy_modularity_communities')
                            parts = [list(c) for c in nx.algorithms.community.greedy_modularity_communities(G)]
                    elif args.partition_method == 'greedy':
                        parts = [list(c) for c in nx.algorithms.community.greedy_modularity_communities(G)]
                    else:
                        # components (connected components)
                        parts = [list(c) for c in nx.connected_components(G)]
                    knotify(f'Partitioning finished in {time.time()-tpart:.1f}s; found {len(parts)} parts')

                    # compute per-part clauses and separator variables
                    var2part = {}
                    for i, pset in enumerate(parts):
                        for v in pset:
                            var2part[v] = i

                    part_clauses = {i: [] for i in range(len(parts))}
                    separators = set()
                    for cl in clauses_all:
                        vars_in = sorted({abs(int(l)) - 1 for l in cl})
                        touched = set(var2part.get(v, -1) for v in vars_in)
                        if len(touched) == 1 and -1 not in touched:
                            part_idx = next(iter(touched))
                            part_clauses[part_idx].append(cl)
                        else:
                            for v in vars_in:
                                if v in var2part:
                                    separators.add(v)

                    # optionally write per-part compact CNF files
                    write_parts_dir = args.write_parts or ''
                    if write_parts_dir:
                        os.makedirs(write_parts_dir, exist_ok=True)

                    # If user requested part-only-write, dump gzipped compact parts and skip in-process estimation
                    if args.part_only_write:
                        for pid, cls in part_clauses.items():
                            if not cls:
                                continue
                            nvars_part = 0
                            for cl in cls:
                                for lit in cl:
                                    nvars_part = max(nvars_part, abs(int(lit)))
                            part_path = os.path.join(write_parts_dir or 'benchmarks/parts', f'part_{pid}_cnf_compact.pkl.gz')
                            try:
                                with gzip.open(part_path, 'wb') as pf:
                                    pickle.dump((nvars_part, cls), pf)
                                print('WROTE part compact CNF ->', part_path)
                            except Exception as e:
                                print('Failed to write part file', part_path, e)
                        print('Completed part-only write; exiting as requested')
                        return

                    # run estimator per part (optionally skip large parts)
                    # iterate parts with progress indicator
                    for pid, cls in safe_tqdm(list(part_clauses.items()), desc='parts'):
                        if not cls:
                            continue
                        nvars_part = 0
                        for cl in cls:
                            for lit in cl:
                                nvars_part = max(nvars_part, abs(int(lit)))

                        # write part file if requested
                        if write_parts_dir:
                            part_path = os.path.join(write_parts_dir, f'part_{pid}_cnf_compact.pkl.gz')
                            try:
                                twrite = time.time()
                                with gzip.open(part_path, 'wb') as pf:
                                    pickle.dump((nvars_part, cls), pf)
                                knotify(f'WROTE part compact CNF -> {part_path} ({time.time()-twrite:.1f}s)')
                            except Exception as e:
                                print('Failed to write part file', part_path, e)

                        # skip estimation for parts larger than threshold if requested
                        if args.part_max_vars and nvars_part > args.part_max_vars:
                            print(f'Skipping part {pid} (n_vars={nvars_part}) > part-max-vars={args.part_max_vars}')
                            rows.append({
                                'approach': f'QLTO_cnf_part_{pid}',
                                'part_id': pid,
                                'part_n_vars': nvars_part,
                                'part_n_clauses': len(cls),
                                'separator_size': len(separators),
                                'skipped_reason': f'n_vars>{args.part_max_vars}'
                            })
                            continue

                        # estimate using existing estimator
                        knotify(f'Estimating part {pid} (n_vars={nvars_part}, n_clauses={len(cls)})')
                        t_est = time.time()
                        report_p = estimate_from_clauses(n_vars=nvars_part, clauses=cls, qlto_extra_ancilla=(args.qlto_ancilla or estimate_qlto_ancilla_count()), sequential=args.sequential, phys_per_logical=args.phys_per_logical, qaoa_layers=args.qaoa_layers)
                        knotify(f'Part {pid} estimate completed in {time.time()-t_est:.1f}s')
                        rows.append({
                            'approach': f'QLTO_cnf_part_{pid}',
                            'part_id': pid,
                            'part_n_vars': report_p['n_vars'],
                            'part_n_clauses': report_p['n_clauses'],
                            'separator_size': len(separators),
                            'pauli_weight_hist': report_p['pauli_weight_hist'],
                            'cnot_per_layer': report_p['cnot_per_layer'],
                            'total_cnot_logical': report_p['total_cnot_logical'],
                            'logical_qubits': report_p['logical_qubits'],
                            'physical_qubits_est': report_p['physical_qubits_est'],
                        })
                    # done with partitioning task
                    continue
                else:
                    report = run_estimate_for_cnf(t['cnf'], n_vars=t.get('n_vars'), qlto_ancilla=args.qlto_ancilla, phys_per_logical=args.phys_per_logical, qaoa_layers=args.qaoa_layers, sequential=args.sequential)
            except Exception as e:
                print('Failed to load CNF:', e)
                continue
            # decide keybits heuristic: if n_vars >= 13248 treat as AES-128, if larger treat AES-256? we keep keybits unspecified
            keybits = None
            if report['n_vars'] >= 13248:
                keybits = 128
            rows.append({
                'approach': f"QLTO_cnf_{os.path.basename(t['cnf'])}",
                'keybits': keybits,
                'n_vars': report['n_vars'],
                'n_clauses': report['n_clauses'],
                'pauli_weight_hist': report['pauli_weight_hist'],
                'cnot_per_layer': report['cnot_per_layer'],
                'qaoa_layers': report['qaoa_layers'],
                'total_cnot_logical': report['total_cnot_logical'],
                'logical_qubits': report['logical_qubits'],
                'physical_qubits_est': report['physical_qubits_est'],
            })

    # write CSV
    if rows:
        write_compact_csv(args.out, rows)

if __name__ == '__main__':
    main()
