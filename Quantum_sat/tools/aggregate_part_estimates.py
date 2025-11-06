#!/usr/bin/env python3
"""Aggregate per-part QLTO estimates by running the estimator on each part CNF.

Writes CSV to `benchmarks/parts_estimates.csv`.

Usage:
  python tools/aggregate_part_estimates.py

Requires: Python in PATH and `tools/qlto_resource_estimator.py` available in repo root.
"""
import subprocess, glob, csv, re, os, sys

PARTS_GLOB = 'benchmarks/parts/part_*_cnf_compact.pkl.gz'
OUT_CSV = 'benchmarks/parts_estimates.csv'

parts = sorted(glob.glob(PARTS_GLOB))
if not parts:
    print('No parts found matching', PARTS_GLOB)
    sys.exit(1)

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['part_file','n_vars','n_clauses','cnot_per_layer','total_cnot_logical','logical_qubits','physical_qubits_est'])
    for p in parts:
        print('Estimating', p)
        r = subprocess.run(['python','benchmarks/qlto_resource_estimator.py','--cnf',p,'--phys-per-logical','1000','--qaoa-layers','2','--sequential'], capture_output=True, text=True)
        txt = r.stdout + '\n' + r.stderr
        # crude regex parsing (robustness depends on estimator output format)
        nvars = re.search(r"n_vars\W*[:=]\W*(\d+)", txt)
        nclauses = re.search(r"n_clauses\W*[:=]\W*(\d+)", txt)
        cnot = re.search(r"cnot_per_layer\W*[:=]\W*([\d,]+)", txt)
        tot = re.search(r"total_cnot_logical\W*[:=]\W*([\d,]+)", txt)
        lq = re.search(r"logical_qubits\W*[:=]\W*(\d+)", txt)
        pq = re.search(r"physical_qubits_est\W*[:=]\W*([\d,]+)", txt)
        row = [os.path.basename(p),
               int(nvars.group(1)) if nvars else '',
               int(nclauses.group(1)) if nclauses else '',
               int(cnot.group(1).replace(',','')) if cnot else '',
               int(tot.group(1).replace(',','')) if tot else '',
               int(lq.group(1)) if lq else '',
               int(pq.group(1).replace(',','')) if pq else '']
        w.writerow(row)
print('WROTE', OUT_CSV)
