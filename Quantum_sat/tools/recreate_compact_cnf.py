#!/usr/bin/env python3
import pickle, gzip, os, sys
pkl_path = os.path.join('benchmarks', 'out', 'aes10_clauses.pkl')
out_path = os.path.join('benchmarks', 'out', 'aes10_clauses_compact.pkl.gz')
if not os.path.exists(pkl_path):
    print('Source CNF pickle not found:', pkl_path)
    sys.exit(2)
print('Loading', pkl_path)
with open(pkl_path, 'rb') as f:
    clauses = pickle.load(f)
# compute n_vars as max absolute literal index
maxv = 0
for cl in clauses:
    for lit in cl:
        maxv = max(maxv, abs(int(lit)))
Nvars = maxv
print(f'Found {len(clauses)} clauses, estimated Nvars={Nvars}')
print('Writing compact gz:', out_path)
with gzip.open(out_path, 'wb') as g:
    pickle.dump((Nvars, clauses), g)
print('WROTE', out_path)
# print file size
print('Size:', os.path.getsize(out_path), 'bytes')
