# tools/ — quick guide for readers

This folder contains utility scripts, small experiment drivers, and lightweight validators used while developing the project. The intent of this README is to help an audience quickly find useful helpers and to explain which files are production utilities vs. tests/examples.

If you already moved test-scripts into `tests/`, this document explains what remains here and how to use the most important tools.

## What this folder contains (high level)

- Utilities and partitioning helpers (e.g., hypergraph partition driver, recursive partitioner).
- QUBO & CNF adapters for quick experiments and resource estimation.
- Small demos and example scripts (TSP pipeline, AES quick checks).
- A handful of short assertion/validation scripts — these were suggested to move to `tests/` to separate concerns.

## Key scripts (what to use first)

- `hypergraph_partition.py` — produce an hMetis-style `.hgr` from a compact CNF and optionally run an external partitioner (hMetis/PaToH). Use when you want hypergraph-based cuts rather than graph-projection.
- `recursive_partitioner.py` — recursively split large parts (uses hypergraph partitioner when provided, falls back to in-process partitioning). Useful for guaranteeing part sizes.
- `qubo_to_cnf_adapter.py` — heuristic adapter that converts a QUBO pickle into a compact CNF-like pickle suitable for `qlto_resource_estimator.py` (useful for pauli-weight estimates on QUBOs).
- `recreate_compact_cnf.py` — small helper to rebuild the compact `(n_vars, clauses)` pickles from raw CNF files.
- `tsp_pipeline.py`, `tsp_to_qubo.py`, `tsp_demo_compare.py` — example/demo tools that generate TSP QUBOs, partition them (k-means), and compare a classical OR-Tools baseline vs. the estimator pipeline.

## Usage snippets (copy-paste friendly)

Partition a gadgetized CNF with hMetis and write parts:

```powershell
python .\tools\hypergraph_partition.py --cnf benchmarks/out/aes10_gadget_clauses_compact.pkl.gz --parts 64 --out-dir benchmarks/parts_hmetis --partitioner C:\path\to\hmetis.exe
```

Recursive partitioning to enforce part sizes <= 1200 variables:

```powershell
python .\tools\recursive_partitioner.py --cnf benchmarks/out/aes10_gadget_clauses_compact.pkl.gz --max-vars 1200 --out-dir benchmarks/parts_recursive --partitioner C:\path\to\hmetis.exe --parts 4
```

Convert a QUBO part to compact CNF and estimate resources for that part:

```powershell
python .\tools\qubo_to_cnf_adapter.py --in benchmarks/out/tsp_part_0_qubo.pkl.gz --out benchmarks/out/tsp_part_0_cnf_compact.pkl.gz
python .\benchmarks\qlto_resource_estimator.py --cnf benchmarks/out/tsp_part_0_cnf_compact.pkl.gz --phys-per-logical 1000 --qaoa-layers 2
```

Run the TSP demo (generates instance, partitions, converts parts, runs OR-Tools baseline):

```powershell
python .\tools\tsp_demo_compare.py --n 100 --parts 8 --out-dir benchmarks/out/tsp_demo --run-estimator
```

## Tests vs Tools — suggested tidy-up

Some short scripts are primarily small checks or validators and are better suited in a `tests/` or `examples/` folder for clarity. If you prefer to tidy the repository I recommend moving these files into `tests/`:

- `test_1round_aes.py`
- `test_qlto_scale.py`
- `verify_qaoa_solution.py`
- `verify_sbox.py`
- `check_mixcolumns.py`
