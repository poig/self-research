# Benchmarks README

This folder contains helper scripts and tools used to convert SAT/CNF encodings into canonical Pauli Hamiltonians and to estimate QLTO/QAOA resources. Some operations (full canonical expansion) are extremely heavy for large CNFs — use the fast paths where possible.

## Files

- `export_aes10_ham.py` — Encoder + exporter for AES-128 (10 rounds)
  - Encodes AES-128 plaintext/ciphertext into a SAT CNF using `src.solvers.aes_full_encoder.encode_aes_128`.
  - Writes a raw CNF pickle: `benchmarks/out/aes10_clauses.pkl`.
  - Writes a compact gzipped CNF used by the estimator: `benchmarks/out/aes10_clauses_compact.pkl.gz` (tuple `(n_vars, clauses)`).
  - Optionally exports a full canonical Hamiltonian (list of `(coeff, tuple(qubit_indices))`) using a streamed exporter to avoid OOM:
    - Fast/sample mode: `--fast --max-clauses <k>` builds a sampled canonical ham using the first `k` clauses.
    - Full but streamed mode: `--force` attempts the full canonical expansion and writes `benchmarks/out/aes10_ham_canonical.pkl.gz`.
    - Parallel mode: `--workers N` enables per-worker sqlite DB expansion and merging (recommended when you want to utilize multiple cores). Default is `0` (single-threaded).
  - Partial-output: if interrupted (Ctrl+C) or on error, the exporter writes a partial dump `<out>.partial.pkl.gz` or `<out>.error.pkl.gz` containing whatever terms were accumulated.
  - Tuning tips: increase `--workers`, or change `batch_size`/`chunk_size` (edit script) to tune throughput. On Windows prefer fewer workers and large chunks. Use NVMe or RAM disk for DB files if possible.

- `qlto_resource_estimator.py` — Simple resource estimator
  - Accepts either a canonical Hamiltonian (`--ham <file>`), or a CNF (`--cnf <file>`).
  - Can load gzipped pickles (`.gz`) produced by `export_aes10_ham.py`.
  - If given a CNF, the estimator uses a conservative mapping of clause -> single Pauli term of weight equal to clause arity (lossy but fast) and prints a resource table including:
    - Variables (problem qubits), pauli-term histogram, CNOT cost per layer, logical qubits, and estimated physical qubits.
  - Example:

    ```bash
    python benchmarks/qlto_resource_estimator.py --cnf benchmarks/out/aes10_clauses_compact.pkl.gz --phys-per-logical 1000 --qaoa-layers 2 --sequential

    python benchmarks/qlto_resource_estimator.py --ham benchmarks/out/aes10_ham_canonical.pkl.gz --n-vars 13248
    ```

- `out/` — Output directory (created by scripts)
  - `aes10_clauses.pkl` — raw CNF pickled as a list of clauses (tuples of ints)
  - `aes10_clauses_compact.pkl.gz` — compact gzipped pickled `(n_vars, clauses)` for the estimator
  - `aes10_ham_canonical.pkl.gz` — (optional) gzipped canonical Hamiltonian produced by full expansion
  - `*.tmp.sqlite` — temporary DB files used during streaming expansion; worker DBs are named `*.worker.<i>.sqlite` and are deleted after merging.

## Quick workflow

1. Encode and export CNF (fast):

    ```bash
    python benchmarks/export_aes10_ham.py --fast --max-clauses 2000
    ```

  Notes (PowerShell): to run the same command in Windows PowerShell you can use a single-line call. Example:

  ```powershell
  python .\benchmarks\export_aes10_ham.py --fast --max-clauses 2000
  ```

    This writes sample Hamiltonian and compact CNF for quick testing.

2. Estimate QLTO resources from CNF (fast):

    ```bash
    python benchmarks/qlto_resource_estimator.py --cnf benchmarks/out/aes10_clauses_compact.pkl.gz
    ```

3. (Optional, slow) Produce canonical Hamiltonian for exact counts:

    ```bash
    python benchmarks/export_aes10_ham.py --force [--workers N]
    ```

  Important: to avoid output filename collisions between AES-128 and AES-256 runs use the `--out-prefix` flag to provide a unique prefix for all generated files. For example (PowerShell):

  ```powershell
  # AES-128 (explicit prefix)
  python .\benchmarks\export_aes10_ham.py --cipher aes128 --out-prefix aes10_128 --force --workers 4 --batch-size 10000 --chunk-size 100000

  # AES-256 (explicit prefix)
  python .\benchmarks\export_aes10_ham.py --cipher aes256 --out-prefix aes10_256 --force --workers 4 --batch-size 10000 --chunk-size 100000
  ```

  The exporter will still create files in `benchmarks/out` named like `<out-prefix>_clauses.pkl`, `<out-prefix>_clauses_compact.pkl.gz`, and `<out-prefix>_ham_canonical.pkl`. If you omit `--out-prefix` the exporter uses a cipher-aware default (`aes10` for AES-128 and `aes10_aes256` for AES-256), but providing a prefix is the safest way to avoid overwrites.

    If interrupted, inspect `benchmarks/out/aes10_ham_canonical.pkl.partial.pkl.gz`.

## Partitioning and per-part estimation (updated)

Note: the older helper scripts `tools/partition_and_estimate.py` and `benchmarks/plot_aes_estimates.py` have been deprecated — their functionality is now consolidated into `benchmarks/qlto_resource_estimator.py` which provides both partitioning/part-file handling and simple plotting. The legacy helpers remain in the repo for reference but we recommend using the unified estimator.

`qlto_resource_estimator.py` supports multiple workflows for partitioning and per-part estimation:

- In-process partitioning and per-part estimates (uses `networkx` and optional `python-louvain`): pass `--cnf` plus `--partition` and related flags.
- Write per-part compact CNF files for downstream processing with `--write-parts <dir>`.
- Batch-process existing per-part files (directory of part files) with `--parts-dir <dir>`; this runs the estimator over each part file, writes a combined CSV (`--out`) and can optionally create simple diagnostic plots with `--plot`.

Key flags (examples):

- `--partition` : run partitioning and per-part estimates in-process (requires `networkx`).
- `--partition-method {louvain,greedy,components}` : partitioning algorithm (default: `louvain`, falls back to greedy modularity when louvain isn't installed).
- `--partition-use-clique` : use clique projection when building the variable graph (denser graph; can be memory-heavy).
- `--write-parts <dir>`: write per-part compact CNF files into `<dir>` (named `part_<id>_cnf_compact.pkl.gz`).
- `--part-max-vars <N>`: only run in-process estimates for parts with <= N vars (0 = no limit). Large parts will be skipped and marked in the CSV.
- `--parts-dir <dir>`: batch-process all part files already written into `<dir>` (estimator accepts `.pkl` and `.pkl.gz` compact part files).
- `--parts-glob <pattern>`: glob pattern to match filenames inside `--parts-dir` (default `part_*_cnf*.pkl*`).
- `--plot` : produce simple diagnostic plots when using `--parts-dir` (requires `matplotlib` and `numpy`).

Example (PowerShell) — partition and write parts:

```powershell
python .\benchmarks\qlto_resource_estimator.py --cnf .\benchmarks\out\aes10_clauses_compact.pkl.gz --partition --write-parts .\benchmarks\parts --part-max-vars 800 --out .\benchmarks\aes10_128_parts.csv
```

Example (PowerShell) — batch-process existing parts and plot:

```powershell
python .\benchmarks\qlto_resource_estimator.py --parts-dir .\benchmarks\parts --out .\benchmarks\parts_estimates.csv --plot --phys-per-logical 1000 --qaoa-layers 2 --sequential
```

The batch `--parts-dir` mode replaces the earlier ad-hoc `Get-ChildItem | ForEach-Object` PowerShell loop. Use `--parts-glob` to match custom filenames.

If you still rely on the legacy helpers (`tools/partition_and_estimate.py`, `benchmarks/plot_aes_estimates.py`), they will continue to work for now, but we recommend switching to the unified estimator for a simpler, single-command workflow.

## Performance notes

- Full canonical expansion expands each clause into up to 2^k projector terms (k = clause arity). For large CNFs this becomes impractical.
- Streaming via per-worker sqlite DBs avoids keeping the entire Pauli dictionary in memory, and merging worker DBs avoids heavy IPC overhead.
- For best performance on Windows:
  - Use moderate worker counts (4–8) and large chunk sizes.
  - Prefer an NVMe or RAM disk for temporary worker DB files.
  - Use `--out-prefix` when running multiple experiments to keep outputs separate.
  - If you have fewer cores or a slower disk, reduce `--workers` and increase `--chunk-size` so each worker does more work between merges.
  - If you want a quick validation run before committing to a long run, use `--fast --max-clauses 2000` to create a compact CNF and test the estimator/plotting steps.

If you want, I can:
- Add CLI flags for `--batch-size` and `--chunk-size` to make tuning easier without editing the script.
- Add automatic periodic checkpoints and a resumable export mode.

---
Created by the exporter helper. If you want more detailed docs for any file, tell me which one and I'll expand it.

## Recommended "absolute optimal" flags (Windows)

For large AES CNFs on a modern Windows workstation (NVMe + 8+ logical cores) the following flags are a practical, near-optimal starting point that balances process overhead, SQLite commit frequency, and IO contention:

```bash
python benchmarks/export_aes10_ham.py --force \
  --workers 4 \
  --batch-size 10000 \
  --chunk-size 100000 \
  --checkpoint-every 20000 \
  --checkpoint-seconds 300

python benchmarks/export_aes10_ham.py --force  --workers 4  --batch-size 10000  --chunk-size 100000  --checkpoint-every 20000  --checkpoint-seconds 300
```

Why these values?
- `--workers 4`: moderate parallelism to utilize multiple cores while limiting Windows process-management overhead.
- `--batch-size 10000`: larger SQLite batches reduce commit frequency and CPU overhead inside workers.
- `--chunk-size 100000`: each worker handles a large clause block so spawn/merge costs are amortized.
- `--checkpoint-every 20000` and `--checkpoint-seconds 300`: periodic checkpoints provide recoverability without too-frequent IO.

If your machine has fewer cores or a slower disk, lower `--workers` and increase `--chunk-size` to keep per-worker work high. If you have a very fast NVMe/RAM disk and many cores, you can try `--workers 6` or `8` and increase batch/chunk sizes accordingly.

## End-to-end: run benchmark and plot (AES-128 and AES-256)

The pipeline is now unified: use `export_aes10_ham.py` to create CNFs and `qlto_resource_estimator.py` to perform estimation, partitioning, per-part writes, batch processing, and simple plotting. Below are compact PowerShell flows that show the recommended single-script usage.

1) Quick sample (safe) — produce a compact CNF for fast iteration

```powershell
python .\benchmarks\export_aes10_ham.py --fast --max-clauses 2000
```

This writes `benchmarks/out/<prefix>_clauses_compact.pkl.gz` (default prefix `aes10`).

2) Full streamed export (slow) — AES-128 and AES-256 with distinct prefixes

Use `--out-prefix` to avoid collisions between AES-128 and AES-256 outputs.

```powershell
# AES-128 (full streamed canonical expansion; may be very large)
python .\benchmarks\export_aes10_ham.py --cipher aes128 --out-prefix aes10_128 --force --workers 4 --batch-size 10000 --chunk-size 100000 --checkpoint-every 20000 --checkpoint-seconds 300

# AES-256 (will be much larger; use a different prefix)
python .\benchmarks\export_aes10_ham.py --cipher aes256 --out-prefix aes10_256 --force --workers 4 --batch-size 10000 --chunk-size 100000 --checkpoint-every 20000 --checkpoint-seconds 300
```

3) Partition and write per-part compact CNFs only (fast-ish compared to estimating large parts)

If you want to extract parts for offline processing or to run a hypergraph partitioner, use `--partition` with `--part-only-write` to write compact gzipped parts and exit.

```powershell
python .\benchmarks\qlto_resource_estimator.py --cnf .\benchmarks\out\aes10_128_clauses_compact.pkl.gz --partition --write-parts .\benchmarks\parts --part-only-write
```

This produces `benchmarks/parts/part_<id>_cnf_compact.pkl.gz` and then exits.

4) Batch-process per-part files and optionally plot diagnostics

After parts are written, run the unified estimator in batch mode to compute per-part estimates and write a combined CSV (and optional plots):

```powershell
python .\benchmarks\qlto_resource_estimator.py --parts-dir .\benchmarks\parts --out .\benchmarks\parts_estimates.csv --plot --phys-per-logical 1000 --qaoa-layers 2 --sequential
```

The command processes all `part_*_cnf*.pkl*` files in `benchmarks\parts` (use `--parts-glob` to change the pattern), writes `benchmarks\parts_estimates.csv`, and creates simple PNG diagnostics if `--plot` is given.

5) Single-row estimate of the full CNF (no partitioning)

```powershell
python .\benchmarks\qlto_resource_estimator.py --cnf .\benchmarks\out\aes10_128_clauses_compact.pkl.gz --phys-per-logical 1000 --qaoa-layers 2 --out .\benchmarks\aes10_128_estimate.csv
```

Optimal single-line PowerShell pipeline (generate → partition → estimate → plot)

Below is a concise, practical PowerShell-friendly sequence that does the common end-to-end flow: (A) export a gadgetized CNF (Tseitin S-box gadgets), (B) partition the CNF using the hypergraph driver if available (hMetis/PaToH), (C) write per-part compact CNFs, and (D) batch-run the estimator over parts and produce CSV + plots.

Replace the `C:\path\to\hmetis.exe` placeholder with your local hMetis/PaToH binary if you want the hypergraph partitioner to run automatically. If you don't have an external partitioner installed, the estimator's `--partition` mode can produce parts in-process (see fallback below).

```powershell
python .\benchmarks\export_aes10_ham.py --cipher aes128 --out-prefix aes10_gadget --sbox-mode tseitin ; \
python .\tools\hypergraph_partition.py --cnf .\benchmarks\out\aes10_gadget_clauses_compact.pkl.gz --parts 32 --out-dir .\benchmarks\parts --hmetis-bin C:\path\to\hmetis.exe ; \
python .\benchmarks\qlto_resource_estimator.py --parts-dir .\benchmarks\parts --out .\benchmarks\parts_estimates.csv --plot --phys-per-logical 1000 --qaoa-layers 2 --sequential
```

Why these flags?
- `--sbox-mode tseitin` gadgetizes AES S-boxes (trades ancilla vars + small clauses for much lower clause arity and dramatically improved partitionability).
- `--parts 32` is a practical starting point; tune up/down depending on target hardware and desired part granularity.
- `--phys-per-logical 1000` and `--qaoa-layers 2` are commonly-used estimate settings — change them to reflect your stack.

Fallback when no external partitioner is installed

If you don't have hMetis/PaToH installed, run the estimator's partitioner to write parts and then batch-process them. This uses the built-in graph projection partitioner (NetworkX + python-louvain if available):

```powershell
python .\benchmarks\export_aes10_ham.py --cipher aes128 --out-prefix aes10_gadget --sbox-mode tseitin ; \
python .\benchmarks\qlto_resource_estimator.py --cnf .\benchmarks\out\aes10_gadget_clauses_compact.pkl.gz --partition --write-parts .\benchmarks\parts --part-only-write ; \
python .\benchmarks\qlto_resource_estimator.py --parts-dir .\benchmarks\parts --out .\benchmarks\parts_estimates.csv --plot --phys-per-logical 1000 --qaoa-layers 2 --sequential
```

Practical tips
- Use `--out-prefix` for separate AES-128/AES-256 runs to avoid filename collisions.
- Start with `--parts 16` or `32` and inspect `benchmarks/parts` to ensure parts aren't too large; use `--part-max-vars` on the estimator to skip extremely large parts when producing CSVs.
- If you want a quick iteration before running a larger split, add `--fast --max-clauses 2000` to the export command (it writes a small sample CNF useful for tuning partitioning and plotting).

If you'd like, I can add a small PowerShell script `tools/run_one_click.ps1` that wraps this pipeline, detects a local hMetis/PaToH, and runs the appropriate flow automatically.

## New tools: gadgetize export, hypergraph partitioning, and recursive partitioning

I added three utilities to help produce practical parts for per-part QLTO runs:

- `--gadgetize` (flag on the exporter): a convenience flag which sets `--sbox-mode=tseitin` for `export_aes10_ham.py` and produces a gadgetized compact CNF.
- `tools/hypergraph_partition.py`: converts a compact CNF (`(n_vars, clauses)`) into an hMetis-style `.hgr`, optionally runs an external partitioner (hMetis/PaToH) and writes per-part compact CNFs.
- `tools/recursive_partitioner.py`: recursively partitions large parts (using hypergraph partitioner if provided, else an in-process star-projection + community detection) until each part has <= `--max-vars` variables; writes `part_<id>_cnf_compact.pkl.gz` files.

Below are copy-paste PowerShell command blocks that run these tools in a practical order. They are intentionally explicit so you can run them step-by-step and inspect outputs.

1) Export gadgetized CNF (fast sample to iterate quickly)

```powershell
python .\benchmarks\export_aes10_ham.py --cipher aes128 --out-prefix aes10_gadget --gadgetize --fast --max-clauses 2000

# Output: benchmarks/out/aes10_gadget_clauses_compact.pkl.gz
```

2) Hypergraph partition (optional — requires hMetis/PaToH installed)

```powershell
# If you have hMetis/PaToH installed, run the hypergraph partitioner (example: 64 parts)
python .\tools\hypergraph_partition.py --cnf .\benchmarks\out\aes10_gadget_clauses_compact.pkl.gz --parts 64 --out-dir .\benchmarks\parts_hmetis --partitioner C:\path\to\hmetis.exe

# The script writes .hgr and (if partitioner ran) part_<id>_cnf_compact.pkl.gz into --out-dir
```

3) Recursive partitioning to enforce max part size (recommended)

```powershell
# Use recursive partitioner to ensure every part has <= 1200 variables.
# Provide --partitioner if you want external hypergraph partitioning used for splits.
python .\tools\recursive_partitioner.py --cnf .\benchmarks\out\aes10_gadget_clauses_compact.pkl.gz --max-vars 1200 --out-dir .\benchmarks\parts_recursive --partitioner C:\path\to\hmetis.exe --parts 4

# Output: benchmarks/parts_recursive/part_<id>_cnf_compact.pkl.gz  (each part n_vars <= 1200)
```

4) Batch-process parts and plot diagnostics

```powershell
python .\benchmarks\qlto_resource_estimator.py --parts-dir .\benchmarks\parts_recursive --out .\benchmarks\parts_recursive_estimates.csv --plot --phys-per-logical 1000 --qaoa-layers 2 --sequential
```

Notes and tips
- If you don't provide a `--partitioner` path, the hypergraph script will only write the `.hgr` file for you to run externally; the recursive partitioner will fall back to an in-process star-projection + community detection strategy.
- Start with `--parts 4` or `8` in the recursive partitioner to split each large block into a few sub-blocks per iteration; increase if you want more aggressive splitting.
- If you want per-part CNFs with duplicated border clauses (self-contained parts), tell me and I'll add an option to duplicate separator clauses into the adjacent parts.
- The sample `--fast --max-clauses 2000` export is useful for tuning partition parameters before running the full export.

## TSP: quick QUBO encoder + estimator adapter

If you want to try the same decomposition + estimation workflow on small TSP instances, two helper scripts were added:

- `tools/tsp_to_qubo.py`: generate a small metric TSP instance (or read a CSV distance matrix) and produce a QUBO pickle: `(n_vars, linear_terms, quad_terms)`.
- `tools/qubo_to_cnf_adapter.py`: convert the QUBO pickle into a compact CNF-like gzipped pickle `(n_vars, clauses)` which the estimator can load directly for Pauli-weight / CNOT-cost estimation (heuristic mapping: quadratic terms -> 2-literal clauses).

Quick PowerShell example (generate a small random 6-city instance, convert, and estimate):

```powershell
python .\tools\tsp_to_qubo.py --n 6 --random --out benchmarks/out/tsp_n6_qubo.pkl.gz
python .\tools\qubo_to_cnf_adapter.py --in benchmarks/out/tsp_n6_qubo.pkl.gz --out benchmarks/out/tsp_n6_qubo_cnf_compact.pkl.gz
python .\benchmarks\qlto_resource_estimator.py --cnf benchmarks/out/tsp_n6_qubo_cnf_compact.pkl.gz --phys-per-logical 1000 --qaoa-layers 2
```

Notes:
- The `qubo_to_cnf_adapter.py` performs a heuristic translation intended for resource estimation only (it does not produce an exact SAT encoding of the QUBO objective). Use the generated compact CNF to get pauli-weight histograms and cnot-cost estimates for planning.
- For larger TSP instances prefer clustering + local-search hybrid strategies (cluster cities, solve clusters or k-opt neighborhoods via QLTO), as described in the playbook above.

