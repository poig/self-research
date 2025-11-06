"""Attempt to build full 10-round AES Hamiltonian and export canonical pickled pauli-list.

This may be heavy — if it's too slow you can stop it. It uses
`src.solvers.aes_full_encoder.encode_aes_128` + `src.solvers.qlto_qaoa_sat.export_canonical_hamiltonian`.

python benchmarks/export_aes10_ham.py --force
python benchmarks/export_aes10_ham.py --fast --max-clauses 2000
"""
import os
import sys
import pickle
import argparse
import sqlite3
import glob
import gzip
import time
# Optional progress bar
try:
    import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

sys.path.insert(0, '.')

import src.solvers.aes_full_encoder as enc_mod
from src.solvers.qlto_qaoa_sat import export_canonical_hamiltonian, SATProblem, SATClause, sat_to_hamiltonian


def expand_clause_projector(clause):
    """Expand a single CNF clause into projector Pauli terms (pairs, projected_count)."""
    clause_projector_terms = {(): (dict(), 1.0)}
    for lit in clause:
        var_idx = abs(lit) - 1
        literal_op = {'I': 0.5, 'Z': 0.5 if lit > 0 else -0.5}
        new_projector = {}
        for op_label, op_coeff in literal_op.items():
            for _, (existing_pauli_dict, existing_coeff) in clause_projector_terms.items():
                new_pauli = existing_pauli_dict.copy()
                if op_label != 'I':
                    cur_op = new_pauli.get(var_idx, 'I')
                    if cur_op == 'I':
                        new_pauli[var_idx] = 'Z'
                    elif cur_op == 'Z':
                        new_pauli.pop(var_idx, None)

                pauli_items = tuple(sorted(new_pauli.keys()))
                new_coeff = existing_coeff * op_coeff
                if pauli_items in new_projector:
                    new_projector[pauli_items] = (new_pauli, new_projector[pauli_items][1] + new_coeff)
                else:
                    new_projector[pauli_items] = (new_pauli, new_coeff)

        clause_projector_terms = new_projector

    projected_count = max(1, len(clause_projector_terms))
    pairs = []
    for pauli_items, (_, coeff) in clause_projector_terms.items():
        if abs(coeff) < 1e-12:
            continue
        if not pauli_items:
            key = ''
        else:
            key = ','.join(str(i) for i in pauli_items)
        pairs.append((key, coeff))
    return pairs, projected_count


def _expand_chunk_worker(chunk, worker_db_path, flush_batch=5000):
    """Worker: expand clauses and write accumulated pairs into a local sqlite DB.
    Returns (worker_db_path, projected_term_count).
    """
    import sqlite3 as _sqlite
    try:
        if os.path.exists(worker_db_path):
            try:
                os.remove(worker_db_path)
            except Exception:
                pass
        wconn = _sqlite.connect(worker_db_path)
        wcur = wconn.cursor()
        wcur.execute('CREATE TABLE terms(pauli TEXT PRIMARY KEY, coeff REAL)')
        wconn.commit()

        local_batch = []
        proj = 0

        def wflush(batch):
            if not batch:
                return
            wcur.execute('BEGIN')
            for key, coeff in batch:
                wcur.execute(
                    'INSERT INTO terms(pauli, coeff) VALUES (?, ?) ON CONFLICT(pauli) DO UPDATE SET coeff = coeff + excluded.coeff',
                    (key, float(coeff))
                )
            wconn.commit()

        for clause in chunk:
            pairs, pcount = expand_clause_projector(clause)
            proj += pcount
            local_batch.extend(pairs)
            if len(local_batch) >= flush_batch:
                wflush(local_batch)
                local_batch = []

        if local_batch:
            wflush(local_batch)

        wconn.close()
        return worker_db_path, proj
    except Exception:
        return worker_db_path, 0

# Choose a fixed plaintext/ciphertext (example values)
plaintext_hex = "00112233445566778899aabbccddeeff"
ciphertext_hex = "69c4e0d86a7b0430d8cdb78070b4c55a"  # example - may be inconsistent

plaintext = bytes.fromhex(plaintext_hex)
ciphertext = bytes.fromhex(ciphertext_hex)

# allocate master key vars (1..128)
master_key_vars = list(range(1, 129))

# Add a small CLI so the user can opt into a fast/sample export or force the full export.
parser = argparse.ArgumentParser(description='Encode AES and (optionally) export canonical Hamiltonian.')
parser.add_argument('--fast', action='store_true', help='Export a small sampled Hamiltonian using the first --max-clauses clauses (safe).')
parser.add_argument('--max-clauses', type=int, default=2000, help='Maximum number of clauses to use for --fast (default: 2000).')
parser.add_argument('--force', action='store_true', help='Force a full canonical export even if the problem is large (may be extremely slow).')
parser.add_argument('--workers', type=int, default=0, help='Number of worker processes for parallel clause expansion (0 = disabled).')
parser.add_argument('--batch-size', type=int, default=2000, help='Batch size (number of projector terms) to flush to the DB (default: 2000).')
parser.add_argument('--chunk-size', type=int, default=0, help='Chunk size (#clauses) passed to each worker. If 0 (default), uses batch_size*8.')
parser.add_argument('--checkpoint-every', type=int, default=0, help='Write an automatic partial checkpoint every N clauses (0 = disabled).')
parser.add_argument('--checkpoint-seconds', type=int, default=0, help='Write an automatic partial checkpoint every N seconds (0 = disabled).')
args = parser.parse_args()

parser.add_argument('--cipher', choices=['aes128', 'aes256'], default='aes128', help='Which AES cipher to encode (aes128 or aes256).')
parser.add_argument('--out-prefix', type=str, default=None, help='Optional output prefix to use for all generated files (overrides automatic prefix).')
parser.add_argument('--sbox-mode', choices=['naive','tseitin','espresso'], default='naive', help='S-box encoding mode forwarded to encoder (default: naive).')
parser.add_argument('--gadgetize', action='store_true', help='Convenience flag: enable S-box gadgetization (sets --sbox-mode=tseitin).')

# Note: parser.parse_args() was already called above; reparse to include new args
args = parser.parse_args()

print('Encoding full AES CNF (this can be slow and large) ...')

# Call appropriate encoder depending on selected cipher
if args.cipher == 'aes256':
    master_key_vars = list(range(1, 256 + 1))
    if hasattr(enc_mod, 'encode_aes_256'):
        ret = enc_mod.encode_aes_256(plaintext, ciphertext, master_key_vars, sbox_mode=args.sbox_mode)
    elif hasattr(enc_mod, 'encode_aes'):
        ret = enc_mod.encode_aes(plaintext, ciphertext, master_key_vars, key_bits=256, sbox_mode=args.sbox_mode)
    else:
        raise RuntimeError('AES-256 encoder not found in src.solvers.aes_full_encoder')
    out_prefix = 'aes10_aes256'
else:
    master_key_vars = list(range(1, 129))
    if hasattr(enc_mod, 'encode_aes_128'):
        ret = enc_mod.encode_aes_128(plaintext, ciphertext, master_key_vars, sbox_mode=args.sbox_mode)
    elif hasattr(enc_mod, 'encode_aes'):
        ret = enc_mod.encode_aes(plaintext, ciphertext, master_key_vars, key_bits=128, sbox_mode=args.sbox_mode)
    else:
        raise RuntimeError('AES-128 encoder not found in src.solvers.aes_full_encoder')
    out_prefix = 'aes10'

# Normalize encoder return signature
if isinstance(ret, tuple) and len(ret) == 3:
    clauses, Nvars, round_keys = ret
else:
    try:
        round_keys, clauses, next_var_id = ret
        Nvars = next_var_id
    except Exception:
        raise RuntimeError('Unexpected return signature from AES encoder')

print('Encoded: clauses=', len(clauses), 'vars=', Nvars)

outdir = os.path.join('benchmarks', 'out')
os.makedirs(outdir, exist_ok=True)
# If the user supplied an explicit output prefix, use it to avoid filename collisions
if getattr(args, 'out_prefix', None):
    out_prefix = args.out_prefix

# If user requested gadgetization shorthand, enable tseitin sbox mode
if getattr(args, 'gadgetize', False):
    print('GADGETIZE: setting sbox-mode = tseitin')
    args.sbox_mode = 'tseitin'

cnf_path = os.path.join(outdir, f'{out_prefix}_clauses.pkl')
with open(cnf_path, 'wb') as f:
    pickle.dump(clauses, f)
print('Wrote CNF pickle to', cnf_path)

# Also write a compact gzipped clause-ham so the estimator can load it directly
compact_path = os.path.join(outdir, f'{out_prefix}_clauses_compact.pkl.gz')
try:
    with gzip.open(compact_path, 'wb') as f:
        pickle.dump((Nvars, clauses), f)
    print('Wrote compact gzipped CNF for estimator to', compact_path)
except Exception:
    pass

# Try to export canonical Hamiltonian using the qlto exporter
ham_path = os.path.join(outdir, f'{out_prefix}_ham_canonical.pkl')
print('Attempting export via export_canonical_hamiltonian(...)')


def export_canonical_via_stream(clauses, n_vars, out_path, progress_every=50000, batch_size=2000, workers=0, chunk_size=None):
    """
    Stream-export a canonical Hamiltonian from a large clause list without
    keeping the full pauli dict in memory. Accumulates term coefficients in
    an on-disk sqlite DB (pauli key -> coeff) and finally writes a gzipped
    pickle of the canonical list [(coeff, (q0,q1,...)), ...].
    """
    start = time.time()
    db_path = out_path + '.tmp.sqlite'
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('CREATE TABLE terms(pauli TEXT PRIMARY KEY, coeff REAL)')
    conn.commit()

    def upsert_pairs(pairs):
        if not pairs:
            return
        cur.execute('BEGIN')
        for key, coeff in pairs:
            # Use UPSERT to accumulate coeffs
            cur.execute(
                'INSERT INTO terms(pauli, coeff) VALUES (?, ?) ON CONFLICT(pauli) DO UPDATE SET coeff = coeff + excluded.coeff',
                (key, float(coeff))
            )
        conn.commit()

    def write_partial_dump(suffix='.partial.pkl.gz'):
        """Read current DB and write a gzipped partial canonical pickle."""
        try:
            cur.execute('SELECT pauli, coeff FROM terms')
            rows = cur.fetchall()
            terms = []
            for pauli_key, coeff in rows:
                if abs(coeff) < 1e-12:
                    continue
                if pauli_key == '':
                    qs = ()
                else:
                    qs = tuple(int(x) for x in pauli_key.split(','))
                terms.append((float(coeff), qs))

            outf = out_path + suffix
            with gzip.open(outf, 'wb') as f:
                pickle.dump(terms, f)
            print(f"WROTE partial canonical dump ({len(terms)} terms) -> {outf}")
            return outf
        except Exception as e:
            print('Failed to write partial dump:', e)
            return None

    total = len(clauses)
    processed = 0
    last_checkpoint_time = time.time()
    batch = []
    # Estimate projected-term upper bound: each clause with k literals expands to up to 2^k projector terms
    est_terms = 0
    est_capped = False
    CAP_EXP = 40
    for c in clauses:
        k = len(c)
        if k > CAP_EXP:
            est_terms += (1 << CAP_EXP)
            est_capped = True
        else:
            est_terms += (1 << k)

    if est_capped:
        print(f"Estimated upper-bound projector terms: {est_terms} (capped per-clause at 2^{CAP_EXP})")
    else:
        print(f"Estimated upper-bound projector terms: {est_terms}")
    projected_terms_total = 0

    if HAVE_TQDM:
        # Use projected-term estimate as the progress bar total so the
        # percentage reflects the expanded work (projector terms), not
        # the input clause count which is much smaller.
        try:
            pbar_total = est_terms if est_terms > 0 else total
        except Exception:
            pbar_total = total
        pbar = tqdm.tqdm(total=pbar_total, desc='Export proj_terms', unit='term')
    else:
        pbar = None

    # If workers requested, run a per-worker DB parallel expansion and merge
    def _merge_worker_dbs():
        """Attach and merge any worker DB files matching the temp pattern."""
        merged = 0
        pattern = db_path + '.worker.*.sqlite'
        for worker_db in glob.glob(pattern):
            try:
                safe_path = worker_db.replace("'", "''")
                cur.execute(f"ATTACH DATABASE '{safe_path}' AS wdb")
                cur.execute("INSERT INTO terms(pauli, coeff) SELECT pauli, coeff FROM wdb.terms ON CONFLICT(pauli) DO UPDATE SET coeff = coeff + excluded.coeff")
                conn.commit()
                cur.execute("DETACH DATABASE wdb")
                try:
                    os.remove(worker_db)
                except Exception:
                    pass
                merged += 1
            except Exception as e:
                print('Failed to merge worker DB', worker_db, e)
        return merged

    if workers and workers > 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if chunk_size is None or chunk_size <= 0:
            chunk_size = max(1, batch_size * 8)
        futures = []
        idx = 0
        chunk = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for clause in clauses:
                chunk.append(clause)
                if len(chunk) >= chunk_size:
                    worker_db = db_path + f'.worker.{idx}.sqlite'
                    futures.append(ex.submit(_expand_chunk_worker, chunk, worker_db))
                    idx += 1
                    chunk = []
            if chunk:
                worker_db = db_path + f'.worker.{idx}.sqlite'
                futures.append(ex.submit(_expand_chunk_worker, chunk, worker_db))

            # Merge worker DBs as they finish
            for fut in as_completed(futures):
                try:
                    worker_db, proj_count = fut.result()
                except Exception as e:
                    print('Worker failed:', e)
                    continue
                projected_terms_total += proj_count
                # Merge using ATTACH/INSERT to leverage SQLite performance
                try:
                    safe_path = worker_db.replace("'", "''")
                    cur.execute(f"ATTACH DATABASE '{safe_path}' AS wdb")
                    cur.execute("INSERT INTO terms(pauli, coeff) SELECT pauli, coeff FROM wdb.terms ON CONFLICT(pauli) DO UPDATE SET coeff = coeff + excluded.coeff")
                    conn.commit()
                    cur.execute("DETACH DATABASE wdb")
                    try:
                        os.remove(worker_db)
                    except Exception:
                        pass
                except Exception as e:
                    print('Failed to merge worker DB', worker_db, e)
                    try:
                        wconn = sqlite3.connect(worker_db)
                        wcur = wconn.cursor()
                        wcur.execute('SELECT pauli, coeff FROM terms')
                        rows = wcur.fetchall()
                        wconn.close()
                        upsert_pairs(rows)
                    except Exception as e2:
                        print('Fallback merge also failed for', worker_db, e2)
                if pbar is not None:
                    try:
                        pbar.update(proj_count)
                        pbar.set_postfix({'proj_terms': projected_terms_total})
                    except Exception:
                        pass
        # Try to merge any remaining worker DBs (in case some finished earlier)
        _merge_worker_dbs()

        # Finalize early: worker path has produced the DB; skip main-thread expansion
        # Read back accumulated terms and write canonical list
        cur.execute('SELECT pauli, coeff FROM terms')
        rows = cur.fetchall()
        terms = []
        for pauli_key, coeff in rows:
            if abs(coeff) < 1e-12:
                continue
            if pauli_key == '':
                qs = ()
            else:
                qs = tuple(int(x) for x in pauli_key.split(','))
            terms.append((float(coeff), qs))

        conn.close()
        try:
            os.remove(db_path)
        except Exception:
            pass

        with gzip.open(out_path + '.gz', 'wb') as f:
            pickle.dump(terms, f)

        elapsed_total = time.time() - start
        print(f"Stream export (worker mode) complete: wrote {len(terms)} terms to {out_path+'.gz'} in {elapsed_total:.1f}s")
        return out_path + '.gz'

    try:
        for ci, clause in enumerate(clauses):
            # Build projector expansion for this clause (incrementally)
            # clause is iterable of ints
            clause_projector_terms = {(): (dict(), 1.0)}
            for lit in clause:
                var_idx = abs(lit) - 1
                literal_op = {'I': 0.5, 'Z': 0.5 if lit > 0 else -0.5}

                new_projector = {}
                for op_label, op_coeff in literal_op.items():
                    for existing_key, (existing_pauli_dict, existing_coeff) in clause_projector_terms.items():
                        new_pauli = existing_pauli_dict.copy()
                        if op_label != 'I':
                            cur_op = new_pauli.get(var_idx, 'I')
                            if cur_op == 'I':
                                new_pauli[var_idx] = 'Z'
                            elif cur_op == 'Z':
                                # Z*Z -> identity: remove entry
                                new_pauli.pop(var_idx, None)

                        pauli_items = tuple(sorted(new_pauli.keys()))
                        new_coeff = existing_coeff * op_coeff
                        if pauli_items in new_projector:
                            new_projector[pauli_items] = (new_pauli, new_projector[pauli_items][1] + new_coeff)
                        else:
                            new_projector[pauli_items] = (new_pauli, new_coeff)

                clause_projector_terms = new_projector

            # Convert clause_projector_terms to pauli-key strings and add to batch
            for pauli_items, (pauli_dict, coeff) in clause_projector_terms.items():
                if abs(coeff) < 1e-12:
                    continue
                # Represent key as comma-separated indices (empty string for identity)
                if not pauli_items:
                    key = ''
                else:
                    key = ','.join(str(i) for i in pauli_items)
                batch.append((key, coeff))

            # Update projected-term counter (number of projector terms produced by this clause)
            projected_terms_total += max(1, len(clause_projector_terms))

            processed += 1
            if len(batch) >= batch_size:
                upsert_pairs(batch)
            if pbar is not None:
                # batch contains projector-term entries; update progress by that amount
                pbar.update(len(batch))
                try:
                    pbar.set_postfix({'clauses': processed, 'proj_terms': projected_terms_total})
                except Exception:
                    pass
            batch = []

            # Periodic checkpointing: clause-count or time-based
            do_checkpoint = False
            if args.checkpoint_every and args.checkpoint_every > 0 and (processed % args.checkpoint_every == 0):
                do_checkpoint = True
            if args.checkpoint_seconds and args.checkpoint_seconds > 0 and (time.time() - last_checkpoint_time >= args.checkpoint_seconds):
                do_checkpoint = True
            if do_checkpoint:
                # Merge worker DBs first to ensure checkpoint includes worker contributions
                if workers and workers > 0:
                    try:
                        merged = _merge_worker_dbs()
                        if merged:
                            print(f'Merged {merged} worker DB(s) into main DB for checkpoint')
                    except Exception:
                        pass
                if batch:
                    upsert_pairs(batch)
                write_partial_dump(suffix=f'.checkpoint.{processed}.pkl.gz')
                last_checkpoint_time = time.time()

        # periodic progress/ETA when tqdm not present
        if pbar is None and (ci + 1) % progress_every == 0:
            elapsed = time.time() - start
            rate = processed / max(1.0, elapsed)
            remaining = max(0.0, (total - processed) / max(1e-9, rate))
            print(f"  Processed {ci+1}/{total} clauses (elapsed {elapsed:.1f}s) ETA {remaining:.1f}s; projected_terms={projected_terms_total}")

    except KeyboardInterrupt:
        print('\n*** KeyboardInterrupt detected: flushing and writing partial dump...')
        if batch:
            upsert_pairs(batch)
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        # If running with workers, try to merge worker DBs before writing a partial dump
        if workers and workers > 0:
            try:
                merged = _merge_worker_dbs()
                if merged:
                    print(f'Merged {merged} worker DB(s) into main DB before partial dump')
            except Exception:
                pass

        write_partial_dump(suffix='.partial.pkl.gz')
        conn.close()
        raise
    except Exception as e:
        print(f"\n*** ERROR during streaming export: {e} -- flushing and writing partial dump...")
        try:
            if batch:
                upsert_pairs(batch)
        except Exception:
            pass
        try:
            if pbar is not None:
                pbar.close()
        except Exception:
            pass
        write_partial_dump(suffix='.error.pkl.gz')
        conn.close()
        raise
    else:
        # Normal completion: flush remaining
        if batch:
            upsert_pairs(batch)
            if pbar is not None:
                pbar.update(len(batch))
                try:
                    pbar.set_postfix({'proj_terms': projected_terms_total})
                except Exception:
                    pass
        if pbar is not None:
            pbar.close()

    # Read back accumulated terms and write canonical list
    cur.execute('SELECT pauli, coeff FROM terms')
    rows = cur.fetchall()
    terms = []
    for pauli_key, coeff in rows:
        if abs(coeff) < 1e-12:
            continue
        if pauli_key == '':
            qs = ()
        else:
            qs = tuple(int(x) for x in pauli_key.split(','))
        terms.append((float(coeff), qs))

    # Cleanup DB
    conn.close()
    try:
        os.remove(db_path)
    except Exception:
        pass

    # Write gzipped pickle to out_path
    with gzip.open(out_path + '.gz', 'wb') as f:
        pickle.dump(terms, f)

    elapsed_total = time.time() - start
    print(f"Stream export complete: wrote {len(terms)} terms to {out_path+'.gz'} in {elapsed_total:.1f}s")
    return out_path + '.gz'

# Safety thresholds: avoid accidentally trying to expand a million-clause CNF
MAX_VARS_WARN = 2048
MAX_CLAUSES_WARN = 100000

if not args.force and (Nvars > MAX_VARS_WARN or len(clauses) > MAX_CLAUSES_WARN):
    print('\n- The problem appears very large (vars: {}, clauses: {}).'.format(Nvars, len(clauses)))
    print("- By default we will NOT attempt the full canonical Hamiltonian export to avoid huge memory/time usage.")
    print("- Options:")
    print("    * Re-run with --force to attempt the full export (may take extremely long and use lots of RAM).")
    print("    * Run the estimator directly on the CNF pickle instead:\n      python benchmarks/qlto_resource_estimator.py --cnf {}\n".format(cnf_path))
else:
    try:
        # Build a SATProblem object from the raw clause list so the exporter
        # can call the module's sat_to_hamiltonian(problem) correctly.
        # Ensure clauses are tuples
        if args.fast:
            # Sample a small prefix of clauses to build a lightweight Hamiltonian
            sample_k = min(args.max_clauses, len(clauses))
            print(f"--fast enabled: building Hamiltonian from first {sample_k} clauses (of {len(clauses)})")
            sampled = clauses[:sample_k]
            sat_clauses = [SATClause(tuple(c)) for c in sampled]
            problem = SATProblem(n_vars=Nvars, clauses=sat_clauses)
            fast_ham_path = ham_path.replace('.pkl', '_fast.pkl')
            export_canonical_hamiltonian(fast_ham_path, cnf=problem)
            print('WROTE sampled canonical ham to', fast_ham_path)
        else:
            # If --force was requested for a large problem, use the streamed exporter
            if args.force:
                print("--force: using streamed exporter with on-disk accumulation (shows progress).")
                out_gz = export_canonical_via_stream(clauses, Nvars, ham_path)
                print('WROTE canonical ham (streamed gz) to', out_gz)
            else:
                sat_clauses = [SATClause(tuple(c)) for c in clauses]
                problem = SATProblem(n_vars=Nvars, clauses=sat_clauses)

                # Try export via exporter which will call sat_to_hamiltonian(problem)
                export_canonical_hamiltonian(ham_path, cnf=problem)
                print('WROTE canonical ham to', ham_path)
    except Exception as e:
        # If export fails (e.g., Qiskit not installed or memory limits), show a
        # helpful message and leave the CNF pickle in place so the user can run
        # the estimator with --cnf instead.
        print('Failed to export canonical ham:', e)
        print('You can still run the estimator on the CNF via:')
        print('  python benchmarks/qlto_resource_estimator.py --cnf', cnf_path)

print('Done.')
