"""
Scope of the "W regresses through the origin on P(anc=1), but not on I(S:A)"
claim, computed from harmonized_sweep.csv.

The unrestricted form ("the P intercept is always smaller") is false.  This
script finds the restriction under which it holds without exception, so the
paper can state the claim with its domain attached rather than have a referee
find the counterexamples in the CSV.
"""

import csv

import numpy as np

R2_GATE = 0.60
INT_FLOOR = 0.002


def load(path="harmonized_sweep.csv"):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["degenerate"].lower() == "true" or r["r2_I"] in ("", "nan"):
            continue
        rows.append(dict(
            init=r["init"], N=int(r["N"]), family=r["family"],
            r2_I=float(r["r2_I"]), r2_P=float(r["r2_P"]),
            int_I=float(r["intercept_I"]), int_P=float(r["intercept_P"]),
            maxW=float(r["max_absW"]),
        ))
    return rows


def main():
    rows = load()
    print(f"non-degenerate rows: {len(rows)}\n")

    smaller = [r for r in rows if abs(r["int_P"]) < abs(r["int_I"])]
    print(f"UNRESTRICTED  |int_P| < |int_I| : {len(smaller)} / {len(rows)}")
    bad = [r for r in rows if abs(r["int_P"]) >= abs(r["int_I"])]
    print(f"  counterexamples: {len(bad)}")
    print(f"  {'init':>5} {'N':>2} {'family':>16} {'R2_I':>7} {'int_I':>9} {'int_P':>9}")
    for r in sorted(bad, key=lambda x: (x["init"], x["N"])):
        print(f"  {r['init']:>5} {r['N']:>2} {r['family']:>16} {r['r2_I']:>7.3f} "
              f"{r['int_I']:>+9.4f} {r['int_P']:>+9.4f}")

    sub = [r for r in rows if r["r2_I"] >= R2_GATE and abs(r["int_I"]) > INT_FLOOR]
    ok = [r for r in sub if abs(r["int_P"]) < abs(r["int_I"])]
    print(f"\nRESTRICTED  (R2_I >= {R2_GATE} and |int_I| > {INT_FLOOR})")
    print(f"  |int_P| < |int_I| : {len(ok)} / {len(sub)}")
    print(f"  {'init':>5} {'N':>2} {'family':>16} {'R2_I':>7} {'R2_P':>7} "
          f"{'int_I':>9} {'int_P':>9} {'factor':>8}")
    factors = []
    for r in sorted(sub, key=lambda x: (x["init"], x["N"])):
        f = abs(r["int_I"]) / max(abs(r["int_P"]), 1e-12)
        factors.append(f)
        print(f"  {r['init']:>5} {r['N']:>2} {r['family']:>16} {r['r2_I']:>7.3f} "
              f"{r['r2_P']:>7.3f} {r['int_I']:>+9.4f} {r['int_P']:>+9.4f} {f:>8.1f}")
    print(f"\n  median factor: {np.median(factors):.1f}x    "
          f"min: {min(factors):.1f}x    max: {max(factors):.1f}x")

    print("\nMATCHED-N INIT PAIRS  (same H construction, only the product init differs)")
    print(f"  {'family':>16} {'N':>2} {'R2_I |+>':>10} {'R2_I |0>':>10}")
    by = {(r["family"], r["N"], r["init"]): r for r in rows}
    for fam in ["paper-fig1", "ordered", "chaotic"]:
        for n in range(3, 8):
            a, b = by.get((fam, n, "plus")), by.get((fam, n, "zero"))
            if a and b:
                print(f"  {fam:>16} {n:>2} {a['r2_I']:>10.3f} {b['r2_I']:>10.3f}")


if __name__ == "__main__":
    main()
