"""The open problem: a locally-routable design register in 2-D. Construct it.

TIER C - NO CIRCUIT. Exact combinatorics over GF(2).

THE PROBLEM, and it is the critical path to any hardware demonstration. The
design register needs columns c_j over k qubits satisfying THREE constraints
that were each introduced for a different reason:

    (a) RESOLUTION V      no GF(2) dependency among any 4 columns, so a
                          two-factor interaction is never confounded with a
                          main effect. Without it the Hessian aliases into the
                          gradient (measured: error grows to 87%).
    (b) CONSTANT WEIGHT   makes Phat constant on the design's characters, giving
                          EXACTLY ZERO directional error under any exchangeable
                          readout noise, for +1 qubit (measured: 0.00e+00).
    (c) CONNECTED SUPPORT on the device graph, so the parity CNOTs are all
                          nearest-neighbour. Without it routing blows up with k
                          (measured 4.7x, 6.2x, 8.7x on heavy-hex) and the
                          m=16 circuit reaches depth 1904 / 980 two-qubit gates,
                          which is no signal at all.

PROVED ALREADY (Part XXVII): ON A LINE THIS IS IMPOSSIBLE at log width. A
locally routable parity on a line is a CNOT ladder over an INTERVAL; in the
prefix basis [a,b] = p_b XOR p_{a-1}, and since prefixes are independent,
[a,b] XOR [c,d] is an interval IFF two of {a-1,b,c-1,d} coincide - verified
exhaustively, zero mismatches at k = 6, 8, 10. Resolution V therefore forces all
endpoints distinct, so 4m <= k+1 and k >= 4m-1: LINEAR, and the whole 2 log2 m
advantage is gone.

THE OPEN CASE. A device is not a line. On a bounded-degree lattice the available
supports are connected SUBGRAPHS, and the counting is different by an
exponential:

    intervals on a line of k sites                 ~ k^2 / 2
    connected subgraphs of size w, degree D        ~ k D^w

so the interval collapse argument does not run, and whether a log-width family
exists is genuinely open. This file tries to construct one.

WHAT WOULD SETTLE IT EITHER WAY:
    a family with m growing FASTER than linearly in k   -> the hardware route
                                                          reopens
    m stuck at Theta(k) on 2-D as well                 -> a general no-go:
                                                          log-width multiplexed
                                                          estimation cannot be
                                                          routed on bounded-degree
                                                          hardware, which applies
                                                          well beyond this project
"""
import sys
import itertools
import numpy as np


def grid_graph(L, degree4=True):
    """L x L lattice. degree4 = square grid; otherwise drop edges to get a
    degree-3 (heavy-hex-like) graph."""
    adj = {}
    idx = lambda r, c: r * L + c
    for r in range(L):
        for c in range(L):
            v = idx(r, c)
            adj.setdefault(v, set())
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 < L and c2 < L:
                    if not degree4 and dr == 0 and (r + c) % 2 == 1:
                        continue          # thin the horizontal edges -> deg 3
                    w = idx(r2, c2)
                    adj[v].add(w)
                    adj.setdefault(w, set()).add(v)
    return adj


def connected_subsets(adj, k, w, cap=200000):
    """All connected vertex subsets of size w, as bitmasks."""
    out = set()
    def grow(cur, frontier):
        if len(out) >= cap:
            return
        if bin(cur).count("1") == w:
            out.add(cur)
            return
        for v in list(frontier):
            if cur >> v & 1:
                continue
            nf = (frontier | adj.get(v, set())) - {v}
            grow(cur | (1 << v), nf)
    for s in range(k):
        grow(1 << s, set(adj.get(s, ())))
    return sorted(out)


def greedy_res_v(cands):
    """Largest family found greedily with NO GF(2) relation among any 4:
    (a) no column is the XOR of two others, (b) all pairwise XORs distinct."""
    chosen, pxor = [], set()
    for v in cands:
        new = {v ^ c for c in chosen}
        if v in pxor:
            continue
        if new & pxor:
            continue
        if len(new) != len(chosen):
            continue
        pxor |= new
        chosen.append(v)
    return chosen


def part1():
    print("PART 1  CONNECTED, CONSTANT-WEIGHT, RESOLUTION-V FAMILIES ON 2-D.")
    print("        m is what the family supports; the design needs 2m columns")
    print("        for a three-level register, so usable m = |family| / 2.")
    print("")
    print("   %4s %5s %4s %10s %9s %10s %14s %12s"
          % ("L", "k", "w", "connected", "family", "usable m", "1-D bound k/4",
             "verdict"))
    for L in (3, 4, 5):
        k = L * L
        adj = grid_graph(L, degree4=True)
        for w in (3, 4, 5):
            if w > k:
                continue
            cands = connected_subsets(adj, k, w)
            fam = greedy_res_v(cands)
            um = len(fam) // 2
            lin = k // 4
            print("   %4d %5d %4d %10d %9d %10d %14d %12s"
                  % (L, k, w, len(cands), len(fam), um, lin,
                     "BEATS 1-D" if um > lin else "at/below"))
    print("")
    print("   The 1-D bound column is k/4, the PROVED ceiling on a line.")
    print("   Anything above it is the 2-D lattice buying something a line")
    print("   cannot, which is the whole question.")
    print("")


def part2():
    print("PART 2  HOW DOES IT SCALE? m against k, on the square lattice.")
    print("        LOG WIDTH would mean m ~ 2^(k/2). LINEAR means m ~ k.")
    print("        The 1-D theorem gives m = k/4 exactly.")
    print("")
    print("   %4s %6s %4s %10s %12s %14s"
          % ("L", "k", "w", "usable m", "m / k", "m / 2^(k/2)"))
    xs, ys = [], []
    for L in (3, 4, 5, 6):
        k = L * L
        adj = grid_graph(L, degree4=True)
        best, bw = 0, 0
        for w in (3, 4, 5):
            if w > k:
                continue
            fam = greedy_res_v(connected_subsets(adj, k, w))
            if len(fam) // 2 > best:
                best, bw = len(fam) // 2, w
        xs.append(np.log(k)); ys.append(np.log(max(best, 1)))
        print("   %4d %6d %4d %10d %12.4f %14.3g"
              % (L, k, bw, best, best / k, best / 2.0 ** (k / 2)))
    sl = float(np.polyfit(xs, ys, 1)[0])
    print("")
    print("   SLOPE of log m on log k = %.4f" % sl)
    print("   slope ~ 1  -> LINEAR, same as the line: a general no-go.")
    print("   slope > 1  -> the 2-D lattice beats the 1-D ceiling and the")
    print("                 hardware route reopens.")
    print("")


def part3():
    print("PART 3  DEGREE 3 - the actual heavy-hex constraint.")
    print("")
    print("   %4s %6s %4s %10s %12s"
          % ("L", "k", "w", "usable m", "m / k"))
    for L in (4, 5, 6):
        k = L * L
        adj = grid_graph(L, degree4=False)
        best, bw = 0, 0
        for w in (3, 4, 5):
            fam = greedy_res_v(connected_subsets(adj, k, w))
            if len(fam) // 2 > best:
                best, bw = len(fam) // 2, w
        print("   %4d %6d %4d %10d %12.4f" % (L, k, bw, best, best / k))
    print("")
    print("   Degree 3 is what a real device offers. If m/k is flat here the")
    print("   no-go extends to the hardware that matters.")


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print("TIER C - NO CIRCUIT. Exact GF(2) combinatorics.")
    print("")
    want = sys.argv[1:] or ["1", "2", "3"]
    for k, fn in (("1", part1), ("2", part2), ("3", part3)):
        if k in want:
            fn()
