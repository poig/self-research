"""Does a circulant mixer, built as a CIRCUIT, break the separability that makes
QLTO's hypercube walk a classical computation - and does it cross a barrier?

Part VI proved the shipped walk factorises:

    H_walk = sum_i ( beta X_i + alpha_i Z_i )  ->  tensor product, zero
    entanglement, O(n) classical.

Part VII then compared mixer against potential with dense `expm` and NO CIRCUIT
(tier C), concluding the potential DEGREE was the binding constraint and the
mixer was "the wrong knob". That table is tier C by R1, and its rows are not
comparable - a cycle on 2^n vertices against a hypercube on n qubits. This file
rebuilds the comparison as real Qiskit circuits.

THE CONSTRUCTION, Wang's lecture notes (CQCWS1_19335, slide 9). A circulant is
diagonal in the Fourier basis, so

    e^{-i C t}  =  F^dagger  e^{-i Lambda t}  F,   Lambda = DFT(first row of C)

QFT -> diagonal phases -> inverse QFT. The same shape as `_qpe_template`'s
ladder and as Part VIII's radius register: one primitive, three roles.

TIERS.
  PART 0  tier B - Operator vs scipy expm. Exactness identity, no shots.
  PART 1  tier B - operator norm and entanglement entropy. Mechanism only.
  PART 2  tier A - transpiled, gate counts on rz/sx/x/cx.
  PART 3  tier A - AerSimulator with shots.

TWO TEST FLAWS FOUND ON THE FIRST RUN, both fixed here, both worth recording
because each made a result look like something it was not.

  THE CONVENTION TEST WAS VACUOUS. Trying both DFT signs on cycle/complete/
  Mobius gave IDENTICAL error, because every undirected circulant has
  c[m] = c[N-m], so Lambda is real and the sign cancels. A CHIRAL row
  (c[1]=0.5i, c[N-1]=-0.5i) is added: Hermitian, so still unitary, but
  Lambda_k = -2a sin(2 pi k/N) is ODD in k, so the sign now matters. A truly
  DIRECTED row is non-Hermitian and DiagonalGate rejects it outright - which is
  itself the reason the slides file directed walks under Non-Unitary.

  THE BARRIER WAS POSED IN THE WRONG METRIC. The first run used a Hamming-weight
  double well for every mixer, and the HYPERCUBE won - P(target) 0.4390 against
  the cycle's 0.2286. That is not a result about tunnelling. On Z_32 vertex 0's
  neighbours are 1 and 31, so a Hamming-weight potential is DISORDER in the
  cycle's own geometry - the Anderson-localisation regime Part III cites from
  Yin et al., not a barrier. PART 3 now poses the well in each mixer's OWN
  metric: Hamming weight for the hypercube, circular distance for the
  circulants.

  >> MIXER AND POTENTIAL ARE NOT INDEPENDENT KNOBS. The potential must be
  >> smooth in the mixer's metric. Part VII treats them as separable choices.

  AND THE METRIC MATCH IS STILL NOT A FAIR MATCH. With the well posed in each
  mixer's own metric the hypercube STILL wins (PART 3). Matching height and
  shape does not match the DENSITY OF STATES at the barrier: a Hamming-weight
  barrier on the hypercube is C(n, n/2) ~ 2^n/sqrt(n) states wide, while the
  cycle's is O(1) vertices wide. Liu-Su-Li s S_0 is an INFIMUM OVER PATHS, and
  the hypercube has n! paths from w=0 to w=n against the cycle s two. The
  hypercube gets a short Agmon distance for free. So PART 3 as it stands
  measures density of states, not tunnelling, and the fair comparison is at
  matched S_0 - which is the next build, not this one.
"""
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, DiagonalGate, StatePreparation
from qiskit.quantum_info import Operator, Statevector, partial_trace, entropy
from qiskit_aer import AerSimulator

BASIS = ['rz', 'sx', 'x', 'cx']


# -- graphs ----------------------------------------------------------------

def circulant_row(name, N):
    """First row c of a circulant adjacency, C[j,k] = c[(k-j) mod N]."""
    c = np.zeros(N, dtype=complex)
    if name == 'cycle':
        c[1] = c[N - 1] = 1.0
    elif name == 'complete':
        c[:] = 1.0                    # self-loop included, matching the slides'
    elif name == 'mobius':            # Lambda = diag(N,0,...,0)
        c[1] = c[N - 1] = 1.0
        c[N // 2] = 1.0
    elif name == 'chiral':
        # HERMITIAN but complex: c[m] = conj(c[N-m]). Lambda_k = -2a sin(2 pi k/N)
        # is ODD in k, so flipping the DFT sign FLIPS the spectrum - the only
        # kind of circulant that can discriminate the two conventions. A truly
        # DIRECTED row (c[m] != conj(c[N-m])) is non-Hermitian, its evolution is
        # not unitary, and DiagonalGate rejects it - which is exactly why the
        # slides put directed walks under the Non-Unitary branch.
        c[1] = 0.5j
        c[N - 1] = -0.5j
    else:
        raise ValueError(name)
    return c


def circulant_matrix(c):
    N = len(c)
    return np.array([[c[(k - j) % N] for k in range(N)] for j in range(N)],
                    dtype=complex)


# -- circuits --------------------------------------------------------------

def circulant_walk_qc(c, t, conj=False):
    """F^dag e^{-i Lambda t} F. `conj` flips the DFT sign; PART 0 decides."""
    N = len(c)
    n = int(np.log2(N))
    sign = 1.0 if not conj else -1.0
    lam = np.array([sum(c[m] * np.exp(sign * 2j * np.pi * m * k / N)
                        for m in range(N)) for k in range(N)])
    qc = QuantumCircuit(n)
    qc.append(QFTGate(n).inverse(), range(n))
    qc.append(DiagonalGate(list(np.exp(-1j * lam * t))), range(n))
    qc.append(QFTGate(n), range(n))
    return qc


def hypercube_walk_qc(n, t):
    """e^{-i t sum_i X_i} = tensor_i RX(2t). One gate per qubit, no entangler."""
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.rx(2.0 * t, q)
    return qc


def potential_qc(n, V, t):
    qc = QuantumCircuit(n)
    qc.append(DiagonalGate(list(np.exp(-1j * np.asarray(V, float) * t))),
              range(n))
    return qc


def trotter_walk(n, mixer_qc, V, t, steps):
    """(e^{-i A dt} e^{-i V dt})^steps - the split-operator walk."""
    dt = t / steps
    qc = QuantumCircuit(n)
    for _ in range(steps):
        qc.compose(mixer_qc(dt), inplace=True)
        qc.compose(potential_qc(n, V, dt), inplace=True)
    return qc


# -- parts -----------------------------------------------------------------

def part0():
    print("PART 0  is F^dag e^{-i Lambda t} F actually e^{-i C t}?")
    print("        TIER B - Operator vs scipy expm, no shots")
    print("        `chiral` is the only row that CAN distinguish the two")
    print("        conventions; the undirected ones cannot, by symmetry.")
    print("")
    print("  %-10s %3s   %-16s %-16s" % ("graph", "n", "err conj=False",
                                         "err conj=True"))
    votes = []
    for n in (3, 4):
        N = 1 << n
        for name in ('cycle', 'complete', 'mobius', 'chiral'):
            c = circulant_row(name, N)
            ref = expm(-1j * circulant_matrix(c) * 0.37)
            errs = {}
            for conj in (False, True):
                U = Operator(circulant_walk_qc(c, 0.37, conj)).data
                errs[conj] = float(np.linalg.norm(U - ref))
            print("  %-10s %3d   %-16.2e %-16.2e"
                  % (name, n, errs[False], errs[True]))
            if abs(errs[False] - errs[True]) > 1e-9:
                votes.append(min(errs, key=errs.get))
    print("")
    if votes:
        uniq = set(votes)
        print("  DISCRIMINATING rows agree on conj=%s  (%d of them)"
              % (list(uniq)[0] if len(uniq) == 1 else uniq, len(votes)))
        return list(uniq)[0] if len(uniq) == 1 else False
    print("  no row discriminated - the test is vacuous")
    return False


def part1(conj):
    print("")
    print("")
    print("PART 1  does the mixer alone break separability, with only a")
    print("        DEGREE-1 drift?   TIER B - exact, mechanism only")
    print("")
    n, t = 4, 0.6
    N = 1 << n
    rng = np.random.default_rng(7)
    alpha = rng.normal(size=n)
    V = np.array([sum(alpha[q] * (1.0 - 2.0 * ((x >> q) & 1)) for q in range(n))
                  for x in range(N)], dtype=float)

    psi0 = np.array([1.0 + 0j])
    for q in range(n):
        th = rng.uniform(0.3, np.pi - 0.3)
        psi0 = np.kron(psi0, np.array([np.cos(th / 2), np.sin(th / 2)]))

    rx = np.array([[np.cos(t), -1j * np.sin(t)], [-1j * np.sin(t), np.cos(t)]])
    fac = np.array([[1.0 + 0j]])
    for q in range(n):
        fac = np.kron(fac, rx)

    mixers = [
        ('hypercube sum X_i', lambda dt: hypercube_walk_qc(n, dt)),
        ('cycle    C_16',
         lambda dt: circulant_walk_qc(circulant_row('cycle', N), dt, conj)),
        ('complete K_16',
         lambda dt: circulant_walk_qc(circulant_row('complete', N), dt, conj)),
        ('mobius   M_16',
         lambda dt: circulant_walk_qc(circulant_row('mobius', N), dt, conj)),
    ]
    print("  %-20s %18s %20s" % ("mixer", "||mixer - prodRX||",
                                 "walk mid-cut entropy"))
    for label, mk in mixers:
        dfac = float(np.linalg.norm(Operator(mk(t)).data - fac))
        U = Operator(trotter_walk(n, mk, V, t, steps=8)).data
        ent = float(entropy(partial_trace(Statevector(U @ psi0),
                                          list(range(n // 2))), base=2))
        print("  %-20s %18.2e %20.4f" % (label, dfac, ent))
    print("")
    print("  entropy is from a PRODUCT input, so any nonzero value is")
    print("  correlation the walk created. Part VI predicts exactly 0 for the")
    print("  hypercube, and the drift here is DEGREE-1 for every row.")


def part2():
    print("")
    print("")
    print("PART 2  circuit cost.   TIER A - transpiled to rz/sx/x/cx")
    print("")
    print("  %-10s %3s %8s %8s %8s" % ("graph", "n", "depth", "2q", "total"))
    for n in (3, 4, 5):
        N = 1 << n
        for name in ('complete', 'cycle'):
            tq = transpile(circulant_walk_qc(circulant_row(name, N), 0.37),
                           basis_gates=BASIS, optimization_level=3)
            o = tq.count_ops()
            print("  %-10s %3d %8d %8d %8d"
                  % (name, n, tq.depth(), o.get('cx', 0), sum(o.values())))
        tq = transpile(hypercube_walk_qc(n, 0.37), basis_gates=BASIS,
                       optimization_level=3)
        o = tq.count_ops()
        print("  %-10s %3d %8d %8d %8d"
              % ('hypercube', n, tq.depth(), o.get('cx', 0), sum(o.values())))
    print("")
    print("  CAVEAT: DiagonalGate synthesises an ARBITRARY diagonal in O(2^n).")
    print("  The slides' 2*log2(n)+1 is for the COMPLETE graph specifically,")
    print("  whose spectrum (N,0,...,0) is one multi-controlled phase. These")
    print("  are correctness numbers and an UPPER BOUND, not the efficient")
    print("  construction, which is a separate build.")


def graph_adjacency(kind, n):
    """Dense adjacency. `kind` is 'hypercube' or a circulant name."""
    N = 1 << n
    if kind == 'hypercube':
        A = np.zeros((N, N))
        for x in range(N):
            for q in range(n):
                A[x, x ^ (1 << q)] = 1.0
        return A
    return np.real(circulant_matrix(circulant_row(kind, N)))


def kinetic_qc(kind, n, h, dt):
    """e^{-i h^2 (D - A) dt}, dropping the constant D for a regular graph.

    SIGN. The Schroedinger kinetic term is -h^2 Laplacian = +h^2 (D - A), so up
    to a constant the generator is -h^2 A and the evolution is e^{+i h^2 A dt}.
    v136's first two rounds used e^{-i A dt} - the wrong sign - which inverts the
    dispersion and makes the well's ground state the HIGHEST momentum state.
    """
    N = 1 << n
    a = h * h * dt
    if kind == 'hypercube':
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.rx(-2.0 * a, q)          # e^{+i a X} = RX(-2a)
        return qc
    c = circulant_row(kind, N)
    lam = np.array([sum(c[m] * np.exp(2j * np.pi * m * k / N)
                        for m in range(N)) for k in range(N)])
    qc = QuantumCircuit(n)
    qc.append(QFTGate(n).inverse(), range(n))
    qc.append(DiagonalGate(list(np.exp(1j * np.real(lam) * a))), range(n))
    qc.append(QFTGate(n), range(n))
    return qc


def agmon_distance(A, V, E, src, dst):
    """Discrete Agmon distance: shortest path under edge weight sqrt(max(V-E,0)).

    Liu-Su-Li's S_0 = int sqrt(f) is an INFIMUM OVER PATHS in the metric
    sqrt(f - E) dx^2. On a graph that is exactly a shortest-path problem, so it
    is computable rather than assumed - and it is the quantity their exponent
    e^{S_0/h} is built from.
    """
    N = len(V)
    w = np.sqrt(np.maximum(V - E, 0.0))
    dist = np.full(N, np.inf)
    dist[src] = 0.0
    seen = np.zeros(N, bool)
    for _ in range(N):
        u = int(np.argmin(np.where(seen, np.inf, dist)))
        if seen[u] or not np.isfinite(dist[u]):
            break
        seen[u] = True
        for v in np.nonzero(A[u])[0]:
            # half the barrier cost at each endpoint - the trapezoid rule for
            # the path integral, so the endpoints are not double counted
            cand = dist[u] + 0.5 * (w[u] + w[v])
            if cand < dist[v]:
                dist[v] = cand
    return float(dist[dst])


def part3():
    print("")
    print("")
    print("PART 3  Liu-Su-Li's ACTUAL algorithm: start in one well's LOCAL")
    print("        GROUND STATE, evolve to t = pi/DeltaE, land in the other.")
    print("        TIER A - StatePreparation + Trotter on AerSimulator, shots")
    print("")
    n, shots = 5, 20000
    N = 1 << n
    be = AerSimulator(seed_simulator=11)

    print("  %-11s %5s %10s %10s %10s %9s %9s"
          % ("graph", "h", "DeltaE", "t=pi/dE", "S_0", "P(far) B", "P(far) A"))
    for kind, coord in (('hypercube', lambda x: bin(x).count('1')),
                        ('cycle', lambda x: min(x, N - x)),
                        ('mobius', lambda x: min(x, N - x)),
                        ('complete', lambda x: min(x, N - x))):
        A = graph_adjacency(kind, n)
        p = np.array([coord(x) for x in range(N)], float)
        span = p.max() - p.min()
        # symmetric double well: minima at the two ends of the coordinate
        V = 3.0 * np.exp(-((p - 0.5 * span) ** 2) / (0.10 * span ** 2))
        left = np.nonzero(p <= 0.25 * span)[0]
        right = np.nonzero(p >= 0.75 * span)[0]

        for h in (0.9, 0.6):
            H = h * h * (np.diag(A.sum(1)) - A) + np.diag(V)
            ev, U = np.linalg.eigh(H)
            phi0, phi1 = U[:, 0], U[:, 1]
            dE = float(ev[1] - ev[0])
            # Phi_- = the combination localised on the LEFT well
            m = (phi0 + phi1) / np.sqrt(2.0)
            if np.sum(np.abs(m[left]) ** 2) < np.sum(np.abs(m[right]) ** 2):
                m = (phi0 - phi1) / np.sqrt(2.0)
            m = m / np.linalg.norm(m)
            loc = float(np.sum(np.abs(m[left]) ** 2))
            t = float(np.pi / dE) if dE > 1e-12 else np.inf
            S0 = agmon_distance(A, V, float(ev[0]),
                                int(left[np.argmin(V[left])]),
                                int(right[np.argmin(V[right])]))

            # tier B reference: exact evolution
            psi_t = expm(-1j * H * t) @ m
            pB = float(np.sum(np.abs(psi_t[right]) ** 2))

            # tier A: the circuit
            steps = 60
            qc = QuantumCircuit(n, n)
            qc.append(StatePreparation(m.astype(complex)), range(n))
            dt = t / steps
            for _ in range(steps):
                qc.compose(kinetic_qc(kind, n, h, dt), inplace=True)
                qc.compose(potential_qc(n, V, dt), inplace=True)
            qc.measure(range(n), range(n))
            tq = transpile(qc, be, basis_gates=BASIS, optimization_level=1)
            cnt = be.run(tq, shots=shots).result().get_counts()
            tot = sum(cnt.values())
            pA = sum(v for k, v in cnt.items() if int(k, 2) in set(right)) / tot

            print("  %-11s %5.2f %10.2e %10.1f %10.4f %9.4f %9.4f"
                  % (kind, h, dE, t, S0, pB, pA))
        print("           (localisation of Phi_- in the left well: %.4f)" % loc)
    print("")
    print("  P(far) B is exact expm on the SAME state - the reference the")
    print("  circuit is checked against. P(far) A is the circuit with shots.")
    print("  A gap between them is Trotter error, not physics.")


def part4():
    print("")
    print("")
    print("PART 4  does DeltaE track Liu-Su-Li's e^{-S_0/h}?")
    print("        TIER C - dense spectra only. This is the THEORY check;")
    print("        if the slope is not -1 their continuum exponent does not")
    print("        survive discretisation, whatever the circuit does.")
    print("")
    n = 5
    N = 1 << n
    hs = np.array([1.2, 1.0, 0.85, 0.7, 0.6, 0.5])
    print("  %-11s %10s %10s %14s" % ("graph", "S_0", "fit slope",
                                      "predicted -1"))
    for kind, coord in (('hypercube', lambda x: bin(x).count('1')),
                        ('cycle', lambda x: min(x, N - x)),
                        ('mobius', lambda x: min(x, N - x))):
        A = graph_adjacency(kind, n)
        p = np.array([coord(x) for x in range(N)], float)
        span = p.max() - p.min()
        V = 3.0 * np.exp(-((p - 0.5 * span) ** 2) / (0.10 * span ** 2))
        left = np.nonzero(p <= 0.25 * span)[0]
        right = np.nonzero(p >= 0.75 * span)[0]
        y, S0ref = [], None
        for h in hs:
            H = h * h * (np.diag(A.sum(1)) - A) + np.diag(V)
            ev = np.linalg.eigvalsh(H)
            y.append(np.log(max(ev[1] - ev[0], 1e-300)) - 0.5 * np.log(h))
            if S0ref is None:
                S0ref = agmon_distance(A, V, float(ev[0]),
                                       int(left[np.argmin(V[left])]),
                                       int(right[np.argmin(V[right])]))
        # ln(dE) - 0.5 ln h  =  -S_0/h + const   ->  slope against S_0/h is -1
        if S0ref < 1e-9:
            print("  %-11s %10.4f %10s %14s"
                  % (kind, S0ref, "n/a", "NO BARRIER"))
            continue
        x = S0ref / hs
        sl = float(np.polyfit(x, np.array(y), 1)[0])
        print("  %-11s %10.4f %10.4f %14s"
              % (kind, S0ref, sl, "%.2f" % -1.0))
    print("")
    print("  slope is d[ln dE - 0.5 ln h] / d[S_0/h]. Liu-Su-Li Eq. 7 predicts")
    print("  exactly -1 in the semiclassical limit.")


def reduced_chain(kind, n):
    """(hopping t[.], diagonal-degree d[.], coordinate p[.]) for a 1-D chain.

    THE HYPERCUBE REDUCES. A potential depending only on Hamming weight commutes
    with every permutation, so the dynamics stay in the SYMMETRIC subspace. That
    is Liu-Su-Li's own Remark 2.2: with infinitely (here n! ) many geodesics of
    equal Agmon length, "the problem can be reduced to one in a lower-dimensional
    space". Since sum_i X_i = 2 J_x on that subspace,

        <w+1| sum_i X_i |w>  =  sqrt( (w+1)(n-w) )

    so the reduced chain has POSITION-DEPENDENT hopping, largest at the middle -
    which is exactly where the barrier sits. v136 PART 4 assumed uniform hopping
    and measured slope -1.99 instead of -1. This is that factor.
    """
    if kind == 'hypercube':
        p = np.arange(n + 1, dtype=float)
        t = np.array([np.sqrt((w + 1.0) * (n - w)) for w in range(n)])
        d = np.full(n + 1, float(n))          # every hypercube vertex has degree n
        return t, d, p
    if kind == 'cycle':                        # N sites, uniform hopping, folded
        N = 1 << n if n < 12 else n            # accept either a qubit count or N
        half = N // 2
        p = np.arange(half + 1, dtype=float)
        t = np.ones(half)
        d = np.full(half + 1, 2.0)
        return t, d, p
    raise ValueError(kind)


def chain_hamiltonian(t, d, V, h):
    """h^2 (D - A) + V on the chain. The SIGN is +h^2(D-A) = -h^2 Laplacian."""
    m = len(V)
    H = np.diag(h * h * d + V)
    for i in range(m - 1):
        H[i, i + 1] = H[i + 1, i] = -h * h * t[i]
    return H


def agmon_chain(t, V, E, corrected=True):
    """Agmon distance along the chain.

        corrected:   S0 = sum_edges sqrt( (Vbar - E) / t_edge ) * dx
        uncorrected: S0 = sum_edges sqrt(  Vbar - E )          * dx

    WKB for -h^2 (t psi')' + V psi = E psi gives t (S')^2 = V - E, hence
    S' = sqrt((V-E)/t). With uniform t the two agree, which is why the CYCLE
    confirmed the exponent and the hypercube did not.
    """
    s = 0.0
    for i in range(len(V) - 1):
        vb = 0.5 * (V[i] + V[i + 1]) - E
        if vb <= 0:
            continue
        s += np.sqrt(vb / t[i]) if corrected else np.sqrt(vb)
    return float(s)


def part5():
    print("")
    print("")
    print("PART 5  THE FIX: reduce to the symmetric subspace, and use the")
    print("        Agmon metric for POSITION-DEPENDENT hopping.")
    print("        TIER C - dense spectra of the reduced chain. Theory check.")
    print("")
    print("  slope = d[ ln DeltaE - 0.5 ln h ] / d[ S0/h ];  Eq. 7 predicts -1")
    print("")
    print("  %-11s %4s %8s %9s %11s %11s"
          % ("graph", "n", "S0 raw", "S0 corr", "slope raw", "slope corr"))
    for kind, ns in (('hypercube', (5, 10, 20, 40, 80)),
                     ('cycle', (5, 6, 7))):
        for n in ns:
            t, d, p = reduced_chain(kind, n)
            span = p.max() - p.min()
            V = 3.0 * np.exp(-((p - 0.5 * span) ** 2) / (0.06 * span ** 2))

            # ground energy at a reference h, used for the Agmon E offset
            E0ref = float(np.linalg.eigvalsh(chain_hamiltonian(t, d, V, 0.3))[0])
            S_raw = agmon_chain(t, V, E0ref, corrected=False)
            S_cor = agmon_chain(t, V, E0ref, corrected=True)

            # choose h so that S/h lands in the semiclassical window [3, 12]
            out = {}
            for lab, S in (('raw', S_raw), ('cor', S_cor)):
                hs = S / np.linspace(3.0, 12.0, 7)
                y, x = [], []
                for h in hs:
                    ev = np.linalg.eigvalsh(chain_hamiltonian(t, d, V, h))
                    dE = ev[1] - ev[0]
                    if dE <= 0 or not np.isfinite(dE):
                        continue
                    y.append(np.log(dE) - 0.5 * np.log(h))
                    x.append(S / h)
                out[lab] = (float(np.polyfit(x, y, 1)[0])
                            if len(x) > 2 else float('nan'))
            print("  %-11s %4d %8.3f %9.3f %11.4f %11.4f"
                  % (kind, n, S_raw, S_cor, out['raw'], out['cor']))
    print("")
    print("  cycle rows: hopping is UNIFORM, so raw and corrected coincide -")
    print("  the control that says the correction is not just a free parameter.")


def part6():
    print("")
    print("")
    print("PART 6  distinct eigenvalues: the quantity that predicts BOTH the")
    print("        circuit cost and the DLA.   TIER C - spectra only")
    print("")
    print("  Wang (CQCWS1_19335 slide 9): an efficient circuit needs either")
    print("    (1) O(polylog N) DISTINCT eigenvalues, or")
    print("    (2) eigenvalues in closed form.")
    print("  Bridi et al. (2508.05749) Thm 1: dim(g_QWOA) <= m^2 + 1, m the")
    print("  number of DISTINCT eigenvalues of the problem Hamiltonian.")
    print("  Same quantity. Few eigenvalues buys a cheap circuit AND a")
    print("  collapsed DLA - cheap and useless together.")
    print("")
    print("  %-11s %4s %8s %10s %-26s" % ("graph", "N", "distinct", "route",
                                          "consequence"))
    for n in (4, 5):
        N = 1 << n
        for kind in ('complete', 'cycle', 'mobius', 'hypercube'):
            A = graph_adjacency(kind, n)
            ev = np.linalg.eigvalsh(A)
            k = len(np.unique(np.round(ev, 8)))
            if k <= 4:
                route, cons = "(1) few", "small DLA, nothing to learn"
            elif kind in ('cycle', 'mobius'):
                route, cons = "(2) closed", "LARGE DLA and still cheap"
            else:
                route, cons = "(2) closed", "large DLA, cheap"
            print("  %-11s %4d %8d %10s %-26s" % (kind, N, k, route, cons))
    print("")
    print("  The complete graph's 2 eigenvalues are why PART 3 gave it S0 = 0")
    print("  and Phi_- localisation 0.50: there is no barrier and no structure.")
    print("  Bridi report DLA dimension exactly 2 for it.")


def part7():
    print("")
    print("")
    print("PART 7  THE HYPERCUBE IS NOT A PARTICLE, IT IS A SPIN.")
    print("        TIER C - dense spectra of the reduced chain.")
    print("")
    print("  PART 5's corrected Agmon metric did not fix the slope; it made it")
    print("  worse with n (-1.18 -> -2.72). The reason is structural, not a")
    print("  bad metric:  n*I - 2 J_x  is NOT the Laplacian of the reduced")
    print("  chain. Its row sums are")
    print("")
    print("      n - sqrt(w(n-w+1)) - sqrt((w+1)(n-w))   !=  0")
    print("")
    print("  which is n - sqrt(n) at the ends and ~0 in the middle - a large")
    print("  BUILT-IN inverted potential of size O(h^2 n) that no choice of")
    print("  Agmon metric on V alone can see.")
    print("")
    print("  So the hypercube walk is a LARGE-SPIN system (Lipkin-Meshkov-Glick")
    print("  shaped), whose semiclassical parameter is 1/n, NOT h. Liu-Su-Li's")
    print("  framework is a particle with -h^2 Laplacian on R^d; sweeping h at")
    print("  fixed n is simply not its semiclassical limit.")
    print("")
    print("  THE TEST. If the hypercube is a spin, then with the potential")
    print("  scaled EXTENSIVELY (V = n v(w/n)) and h FIXED, ln DeltaE should be")
    print("  linear in n with a constant slope -S~ (an intensive action).")
    print("")
    print("  %-11s %5s %13s %13s %11s"
          % ("h fixed", "n", "DeltaE", "ln dE", "d lndE/dn"))
    print("  Points are DISCARDED below the double-precision eigenvalue")
    print("  floor eps*||H||: a splitting of 1e-14 against ||H|| ~ 150 is")
    print("  LAPACK resolution, not physics. The first run of this part fitted")
    print("  through the floor and reported r = -0.58; the usable points had")
    print("  already given -0.959 and -0.970.")
    print("")
    for h in (1.0, 0.7):
        ns_all = np.array([8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
        ns, ys = [], []
        prev = None
        for n in ns_all:
            w = np.arange(n + 1, dtype=float)
            u = w / n
            V = float(n) * 1.2 * np.exp(-((u - 0.5) ** 2) / 0.06)
            t = np.array([np.sqrt((k + 1.0) * (n - k)) for k in range(n)])
            H = np.diag(h * h * float(n) + V)
            for i in range(n):
                H[i, i + 1] = H[i + 1, i] = -h * h * t[i]
            ev = np.linalg.eigvalsh(H)
            dE = float(ev[1] - ev[0])
            floor = 50.0 * np.finfo(float).eps * float(np.abs(ev).max())
            if dE < floor:
                print("  h=%-9.2f %5d %13.3e %13s   below floor %.1e"
                      % (h, n, dE, "--", floor))
                continue
            y = np.log(dE)
            ns.append(n); ys.append(y)
            slope = "" if prev is None else "%11.4f" % ((y - prev[1]) /
                                                        (n - prev[0]))
            print("  h=%-9.2f %5d %13.3e %13.4f %s"
                  % (h, n, dE, y, slope))
            prev = (n, y)
        ns = np.array(ns); ys = np.array(ys)
        fit = float(np.polyfit(ns, ys, 1)[0])
        r = np.corrcoef(ns, ys)[0, 1]
        print("      -> fit over %d usable points: d(ln dE)/dn = %.4f"
              % (len(ns), fit))
        print("         linear correlation r = %.6f" % r)
        print("")
    print("  A CONSTANT slope and r ~ -1 means ln DeltaE = -n S~ + const, i.e.")
    print("  the tunnelling is exponential in the SPIN SIZE, exactly as a")
    print("  large-spin WKB predicts - and NOT of the form e^{-S0/h} with S0 a")
    print("  path integral over a fixed landscape.")
    print("")
    print("  CONSEQUENCE FOR QLTO. The design/walk register at q=2 levels per")
    print("  parameter is a spin-n/2 system, not a discretised particle. Every")
    print("  continuum tunnelling bound quoted for it - including v99 PART 4's")
    print("  exp(width) vs exp(height) reading - is being applied outside its")
    print("  hypotheses. The CYCLE register (PART 5, slope -0.998) is a genuine")
    print("  particle-on-a-lattice and the bound does hold there.")


if __name__ == '__main__':
    print(__doc__.split('\n')[0])
    print("=" * 72)
    conj = part0()
    part1(conj)
    part2()
    part3()
    part4()
    part5()
    part6()
    part7()
