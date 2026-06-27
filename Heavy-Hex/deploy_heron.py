#!/usr/bin/env python3
"""
Submit (1+x^2)(1+y^2) flag-qubit QEC experiment to IBM Heron r2.

Decodes via consistency projection + linear basis: for each stabilizer, require
all 4 rounds to agree on the syndrome value. ~34/48 bits pass this 4/4 check at
Heron noise, and Gaussian elimination on H_clean * E = S_clean projects the full
syndrome onto Col(H). The 24-dim basis decoder then finds the correction.

At Heron-default noise (1.2% CX, 1% readout): 97% decode rate, 2% true LER.

Usage:
  export IBM_QUANTUM_TOKEN='your_token'
  python3 deploy_heron.py                        # single run (interactive backend chooser)
  python3 deploy_heron.py --list-backends        # show available QPUs + queue depth, exit
  python3 deploy_heron.py --backend ibm_kyiv     # submit to a named QPU (no prompt)
  python3 deploy_heron.py --rounds 4 --shots 1000 # 4 rounds, 1000 shots

When run interactively with no --backend, the script lists every operational QPU
(>=156 qubits) with its pending-job queue and prompts you to choose; Enter selects
the least-busy one. Piped/non-interactive runs auto-pick least-busy so they never block.

Saves job ID to ~/.planewarp_jobs.json.
Press Ctrl+C after submission to detach.
Retrieve:  python3 deploy_retrieve.py <job_id>
"""

import json, sys, getpass, os, time
from pathlib import Path

import numpy as np

IBM_TOKEN_ENV = "IBM_QUANTUM_TOKEN"
SAVE_FILE = Path.home() / ".planewarp_jobs.json"
CLEAN_DIR = Path.home() / ".planewarp_clean"


def get_token():
    token = os.environ.get(IBM_TOKEN_ENV)
    if token:
        return token
    token = getpass.getpass("IBM Quantum API token: ")
    if token:
        return token
    print("No token provided.", file=sys.stderr)
    sys.exit(1)


def build_flag_circuit(r, s, rounds, final_data_readout=False, use_buffer=False,
                       share_pairs=False):
    """Flag-qubit circuit for (1+x²)(1+y²) code on heavy-hex.

    Args:
        use_buffer: If True, replace CX(data, flag) with
            CX(data, spare) + CX(spare, flag) + reset(spare)
            for local heavy-hex routing with zero SWAPs.
        share_pairs: If True, measure each weight-2 vertical pair
            vpair(i,j)=D(i,j)^D(i+2,j) ONCE (one ancilla per (i,j), 4 CZ via
            spare), instead of extracting a0=vpair(i,j) and a1=vpair(i,j+2)
            separately for every plaquette. Since a1(i,j) == a0(i,j+2), the
            unshared circuit extracts every pair twice. The plaquette syndrome is
            reassembled in all_syndromes_shared() as syn(i,j)=m(i,j)^m(i,(j+2)%s),
            bit-identical to the unshared syn.

            Routing: each pair is measured via two buffered legs (like use_buffer)
            — data→spare→a0, reset(spare), data→spare→a0, reset(spare) — so all
            interactions are local on the heavy-hex graph and Sabre inserts zero
            routing SWAPs. CZ/round = 4*r*s (vs 8*r*s for pure buffer mode),
            and ancilla usage is halved. --buffer is redundant when this flag is
            set (ignored with a note).

            Trade-off: a shared-ancilla measurement fault now flips two adjacent
            plaquette detectors instead of one (a benign horizontal matching edge
            for a matching decoder; validate against the tesseract decoder on HW).
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from pw_qiskit import heavy_hex_flag_layout

    data_map, anc_maps, _, _ = heavy_hex_flag_layout(r, s)
    n_data = r * s
    # The layout always allocates 2*r*s physical ancilla slots and anc_maps indexes
    # into that full range, so the QUANTUM register stays full-size even when sharing.
    # (Shrinking it pushes anc_maps[(i,j,0)] out of range.) In share mode the a1
    # ancillas simply receive no gates — idle, unentangled, no two-qubit error.
    n_anc_phys = 2 * r * s
    # Classical syndrome bits actually recorded: one per measured ancilla.
    n_meas = r * s if share_pairs else 2 * r * s
    n_spare = 12 if use_buffer else 0
    total = n_data + n_anc_phys + n_spare

    qr = QuantumRegister(total, "q")
    cr_syn = [ClassicalRegister(n_meas, f"syn_{c}") for c in range(rounds)]
    registers = [qr, *cr_syn]
    cr_data = None
    if final_data_readout:
        cr_data = ClassicalRegister(n_data, "data")
        registers.append(cr_data)
    qc = QuantumCircuit(*registers)

    # Spare qubit index helper
    def spare_idx(d):
        return n_data + n_anc_phys + (d % n_spare)

    for rnd in range(rounds):
        for i in range(r):
            for j in range(s):
                if share_pairs:
                    # Direct share-pairs: 2 CX per pair, both local on heavy-hex
                    # when VF2Layout finds the correct mapping.
                    # Plaquettes are reassembled classically in all_syndromes_shared.
                    a = anc_maps[(i, j, 0)]
                    qc.reset(a)
                    qc.cx(data_map[i][j], a)          # data → flag (local edge)
                    qc.cx(data_map[(i + 2) % r][j], a) # data2 → flag (local edge)
                    qc.measure(a, cr_syn[rnd][i * s + j])
                    continue
                if use_buffer:
                    s0 = spare_idx(i * s + j)
                    s1 = spare_idx(i * s + j + n_data // 2)
                a0 = anc_maps[(i, j, 0)]
                a1 = anc_maps[(i, j, 1)]

                # anc0: row pair  (i,j) + (i+2,j)
                qc.reset(a0)
                if use_buffer:
                    qc.cx(data_map[i][j], s0)
                    qc.cx(s0, a0)
                    qc.reset(s0)
                    qc.cx(data_map[(i + 2) % r][j], s0)
                    qc.cx(s0, a0)
                    qc.reset(s0)
                else:
                    qc.cx(data_map[i][j], a0)
                    qc.cx(data_map[(i + 2) % r][j], a0)

                # anc1: col pair  (i,j+2) + (i+2,j+2)
                qc.reset(a1)
                if use_buffer:
                    qc.cx(data_map[i][(j + 2) % s], s1)
                    qc.cx(s1, a1)
                    qc.reset(s1)
                    qc.cx(data_map[(i + 2) % r][(j + 2) % s], s1)
                    qc.cx(s1, a1)
                    qc.reset(s1)
                else:
                    qc.cx(data_map[i][(j + 2) % s], a1)
                    qc.cx(data_map[(i + 2) % r][(j + 2) % s], a1)

                qc.measure(a0, cr_syn[rnd][i * s * 2 + j * 2])
                qc.measure(a1, cr_syn[rnd][i * s * 2 + j * 2 + 1])
        qc.barrier()

    if final_data_readout and cr_data is not None:
        for i in range(r):
            for j in range(s):
                qc.measure(data_map[i][j], cr_data[i * s + j])

    return qc, data_map, anc_maps


LOGICAL_OBS = [0, 2, 4, 6]
"""Logical Z observable: alternating qubits in row 0 (Z⊗Z⊗Z⊗Z on col 0,2,4,6)."""


def all_syndromes_shared(pub_result, rounds, r, s):
    """Extract (shots, rounds, r, s) from a share_pairs circuit.

    Each register holds r*s once-measured vertical pairs m(i,j)=D(i,j)^D(i+2,j).
    The plaquette syndrome is syn(i,j)=m(i,j) ^ m(i,(j+2)%s), bit-identical to the
    a0^a1 produced by all_syndromes() on the unshared circuit.
    """
    n_stab = r * s
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots
    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
        m = bits[:, :n_stab].reshape(shots, r, s)        # m[:, i, j]
        syn[:, c] = m ^ np.roll(m, shift=-2, axis=2)     # XOR with column (j+2)%s
    return syn


def all_syndromes(pub_result, rounds, r, s):
    """Extract (shots, rounds, r, s) from a SamplerV2 PubResult.

    BitArray order='little' so index 0 = first measured ancilla.
    """
    n_stab = r * s
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots
    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
        # Columns are [a0_q, a1_q] per stabilizer q (q = i*s + j). XOR each pair
        # in one vectorized op instead of an n_stab-long Python loop.
        pair = bits[:, :2 * n_stab].reshape(shots, n_stab, 2)
        syn[:, c] = (pair[:, :, 0] ^ pair[:, :, 1]).reshape(shots, r, s)
    return syn


def print_logical_diagnostics(correction):
    r, s = correction.shape
    for px in (0, 1):
        for py in (0, 1):
            hr, hs = r // 2, s // 2
            for si in range(hr):
                rp = 0
                for sj in range(hs):
                    rp ^= correction[px + 2 * si, py + 2 * sj]
                if rp:
                    print(f"    Sub-lattice ({px},{py}) row {si}: ODD")
            for sj in range(hs):
                cp = 0
                for si in range(hr):
                    cp ^= correction[px + 2 * si, py + 2 * sj]
                if cp:
                    print(f"    Sub-lattice ({px},{py}) col {sj}: ODD")


def build_initial_layout(backend, n_data, n_flag, n_spare=12):
    """Autodetect heavy-hex physical qubit indices and build an initial_layout
    that maps data→degree-4, flag→degree-2, spare→remaining degree-2.

    Returns a list of length n_data + n_flag + n_spare mapping each virtual
    qubit to a physical qubit index, or None if detection fails.
    """
    cm = backend.coupling_map
    if cm is None:
        return None
    # Count neighbors per physical qubit
    from collections import Counter
    deg = Counter()
    for a, b in cm:
        deg[a] += 1
        deg[b] += 1

    # On heavy-hex: degree-4 nodes = data, degree-2 = flag/spare
    d4 = sorted([q for q, d in deg.items() if d == 4])
    d2 = sorted([q for q, d in deg.items() if d == 2])
    all_phys = set(deg.keys())
    deg0 = sorted(all_phys - set(d4) - set(d2))  # degree-1 (edge) or other

    needed_d4 = n_data
    needed_d2 = n_flag + n_spare
    if len(d4) < needed_d4 or len(d2) < needed_d2:
        # Fall back to degree-1 nodes if available
        extra = sorted(deg0)
        d2 = sorted(d2 + extra[:max(0, needed_d2 - len(d2))])

    # Use first n_data degree-4 nodes for data, first n_flag degree-2 for flags,
    # next n_spare degree-2 for spares
    layout = d4[:needed_d4] + d2[:needed_d2]
    # Pad if we don't have enough
    while len(layout) < needed_d4 + needed_d2:
        layout.append(max(all_phys) + 1 + len(layout))
def clean_stats():
    """Aggregate statistics from all saved clean-shot files."""
    if not CLEAN_DIR.exists():
        print("No clean-shot files found.")
        return
    files = sorted(CLEAN_DIR.glob("*.npz"))
    if not files:
        print(f"No .npz files in {CLEAN_DIR}")
        return
    total = 0
    clean_total = 0
    print(f"\n=== Clean-shot archive ({len(files)} files) ===")
    for f in files:
        data = np.load(f)
        n = int(data['n_shots'])
        nc = int(data['n_clean'])
        pct = data['clean_pct']
        job_id = str(data['job_id'])
        print(f"  {f.name}")
        print(f"    Job:       {job_id}")
        print(f"    Shots:     {nc}/{n} clean ({pct:.1f}%)")
        total += n
        clean_total += nc
    print(f"  Total: {clean_total}/{total} clean ({100*clean_total/max(1,total):.1f}%)")
    print(f"  Archive: {CLEAN_DIR}")


def candidate_backends(service, min_qubits=156):
    """Operational, non-simulator backends with >= min_qubits."""
    try:
        pool = service.backends(min_num_qubits=min_qubits, simulator=False, operational=True)
    except TypeError:
        # Older runtime without these kwargs: filter manually so a >=min_qubits
        # simulator can't be silently selected.
        pool = [b for b in service.backends()
                if getattr(b, "num_qubits", 0) >= min_qubits
                and not getattr(getattr(b, "configuration", lambda: None)(), "simulator", False)]
    return [b for b in pool if getattr(b, "num_qubits", 0) >= min_qubits]


def backend_rows(backends):
    """Fetch (backend, qubits, pending_jobs, status_msg) for each, tolerating errors."""
    rows = []
    for b in backends:
        pending, msg = None, "operational"
        try:
            st = b.status()
            pending = getattr(st, "pending_jobs", None)
            if not getattr(st, "operational", True):
                msg = "OFFLINE"
        except Exception:
            msg = "status unavailable"
        rows.append((b, b.num_qubits, pending, msg))
    return rows


def print_backend_table(rows):
    print(f"\n{'#':>2}  {'Backend':<22}{'Qubits':>7}{'Queue':>7}  Status")
    print(f"{'-'*2}  {'-'*22}{'-'*7}{'-'*7}  {'-'*18}")
    for idx, (b, nq, pending, msg) in enumerate(rows):
        q = "?" if pending is None else str(pending)
        print(f"{idx:>2}  {b.name:<22}{nq:>7}{q:>7}  {msg}")


def select_backend(service, opts, min_qubits=156):
    """Resolve which backend to use from --backend / --list-backends / interactive prompt."""
    candidates = candidate_backends(service, min_qubits)
    if not candidates:
        print(f"No operational backend (>= {min_qubits} qubits) found.", file=sys.stderr)
        sys.exit(1)

    # Explicit name wins, and is the scriptable path.
    if getattr(opts, "backend", None):
        for b in candidates:
            if b.name == opts.backend:
                return b
        try:
            return service.backend(opts.backend)   # allow names outside the >=156 filter
        except Exception:
            print(f"Backend '{opts.backend}' not found. Available candidates:", file=sys.stderr)
            print_backend_table(backend_rows(candidates))
            sys.exit(1)

    rows = backend_rows(candidates)

    if getattr(opts, "list_backends", False):
        print_backend_table(rows)
        sys.exit(0)

    # Default = least busy (fewest pending jobs; unknown queue sorts last).
    default_idx = min(range(len(rows)),
                      key=lambda i: (rows[i][2] is None, rows[i][2] if rows[i][2] is not None else 0))

    # Non-interactive (piped/CI): don't block, take the least-busy default.
    if not sys.stdin.isatty():
        chosen = candidates[default_idx]
        print(f"Non-interactive: auto-selected least-busy backend {chosen.name}.")
        return chosen

    print_backend_table(rows)
    while True:
        raw = input(f"\nSelect backend [0-{len(rows)-1}] "
                    f"(Enter = {default_idx}: {candidates[default_idx].name}, least busy): ").strip()
        if raw == "":
            return candidates[default_idx]
        if raw.isdigit() and 0 <= int(raw) < len(rows):
            return candidates[int(raw)]
        print("  Invalid selection — enter a row number or press Enter for the default.")





def main():
    import argparse
    ap = argparse.ArgumentParser(description="Deploy (1+x²)(1+y²) flag-qubit QEC on Heron r2")
    ap.add_argument('--shots', type=int, default=200, help='shots per job')
    ap.add_argument('--rounds', type=int, default=4, help='syndrome extraction rounds (must be ≥4 for projection)')
    ap.add_argument('--clean-stats', action='store_true',
                    help='print aggregate statistics of all saved clean shots (legacy)')
    ap.add_argument('--buffer', action='store_true',
                    help='use buffer-plane (CX via spare qubits) for zero-SWAP routing')
    ap.add_argument('--share-pairs', action='store_true',
                    help='measure each weight-2 vertical pair once and reassemble '
                         'plaquettes classically: CZ/round 4*r*s->2*r*s (direct) or '
                         '8*r*s->2*r*s (vs buffer), half the ancillas, identical syndrome')
    ap.add_argument('--readout', action='store_true',
                    help='measure all data qubits in final round for ground-truth logical error detection')
    ap.add_argument('--backend', '-b', type=str, default=None, metavar='NAME',
                    help='submit to this backend by name (skips the interactive chooser)')
    ap.add_argument('--list-backends', action='store_true',
                    help='list available QPUs (name, qubits, queue depth) and exit')
    ap.add_argument('--dry-run', action='store_true',
                    help='transpile only, print stats, do not submit')
    ap.add_argument('--check-job', type=str, default=None, metavar='JOB_ID',
                    help='check status and queue position of a submitted job, then exit')
    opts = ap.parse_args()

    if opts.clean_stats:
        return clean_stats()

    token = get_token()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from pw_qiskit import PlaneWarp

    r, s = 6, 8

    # ---------- check a submitted job ----------
    if opts.check_job:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        job = service.job(opts.check_job)
        st = str(job.status())
        msg = f"Job {opts.check_job[:12]}...  status={st}"
        try:
            b = job.backend()
            bst = b.status()
            msg += f"  backend={b.name}  backend_queue={bst.pending_jobs}"
        except Exception:
            pass
        print(msg)
        if st == "DONE":
            print("  Result available — use deploy_retrieve.py to decode.")
        return
    rounds = opts.rounds      # syndrome extraction rounds (must be ≥4)
    shots = opts.shots        # shots per job
    use_buffer = opts.buffer  # buffer-plane routing via spare qubits
    share_pairs = opts.share_pairs  # share weight-2 pair measurements across plaquettes
    if share_pairs and use_buffer:
        print("Note: --share-pairs now uses direct CX (no spare); --buffer does not conflict.")
        use_buffer = False
    dry_run = opts.dry_run    # transpile only, no submit
    # resilience_level=2 applies ZNE (3 noise factors → 3× execution cost).
    # Without it, ~2.3 data errors/round overwhelm the distance-3 code.

    # ---------- pick backend ----------
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = select_backend(service, opts, min_qubits=156)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    # ---------- build ----------
    # Data readout is NOT used for coset recovery:
    #   canonicalize applies sub-lattice row/col flips which ARE the logical operators,
    #   so it zeros out the very k we're trying to recover.
    #   Even parity-based recovery fails at Heron's ~1% readout noise
    #   (expected ~1 false sub-lattice parity flip per shot).
    # Single-observable LER (~2% at Heron noise) is already characterized.
    final_readout = opts.readout
    if share_pairs:
        cx_per_round = 2 * r * s   # 2 CZ/pair direct CX, r*s pairs, all local
    elif use_buffer:
        cx_per_round = 8 * r * s
    else:
        cx_per_round = 4 * r * s
    print("Building 6x8 flag circuit ...")
    qc, _, _ = build_flag_circuit(r, s, rounds, final_data_readout=final_readout,
                                  use_buffer=use_buffer, share_pairs=share_pairs)
    print(f"  Virtual qubits: {qc.num_qubits}")
    mode_label = 'via buffer-plane' if (use_buffer or share_pairs) else 'direct'
    print(f"  CX / round:     {cx_per_round}  ({mode_label})")

    # ---------- transpile ----------
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import SetLayout, SabreLayout, SabreSwap, BasisTranslator
    from qiskit.transpiler.layout import Layout
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as SEL

    cm = backend.target.build_coupling_map()
    basis_gates = list(backend.target.operation_names)

    # Build initial layout from physical topology: data→degree-4, flag→degree-2.
    from collections import Counter
    deg = Counter()
    for a, b in cm:
        deg[a] += 1
        deg[b] += 1
    d4 = sorted([q for q, d in deg.items() if d == 4])
    d2 = sorted([q for q, d in deg.items() if d == 2])
    extra = sorted(set(deg.keys()) - set(d4) - set(d2))
    flags_d2 = sorted(d2 + extra)[:2 * r * s]
    phys_layout = d4[:r * s] + flags_d2[:2 * r * s]
    while len(phys_layout) < qc.num_qubits:
        phys_layout.append(max(deg.keys()) + 1 + len(phys_layout))
    qregs = qc.qregs
    initial_layout = Layout.from_intlist(phys_layout[:qc.num_qubits], *qregs)

    print("Transpiling (SetLayout → SabreLayout x500 → SabreSwap x500) ...")
    pm = PassManager()
    pm.append(SetLayout(initial_layout))
    pm.append(SabreLayout(backend.target, max_iterations=500, seed=0))
    pm.append(SabreSwap(backend.target, trials=500))
    pm.append(BasisTranslator(SEL, basis_gates))
    qc_t = pm.run(qc)
    ops = qc_t.count_ops()
    # Count all two-qubit gates (CZ, ECR, CX depending on backend basis)
    two_q = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'swap'))
    ecr = ops.get("ecr", 0)   # native 2q count for the saved record (Heron uses cz, so usually 0)
    # On Heron, Sabre decomposes routing SWAPs into 3 CZ each *before* emitting the
    # final circuit, so ops.get("swap") is always 0 and gives a false "0 SWAPs" read.
    # Infer the implied routing SWAP count from the 2Q overhead instead.
    baseline_2q = cx_per_round * rounds
    overhead_2q = max(0, two_q - baseline_2q)
    implied_swaps = overhead_2q // 3   # each routing SWAP → 3 CZ in native basis
    print(f"  Physical qubits: {qc_t.num_qubits}")
    print(f"  Depth:           {qc_t.depth()}")
    print(f"  Two-qubit gates: {two_q}  (baseline: {baseline_2q},  overhead: +{overhead_2q})")
    print(f"  Implied routing SWAPs: {implied_swaps}  (overhead // 3; ops['swap']=0 is always misleading on Heron)")
    print(f"  Gate breakdown:  {dict((k,v) for k,v in ops.items() if v > 0)}")
    if implied_swaps > 0:
        print(f"  WARNING: ~{implied_swaps} implied routing SWAPs ({overhead_2q} extra CZ) — "
              f"ancilla not fully local on this layout; try --buffer or --share-pairs.")
    if two_q > baseline_2q * 1.5:
        print("  WARNING: >50% overhead — spare placement may not match topology.")

    if dry_run:
        print("\nDry run complete. Submit with `--share-pairs` (omit `--dry-run`).")
        return

    # ---------- submit via SamplerV2 ----------
    print(f"\nSubmitting {shots} shots x {rounds} rounds ...")
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_t], shots=shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Dashboard: https://quantum.ibm.com/jobs/{job_id}")

    # Persist for retrieval
    jobs = {}
    if SAVE_FILE.exists():
        try:
            jobs = json.loads(SAVE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"Warning: could not read {SAVE_FILE}, starting fresh.", file=sys.stderr)
            jobs = {}
    jobs[job_id] = {
        "r": r, "s": s, "rounds": rounds, "shots": shots,
        "backend": backend.name, "twirling": False, "submitted": time.time(),
    }
    SAVE_FILE.write_text(json.dumps(jobs, indent=2))
    print(f"  Saved to {SAVE_FILE}")

    # ---------- wait (detachable) ----------
    print("\nWaiting for result (Ctrl+C to detach) ...")
    try:
        result = job.result()
    except KeyboardInterrupt:
        print("\nDetached (job still queued / running on IBM).")
        print(f"  Retrieve:  python3 deploy_retrieve.py {job_id}")
        sys.exit(0)

    # ---------- decode all shots ----------
    pub_result = result[0]
    extract = all_syndromes_shared if share_pairs else all_syndromes
    all_syn = extract(pub_result, rounds, r, s)
    n_shots = all_syn.shape[0]

    # Per-round syndrome diagnostics
    for c in range(rounds):
        wt = all_syn[:, c, :, :].sum() / n_shots
        print(f"  Round {c} syn wt: {wt:.1f} / {r*s} ({100*wt/(r*s):.1f}%)")

    # Extract data readout if present
    data_raw = None
    if hasattr(pub_result.data, "data"):
        dbits = getattr(pub_result.data, "data").to_bool_array(order='little')
        data_raw = dbits.astype(np.uint8).reshape(n_shots, r, s)

    # ---------- Consistency projection + basis decode ----------
    pw = PlaneWarp()
    decode_ok = 0
    decode_corr = np.zeros((n_shots, r, s), dtype=np.uint8)
    n_clean_list = []
    and_all = all_syn.all(axis=1).astype(np.uint8)
    and_decode = 0
    total_and = 0  # AND decoded count (for comparison)
    proj_fail_clean = 0  # failed due to <24 clean bits
    proj_fail_solve = 0  # failed due to inconsistent system
    proj_fail_decode = 0 # projection succeeded but decoder returned ok=False

    # Load 24-dim basis for column-space decoding
    bp24 = CLEAN_DIR / 'basis_24.npz'
    if not bp24.exists():
        print(f"  ERROR: 24-dim basis not found at {bp24}. Run build_basis.py first.", file=sys.stderr)
        sys.exit(1)
    bd24 = np.load(bp24)
    basis_syn = [np.asarray(bd24['syn'][i], dtype=np.uint8).reshape(r, s) for i in range(len(bd24['syn']))]
    basis_corr = [np.asarray(bd24['corr'][i], dtype=np.uint8).reshape(r, s) for i in range(len(bd24['corr']))]
    print(f"  Basis loaded: {len(basis_syn)} dims (full column space)")

    # Use last 4 rounds for projection
    proj_rounds = min(4, rounds)
    for idx in range(n_shots):
        # AND decode (for comparison)
        C_and, ok_and = pw.decode_linear_basis(basis_syn, basis_corr, and_all[idx])
        if ok_and: and_decode += 1

        # Projection decode
        rounds_4d = np.stack([all_syn[idx, c] for c in range(rounds - proj_rounds, rounds)])
        C, ok = pw.project_decode(basis_syn, basis_corr, rounds_4d)
        if ok:
            decode_ok += 1
            decode_corr[idx] = C

    # Single-observable LER from data readout
    single_err = 0
    and_single_err = 0
    proj_perf = 0
    proj_det = 0
    proj_log = 0
    if data_raw is not None:
        for idx in range(n_shots):
            E = data_raw[idx]
            # Projection LER
            C = decode_corr[idx]
            if C.sum() > 0 or True:  # correction was applied (or not)
                res = (E ^ C).astype(np.uint8)
                sr = pw.syndrome_of(res)
                if sr.sum() == 0:
                    if pw.is_stabilizer(res):
                        proj_perf += 1
                    else:
                        proj_log += 1
                else:
                    proj_det += 1
            # AND LER
            C_and, _ = pw.decode_linear_basis(basis_syn, basis_corr, and_all[idx])
            res_and = (E ^ C_and).astype(np.uint8)
            sr_and = pw.syndrome_of(res_and)
            and_log = (sr_and.sum() == 0 and not pw.is_stabilizer(res_and))
            if and_log:
                and_single_err += 1

    print(f"\n=== Heron r2 results ===")
    print(f"  Grid:     {r}×{s}")
    print(f"  Rounds:   {rounds} (last {proj_rounds} for projection)")
    print(f"  Shots:    {n_shots}")
    print(f"  Method:   consistency projection → Col(H) → basis decode")
    print()
    print(f"  Projection decode:  {decode_ok}/{n_shots} ({100*decode_ok/n_shots:.1f}%)")
    print(f"  AND decode:         {and_decode}/{n_shots} ({100*and_decode/n_shots:.1f}%)  (reference)")
    print()
    if data_raw is not None:
        print(f"  Projection outcomes ({decode_ok} decoded):")
        print(f"    Perfect (E⊕C=0):         {proj_perf} ({100*proj_perf/max(1,decode_ok):.1f}%)")
        print(f"    Detectable (syn≠0):       {proj_det} ({100*proj_det/max(1,decode_ok):.1f}%)")
        print(f"    Logical (syn=0, not stab): {proj_log} ({100*proj_log/max(1,decode_ok):.1f}%)")
        print(f"  True LER (undetectable):    {100*proj_log/n_shots:.2f}%")
        print(f"  AND LER:                    {100*and_single_err/n_shots:.2f}%  (reference)")

    jobs[job_id].update({
        "completed": time.time(),
        "shots_completed": n_shots,
        "rounds": rounds,
        "proj_decode_pct": round(100 * decode_ok / n_shots, 1),
        "and_decode_pct": round(100 * and_decode / n_shots, 1),
        "proj_perf": int(proj_perf),
        "proj_det": int(proj_det),
        "proj_log": int(proj_log),
        "true_ler_pct": round(100 * proj_log / n_shots, 2),
        "round_syn_wt": [float(all_syn[:,c,:,:].sum() / n_shots) for c in range(rounds)],
        "two_q_count": two_q,
        "ecr_count": ecr,
        "implied_swaps": implied_swaps,
        "overhead_2q": overhead_2q,
    })
    SAVE_FILE.write_text(json.dumps(jobs, indent=2))
    print(f"\nResults saved to {SAVE_FILE}")


if __name__ == "__main__":
    main()
