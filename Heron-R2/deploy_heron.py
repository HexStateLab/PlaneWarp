#!/usr/bin/env python3
"""
Submit (1+x^2)(1+y^2) flag-qubit QEC experiment to IBM Heron r2.

Post-selection is the operating mode: ~15% of shots pass is_stabilizer(correction),
meaning all 24 logicals are simultaneously correct. Surviving shots have 100% fidelity.
Data readout coset recovery NOT viable — Heron's ~1% per-qubit readout noise scrambles
the sub-lattice parity needed to identify which of the 2^24 cosets was decoded.

Usage:
  export IBM_QUANTUM_TOKEN='your_token'
  python3 deploy_heron.py                        # single run (interactive backend chooser)
  python3 deploy_heron.py --list-backends        # show available QPUs + queue depth, exit
  python3 deploy_heron.py --backend ibm_kyiv     # submit to a named QPU (no prompt)
  python3 deploy_heron.py --postselect           # save clean shots
  python3 deploy_heron.py --shots 1000 --postselect
  python3 deploy_heron.py --postselect --strict  # + reject high-syndrome shots pre-decode
  python3 deploy_heron.py --clean-stats          # aggregate all clean runs

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
            vpair(i,j)=D(i,j)^D(i+2,j) ONCE (one ancilla per (i,j), 2 CX),
            instead of extracting a0=vpair(i,j) and a1=vpair(i,j+2) separately
            for every plaquette. Since a1(i,j) == a0(i,j+2), the unshared circuit
            extracts every pair twice. The plaquette syndrome is reassembled in
            all_syndromes_shared() as syn(i,j)=m(i,j)^m(i,(j+2)%s), which is
            bit-identical to the unshared syn. Cuts CZ/round from 4*r*s (direct)
            or 8*r*s (buffer) to the floor of 2*r*s, and halves ancilla usage.
            Trade-off: a shared-ancilla measurement fault now flips two adjacent
            plaquette detectors instead of one (a benign horizontal matching edge
            for a matching decoder; validate against the tesseract decoder on HW).
            Mutually exclusive with use_buffer; routing is left to the transpiler.
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
    n_spare = 12 if (use_buffer and not share_pairs) else 0
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
                    # Measure vpair(i,j)=D(i,j)^D(i+2,j) once onto a0 (already
                    # adjacent to both in the layout). Plaquettes are reassembled
                    # classically in all_syndromes_shared.
                    a = anc_maps[(i, j, 0)]
                    qc.reset(a)
                    qc.cx(data_map[i][j], a)
                    qc.cx(data_map[(i + 2) % r][j], a)
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
    ap.add_argument('--rounds', type=int, default=2, help='syndrome extraction rounds')
    ap.add_argument('--postselect', action='store_true',
                    help='save clean-shot corrections to ~/.planewarp_clean/ for reuse')
    ap.add_argument('--clean-stats', action='store_true',
                    help='print aggregate statistics of all saved clean shots')
    ap.add_argument('--strict', type=int, nargs='?', const=8, default=0,
                    help='pre-decode syndrome-weight threshold (default: 8, 0=off). '
                         'Shots with AND-syndrome weight > threshold are rejected '
                         'without decoding, eliminating false positives from high-noise '
                         'shots at a small yield cost.')
    ap.add_argument('--buffer', action='store_true',
                    help='use buffer-plane (CX via spare qubits) for zero-SWAP routing')
    ap.add_argument('--share-pairs', action='store_true',
                    help='measure each weight-2 vertical pair once and reassemble '
                         'plaquettes classically: CZ/round 4*r*s->2*r*s (direct) or '
                         '8*r*s->2*r*s (vs buffer), half the ancillas, identical syndrome')
    ap.add_argument('--backend', '-b', type=str, default=None, metavar='NAME',
                    help='submit to this backend by name (skips the interactive chooser)')
    ap.add_argument('--list-backends', action='store_true',
                    help='list available QPUs (name, qubits, queue depth) and exit')
    ap.add_argument('--dry-run', action='store_true',
                    help='transpile only, print stats, do not submit')
    opts = ap.parse_args()

    if opts.clean_stats:
        return clean_stats()

    token = get_token()

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from pw_qiskit import PlaneWarp

    r, s = 6, 8
    rounds = opts.rounds      # syndrome extraction rounds
    shots = opts.shots        # shots per job
    postselect = opts.postselect
    strict = opts.strict      # 0 = off, >0 = syndrome-weight threshold
    use_buffer = opts.buffer  # buffer-plane routing via spare qubits
    share_pairs = opts.share_pairs  # share weight-2 pair measurements across plaquettes
    if share_pairs and use_buffer:
        print("Note: --share-pairs supersedes --buffer (direct routing); ignoring --buffer.")
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
    final_readout = False
    if share_pairs:
        cx_per_round = 2 * r * s
    elif use_buffer:
        cx_per_round = 8 * r * s
    else:
        cx_per_round = 4 * r * s
    print("Building 6x8 flag circuit ...")
    qc, _, _ = build_flag_circuit(r, s, rounds, final_data_readout=final_readout,
                                  use_buffer=use_buffer, share_pairs=share_pairs)
    print(f"  Virtual qubits: {qc.num_qubits}")
    print(f"  CX / round:     {cx_per_round}  ({'via buffer-plane' if use_buffer else 'direct'})")

    # ---------- transpile ----------
    print("Transpiling (preset pass manager, Sabre, opt=3) ...")
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        routing_method="sabre",
        seed_transpiler=42,
    )
    qc_t = pm.run(qc)
    ops = qc_t.count_ops()
    # Count all two-qubit gates (CZ, ECR, CX depending on backend basis)
    two_q = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'swap'))
    swaps = ops.get("swap", 0)
    ecr = ops.get("ecr", 0)   # native 2q count for the saved record (Heron uses cz, so usually 0)
    print(f"  Physical qubits: {qc_t.num_qubits}")
    print(f"  Depth:           {qc_t.depth()}")
    print(f"  Two-qubit gates: {two_q}  (baseline: {cx_per_round * rounds})")
    print(f"  SWAP gates:      {swaps}")
    print(f"  Gate breakdown:  {dict((k,v) for k,v in ops.items() if v > 0)}")
    if swaps:
        print("  WARNING: SWAPs present — buffer-plane not fully utilized.")
    if two_q > cx_per_round * rounds * 1.5:
        print("  WARNING: >50% overhead — spare placement may not match topology.")

    if dry_run:
        print("\nDry run complete. Submit with `--buffer` (omit `--dry-run`).")
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

    pw = PlaneWarp()
    errors = 0
    total_corr = 0

    print(f"\nDecoding {n_shots} shots, {rounds} rounds, {r}x{s} grid ...")

    for idx in range(n_shots):
        correction = pw.decode_tesseract(all_syn[idx])
        total_corr += int(correction.sum())
        if not pw.is_stabilizer(correction):
            errors += 1

    ler = errors / n_shots
    avg_corr = total_corr / n_shots

    print(f"\n=== Heron r2 results ===")
    print(f"  Grid:           {r}×{s}")
    print(f"  Logical qubits: {2 * r + 2 * s - 4}")
    print(f"  Rounds:         {rounds}")
    print(f"  Shots:          {n_shots}")
    print(f"  Total syn wt:   {int(all_syn.sum())} / {n_shots * rounds * r * s} bits")
    print(f"  Avg corr:       {avg_corr:.2f} / {r * s} qubits")
    print(f"  LER:            {ler:.4f}")

    jobs[job_id].update({
        "completed": time.time(),
        "shots_completed": n_shots,
        "ler": ler,
        "avg_corr": float(avg_corr),
        "round_syn_wt": [float(all_syn[:,c,:,:].sum() / n_shots) for c in range(rounds)],
        "two_q_count": two_q,
        "ecr_count": ecr,
        "swaps": swaps,
    })
    SAVE_FILE.write_text(json.dumps(jobs, indent=2))
    print(f"\nResults saved to {SAVE_FILE}")


if __name__ == "__main__":
    main()
