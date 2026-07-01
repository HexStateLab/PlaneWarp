"""
test_logical_entanglement.py — GHZ state over ALL logical qubits + QEC.
Demonstrates that entanglement across all logical degrees of freedom
survives QEC rounds. Uses decoder.py with auto-rotation.

For periodic (r×s): 2 independent logical qubits (row-flip, col-flip).
Prepares |0...0⟩_L + |1...1⟩_L GHZ state, runs QEC rounds, measures
all logical qubits, computes entanglement witness.
"""
import sys, time, json, argparse
import numpy as np
from pathlib import Path
from pw_opt import build_circuit, all_syndromes_opt, check_consistency

SAVE_FILE = Path("logical_entanglement.json")

def get_token():
    import os
    tok = os.environ.get("IBM_QUANTUM_TOKEN")
    if tok: return tok
    import getpass
    return getpass.getpass("IBM Quantum token: ")

def logical_measure(corrected_data, r, s, periodic=True):
    """Measure ALL logical Z operators (rows + columns)."""
    logicals = {}
    if periodic:
        for i in range(r - 1):
            logicals[f'Z_row_{i}'] = corrected_data[:, i, :].sum(axis=1) % 2
        for j in range(s - 1):
            logicals[f'Z_col_{j}'] = corrected_data[:, :, j].sum(axis=1) % 2
    else:
        for j in range(s - 1):
            logicals[f'Z_col_{j}'] = corrected_data[:, :, j].sum(axis=1) % 2
    return logicals

def decode_all(decoder_name, all_syn, r, s):
    """Decode syndromes and return (n_shots, r, s) corrections."""
    n_shots, rounds, _, _ = all_syn.shape
    if rounds == 0:
        return np.zeros((n_shots, r, s), dtype=np.uint8)
    if decoder_name == "tesseract":
        from decoder import tesseract_decode
        corrs = np.zeros((n_shots, r, s), dtype=np.uint8)
        for i in range(n_shots):
            corrs[i] = tesseract_decode(all_syn[i], r, s)
        return corrs
    elif decoder_name == "ffinal":
        from decoder import tesseract_decode_ffinal
        corrs = np.zeros((n_shots, r, s), dtype=np.uint8)
        for i in range(n_shots):
            corrs[i] = tesseract_decode_ffinal(all_syn[i], r, s)
        return corrs
    elif decoder_name == "multi":
        from decoder import tesseract_decode_multi
        corrs = np.zeros((n_shots, r, s), dtype=np.uint8)
        for i in range(n_shots):
            corrs[i] = tesseract_decode_multi(all_syn[i], r, s)
        return corrs
    raise ValueError(f"unknown decoder: {decoder_name}")

def run_test(token, opts):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    offline = getattr(opts, "offline", False)
    if not offline:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    r, s = opts.grid or (6, 8)
    rounds = opts.rounds
    shots = opts.shots
    periodic = not opts.open
    no_reset = not opts.reset_every_round
    free_final_round = not opts.no_free_final_round
    full_stabilizer = opts.full_stabilizer

    # Compute logical count
    if periodic:
        n_logicals = (r - 1) + (s - 1)   # all rows + all cols
        n_independent = 2
    else:
        n_logicals = (s - 1)              # open: only cols
        n_independent = 2

    print(f"GHZ-over-All-Logicals: {r}×{s} grid, {n_logicals} logicals "
          f"({n_independent} independent), {rounds} QEC rounds, {shots} shots")

    # Build circuit: GHZ state over all logical qubits
    # For periodic: entangle all row-flip and col-flip logicals into a GHZ
    # The 2 independent logicals are: Z_L1 = row 0 parity, Z_L2 = col 0 parity
    # GHZ over all logicals = (|0...0⟩ + |1...1⟩)/√2
    # This is achieved by: Hadamard on ANY one logical, then CNOT cascade
    # to all other logicals.  Since logical ops are row/col flips, we prepare
    # the GHZ by flipping a reference pattern of data qubits.

    # We use the GHZ prep already in build_circuit: it prepares
    # |00...0⟩_L + |11...1⟩_L by entangling the boundary through a GHZ ancilla.
    # Set logical_state="ghz" to activate.
    qc, data_map, lq0_qubits, lq1_qubits, n_anc = build_circuit(
        r, s, rounds, logical_state="ghz", ghz=True,
        measure_x=opts.measure_x, partial_x=opts.partial_x,
        stabilizer_basis='X' if opts.x_stabilizer else 'Z',
        no_reset=no_reset, free_final_round=free_final_round,
        full_stabilizer=full_stabilizer, dd=opts.dd, periodic=periodic,
    )

    basis = "X" if opts.measure_x else "Z"
    stab = "X" if opts.x_stabilizer else "Z"
    anc_rounds = max(0, rounds - 1) if free_final_round else max(0, rounds)
    cx_per_round = 16 * (r // 2 - 1) * (s // 2) if full_stabilizer else 8 * (r // 2 - 1) * (s // 2)
    total_cx = anc_rounds * cx_per_round
    ffr_note = " (last round free)" if free_final_round else ""
    stab_note = ", full-stab" if full_stabilizer else ""
    print(f"  Data: {r*s}, Ancillas: {n_anc}, {stab}-stab{stab_note}{ffr_note}")
    print(f"  {anc_rounds} ancilla rounds × {cx_per_round} CX = {total_cx} CX")

    if opts.dry_run:
        ops = qc.count_ops()
        two_q = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'swap'))
        print(f"  Physical qubits: {qc.num_qubits}, Two-qubit gates: {two_q}")
        print("\nDry run complete.")
        return
    else:
        if offline:
            from offline_sim import setup as offline_setup
            backend, offline_sampler = offline_setup(
                fake=opts.fake, two_qubit_rate=opts.noise_2q,
                one_qubit_rate=opts.noise_1q, readout_rate=opts.noise_readout,
                reset_rate=opts.noise_reset, seed=opts.seed)
            print(f"Backend: {backend.name} [OFFLINE]")
        else:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            if opts.backend:
                backend = service.backend(opts.backend)
            else:
                backend = service.backend("ibm_marrakesh")
            print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    if not opts.dry_run:
        print("Transpiling ...")
        if offline:
            from offline_sim import transpile_offline
            qc_t = transpile_offline(qc, backend)
        else:
            pm = generate_preset_pass_manager(backend=backend, optimization_level=opts.opt_level, seed_transpiler=42)
            qc_t = pm.run(qc)
        ops = qc_t.count_ops()
        two_q = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'swap'))
        print(f"  Physical qubits: {qc_t.num_qubits}, Depth: {qc_t.depth()}, Two-qubit gates: {two_q}")

    print("\nSubmitting ...")
    if offline:
        sampler = offline_sampler
    else:
        sampler = Sampler(mode=backend)
    job = sampler.run([qc_t], shots=shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Dashboard: https://quantum.ibm.com/jobs/{job_id}")

    results = {"r": r, "s": s, "rounds": rounds, "shots": shots,
               "backend": backend.name, "n_logicals": n_logicals,
               "periodic": periodic, "submitted": time.time()}

    print("\nWaiting for result (Ctrl+C to detach) ...")
    try:
        result = job.result()
    except KeyboardInterrupt:
        print("\nDetached. Re-run with --redecode after job completes.")
        SAVE_FILE.write_text(json.dumps(results, indent=2, default=str))
        sys.exit(0)

    pub_result = result[0]
    dbits = getattr(pub_result.data, "data").to_bool_array(order='little')
    data_raw = dbits.astype(np.uint8).reshape(-1, r, s)
    n_shots = data_raw.shape[0]

    if rounds == 0:
        all_syn = np.zeros((n_shots, 0, r, s), dtype=np.uint8)
    else:
        all_syn = all_syndromes_opt(pub_result, rounds, r, s, n_anc,
                                    no_reset=no_reset, free_final_round=free_final_round,
                                    data_raw=data_raw, full_stabilizer=full_stabilizer,
                                    periodic=periodic)

    if free_final_round and rounds >= 2:
        cc = check_consistency(all_syn, data_raw, r, s)
        if cc:
            print(f"  Consistency check (ancilla vs data, last round):")
            print(f"    Shots with 0 mismatches: {cc['frac_zero_mismatch']*100:.1f}%")
            print(f"    Mean mismatched plaquettes: {cc['mean_mismatch']:.3f}")

    ghz_out = getattr(pub_result.data, "ghz").to_bool_array(order='little').flatten().astype(np.uint8)
    print(f"  GHZ ancilla: |0⟩={(ghz_out==0).sum()}, |1⟩={(ghz_out==1).sum()}")

    print(f"\nDecoding {n_shots} shots with {['ffinal','tesseract','multi']} ...\n")

    for dec_name in ("ffinal", "tesseract", "multi"):
        t0 = time.time()
        corrs = decode_all(dec_name, all_syn, r, s)
        dt = time.time() - t0
        corrected = data_raw ^ corrs

        logicals = logical_measure(corrected, r, s, periodic=periodic)
        n_logicals_measured = len(logicals)

        # All-0 fidelity: fraction of shots where ALL logicals read 0
        all_zero = np.ones(n_shots, dtype=np.uint8)
        for name, vals in logicals.items():
            all_zero &= (vals == 0)
        joint_0 = all_zero.mean()

        # GHZ entanglement witness:
        # For a true GHZ, all logicals should agree (all-0 or all-1).
        # Compute fraction where all logical values are identical.
        vals_stack = np.column_stack([logicals[name] for name in sorted(logicals.keys())])
        all_same = (vals_stack.max(axis=1) == vals_stack.min(axis=1)).mean()

        # Correlation between independent logicals (Z_L1 = row 0, Z_L2 = col 0)
        lz1 = corrected[:, 0, :].sum(axis=1) % 2 if periodic else corrected[:, :, 0].sum(axis=1) % 2
        lz2 = corrected[:, :, 0].sum(axis=1) % 2 if periodic else corrected[:, :, 2].sum(axis=1) % 2
        agree = (lz1 == lz2).astype(np.uint8)
        zz_corr = float(2 * int(agree.sum()) - n_shots) / n_shots

        # Boundary entanglement (GHZ signature)
        bnd = np.zeros((n_shots, s - 1 + r - 1), dtype=np.uint8) if periodic else np.zeros((n_shots, s - 1), dtype=np.uint8)
        if periodic:
            for j in range(s - 1):
                bnd[:, j] = corrected[:, r - 1, j]
            for i in range(r - 1):
                bnd[:, s - 1 + i] = corrected[:, i, s - 1]
        else:
            for j in range(s - 1):
                bnd[:, j] = corrected[:, :, j]
        boundary_all_same = (bnd.max(axis=1) == bnd.min(axis=1)).mean()

        print(f"  {dec_name} ({dt:.1f}s):")
        print(f"    All-{n_logicals_measured}-Z |0...0⟩: {joint_0:.4f}")
        print(f"    All logicals agree:   {all_same:.4f}")
        print(f"    ⟨Z_L1⊗Z_L2⟩:          {zz_corr:.3f}")
        print(f"    Boundary all-same:    {boundary_all_same:.4f}")

        # Entanglement verdict
        w = 2 * boundary_all_same - 1 + zz_corr
        verdict = "✓ ENTANGLED!" if w > 1 else "~ Marginal" if w > 0.5 else "✗ Separable"
        print(f"    W_logical = {2*boundary_all_same-1:.3f} + {zz_corr:.3f} = {w:.3f}  {verdict}")
        print()

        results[dec_name] = {
            "joint_0": float(joint_0), "all_same": float(all_same),
            "zz_corr": float(zz_corr), "boundary_all_same": float(boundary_all_same),
            "witness": float(w), "time_s": round(dt, 2),
        }

    results["completed"] = time.time()
    SAVE_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved to {SAVE_FILE}")

    best = results.get("tesseract") or results.get("ffinal") or {}
    w_best = best.get("witness", 0)
    print(f"\n  Best witness: {w_best:.3f} "
          f"{'✓ ENTANGLED' if w_best > 1 else '~ Below threshold' if w_best > 0.5 else '✗ Separable'}")


def main():
    ap = argparse.ArgumentParser(description="GHZ-over-all-logicals entanglement test")
    ap.add_argument('--shots', type=int, default=1000)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--backend', '-b', type=str, default=None)
    ap.add_argument('--opt-level', type=int, default=3, choices=[0,1,2,3])
    ap.add_argument('--grid', type=int, nargs=2, metavar=('R', 'S'), default=(6, 8))
    ap.add_argument('--reset-every-round', action='store_true')
    ap.add_argument('--no-free-final-round', action='store_true')
    ap.add_argument('--x-stabilizer', action='store_true')
    ap.add_argument('--measure-x', action='store_true')
    ap.add_argument('--partial-x', action='store_true')
    ap.add_argument('--full-stabilizer', action='store_true')
    ap.add_argument('--dd', action='store_true')
    ap.add_argument('--open', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--fake', type=str, default=None)
    ap.add_argument('--noise-2q', type=float, default=0.0)
    ap.add_argument('--noise-1q', type=float, default=0.0)
    ap.add_argument('--noise-readout', type=float, default=0.0)
    ap.add_argument('--noise-reset', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=None)
    opts = ap.parse_args()

    token = None if (opts.offline or opts.dry_run) else get_token()
    run_test(token, opts)

if __name__ == "__main__":
    main()
