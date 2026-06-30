"""
run_opt.py — Experiment runner using pw_opt.py's optimized share-pair builder.

Imports build_circuit and all_syndromes_opt from pw_opt for the circuit
construction and syndrome extraction. All analysis (logical fidelity,
entanglement witness, all-logicals) lives here.
"""
import sys, time, json, argparse
import numpy as np
from pathlib import Path
from pw_opt import build_circuit, all_syndromes_opt, check_consistency

SAVE_FILE = Path("jobs.json")
RAW_DIR = Path("raw")

def get_token():
    import os
    tok = os.environ.get("IBM_QUANTUM_TOKEN")
    if tok:
        return tok
    import getpass
    return getpass.getpass("IBM Quantum token: ")

def logical_measure(corrected_data, r, s, periodic=True):
    if periodic:
        lz1 = corrected_data[:, 0, :].sum(axis=1) % 2
        lz2 = corrected_data[:, :, 0].sum(axis=1) % 2
    else:
        lz1 = corrected_data[:, :, 0].sum(axis=1) % 2
        lz2 = corrected_data[:, :, 2].sum(axis=1) % 2
    return lz1, lz2

def all_logicals_measure(corrected_data, r, s, basis='Z', periodic=True):
    logicals = {}
    if basis == 'Z':
        if periodic:
            for i in range(r - 1):
                logicals[f'Z_row_{i}'] = corrected_data[:, i, :].sum(axis=1) % 2
            for j in range(s - 1):
                logicals[f'Z_col_{j}'] = corrected_data[:, :, j].sum(axis=1) % 2
        else:
            for j in range(s - 1):
                logicals[f'Z_col_{j}'] = corrected_data[:, :, j].sum(axis=1) % 2
    elif basis == 'X':
        corrected_X = corrected_data.copy()
        corrected_X ^= 1
        if periodic:
            for i in range(r - 1):
                logicals[f'X_row_{i}'] = corrected_X[:, i, :].sum(axis=1) % 2
            for j in range(s - 1):
                logicals[f'X_col_{j}'] = corrected_X[:, :, j].sum(axis=1) % 2
        else:
            for j in range(s - 1):
                logicals[f'X_col_{j}'] = corrected_X[:, :, j].sum(axis=1) % 2
    return logicals

def compute_fidelity(lz1, lz2, z1, z2):
    return ((lz1 == z1) & (lz2 == z2)).mean()

def decode(decoder_name, all_syn, r, s):
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
    raise ValueError(f"unknown decoder: {decoder_name}")

def run_test(token, opts):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    offline = getattr(opts, "offline", False)
    if not offline:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    r, s = opts.grid or (6, 8)
    rounds = opts.rounds
    shots = opts.shots
    logical_state = opts.state
    bell = (logical_state == "bell")
    ghz = (logical_state == "ghz")
    bell_measure = opts.bell_measure
    bell_after_qec = opts.bell_after_qec
    ghz_measure = opts.ghz_measure
    no_reset = not opts.reset_every_round
    free_final_round = not opts.no_free_final_round
    full_stabilizer = opts.full_stabilizer
    periodic = not opts.open

    if free_final_round:
        readout_is_x = opts.measure_x
        stab_is_x = opts.x_stabilizer
        if readout_is_x != stab_is_x:
            print("WARNING: --no-free-final-round forced: readout basis != stabilizer basis")
            free_final_round = False
        if opts.partial_x:
            print("WARNING: --no-free-final-round forced: partial_x is mixed basis")
            free_final_round = False

    offline_sampler = None
    if opts.dry_run:
        backend = None
        print("Backend: [dry-run]")
    elif offline:
        from offline_sim import setup as offline_setup
        backend, offline_sampler = offline_setup(
            fake=opts.fake,
            two_qubit_rate=opts.noise_2q,
            one_qubit_rate=opts.noise_1q,
            readout_rate=opts.noise_readout,
            reset_rate=opts.noise_reset,
            seed=opts.seed,
        )
        src = f"fake={opts.fake}" if opts.fake else (
            f"2q={opts.noise_2q} 1q={opts.noise_1q} ro={opts.noise_readout} rst={opts.noise_reset}"
        )
        print(f"Backend: {backend.name} [OFFLINE simulator, noise: {src}]")
    else:
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        if opts.backend:
            backend = service.backend(opts.backend)
        else:
            backend = service.backend("ibm_marrakesh")
        print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    qc, data_map, lq0_qubits, lq1_qubits, n_anc = build_circuit(
        r, s, rounds, logical_state=logical_state, bell=bell,
        ghz=ghz, ghz_measure=ghz_measure,
        bell_measure=bell_measure, measure_x=opts.measure_x,
        partial_x=opts.partial_x,
        stabilizer_basis='X' if opts.x_stabilizer else 'Z',
        no_reset=no_reset,
        free_final_round=free_final_round,
        bell_after_qec=bell_after_qec,
        full_stabilizer=full_stabilizer,
        dd=opts.dd,
        periodic=periodic,
    )
    if opts.partial_x:
        basis = "X_partial"
    else:
        basis = "X" if opts.measure_x else "Z"
    if bell:
        label = f"Bell-{basis}{'-M' if bell_measure else ''}"
    elif ghz:
        label = f"GHZ{'zM' if ghz_measure else ''}"
    else:
        label = f"|{logical_state}⟩"
    stab = "X" if opts.x_stabilizer else "Z"
    anc_rounds = max(0, rounds - 1) if free_final_round else max(0, rounds)
    cx_per_round = 16 * (r // 2 - 1) * (s // 2) if full_stabilizer else 8 * (r // 2 - 1) * (s // 2)
    total_cx = anc_rounds * cx_per_round
    ffr_note = f" (last round free from data)" if free_final_round else ""
    stab_note = f", full-stab" if full_stabilizer else ""
    bc_str = "open" if not periodic else "periodic"
    print(f"Circuit: {r}×{s} grid ({bc_str} BC), {rounds} rounds, {label}, {shots} shots")
    print(f"  Data: {r*s}, Ancillas: {n_anc}, {stab}-stab{stab_note}, no_reset={no_reset}")
    print(f"  {anc_rounds} ancilla round{'s' if anc_rounds != 1 else ''} × {cx_per_round} CX = {total_cx} CX{ffr_note}")

    if opts.dry_run:
        ops = qc.count_ops()
        two_q = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'swap'))
        print(f"  Physical qubits: {qc.num_qubits}, Two-qubit gates: {two_q}")
        print("\nDry run complete.")
        return
    else:
        print("Transpiling ...")
        if offline:
            from offline_sim import transpile_offline
            qc_t = transpile_offline(qc, backend)
        else:
            pm = generate_preset_pass_manager(
                backend=backend, optimization_level=opts.opt_level,
                seed_transpiler=42,
            )
            qc_t = pm.run(qc)
        ops = qc_t.count_ops()
        two_q = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'swap'))
        print(f"  Physical qubits: {qc_t.num_qubits}, Depth: {qc_t.depth()}, Two-qubit gates: {two_q}")

    print(f"\nSubmitting ...")
    if offline:
        sampler = offline_sampler
    else:
        sampler = Sampler(mode=backend)
    job = sampler.run([qc_t], shots=shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Dashboard: https://quantum.ibm.com/jobs/{job_id}")

    jobs = {}
    if SAVE_FILE.exists():
        try:
            jobs = json.loads(SAVE_FILE.read_text())
        except:
            jobs = {}
    jobs[job_id] = {
        "r": r, "s": s, "rounds": rounds, "shots": shots,
        "backend": backend.name, "logical_state": logical_state,
        "bell_measure": bell_measure, "ghz_measure": ghz_measure, "no_reset": no_reset,
        "measure_x": opts.measure_x, "partial_x": opts.partial_x,
        "full_stabilizer": full_stabilizer, "periodic": periodic,
        "submitted": time.time(),
    }
    SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))

    print("\nWaiting for result (Ctrl+C to detach) ...")
    try:
        result = job.result()
    except KeyboardInterrupt:
        print("\nDetached.")
        sys.exit(0)

    pub_result = result[0]

    dbits = getattr(pub_result.data, "data").to_bool_array(order='little')
    data_raw = dbits.astype(np.uint8).reshape(-1, r, s)
    n_shots = data_raw.shape[0]

    if rounds == 0:
        all_syn = np.zeros((n_shots, 0, r, s), dtype=np.uint8)
    else:
        all_syn = all_syndromes_opt(pub_result, rounds, r, s, n_anc, no_reset=no_reset,
                                    free_final_round=free_final_round, data_raw=data_raw,
                                    full_stabilizer=full_stabilizer, periodic=periodic)

    # Consistency check: compare last ancilla round vs data-readout syndrome
    if free_final_round and rounds >= 2:
        cc = check_consistency(all_syn, data_raw, r, s)
        if cc:
            print(f"  Consistency check (ancilla vs data, last round):")
            print(f"    Shots with 0 mismatches: {cc['frac_zero_mismatch']*100:.1f}%")
            print(f"    Shots with 1 mismatch:   {cc['frac_one_mismatch']*100:.1f}%")
            print(f"    Mean mismatched plaquettes: {cc['mean_mismatch']:.3f}")

    bell_out = None
    if bell:
        bell_out = getattr(pub_result.data, "bell").to_bool_array(order='little').flatten().astype(np.uint8)
        print(f"  Bell prep: |0⟩: {(bell_out == 0).sum()}, |1⟩: {(bell_out == 1).sum()}")

    ghz_out = None
    if ghz:
        ghz_out = getattr(pub_result.data, "ghz").to_bool_array(order='little').flatten().astype(np.uint8)
        print(f"  GHZ prep: |0⟩: {(ghz_out == 0).sum()}, |1⟩: {(ghz_out == 1).sum()}")

    bell_m = None
    if bell_measure:
        bell_m = getattr(pub_result.data, "bell_m").to_bool_array(order='little').flatten().astype(np.uint8)
        print(f"  Bell measure: |0⟩ (X_L1 X_L₂=+1): {(bell_m == 0).sum()}, "
              f"|1⟩ (X_L1 X_L₂=-1): {(bell_m == 1).sum()}")

    ghz_m = None
    if ghz_measure:
        ghz_m = getattr(pub_result.data, "ghz_m").to_bool_array(order='little').flatten().astype(np.uint8)
        print(f"  GHZ measure: |0⟩ (X⊗12=+1): {(ghz_m == 0).sum()}, "
              f"|1⟩ (X⊗12=-1): {(ghz_m == 1).sum()}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = dict(syndromes=all_syn, data_raw=data_raw,
                  r=r, s=s, rounds=rounds,
                  logical_state=logical_state, measure_x=opts.measure_x,
                  partial_x=opts.partial_x,
                  bell_measure=bell_measure, ghz_measure=ghz_measure, no_reset=no_reset,
                  free_final_round=free_final_round, periodic=periodic)
    if bell_out is not None:
        kwargs["bell_out"] = bell_out
    if ghz_out is not None:
        kwargs["ghz_out"] = ghz_out
    if bell_m is not None:
        kwargs["bell_m"] = bell_m
    if ghz_m is not None:
        kwargs["ghz_m"] = ghz_m
    np.savez_compressed(RAW_DIR / f"{job_id}.npz", **kwargs)

    print(f"\nDecoding {n_shots} shots ({basis}-basis readout) ...\n")

    # For bell-after-qec: compute witness from raw data without decoder
    if bell and bell_after_qec:
        if periodic:
            lz1_raw = data_raw[:, 0, :].sum(axis=1) % 2
            lz2_raw = data_raw[:, :, 0].sum(axis=1) % 2
        else:
            lz1_raw = data_raw[:, :, 0].sum(axis=1) % 2
            lz2_raw = data_raw[:, :, 2].sum(axis=1) % 2
        agree_z = (lz1_raw == lz2_raw).astype(np.uint8)
        z_vals = np.where(agree_z, 1, -1)
        z_corr = float(z_vals.mean())
        # ⟨XX⟩ from bell measurement
        x_vals = np.where(bell_m == 0, 1, -1)
        x_corr = float(x_vals.mean())
        witness = z_corr + x_corr

        # Post-select on |Φ⁺⟩ (bell_out=0): W = ZZ + XX both = +1
        sel = (bell_out == 0)
        n_sel = sel.sum()
        z_sel = float(z_vals[sel].mean()) if n_sel > 0 else 0.0
        x_sel = float(x_vals[sel].mean()) if n_sel > 0 else 0.0
        w_sel = z_sel + x_sel

        print(f"\n  Bell witness (bell-after-qec, raw data):")
        print(f"    Bell prep: |0⟩={(bell_out==0).sum()} |1⟩={(bell_out==1).sum()}")
        print(f"    All shots:  ⟨ZZ⟩={z_corr:.3f}  ⟨XX⟩={x_corr:.3f}  W={witness:.3f}")
        print(f"    Post-sel Φ⁺ ({n_sel}/{n_shots}): ⟨ZZ⟩={z_sel:.3f}  ⟨XX⟩={x_sel:.3f}  W={w_sel:.3f}")
        info = {"Z_corr": float(z_corr), "X_corr": float(x_corr),
                "witness": float(witness), "Z_sel": float(z_sel), "X_sel": float(x_sel),
                "witness_sel": float(w_sel), "n_sel": int(n_sel),
                "basis": "bell_after_qec", "time_s": 0}
        jobs[job_id]["bell_after_qec"] = info

    decoders = ("ffinal", "tesseract")
    for dec_name in decoders:
        t0 = time.time()
        corrs = decode(dec_name, all_syn, r, s)
        dt = time.time() - t0
        corrected = data_raw ^ corrs
        lz1, lz2 = logical_measure(corrected, r, s, periodic=periodic)

        if bell:
            if opts.partial_x:
                if periodic:
                    x_flat = [0 * s + j for j in range(s)] + [i * s + 0 for i in range(1, r)]
                else:
                    x_flat = [i * s + 0 for i in range(r)] + [i * s + 2 for i in range(r)]
                x_partial = corrected.reshape(n_shots, -1)[:, x_flat]
                x_prod_vals = x_partial.sum(axis=1) % 2
                x_corr = float(2 * int((x_prod_vals == 0).sum()) - n_shots) / n_shots
                print(f"  {dec_name} ({dt:.1f}s):")
                print(f"    ⟨X_L1⊗X_L₂⟩ = {x_corr:.3f} (from {len(x_flat)} H-rotated qubits)")
                info = {"X_corr": float(x_corr), "basis": "X_partial", "time_s": round(dt, 2)}
                jobs[job_id][dec_name] = info
            elif bell_m is not None and bell_out is not None:
                # Single-run Bell witness: post-select on bell_out=0 (|Φ⁺⟩)
                sel = (bell_out == 0)
                n_sel = sel.sum()
                if n_sel > 0:
                    agree_z = (lz1[sel] == lz2[sel]).astype(np.uint8)
                    z_sel = float(2 * int(agree_z.sum()) - n_sel) / n_sel
                    x_sel = float(2 * int((bell_m[sel] == 0).sum()) - n_sel) / n_sel
                else:
                    z_sel = x_sel = 0.0
                w_sel = z_sel + x_sel
                print(f"  {dec_name} ({dt:.1f}s):")
                print(f"    Bell prep: |0⟩={(bell_out==0).sum()} |1⟩={(bell_out==1).sum()}")
                print(f"    Post-sel Φ⁺ ({n_sel}/{n_shots}):")
                print(f"      ⟨Z_L1⊗Z_L2⟩_sel = {z_sel:.3f}")
                print(f"      ⟨X_L1⊗X_L₂⟩_sel = {x_sel:.3f}")
                print(f"      W_sel = {z_sel:.3f} + {x_sel:.3f} = {w_sel:.3f}")
                info = {"Z_sel": float(z_sel), "X_sel": float(x_sel),
                        "witness_sel": float(w_sel), "n_sel": int(n_sel),
                        "basis": "bell", "time_s": round(dt, 2)}
                jobs[job_id][dec_name] = info
            else:
                # Bell run (Z or X basis): compute correlation from data, post-selected on bell=0
                agree_z = (lz1 == lz2).astype(np.uint8)
                z_vals = np.where(agree_z, 1, -1)
                z_all = z_vals.mean()
                sel = (bell_out == 0)
                n_sel = sel.sum()
                z_post = z_vals[sel].mean() if n_sel > 0 else 0
                if opts.measure_x:
                    corr_key = "X_corr"; post_key = "X_post_sel"; basis_label = "X"
                else:
                    corr_key = "Z_corr"; post_key = "Z_post_sel"; basis_label = "Z"
                print(f"  {dec_name} ({dt:.1f}s):")
                print(f"    Bell prep: |0⟩={(bell_out==0).sum()} |1⟩={(bell_out==1).sum()}")
                print(f"    ⟨{basis_label}_L1⊗{basis_label}_L2⟩ = {z_all:.3f}  (all shots)")
                print(f"    Post-selected bell=0 ({n_sel}/{n_shots}): ⟨{basis_label}{basis_label}⟩_0 = {z_post:.3f}")
                info = {corr_key: float(z_all), post_key: float(z_post),
                        "n_sel": int(n_sel), "basis": basis_label, "time_s": round(dt, 2)}
                jobs[job_id][dec_name] = info
        elif ghz:
            bnd = np.zeros((n_shots, s - 1 + r - 1), dtype=np.uint8)
            for j in range(s - 1):
                bnd[:, j] = corrected[:, r - 1, j]
            for i in range(r - 1):
                bnd[:, s - 1 + i] = corrected[:, i, s - 1]
            all_same = (bnd.max(axis=1) == bnd.min(axis=1)).mean()
            logicals = all_logicals_measure(corrected, r, s, basis='Z')
            n_logicals = r + s - 2
            all_ok = np.ones(n_shots, dtype=np.uint8)
            for name, vals in logicals.items():
                all_ok &= (vals == 0)
            joint_fidelity = all_ok.mean()
            print(f"  {dec_name} ({dt:.1f}s):")
            print(f"    GHZ ancilla: |0⟩={(ghz_out==0).sum()} |1⟩={(ghz_out==1).sum()}")
            print(f"    Boundary all-same = {all_same:.3f}")
            print(f"    Boundary |0...0⟩ = {(bnd.sum(axis=1)==0).mean():.3f}")
            if ghz_m is not None:
                x_cond = float(2 * int((ghz_m == ghz_out).sum()) - n_shots) / n_shots
                w = (2 * all_same - 1) + x_cond
                print(f"    ⟨X⊗12⟩_cond (ghz_m == ghz_out) = {x_cond:.3f}")
                print(f"    W_GHZ = {2*all_same-1:.3f} + {x_cond:.3f} = {w:.3f}")
            for name, vals in logicals.items():
                print(f"    {name} error rate = {vals.mean():.4f}")
            print(f"    All-{n_logicals}-Z fidelity = {joint_fidelity:.4f}")
            info = {"ghz_out_0": int((ghz_out==0).sum()), "ghz_out_1": int((ghz_out==1).sum()),
                    "boundary_all_same": float(all_same),
                    "joint_fidelity": float(joint_fidelity),
                    "basis": "GHZ", "time_s": round(dt, 2)}
            if ghz_m is not None:
                info["X_cond"] = float(x_cond)
                info["witness_ghz"] = float(w)
            jobs[job_id][dec_name] = info
        elif opts.all_logicals:
            basis_label = "X" if opts.measure_x else "Z"
            logicals = all_logicals_measure(corrected, r, s, basis=basis_label)
            n_logicals = r + s - 2
            all_ok = np.ones(n_shots, dtype=np.uint8)
            for name, vals in logicals.items():
                all_ok &= (vals == 0)
                print(f"  {dec_name}: {name} error rate = {vals.mean():.4f}")
            joint_fidelity = all_ok.mean()
            print(f"  {dec_name}: All-{n_logicals}-{basis_label} fidelity = {joint_fidelity:.4f}")
            info = {
                "joint_fidelity": float(joint_fidelity),
                "time_s": round(dt, 2),
                "expected_state": f"|{'0'*n_logicals}⟩",
            }
            jobs[job_id][dec_name] = info
        else:
            expected_z1 = int(logical_state[0])
            expected_z2 = int(logical_state[1])
            fidelity = compute_fidelity(lz1, lz2, expected_z1, expected_z2)
            print(f"  {dec_name} ({dt:.1f}s):")
            print(f"    fidelity = {fidelity:.3f}   (expected |{expected_z1}{expected_z2}⟩_L)")
            for z1 in (0, 1):
                for z2 in (0, 1):
                    cnt = ((lz1 == z1) & (lz2 == z2)).sum()
                    exp = "← expected" if (z1 == expected_z1 and z2 == expected_z2) else ""
                    print(f"      |{z1}{z2}⟩: {cnt:>4d} ({100*cnt/n_shots:.1f}%)  {exp}")
            corr = float(2 * int((lz1 == lz2).sum()) - n_shots) / n_shots
            print(f"    ⟨Z_L1⊗Z_L2⟩ = {corr:.3f}")
            jobs[job_id][dec_name] = {
                "fidelity": float(fidelity),
                "correlation": float(corr),
                "time_s": round(dt, 2),
                "expected_z1": expected_z1,
                "expected_z2": expected_z2,
            }

    jobs[job_id]["completed"] = time.time()
    jobs[job_id]["n_shots"] = n_shots
    SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))
    print(f"\nResults saved to {SAVE_FILE}")

    # Summary from best decoder
    if bell_after_qec:
        baq = jobs[job_id].get("bell_after_qec", {})
        w_sel = baq.get("witness_sel", 0)
        n_sel = baq.get("n_sel", 0)
        z_sel = baq.get("Z_sel", 0)
        x_sel = baq.get("X_sel", 0)
        print(f"\n  {'✓ ENTANGLED!' if w_sel > 1 else '~ Below threshold' if w_sel > 0.5 else '✗ Separable'} "
              f"W_post({n_sel}) = {z_sel:.3f} + {x_sel:.3f} = {w_sel:.3f}")
    else:
        best = jobs[job_id].get("ffinal") or jobs[job_id].get("tesseract") or {}
        if ghz:
            f = best.get("fidelity", 0)
            w = best.get("witness_ghz", None)
            if w is not None:
                x = best.get("X_cond", 0)
                ba = best.get("boundary_all_same", 0)
                print(f"\n  {'✓ ENTANGLED!' if w > 1 else '~ Below threshold' if w > 0.5 else '✗ Separable'} "
                      f"W_GHZ = {2*ba-1:.3f} + {x:.3f} = {w:.3f}")
            else:
                print(f"\n  GHZ: All-12-Z fidelity={f:.3f}")
        elif bell:
            if opts.partial_x:
                xn = best.get("X_corr", 0)
                print(f"\n  {'✓' if abs(xn) > 0.6 else '~' if abs(xn) > 0.2 else '✗'} "
                      f"⟨X_L1⊗X_L₂⟩={xn:.3f}: {'Strong' if abs(xn) > 0.6 else 'Weak' if abs(xn) > 0.2 else 'Degraded'}")
            elif opts.bell_measure:
                w = best.get("witness_sel", 0)
                z = best.get("Z_sel", 0)
                x = best.get("X_sel", 0)
                print(f"\n  {'✓ ENTANGLED!' if w > 1 else '~ Below threshold' if w > 0.5 else '✗ Separable'} "
                      f"W = {z:.3f} + {x:.3f} = {w:.3f}")
            else:
                if opts.measure_x:
                    bl = "X"
                    val = best.get("X_post_sel", best.get("X_corr", 0))
                else:
                    bl = "Z"
                    val = best.get("Z_post_sel", best.get("Z_corr", 0))
                n = best.get("n_sel", 0)
                print(f"\n  {bl}-basis bell run: ⟨{bl}{bl}⟩_0 = {val:.3f} ({n} bell=0 shots)")
        elif opts.all_logicals:
            basis_label = "X" if opts.measure_x else "Z"
            f = best.get("fidelity", 0)
            label = f"All-{r+s-2}-{basis_label}"
            print(f"\n  {'✓' if f > 0.8 else '~' if f > 0.6 else '✗'} "
                  f"{label}={f:.3f}: {'Preserved!' if f > 0.8 else 'Partial' if f > 0.6 else 'Degraded'}")
        else:
            f = best.get("fidelity", 0)
            print(f"\n  {'✓' if f > 0.8 else '~' if f > 0.6 else '✗'} "
                  f"Fidelity={f:.3f}: {'Preserved!' if f > 0.8 else 'Partial' if f > 0.6 else 'Degraded'}")



def decode_last_job():
    """Re-decode cached jobs and compute combined Bell witness.

    Picks the most recent bell job with rounds > 0 for detailed re-decoding.
    Then searches all bell jobs for Z + X companion pairs and computes
    the combined witness W = ⟨Z Z⟩ + ⟨X X⟩.
    """
    if not SAVE_FILE.exists():
        print("No jobs.json found.")
        return
    jobs = json.loads(SAVE_FILE.read_text())
    if not jobs:
        print("No jobs in jobs.json.")
        return

    # Prefer a bell job with QEC rounds for detailed re-decoding
    bell_qec = [jid for jid, j in jobs.items()
                if j.get("logical_state") == "bell" and j.get("rounds", 0) > 0]
    if bell_qec:
        job_id = max(bell_qec, key=lambda jid: jobs[jid].get("submitted", 0))
    else:
        job_id = max(jobs, key=lambda jid: jobs[jid].get("submitted", 0))
    info = jobs[job_id]
    npz_path = RAW_DIR / f"{job_id}.npz"
    if not npz_path.exists():
        print(f"No cached data for job {job_id}: {npz_path} not found.")
        return

    data = np.load(npz_path)
    all_syn = data["syndromes"]
    data_raw = data["data_raw"]
    r = int(data["r"])
    s = int(data["s"])
    rounds = int(data["rounds"])
    n_shots = all_syn.shape[0]
    logical_state = str(data.get("logical_state", info.get("logical_state", "00")))
    measure_x = bool(data.get("measure_x", False))
    partial_x = bool(data.get("partial_x", False))
    bell_measure = bool(data.get("bell_measure", False))
    ghz_measure = bool(data.get("ghz_measure", False))
    bell_out = data.get("bell_out", None) if "bell_out" in data else None
    bell_m = data.get("bell_m", None) if "bell_m" in data else None
    ghz_out = data.get("ghz_out", None) if "ghz_out" in data else None
    ghz_m = data.get("ghz_m", None) if "ghz_m" in data else None

    free_final_round = bool(data.get("free_final_round", False))
    periodic = bool(data.get("periodic", True))
    if partial_x:
        basis = "X_partial"
    else:
        basis = "X" if measure_x else "Z"
    bc_label = "open" if not periodic else "periodic"
    ffr = " free-final" if free_final_round else ""
    print(f"Re-decoding job {job_id}")
    print(f"  {r}×{s} grid, {rounds} rounds, {n_shots} shots, {basis}-basis{ffr}, {bc_label} BC")
    print(f"  Logical state: {logical_state}\n")

    if rounds == 0:
        print("  No QEC rounds — decoding skipped (raw data only)\n")
        if periodic:
            lz1 = data_raw[:, 0, :].sum(axis=1) % 2
            lz2 = data_raw[:, :, 0].sum(axis=1) % 2
        else:
            lz1 = data_raw[:, :, 0].sum(axis=1) % 2
            lz2 = data_raw[:, :, 2].sum(axis=1) % 2
        if logical_state == "bell":
            agree = (lz1 == lz2).astype(np.uint8)
            z_all = float(2 * int(agree.sum()) - n_shots) / n_shots
            print(f"  Raw ⟨Z_L1⊗Z_L2⟩ = {z_all:.3f} (all)")
            if bell_out is not None:
                sel = (bell_out == 0)
                z_post = float(2 * int(agree[sel].sum()) - sel.sum()) / sel.sum() if sel.sum() > 0 else 0
                print(f"  Post-selected bell=0 ({sel.sum()}/{n_shots}): ⟨Z Z⟩_0 = {z_post:.3f}")
        else:
            z1, z2 = int(logical_state[0]), int(logical_state[1])
            f = compute_fidelity(lz1, lz2, z1, z2)
            print(f"  Raw fidelity = {f:.3f}   (expected |{z1}{z2}⟩_L)")
        print(f"  ✓ Raw {n_shots} shots")

    # Decoder loop (only if rounds > 0)
    if rounds > 0:
        decoders = ("ffinal", "tesseract")
        for dec_name in decoders:
            t0 = time.time()
            corrs = decode(dec_name, all_syn, r, s)
            dt = time.time() - t0
            corrected = data_raw ^ corrs

            if logical_state == "bell":
                if partial_x:
                    if periodic:
                        x_flat = [0 * s + j for j in range(s)] + [i * s + 0 for i in range(1, r)]
                    else:
                        x_flat = [i * s + 0 for i in range(r)] + [i * s + 2 for i in range(r)]
                    x_partial = corrected.reshape(n_shots, -1)[:, x_flat]
                    x_prod_vals = x_partial.sum(axis=1) % 2
                    x_corr = float(2 * int((x_prod_vals == 0).sum()) - n_shots) / n_shots
                    print(f"  {dec_name} ({dt:.1f}s):")
                    print(f"    ⟨X_L1⊗X_L₂⟩ = {x_corr:.3f} (from {len(x_flat)} H-rotated qubits)")
                    info[dec_name] = {"X_corr": float(x_corr), "basis": "X_partial"}
                elif bell_m is not None and bell_out is not None:
                    # Single-run Bell witness: post-select on bell_out=0 (|Φ⁺⟩)
                    sel = (bell_out == 0)
                    n_sel = sel.sum()
                    if n_sel > 0:
                        lz1, lz2 = logical_measure(corrected[sel], r, s, periodic=periodic)
                        agree_z = (lz1 == lz2).astype(np.uint8)
                        z_sel = float(2 * int(agree_z.sum()) - n_sel) / n_sel
                        x_sel = float(2 * int((bell_m[sel] == 0).sum()) - n_sel) / n_sel
                    else:
                        z_sel = x_sel = 0.0
                    w_sel = z_sel + x_sel
                    print(f"  {dec_name} ({dt:.1f}s):")
                    print(f"    Bell prep: |0⟩={(bell_out==0).sum()} |1⟩={(bell_out==1).sum()}")
                    print(f"    Post-sel Φ⁺ ({n_sel}/{n_shots}):")
                    print(f"      ⟨Z_L1⊗Z_L2⟩_sel = {z_sel:.3f}")
                    print(f"      ⟨X_L1⊗X_L₂⟩_sel = {x_sel:.3f}")
                    print(f"      W_sel = {z_sel:.3f} + {x_sel:.3f} = {w_sel:.3f}")
                    info[dec_name] = {"Z_sel": float(z_sel), "X_sel": float(x_sel),
                                      "witness_sel": float(w_sel), "n_sel": int(n_sel),
                                      "basis": "bell"}
                else:
                    # Bell run (Z or X basis)
                    lz1, lz2 = logical_measure(corrected, r, s, periodic=periodic)
                    agree_z = (lz1 == lz2).astype(np.uint8)
                    z_vals = np.where(agree_z, 1, -1)
                    z_all = z_vals.mean()
                    sel = (bell_out == 0)
                    n_sel = sel.sum()
                    z_post = z_vals[sel].mean() if n_sel > 0 else 0
                    if measure_x:
                        corr_key = "X_corr"; post_key = "X_post_sel"; basis_label = "X"
                    else:
                        corr_key = "Z_corr"; post_key = "Z_post_sel"; basis_label = "Z"
                    print(f"  {dec_name} ({dt:.1f}s):")
                    print(f"    Bell prep: |0⟩={(bell_out==0).sum()} |1⟩={(bell_out==1).sum()}")
                    print(f"    ⟨{basis_label}_L1⊗{basis_label}_L2⟩ = {z_all:.3f}  (all shots)")
                    print(f"    Post-selected bell=0 ({n_sel}/{n_shots}): ⟨{basis_label}{basis_label}⟩_0 = {z_post:.3f}")
                    info[dec_name] = {corr_key: float(z_all), post_key: float(z_post),
                                      "n_sel": int(n_sel), "basis": basis_label}
            elif logical_state == "ghz":
                bnd = np.zeros((n_shots, s - 1 + r - 1), dtype=np.uint8)
                for j in range(s - 1):
                    bnd[:, j] = corrected[:, r - 1, j]
                for i in range(r - 1):
                    bnd[:, s - 1 + i] = corrected[:, i, s - 1]
                all_same = (bnd.max(axis=1) == bnd.min(axis=1)).mean()
                print(f"  {dec_name} ({dt:.1f}s):")
                print(f"    GHZ ancilla: |0⟩={(ghz_out==0).sum()} |1⟩={(ghz_out==1).sum()}")
                print(f"    Boundary all-same = {all_same:.3f}")
                info[dec_name] = {"boundary_all_same": float(all_same), "basis": "GHZ"}
                if ghz_m is not None:
                    x_cond = float(2 * int((ghz_m == ghz_out).sum()) - n_shots) / n_shots
                    w = (2 * all_same - 1) + x_cond
                    print(f"    ⟨X⊗12⟩_cond = {x_cond:.3f}")
                    print(f"    W_GHZ = {2*all_same-1:.3f} + {x_cond:.3f} = {w:.3f}")
                    info[dec_name]["X_cond"] = float(x_cond)
                    info[dec_name]["witness_ghz"] = float(w)
            else:
                lz1, lz2 = logical_measure(corrected, r, s, periodic=periodic)
                expected_z1 = int(logical_state[0])
                expected_z2 = int(logical_state[1])
                fidelity = compute_fidelity(lz1, lz2, expected_z1, expected_z2)
                print(f"  {dec_name} ({dt:.1f}s):")
                print(f"    fidelity = {fidelity:.3f}   (expected |{expected_z1}{expected_z2}⟩_L)")
                for z1 in (0, 1):
                    for z2 in (0, 1):
                        cnt = ((lz1 == z1) & (lz2 == z2)).sum()
                        exp = "← expected" if (z1 == expected_z1 and z2 == expected_z2) else ""
                        print(f"      |{z1}{z2}⟩: {cnt:>4d} ({100*cnt/n_shots:.1f}%)  {exp}")
                corr = float(2 * int((lz1 == lz2).sum()) - n_shots) / n_shots
                print(f"    ⟨Z_L1⊗Z_L2⟩ = {corr:.3f}")
            print()

    SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))
    print(f"  ✓ Re-decoded {n_shots} shots")

    # Try to combine Z + X for Bell witness from two separate jobs
    def get_corr(j, key, fallback=None, dec='ffinal'):
        v = j.get(dec, {}).get(key) or j.get('tesseract', {}).get(key)
        if v is None and fallback is not None:
            v = j.get(dec, {}).get(fallback) or j.get('tesseract', {}).get(fallback)
        return v

    # Show all bell jobs for context
    print("\n  Bell jobs in jobs.json:")
    for jid, j in sorted(jobs.items(), key=lambda x: x[1].get("submitted", 0)):
        if j.get("logical_state") == "bell":
            rnds = j.get("rounds", "?")
            bm = "/bellM" if j.get("bell_measure") else ""
            mx = "X" if j.get("measure_x") else "Z"
            xs = "X" if j.get("x_stabilizer") else "Z"
            ro = "/R" if not j.get("no_reset", True) else ""
            baq = "/afterQEC" if j.get("bell_after_qec") else ""
            w_ = j.get("ffinal", {}) or j.get("bell_after_qec", {})
            zc = w_.get("Z_corr", "?")
            xc = w_.get("X_corr", "?")
            wt = w_.get("witness", "")
            wt_str = f"  W={wt:.3f}" if wt != "" else ""
            print(f"    {jid[:12]}  {rnds}r  read={mx}{bm}  stab={xs}{ro}  "
                  f"⟨ZZ⟩={zc}  ⟨XX⟩={xc}{wt_str}")

    # Two-run combination for jobs without single-run witness
    z_cands = [(jid, j) for jid, j in jobs.items()
               if j.get("logical_state") == "bell"
               and not j.get("bell_measure") and not j.get("measure_x") and not j.get("partial_x")
               and j.get("rounds", -1) >= 0
               and get_corr(j, 'Z_corr') is not None]
    x_cands = [(jid, j) for jid, j in jobs.items()
               if j.get("logical_state") == "bell"
               and (j.get("bell_measure") or j.get("measure_x") or j.get("partial_x"))
               and j.get("rounds", -1) >= 0
               and get_corr(j, 'X_corr') is not None]

    z_job = x_job = None
    if z_cands and x_cands:
        best = -1
        for zid, zj in z_cands:
            for xid, xj in x_cands:
                if zj.get("rounds") == xj.get("rounds"):
                    zn = get_corr(zj, 'Z_corr') or 0
                    xn = get_corr(xj, 'X_corr') or 0
                    score = abs(zn) + abs(xn)
                    if score > best:
                        best = score
                        z_job, x_job = zid, xid

    if z_job and x_job:
        rnds = jobs[z_job].get("rounds")
        zn = get_corr(jobs[z_job], 'Z_corr')
        xn = get_corr(jobs[x_job], 'X_corr')
        if zn is not None and xn is not None:
            witness = zn + xn
            merged = {"Z_corr": zn, "X_corr": xn, "witness": witness, "rounds": rnds}
            for jid in (z_job, x_job):
                jobs[jid]["merged_witness"] = merged
            SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))
            print(f"\n  Combined Bell witness from jobs {z_job[:8]} (Z) + {x_job[:8]} (X):")
            print(f"    ⟨Z Z⟩ = {zn:.3f}  ⟨X X⟩ = {xn:.3f}  W = {zn:.3f} + {xn:.3f} = {witness:.3f}")
            print(f"    {'✓ ENTANGLED' if witness > 1 else '~ Below threshold' if witness > 0.5 else '✗ Separable'}")
            print(f"  ✓ Merged witness saved to both job entries in {SAVE_FILE}")
    elif not z_cands and not x_cands:
        pass  # no two-run combos possible
    else:
        missing = []
        if not z_cands:
            missing.append("Z-basis Bell run (`--state bell`)")
        if not x_cands:
            missing.append("X-basis Bell run (`--state bell --bell-measure`)")
        print(f"\n  Tip: need both a {missing[0]}" + (" and " + missing[1] if len(missing) > 1 else "") +
              " to compute the full Bell witness W = ⟨Z Z⟩ + ⟨X X⟩")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run (1+x²)(1+y²) code experiment using pw_opt circuit builder"
    )
    ap.add_argument('--redecode', action='store_true',
                    help='Re-decode last cached job without resubmitting')
    ap.add_argument('--shots', type=int, default=1000)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--backend', '-b', type=str, default=None, metavar='NAME')
    ap.add_argument('--opt-level', type=int, default=3, choices=[0,1,2,3],
                    help='Transpiler optimization level (default: 3 — safe since no if_test in circuits)')
    ap.add_argument('--reset-every-round', action='store_true',
                    help='Reset ancillas every round (default: no-reset, save resets)')
    ap.add_argument('--no-free-final-round', action='store_true',
                    help='Disable free final round; run all rounds as ancilla rounds (costs 64 extra CX)')
    ap.add_argument('--x-stabilizer', action='store_true',
                    help='Measure X⊗X stabilizers instead of Z⊗Z')
    ap.add_argument('--measure-x', action='store_true',
                    help='Apply H to all data qubits before readout (X-basis)')
    ap.add_argument('--partial-x', action='store_true',
                    help='H on row 0 ∪ col 0, read X_L1 X_L2 from data (0 CX; use instead of --bell-measure for final readout)')
    ap.add_argument('--bell-measure', action='store_true',
                    help='Ancilla-based Bell X measurement mid-circuit (13 CX; only needed for non-destructive readout)')
    ap.add_argument('--bell-after-qec', action='store_true',
                    help='Create Bell state AFTER QEC rounds (avoids mid-circuit collapse). '
                         'QEC runs on |00⟩, then Bell creation at end. Requires --state bell and --bell-measure.')
    ap.add_argument('--full-stabilizer', action='store_true',
                    help='Measure full 4-qubit stabilizer S(i,j) instead of 2-qubit V(i,j). '
                         'Requires 4 CX per ancilla (128/round instead of 64) but preserves Bell '
                         'state through multi-round QEC by not differentiating |00⟩_L from |11⟩_L.')
    ap.add_argument('--ghz-measure', action='store_true',
                    help='Ancilla-based GHZ boundary X⊗12 measurement (13 CX; prefer --partial-x for final readout)')
    ap.add_argument('--all-logicals', action='store_true',
                    help='Report all Z-type logicals')
    ap.add_argument('--grid', type=int, nargs=2, metavar=('R', 'S'),
                    help='Grid dimensions (default: 6 8)')
    ap.add_argument('--state', type=str, default="00",
                    choices=["00", "01", "10", "11", "bell", "ghz"],
                    help="logical state (or 'bell'/'ghz' for entangled states)")
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--offline', action='store_true',
                    help='Run on the local offline dynamic-circuit simulator (no IBM token/queue)')
    ap.add_argument('--fake', type=str, default=None, metavar='NAME',
                    help='Offline: device-calibrate noise from a fake backend (e.g. fez, marrakesh, torino)')
    ap.add_argument('--noise-2q', type=float, default=0.0,
                    help='Offline: depolarizing prob on 2-qubit gates (ignored if --fake given)')
    ap.add_argument('--noise-1q', type=float, default=0.0,
                    help='Offline: depolarizing prob on 1-qubit gates')
    ap.add_argument('--noise-readout', type=float, default=0.0,
                    help='Offline: symmetric readout bit-flip prob')
    ap.add_argument('--noise-reset', type=float, default=0.0,
                    help='Offline: prob a reset leaves |1> (imperfect active reset)')
    ap.add_argument('--seed', type=int, default=None,
                    help='Offline: RNG seed for reproducible sampling')
    ap.add_argument('--dd', action='store_true',
                    help='Dynamic decoupling: X gates on all data qubits between rounds')
    ap.add_argument('--open', action='store_true',
                    help='Open boundary conditions (no vertical wrapping). '
                         'X_L1=col0, X_L2=col2 — both commute with all V(i,j), '
                         'so Bell state survives multi-round QEC.')
    opts = ap.parse_args()
    if opts.redecode:
        decode_last_job()
    else:
        token = None if (opts.offline or opts.dry_run) else get_token()
        run_test(token, opts)

if __name__ == "__main__":
    main()