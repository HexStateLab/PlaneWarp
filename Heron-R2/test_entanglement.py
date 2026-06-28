#!/usr/bin/env python3
"""
test_entanglement.py — Test logical qubit preservation for (1+x²)(1+y²) code.

Logical operators (from plane_warp.c analysis):
  Z_L1 = Z on ALL qubits in row 0       (s qubits, horizontal cycle)
  Z_L2 = Z on ALL qubits in column 0    (r qubits, vertical cycle)
  X_L1 = X on ALL qubits in column 0    (vertical, anticommutes with Z_L1 at (0,0))
  X_L2 = X on ALL qubits in row 0       (horizontal, anticommutes with Z_L2 at (0,0))

|00⟩_L is the +1 eigenstate of both Z_L1 and Z_L2.
All data qubits in |0⟩ is |00⟩_L (verified by C-code is_stabilizer test).

States:
  |00⟩  — all |0⟩ (baseline preservation test)
  |01⟩  — X_L2|00⟩ = X on row 0
  |10⟩  — X_L1|00⟩ = X on column 0
  |11⟩  — X_L1 X_L2|00⟩ = X on row 0 + column 0

Usage:
  export IBM_QUANTUM_TOKEN='your_token'
  python3 test_entanglement.py --shots 1000          # |00⟩ test
  python3 test_entanglement.py --state 01 --shots 1000
  python3 test_entanglement.py --states all --shots 500  # run all 4 states
  python3 test_entanglement.py --dry-run
"""

import json, sys, getpass, os, time
from pathlib import Path
import numpy as np

IBM_TOKEN_ENV = "IBM_QUANTUM_TOKEN"
SAVE_FILE = Path.home() / ".planewarp_entanglement.json"
RAW_DIR = Path.home() / ".planewarp_raw"


def get_token():
    token = os.environ.get(IBM_TOKEN_ENV)
    if token:
        return token
    token = getpass.getpass("IBM Quantum API token: ")
    if token:
        return token
    print("No token provided.", file=sys.stderr)
    sys.exit(1)


def build_circuit(r, s, rounds, logical_state="00", share_pairs=False,
                  bell=False, measure_x=False, opt=False):
    """Build QEC circuit for a given logical state.

    Logical operators:
      Z_L1 = Z on row 0,  X_L1 = X on column 0
      Z_L2 = Z on column 0, X_L2 = X on row 0

    |00⟩_L ≡ all |0⟩.

    Bell state (|00⟩_L + |11⟩_L)/√2:
      Uses one extra ancilla qubit.
      H → controlled-X_L1 (CX to column 0) → controlled-X_L2 (CX to row 0)
        → H → measure ancilla.
      |0⟩ outcome → |Φ⁺⟩_L, |1⟩ outcome → |Φ⁻⟩_L.

    When opt=True: uses optimized share-pair with software-reconstructed V(2).
    32 ancillas instead of 48, 64 CX, 0 SWAPs on heavy-hex.

    When share_pairs=True: 48 ancillas, V(i,j+2) reconstructed in software.

    When both False: standard flag circuit, 2 ancillas per stabilizer, 192 CX.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from pw_qiskit import heavy_hex_flag_layout

    data_map, anc_maps, _, _ = heavy_hex_flag_layout(r, s)
    n_data = r * s
    n_bell = 1 if bell else 0

    if opt:
        hr = r // 2
        hs = s // 2
        n_anc = (hr - 1) * hs * 4  # (r//2 - 1) × (s//2) × 4 sectors
        total = n_data + n_anc + n_bell
        qr = QuantumRegister(total, "q")
        cr_syn = [ClassicalRegister(n_anc, f"syn_{c}") for c in range(rounds)]
        cr_data = ClassicalRegister(n_data, "data")
        cr_bell = ClassicalRegister(1, "bell") if bell else None
        qc = QuantumCircuit(qr, *cr_syn, cr_data, *(cr_bell,) if bell else ())

        if bell:
            b_idx = n_data + n_anc
            qc.h(b_idx)
            for i in range(r):
                qc.cx(b_idx, data_map[i][0])
            for j in range(s):
                qc.cx(b_idx, data_map[0][j])
            qc.h(b_idx)
            qc.measure(b_idx, cr_bell[0])
        else:
            if "1" in logical_state:
                if logical_state[1] == "1":
                    for jj in range(s):
                        qc.x(data_map[0][jj])
                if logical_state[0] == "1":
                    for ii in range(r):
                        qc.x(data_map[ii][0])

        for rnd in range(rounds):
            anc_idx = n_data
            for px in range(2):
                for py in range(2):
                    for p in range(hr - 1):
                        for q in range(hs):
                            i = 2 * p + px
                            j = 2 * q + py
                            qc.reset(anc_idx)
                            qc.cx(data_map[i][j], anc_idx)
                            qc.cx(data_map[(i + 2) % r][j], anc_idx)
                            qc.measure(anc_idx, cr_syn[rnd][anc_idx - n_data])
                            anc_idx += 1
            qc.barrier()
    else:
        n_anc_phys = 2 * r * s
        n_meas = r * s if share_pairs else 2 * r * s
        total = n_data + n_anc_phys + n_bell
        qr = QuantumRegister(total, "q")
        cr_syn = [ClassicalRegister(n_meas, f"syn_{c}") for c in range(rounds)]
        cr_data = ClassicalRegister(n_data, "data")

        if bell:
            cr_bell = ClassicalRegister(1, "bell")
            qc = QuantumCircuit(qr, *cr_syn, cr_data, cr_bell)
            b_idx = n_data + n_anc_phys
            qc.h(b_idx)
            for i in range(r):
                qc.cx(b_idx, data_map[i][0])
            for j in range(s):
                qc.cx(b_idx, data_map[0][j])
            qc.h(b_idx)
            qc.measure(b_idx, cr_bell[0])
        else:
            qc = QuantumCircuit(qr, *cr_syn, cr_data)
            if "1" in logical_state:
                if logical_state[1] == "1":
                    for jj in range(s):
                        qc.x(data_map[0][jj])
                if logical_state[0] == "1":
                    for ii in range(r):
                        qc.x(data_map[ii][0])

        for rnd in range(rounds):
            for ii in range(r):
                for jj in range(s):
                    if share_pairs:
                        a = anc_maps[(ii, jj, 0)]
                        qc.reset(a)
                        qc.cx(data_map[ii][jj], a)
                        qc.cx(data_map[(ii + 2) % r][jj], a)
                        qc.measure(a, cr_syn[rnd][ii * s + jj])
                        continue
                    a0 = anc_maps[(ii, jj, 0)]
                    a1 = anc_maps[(ii, jj, 1)]
                    qc.reset(a0)
                    qc.cx(data_map[ii][jj], a0)
                    qc.cx(data_map[(ii + 2) % r][jj], a0)
                    qc.reset(a1)
                    qc.cx(data_map[ii][(jj + 2) % s], a1)
                    qc.cx(data_map[(ii + 2) % r][(jj + 2) % s], a1)
                    qc.measure(a0, cr_syn[rnd][ii * s * 2 + jj * 2])
                    qc.measure(a1, cr_syn[rnd][ii * s * 2 + jj * 2 + 1])
            qc.barrier()

    if measure_x:
        for ii in range(r):
            for jj in range(s):
                qc.h(data_map[ii][jj])
        qc.barrier()

    for ii in range(r):
        for jj in range(s):
            qc.measure(data_map[ii][jj], cr_data[ii * s + jj])

    lq0_qubits = [data_map[0][jj] for jj in range(s)]
    lq1_qubits = [data_map[ii][0] for ii in range(r)]

    return qc, data_map, lq0_qubits, lq1_qubits


def all_syndromes(pub_result, rounds, r, s):
    """Extract (shots, rounds, r, s) from a SamplerV2 PubResult (non-shared)."""
    n_stab = r * s
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots
    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
        pair = bits[:, :2 * n_stab].reshape(shots, n_stab, 2)
        syn[:, c] = (pair[:, :, 0] ^ pair[:, :, 1]).reshape(shots, r, s)
    return syn


def all_syndromes_shared(pub_result, rounds, r, s):
    """Extract (shots, rounds, r, s) from share_pairs circuit."""
    n_stab = r * s
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots
    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
        m = bits[:, :n_stab].reshape(shots, r, s)
        syn[:, c] = m ^ np.roll(m, shift=-2, axis=2)
    return syn


def all_syndromes_opt(pub_result, rounds, r, s):
    """Extract (shots, rounds, r, s) from optimized share-pair circuit.

    Measurements come in sector-major order (px, py, p, q).
    V(i,j) measured for i=0..r-3, all j (32 of 48 for 6×8).
    V(r-2,:) = Σ V(0:r-2:2,:) and V(r-1,:) = Σ V(1:r-1:2,:) mod 2.
    S(i,j) = V(i,j) ⊕ V(i,j+2 mod s).
    """
    hr, hs = r // 2, s // 2
    n_anc = (hr - 1) * hs * 4  # (r//2 - 1) × (s//2) × 4 sectors
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots

    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
        m = bits[:, :n_anc].astype(np.uint8)

        V = np.zeros((shots, r, s), dtype=np.uint8)
        idx = 0
        for px in range(2):
            for py in range(2):
                for p in range(hr - 1):
                    for q in range(hs):
                        i = 2 * p + px
                        j = 2 * q + py
                        V[:, i, j] = m[:, idx]
                        idx += 1

        V[:, r-2, :] = V[:, 0:r-2:2, :].sum(axis=1) % 2
        V[:, r-1, :] = V[:, 1:r-1:2, :].sum(axis=1) % 2

        syn[:, c] = V ^ np.roll(V, shift=-2, axis=2)
    return syn


def logical_measure(corrected_data, r, s):
    """Measure logical Z on both qubits from corrected data.

    Returns (lz1, lz2) arrays of shape (n_shots,).
      lz1 = parity of corrected_data[0, :]  — Z_L1 on row 0
      lz2 = parity of corrected_data[:, 0]  — Z_L2 on column 0
    """
    lz1 = corrected_data[:, 0, :].sum(axis=1) % 2    # Z_L1: full row 0
    lz2 = corrected_data[:, :, 0].sum(axis=1) % 2    # Z_L2: full column 0
    return lz1, lz2


def compute_fidelity(lz1, lz2, expected_z1, expected_z2):
    """Fraction of shots matching expected logical values."""
    correct = ((lz1 == expected_z1) & (lz2 == expected_z2)).sum()
    return correct / len(lz1)


def decode(decoder_name, all_syn, r, s):
    """Run a decoder over all shots. Returns (n_shots, r, s) corrections."""
    n_shots = all_syn.shape[0]
    if all_syn.shape[1] == 0:
        return np.zeros((n_shots, r, s), dtype=np.uint8)
    corrs = np.empty((n_shots, r, s), dtype=np.uint8)
    if decoder_name == "tesseract":
        from pw_qiskit import PlaneWarp
        pw = PlaneWarp()
        for i in range(n_shots):
            corrs[i] = pw.decode_tesseract(all_syn[i])
    elif decoder_name == "waxis":
        from waxis_decode import WaxisDecoder
        dec = WaxisDecoder(r, s)
        for i in range(n_shots):
            corrs[i] = dec.decode(all_syn[i])
    else:
        raise ValueError(f"unknown decoder: {decoder_name}")
    return corrs


def run_test(token, opts):
    """Full pipeline: build, submit, retrieve, decode, analyze."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    r, s = 6, 8
    rounds = opts.rounds
    shots = opts.shots
    share_pairs = opts.share_pairs
    logical_state = opts.state
    opt = opts.opt

    # Pick backend
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    if opts.backend:
        backend = service.backend(opts.backend)
    else:
        backend = service.backend("ibm_kingston")
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")

    bell = (logical_state == "bell")
    # Build circuit
    qc, data_map, lq0_qubits, lq1_qubits = build_circuit(
        r, s, rounds, logical_state=logical_state, share_pairs=share_pairs, bell=bell,
        measure_x=opts.measure_x, opt=opt,
    )
    basis = "X" if opts.measure_x else "Z"
    label = f"Bell-{basis}" if bell else f"|{logical_state}⟩"
    hr, hs = r // 2, s // 2
    n_anc = 4 * (hr - 1) * hs if opt else (r * s if share_pairs else 2 * r * s)
    print(f"Circuit: {r}×{s} grid, {rounds} rounds, {label}, {shots} shots")
    print(f"  Data qubits: {r * s}, Ancillas: {n_anc}{', +1 Bell ancilla' if bell else ''}")
    if opt:
        print(f"  Optimized: V(2) software-reconstructed, {n_anc} ancillas, 64 CX")
    if opts.measure_x:
        print(f"  Readout: H⊗n applied before measurement (X-basis)")

    # Transpile
    print("Transpiling ...")
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
        routing_method="sabre",
        seed_transpiler=42,
    )
    qc_t = pm.run(qc)
    ops = qc_t.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ('cz', 'ecr', 'cx', 'swap'))
    print(f"  Physical qubits: {qc_t.num_qubits}")
    print(f"  Depth: {qc_t.depth()}")
    print(f"  Two-qubit gates: {two_q}")

    if opts.dry_run:
        print("\nDry run complete.")
        return

    # Submit
    print(f"\nSubmitting ...")
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_t], shots=shots)
    job_id = job.job_id()
    print(f"  Job ID: {job_id}")
    print(f"  Dashboard: https://quantum.ibm.com/jobs/{job_id}")

    # Save job info
    jobs = {}
    if SAVE_FILE.exists():
        try:
            jobs = json.loads(SAVE_FILE.read_text())
        except:
            jobs = {}
    jobs[job_id] = {
        "r": r, "s": s, "rounds": rounds, "shots": shots,
        "backend": backend.name, "logical_state": logical_state,
        "share_pairs": share_pairs, "opt": opt,
        "submitted": time.time(),
    }
    SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))

    # Wait for result
    print("\nWaiting for result (Ctrl+C to detach) ...")
    try:
        result = job.result()
    except KeyboardInterrupt:
        print("\nDetached.")
        sys.exit(0)

    # --- Process result ---
    pub_result = result[0]
    if rounds == 0:
        n_shots = getattr(pub_result.data, "data").num_shots
        all_syn = np.zeros((n_shots, 0, r, s), dtype=np.uint8)
    elif opt:
        all_syn = all_syndromes_opt(pub_result, rounds, r, s)
    elif share_pairs:
        all_syn = all_syndromes_shared(pub_result, rounds, r, s)
    else:
        all_syn = all_syndromes(pub_result, rounds, r, s)
    n_shots = all_syn.shape[0]

    dbits = getattr(pub_result.data, "data").to_bool_array(order='little')
    data_raw = dbits.astype(np.uint8).reshape(n_shots, r, s)

    # Bell ancilla outcome (if bell=True)
    bell_out = None
    if bell:
        bell_out = getattr(pub_result.data, "bell").to_bool_array(order='little').flatten().astype(np.uint8)
        print(f"  Bell ancilla outcomes: |0⟩: {(bell_out == 0).sum()}, |1⟩: {(bell_out == 1).sum()}")

    basis = "X" if opts.measure_x else "Z"

    # Save raw data
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kwargs = dict(syndromes=all_syn, data_raw=data_raw,
                  r=r, s=s, rounds=rounds, share_pairs=share_pairs,
                  logical_state=logical_state, measure_x=opts.measure_x,
                  opt=opt)
    if bell_out is not None:
        kwargs["bell_out"] = bell_out
    np.savez_compressed(RAW_DIR / f"{job_id}.npz", **kwargs)

    print(f"\nDecoding {n_shots} shots ({basis}-basis readout) ...\n")

    for dec_name in ("tesseract", "waxis"):
        t0 = time.time()
        corrs = decode(dec_name, all_syn, r, s)
        dt = time.time() - t0
        corrected = data_raw ^ corrs
        lz1, lz2 = logical_measure(corrected, r, s)

        if bell:
            op = "X" if opts.measure_x else "Z"
            agree = (lz1 == lz2).astype(np.uint8)
            # |Φ⁺⟩ (b=0): Z_L1==Z_L2 AND X_L1==X_L2
            # |Φ⁻⟩ (b=1): Z_L1==Z_L2 AND X_L1!=X_L2
            expected_agree = np.ones(n_shots, dtype=np.uint8)  # Z-basis: always agree
            if opts.measure_x:
                expected_agree = 1 - bell_out                # X-basis: agree only for b=0
            correct_corr = (agree == expected_agree).sum()
            bell_fidelity = correct_corr / n_shots
            corr_val = float(2 * int(agree.sum()) - n_shots) / n_shots

            print(f"  {dec_name} ({dt:.1f}s):")
            print(f"    Bell fidelity ({op}-basis, corr matches bell ancilla) = {bell_fidelity:.3f}")
            print(f"    <{op}_L1 {op}_L2> = {corr_val:.3f}")

            for b in (0, 1):
                for z1 in (0, 1):
                    for z2 in (0, 1):
                        cnt = ((bell_out == b) & (lz1 == z1) & (lz2 == z2)).sum()
                        exp = "← expected" if (b == 0 and z1 == z2) or (b == 1 and z1 != z2) else ""
                        print(f"      b={b} |{z1}{z2}>: {cnt:>4d} ({100*cnt/n_shots:.1f}%)  {exp}")

            key = f"bell_fidelity_{op}"
            jobs[job_id][dec_name] = {
                key: float(bell_fidelity),
                f"correlation_{op}": float(corr_val),
                "time_s": round(dt, 2),
                "logical_state": "bell",
            }
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

    op = "X" if opts.measure_x else "Z"
    f = jobs[job_id].get("waxis", {}).get(f"bell_fidelity_{op}" if bell else "fidelity", 0)
    label = f"Bell-{op}" if bell else "Fidelity"
    print(f"\n  {'✓' if f > 0.8 else '~' if f > 0.6 else '✗'} "
          f"{label}={f:.3f}: {'Preserved!' if f > 0.8 else 'Partial' if f > 0.6 else 'Degraded'}")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Test logical qubit preservation for (1+x²)(1+y²) code"
    )
    ap.add_argument('--shots', type=int, default=1000)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--backend', '-b', type=str, default=None, metavar='NAME')
    ap.add_argument('--share-pairs', action='store_true')
    ap.add_argument('--opt', action='store_true',
                    help='Optimized share-pair: software-reconstructed V(2), 32 anc, 64 CX')
    ap.add_argument('--measure-x', action='store_true',
                    help='Apply H before readout for X-basis measurement')
    ap.add_argument('--state', type=str, default="00",
                    choices=["00", "01", "10", "11", "bell"],
                    help="logical state to prepare (or 'bell' for Bell pair)")
    ap.add_argument('--dry-run', action='store_true')
    opts = ap.parse_args()
    run_test(get_token(), opts)


if __name__ == "__main__":
    main()
