#!/usr/bin/env python3
"""
bell.py — Two-logical Bell QEC test. Uses pw_opt circuit builder + decoder.py.

Prepares |Φ⁺⟩_L = (|00⟩_L + |11⟩_L)/√2, runs QEC, measures logical correlation.

Usage:
  export IBM_QUANTUM_TOKEN='...'
  python bell.py                          # Z-basis, 2 rounds, 1000 shots
  python bell.py --measure-x              # X-basis (H before readout)
  python bell.py --backend ibm_fez        # target backend
  python bell.py --redecode               # re-decode last saved job
  python bell.py --dry-run
"""

import json, sys, os, time, argparse
from pathlib import Path
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from pw_opt import build_circuit, all_syndromes_opt

SAVE_FILE = Path("bell_jobs.json")
RAW_DIR   = Path("bell_raw")


# ── decoding ──

def decode_batch(dec_name, all_syn, r, s):
    n_shots = all_syn.shape[0]
    if dec_name == "raw" or all_syn.shape[1] == 0:
        return np.zeros((n_shots, r, s), dtype=np.uint8)
    corrs = np.zeros((n_shots, r, s), dtype=np.uint8)
    if dec_name in ("ffinal", "tesseract", "multi"):
        from decoder import (tesseract_decode_ffinal, tesseract_decode,
                             tesseract_decode_multi)
        fn = {"ffinal": tesseract_decode_ffinal, "tesseract": tesseract_decode,
              "multi": tesseract_decode_multi}[dec_name]
        for i in range(n_shots):
            corrs[i] = fn(all_syn[i], r, s)
    return corrs


# ── analysis ──

def measure_logicals(data, r, s):
    lz1 = data[:, 0, :].sum(axis=1) % 2
    lz2 = data[:, :, 0].sum(axis=1) % 2
    return lz1, lz2


def analyze(job_id, all_syn, data_raw, bell_out, bell_m, r, s, rounds,
            measure_x=False, belt_measure=False, no_reset=True,
            free_final_round=True):
    """Decode and print Bell analysis. Reusable by both live and --redecode paths."""
    n_shots = data_raw.shape[0]
    basis = "X" if measure_x else "Z"

    print(f"\nDecoding {n_shots} shots ({basis}-basis, {rounds} rounds) ...\n")

    for dec_name in ("raw", "ffinal", "tesseract", "multi"):
        t0 = time.time()
        corrs = decode_batch(dec_name, all_syn, r, s)
        dt = time.time() - t0
        corrected = data_raw ^ corrs
        lz1, lz2 = measure_logicals(corrected, r, s)

        agree = (lz1 == lz2)
        n_agree = int(agree.sum())
        all_corr = (2.0 * n_agree - n_shots) / n_shots

        sel = (bell_out == 0)
        n_sel = sel.sum()
        n_agree_sel = int(agree[sel].sum())
        zz_sel = (2.0 * n_agree_sel - n_sel) / n_sel if n_sel > 0 else 0

        print(f"  {dec_name} ({dt:.1f}s):  ⟨Z_L1⊗Z_L₂⟩={all_corr:+.3f}  "
              f"⟨ZZ⟩_sel={zz_sel:+.3f} (n_sel={n_sel})")

        if bell_m is not None:
            n_xx = int((bell_m[sel]==0).sum())
            xx_sel = (2.0 * n_xx - n_sel) / n_sel if n_sel > 0 else 0
            w = zz_sel + xx_sel
            print(f"    ⟨XX⟩_sel={xx_sel:+.3f}  W={w:+.3f}  "
                  f"{'ENTANGLED' if w>1 else 'marginal' if w>0.5 else 'separable'}")

        print()

    lz1_raw, lz2_raw = measure_logicals(data_raw, r, s)
    agree_raw = (lz1_raw == lz2_raw)
    n_sel_v = int((bell_out==0).sum())
    n_agree_v = int(agree_raw[bell_out==0].sum())
    zz_sel = (2.0 * n_agree_v - n_sel_v) / n_sel_v if n_sel_v > 0 else 0
    verdict = ("ENTANGLED" if abs(zz_sel) > 0.3
               else "marginal" if abs(zz_sel) > 0.1
               else "no correlation")
    print(f"  Verdict: ⟨ZZ⟩_sel={zz_sel:+.3f} → {verdict}")

    # Update job record
    if SAVE_FILE.exists():
        jobs = json.loads(SAVE_FILE.read_text())
    else:
        jobs = {}
    if job_id and job_id in jobs:
        jobs[job_id]["redecoded"] = time.time()
        SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))


# ── submit & wait ──

def submit(token, opts):
    r, s = opts.grid
    rounds = opts.rounds
    shots = opts.shots

    # Build
    no_reset = opts.no_reset
    free_final = not opts.no_reset
    qc, data_map, lq0, lq1, n_anc = build_circuit(
        r, s, rounds, logical_state="bell", bell=True,
        bell_ancilla=False,  # direct H on data[0][0], no ancilla overhead
        no_reset=no_reset, free_final_round=free_final,
        periodic=True, compact=True,
        full_stabilizer=opts.full_stabilizer,
        dd=opts.dd,
        initial_reset=opts.initial_reset,
        share_extra_ancilla=opts.share_extra,
        bell_measure=opts.bell_measure, measure_x=opts.measure_x)

    ops = qc.count_ops()
    basis = "X" if opts.measure_x else "Z"
    bm = "-M" if opts.bell_measure else ""
    print(f"Bell QEC: {r}x{s}, {rounds}r, {shots} shots, {basis}-basis{bm}")
    print(f"  Qubits={qc.num_qubits}  CX={ops.get('cx',0)}  "
          f"depth={qc.depth()}  meas={ops.get('measure',0)}")

    if opts.dry_run:
        print("Dry run complete.")
        return

    # Backend
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backend = service.backend(opts.backend) if opts.backend else service.backend("ibm_kingston")
    print(f"Backend: {backend.name}")

    # Transpile
    print("Transpiling ...")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                       seed_transpiler=42)
    qc_t = pm.run(qc)
    ops_t = qc_t.count_ops()
    two_q = sum(v for k, v in ops_t.items() if k in ("cz", "ecr", "cx"))
    print(f"  Phys qubits={qc_t.num_qubits}  depth={qc_t.depth()}  "
          f"2q gates={two_q}")

    # Submit
    print("Submitting ...")
    sampler = Sampler(mode=backend)
    job = sampler.run([qc_t], shots=shots)
    job_id = job.job_id()
    print(f"  Job: {job_id}")
    print(f"  https://quantum.ibm.com/jobs/{job_id}")

    # Save job info immediately (before waiting)
    jobs = {}
    if SAVE_FILE.exists():
        try: jobs = json.loads(SAVE_FILE.read_text())
        except: pass
    jobs[job_id] = {
        "r": r, "s": s, "rounds": rounds, "shots": shots,
        "backend": backend.name, "measure_x": opts.measure_x,
        "bell_measure": opts.bell_measure,
        "submitted": time.time(),
    }
    SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))

    # Wait
    print("Waiting (Ctrl+C to detach, re-decode later with --redecode) ...")
    try:
        result = job.result()
    except KeyboardInterrupt:
        print(f"\nDetached. Job {job_id} saved. Re-run with: python bell.py --redecode")
        sys.exit(0)

    pub_result = result[0]

    # Extract
    dbits = getattr(pub_result.data, "data").to_bool_array(order='little')
    n_shots = dbits.shape[0]
    data_raw = dbits.astype(np.uint8).reshape(n_shots, r, s)

    all_syn = all_syndromes_opt(pub_result, rounds, r, s, n_anc,
                                no_reset=no_reset, free_final_round=free_final,
                                data_raw=data_raw,
                                full_stabilizer=opts.full_stabilizer) if rounds > 0 else \
              np.zeros((n_shots, 0, r, s), dtype=np.uint8)

    # Bell ancilla
    try:
        bell_out = getattr(pub_result.data, "bell").to_bool_array(order='little').flatten().astype(np.uint8)
    except AttributeError:
        bell_out = np.zeros(n_shots, dtype=np.uint8)
    print(f"  Bell prep: |Φ⁺⟩={(bell_out==0).sum()}  |Φ⁻⟩={(bell_out==1).sum()}")

    bell_m = None
    if opts.bell_measure:
        try:
            bell_m = getattr(pub_result.data, "bell_m").to_bool_array(order='little').flatten().astype(np.uint8)
            print(f"  Bell X: +1={(bell_m==0).sum()}  -1={(bell_m==1).sum()}")
        except: pass

    # Save raw data for --redecode
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW_DIR / f"{job_id}.npz",
                        all_syn=all_syn, data_raw=data_raw,
                        bell_out=bell_out, bell_m=bell_m if bell_m is not None else np.array([]),
                        r=r, s=s, rounds=rounds,
                        measure_x=opts.measure_x, bell_measure=opts.bell_measure,
                        no_reset=no_reset, free_final_round=free_final,
                        full_stabilizer=opts.full_stabilizer)

    # Mark completed
    if job_id in jobs:
        jobs[job_id]["completed"] = time.time()
        SAVE_FILE.write_text(json.dumps(jobs, indent=2, default=str))

    # Analyze
    analyze(job_id, all_syn, data_raw, bell_out, bell_m, r, s, rounds,
            measure_x=opts.measure_x, belt_measure=opts.bell_measure)


# ── re-decode ──

def redecode():
    if not SAVE_FILE.exists():
        print("No bell_jobs.json found.")
        return
    jobs = json.loads(SAVE_FILE.read_text())
    if not jobs:
        print("No jobs in bell_jobs.json.")
        return

    # Pick the latest completed job
    completed = [(jid, j) for jid, j in jobs.items()
                 if isinstance(j, dict) and j.get("completed")]
    if not completed:
        print("No completed jobs found. Try again after the job finishes.")
        return
    job_id = max(completed, key=lambda x: x[1].get("completed", 0))[0]
    info = jobs.get(job_id, {})
    if not isinstance(info, dict):
        print(f"Corrupt entry for {job_id}")
        return

    npz_path = RAW_DIR / f"{job_id}.npz"
    if not npz_path.exists():
        print(f"No cached data for {job_id}: {npz_path} not found.")
        return

    data = np.load(npz_path, allow_pickle=True)
    all_syn = data["all_syn"]
    data_raw = data["data_raw"]
    bell_out = data["bell_out"]
    bell_m_arr = data["bell_m"]
    bell_m = bell_m_arr if bell_m_arr.size > 0 else None
    r = int(data["r"])
    s = int(data["s"])
    rounds = int(data["rounds"])
    measure_x = bool(data.get("measure_x", False))
    bell_measure = bool(data.get("bell_measure", False))

    basis = "X" if measure_x else "Z"
    print(f"Re-decoding job {job_id}")
    print(f"  {r}x{s}, {rounds} rounds, {all_syn.shape[0]} shots, {basis}-basis")
    print(f"  Backend: {info.get('backend', '?')}")
    print(f"  Submitted: {info.get('submitted', '?')}")
    print(f"  Bell prep: |Φ⁺⟩={(bell_out==0).sum()}  |Φ⁻⟩={(bell_out==1).sum()}")

    analyze(job_id, all_syn, data_raw, bell_out, bell_m, r, s, rounds,
            measure_x=measure_x, belt_measure=bell_measure)


# ── main ──

def main():
    ap = argparse.ArgumentParser(description="Bell QEC test")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--backend", "-b", type=str, default=None)
    ap.add_argument("--grid", type=int, nargs=2, metavar=("R", "S"), default=(6, 8))
    ap.add_argument("--measure-x", action="store_true",
                    help="H-before-readout for X-basis (0 extra CX)")
    ap.add_argument("--bell-measure", action="store_true",
                    help="Mid-circuit X Bell readout")
    ap.add_argument("--redecode", action="store_true",
                    help="Re-decode the last completed job (no re-submission)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--full-stabilizer", action="store_true",
                    help="Full 4-qubit S(i,j) stabilizer (preserves Bell state through multi-round QEC)")
    ap.add_argument("--dd", action="store_true",
                    help="Dynamic decoupling: X gates on all data qubits between rounds")
    ap.add_argument("--no-reset", action="store_true",
                    help="Skip ancilla resets (XOR differencing, fewer gates)")
    ap.add_argument("--initial-reset", action="store_true",
                    help="Reset all data qubits to |0⟩ before state prep")
    ap.add_argument("--share-extra", action="store_true",
                    help="Share ancilla qubit for Bell prep/measure (saves 1 qubit)")
    opts = ap.parse_args()

    if opts.redecode:
        redecode()
        return

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token and not opts.dry_run:
        import getpass
        token = getpass.getpass("IBM Quantum token: ")

    submit(token, opts)


if __name__ == "__main__":
    main()
