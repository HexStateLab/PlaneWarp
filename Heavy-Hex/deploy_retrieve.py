#!/usr/bin/env python3
"""
Retrieve a completed Heron r2 job and decode all shots via consistency
projection + linear basis decoder.

Usage:
  python3 deploy_retrieve.py <job_id>

Job parameters read from ~/.planewarp_jobs.json (created by deploy_heron.py).
"""

import json, sys, os, time
from pathlib import Path

import numpy as np

SAVE_FILE = Path.home() / ".planewarp_jobs.json"
IBM_TOKEN_ENV = "IBM_QUANTUM_TOKEN"
CLEAN_DIR = Path.home() / ".planewarp_clean"


def get_token():
    token = os.environ.get(IBM_TOKEN_ENV)
    if token:
        return token
    from getpass import getpass
    token = getpass("IBM Quantum API token: ")
    if token:
        return token
    print("No token.", file=sys.stderr)
    sys.exit(1)


def all_syndromes(pub_result, rounds, r, s):
    n_stab = r * s
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots
    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    bits0 = getattr(pub_result.data, "syn_0").to_bool_array()
    shared = (bits0.shape[1] == n_stab)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array()
        for q in range(n_stab):
            i, j = q // s, q % s
            if shared:
                syn[:, c, i, j] = bits[:, q].astype(np.uint8)
            else:
                syn[:, c, i, j] = bits[:, 2 * q] ^ bits[:, 2 * q + 1]
    return syn


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Retrieve and decode Heron r2 job")
    ap.add_argument('job_id', nargs='?', help='job ID to retrieve')
    opts = ap.parse_args()

    job_id = opts.job_id
    if not job_id:
        print("Usage: python3 deploy_retrieve.py <job_id>", file=sys.stderr)
        sys.exit(1)

    token = get_token()

    from qiskit_ibm_runtime import QiskitRuntimeService
    from pw_qiskit import PlaneWarp

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    job = service.job(job_id)
    status = str(job.status())
    print(f"Job: {job_id}")
    print(f"Status: {status}")

    if status in ("ERROR", "CANCELLED"):
        print(f"Job ended with {status}.  Cannot retrieve results.", file=sys.stderr)
        sys.exit(1)
    if status != "DONE":
        print("Job not yet completed.  Try again later.", file=sys.stderr)
        sys.exit(1)

    # Load saved params
    params = {}
    if SAVE_FILE.exists():
        saved = json.loads(SAVE_FILE.read_text())
        params = saved.get(job_id, {})

    r = params.get("r", 6)
    s = params.get("s", 8)
    rounds = params.get("rounds", 4)

    result = job.result()
    pub_result = result[0]

    all_syn = all_syndromes(pub_result, rounds, r, s)
    n_shots = all_syn.shape[0]

    # Data readout
    data_raw = None
    if hasattr(pub_result.data, "data"):
        dbits = getattr(pub_result.data, "data").to_bool_array(order='little')
        data_raw = dbits.astype(np.uint8).reshape(n_shots, r, s)

    pw = PlaneWarp()

    # Load 24-dim basis
    bp24 = CLEAN_DIR / 'basis_24.npz'
    if not bp24.exists():
        print(f"  ERROR: 24-dim basis not found at {bp24}. Run build_basis.py first.", file=sys.stderr)
        sys.exit(1)
    bd24 = np.load(bp24)
    basis_syn = [np.asarray(bd24['syn'][i], dtype=np.uint8).reshape(r, s) for i in range(len(bd24['syn']))]
    basis_corr = [np.asarray(bd24['corr'][i], dtype=np.uint8).reshape(r, s) for i in range(len(bd24['corr']))]
    print(f"  Basis loaded: {len(basis_syn)} dims (full column space)")

    # Consistency projection + basis decode
    proj_rounds = min(4, rounds)
    decode_ok = 0
    decode_corr = np.zeros((n_shots, r, s), dtype=np.uint8)
    decoded_mask = np.zeros(n_shots, dtype=bool)
    for idx in range(n_shots):
        rounds_4d = np.stack([all_syn[idx, c] for c in range(rounds - proj_rounds, rounds)])
        C, ok = pw.project_decode(basis_syn, basis_corr, rounds_4d)
        if ok:
            decode_ok += 1
            decode_corr[idx] = C
            decoded_mask[idx] = True

    # Single-observable LER from data readout
    proj_perf = 0
    proj_det = 0
    proj_log = 0
    if data_raw is not None:
        for idx in range(n_shots):
            if not decoded_mask[idx]:
                continue
            E = data_raw[idx]
            res = (E ^ decode_corr[idx]).astype(np.uint8)
            sr = pw.syndrome_of(res)
            if sr.sum() == 0:
                if pw.is_stabilizer(res):
                    proj_perf += 1
                else:
                    proj_log += 1
            else:
                proj_det += 1

    # Per-round syndrome diagnostics
    for c in range(rounds):
        wt = all_syn[:, c, :, :].sum() / n_shots
        print(f"  Round {c} syn wt: {wt:.1f} / {r*s} ({100*wt/(r*s):.1f}%)")

    print(f"\n=== Heron r2 results ===")
    print(f"  Grid:     {r}×{s}")
    print(f"  Rounds:   {rounds} (last {proj_rounds} for projection)")
    print(f"  Shots:    {n_shots}")
    print(f"  Method:   consistency projection → Col(H) → basis decode")
    print()
    print(f"  Projection decode:  {decode_ok}/{n_shots} ({100*decode_ok/n_shots:.1f}%)")
    print()
    if data_raw is not None:
        print(f"  Projection outcomes ({decode_ok} decoded):")
        print(f"    Perfect (E⊕C=0):         {proj_perf} ({100*proj_perf/max(1,decode_ok):.1f}%)")
        print(f"    Detectable (syn≠0):       {proj_det} ({100*proj_det/max(1,decode_ok):.1f}%)")
        print(f"    Logical (syn=0, not stab): {proj_log} ({100*proj_log/max(1,decode_ok):.1f}%)")
        print(f"  True LER (undetectable):    {100*proj_log/n_shots:.2f}%")
    else:
        print(f"  No data readout (rerun with --readout for LER).")

    # Update save file
    if SAVE_FILE.exists():
        saved = json.loads(SAVE_FILE.read_text())
        if job_id in saved:
            saved[job_id].update({
                "retrieved": True,
                "shots_completed": n_shots,
                "proj_decode_pct": round(100 * decode_ok / n_shots, 1),
                "proj_perf": int(proj_perf),
                "proj_det": int(proj_det),
                "proj_log": int(proj_log),
                "true_ler_pct": round(100 * proj_log / n_shots, 2) if data_raw is not None else None,
            })
            SAVE_FILE.write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    main()
