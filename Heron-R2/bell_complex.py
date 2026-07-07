#!/usr/bin/env python3
"""Bell witness: Bell ancilla prep, identical QEC schedule in both arms,
data-readout ZZ (Z basis) and XX (partial-X H on the X_L1X_L2 support).
No mid-circuit Bell-measure ancilla.

Correctness rules encoded here:
  - Corrections decoded from S_Z apply ONLY to Z-basis readout (the ZZ
    arm). X errors do not flip X-basis bits, so the XX arm is never
    corrected with an X-error pattern — the Z-syndrome record is used
    there for POSTSELECTION only. (Mirrored when --stabilizer-basis X.)
  - The prep frame bit m applies ONLY to XX: both Bell signs have
    ZZ = +1, so frame-correcting ZZ mixes in a coin flip.
  - W = ZZ_raw + XX_frame.  Separable bound W <= 1; certified when
    W - 2*sigma > 1.
  - Weight-2 V checks of one basis anticommute with the other basis's
    logical string (single-qubit overlaps): at rounds >= 1 WITHOUT
    --full-stabilizer the checks themselves destroy the Bell coherence
    of the complementary correlator, on perfect hardware.
  - Data-derived syndromes: the ZZ arm's Z readout yields a full final
    S_Z (appended as an extra decode round); the XX arm's mixed readout
    only yields S_Z on plaquettes not touching the H'd support — used
    for its postselection mask.
"""

import sys, os, time, argparse
import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from pw_opt import build_circuit, all_syndromes_opt, accumulated_syndromes, synthesize_paired_layout, steane_syndromes


def _transpile_pytket(qc, backend):
    """Transpile via pytket with full placement+routing+rebase.

    Uses the default Qiskit O3 transpiler (the pytket implicit-swap model is
    incompatible with IBM Runtime's ISA validator without post-processing).
    Kept as a future extension point — passes through to Qiskit for now.
    """
    from qiskit.transpiler.preset_passmanagers import (
        generate_preset_pass_manager)
    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=3,
                                      seed_transpiler=42)
    return pm.run(qc)


def _decoder():
    try:
        from decoder import tesseract_decode_ffinal
        return tesseract_decode_ffinal, None
    except (ImportError, OSError) as e:      # missing .so / binary included
        return None, e


def data_syndrome(data):
    V = data ^ np.roll(data, -2, axis=1)
    return V ^ np.roll(V, -2, axis=2)


def plaquette_validity(r, s):
    """Plaquettes whose 4 qubits avoid the partial-X support (row0 + col0):
    only these give a valid data-derived S_Z from the XX arm's mixed-basis
    readout."""
    support = {(0, j) for j in range(s)} | {(i, 0) for i in range(1, r)}
    valid = np.ones((r, s), dtype=bool)
    for i in range(r):
        for j in range(s):
            for di, dj in ((0, 0), (2, 0), (0, 2), (2, 2)):
                if ((i + di) % r, (j + dj) % s) in support:
                    valid[i, j] = False
                    break
    return valid


def decode_stream(stream, r, s, fn):
    """Unique-decode: one subprocess call per distinct syndrome stream."""
    nsh = stream.shape[0]
    if stream.shape[1] == 0 or fn is None:
        return np.zeros((nsh, r, s), dtype=np.uint8), False
    flat = stream.reshape(nsh, -1)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    cu = np.zeros((len(uniq), r, s), dtype=np.uint8)
    for k in range(len(uniq)):
        cu[k] = fn(uniq[k].reshape(-1, r, s), r, s)
    return cu[inv], True


def witness(zz_bits, xx_bits):
    zz = 1.0 - 2.0 * zz_bits.mean()
    xx = 1.0 - 2.0 * xx_bits.mean()
    sig = np.sqrt(max(1e-12, 1 - zz * zz) / len(zz_bits) +
                  max(1e-12, 1 - xx * xx) / len(xx_bits))
    return zz + xx, zz, xx, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--backend", "-b", type=str, default="ibm_fez")
    ap.add_argument("--grid", type=int, nargs=2, default=(4, 8))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--full-stabilizer", action="store_true")
    ap.add_argument("--steane", action="store_true",
                    help="Steane-style EC: encoded-ancilla block per round, "
                         "transversal CX, destructive block readout; syndrome "
                         "reconstructed in software (free round every round). "
                         "Implies full weight-4 S extraction.")
    ap.add_argument("--stabilizer-basis", type=str, default="Z")
    ap.add_argument("--output-qasm", type=str, default=None,
                    help="Write OpenQASM 3.0 of ZZ arm to this file")
    ap.add_argument("--qiskit", action="store_true",
                    help="Use Qiskit O3 transpiler instead of pytket")
    ap.add_argument("--job-id", type=str, default=None,
                    help="Skip execution: fetch this completed IBM job and "
                         "run the decode/analysis on it. Use 'last' for the "
                         "most recent completed job. ALL other flags "
                         "(--grid/--rounds/--stabilizer-basis/--steane/"
                         "--full-stabilizer) must match the submitted run — "
                         "they determine register layout and reconstruction.")
    ap.add_argument("--transpile-seeds", type=int, default=16,
                    help="Qiskit path only: transpile with this many Sabre "
                         "seeds and submit the lowest-2q result per arm "
                         "(measured ~8%% 2q reduction on heavy-hex; set 1 "
                         "to disable)")
    ap.add_argument("--dynamic", action="store_true",
                    help="Inject dynamic circuit correction stage from "
                         "dc_decode.py.  Computes tesseract correction "
                         "mid-circuit from measured syndrome and applies "
                         "conditional X to data — no post-hoc decode. "
                         "Best with --steane (where syndrome registers "
                         "carry full r*s bits).")
    ap.add_argument("--dynamic-mode", choices=("expr", "coherent", "inplace"),
                    default="expr",
                    help="expr: classical-expression if_tests (measured "
                         "stall cost ~0.27 of raw <ZZ> on Heron). "
                         "coherent: deferred-measurement CX network — "
                         "zero classical branches, zero stall; prefix-XOR "
                         "particular solution only; +2n qubits; correction "
                         "recorded in dc_corr_m for audit. "
                         "inplace: coherent stage on the BLOCK qubits — "
                         "zero fresh qubits, near-native routing, kernel "
                         "fixup included; block readout stays raw (prefix "
                         "uncomputed); audit from steane_t + dc_row_m.")
    ap.add_argument("--dynamic-confirm", type=int, choices=(1, 2, 3),
                    default=1,
                    help="Temporal defect confirmation for --dynamic "
                         "(expr mode only).  1: raw last-round syndrome "
                         "(current behaviour).  2: AND of last two Z "
                         "rounds — correct only PERSISTENT defects, "
                         "vetoing measurement/ancilla-prep (syndrome) "
                         "errors that gate errors survive.  3: majority "
                         "vote over last three Z rounds (two-sided; "
                         "tolerates one bad round anywhere).  Requires "
                         "that many Z-type rounds in the schedule; "
                         "expression size grows ~2.3x / ~7x.")
    opts = ap.parse_args()

    r, s = opts.grid; n = opts.rounds; shots = opts.shots
    basis = opts.stabilizer_basis.upper()
    dry = opts.dry_run

    if opts.steane and not opts.full_stabilizer:
        # Steane extraction measures the full weight-4 S transversally —
        # the weight-2 V dephasing hazard below cannot arise.
        opts.full_stabilizer = True

    if n > 0 and not opts.full_stabilizer:
        killed = "XX" if basis == "Z" else "ZZ"
        print(f"!! rounds>=1 with weight-2 V_{basis} checks: they "
              f"anticommute with the {killed} logical string — the checks "
              f"themselves dephase {killed} even on perfect hardware. "
              f"Use --full-stabilizer unless this is the negative control.")
    if n == 1 and basis == "X":
        print("!! a single X-basis round is a random gauge fix — no "
              "Z-error information; decoding engages from rounds >= 2.")

    if dry:
        if opts.job_id:
            print("--job-id needs the real service; drop --dry-run."); return
        from qiskit_ibm_runtime.fake_provider.backends.fez.fake_fez import FakeFez
        be = FakeFez()
    else:
        token = os.environ.get("IBM_QUANTUM_TOKEN") or \
            __import__("getpass").getpass("IBM: ")
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
        be = svc.backend(opts.backend)

    kw = dict(logical_state="00", bell=True, bell_ancilla=True,
              stabilizer_basis=basis,
              full_stabilizer=opts.full_stabilizer,
              no_reset=False, periodic=True, compact=True,
              steane_ec=opts.steane)

    plan = None
    if opts.steane:
        pass  # steane replaces ancilla-based extraction: no paired layout
    elif opts.full_stabilizer and r in (4, 6) and s % 4 == 0 and n > 0:
        if not dry:
            print("synthesizing heavy-hex paired layout …")
        plan = synthesize_paired_layout(be, r, s, verbose=not dry)
        if plan is not None:
            kw["full_stabilizer"] = "paired"
            kw["no_reset"] = True
            kw["rung_plan"] = plan
        else:
            print("!! paired layout synthesis failed — falling back to "
                  "legacy full_stabilizer (transpiler-routed degree-4 stars)")
            kw["full_stabilizer"] = True
            kw["no_reset"] = True

    qzz, *qzz_ret = build_circuit(r, s, n, **kw)
    n_anc = qzz_ret[3] if len(qzz_ret) > 3 else (r - 1) * s
    qxx, *_ = build_circuit(r, s, n, partial_x=True, **kw)

    if opts.output_qasm:
        from qiskit import qasm3
        with open(opts.output_qasm, "w") as fp:
            fp.write(qasm3.dumps(qzz))
        print(f"ZZ arm QASM → {opts.output_qasm}")
        return

    # --- dynamic correction injection (Stage 5: dc_decode.py) ---
    if opts.dynamic:
        from dc_decode import (inject_dynamic_stage, steane_bit_fn,
                               confirmed_bit_fn)
        if not opts.steane:
            print("!! --dynamic requires --steane (full r*s syndrome "
                  "registers); non-Steane syndrome is share-pair "
                  "with fewer total bits.  Ignoring.")
        elif n == 0:
            print("!! --dynamic with rounds=0: no syndrome round to decode. "
                  "Ignoring.")
        else:
            # The steane_{t} registers hold RAW block readout, not syndrome
            # — steane_bit_fn applies the XOR-of-4 roll math at the
            # expression level.  The decoder consumes S_Z (X-error
            # corrections), so use the LAST Z-type round, and inject the
            # ZZ arm only: X corrections apply to Z-basis readout (rule 1);
            # injecting post-H in the XX arm would flip X-basis outcomes
            # on X-error syndromes — the decoded_arm bug, physically.
            z_rounds = [t for t in range(n) if basis[t % len(basis)] == "Z"]
            if not z_rounds:
                print("!! --dynamic: no Z-type round in the schedule — "
                      "nothing to feed the X-error decoder.  Ignoring.")
            else:
                lz = z_rounds[-1]
                n_data = r * s
                if opts.dynamic_mode == "coherent":
                    from dc_decode import add_coherent_correction
                    if opts.dynamic_confirm > 1:
                        print("!! --dynamic-confirm is expr-mode only: the "
                              "coherent stage computes S from the live block "
                              "qubits of ONE round (no prior-round record "
                              "exists coherently).  Ignoring.")
                    blk = list(range(qzz.num_qubits - n_data,
                                     qzz.num_qubits))
                    qzz = add_coherent_correction(
                        qzz, r, s, list(range(n_data)), blk,
                        f"steane_{lz}")
                    print(f"  coherent correction injected: ZZ arm, CX "
                          f"network before steane_{lz} measurement, zero "
                          f"classical branches; XX arm postselect-only")
                elif opts.dynamic_mode == "inplace":
                    from dc_decode import add_inplace_coherent_correction
                    if opts.dynamic_confirm > 1:
                        print("!! --dynamic-confirm is expr-mode only: the "
                              "in-place stage computes from the live block "
                              "qubits of ONE round.  Ignoring.")
                    blk = list(range(qzz.num_qubits - n_data,
                                     qzz.num_qubits))
                    qzz = add_inplace_coherent_correction(
                        qzz, r, s, list(range(n_data)), blk,
                        f"steane_{lz}")
                    print(f"  in-place coherent correction injected: ZZ "
                          f"arm, block qubits reused as compute register "
                          f"(0 fresh qubits), kernel fixup in-branch, "
                          f"prefix uncomputed before steane_{lz} readout; "
                          f"XX arm postselect-only")
                else:
                    k = opts.dynamic_confirm
                    if k > 1 and len(z_rounds) < k:
                        print(f"!! --dynamic-confirm {k}: only "
                              f"{len(z_rounds)} Z round(s) in schedule — "
                              f"falling back to raw last-round syndrome")
                        k = 1
                    window = z_rounds[-k:]   # oldest first
                    cregs = [[cr for cr in qzz.cregs
                              if cr.name == f"steane_{t}"][0]
                             for t in window]
                    if k == 1:
                        bit_fn = steane_bit_fn(cregs[0], r, s)
                        src = f"steane_{lz} (last Z round)"
                    else:
                        # NOTE: the stage is injected before the FINAL data
                        # measurement (inject_dynamic_stage), which is after
                        # all steane_{t} blocks have been destructively read
                        # out — every register in the window is already
                        # populated at evaluation time.
                        bit_fn = confirmed_bit_fn(cregs, r, s)
                        kind = "AND-pair" if k == 2 else "majority-of-3"
                        src = (f"{kind} of steane_{{{','.join(map(str, window))}}}"
                               f" (persistent-defect filter: syndrome "
                               f"errors vetoed, gate/data errors kept)")
                    qzz = inject_dynamic_stage(
                        qzz, r, s, list(range(n_data)),
                        bit_fn, classical=True)
                    print(f"  dynamic correction injected: ZZ arm, from "
                          f"{src}, classical expressions; XX arm "
                          f"postselect-only per rule 1")

    for lab, qc in [("ZZ arm", qzz), ("XX arm", qxx)]:
        ops = qc.count_ops()
        print(f"{lab}: {qc.num_qubits}q, {ops.get('cx', 0)} CX, "
              f"depth {qc.depth()}")
    print(f"Backend: {be.name}  shots: {shots}  rounds: {n}  basis: {basis}")

    fn, ferr = _decoder()
    print("decoder.py:", "available" if fn else f"NOT loaded ({ferr}) — "
          f"corrections disabled, raw + postsel only")

    if dry:
        print("Dry run."); return

    if opts.job_id:
        jid = opts.job_id
        if jid == "last":
            done = [jb for jb in svc.jobs(limit=20, pending=False)
                    if "DONE" in str(jb.status()).upper()
                    or "COMPLETED" in str(jb.status()).upper()]
            if not done:
                print("no completed jobs found."); return
            jid = done[0].job_id()
        j = svc.job(jid)
        st = str(j.status())
        print(f"  fetching job {jid} (status: {st}) — decoding with "
              f"grid {r}x{s}, rounds {n}, basis {basis}, "
              f"steane={opts.steane}; these MUST match the submitted run.")
        res = j.result()
        if len(res) != 2:
            print(f"!! expected 2 pubs (ZZ, XX), job has {len(res)} — "
                  f"not a bell_complex job?"); return
        # registers must match the flags, else the reconstruction silently
        # decodes garbage (the zeroed-correlator bug class)
        pfx = "steane_" if opts.steane else "syn_"
        have = 0
        while hasattr(res[0].data, f"{pfx}{have}"):
            have += 1
        if opts.steane and have == 0 and hasattr(res[0].data, "syn_0"):
            print("!! job has syn_* registers, not steane_* — it was "
                  "submitted WITHOUT --steane"); return
        if not opts.steane and have == 0 and hasattr(res[0].data, "steane_0"):
            print("!! job has steane_* registers — it was submitted "
                  "WITH --steane"); return
        # expected register count mirrors build_circuit's qec_rounds
        want_regs = n  # (free_final_round would subtract 1; not a CLI flag here)
        if have != want_regs:
            print(f"!! job has {have} {pfx}* register(s) but --rounds {n} "
                  f"expects {want_regs} — pass --rounds {have}"); return
        dw = res[0].data.data.to_bool_array(order="little").shape[1]
        if dw != r * s:
            print(f"!! data register width {dw} != r*s = {r * s} — "
                  f"--grid doesn't match the submitted run"); return
    else:
        if opts.qiskit:
            def _best(qc):
                best_t, best_c = None, None
                for sd in range(max(1, opts.transpile_seeds)):
                    pm = generate_preset_pass_manager(
                        backend=be, optimization_level=3, seed_transpiler=sd,
                        initial_layout=plan["initial_layout"] if plan else None)
                    t = pm.run(qc)
                    c = sum(v for k, v in t.count_ops().items()
                            if k in ("cz", "ecr", "cx", "swap"))
                    if best_c is None or c < best_c:
                        best_t, best_c = t, c
                return best_t
            qzz_t, qxx_t = _best(qzz), _best(qxx)
        else:
            qzz_t = _transpile_pytket(qzz, be)
            qxx_t = _transpile_pytket(qxx, be)
        t2q = sum(v for t in (qzz_t, qxx_t) for k, v in t.count_ops().items()
                  if k in ("cz", "ecr", "cx", "swap"))
        print(f"  transpiled 2q: {t2q}")

        j = Sampler(mode=be).run([qzz_t, qxx_t], shots=shots)
        print(f"  job: {j.job_id()}\n  https://quantum.ibm.com/jobs/{j.job_id()}")
        print("  waiting …")
        try:
            res = j.result()
        except KeyboardInterrupt:
            print("\nDetached."); return

    # decode-arm follows the STREAM's basis type (basis[0] for multi-char
    # sequences like "ZX"): S_Z detects X errors, which flip Z-basis readout
    # only — rule 1 above.  `basis == "Z"` alone mis-routes "ZX" to XX.
    decoded_arm = "ZZ" if basis[0] == "Z" else "XX"
    valid = plaquette_validity(r, s)
    out = {}

    for arm, pub in zip(("ZZ", "XX"), res):
        db = pub.data.data.to_bool_array(order="little").astype(np.uint8)
        data = db.reshape(-1, r, s)
        nsh = data.shape[0]
        if n == 0:
            syn = np.zeros((nsh, 0, r, s), dtype=np.uint8)
        elif opts.steane:
            syn = steane_syndromes(pub, n, r, s)
        else:
            syn = all_syndromes_opt(pub, n, r, s, n_anc,
                                    no_reset=kw.get("no_reset", False),
                                    full_stabilizer=kw.get("full_stabilizer", False))
        m = pub.data.bell.to_bool_array(order="little")[:, 0].astype(np.uint8)

        # decoder-ready accumulated stream (re-references X-basis rounds
        # to their random first gauge sample; passthrough for Z-basis)
        if syn.shape[1] > 0:
            # accumulated_syndromes splits by individual basis chars (Z/X);
            # for multi-char bases like "ZX", grab the requested base stream
            stream_key = basis[0] if len(basis) > 1 else basis
            _, stream = accumulated_syndromes(syn, basis)[stream_key]
        else:
            stream = syn

        # the readout-basis-matched arm also yields a data-derived final
        # syndrome — append as the last decode round; the mixed XX readout
        # only yields valid plaquettes (postselection only).  Valid whenever
        # the decode stream is Z-type (basis[0]), since the ZZ arm's data
        # readout is Z regardless of the stabilizer sequence.
        d_syn = data_syndrome(data) if basis[0] == "Z" else None

        # audit: reconstruct what the in-circuit confirmation filter did.
        # steane registers return with every shot, so the veto decisions
        # are fully replayable offline.
        if opts.dynamic and opts.dynamic_confirm > 1 and opts.steane \
                and syn.shape[1] >= opts.dynamic_confirm:
            from dc_decode import numpy_confirmed
            zr = [t for t in range(n) if basis[t % len(basis)] == "Z"]
            if len(zr) >= opts.dynamic_confirm:
                widx = zr[-opts.dynamic_confirm:]
                _, vst = numpy_confirmed(syn, widx)
                tot = max(vst["raw_defects"], 1)
                print(f"{arm}: confirm-{opts.dynamic_confirm} audit "
                      f"(rounds {widx}): {vst['raw_defects']} raw defects "
                      f"in last round, {vst['confirmed']} confirmed, "
                      f"{vst['vetoed']} vetoed as syndrome errors "
                      f"({100 * vst['vetoed'] / tot:.1f}%)")

        corr = np.zeros_like(data)
        decoded = False
        if opts.dynamic:
            # data qubits already corrected mid-circuit; skip post-hoc decode
            # syndrome stream is from BEFORE the correction — using it would
            # double-correct (the bug class that produced the LER ceiling).
            pass
        elif arm == decoded_arm and fn is not None:
            full = stream
            if basis[0] == "Z":
                full = np.concatenate([stream, d_syn[:, None]], axis=1)
            corr, decoded = decode_stream(full, r, s, fn)
        fixed = data ^ corr

        l1 = fixed[:, 0, :].sum(axis=1) % 2
        l2 = fixed[:, :, 0].sum(axis=1) % 2
        bits = l1 ^ l2

        # quiet mask: all accumulated rounds zero, plus the arm's valid
        # data-derived plaquettes
        # With --dynamic the syndrome stream was measured BEFORE correction
        # and no longer reflects the corrected state — use data-derived S only.
        mask = np.ones(nsh, dtype=bool)
        if stream.shape[1] > 0 and not opts.dynamic:
            mask &= stream.reshape(nsh, -1).sum(axis=1) == 0
        if basis[0] == "Z":
            if arm == "ZZ":
                mask &= d_syn.reshape(nsh, -1).sum(axis=1) == 0
            else:
                mask &= d_syn[:, valid].sum(axis=1) == 0

        raw = 1.0 - 2.0 * bits.mean()
        frame = 1.0 - 2.0 * (bits ^ m).mean()
        wt = int(corr.sum(axis=(1, 2)).mean())
        label = " (decoded)" if decoded else ""
        label = label if label else (" (dynamic)" if opts.dynamic else "")
        print(f"  <{arm}> raw={raw:+.4f}  frame={frame:+.4f}  "
              f"corr wt={wt}{label}  "
              f"m=0/1: {int((m == 0).sum())}/{int((m == 1).sum())}  "
              f"quiet keep={mask.mean():.2f}")
        # --dynamic audit: the block register the expressions consumed is
        # in the results, so what the in-circuit conditionals applied is
        # fully reconstructible (bit-exact replica; see dc_decode)
        if opts.dynamic and arm == "ZZ" and n > 0:
            from dc_decode import numpy_correction
            z_rounds = [t for t in range(n) if basis[t % len(basis)] == "Z"]
            if z_rounds:
                if opts.dynamic_mode == "coherent" and \
                        hasattr(pub.data, "dc_corr_m"):
                    # direct in-circuit record: particular solution from
                    # dc_corr_m XOR the measured-majority row flips
                    Edc = pub.data.dc_corr_m.to_bool_array(
                        order='little').astype(np.uint8).reshape(nsh, r, s)
                    hr, hs = r // 2, s // 2
                    rowm = pub.data.dc_row_m.to_bool_array(
                        order='little').astype(np.uint8)
                    ridx = 0
                    for px in (0, 1):
                        for py in (0, 1):
                            for ha in range(1, hr):
                                a = (2 * ha) + px
                                grp = rowm[:, ridx:ridx + hs - 1]
                                ridx += hs - 1
                                rf = (grp.sum(axis=1)
                                      >= hs // 2 + 1).astype(np.uint8)
                                Edc[:, a, py::2] ^= rf[:, None]
                    src = "recorded (dc_corr_m ^ dc_row_m)"
                elif opts.dynamic_mode == "inplace" and \
                        hasattr(pub.data, "dc_row_m"):
                    # raw block readout is preserved (prefix uncomputed):
                    # replay the exact applied correction, fixup included
                    from dc_decode import numpy_inplace_replica
                    lz = z_rounds[-1]
                    braw = getattr(pub.data, f"steane_{lz}").to_bool_array(
                        order='little').astype(np.uint8).reshape(nsh, r, s)
                    rowm = pub.data.dc_row_m.to_bool_array(
                        order='little').astype(np.uint8)
                    Edc = numpy_inplace_replica(braw, rowm, r, s)
                    src = f"recorded (steane_{lz} + dc_row_m, fixup incl.)"
                else:
                    Edc = numpy_correction(syn[:, z_rounds[-1]], r, s)
                    src = "replica"
                fired = float(Edc.any(axis=(1, 2)).mean())
                mwt = float(Edc.sum(axis=(1, 2)).mean())
                lbits = ((Edc[:, 0, :].sum(axis=1)
                          + Edc[:, :, 0].sum(axis=1)) % 2).astype(np.uint8)
                lflip = float(lbits.mean())
                undone = 1.0 - 2.0 * (bits ^ lbits).mean()
                print(f"    dynamic audit [{src}]: correction fired on "
                      f"{fired:.1%} of shots, mean wt {mwt:.2f}, "
                      f"logical(Z_L1Z_L2) action on {lflip:.1%}; "
                      f"counterfactual <ZZ> without it: {undone:+.4f} "
                      f"(vs raw {raw:+.4f})")
        out[arm] = (bits, m, mask)

    # witness: ZZ raw, XX frame-corrected — never the other way around
    zz_bits, _, mzq = out["ZZ"]
    xx_bits, m_xx, mxq = out["XX"]
    W, zz, xx, sig = witness(zz_bits, xx_bits ^ m_xx)
    print(f"\n  W = <ZZ> + <XX>_frame = {zz:+.4f} + {xx:+.4f} = "
          f"{W:+.4f} ± {sig:.4f}"
          f"  ->  {'ENTANGLED (W-2σ > 1)' if W - 2 * sig > 1 else 'not certified'}")
    if mzq.sum() > 20 and mxq.sum() > 20:
        Wp, zzp, xxp, sigp = witness(zz_bits[mzq], (xx_bits ^ m_xx)[mxq])
        print(f"  postsel: {zzp:+.4f} + {xxp:+.4f} = {Wp:+.4f} ± {sigp:.4f} "
              f"(keep Z:{mzq.mean():.2f} X:{mxq.mean():.2f})"
              f"  ->  {'ENTANGLED (W-2σ > 1)' if Wp - 2 * sigp > 1 else 'not certified'}")
    print("  done.")


if __name__ == "__main__":
    main()