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
    opts = ap.parse_args()

    r, s = opts.grid; n = opts.rounds; shots = opts.shots
    n_anc = 4 * (r // 2 - 1) * (s // 2)
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

    qzz, *_ = build_circuit(r, s, n, **kw)
    qxx, *_ = build_circuit(r, s, n, partial_x=True, **kw)

    if opts.output_qasm:
        from qiskit import qasm3
        with open(opts.output_qasm, "w") as fp:
            fp.write(qasm3.dumps(qzz))
        print(f"ZZ arm QASM → {opts.output_qasm}")
        return

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

    if opts.qiskit:
        pm = generate_preset_pass_manager(backend=be, optimization_level=3,
                                          seed_transpiler=42,
                                          initial_layout=plan["initial_layout"] if plan else None)
        qzz_t, qxx_t = pm.run(qzz), pm.run(qxx)
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

    decoded_arm = "ZZ" if basis == "Z" else "XX"
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
        # only yields valid plaquettes (postselection only)
        d_syn = data_syndrome(data) if basis == "Z" else None

        corr = np.zeros_like(data)
        decoded = False
        if arm == decoded_arm and fn is not None:
            full = stream
            if basis == "Z":
                full = np.concatenate([stream, d_syn[:, None]], axis=1)
            corr, decoded = decode_stream(full, r, s, fn)
        fixed = data ^ corr

        l1 = fixed[:, 0, :].sum(axis=1) % 2
        l2 = fixed[:, :, 0].sum(axis=1) % 2
        bits = l1 ^ l2

        # quiet mask: all accumulated rounds zero, plus the arm's valid
        # data-derived plaquettes
        mask = np.ones(nsh, dtype=bool)
        if stream.shape[1] > 0:
            mask &= stream.reshape(nsh, -1).sum(axis=1) == 0
        if basis == "Z":
            if arm == "ZZ":
                mask &= d_syn.reshape(nsh, -1).sum(axis=1) == 0
            else:
                mask &= d_syn[:, valid].sum(axis=1) == 0

        raw = 1.0 - 2.0 * bits.mean()
        frame = 1.0 - 2.0 * (bits ^ m).mean()
        wt = int(corr.sum(axis=(1, 2)).mean())
        print(f"  <{arm}> raw={raw:+.4f}  frame={frame:+.4f}  "
              f"corr wt={wt}{' (decoded)' if decoded else ''}  "
              f"m=0/1: {int((m == 0).sum())}/{int((m == 1).sum())}  "
              f"quiet keep={mask.mean():.2f}")
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
