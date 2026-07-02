#!/usr/bin/env python3
"""All-logicals GHZ test for IonQ Forte — corrected.

Scheme (single circuit, both bases):
  1. Prepare |0...0>_L, then measure the joint operator X_all = prod_i X_Li
     via the bell ancilla.  Outcome m1 projects onto
         (|0...0>_L + (-1)^m1 |1...1>_L)/sqrt(2)   — a logical GHZ state.
  2. Run n rounds of Z stabilizers.
  3. Measure X_all AGAIN (bell_measure, fresh ancilla).  On a coherent GHZ
     state m2 == m1 deterministically; on a classical mixture P(m2==m1)=1/2.
     X_all commutes with every Z_i Z_j pair, so this probe does not disturb
     the Z-agreement statistic measured next.
  4. Measure all data in Z; compute all row/col Z logicals.

Witness:
  P  = P(all Z logicals equal)            (population term)
  C  = 2*P(m2 == m1) - 1                  (coherence term)
  F ~= (P + C)/2 ;  F > 0.5  =>  logical GHZ entanglement.
The old script compared Z-logical outcomes to m1.  m1 is an X-type outcome:
it is UNCORRELATED with Z readout, so that witness reads ~0 even for a
perfect GHZ state.  That is why it "did not work".
"""

import argparse, os
import numpy as np
from pw_opt import build_circuit, _check_anchors, _unpack_indices


# ---------- result parsing (counts -> per-shot register arrays) ----------

def counts_to_arrays(counts, qc):
    """Expand a counts dict into per-shot uint8 arrays keyed by creg name.
    Qiskit count keys list registers last-added-first, space separated;
    bits within a register are big-endian (leftmost = highest index)."""
    regs = list(qc.cregs)
    sizes = [rg.size for rg in regs]
    per_reg = {rg.name: [] for rg in regs}
    for key, cnt in counts.items():
        parts = key.split()
        if len(parts) != len(regs):            # unspaced key: split by size
            k, parts = key.replace(" ", ""), []
            for sz in reversed(sizes):
                parts.append(k[:sz]); k = k[sz:]
        for part, rg in zip(parts, reversed(regs)):
            bits = np.frombuffer(part[::-1].encode(), np.uint8) - ord("0")
            per_reg[rg.name].append(np.tile(bits, (cnt, 1)))
    return {nm: np.concatenate(v).astype(np.uint8) for nm, v in per_reg.items()}


def syndromes_from_arrays(arrs, rounds, r, s, n_anc, no_reset=True,
                          full_stabilizer=False, periodic=True):
    """Same logic as pw_opt.all_syndromes_opt, but on plain arrays."""
    if rounds == 0:
        return None
    shots = arrs["syn_0"].shape[0]
    m_raw = np.stack([arrs[f"syn_{c}"][:, :n_anc] for c in range(rounds)], axis=1)
    m_par = m_raw.copy()
    if no_reset:
        m_par[:, 1:] ^= m_raw[:, :-1]
    ui, uj = _unpack_indices(r, s)
    V = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    V[:, :, ui, uj] = m_par
    if periodic:
        V[:, :, r - 2, :] = V[:, :, 0:r - 2:2, :].sum(axis=2) % 2
        V[:, :, r - 1, :] = V[:, :, 1:r - 1:2, :].sum(axis=2) % 2
    return V if full_stabilizer else V ^ np.roll(V, shift=-2, axis=3)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--backend", "-b", default="aer",
                    help="'aer' (local sim) or an IonQ backend, e.g. "
                         "'ionq_simulator', 'qpu.forte-1'")
    ap.add_argument("--grid", type=int, nargs=2, default=(4, 4))
    ap.add_argument("--periodic", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-qasm", default=None)
    opts = ap.parse_args()

    r, s = opts.grid; n = opts.rounds
    n_anc = 4 * (r // 2 - 1) * (s // 2)
    n_logicals = (r - 1) + (s - 1)

    qc, *_ = build_circuit(
        r, s, n, logical_state="00",
        bell=True, bell_ancilla=True,          # X_all prep -> m1
        bell_measure=True,                     # X_all probe -> m2 (coherence)
        stabilizer_basis="Z", periodic=opts.periodic,
        full_stabilizer=True)   # weight-4 checks: the only ones that commute
                                # with the all-logicals X operator (see notes)

    ops = qc.count_ops()
    print(f"All-logicals GHZ: {r}x{s} (periodic={opts.periodic}), "
          f"{n_logicals} logicals, {n} rounds")
    print(f"  {qc.num_qubits}q, {ops.get('cx', 0)} CX, depth {qc.depth()}")

    if opts.output_qasm:
        from qiskit import qasm3
        with open(opts.output_qasm, "w") as f:
            f.write(qasm3.dumps(qc))
        print(f"QASM -> {opts.output_qasm}"); return
    if opts.dry_run:
        print("Dry run."); return

    # ---- execution: IonQ provider (NOT IBM Runtime) or local Aer ----
    if opts.backend == "aer":
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        be = AerSimulator()
        job = be.run(transpile(qc, be), shots=opts.shots)
    else:
        from qiskit_ionq import IonQProvider          # pip install qiskit-ionq
        provider = IonQProvider(os.environ.get("IONQ_API_KEY"))
        be = provider.get_backend(opts.backend)
        from qiskit import transpile
        qc_t = transpile(qc, be)
        print(f"Backend: {be.name()}  (verify mid-circuit measurement support!)")
        job = be.run(qc_t, shots=opts.shots)
        print(f"  job: {job.job_id()}")

    counts = job.result().get_counts()
    arrs = counts_to_arrays(counts, qc)
    shots = arrs["data"].shape[0]
    data = arrs["data"].reshape(-1, r, s)
    m1 = arrs["bell"][:, 0]
    m2 = arrs["bell_m"][:, 0]

    # optional decoding of Z errors from syndromes
    fixed = data
    if n > 0:
        syn = syndromes_from_arrays(arrs, n, r, s, n_anc,
                                    full_stabilizer=True,
                                    periodic=opts.periodic)
        try:
            from decoder import tesseract_decode_ffinal
            full = np.zeros((shots, n + 1, r, s), dtype=np.uint8)  # decoder API
            full[:, :n] = syn
            corr = np.array([tesseract_decode_ffinal(full[i], r, s)
                             for i in range(shots)])
            fixed = data ^ corr
        except ImportError:
            print("  (decoder.py not found — reporting raw, undecoded data)")

    # Z logicals: row parities 0..r-2, col parities 0..s-2
    logicals = {f"Z_row_{i}": fixed[:, i, :].sum(1) % 2 for i in range(r - 1)}
    logicals |= {f"Z_col_{c}": fixed[:, :, c].sum(1) % 2 for c in range(s - 1)}
    vals = np.column_stack([logicals[k] for k in sorted(logicals)])

    P = (vals.max(1) == vals.min(1)).mean()        # all Z logicals agree
    C = 2 * (m1 == m2).mean() - 1                  # X_all coherence
    F = 0.5 * (P + C)

    print(f"\n  m1 split: {np.bincount(m1, minlength=2)}  (want ~50/50)")
    print(f"  Population  P(all Z logicals equal) = {P:.4f}")
    print(f"  Coherence   C = 2*P(m2==m1)-1       = {C:+.4f}")
    print(f"  Fidelity estimate F ~ (P+C)/2       = {F:.4f}"
          f"   -> {'GHZ ENTANGLED' if F > 0.5 else 'not certified'}")
    print(f"  P(all zero) = {(vals.sum(1) == 0).mean():.4f}  "
          f"(should be ~{0.5:.2f} for GHZ, ~1.0 for product |0..0>)")
    for nm in sorted(logicals):
        print(f"    {nm}: <Z>={1 - 2 * logicals[nm].mean():+.3f}")
    print("  done.")


if __name__ == "__main__":
    main()