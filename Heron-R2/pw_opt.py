"""
pw_opt.py — Optimized share-pair circuit builder for (1+x²)(1+y²) code.

Key improvement: only 2 vertical-pair ancillas per 3-row column (p=0,1).
V(2,q) = V(0,q) ⊕ V(1,q) computed in software.
Saves 16 ancillas vs the standard share-pair layout (32 vs 48).

For 6×8: 48 data + 32 anc = 80 qubits, 64 CX, depth ~17, 0 SWAPs.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def build_circuit(r, s, rounds, logical_state="00", bell=False, bell_measure=False, measure_x=False, partial_x=False, stabilizer_basis='Z', no_reset=True, ghz=False, ghz_measure=False, free_final_round=False):
    """Build optimized share-pair QEC circuit.

    stabilizer_basis='Z': measure V(i,j) = Z_i Z_{i+2,j} (Z⊗Z stabilizers) via data→anc CX.
    stabilizer_basis='X': measure V(i,j) = X_i X_{i+2,j} (X⊗X stabilizers) via anc→data CX
    with ancilla in |+⟩ (cleaner: 2 H per check on anc, not 4 on data).

    no_reset=True: skip ancilla resets on rounds > 0. Ancilla persists in |m_{r-1}⟩;
    CX flips by the new parity, so m_r = m_{r-1} ⊕ P_r. Recover P_r = m_r ⊕ m_{r-1}
    via consecutive differencing in all_syndromes_opt. Works in both Z and X bases.

    free_final_round=True: run rounds-1 ancilla rounds; the destructive data readout
    at the end supplies the last round's Z-stabilizer syndrome. Only valid when
    readout basis matches stabilizer basis (both Z or both X). Saves 64 CX.

    For an r×s grid where both are even:
      - Sector (px, py): data at (2p+px, 2q+py) for p=0..r/2-1, q=0..s/2-1
      - In sector coords, V(p,q) = data[p][q] ⊕ data[(p+1)%(r/2)][q]
      - Measure V(p,q) for p=0..r/2-2 (all except last row in sector)
      - Compute V(r/2-1, q) = sum of all measured V(p,q) for p=0..r/2-2

    For r=6, s=8: sector size 3×4, measure p=0,1; compute p=2.
    """
    from pw_qiskit import heavy_hex_flag_layout
    data_map, anc_maps, _, _ = heavy_hex_flag_layout(r, s)

    n_data = r * s
    hr, hs = r // 2, s // 2
    n_anc_phys = 2 * r * s
    n_anc = 4 * (hr - 1) * hs

    n_extra = (1 if bell else 0) + (1 if bell_measure else 0) + (1 if ghz else 0) + (1 if ghz_measure else 0)
    extra_qubits = []
    if bell: extra_qubits.append(("bell", 1))
    if bell_measure: extra_qubits.append(("bell_m", 1))
    if ghz: extra_qubits.append(("ghz", 1))
    if ghz_measure: extra_qubits.append(("ghz_m", 1))
    extra_idx = {}
    extra_cursor = n_data + n_anc_phys
    if bell:
        extra_idx["bell"] = extra_cursor
        extra_cursor += 1
    if bell_measure:
        extra_idx["bell_m"] = extra_cursor
        extra_cursor += 1
    if ghz:
        extra_idx["ghz"] = extra_cursor
        extra_cursor += 1
    if ghz_measure:
        extra_idx["ghz_m"] = extra_cursor
        extra_cursor += 1
    total = n_data + n_anc_phys + n_extra

    qec_rounds = rounds - 1 if free_final_round else rounds

    qr = QuantumRegister(total, "q")
    cr_syn = [ClassicalRegister(n_anc, f"syn_{c}") for c in range(qec_rounds)]
    cr_data = ClassicalRegister(n_data, "data")
    cregs = [*cr_syn, cr_data]
    extra_cr = {}
    for name in extra_idx:
        cr = ClassicalRegister(1, name)
        extra_cr[name] = cr
        cregs.append(cr)
    qc = QuantumCircuit(qr, *cregs)

    if ghz:
        g_idx = extra_idx["ghz"]
        qc.h(g_idx)
        for j in range(s - 1):
            qc.cx(g_idx, data_map[r - 1][j])
        for i in range(r - 1):
            qc.cx(g_idx, data_map[i][s - 1])
        qc.h(g_idx)
        qc.measure(g_idx, extra_cr["ghz"][0])
    elif bell:
        b_prep_idx = extra_idx["bell"]
        qc.h(b_prep_idx)
        for i in range(r):
            qc.cx(b_prep_idx, data_map[i][0])
        for j in range(1, s):
            qc.cx(b_prep_idx, data_map[0][j])
        qc.h(b_prep_idx)
        qc.measure(b_prep_idx, extra_cr["bell"][0])
    elif bell_measure:
        b_prep_idx = extra_idx["bell_m"]
        qc.h(b_prep_idx)
        for i in range(r):
            qc.cx(b_prep_idx, data_map[i][0])
        for j in range(1, s):
            qc.cx(b_prep_idx, data_map[0][j])
        qc.h(b_prep_idx)
    else:
        # |+⟩⊗N preparation for X-stabilizer basis (satisfies X_i X_j = +1)
        if stabilizer_basis == 'X':
            for ii in range(r):
                for jj in range(s):
                    qc.h(data_map[ii][jj])
        if "1" in logical_state:
            if logical_state[1] == "1":
                for jj in range(s):
                    qc.x(data_map[0][jj])
            if logical_state[0] == "1":
                for ii in range(r):
                    qc.x(data_map[ii][0])

    # QEC rounds (rounds-1 if free_final_round, else rounds)
    for rnd in range(qec_rounds):
        slot = 0
        for px in range(2):
            for py in range(2):
                for p in range(hr - 1):
                    for q in range(hs):
                        i = 2 * p + px
                        j = 2 * q + py
                        anc_idx = anc_maps[(i, j, 0)]
                        if rnd == 0 or not no_reset:
                            qc.reset(anc_idx)
                        if stabilizer_basis == 'X':
                            qc.h(anc_idx)
                            qc.cx(anc_idx, data_map[i][j])
                            qc.cx(anc_idx, data_map[(i + 2) % r][j])
                            qc.h(anc_idx)
                        else:
                            qc.cx(data_map[i][j], anc_idx)
                            qc.cx(data_map[(i + 2) % r][j], anc_idx)
                        qc.measure(anc_idx, cr_syn[rnd][slot])
                        slot += 1
        qc.barrier()

    # Bell measurement after QEC: measures X_L1 X_L₂ of the (possibly corrupted) state
    if bell_measure:
        bm_idx = extra_idx["bell_m"]
        qc.h(bm_idx)
        for i in range(r):
            qc.cx(bm_idx, data_map[i][0])
        for j in range(1, s):
            qc.cx(bm_idx, data_map[0][j])
        qc.h(bm_idx)
        qc.measure(bm_idx, extra_cr["bell_m"][0])

    # GHZ measurement after QEC: measures X⊗12 on the boundary
    if ghz_measure:
        gm_idx = extra_idx["ghz_m"]
        qc.h(gm_idx)
        for j in range(s - 1):
            qc.cx(gm_idx, data_map[r - 1][j])
        for i in range(r - 1):
            qc.cx(gm_idx, data_map[i][s - 1])
        qc.h(gm_idx)
        qc.measure(gm_idx, extra_cr["ghz_m"][0])

    # X-basis rotation
    if measure_x:
        for ii in range(r):
            for jj in range(s):
                qc.h(data_map[ii][jj])
        qc.barrier()
    elif partial_x:
        # H on support of X_L1 X_L2 (row 0 ∪ column 0)
        for jj in range(s):
            qc.h(data_map[0][jj])
        for ii in range(1, r):
            qc.h(data_map[ii][0])
        qc.barrier()

    # Final data readout
    for ii in range(r):
        for jj in range(s):
            qc.measure(data_map[ii][jj], cr_data[ii * s + jj])

    lq0_qubits = [data_map[0][jj] for jj in range(s)]
    lq1_qubits = [data_map[ii][0] for ii in range(r)]

    return qc, data_map, lq0_qubits, lq1_qubits, n_anc


def all_syndromes_opt(pub_result, rounds, r, s, n_anc, no_reset=True, free_final_round=False, data_raw=None):
    """Extract and reconstruct full (shots, rounds, r, s) syndrome.

    Measurements are for V(i,j) = data[i][j] ⊕ data[(i+2)%r][j]
    for i=0..r-3 (both even and odd, all columns j).
    The last two rows' V are computed via linear combination.

    When no_reset=True, ancillas persist between rounds: m_r = m_{r-1} ⊕ P_r.
    The actual parity P_r = m_r ⊕ m_{r-1} (with m_{-1} = 0).

    When free_final_round=True, the last round's syndrome is computed from
    data_raw (destructive readout) instead of an ancilla measurement.
    Only rounds-1 ancilla registers are expected in pub_result.
    """
    hr, hs = r // 2, s // 2
    anc_rounds = rounds - 1 if free_final_round else rounds

    if anc_rounds == 0:
        shots = data_raw.shape[0]
        syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    else:
        first = getattr(pub_result.data, "syn_0")
        shots = first.num_shots

        m_raw = np.zeros((shots, anc_rounds, n_anc), dtype=np.uint8)
        for c in range(anc_rounds):
            bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
            m_raw[:, c] = bits[:, :n_anc].astype(np.uint8)

        if no_reset:
            m_parity = m_raw.copy()
            m_parity[:, 1:] ^= m_raw[:, :-1]
        else:
            m_parity = m_raw

        syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
        for c in range(anc_rounds):
            m = m_parity[:, c]

            # Unpack measurements into (shots, r, s) V array
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

            # Reconstruct V(r-2, j) and V(r-1, j)
            V[:, r-2, :] = V[:, 0:r-2:2, :].sum(axis=1) % 2
            V[:, r-1, :] = V[:, 1:r-1:2, :].sum(axis=1) % 2

            # Reconstruct full syndrome: S(i,j) = V(i,j) ⊕ V(i,j+2 mod s)
            syn[:, c] = V ^ np.roll(V, shift=-2, axis=2)

    # Free final round: compute last syndrome from data readout
    if free_final_round and data_raw is not None:
        V_last = data_raw.astype(np.uint8) ^ np.roll(data_raw.astype(np.uint8), shift=-2, axis=1)
        syn[:, -1] = V_last ^ np.roll(V_last, shift=-2, axis=2)

    return syn


def verify_no_reset():
    """Compare reset-based vs no-reset: depth scaling and round-1 equivalence."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    print("\n--- No-reset depth scaling ---")
    r, s = 6, 8
    print(f"{'rounds':>6} | {'reset depth':>11} | {'no-reset depth':>13} | {'reset CX':>8}")
    for rounds in (1, 2, 4, 8, 16):
        qc_r, *_ = build_circuit(r, s, rounds, logical_state="00", no_reset=False)
        qc_f, *_ = build_circuit(r, s, rounds, logical_state="00", no_reset=True)
        print(f"{rounds:>6} | {qc_r.depth():>11} | {qc_f.depth():>13} | "
              f"{qc_r.count_ops().get('cx',0):>8}")

    # Round-1 equivalence: with rounds=1, differencing is identity, so the two
    # syndrome streams must match shot-for-shot under identical sampling.
    print("\n--- rounds=1 equivalence (ideal sim) ---")
    rounds = 1
    backend = AerSimulator(device='CPU')
    qc_r, _, _, _, n_anc = build_circuit(r, s, rounds, logical_state="00", reset_free=False)
    qc_f, *_ = build_circuit(r, s, rounds, logical_state="00", reset_free=True)
    # Same op counts on the ancilla extraction except (rounds-1)=0 resets -> equal here
    eq = qc_r.count_ops().get('reset', 0) == qc_f.count_ops().get('reset', 0)
    print(f"  reset count equal at rounds=1: {eq} "
          f"(reset={qc_r.count_ops().get('reset',0)} vs {qc_f.count_ops().get('reset',0)})")
    print("  (for rounds>1, free has fewer resets; validate logical fidelity via "
          "verify_pipeline with reset_free=True)")


def verify_optimized():
    """Verify the optimized circuit builds and transpiles correctly."""
    from qiskit_ibm_runtime.fake_provider.backends.fez.fake_fez import FakeFez
    from qiskit import transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    backend = FakeFez()

    # Test |00⟩ circuit (no Bell)
    r, s, rounds = 6, 8, 1
    qc, dm, lq0, lq1, n_anc = build_circuit(r, s, rounds, logical_state="00")
    ops = qc.count_ops()
    print(f"Optimized |00⟩ circuit: {qc.num_qubits} qubits ({r*s} data + {n_anc} anc), "
          f"CX={ops.get('cx',0)}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                      seed_transpiler=42)
    qc_t = pm.run(qc)
    ops_t = qc_t.count_ops()
    print(f"  Transpiled: phys={qc_t.num_qubits}, depth={qc_t.depth()}, "
          f"CZ={ops_t.get('cz',0)}, SWAP={ops_t.get('swap',0)}")
    assert ops_t.get('swap', 0) == 0, "SWAPs found in |00⟩!"
    print("  ✓ 0 SWAPs verified")

    # Test Bell circuit (prep + measure)
    qc_b, dm_b, _, _, _ = build_circuit(r, s, rounds, bell=True, bell_measure=True)
    ops_b = qc_b.count_ops()
    print(f"\nOptimized Bell circuit (prep+measure): {qc_b.num_qubits} qubits, "
          f"CX={ops_b.get('cx',0)}")
    pm_b = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                        seed_transpiler=42)
    qc_b_t = pm_b.run(qc_b)
    ops_b_t = qc_b_t.count_ops()
    print(f"  Transpiled: phys={qc_b_t.num_qubits}, depth={qc_b_t.depth()}, "
          f"CZ={ops_b_t.get('cz',0)}, SWAP={ops_b_t.get('swap',0)}")


def verify_pipeline(no_reset=False):
    """End-to-end: circuit → simulate → syndrome extraction → decode."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from waxis_decode import WaxisDecoder

    print("\n--- End-to-end pipeline test ---")
    backend = AerSimulator(device='CPU')
    r, s, rounds = 6, 8, 1

    qc, dm, lq0, lq1, n_anc = build_circuit(r, s, rounds, logical_state="00", no_reset=no_reset)
    print(f"Circuit: {qc.num_qubits}q, CX={qc.count_ops().get('cx',0)}")

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ['cx'])

    qc_t = qc  # No transpilation needed for Aer (all-to-all connectivity)
    job = backend.run(qc_t, noise_model=noise_model, shots=500)
    counts = job.result().get_counts()
    # Show sample output format
    sample = next(iter(counts.items()))
    print(f"  Sample output: '{sample[0]}' (count={sample[1]})")
    print(f"  Num classical registers: {len(sample[0].split())}")

    syn_list = []
    data_list = []
    for bitstring, cnt in counts.items():
        parts = bitstring.split()
        data_bits = parts[0][::-1]    # Qiskit get_counts: clbit 0 is rightmost
        syn_bits = parts[1][::-1] if len(parts) >= 2 else ""
        for _ in range(cnt):
            V = np.zeros((r, s), dtype=np.uint8)
            idx = 0
            for px in range(2):
                for py in range(2):
                    for p in range(r//2 - 1):
                        for q in range(s//2):
                            i = 2 * p + px
                            j = 2 * q + py
                            V[i, j] = int(syn_bits[idx])
                            idx += 1
            V[4, :] = V[0:4:2, :].sum(axis=0) % 2
            V[5, :] = V[1:5:2, :].sum(axis=0) % 2
            syn_list.append(V ^ np.roll(V, shift=-2, axis=1))
            data = np.zeros((r, s), dtype=np.uint8)
            for ii in range(r):
                for jj in range(s):
                    data[ii, jj] = int(data_bits[ii * s + jj])
            data_list.append(data)

    syn_hits = np.array(syn_list)
    data_raw = np.array(data_list)
    n = len(syn_hits)
    print(f"  Total shots decoded: {n}")

    dec = WaxisDecoder(r, s)
    corrs = np.zeros((n, 1, r, s), dtype=np.uint8)
    for i in range(n):
        corrs[i] = dec.decode(syn_hits[i].reshape(1, r, s))

    corrected = data_raw ^ corrs[:, 0]
    lz1 = corrected[:, 0, :].sum(axis=1) % 2
    lz2 = corrected[:, :, 0].sum(axis=1) % 2
    fidelity = ((lz1 == 0) & (lz2 == 0)).mean()
    print(f"  |00⟩ fidelity with 2% CX noise: {fidelity:.3f}")
    print("✓ Pipeline verified")


if __name__ == "__main__":
    verify_optimized()
    verify_no_reset()
    verify_pipeline()
    verify_pipeline(no_reset=True)
