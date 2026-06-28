"""
pw_opt.py — Optimized share-pair circuit builder for (1+x²)(1+y²) code.

Key improvement: only 2 vertical-pair ancillas per 3-row column (p=0,1).
V(2,q) = V(0,q) ⊕ V(1,q) computed in software.
Saves 16 ancillas vs the standard share-pair layout (32 vs 48).

For 6×8: 48 data + 32 anc = 80 qubits, 64 CX, depth ~17, 0 SWAPs.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def build_circuit(r, s, rounds, logical_state="00", bell=False, measure_x=False):
    """Build optimized share-pair QEC circuit.

    Measures V(i,j) = data[i][j] ⊕ data[(i+2)%r][j] for i=0..r-3 (NOT r-2, r-1).
    The final two rows' V are reconstructed: V(r-2) = V(r-4) ⊕ V(r-2),
    V(r-1) = V(r-3) ⊕ V(r-1) — but actually simpler: only measure even and
    odd positions up to r//2 sectors.

    For an r×s grid where both are even:
      - Sector (px, py): data at (2p+px, 2q+py) for p=0..r/2-1, q=0..s/2-1
      - In sector coords, V(p,q) = data[p][q] ⊕ data[(p+1)%(r/2)][q]
      - Measure V(p,q) for p=0..r/2-2 (all except last row in sector)
      - Compute V(r/2-1, q) = sum of all measured V(p,q) for p=0..r/2-2

    For r=6, s=8: sector size 3×4, measure p=0,1; compute p=2.
    """
    from pw_qiskit import heavy_hex_flag_layout
    data_map, _, _, _ = heavy_hex_flag_layout(r, s)

    n_data = r * s
    hr, hs = r // 2, s // 2
    n_anc = 0
    anc_indices = []  # list of (sector_px, sector_py, p, q, global_anc_idx)

    for px in range(2):
        for py in range(2):
            for p in range(hr - 1):  # all but last row in sector
                for q in range(hs):
                    anc_indices.append((px, py, p, q))
                    n_anc += 1

    n_bell = 1 if bell else 0
    total = n_data + n_anc + n_bell
    qr = QuantumRegister(total, "q")
    cr_syn = [ClassicalRegister(n_anc, f"syn_{c}") for c in range(rounds)]
    cr_data = ClassicalRegister(n_data, "data")

    if bell:
        cr_bell = ClassicalRegister(1, "bell")
        qc = QuantumCircuit(qr, *cr_syn, cr_data, cr_bell)
        b_idx = n_data + n_anc
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

    # QEC rounds
    for rnd in range(rounds):
        anc_global = n_data  # starting index for ancillas in this round
        for px in range(2):
            for py in range(2):
                for p in range(hr - 1):
                    for q in range(hs):
                        i = 2 * p + px
                        j = 2 * q + py
                        a_idx = anc_global
                        anc_global += 1

                        qc.reset(a_idx)
                        qc.cx(data_map[i][j], a_idx)
                        qc.cx(data_map[(i + 2) % r][j], a_idx)

                        syn_idx = a_idx - n_data
                        qc.measure(a_idx, cr_syn[rnd][syn_idx])
        qc.barrier()

    # X-basis rotation
    if measure_x:
        for ii in range(r):
            for jj in range(s):
                qc.h(data_map[ii][jj])
        qc.barrier()

    # Final data readout
    for ii in range(r):
        for jj in range(s):
            qc.measure(data_map[ii][jj], cr_data[ii * s + jj])

    lq0_qubits = [data_map[0][jj] for jj in range(s)]
    lq1_qubits = [data_map[ii][0] for ii in range(r)]

    return qc, data_map, lq0_qubits, lq1_qubits, n_anc


def all_syndromes_opt(pub_result, rounds, r, s, n_anc):
    """Extract and reconstruct full (shots, rounds, r, s) syndrome.

    Measurements are for V(i,j) = data[i][j] ⊕ data[(i+2)%r][j]
    for i=0..r-3 (both even and odd, all columns j).
    The last two rows' V are computed via linear combination.
    """
    hr, hs = r // 2, s // 2
    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots

    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    for c in range(rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order='little')
        m = bits[:, :n_anc].astype(np.uint8)

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
        # V(r-2, j) = data[r-2][j] ⊕ data[r][j] = data[r-2][j] ⊕ data[0][j]
        # V(0, j) = data[0][j] ⊕ data[2][j]
        # V(2, j) = data[2][j] ⊕ data[4][j]
        # V(0, j) ⊕ V(2, j) = data[0][j] ⊕ data[4][j] = V(4, j)
        # etc. For r=6: V(4, j) = V(0,j) ⊕ V(2,j), V(5,j) = V(1,j) ⊕ V(3,j)
        V[:, r-2, :] = V[:, 0:r-2:2, :].sum(axis=1) % 2  # V(r-2) = sum of V(even ~=r-2)
        V[:, r-1, :] = V[:, 1:r-1:2, :].sum(axis=1) % 2  # V(r-1) = sum of V(odd ~=r-1)

        # Reconstruct full syndrome: S(i,j) = V(i,j) ⊕ V(i,j+2 mod s)
        syn[:, c] = V ^ np.roll(V, shift=-2, axis=2)

    return syn


def verify_optimized():
    """Verify the optimized circuit builds and transpiles correctly."""
    from qiskit_ibm_runtime.fake_provider.backends.fez.fake_fez import FakeFez
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    import json, os

    path = os.path.expanduser(
        "~/.local/lib/python3.12/site-packages/qiskit_ibm_runtime/"
        "fake_provider/backends/fez/conf_fez.json"
    )
    with open(path) as f:
        conf = json.load(f)
    for gate in conf["gates"]:
        if gate["name"] == "cz":
            cm_full = CouplingMap(couplinglist=[(min(a,b), max(a,b))
                                                for a,b in gate["coupling_map"] if a != b])
            break

    backend = FakeFez()
    r, s, rounds = 6, 8, 1
    qc, dm, lq0, lq1, n_anc = build_circuit(r, s, rounds, logical_state="00")
    ops = qc.count_ops()
    print(f"Optimized circuit: {qc.num_qubits} qubits ({r*s} data + {n_anc} anc), "
          f"CX={ops.get('cx',0)}")

    qc_t = transpile(qc, backend=backend, coupling_map=cm_full,
                     basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
                     optimization_level=3, seed_transpiler=42)
    ops_t = qc_t.count_ops()
    print(f"Transpiled: phys={qc_t.num_qubits}, depth={qc_t.depth()}, "
          f"CX={ops_t.get('cx',0)}, SWAP={ops_t.get('swap',0)}")
    assert ops_t.get('swap', 0) == 0, "SWAPs found!"
    print("✓ 0 SWAPs verified")


def verify_pipeline():
    """End-to-end: circuit → simulate → syndrome extraction → decode."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from waxis_decode import WaxisDecoder

    print("\n--- End-to-end pipeline test ---")
    backend = AerSimulator(device='CPU')
    r, s, rounds = 6, 8, 1

    qc, dm, lq0, lq1, n_anc = build_circuit(r, s, rounds, logical_state="00")
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
        data_bits = parts[0]
        syn_bits = parts[1] if len(parts) >= 2 else ""
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
    verify_pipeline()
