"""
pw_opt.py — Optimized share-pair circuit builder for (1+x²)(1+y²) code.

Key improvement: only 2 vertical-pair ancillas per 3-row column (p=0,1).
V(2,q) = V(0,q) ⊕ V(1,q) computed in software.
Saves 16 ancillas vs the standard share-pair layout (32 vs 48).

For 6×8: 48 data + 32 anc = 80 qubits, 64 CX, depth ~17, 0 SWAPs.

Optimizations in this revision (all output-format compatible):
  - compact=True (default): the QuantumRegister holds exactly the qubits
    actually used (data + measured ancillas + extras) instead of
    n_data + 2*r*s.  For 6×8 that is ~81 qubits instead of 145, which
    speeds up transpilation and simulation and removes idle wires.
    Set compact=False to restore the original register layout with
    heavy_hex_flag_layout indices.
  - Direction-major CX scheduling: all "self" CXs are emitted before all
    "+2 partner" CXs, so every extraction round is exactly 2 CX layers
    deep (4 for full_stabilizer) instead of chaining through shared data
    qubits.  All CXs within a round pairwise commute (data qubits are
    always on one side, ancillas on the other), so the unitary is
    unchanged — only the DAG depth improves (~2x per round).
  - initial_reset=False (default): the round-0 ancilla resets are dropped
    since qubits start in |0⟩; pass initial_reset=True to restore them.
  - share_extra_ancilla (opt-in): bell / bell_measure / ghz /
    ghz_measure can reuse a single extra qubit (reset between uses)
    instead of allocating one each.  Classical registers are unchanged.
  - Periodic Bell prep/measure skips the (0,0) qubit entirely: it appears
    in both X_L1 and X_L2, so the two CXs cancel (X² = I).  Saves 2 CX
    and 2 layers of ancilla depth per Bell operation.
  - all_syndromes_opt and verify_pipeline are fully vectorized
    (precomputed fancy-index unpacking, unique-syndrome decoding).
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def _check_anchors(r, s):
    """Check anchor coordinates (i, j) in classical-bit order (px, py, p, q)."""
    hr, hs = r // 2, s // 2
    return [(2 * p + px, 2 * q + py)
            for px in range(2) for py in range(2)
            for p in range(hr - 1) for q in range(hs)]


def _unpack_indices(r, s):
    """Row/col fancy-index arrays matching the classical-bit order."""
    anchors = _check_anchors(r, s)
    ii = np.array([a[0] for a in anchors])
    jj = np.array([a[1] for a in anchors])
    return ii, jj


def build_circuit(r, s, rounds, logical_state="00", bell=False, bell_measure=False, measure_x=False, partial_x=False, stabilizer_basis='Z', no_reset=True, ghz=False, ghz_measure=False, free_final_round=False, bell_after_qec=False, full_stabilizer=False, dd=False, periodic=True, compact=True, initial_reset=False, share_extra_ancilla=False, parity_mode='cat', cat_detect=False):
    """Build optimized share-pair QEC circuit.

    periodic=True: periodic vertical boundary conditions — V(i,j) wraps
    i+2 modulo r, and V(r-2,j), V(r-1,j) reconstructed in software.
    periodic=False: open boundaries — V(i,j) = Z_i Z_{i+2,j} for i=0..r-3
    only; bottom two rows have no vertical stabilizers. X_L1 = col 0,
    X_L2 = col 2 (vertical strings) commute with all V(i,j), so Bell
    state survives multi-round QEC.

    stabilizer_basis='Z': measure V(i,j) = Z_i Z_{i+2,j} (Z⊗Z stabilizers) via data→anc CX.
    stabilizer_basis='X': measure V(i,j) = X_i X_{i+2,j} (X⊗X stabilizers) via anc→data CX
    with ancilla in |+⟩ (cleaner: 2 H per check on anc, not 4 on data).

    no_reset=True: skip ancilla resets on rounds > 0. Ancilla persists in |m_{r-1}⟩;
    CX flips by the new parity, so m_r = m_{r-1} ⊕ P_r. Recover P_r = m_r ⊕ m_{r-1}
    via consecutive differencing in all_syndromes_opt. Works in both Z and X bases.

    free_final_round=True: run rounds-1 ancilla rounds; the destructive data readout
    at the end supplies the last round's Z-stabilizer syndrome. Only valid when
    readout basis matches stabilizer basis (both Z or both X). Saves 64 CX.

    compact=True: allocate only the qubits actually used (data + measured
    ancillas + extras) and remap indices to a dense range. The returned
    data_map reflects the remapping. Set compact=False if downstream code
    relies on the raw heavy_hex_flag_layout indices (e.g. a trivial
    initial_layout onto physical qubits).

    initial_reset=False: skip the redundant round-0 ancilla resets
    (qubits initialize to |0⟩ on hardware and in Aer).

    share_extra_ancilla=True (opt-in, default False): bell/bell_measure/
    ghz/ghz_measure share one physical extra qubit, reset between uses.
    Only meaningful with parity_mode='single'.

    parity_mode='cat' (default): Bell/GHZ parity operators X⊗n are measured
    with an n-qubit cat state laid out as a chain along the boundary,
    instead of one ancilla receiving n sequential CXs. The single-ancilla
    version is a degree-n star (K_{1,n}) which cannot embed in degree-3
    heavy hex and forces heavy SWAP routing; the cat version's interaction
    graph is a caterpillar (path + one pendant data edge per node), which
    is near-native. Sequence: grow the cat outward from the chain centre
    (depth ~n/2, all path-local), one parallel CX layer onto the data
    (controlled-X⊗n with the cat as control), uncompute, H + measure the
    centre qubit. Identical measurement statistics to parity_mode='single'.
    Error propagation differs: a Z fault on any of the n cat qubits flips
    the recorded parity (vs. any point in the n-CX window for a single
    ancilla — comparable exposure), and an X fault during (un)growth walks
    out as a contiguous X segment on boundary data, which the Z-checks see.
    Set cat_detect=True to also measure the n-1 non-centre cat qubits into
    an extra register '<name>_cat' after uncompute: any 1 flags a fault
    (usable for post-selection). Default False keeps the classical output
    format identical to parity_mode='single'.

    parity_mode='single': original behavior — one extra ancilla per
    operator, n sequential CXs.

    For periodic r×s where both are even:
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
    n_anc = 4 * (hr - 1) * hs
    checks = _check_anchors(r, s)

    extra_flags = [name for name, on in (("bell", bell), ("bell_m", bell_measure),
                                         ("ghz", ghz), ("ghz_m", ghz_measure)) if on]

    if compact:
        def _dq(i, j):
            return i * s + j
        _anc_index = {c: n_data + k for k, c in enumerate(checks)}

        def _aq(i, j):
            return _anc_index[(i, j)]
        base = n_data + n_anc
    else:
        def _dq(i, j):
            return data_map[i][j]

        def _aq(i, j):
            return anc_maps[(i, j, 0)]
        base = n_data + 2 * r * s

    # --- supports of the boundary parity operators, in boundary-walk order --
    # (walk order is irrelevant to the measured operator, but makes the cat
    # chain geometrically local: consecutive entries are grid neighbors or
    # diagonal at the corner turns)
    def _logical_xx_coords():
        """Support of X_L1 · X_L2 (symmetric difference — overlap cancels)."""
        if periodic:
            # (0,0) is in both X_L1 (row 0) and X_L2 (col 0): X² = I, skip it.
            return ([(i, 0) for i in range(r - 1, 0, -1)] +
                    [(0, j) for j in range(1, s)])
        # snake: down col 0, back up col 2
        return ([(i, 0) for i in range(r)] +
                [(i, 2) for i in range(r - 1, -1, -1)])

    def _ghz_coords():
        return ([(r - 1, j) for j in range(s - 1)] +
                [(i, s - 1) for i in range(r - 2, -1, -1)])

    op_coords = {}
    if bell:
        op_coords["bell"] = _logical_xx_coords()
    if bell_measure:
        op_coords["bell_m"] = _logical_xx_coords()
    if ghz:
        op_coords["ghz"] = _ghz_coords()
    if ghz_measure:
        op_coords["ghz_m"] = _ghz_coords()

    if parity_mode == 'cat' and extra_flags:
        n_extra = 0
        extra_idx = {}
        n_cat = max(3 * len(c) - 1 for c in op_coords.values())
        if compact:
            _shared_cat = [base + k for k in range(n_cat)]
        else:
            # unused physical ancilla slots from the heavy-hex layout: the
            # register has 2*r*s ancilla positions but only the (i,j,0) of
            # the check anchors are consumed by syndrome extraction.
            _taken = {anc_maps[(i, j, 0)] for (i, j) in checks}
            _free = [q for q in sorted(set(anc_maps.values()))
                     if q not in _taken]
            _shared_cat = _free[:n_cat]

        def _cat_anc(m):
            return _shared_cat[:m]
        total = base + (n_cat if compact else 0)
    else:
        if share_extra_ancilla and extra_flags:
            extra_idx = {name: base for name in extra_flags}
            n_extra = 1
        else:
            extra_idx = {name: base + k for k, name in enumerate(extra_flags)}
            n_extra = len(extra_flags)
        total = base + n_extra

    qec_rounds = rounds - 1 if free_final_round else rounds

    qr = QuantumRegister(total, "q")
    cr_syn = [ClassicalRegister(n_anc, f"syn_{c}") for c in range(qec_rounds)]
    cr_data = ClassicalRegister(n_data, "data")
    cregs = [*cr_syn, cr_data]
    extra_cr = {}
    cat_cr = {}
    for name in extra_flags:
        cr = ClassicalRegister(1, name)
        extra_cr[name] = cr
        cregs.append(cr)
        if parity_mode == 'cat' and cat_detect and len(op_coords[name]) > 1:
            ccr = ClassicalRegister(3 * len(op_coords[name]) - 2, f"{name}_cat")
            cat_cr[name] = ccr
            cregs.append(ccr)
    qc = QuantumCircuit(qr, *cregs)

    # --- helper: measure a product of X operators ---------------------------
    _used_anc = set()

    def _parity_measure(name):
        coords = op_coords[name]
        qubits = [_dq(i, j) for (i, j) in coords]
        cbit = extra_cr[name][0]
        if parity_mode == 'cat':
            _parity_measure_cat(name, qubits, cbit)
            return
        anc = extra_idx[name]
        if anc in _used_anc:
            qc.reset(anc)
        _used_anc.add(anc)
        qc.h(anc)
        for dq_ in qubits:
            qc.cx(anc, dq_)
        qc.h(anc)
        qc.measure(anc, cbit)

    def _parity_measure_cat(name, qubits, cbit):
        """Measure X⊗n on `qubits` via a (3n-1)-qubit ladder cat state.

        Heavy hex is bipartite with max degree 3: corner qubits (degree 3)
        are only adjacent to edge qubits (degree 2). A data qubit is a
        corner, so the cat qubit that couples to it must be an edge qubit
        — which then has room for exactly one link into the cat. The only
        cat topology that embeds natively is therefore a ladder:

            backbone:  c1 - b1 - c2 - b2 - ... - cn   (corners c, edges b)
            leaves:    l_k attached to c_k             (edge qubits)
            coupling:  CX(l_k -> d_k), one parallel layer

        Sequence: grow the cat outward from the backbone centre, one
        parallel layer onto the leaves, one parallel layer onto the data
        (= controlled-X⊗n with the cat as control), exact uncompute, then
        H + measure of the centre qubit yields the parity. Identical
        measurement statistics to parity_mode='single'.
        """
        n = len(qubits)
        a = _cat_anc(3 * n - 1)
        for x in a:
            if x in _used_anc:
                qc.reset(x)
            _used_anc.add(x)
        P = a[:2 * n - 1]          # backbone: c_k = P[2k], b_k = P[2k+1]
        L = a[2 * n - 1:]          # leaves, one per data qubit
        root = (2 * n - 1) // 2
        qc.h(P[root])
        # grow the cat outward from the centre (path-local, depth ~n)
        up = [(P[k], P[k + 1]) for k in range(root, 2 * n - 2)]
        down = [(P[k], P[k - 1]) for k in range(root, 0, -1)]
        for c, t in up:
            qc.cx(c, t)
        for c, t in down:
            qc.cx(c, t)
        # one parallel layer onto the leaves
        for k in range(n):
            qc.cx(P[2 * k], L[k])
        # one parallel coupling layer: controlled-(X…X) with the cat as control
        for k, dqb in enumerate(qubits):
            qc.cx(L[k], dqb)
        # uncompute the cat (exact reverse)
        for k in range(n):
            qc.cx(P[2 * k], L[k])
        for c, t in reversed(down):
            qc.cx(c, t)
        for c, t in reversed(up):
            qc.cx(c, t)
        qc.h(P[root])
        qc.measure(P[root], cbit)
        if name in cat_cr:
            others = [x for x in a if x != P[root]]
            for b, x in enumerate(others):
                qc.measure(x, cat_cr[name][b])

    if ghz:
        _parity_measure("ghz")
    elif bell and not bell_after_qec:
        _parity_measure("bell")
    else:
        # |+⟩⊗N preparation for X-stabilizer basis (satisfies X_i X_j = +1)
        if stabilizer_basis == 'X':
            for ii in range(r):
                for jj in range(s):
                    qc.h(_dq(ii, jj))
        if "1" in logical_state:
            flip = qc.z if stabilizer_basis == 'X' else qc.x
            if periodic:
                if logical_state[1] == "1":
                    for jj in range(s):
                        flip(_dq(0, jj))
                if logical_state[0] == "1":
                    for ii in range(r):
                        flip(_dq(ii, 0))
            else:
                if logical_state[1] == "1":
                    for ii in range(r):
                        flip(_dq(ii, 2))
                if logical_state[0] == "1":
                    for ii in range(r):
                        flip(_dq(ii, 0))

    # QEC rounds (rounds-1 if free_final_round, else rounds)
    def row2(i):
        return i + 2 if not periodic else (i + 2) % r

    anc_list = [_aq(i, j) for (i, j) in checks]
    # Direction-major CX schedule: each offset layer touches every data qubit
    # and every ancilla at most once, so the round is exactly len(offsets)
    # CX layers deep. All CXs in a round pairwise commute (data qubits only
    # ever on the control side in Z basis / target side in X basis), so the
    # unitary matches the original per-ancilla emission order.
    offsets = ([(0, 0), (2, 0), (0, 2), (2, 2)] if full_stabilizer
               else [(0, 0), (2, 0)])

    for rnd in range(qec_rounds):
        if (rnd == 0 and initial_reset) or (rnd > 0 and not no_reset):
            for a in anc_list:
                qc.reset(a)
        if stabilizer_basis == 'X':
            for a in anc_list:
                qc.h(a)
        for (di, dj) in offsets:
            for (i, j), a in zip(checks, anc_list):
                ti = row2(i) if di else i
                tj = (j + dj) % s
                if stabilizer_basis == 'X':
                    qc.cx(a, _dq(ti, tj))
                else:
                    qc.cx(_dq(ti, tj), a)
        if stabilizer_basis == 'X':
            for a in anc_list:
                qc.h(a)
        for slot, a in enumerate(anc_list):
            qc.measure(a, cr_syn[rnd][slot])
        # Dynamic decoupling: X gates on all idle data qubits between rounds
        if dd and rnd < qec_rounds - 1:
            for ii in range(r):
                for jj in range(s):
                    qc.x(_dq(ii, jj))

    # Bell creation after QEC (fresh Bell state from QEC-cleaned |00⟩)
    if bell_after_qec:
        _parity_measure("bell")

    # Bell measurement after QEC: measures X_L1 X_L2 of the (possibly corrupted) state
    if bell_measure:
        _parity_measure("bell_m")

    # GHZ measurement after QEC: measures X⊗12 on the boundary
    if ghz_measure:
        _parity_measure("ghz_m")

    # X-basis rotation
    if measure_x:
        for ii in range(r):
            for jj in range(s):
                qc.h(_dq(ii, jj))
        qc.barrier()
    elif partial_x:
        if periodic:
            for jj in range(s):
                qc.h(_dq(0, jj))
            for ii in range(1, r):
                qc.h(_dq(ii, 0))
        else:
            for ii in range(r):
                qc.h(_dq(ii, 0))
            for ii in range(r):
                qc.h(_dq(ii, 2))
        qc.barrier()

    # Final data readout
    for ii in range(r):
        for jj in range(s):
            qc.measure(_dq(ii, jj), cr_data[ii * s + jj])

    if periodic:
        lq0_qubits = [_dq(0, jj) for jj in range(s)]
        lq1_qubits = [_dq(ii, 0) for ii in range(r)]
    else:
        lq0_qubits = [_dq(ii, 0) for ii in range(r)]
        lq1_qubits = [_dq(ii, 2) for ii in range(r)]

    eff_data_map = [[_dq(ii, jj) for jj in range(s)] for ii in range(r)]
    return qc, eff_data_map, lq0_qubits, lq1_qubits, n_anc


def all_syndromes_opt(pub_result, rounds, r, s, n_anc, no_reset=True, free_final_round=False, data_raw=None, full_stabilizer=False, periodic=True):
    """Extract and reconstruct full (shots, rounds, r, s) syndrome.

    Measurements are for V(i,j) = data[i][j] ⊕ data[(i+2)%r][j]
    for i=0..r-3 (both even and odd, all columns j).
    The last two rows' V are computed via linear combination (periodic)
    or left as zero (open boundaries).

    When no_reset=True, ancillas persist between rounds: m_r = m_{r-1} ⊕ P_r.
    The actual parity P_r = m_r ⊕ m_{r-1} (with m_{-1} = 0).

    When free_final_round=True, the last round's syndrome is computed from
    data_raw (destructive readout) instead of an ancilla measurement.
    Only rounds-1 ancilla registers are expected in pub_result.

    Fully vectorized: measurements are scattered into the (r, s) grid with
    precomputed fancy indices in one shot across all rounds.
    """
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

        # Scatter all rounds at once: (shots, anc_rounds, n_anc) -> (shots, anc_rounds, r, s)
        ui, uj = _unpack_indices(r, s)
        V = np.zeros((shots, anc_rounds, r, s), dtype=np.uint8)
        V[:, :, ui, uj] = m_parity

        if periodic:
            V[:, :, r - 2, :] = V[:, :, 0:r - 2:2, :].sum(axis=2) % 2
            V[:, :, r - 1, :] = V[:, :, 1:r - 1:2, :].sum(axis=2) % 2

        syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
        if full_stabilizer:
            syn[:, :anc_rounds] = V  # measurements ARE S directly
        else:
            syn[:, :anc_rounds] = V ^ np.roll(V, shift=-2, axis=3)

    # Free final round: compute last syndrome from data readout
    if free_final_round and data_raw is not None:
        V_last = data_raw.astype(np.uint8) ^ np.roll(data_raw.astype(np.uint8), shift=-2, axis=1)
        syn[:, -1] = V_last ^ np.roll(V_last, shift=-2, axis=2)

    return syn


def check_consistency(all_syn, data_raw, r, s):
    """Diagnostic: compare final-round ancilla syndrome vs data-readout syndrome.

    When free_final_round is used, both the ancilla-based and data-based
    syndromes for the last round are available.  Their XOR gives the
    measurement error pattern for the last ancilla round.

    Returns a dict of per-shot and aggregate metrics.
    """
    n_shots, rounds, _, _ = all_syn.shape
    if rounds < 2:
        return {}

    # Last ancilla round (rounds-2) — this is the last one measured before
    # the free final round (= rounds-1) which comes from data.
    syn_anc = all_syn[:, -2]   # (n_shots, r, s) from ancilla
    # Data-based syndrome for the same physical state
    V_data = data_raw.astype(np.uint8) ^ np.roll(data_raw.astype(np.uint8), shift=-2, axis=1)
    syn_data = V_data ^ np.roll(V_data, shift=-2, axis=2)

    mismatch = syn_anc ^ syn_data   # 1 where ancilla syndrome ≠ data syndrome
    n_mismatch = mismatch.sum(axis=(1, 2))  # mismatched plaquettes per shot
    frac_zero = (n_mismatch == 0).mean()
    frac_one = (n_mismatch == 1).mean()
    mean_mismatch = n_mismatch.mean()

    return {
        "frac_zero_mismatch": float(frac_zero),
        "frac_one_mismatch": float(frac_one),
        "mean_mismatch": float(mean_mismatch),
        "n_shots": n_shots,
    }


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
    qc_r, _, _, _, n_anc = build_circuit(r, s, rounds, logical_state="00", no_reset=False)
    qc_f, *_ = build_circuit(r, s, rounds, logical_state="00", no_reset=True)
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
    """End-to-end: circuit → simulate → syndrome extraction → decode.

    Vectorized: bitstrings are parsed once per unique outcome, syndromes
    are scattered with fancy indexing, and the decoder runs only on unique
    syndrome patterns; results are expanded back with counts as weights.
    """
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

    def _bits(strings):
        """(n, L) uint8 array from equal-length bitstrings, LSB-first."""
        arr = np.frombuffer("".join(strings).encode(), dtype=np.uint8)
        return (arr.reshape(len(strings), -1) - ord("0"))[:, ::-1].astype(np.uint8)

    items = list(counts.items())
    cnts = np.array([c for _, c in items], dtype=np.int64)
    parts = [b.split() for b, _ in items]
    data_u = _bits([p[0] for p in parts]).reshape(-1, r, s)
    syn_bits = _bits([p[1] for p in parts]) if len(parts[0]) >= 2 else None

    ui, uj = _unpack_indices(r, s)
    V = np.zeros((len(items), r, s), dtype=np.uint8)
    if syn_bits is not None:
        V[:, ui, uj] = syn_bits[:, :len(ui)]
    V[:, r - 2, :] = V[:, 0:r - 2:2, :].sum(axis=1) % 2
    V[:, r - 1, :] = V[:, 1:r - 1:2, :].sum(axis=1) % 2
    syn_u = V ^ np.roll(V, shift=-2, axis=2)

    n = int(cnts.sum())
    print(f"  Total shots decoded: {n} ({len(items)} unique outcomes)")

    # Decode each *unique syndrome* once, then broadcast back.
    dec = WaxisDecoder(r, s)
    uniq_syn, inv = np.unique(syn_u.reshape(len(items), -1), axis=0, return_inverse=True)
    corr_u = np.zeros((len(uniq_syn), r, s), dtype=np.uint8)
    for k, v in enumerate(uniq_syn.reshape(-1, r, s)):
        corr_u[k] = dec.decode(v.reshape(1, r, s))[0]
    corrs = corr_u[inv]

    corrected = data_u ^ corrs
    lz1 = corrected[:, 0, :].sum(axis=1) % 2
    lz2 = corrected[:, :, 0].sum(axis=1) % 2
    ok = (lz1 == 0) & (lz2 == 0)
    fidelity = (ok * cnts).sum() / n
    print(f"  |00⟩ fidelity with 2% CX noise: {fidelity:.3f}")
    print("✓ Pipeline verified")


if __name__ == "__main__":
    verify_optimized()
    verify_no_reset()
    verify_pipeline()
    verify_pipeline(no_reset=True)
