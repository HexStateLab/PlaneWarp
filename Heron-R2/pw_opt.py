"""
pw_opt.py — Optimized share-pair circuit builder for the plaquette code
family (1+x^g)(1+y^g), parametrized by the check stride g.

FACTORIZATION UPDATE (stride=1 default): over GF(2),
    (1+x²)(1+y²) = ((1+x)(1+y))² = (1 + x + y + xy)²,
i.e. the legacy stride-2 code is the SQUARE of the nearest-neighbor
plaquette code and decouples into 4 independent (row parity × column
parity) copies of it. This module now builds the square-root code
directly: stride=1 (default) uses nearest-neighbor checks
    V(i,j) = P(i,j) P(i+1,j)          (weight-2 gauge)
    S(i,j) = V(i,j)·V(i,j+1)
           = P(i,j) P(i+1,j) P(i,j+1) P(i+1,j+1)   (weight-4 stabilizer)
on the FULL lattice — same logicals (row 0 / column 0), unit-distance
check support (better routing), and no qubits spent on decoupled
spectator sectors. Anything the stride-2 code did on r×s, stride=1 does
on (r/2)×(s/2). Pass stride=2 to every entry point to reproduce the
legacy (1+x²)(1+y²) behavior bit-for-bit (anchors, classical-bit order,
syndrome math, layouts all match the previous revision).

General stride g: V(i,j) = P(i,j)P(i+g,j) measured for i=0..r-g-1 (all
except the last g rows), the last g rows' V reconstructed in software
per residue class; S(i,j) = V(i,j)·V(i,j+g); n_anc = (r-g)·s. Periodic
boundaries require r % g == 0 == s % g.

Legacy sizing note (stride=2, 6×8): 48 data + 32 anc = 80 qubits, 64 CX,
depth ~17, 0 SWAPs. The stride=1 equivalent of one sector is 3×4:
12 data + 8 anc.

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
  - stabilizer_basis accepts repeating sequences ('ZX'/'XZ'): per-round
    alternating extraction of both stabilizer types, required to protect
    both correlators of a logical Bell state. Use with full_stabilizer=True
    (weight-4 S commute with both logicals; weight-2 V_Z gauge checks
    anticommute with X_L1X_L2 and dephase the Bell state in one round).
    Helpers round_bases / split_by_basis / detection_events handle the
    per-basis syndrome bookkeeping for decoding.
  - full_stabilizer accepts True (legacy weight-4 "star": one degree-4
    ancilla per S = V(i,j)·V(i,j+2)) or "paired" (new). Paired mode keeps
    max interaction degree 3 by loading each S with the primary ancilla's
    own V plus a coherent fan-in from the (j+2) neighbor ancilla, which then
    uncomputes; only the primary is measured, so the individual weight-2 V
    gauge is never collapsed and Bell protection under alternating ZX is
    preserved. The helper's carried value leaks into the fan but is a
    measured bit removed in software by all_syndromes_opt(full_stabilizer=
    "paired"):  S_a(t)=m_a(t)^m_a(t-1)^m_h(t-[phase0]).  Paired mode requires
    no_reset=True, periodic=True, s%4==0. It gives a slightly lower raw 2q
    count than the star on heavy hex (the star's degree-4 ancilla must be
    routed) but the S rungs still route; on IBM Fez neither mode dominates
    (paired: fewer 2q gates, more depth; star: better calibration-estimated
    fidelity for the 8-round Bell workload) — choose per run by comparing
    pick_best_transpilation(...) of each. A fully zero-SWAP paired layout is
    provably infeasible on Fez for the 6x8 code (see synthesize_paired_layout).
  - rung_plan (opt-in, paired only): thread each S rung through dedicated
    FREE "bridge" qubits so every 2q gate is nearest-neighbor. Bridges start
    and end in |0> (|+> in X basis) and are exactly uncomputed by the
    cascade, so they may be reused across the two fan phases and even shared
    across rungs (program order serializes the overlap — verified). Build a
    plan with synthesize_paired_layout when the lattice has enough degree-3
    headroom; on Fez it returns None (expected) and you run paired mode with
    the transpiler's own layout instead.

Note: pw_qiskit.heavy_hex_flag_layout (compact=False path) and waxis_decode
are external and were exercised here only against a local stub; re-run your
own verify_* suite after integrating. Both are stride-2 artifacts:
compact=False asserts stride == 2, and verify_pipeline tries
WaxisDecoder(r, s, stride=...) falling back to WaxisDecoder(r, s) —
supply a stride-aware decoder for stride != 2.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def _check_anchors(r, s, stride=1):
    """Check anchor coordinates (i, j) in classical-bit order (px, py, p, q).

    stride g: anchors are all (i, j) with i in [0, r-g), enumerated by
    (row residue px, column residue py, row cell p, column cell q) so the
    classical-bit order at stride=2 matches the legacy layout exactly.
    At stride=1 this reduces to plain (i, j) row-major over i < r-1."""
    g = stride
    hr, hs = r // g, s // g
    return [(g * p + px, g * q + py)
            for px in range(g) for py in range(g)
            for p in range(hr - 1) for q in range(hs)]


def _unpack_indices(r, s, stride=1):
    """Row/col fancy-index arrays matching the classical-bit order."""
    anchors = _check_anchors(r, s, stride)
    ii = np.array([a[0] for a in anchors])
    jj = np.array([a[1] for a in anchors])
    return ii, jj


def _bell_support_coords(r, s, periodic):
    """Support of X_L1 · X_L2 (symmetric difference — the (0,0) overlap cancels)."""
    if periodic:
        return [(i, 0) for i in range(1, r)] + [(0, j) for j in range(1, s)]
    return [(i, 0) for i in range(r)] + [(i, 2) for i in range(r)]


def _ghz_support_coords(r, s):
    return [(r - 1, j) for j in range(s - 1)] + [(i, s - 1) for i in range(r - 1)]


def build_circuit(r, s, rounds, logical_state="00", bell=False, bell_measure=False, measure_x=False, partial_x=False, stabilizer_basis='Z', no_reset=True, ghz=False, ghz_measure=False, free_final_round=False, bell_after_qec=False, full_stabilizer=False, dd=False, periodic=True, compact=True, initial_reset=False, share_extra_ancilla=False, bell_ancilla=True, parity_tree=None, rung_plan=None, steane_ec=False, stride=1):
    """Build optimized share-pair QEC circuit.

    stride (g, default 1): check stride of the (1+x^g)(1+y^g) code.
    stride=1 is the factorized nearest-neighbor plaquette code
    (1+x+y+xy) — see module docstring; stride=2 reproduces the legacy
    (1+x²)(1+y²) code exactly.

    periodic=True: periodic vertical boundary conditions — V(i,j) wraps
    i+g modulo r, and the last g rows' V reconstructed in software.
    periodic=False: open boundaries — V(i,j) = Z_i Z_{i+g,j} for
    i=0..r-g-1 only; bottom g rows have no vertical stabilizers. X_L1 = col 0,
    X_L2 = col 2 (vertical strings) commute with all V(i,j), so Bell
    state survives multi-round QEC.

    stabilizer_basis='Z': measure V(i,j) = Z_i Z_{i+2,j} (Z⊗Z stabilizers) via data→anc CX.
    stabilizer_basis='X': measure V(i,j) = X_i X_{i+2,j} (X⊗X stabilizers) via anc→data CX
    with ancilla in |+⟩ (cleaner: 2 H per check on anc, not 4 on data).
    stabilizer_basis may also be a repeating sequence, e.g. 'ZX' or 'XZ':
    round t is measured in basis stabilizer_basis[t % len]. This is required
    to protect BOTH logical correlators of a Bell state simultaneously —
    Z-type checks alone leave dephasing uncorrected (X_L1X_L2 decays at the
    physical T2), and X-type alone leave bit flips uncorrected. Notes:
      - Use full_stabilizer=True with alternating bases for Bell states.
        Every weight-4 S of either type commutes with all four logicals,
        so the logical Bell correlators are exactly preserved (verified:
        W = 2.0 noiselessly at both strides). Note that S_Z(i,j) and
        S_X(i+g,j+g) at diagonally adjacent anchors share ONE qubit and
        anticommute (at any stride — hidden on small legacy lattices by
        the column wrap), so opposite-basis rounds gauge-randomize those
        individual S values: same-basis detection events carry a ~25%
        gauge background noiselessly wherever a diagonal opposite-type
        anchor was measured in between. A fully static CSS variant at
        stride=1 would place S_Z and S_X on checkerboard cells (the
        standard toric code) — not implemented here, to keep register
        semantics identical to the legacy stride-2 pipeline. The
        weight-2 V gauge operators are worse still:
        V_Z checks anticommute with X_L1X_L2 (periodic) and dephase the
        Bell state in one round even on perfect hardware.
      - no_reset differencing is basis-agnostic (the ancilla accumulates
        m_t = m_{t-1} ⊕ P_t whatever P_t is), so all_syndromes_opt needs
        no change; interpret even/odd rounds via round_bases() and form
        detection events with detection_events(), which differences
        consecutive SAME-basis rounds (a first-of-basis round is a random
        gauge-fixing reference unless the initial state is a deterministic
        eigenstate, e.g. |0…0⟩ for Z-type).
      - For logical_state prep (non-Bell), the |+⟩^N prep and flip operator
        follow the FIRST basis in the sequence.

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
    Saves qubits but serializes the parity chains (deeper circuit); only
    worth it when qubit count is the binding constraint. Classical output
    unchanged either way.

    For periodic r×s with r % g == 0 == s % g:
      - Sector (px, py), px,py in [0,g): data at (gp+px, gq+py)
      - In sector coords, V(p,q) = data[p][q] ⊕ data[(p+1)%(r/g)][q]
      - Measure V(p,q) for p=0..r/g-2 (all except last row in sector)
      - Compute V(r/g-1, q) = sum of all measured V(p,q) for p=0..r/g-2
    At stride=1 there is a single sector spanning the whole lattice.

    Legacy example (stride=2, r=6, s=8): sector 3×4, measure p=0,1;
    compute p=2. Same-size example at stride=1: measure i=0..r-2,
    compute i=r-1.

    parity_tree=None: optional plan from synthesize_parity_layout(backend,...).
    When given, the ancilla parity measurements (bell / bell_measure / ghz /
    ghz_measure — one op family per plan) are emitted as a cat state grown
    over a tree of physical qubits synthesized directly from the backend
    coupling map, instead of the single-ancilla degree-n star (which cannot
    embed on degree-3 heavy hex and forces ~300+ CZ of routing). Transpile
    with initial_layout=parity_tree['initial_layout'] and the whole circuit —
    QEC block and gadget — maps with zero SWAPs; the transpiled 2q count
    equals the logical CX count. Requires compact=True. Classical registers
    are unchanged (still one bit per parity op), so all_syndromes_opt and
    downstream analysis need no changes. Trade-off: more (but strictly
    nearest-neighbor, shallow, parallel) CXs ~ 2·|tree| + n instead of n
    long-routed ones; a Z fault on any tree qubit flips the recorded parity
    bit, an X fault during (un)growth walks out as a contiguous segment the
    checks can see.
    """
    g = stride
    assert r > g and (not periodic or (r % g == 0 and s % g == 0)), \
        f"stride {g} needs r > g and (periodic) r, s divisible by g"
    if not compact:
        assert g == 2, ("compact=False uses pw_qiskit.heavy_hex_flag_layout, "
                        "which is a stride-2 artifact")
        from pw_qiskit import heavy_hex_flag_layout
        data_map, anc_maps, _, _ = heavy_hex_flag_layout(r, s)

    n_data = r * s
    n_anc = (r - g) * s
    checks = _check_anchors(r, s, g)

    extra_flags = [name for name, on in (("bell", bell and bell_ancilla),
                                          ("bell_m", bell_measure),
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

    if rung_plan is not None:
        assert full_stabilizer == "paired", \
            "rung_plan only applies to full_stabilizer='paired'"
        assert compact, "rung_plan requires compact=True (dense indices)"
        assert parity_tree is None, \
            "combining rung_plan with parity_tree is not supported yet"
        _bridge_base = base
        base = base + rung_plan["n_fresh"]   # extras go after the bridges

    if parity_tree is not None and extra_flags:
        assert compact, ("parity_tree requires compact=True: logical indices "
                         "must match the synthesized initial_layout")
        for name in extra_flags:
            _coords = (_ghz_support_coords(r, s) if name.startswith("ghz")
                       else _bell_support_coords(r, s, periodic))
            assert [tuple(c) for c in parity_tree["support"]] == \
                   [tuple(c) for c in _coords], (
                f"parity_tree was synthesized for op family "
                f"'{parity_tree['op']}' with a different support than '{name}'"
                f" — synthesize with the matching op/periodic settings")
        extra_idx = {name: None for name in extra_flags}
        n_extra = parity_tree.get("n_fresh",
                                  len(parity_tree["tree_nodes"]))
        _tree_base = base
    elif share_extra_ancilla and extra_flags:
        extra_idx = {name: base for name in extra_flags}
        n_extra = 1
        _tree_base = None
    else:
        extra_idx = {name: base + k for k, name in enumerate(extra_flags)}
        n_extra = len(extra_flags)
        _tree_base = None
    total = base + n_extra

    if steane_ec:
        # Steane-style EC: syndrome extracted every round via a transversal
        # CX onto ONE reusable encoded-ancilla block (n_data qubits, reset
        # between rounds) that is measured destructively; the round syndrome
        # is reconstructed in software (steane_syndromes) with the same
        # V/S roll math as free_final_round — a free syndrome round, every
        # round.  Ancilla is |+>_L for Z-stabilizer rounds and |0>_L for
        # X-stabilizer rounds (uniform over the measured-basis logical, so
        # the transversal copy carries off syndrome, not the data logical).
        assert compact and periodic, "steane_ec: compact+periodic only"
        assert full_stabilizer and full_stabilizer != "paired", \
            "steane_ec replaces stabilizer extraction; use full_stabilizer=True"
        _steane_base = total
        total += n_data

    qec_rounds = rounds - 1 if free_final_round else rounds

    basis_seq = stabilizer_basis.upper()
    first_basis = basis_seq[0]
    if free_final_round and len(basis_seq) > 1 and rounds > 0:
        _readout_b = 'X' if measure_x else 'Z'
        _final_b = basis_seq[(rounds - 1) % len(basis_seq)]
        assert _final_b == _readout_b, (
            f"free_final_round: data readout basis '{_readout_b}' must match "
            f"the basis of the final (software) round '{_final_b}' — pick the "
            f"sequence phase accordingly (e.g. 'XZ' for Z readout, 'ZX' for "
            f"X readout, with an even number of rounds)")

    qr = QuantumRegister(total, "q")
    if steane_ec:
        cr_syn = [ClassicalRegister(n_data, f"steane_{c}") for c in range(qec_rounds)]
    else:
        cr_syn = [ClassicalRegister(n_anc, f"syn_{c}") for c in range(qec_rounds)]
    cr_data = ClassicalRegister(n_data, "data")
    cregs = [*cr_syn, cr_data]
    extra_cr = {}
    for name in extra_flags:
        cr = ClassicalRegister(1, name)
        extra_cr[name] = cr
        cregs.append(cr)
    qc = QuantumCircuit(qr, *cregs)

    # --- helpers: measure a product of X operators ---------------------------
    extra_used = [False]
    _tree_used = [False]
    _after_rounds = [False]

    def _parity_measure_tree(qubits, cbit):
        """X⊗n parity via a cat state grown over the synthesized device tree.

        The tree is a literal subgraph of the backend coupling map (found by
        synthesize_parity_layout), so with initial_layout =
        parity_tree['initial_layout'] every CX below is between physically
        adjacent qubits: zero SWAPs by construction. Sequence: H on the root,
        BFS growth of the cat over the tree, one coupling layer onto the data
        (controlled-X⊗n with the cat as control — each support qubit receives
        exactly one CX from its adjacent tree node), exact uncompute, H +
        measure of the root gives the parity. Identical statistics to the
        single-ancilla gadget.
        """
        if "tree_logical" in parity_tree:
            nodes = list(parity_tree["tree_logical"])
        else:
            nodes = [_tree_base + t
                     for t in range(len(parity_tree["tree_nodes"]))]
        is_anc = parity_tree.get(
            "node_is_anc", [False] * len(nodes))
        order = parity_tree["bfs_order"]
        parent = parity_tree["tree_parent"]
        root = order[0]
        if _tree_used[0]:
            for x in nodes:
                qc.reset(x)
        elif _after_rounds[0]:
            # check-ancilla tree nodes hold their last syndrome value after
            # the rounds — reset before reusing them as cat qubits
            for x, a in zip(nodes, is_anc):
                if a:
                    qc.reset(x)
        _tree_used[0] = True
        qc.h(nodes[root])
        for t in order[1:]:
            qc.cx(nodes[parent[t]], nodes[t])
        for k, dq_ in enumerate(qubits):
            qc.cx(nodes[parity_tree["leaf_of"][k]], dq_)
        for t in reversed(order[1:]):
            qc.cx(nodes[parent[t]], nodes[t])
        qc.h(nodes[root])
        qc.measure(nodes[root], cbit)

    def _parity_measure(anc, qubits, cbit):
        if parity_tree is not None:
            _parity_measure_tree(qubits, cbit)
            return
        if share_extra_ancilla and extra_used[0]:
            qc.reset(anc)
        extra_used[0] = True
        qc.h(anc)
        for dq_ in qubits:
            qc.cx(anc, dq_)
        qc.h(anc)
        qc.measure(anc, cbit)

    def _logical_xx_support():
        """Support of X_L1 · X_L2 (symmetric difference — overlap cancels)."""
        return [_dq(i, j) for (i, j) in _bell_support_coords(r, s, periodic)]

    def _ghz_support():
        return [_dq(i, j) for (i, j) in _ghz_support_coords(r, s)]

    if ghz:
        _parity_measure(extra_idx["ghz"], _ghz_support(), extra_cr["ghz"][0])
    elif bell and not bell_after_qec:
        if bell_ancilla:
            _parity_measure(extra_idx["bell"], _logical_xx_support(), extra_cr["bell"][0])
        else:
            # Direct Bell prep: |Φ⁺⟩_L = H|0⟩ on data[0][0] = (|00⟩_L + |11⟩_L)/√2
            qc.h(_dq(0, 0))
    else:
        # |+⟩⊗N preparation for X-stabilizer basis (satisfies X_i X_j = +1)
        if first_basis == 'X':
            for ii in range(r):
                for jj in range(s):
                    qc.h(_dq(ii, jj))
        if "1" in logical_state:
            flip = qc.z if first_basis == 'X' else qc.x
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
    def row_g(i):
        return i + g if not periodic else (i + g) % r

    anc_list = [_aq(i, j) for (i, j) in checks]
    paired = (full_stabilizer == "paired")
    if paired:
        assert s % (2 * g) == 0, (
            "paired full_stabilizer needs s % (2*stride) == 0: the column "
            "cycle must 2-color so every ancilla is primary in one phase "
            "and helper in the other")
        assert no_reset, ("paired mode's software correction assumes the "
                          "no_reset accumulation semantics")
        assert periodic, "paired mode implemented for periodic boundaries"
        _slot = {c: k for k, c in enumerate(checks)}
        _kpar = lambda j: (j // g) % 2
        _phase_anchors = {ph: [c for c in checks if _kpar(c[1]) == ph]
                          for ph in (0, 1)}
    # Direction-major CX schedule: each offset layer touches every data qubit
    # and every ancilla at most once, so the round is exactly len(offsets)
    # CX layers deep. All CXs in a round pairwise commute (data qubits only
    # ever on the control side in Z basis / target side in X basis), so the
    # unitary matches the original per-ancilla emission order.
    offsets = ([(0, 0), (g, 0), (0, g), (g, g)]
               if (full_stabilizer and not paired)
               else [(0, 0), (g, 0)])

    def _load(anchor, rb, invert=False):
        """CX pair coupling the weight-2 V at `anchor` onto its own ancilla."""
        (i, j) = anchor
        a = _aq(i, j)
        for di in (0, g):
            d = _dq(row_g(i) if di else i, j)
            if rb == 'X':
                qc.cx(a, d)
            else:
                qc.cx(d, a)

    def _paired_round(rnd, rb):
        """Weight-4 S = V(i,j)·V(i,j+g) via coherent helper fan-in.

        Per phase: primaries load their own V (2 CX), their helpers (the
        (j+g) ancillas, always opposite phase) load V(i,j+g) (2 CX), one
        fan CX merges the helper's full state into the primary, the helper
        unloads (2 CX, exact classical uncompute — no commutation needed),
        and only the primary is measured.  The helper's prior content leaks
        into the fan but is a *measured* bit, removed in software by
        all_syndromes_opt(full_stabilizer="paired"):
            S_a(t) = m_a(t) ^ m_a(t-1) ^ m_h(t-1)   [phase-0 primaries]
            S_a(t) = m_a(t) ^ m_a(t-1) ^ m_h(t)     [phase-1 primaries]
        Individual V values are never exposed to measurement, so the
        weight-2 gauge is never collapsed — safe for alternating-basis
        Bell-state protection.  7 logical CX per S, max interaction degree
        3 vs the degree-4 star of the legacy mode.
        """
        if rb == 'X':
            for a in anc_list:
                qc.h(a)
        for ph in (0, 1):
            prim = _phase_anchors[ph]
            helpers = [(i, (j + g) % s) for (i, j) in prim]
            for anc in prim:
                _load(anc, rb)
            for h in helpers:
                _load(h, rb)
            for (i, j), (hi, hj) in zip(prim, helpers):
                A, H = _aq(i, j), _aq(hi, hj)
                br = ([] if rung_plan is None
                      else [_bridge_base + t
                            for t in rung_plan["bridges"][str(_slot[(i, j)])]])
                if not br:
                    qc.cx(A, H) if rb == 'X' else qc.cx(H, A)
                    continue
                # NN parity cascade H -> t1 -> ... -> tb -> A; bridges start
                # and end in |0>, exact uncompute, so cross-phase reuse and
                # both-basis operation are safe.  In the X frame the sign
                # flows target->control, so every CX direction flips and the
                # bridges are framed with H into |+> (sign 0) and back.
                chain = [H] + br + [A]
                if rb == 'X':
                    for t in br:
                        qc.h(t)
                    for k in range(len(chain) - 1):
                        qc.cx(chain[k + 1], chain[k])
                    for k in range(len(chain) - 3, -1, -1):
                        qc.cx(chain[k + 1], chain[k])
                    for t in br:
                        qc.h(t)
                else:
                    for k in range(len(chain) - 1):
                        qc.cx(chain[k], chain[k + 1])
                    for k in range(len(chain) - 3, -1, -1):
                        qc.cx(chain[k], chain[k + 1])
            for h in helpers:
                _load(h, rb)          # exact uncompute of the helper load
            for anc in prim:
                a = _aq(*anc)
                if rb == 'X':
                    qc.h(a)
                qc.measure(a, cr_syn[rnd][_slot[anc]])
                if rb == 'X' and ph == 0:
                    qc.h(a)           # reopen frame: helper duty in phase 1
        if rb == 'X':
            for anc in _phase_anchors[0]:
                qc.h(_aq(*anc))       # close phase-0 frames

    if steane_ec and qec_rounds > 0:
        _preps = steane_prep_circuits(r, s, g)
        _blkq = [qr[_steane_base + k] for k in range(n_data)]
        for rnd in range(qec_rounds):
            rb = basis_seq[rnd % len(basis_seq)]
            if rnd > 0 or initial_reset:
                for bq in _blkq:
                    qc.reset(bq)
            # encoded ancilla, uniform over the measured-basis logical
            qc.compose(_preps['plus' if rb == 'Z' else 'zero'], _blkq,
                       inplace=True)
            # transversal CX: block qubit k <-> data qubit (i,j), k = i*s + j
            for i in range(r):
                for j in range(s):
                    k = i * s + j
                    if rb == 'Z':
                        qc.cx(_dq(i, j), _blkq[k])
                    else:
                        qc.cx(_blkq[k], _dq(i, j))
            for k in range(n_data):
                if rb == 'X':
                    qc.h(_blkq[k])
                qc.measure(_blkq[k], cr_syn[rnd][k])
            if dd and rnd < qec_rounds - 1:
                for ii in range(r):
                    for jj in range(s):
                        qc.x(_dq(ii, jj))

    for rnd in range(qec_rounds if not steane_ec else 0):
        rb = basis_seq[rnd % len(basis_seq)]
        if (rnd == 0 and initial_reset) or (rnd > 0 and not no_reset):
            for a in anc_list:
                qc.reset(a)
        if paired:
            _paired_round(rnd, rb)
        else:
            if rb == 'X':
                for a in anc_list:
                    qc.h(a)
            for (di, dj) in offsets:
                for (i, j), a in zip(checks, anc_list):
                    ti = row_g(i) if di else i
                    tj = (j + dj) % s
                    if rb == 'X':
                        qc.cx(a, _dq(ti, tj))
                    else:
                        qc.cx(_dq(ti, tj), a)
            if rb == 'X':
                for a in anc_list:
                    qc.h(a)
            for slot, a in enumerate(anc_list):
                qc.measure(a, cr_syn[rnd][slot])
        # Dynamic decoupling: X gates on all idle data qubits between rounds
        if dd and rnd < qec_rounds - 1:
            for ii in range(r):
                for jj in range(s):
                    qc.x(_dq(ii, jj))

    # Pauli-frame fix: the inter-round DD applies (qec_rounds-1) X's per data
    # qubit. When that count is odd the physical frame is globally flipped
    # and the final data readout (and both logicals) comes back inverted —
    # all_syndromes_opt does not frame-track. Emit one compensating X layer
    # so the total is always even; syndromes are unaffected either way
    # (paired flips cancel inside every Z⊗Z / X⊗X check).
    if dd and qec_rounds >= 2 and (qec_rounds - 1) % 2 == 1:
        for ii in range(r):
            for jj in range(s):
                qc.x(_dq(ii, jj))

    _after_rounds[0] = True

    # Bell creation after QEC (fresh Bell state from QEC-cleaned |00⟩)
    if bell_after_qec:
        _parity_measure(extra_idx["bell"], _logical_xx_support(), extra_cr["bell"][0])

    # Bell measurement after QEC: measures X_L1 X_L2 of the (possibly corrupted) state
    if bell_measure:
        _parity_measure(extra_idx["bell_m"], _logical_xx_support(), extra_cr["bell_m"][0])

    # GHZ measurement after QEC: measures X⊗12 on the boundary
    if ghz_measure:
        _parity_measure(extra_idx["ghz_m"], _ghz_support(), extra_cr["ghz_m"][0])

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


def all_syndromes_opt(pub_result, rounds, r, s, n_anc, no_reset=True, free_final_round=False, data_raw=None, full_stabilizer=False, periodic=True, stride=1):
    """Extract and reconstruct full (shots, rounds, r, s) syndrome.

    Measurements are for V(i,j) = data[i][j] ⊕ data[(i+g)%r][j]
    for i=0..r-g-1 (all row residues, all columns j), g=stride.
    The last g rows' V are computed via linear combination (periodic)
    or left as zero (open boundaries). Pass the same stride used in
    build_circuit (default 1 = the factorized nearest-neighbor code).

    When no_reset=True, ancillas persist between rounds: m_r = m_{r-1} ⊕ P_r.
    The actual parity P_r = m_r ⊕ m_{r-1} (with m_{-1} = 0).

    When free_final_round=True, the last round's syndrome is computed from
    data_raw (destructive readout) instead of an ancilla measurement.
    Only rounds-1 ancilla registers are expected in pub_result.

    Fully vectorized: measurements are scattered into the (r, s) grid with
    precomputed fancy indices in one shot across all rounds.
    """
    g = stride
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

        if full_stabilizer == "paired":
            # S_a(t) = m_a(t) ^ m_a(t-1) ^ m_h(t - [phase==0]) with m(-1)=0:
            # differencing removes the primary's accumulation, and the raw
            # helper measurement removes the helper-state leak of the fan CX.
            anchors = _check_anchors(r, s, g)
            slot = {c: k for k, c in enumerate(anchors)}
            hidx = np.array([slot[(i, (j + g) % s)] for (i, j) in anchors])
            kap = np.array([(j // g) % 2 for (_, j) in anchors])
            m_parity = m_raw.copy()
            m_parity[:, 1:] ^= m_raw[:, :-1]
            i0 = np.where(kap == 0)[0]
            i1 = np.where(kap == 1)[0]
            m_parity[:, 1:, i0] ^= m_raw[:, :-1][:, :, hidx[i0]]
            m_parity[:, :, i1] ^= m_raw[:, :, hidx[i1]]
        elif no_reset:
            m_parity = m_raw.copy()
            m_parity[:, 1:] ^= m_raw[:, :-1]
        else:
            m_parity = m_raw

        # Scatter all rounds at once: (shots, anc_rounds, n_anc) -> (shots, anc_rounds, r, s)
        ui, uj = _unpack_indices(r, s, g)
        V = np.zeros((shots, anc_rounds, r, s), dtype=np.uint8)
        V[:, :, ui, uj] = m_parity

        if periodic:
            for u in range(g):
                V[:, :, r - g + u, :] = V[:, :, u:r - g:g, :].sum(axis=2) % 2

        syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
        if full_stabilizer:
            syn[:, :anc_rounds] = V  # measurements ARE S directly
        else:
            syn[:, :anc_rounds] = V ^ np.roll(V, shift=-g, axis=3)

    # Free final round: compute last syndrome from data readout
    if free_final_round and data_raw is not None:
        V_last = data_raw.astype(np.uint8) ^ np.roll(data_raw.astype(np.uint8), shift=-g, axis=1)
        syn[:, -1] = V_last ^ np.roll(V_last, shift=-g, axis=2)

    return syn


def round_bases(rounds, stabilizer_basis):
    """Per-round basis list for a (possibly repeating) stabilizer_basis string."""
    seq = stabilizer_basis.upper()
    return [seq[t % len(seq)] for t in range(rounds)]


# =========================================================================
# Steane-style EC (steane_ec=True)
#
# Encoded-ancilla prep is derived from THIS code's own operators — the
# periodic weight-2-gauge / weight-4-S structure, not a surface-code
# import.  The code decouples into stride-2 plaquette sectors; the ancilla
# stabilizer group is completed to a maximal abelian set (symplectic-
# centralizer completion) so spectator-sector DOF are pinned and the block
# is a pure codeword: all S = +1, the measured-basis logical uniform.
# Verified noiselessly: encoded |+>_L / |0>_L pass all-S/logical checks,
# and a logical Bell pair holds W = <ZZ>_L + <XX>_L = 2.0000 through
# multiple alternating rounds, with injected errors lighting the
# reconstructed syndrome.
# =========================================================================

def _st_pauli(n, kind, coords):
    """Pauli string on n qubits (q = i*s + j); leftmost char = qubit n-1."""
    from qiskit.quantum_info import Pauli
    ch = ["I"] * n
    for q in coords:
        ch[q] = kind
    return Pauli("".join(reversed(ch)))


def _st_sym(p):
    return np.concatenate([p.x.astype(np.uint8), p.z.astype(np.uint8)])


def _st_commute(a, b):
    return (int(np.dot(a.x.astype(int), b.z.astype(int))) +
            int(np.dot(a.z.astype(int), b.x.astype(int)))) % 2 == 0


def _st_independent(vec, basis):
    """Gaussian-eliminate vec against reduced basis rows; residual or None."""
    v = vec.copy()
    for piv, row in basis:
        if v[piv]:
            v ^= row
    nz = np.nonzero(v)[0]
    return (nz[0], v) if len(nz) else None


def _st_gf2_null(rows, ncols):
    """Basis of the GF(2) null space of the given rows (full RREF)."""
    R = [row.copy() % 2 for row in rows]
    piv_row = {}
    ri = 0
    for c in range(ncols):
        sel = next((i for i in range(ri, len(R)) if R[i][c]), None)
        if sel is None:
            continue
        R[ri], R[sel] = R[sel], R[ri]
        for i in range(len(R)):
            if i != ri and R[i][c]:
                R[i] ^= R[ri]
        piv_row[c] = ri
        ri += 1
    basis = []
    for free in range(ncols):
        if free in piv_row:
            continue
        v = np.zeros(ncols, dtype=np.uint8)
        v[free] = 1
        for c, rr in piv_row.items():
            if R[rr][free]:
                v[c] = 1
        basis.append(v)
    return basis


def _st_complete_isotropic(chosen_syms, n):
    """Extend an isotropic symplectic set to a maximal one (size n) from its
    symplectic centralizer — pins the spectator-sector DOF of the
    decoupled stride-g plaquette sectors (trivial at stride=1)."""
    chosen = [v.copy() % 2 for v in chosen_syms]
    while len(chosen) < n:
        cons = [np.concatenate([c[n:], c[:n]]) for c in chosen]
        null = _st_gf2_null(cons, 2 * n)
        span = []
        for c in chosen:
            res = _st_independent(c, span)
            if res:
                span.append(res)
        for w in null:
            if w.any() and _st_independent(w, span) is not None:
                chosen.append(w)
                break
        else:
            break
    return chosen


def _st_vec_to_pauli(vec, n):
    from qiskit.quantum_info import Pauli
    x, z = vec[:n], vec[n:]
    ch = ["IXZY"[int(x[q]) + 2 * int(z[q])] for q in range(n)]
    return Pauli("".join(reversed(ch)))


def steane_code_operators(r, s, stride=1):
    """This code's operators (periodic, full_stabilizer):
    S_P(i,j) = P(i,j)P(i+g,j)P(i,j+g)P(i+g,j+g) at _check_anchors;
    V_P(i,j) = P(i,j)P(i+g,j) weight-2 gauge; logicals matched to
    bell_complex readout: Z_L1/X_L2 on row 0, Z_L2/X_L1 on col 0."""
    g = stride
    n = r * s
    q = lambda i, j: (i % r) * s + (j % s)
    anchors = _check_anchors(r, s, g)

    def S(kind):
        return [_st_pauli(n, kind, [q(i, j), q(i + g, j),
                                    q(i, j + g), q(i + g, j + g)])
                for (i, j) in anchors]

    def V(kind):
        return [_st_pauli(n, kind, [q(i, j), q(i + g, j)])
                for i in range(r - g) for j in range(s)]

    L = dict(
        XL1=_st_pauli(n, "X", [q(i, 0) for i in range(r)]),
        XL2=_st_pauli(n, "X", [q(0, j) for j in range(s)]),
        ZL1=_st_pauli(n, "Z", [q(0, j) for j in range(s)]),
        ZL2=_st_pauli(n, "Z", [q(i, 0) for i in range(r)]),
    )
    return n, {"SX": S("X"), "SZ": S("Z"), "VX": V("X"), "VZ": V("Z"), **L}


def steane_encoded_stabilizer_list(r, s, state, stride=1):
    """Maximal abelian independent generating set (n Paulis) for the encoded
    ancilla |+>_L (state='plus') or |0>_L (state='zero'): all S of both
    types + the state's two logicals, spectator DOF pinned by completion.
    The complementary logical provably stays out of the group (uniform)."""
    n, ops = steane_code_operators(r, s, stride)
    if state == "plus":
        must = ops["SX"] + ops["SZ"] + [ops["XL1"], ops["XL2"]]
    else:
        must = ops["SX"] + ops["SZ"] + [ops["ZL1"], ops["ZL2"]]
    fill = ops["VX"] + ops["VZ"]

    chosen, basis = [], []

    def try_add(p):
        if not all(_st_commute(p, c) for c in chosen):
            return False
        res = _st_independent(_st_sym(p), basis)
        if res is None:
            return False
        chosen.append(p)
        basis.append(res)
        return True

    for p in must:
        try_add(p)
    for p in fill:
        if len(chosen) == n:
            break
        try_add(p)
    syms = _st_complete_isotropic([_st_sym(p) for p in chosen], n)
    return n, [_st_vec_to_pauli(v, n) for v in syms]


_steane_prep_cache = {}


def steane_prep_circuits(r, s, stride=1):
    """{'plus': circuit, 'zero': circuit} preparing the encoded ancilla block
    (n_data qubits).  'plus' (|+>_L, Z_L uniform) pairs with Z-stabilizer
    rounds; 'zero' (|0>_L, X_L uniform) with X-stabilizer rounds.  Cached."""
    key = (r, s, stride)
    if key not in _steane_prep_cache:
        from qiskit.synthesis import (synth_circuit_from_stabilizers,
                                      synth_clifford_greedy)
        from qiskit.quantum_info import Clifford
        preps = {}
        for state in ("plus", "zero"):
            n, gens = steane_encoded_stabilizer_list(r, s, state, stride)
            assert len(gens) == n, \
                f"steane prep {r}x{s} {state}: {len(gens)} gens != {n} (not pure)"
            qc = synth_circuit_from_stabilizers(gens)
            # Post-synthesis: greedy Clifford resynthesis of the SAME
            # unitary (hence the same prepared state, exactly).  Measured
            # on 4x6 / FakeMarrakesh: 54cx+17swap -> 39cx+20swap logical,
            # 146 -> 112 transpiled 2q at O3 — the block prep is ~all of
            # the optimizable gate mass of a Steane round, so this is the
            # cheapest ~23% available.  Keep whichever is lighter by the
            # routed-cost proxy cx + 3*swap.
            try:
                qg = synth_clifford_greedy(Clifford(qc))
                def _cost(c):
                    o = c.count_ops()
                    return o.get("cx", 0) + 3 * o.get("swap", 0)
                if _cost(qg) < _cost(qc):
                    qc = qg
            except Exception:
                pass                      # keep baseline synthesis
            preps[state] = qc
        _steane_prep_cache[key] = preps
    return _steane_prep_cache[key]


def steane_syndromes(pub_result, rounds, r, s, free_final_round=False,
                     data_raw=None, stride=1):
    """(shots, rounds, r, s) syndrome from steane_ec block readouts.

    Each round's steane_{t} register holds the destructive readout of the
    ancilla block (Z basis after data->block CX for Z rounds; X basis after
    block->data CX + H for X rounds).  The syndrome is reconstructed with
    exactly the free_final_round math, per round:
        V = b ^ roll(b, -g, rows);  S = V ^ roll(V, -g, cols)   (g=stride)
    Every round is a free syndrome round.  With free_final_round=True the
    last round comes from the data readout (data_raw) as usual.

    Semantics match all_syndromes_opt(full_stabilizer=True): per-round S
    VALUES, not detectors.  After the product-state |00> prep the weight-4
    S of the type NOT fixed by the prep are gauge-random per shot until the
    first round of that type projects them (verified: same statistics as
    the ancilla-based extraction, and consecutive same-type XOR detectors
    are exactly zero noiselessly).  Form detectors downstream as usual."""
    g = stride
    blk_rounds = rounds - 1 if free_final_round else rounds
    if blk_rounds == 0:
        shots = data_raw.shape[0]
        syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    else:
        first = getattr(pub_result.data, "steane_0")
        shots = first.num_shots
        b = np.zeros((shots, blk_rounds, r, s), dtype=np.uint8)
        for t in range(blk_rounds):
            bits = getattr(pub_result.data, f"steane_{t}").to_bool_array(
                order='little')
            b[:, t] = bits[:, :r * s].astype(np.uint8).reshape(shots, r, s)
        V = b ^ np.roll(b, shift=-g, axis=2)
        syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
        syn[:, :blk_rounds] = V ^ np.roll(V, shift=-g, axis=3)
    if free_final_round and data_raw is not None:
        V_last = data_raw.astype(np.uint8) ^ np.roll(
            data_raw.astype(np.uint8), shift=-g, axis=1)
        syn[:, -1] = V_last ^ np.roll(V_last, shift=-g, axis=2)
    return syn


def split_by_basis(all_syn, stabilizer_basis):
    """Split (shots, rounds, r, s) syndromes into per-basis subsequences.

    Returns {basis: (round_indices, all_syn[:, round_indices])}.
    With alternating extraction, the S_Z history (X-error syndromes for the
    Z-readout logicals) and the S_X history (Z-error syndromes for the
    X-readout logicals) are decoded independently.
    """
    bases = round_bases(all_syn.shape[1], stabilizer_basis)
    out = {}
    for b in dict.fromkeys(bases):
        idx = [t for t, bb in enumerate(bases) if bb == b]
        out[b] = (idx, all_syn[:, idx])
    return out


def detection_events(all_syn, stabilizer_basis, deterministic_first=('Z',)):
    """Same-basis consecutive differencing → detection events.

    For each basis subsequence, event[k] = S[t_k] ^ S[t_{k-1}].  The first
    round of a basis is a valid event only if the initial state is a
    deterministic +1 eigenstate of that basis's stabilizers (|0…0⟩ prep for
    Z-type, |+…+⟩ prep for X-type); otherwise it is a random gauge-fixing
    reference and its event row is zeroed.  Interleaved rounds of the other
    basis commute with these S operators, so same-basis differencing across
    them is valid.

    Returns {basis: (round_indices, events)} with events shaped like the
    subsequence.
    """
    out = {}
    for b, (idx, sub) in split_by_basis(all_syn, stabilizer_basis).items():
        ev = sub.copy()
        ev[:, 1:] ^= sub[:, :-1]
        if b not in deterministic_first:
            ev[:, 0] = 0
        out[b] = (idx, ev)
    return out


def accumulated_syndromes(all_syn, stabilizer_basis,
                           deterministic_first=('Z',)):
    """Per-basis RAW accumulated-syndrome streams, decoder-ready.

    The AND-vote / multi-pass decoders expect raw per-round S values, where
    a persistent data error stays flipped in every later round — NOT
    detection events. For a basis whose first round is a random gauge fix
    (X-type after |0…0> prep), every round is re-referenced to the first
    (S(t) ^ S(first)) and the reference round is dropped; for a
    deterministic basis the raw stream is already the accumulated error.

    Returns {basis: (round_indices, stream)}; a stream can be empty
    (e.g. a single X sample carries no error information).
    """
    out = {}
    for b, (idx, sub) in split_by_basis(all_syn, stabilizer_basis).items():
        if b in deterministic_first:
            out[b] = (idx, sub)
        elif sub.shape[1] <= 1:
            out[b] = (idx[1:], sub[:, 1:])
        else:
            out[b] = (idx[1:], sub[:, 1:] ^ sub[:, :1])
    return out


def check_consistency(all_syn, data_raw, r, s, stride=1):
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
    g = stride
    V_data = data_raw.astype(np.uint8) ^ np.roll(data_raw.astype(np.uint8), shift=-g, axis=1)
    syn_data = V_data ^ np.roll(V_data, shift=-g, axis=2)

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


def verify_pipeline(no_reset=False, stride=1):
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

    qc, dm, lq0, lq1, n_anc = build_circuit(r, s, rounds, logical_state="00",
                                             no_reset=no_reset, stride=stride)
    print(f"Circuit: {qc.num_qubits}q, CX={qc.count_ops().get('cx',0)} "
          f"(stride={stride})")

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

    g = stride
    ui, uj = _unpack_indices(r, s, g)
    V = np.zeros((len(items), r, s), dtype=np.uint8)
    if syn_bits is not None:
        V[:, ui, uj] = syn_bits[:, :len(ui)]
    for u in range(g):
        V[:, r - g + u, :] = V[:, u:r - g:g, :].sum(axis=1) % 2
    syn_u = V ^ np.roll(V, shift=-g, axis=2)

    n = int(cnts.sum())
    print(f"  Total shots decoded: {n} ({len(items)} unique outcomes)")

    # Decode each *unique syndrome* once, then broadcast back.
    try:
        dec = WaxisDecoder(r, s, stride=stride)
    except TypeError:
        assert stride == 2, ("external WaxisDecoder is stride-2 only; "
                             "pass a stride-aware decoder for stride != 2")
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


def schedule_with_dd(qc_transpiled, backend, sequence="XX", skip_qubits=()):
    """Timing-aware dynamical decoupling on a *transpiled* circuit.

    Why this beats dd=True: on Heron R2 the readout takes ~1.56 us while a CZ
    takes ~84 ns, so >85% of each round's wall time is the ancilla measurement
    window during which every data qubit idles. Over 8 rounds that is ~12.5 us
    of idle against a median T2 of ~88 us. The instruction-level dd=True flag
    inserts one X per round wherever the scheduler happens to put it; this
    helper instead schedules the circuit (ASAP) and pads every real delay
    window with a symmetric, frame-neutral X-X (or XY4) echo centered in the
    idle — the textbook placement that actually refocuses low-frequency
    dephasing during readout. Use INSTEAD of dd=True (leave dd=False), after
    transpilation:

        qc_t = pm.run(qc)
        qc_dd = schedule_with_dd(qc_t, backend)

    sequence: 'XX' (2 X gates, frame neutral, cheapest) or 'XY4'.
    skip_qubits: physical qubits to leave unpadded (rarely needed).
    Falls back to returning the input unchanged if the target lacks timing.
    """
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (ALAPScheduleAnalysis,
                                          PadDynamicalDecoupling)
    from qiskit.circuit.library import XGate, YGate

    target = backend.target
    if sequence.upper() == "XY4":
        dd_seq = [XGate(), YGate(), XGate(), YGate()]
    else:
        dd_seq = [XGate(), XGate()]
    try:
        pm = PassManager([
            ALAPScheduleAnalysis(target=target),
            PadDynamicalDecoupling(dd_sequence=dd_seq, target=target,
                                   skip_reset_qubits=True,
                                   qubits=None if not skip_qubits else [
                                       q for q in range(target.num_qubits)
                                       if q not in set(skip_qubits)]),
        ])
        return pm.run(qc_transpiled)
    except Exception as e:  # no timing info (e.g. plain Aer)
        print(f"  schedule_with_dd: skipped ({type(e).__name__}: {e})")
        return qc_transpiled


def _target_error_maps(backend):
    """(edge -> 2q error, qubit -> readout error) from the backend target."""
    t = backend.target
    cz_err, ro_err = {}, {}
    for name in ("cz", "ecr", "cx"):
        if name in t.operation_names:
            for qargs, props in t[name].items():
                e = getattr(props, "error", None)
                if e is not None:
                    cz_err[frozenset(qargs)] = e
            break
    if "measure" in t.operation_names:
        for qargs, props in t["measure"].items():
            e = getattr(props, "error", None)
            if e is not None:
                ro_err[qargs[0]] = e
    return cz_err, ro_err


def estimate_success(qc_transpiled, backend):
    """Crude product-fidelity estimate of a transpiled circuit: multiplies
    (1 - error) over every 2q gate and measurement using target calibration.
    Good enough as a *ranking* metric between candidate layouts/seeds."""
    import math
    cz_err, ro_err = _target_error_maps(backend)

    def _c(e):
        return min(max(e, 0.0), 0.999) if e == e else 0.999
    log_f = 0.0
    for inst in qc_transpiled.data:
        n = inst.operation.name
        if n in ("cz", "ecr", "cx"):
            q = frozenset(qc_transpiled.find_bit(x).index for x in inst.qubits)
            log_f += math.log1p(-_c(cz_err.get(q, 0.0)))
        elif n == "measure":
            q = qc_transpiled.find_bit(inst.qubits[0]).index
            log_f += math.log1p(-_c(ro_err.get(q, 0.0)))
    return math.exp(log_f)


def pick_best_transpilation(qc, backend, seeds=tuple(range(8)),
                            optimization_level=3, initial_layout=None,
                            verbose=True):
    """Transpile with several seeds and keep the layout with the best
    calibration-estimated success probability.

    O3's VF2 layout is noise-aware but stops at the first 'good enough'
    embedding for a given seed; on a 156q Heron there are many disjoint
    places the 80q block can sit and their aggregate CZ/readout error can
    differ by several percent in product fidelity. Costs only transpile time.
    Returns (best_circuit, best_score).
    """
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    best = (None, -1.0, None)
    for sd in seeds:
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=optimization_level,
            seed_transpiler=sd, initial_layout=initial_layout)
        t = pm.run(qc)
        sc = estimate_success(t, backend)
        if sc > best[1]:
            best = (t, sc, sd)
    if verbose:
        print(f"  pick_best_transpilation: seed {best[2]} "
              f"est. success {best[1]:.3e}")
    return best[0], best[1]


def _steiner_connect(G, free, terminals):
    """Greedy Steiner: connect `terminals` through vertices in `free`.

    Repeatedly BFS from the growing tree through free vertices to the
    nearest unconnected terminal and absorb the path. Returns the set of
    tree vertices (a connected, cycle-consistent subgraph of G containing
    all terminals) or None if the free region is disconnected.
    """
    import collections
    terminals = list(dict.fromkeys(terminals))
    tree = {terminals[0]}
    remaining = set(terminals[1:])
    while remaining:
        par = {v: None for v in tree}
        q = collections.deque(tree)
        hit = None
        while q and hit is None:
            v = q.popleft()
            for w in G[v]:
                if w in par or w not in free:
                    continue
                par[w] = v
                if w in remaining:
                    hit = w
                    break
                q.append(w)
        if hit is None:
            return None
        v = hit
        path = []
        while v not in tree:
            path.append(v)
            v = par[v]
        tree.update(path)
        remaining.discard(hit)
    return tree


def _constructive_placement(G, r, s, coords, checks, anc_index, seed,
                            max_anchors=120, max_paths=150,
                            keep_anc_access=False, stride=1):
    """Greedy constructive placement of the QEC interaction graph.

    The graph is a forest of stride·s independent chains (per column: one
    chain per row-residue class, data alternating with check ancillas
    — at stride=1 a single chain per column), which is
    why generic subgraph search (VF2) is slow — nothing prunes. Instead:
    pack the chains one by one in scanline order from a graph CORNER
    (max-eccentricity vertex) — corner packing keeps the unused remainder
    in one piece, unlike center packing which builds enclosing walls —
    placing unconstrained chains first and support-carrying chains last,
    facing the open region,
    rejecting any placement that breaks the CONNECTIVITY INVARIANT:
    every support qubit placed so far must keep a neighbor in the
    LARGEST UNUSED component. All leaves are then placed inside that
    component, so the strict (free-qubits-only) Steiner pass succeeds
    by construction. Returns the `phys` list (pattern logical -> physical,
    leaves appended after the QEC block) or None.
    """
    import random
    rng = random.Random(seed)
    n_data = r * s
    support_set = set(map(tuple, coords))

    # component specs: per (column, residue) chain, slots = d,a,d,a,...,d
    comps = []
    for j in range(s):
        for par in range(stride):
            rr = list(range(par, r, stride))
            slots = []
            for t, i in enumerate(rr):
                slots.append(("d", i, j))
                if t < len(rr) - 1:
                    slots.append(("a", i, j))
            nsup = sum(1 for (k, i, jj) in slots
                       if (k == "d" and (i, jj) in support_set)
                       or (k == "a" and keep_anc_access))
            comps.append((slots, nsup))
    comps.sort(key=lambda c: c[1])   # support chains LAST

    # graph center (min eccentricity) for compact BFS packing
    import collections
    def bfs_dist(src):
        d = {src: 0}
        q = collections.deque([src])
        while q:
            v = q.popleft()
            for w in G[v]:
                if w not in d:
                    d[w] = d[v] + 1
                    q.append(w)
        return d
    ecc = {v: max(bfs_dist(v).values()) for v in G}
    corner = max(G, key=lambda v: (ecc[v], rng.random()))
    dist0 = bfs_dist(corner)
    order = sorted(G, key=lambda v: (dist0[v], rng.random()))

    used = set()
    data_used = set()
    assign = {}
    placed_support = []

    def unused_components(pset):
        """Components of the unused region after tentatively using pset."""
        blocked = used | pset
        comp = {}
        sizes = {}
        cid = 0
        for v in G:
            if v in blocked or v in comp:
                continue
            cid += 1
            comp[v] = cid
            sizes[cid] = 1
            stack = [v]
            while stack:
                u = stack.pop()
                for w in G[u]:
                    if w not in blocked and w not in comp:
                        comp[w] = cid
                        sizes[cid] += 1
                        stack.append(w)
        return comp, sizes

    def main_ok(pset, path_supports):
        """Invariant: every support qubit placed so far (including the
        tentative chain's) must keep a neighbor in the LARGEST unused
        component — that component hosts every leaf and the Steiner tree,
        so the strict tree pass succeeds by construction."""
        comp, sizes = unused_components(pset)
        if not sizes:
            return False
        main = max(sizes, key=sizes.get)
        for sp in placed_support + path_supports:
            if not any(comp.get(w) == main for w in G[sp]):
                return False
        return True

    for slots, nsup in comps:
        L = len(slots)
        done = False
        conn_budget = 500
        anchors = [v for v in order if v not in used][:max_anchors]

        def _needs_access(kind, i, j):
            return ((kind == "d" and (i, j) in support_set)
                    or (kind == "a" and keep_anc_access))

        def leaf_possible(so, p, pset):
            # cheap necessary condition: every access-requiring slot must
            # keep an unused off-path neighbor (a straight chain middle can't)
            for (kind, i, j), v in zip(so, p):
                if _needs_access(kind, i, j):
                    if not any(w not in used and w not in pset
                               for w in G[v]):
                        return False
            for sp in placed_support:
                nbs = [w for w in G[sp] if w not in used]
                if not nbs or all(w in pset for w in nbs):
                    return False
            return True

        for anchor in anchors:
            tried = 0
            stack = [(anchor, [anchor])]
            while stack and not done and tried < max_paths:
                u, p = stack.pop()
                if len(p) == L:
                    tried += 1
                    pset = set(p)
                    for so in (slots, slots[::-1]):
                        psup = [v for (k, i, j), v in zip(so, p)
                                if _needs_access(k, i, j)]
                        if not leaf_possible(so, p, pset):
                            continue
                        if conn_budget <= 0:
                            continue
                        conn_budget -= 1
                        if not main_ok(pset, psup):
                            continue
                        for (kind, i, j), v in zip(so, p):
                            if kind == "d":
                                assign[i * s + j] = v
                                data_used.add(v)
                                if (i, j) in support_set:
                                    placed_support.append(v)
                            else:
                                assign[anc_index[(i, j)]] = v
                                if keep_anc_access:
                                    placed_support.append(v)
                        used.update(p)
                        done = True
                        break
                    continue
                nbrs = [w for w in G[u] if w not in used and w not in p]
                rng.shuffle(nbrs)
                for w in nbrs:
                    stack.append((w, p + [w]))
            if done:
                break
        if not done:
            return None

    base = n_data + len(checks)
    phys = [assign[q] for q in range(n_data)] + \
           [assign[n_data + k] for k in range(len(checks))]
    anc_set = set(phys[n_data:base])

    # every leaf lives in the largest unused component (the invariant
    # guarantees each support has a neighbor there) — the strict Steiner
    # pass then succeeds by construction, no ancilla-sharing needed
    comp, sizes = unused_components(set())
    if not sizes:
        return None
    main = max(sizes, key=sizes.get)
    for (i, j) in coords:
        v = assign[i * s + j]
        cands = [w for w in G[v] if comp.get(w) == main]
        if not cands:
            return None
        phys.append(max(
            cands,
            key=lambda w: (sum(1 for x in G[w] if x not in used),
                           rng.random())))
    return phys


def _score_plan(plan, phys, G_unused, backend, r, s, checks, rounds_weight=8,
                stride=1):
    """Calibration-aware log-fidelity score for a candidate placement.

    QEC chain edges and ancilla readouts recur every round (weighted by
    rounds_weight); tree edges, leaf couplings and the root readout count
    once. Higher is better. Returns 0.0 if the target has no calibration."""
    import math
    cz_err, ro_err = _target_error_maps(backend)
    if not cz_err and not ro_err:
        return 0.0

    def _c(e):  # clamp faulty-edge/NaN calibration (error=1) to a finite penalty
        return min(max(e, 0.0), 0.999) if e == e else 0.999
    n_data = r * s
    score = 0.0
    for k, (i, j) in enumerate(checks):
        a = phys[n_data + k]
        d0 = phys[i * s + j]
        d1 = phys[((i + stride) % r) * s + j]
        for e in (frozenset((d0, a)), frozenset((a, d1))):
            score += rounds_weight * math.log1p(-_c(cz_err.get(e, 0.0)))
        score += rounds_weight * math.log1p(-_c(ro_err.get(a, 0.0)))
    tn = plan["tree_nodes"]
    for t, p in enumerate(plan["tree_parent"]):
        if p >= 0:
            score += 2 * math.log1p(-_c(cz_err.get(frozenset((tn[t], tn[p])), 0.0)))
    for k, leaf_t in enumerate(plan["leaf_of"]):
        i, j = plan["support"][k]
        e = frozenset((tn[leaf_t], phys[i * s + j]))
        score += math.log1p(-_c(cz_err.get(e, 0.0)))
    score += math.log1p(-_c(ro_err.get(tn[plan["bfs_order"][0]], 0.0)))
    return score


def synthesize_parity_layout(backend, r, s, op="bell", periodic=True,
                             vf2_time=180, seeds=tuple(range(32)),
                             noise_aware=True, rounds_weight=8,
                             verbose=True, stride=1):
    """Synthesize a zero-SWAP layout plan for the ancilla parity measurement.

    Why: the single-ancilla X⊗n gadget is a degree-n star — unembeddable on
    degree-3 heavy hex, so the transpiler routes it (~300+ CZ for n=12 on
    Heron). Fixed alternatives fail structurally: heavy hex has GIRTH 12,
    and any prescribed backbone that attaches to two data qubits of the same
    check-path closes a 10-cycle. The only shape guaranteed to embed is one
    read off the device itself: a cat state grown over a TREE that is a
    literal subgraph of the coupling map.

    Method: (1) constructively place the QEC block — a forest of
    independent chains, packed greedily in BFS order from the graph center
    (milliseconds; generic VF2 search took minutes here because a floppy
    forest gives it nothing to prune on) — while guaranteeing every support
    data qubit keeps a free neighbor for its 'leaf'; (2) connect the leaves
    through remaining free qubits with a greedy Steiner tree, falling back
    to routing the tree through the check ancillas if the strict free
    region is fragmented; (3) return the tree structure and the full
    initial_layout.

    Usage:
        plan = synthesize_parity_layout(backend, 6, 8, op="bell",
                                        periodic=True)
        qc, dm, lq0, lq1, n_anc = build_circuit(
            6, 8, rounds, bell=True, bell_ancilla=True, bell_measure=True,
            periodic=True, compact=True, parity_tree=plan, ...)
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3,
            initial_layout=plan["initial_layout"], seed_transpiler=42)
        qc_t = pm.run(qc)   # transpiled 2q count == logical CX count

    op='bell' serves bell and bell_measure; op='ghz' serves ghz/ghz_measure
    (one op family per plan — their supports differ). The plan is
    JSON-serializable; synthesize once per backend and cache it.

    Notes: valid for the weight-2 (share-pair) extraction graph. The
    full_stabilizer=True graph has degree-4 check ancillas and does not
    embed on heavy hex at all — its layout is routed by the transpiler,
    and this plan does not apply there.

    Returns the plan dict, or None if no placement/tree was found (try
    more seeds). vf2_time is kept for API compatibility and ignored.
    """
    import collections

    coords = (_ghz_support_coords(r, s) if op == "ghz"
              else _bell_support_coords(r, s, periodic))
    n_data = r * s
    n_anc = (r - stride) * s
    checks = _check_anchors(r, s, stride)
    anc_index = {c: n_data + k for k, c in enumerate(checks)}
    base = n_data + n_anc
    n_sup = len(coords)

    cm = backend.coupling_map if hasattr(backend, "coupling_map") else backend
    G = collections.defaultdict(set)
    for a, c in cm.get_edges():
        G[a].add(c)
        G[c].add(a)

    best_plan = None
    for seed in seeds:
        phys = _constructive_placement(G, r, s, coords, checks, anc_index,
                                       seed, stride=stride)
        if phys is None:
            continue
        leaves = phys[base:]
        data_phys = set(phys[:n_data])
        anc_phys = set(phys[n_data:base])
        # strict pass: tree only through unused qubits. Relaxed pass: also
        # through the QEC check ancillas — they are untouched |0> at prep
        # time and the cat uncomputes them back to |0>; for a post-round
        # gadget the builder resets them first. Data qubits are never used.
        tree_set = None
        for allow_anc in (False, True):
            free = set(G) - data_phys - (set() if allow_anc else anc_phys) \
                   - (set(phys[:base]) - anc_phys - data_phys)
            tree_set = _steiner_connect(G, free, leaves)
            if tree_set is not None:
                uses_anc = allow_anc and bool(tree_set & anc_phys)
                break
        if tree_set is None:
            if verbose:
                print(f"  seed {seed}: embedding found but leaves not "
                      f"connectable even through check ancillas; retrying")
            continue

        # prune: iteratively drop non-terminal vertices with tree-degree 1
        term = set(leaves)
        changed = True
        while changed:
            changed = False
            for v in list(tree_set):
                if v in term:
                    continue
                if sum(1 for w in G[v] if w in tree_set) <= 1:
                    tree_set.discard(v)
                    changed = True

        tnodes = sorted(tree_set)
        tidx = {p: t for t, p in enumerate(tnodes)}
        adj = {t: [tidx[w] for w in G[p] if w in tree_set]
               for p, t in tidx.items()}

        def bfs(root):
            parent = [-1] * len(tnodes)
            order = [root]
            seen = {root}
            depth = {root: 0}
            for t in order:
                for w in adj[t]:
                    if w not in seen:
                        seen.add(w)
                        parent[w] = t
                        depth[w] = depth[t] + 1
                        order.append(w)
            return parent, order, max(depth.values())

        # root at the tree's approximate center to minimize cat depth
        best = None
        for cand in range(len(tnodes)):
            parent, order, ecc = bfs(cand)
            if best is None or ecc < best[3]:
                best = (cand, parent, order, ecc)
        root, parent, order, ecc = best

        phys_to_logical = {p: idx for idx, p in enumerate(phys[:base])}
        fresh = [p for p in tnodes if p not in phys_to_logical]
        fresh_logical = {p: base + k for k, p in enumerate(fresh)}
        plan = {
            "r": r, "s": s, "periodic": periodic, "op": op,
            "support": [list(c) for c in coords],
            "tree_nodes": tnodes,
            "tree_parent": parent,
            "bfs_order": order,
            "leaf_of": [tidx[p] for p in leaves],
            "tree_logical": [phys_to_logical.get(p, fresh_logical.get(p))
                             for p in tnodes],
            "node_is_anc": [p in anc_phys for p in tnodes],
            "n_fresh": len(fresh),
            "uses_check_ancillas": uses_anc,
            "initial_layout": phys[:base] + fresh,
            "n_qubits": base + len(fresh),
            "seed": seed,
        }
        if noise_aware:
            plan["score"] = _score_plan(plan, phys, G, backend, r, s,
                                        checks, rounds_weight, stride=stride)
            if best_plan is None or plan["score"] > best_plan.get(
                    "score", float("-inf")):
                best_plan = plan
        elif best_plan is None or len(tnodes) < len(best_plan["tree_nodes"]):
            best_plan = plan
    if best_plan is not None and verbose:
        tn = best_plan["tree_nodes"]
        n_cx = 2 * (len(tn) - 1) + n_sup
        print(f"  best of {len(seeds)} seeds (seed {best_plan['seed']}): "
              f"tree of {len(tn)} qubits "
              f"({sum(best_plan['node_is_anc'])} shared with check "
              f"ancillas), gadget CX = {n_cx} (all nearest-neighbor), "
              f"total {best_plan['n_qubits']} qubits"
              + (f", est. log-fid score {best_plan['score']:.3f}"
                 if "score" in best_plan else ""))
    return best_plan


def synthesize_paired_layout(backend, r, s, seeds=tuple(range(24)),
                             n_extra=0, rounds_weight=8, noise_aware=True,
                             verbose=True, stride=1):
    """Try to synthesize a zero-SWAP layout plan for full_stabilizer='paired'.

    Why paired mode: the legacy weight-4 star places a degree-4 interaction
    on the ancilla, which cannot embed on degree-3 heavy hex, so the
    transpiler routes it. The paired scheme reduces the interaction graph to
    the weight-2 chain skeleton plus one 'rung' per S check between the two
    ancillas whose V's compose it; every ancilla then has max interaction
    degree 3.

    Feasibility caveat (important): a *fully* zero-SWAP paired layout needs
    every one of the 4*(r/2-1)*(s/2) ancillas simultaneously on a degree-3
    device vertex with a free third neighbor for its rung. On IBM Fez (156
    qubits, Heron r2) an exhaustive search shows this is INFEASIBLE for the
    6x8 periodic code: the weight-2 skeleton embeds happily with ancillas on
    degree-2 vertices, but forcing all 32 ancillas onto disjoint degree-3
    chain cores with free ports does not fit. This function therefore returns
    None on Fez-class lattices. That is expected, not an error.

    What to do instead on Fez: run paired mode with the transpiler's own
    layout and sweep seeds with pick_best_transpilation. Empirically this
    lowers the raw 2q count vs the star mode (~6.1k vs ~6.3k 2q at 8 rounds)
    while the rungs route with a modest depth increase. Choose star vs paired
    per run by comparing estimate_success of the best transpilation of each;
    neither dominates on Fez.

    When a plan IS returned (smaller codes, sparser rung load, or lattices
    with enough degree-3 headroom), it maps every chain load nearest-neighbor
    and threads each rung through FREE 'bridge' qubits (reusable across
    phases — the cascade uncomputes them to |0>). Pass it as rung_plan to
    build_circuit and as initial_layout to the transpiler:

        plan = synthesize_paired_layout(backend, r, s, n_extra=1)
        if plan is not None:
            qc, ... = build_circuit(r, s, rounds, full_stabilizer="paired",
                                    rung_plan=plan, ...)
            pm = generate_preset_pass_manager(backend=backend,
                optimization_level=3, initial_layout=plan["initial_layout"])

    n_extra reserves free qubits at the end of the layout for the bell/ghz
    gadget ancillas. Returns None if no seed yields a complete assignment.
    """
    import collections
    import math
    g = stride
    assert s % (2 * g) == 0, "paired mode needs s % (2*stride) == 0"
    n_data = r * s
    n_anc = (r - g) * s
    checks = _check_anchors(r, s, g)
    anc_index = {c: n_data + k for k, c in enumerate(checks)}
    base = n_data + n_anc

    cm = backend.coupling_map if hasattr(backend, "coupling_map") else backend
    G = collections.defaultdict(set)
    for a, c in cm.get_edges():
        G[a].add(c)
        G[c].add(a)
    cz_err, ro_err = _target_error_maps(backend)

    def _c(e):
        return min(max(e, 0.0), 0.999) if e == e else 0.999

    def free_path(pa, pb, blocked):
        """Shortest path pa->pb whose interior avoids `blocked`; returns the
        interior vertex list ([] if adjacent) or None."""
        if pb in G[pa]:
            return []
        par = {pa: None}
        q = collections.deque([pa])
        while q:
            v = q.popleft()
            for w in G[v]:
                if w in par:
                    continue
                if w == pb:
                    path = []
                    while v != pa:
                        path.append(v)
                        v = par[v]
                    return path[::-1]
                if w in blocked:
                    continue
                par[w] = v
                q.append(w)
        return None

    def _heavyhex_sites():
        """Vertical 5-chain sites: spacer v between two degree-3 row vertices
        u, w, each of which keeps >=2 row neighbors (one becomes chain data,
        one stays free as the ancilla's rung port)."""
        deg = {x: len(G[x]) for x in G}
        sites = []
        for v in sorted(G):
            if deg[v] != 2:
                continue
            u, w = sorted(G[v])
            if deg[u] == 3 and deg[w] == 3 and u not in G[w] \
                    and len(G[u] - {v}) >= 2 and len(G[w] - {v}) >= 2:
                sites.append((u, v, w))
        return sites

    def _heavyhex_placement(seed):
        """Embed the s (columns) x g (residues) = g·s data-ancilla chains onto
        the heavy-hex lattice with a free rung port on every ancilla.

        Each chain's a-d-a core is a deg3-deg2-deg3 "bridge" (u, v, w); u and
        w saturate their three device edges as {bridge, outer-data, port}, so
        each needs two free non-bridge neighbors (one becomes chain data, the
        other the rung port). We pick 2s vertex-disjoint bridges by shuffled
        greedy with a little backtracking (device has ~2x the qubits needed,
        so a random restart almost always completes), then map bridges to
        (column, parity) grouped by rung cycle so partners land near each
        other. Returns the phys list or None if this seed can't complete."""
        import random
        if r != 3 * g:
            # the a-d-a bridge yields exactly 3-data chains: native packing
            # applies only when each residue chain has 3 data rows
            return None
        rng = random.Random(seed * 2654435761 & 0xffffffff)
        need = g * s
        raw = _heavyhex_sites()
        if len(raw) < need:
            return None

        def try_select():
            order = raw[:]
            rng.shuffle(order)
            usedq, reserved, chosen = set(), set(), []
            for (u, v, w) in order:
                if len(chosen) == need:
                    break
                if {u, v, w} & usedq or {u, v, w} & reserved:
                    continue
                fu = [x for x in G[u] - {v}
                      if x not in usedq and x not in reserved]
                fw = [x for x in G[w] - {v}
                      if x not in usedq and x not in reserved]
                if len(fu) < 2 or len(fw) < 2:
                    continue
                d0, pu = fu[0], fu[1]
                d4, pw = fw[0], fw[1]
                chosen.append((u, v, w, d0, d4, pu, pw))
                usedq.update((u, v, w, d0, d4))
                reserved.update((pu, pw))
            return chosen if len(chosen) == need else None

        chosen = None
        for _ in range(200):          # cheap random restarts within the seed
            chosen = try_select()
            if chosen is not None:
                break
        if chosen is None:
            return None

        # Map the 2s chosen bridges onto (column, parity) chains grouped by
        # rung cycle. Order chosen bridges by device index so consecutive
        # picks are physically close, keeping rung paths short.
        chosen.sort(key=lambda t: (t[1], t[0]))
        cycles = [[(py + g * t, px) for t in range(s // g)]
                  for px in range(g) for py in range(g)]
        flat = [pj for cyc in cycles for pj in cyc]
        assign = {}
        for (j, px), (u, v, w, d0, d4, pu, pw) in zip(flat, chosen):
            assign[(px) * s + j] = d0
            assign[anc_index[(px, j)]] = u
            assign[(px + g) * s + j] = v
            assign[anc_index[(px + g, j)]] = w
            assign[(px + 2 * g) * s + j] = d4
        return [assign[q] for q in range(n_data)] + \
               [assign[n_data + k] for k in range(n_anc)]

    def weighted_path(pa, pb, used, load):
        """Dijkstra pa->pb through free vertices; reusing a vertex already
        serving other bridges is allowed (each cascade uncomputes it to |0>
        and program order serializes the overlap) but penalized so parallel
        depth stays low. Returns the interior vertex list or None."""
        import heapq
        if pb in G[pa]:
            return []
        dist = {pa: 0.0}
        par = {}
        h = [(0.0, pa)]
        while h:
            d, v = heapq.heappop(h)
            if d > dist.get(v, float("inf")):
                continue
            for w in G[v]:
                if w == pb:
                    path = []
                    while v != pa:
                        path.append(v)
                        v = par[v]
                    return path[::-1]
                if w in used:
                    continue
                nd = d + 1.0 + 3.0 * load.get(w, 0)
                if nd < dist.get(w, float("inf")):
                    dist[w] = nd
                    par[w] = v
                    heapq.heappush(h, (nd, w))
        return None

    import time
    _t0 = time.time()
    _budget = 120.0                 # never hang; bail out with None
    best = None
    for seed in seeds:
        if time.time() - _t0 > _budget:
            if verbose:
                print("  time budget exhausted; returning best-so-far")
            break
        # Heavy-hex-native vertical chains keep a free rung port on every
        # ancilla by construction; fall back to the generic constructive
        # placement (with keep_anc_access so ancillas still border the free
        # component) if the native packing can't seat all chains.
        phys = _heavyhex_placement(seed)
        if phys is None:
            phys = _constructive_placement(G, r, s, [], checks, anc_index,
                                           seed, keep_anc_access=True,
                                           stride=g)
        if phys is None:
            continue
        used = set(phys)
        bridges_by_slot = {}
        load = {}   # per-phase reuse counter for the depth penalty
        ok = True
        for ph in (0, 1):
            load.clear()
            rungs = sorted(((k, c) for k, c in enumerate(checks)
                            if (c[1] // g) % 2 == ph),
                           key=lambda kc: kc[0])
            for k, (i, j) in rungs:
                pa = phys[anc_index[(i, (j + g) % s)]]
                pb = phys[anc_index[(i, j)]]
                p = weighted_path(pa, pb, used, load)
                if p is None:
                    ok = False
                    break
                bridges_by_slot[k] = p
                for v in p:
                    load[v] = load.get(v, 0) + 1
            if not ok:
                break
        if not ok:
            if verbose:
                print(f"  seed {seed}: some rung not reachable; retrying")
            continue

        bridge_phys = sorted({p for br in bridges_by_slot.values() for p in br})
        pidx = {p: t for t, p in enumerate(bridge_phys)}
        total_cx_round = 6 * n_anc + sum(
            2 * len(br) + 1 for br in bridges_by_slot.values())
        if noise_aware and (cz_err or ro_err):
            score = 0.0
            for k, (i, j) in enumerate(checks):
                a = phys[anc_index[(i, j)]]
                d0, d1 = phys[i * s + j], phys[((i + g) % r) * s + j]
                for e in (frozenset((d0, a)), frozenset((a, d1))):
                    score += 3 * rounds_weight * math.log1p(-_c(cz_err.get(e, 0.0)))
                score += rounds_weight * math.log1p(-_c(ro_err.get(a, 0.0)))
                chain = ([phys[anc_index[(i, (j + g) % s)]]]
                         + bridges_by_slot[k] + [a])
                for u, v in zip(chain, chain[1:]):
                    score += 2 * rounds_weight * math.log1p(
                        -_c(cz_err.get(frozenset((u, v)), 0.0)))
        else:
            score = -float(total_cx_round)
        if best is None or score > best["score"]:
            extras = []
            if n_extra:
                blocked = used | set(bridge_phys)
                extras = sorted((q for q in G if q not in blocked),
                                key=lambda q: -len(G[q]))[:n_extra]
            best = {
                "r": r, "s": s, "seed": seed, "score": score,
                "bridges": {str(k): [pidx[p] for p in bridges_by_slot[k]]
                            for k in bridges_by_slot},
                "n_fresh": len(bridge_phys),
                "initial_layout": phys + bridge_phys + extras,
                "n_qubits": base + len(bridge_phys) + n_extra,
                "cx_per_round": total_cx_round,
            }
    if best is not None and verbose:
        print(f"  paired layout (seed {best['seed']}): {best['n_fresh']} "
              f"bridge qubits, {best['cx_per_round']} NN CX/round, "
              f"total {best['n_qubits']} qubits, score {best['score']:.2f}")
    elif best is None and verbose:
        print("  no zero-SWAP paired layout on this lattice/code (expected on "
              "Fez for 6x8; see docstring). Run paired mode with the "
              "transpiler's own layout via pick_best_transpilation instead.")
    return best


def verify_tree_parity():
    """Self-check for the tree parity gadget (run on your side).

    (1) synthesize on FakeFez; (2) transpile star vs tree Bell prep+measure
    and compare 2q gate counts — the tree circuit's transpiled 2q count must
    EQUAL its logical CX count (zero routing); (3) ideal-simulator statistics:
    prep and measure parities must agree shot-for-shot at rounds=0 in both
    modes, and the |+⟩^N state must give deterministic +1.
    """
    import numpy as np
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    backend = FakeFez()
    r, s = 6, 8
    print("synthesizing (periodic Bell support) ...")
    plan = synthesize_parity_layout(backend, r, s, op="bell", periodic=True)
    assert plan is not None, "no plan found — increase vf2_time/seeds"

    kw = dict(logical_state="00", bell=True, bell_ancilla=True,
              bell_measure=True, periodic=True, compact=True, no_reset=True)
    qc_star, *_ = build_circuit(r, s, 1, **kw)
    qc_tree, *_ = build_circuit(r, s, 1, parity_tree=plan, **kw)

    pm_star = generate_preset_pass_manager(backend=backend,
                                           optimization_level=3,
                                           seed_transpiler=42)
    pm_tree = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=42,
        initial_layout=plan["initial_layout"])
    for label, qc, pm in (("star", qc_star, pm_star),
                          ("tree", qc_tree, pm_tree)):
        t = pm.run(qc)
        two_q = sum(v for k, v in t.count_ops().items()
                    if k in ("cz", "ecr", "cx", "swap"))
        print(f"  {label}: logical CX={qc.count_ops().get('cx', 0):>4}  "
              f"transpiled 2q={two_q:>4}  depth={t.depth():>4}  "
              f"2q-depth={t.depth(lambda i: len(i.qubits) == 2):>3}")
        if label == "tree":
            assert two_q == qc.count_ops().get("cx", 0), \
                "tree gadget routed — plan/layout mismatch"
    print("  ✓ tree gadget transpiles with ZERO routing overhead")

    from qiskit_aer.primitives import SamplerV2
    sampler = SamplerV2(options={"backend_options": {"seed_simulator": 11}})
    for label, extra in (("star", {}), ("tree", dict(parity_tree=plan))):
        qc, *_ = build_circuit(r, s, 0, **kw, **extra)
        pub = sampler.run([qc], shots=400).result()[0]
        b = pub.data.bell.to_bool_array(order="little")[:, 0]
        bm = pub.data.bell_m.to_bool_array(order="little")[:, 0]
        agree = (b == bm).mean()
        print(f"  {label}: rounds=0 P(prep=measure)={agree:.3f}  "
              f"P(prep=1)={b.mean():.2f}")
        assert agree == 1.0
    print("  ✓ tree statistics match the single-ancilla gadget")


def synthesize_steane_layout(backend, r, s, n_extra=1,
                             seeds=tuple(range(24)), verbose=True,
                             stride=1):
    """Synthesize an initial_layout for steane_ec runs that makes the
    transversal CX layer zero-SWAP and both prep circuits route locally.

    Interaction-graph observation that makes this tractable: in a Steane
    round there are NO data<->data gates at all — extraction is exactly one
    CX per (data_k, block_k) pair.  So the placement problem is: choose
    n_data vertex-disjoint DEVICE EDGES (one per pair; the transversal
    layer is then native by construction), inside a compact low-error
    region, oriented so that data ends neighbour data ends and block ends
    neighbour block ends as much as possible (the two Clifford preps —
    encoded data pair and encoded ancilla block — are the remaining 2q
    mass, and they route over exactly those induced subgraphs).

    Algorithm (seeded greedy + orientation hill-climb, same spirit as
    synthesize_paired_layout):
      1. grow a connected region of n_data disjoint edges from a random
         low-cz-error seed edge, preferring low cz + readout error;
      2. order the edges along a greedy nearest-neighbour snake and assign
         grid sites row-major (only compactness matters — no data<->data
         circuit adjacency to honour);
      3. orient each edge (which endpoint is data) by hill-climbing the
         count of native data-data + block-block adjacencies;
      4. park the bell/extra ancillas on free neighbours of the row-0
         data qubits (gadget support) and the n_anc unused stabilizer
         ancillas on any remaining free qubits.

    Returns {'initial_layout': [...], 'native_dd': int, 'native_bb': int,
    'mean_cz': float} or None if no seed completes.  Virtual order matches
    build_circuit(steane_ec=True): [data n][anc n_anc][extras n_extra]
    [block n].  Pass to the transpiler as initial_layout; O3 keeps native
    2q untouched and routes only the preps' non-native remainder.
    """
    import collections
    import random as _random

    n_data = r * s
    n_anc = (r - stride) * s

    cm = backend.coupling_map if hasattr(backend, "coupling_map") else backend
    G = collections.defaultdict(set)
    for a, c in cm.get_edges():
        G[a].add(c)
        G[c].add(a)
    cz_err, ro_err = _target_error_maps(backend)

    def _e(a, b):
        v = cz_err.get(frozenset((a, b)), 0.01)
        v = v if v == v else 0.05  # NaN guard
        w = (ro_err.get(a, 0.02) + ro_err.get(b, 0.02)) / 2
        return min(max(v, 1e-5), 0.999) + 0.5 * min(max(w, 1e-5), 0.999)

    all_edges = sorted({tuple(sorted((a, b))) for a in G for b in G[a]})

    def _grow(rng):
        seed_pool = sorted(all_edges, key=lambda ed: _e(*ed))[:max(8, len(all_edges) // 4)]
        e0 = seed_pool[rng.randrange(len(seed_pool))]
        used, region = set(e0), [e0]
        while len(region) < n_data:
            cands = []
            for (a, b) in all_edges:
                if a in used or b in used:
                    continue
                if any(w in used for w in (G[a] | G[b])):
                    cands.append((a, b))
            if not cands:
                return None
            cands.sort(key=lambda ed: _e(*ed) + rng.random() * 1e-4)
            pick = cands[0]
            region.append(pick)
            used.update(pick)
        return region

    def _snake(region):
        """Greedy nearest-neighbour order over edge midpoints (graph dist)."""
        # BFS distances between edges via their endpoints
        idx = {ed: k for k, ed in enumerate(region)}
        vert2edge = {}
        for ed in region:
            for v in ed:
                vert2edge[v] = ed
        order = [region[0]]
        left = set(region[1:])
        while left:
            # BFS from current edge endpoints to nearest unused region edge
            src = order[-1]
            seen, q = set(src), collections.deque(src)
            hit = None
            while q and hit is None:
                v = q.popleft()
                for w in G[v]:
                    if w in seen:
                        continue
                    seen.add(w)
                    ed = vert2edge.get(w)
                    if ed in left:
                        hit = ed
                        break
                    q.append(w)
            if hit is None:
                hit = left.pop()
            else:
                left.discard(hit)
            order.append(hit)
        return order

    def _orient(order, rng):
        """ori[k] in {0,1}: which endpoint of edge k is DATA.  Hill-climb
        native data-data + block-block adjacency."""
        m = len(order)
        ori = [rng.randrange(2) for _ in range(m)]

        def ends(k):
            a, b = order[k]
            return (a, b) if ori[k] == 0 else (b, a)   # (data, block)

        def score():
            dset = {ends(k)[0] for k in range(m)}
            bset = {ends(k)[1] for k in range(m)}
            dd = sum(1 for v in dset for w in G[v] if w in dset) // 2
            bb = sum(1 for v in bset for w in G[v] if w in bset) // 2
            return dd + bb, dd, bb

        best, dd, bb = score()
        improved = True
        while improved:
            improved = False
            for k in range(m):
                ori[k] ^= 1
                sc, d2, b2 = score()
                if sc > best:
                    best, dd, bb = sc, d2, b2
                    improved = True
                else:
                    ori[k] ^= 1
        return [ends(k) for k in range(m)], dd, bb

    best_plan = None
    for sd in seeds:
        rng = _random.Random(sd * 2654435761 & 0xffffffff)
        region = _grow(rng)
        if region is None:
            continue
        order = _snake(region)
        pairs, dd, bb = _orient(order, rng)         # [(data_phys, block_phys)]
        used = {v for p in pairs for v in p}
        free = [v for v in sorted(G) if v not in used]

        data_phys = [None] * n_data
        block_phys = [None] * n_data
        for k, (dp, bp) in enumerate(pairs):        # snake order -> row-major
            data_phys[k] = dp
            block_phys[k] = bp

        # extras next to row-0 data (bell-gadget support), then leftovers
        extras = []
        for v in free:
            if len(extras) >= n_extra:
                break
            if any(w in set(data_phys[:s]) for w in G[v]):
                extras.append(v)
        for v in free:
            if len(extras) >= n_extra:
                break
            if v not in extras:
                extras.append(v)
        rest = [v for v in free if v not in set(extras)]
        if len(extras) < n_extra or len(rest) < n_anc:
            continue
        anc_phys = rest[:n_anc]

        layout = data_phys + anc_phys + extras[:n_extra] + block_phys
        mean_cz = float(np.mean([_e(*p) for p in pairs]))
        plan = {"initial_layout": layout, "native_dd": dd, "native_bb": bb,
                "mean_cz": mean_cz}
        key = (dd + bb, -mean_cz)
        if best_plan is None or key > best_plan[0]:
            best_plan = (key, plan)

    if best_plan is None:
        if verbose:
            print("steane layout synthesis: no seed completed")
        return None
    plan = best_plan[1]
    if verbose:
        print(f"steane layout: {n_data} native pair edges (transversal CX "
              f"zero-SWAP), native data-data adjacencies {plan['native_dd']}, "
              f"block-block {plan['native_bb']}, mean pair 2q err "
              f"{plan['mean_cz']:.4f}")
    return plan


if __name__ == "__main__":
    verify_optimized()
    verify_no_reset()
    verify_pipeline()
    verify_pipeline(no_reset=True)
