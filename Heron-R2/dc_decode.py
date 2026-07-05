"""Dynamic circuit tesseract decoder for the (1+x^2)(1+y^2) BB code.

Translates plane_warp.c's sector-algebraic decoder into Qiskit dynamic
circuit stages, using only:
  - Classical register expressions (XOR, AND, OR)
  - if_test conditional gates
  - CX gates for the O(n) 2D prefix-XOR when operating on qubit state

No subprocess calls, no calibration statistics, no fudge factors.

Two modes of operation:
  1. Classical-register mode (classical=True, default):
     Builds correction expressions from measured syndrome classical registers.
     Suitable for any grid size but the expression tree depth scales as O(n^2)
     for sector area n.  Practical for 6x6--20x20 grids.
  2. Qubit-mode (classical=False):
     Allocates r*s correction ancilla qubits, loads syndrome from classical
     registers via conditional X, computes the 2D prefix-XOR via CX gates
     (O(r*s) gates), measures and applies.  Requires ~3*r*s extra qubits
     but the circuit depth scales as O(r*s) independent of expression size.
     Suitable for large grids (80x80).
"""

import numpy as np
from qiskit.circuit.classical import expr


def _bit_expr(src, a, b, s):
    """Uniform bit access: src is either a ClassicalRegister (raw syndrome
    bits, indexed row-major) or a callable (a, b) -> classical expression."""
    if callable(src):
        return src(a, b)
    return expr.lift(src[_linear_index(a, b, s)])


def steane_bit_fn(block_creg, r, s):
    """Adapter for Steane-EC block registers: the steane_{t} register holds
    the RAW destructive block readout b, not the syndrome.  The syndrome is
        S(i,j) = b(i,j) ^ b(i+2,j) ^ b(i,j+2) ^ b(i+2,j+2)   (mod r, s)
    — the same V/S roll math the software path (steane_syndromes) applies.
    Returns a callable (a, b) -> expression computing S(a, b) from the raw
    register, for use as the `syn_creg` argument of the correction stages."""
    def fn(a, b):
        acc = None
        for da, db in ((0, 0), (2, 0), (0, 2), (2, 2)):
            bit = expr.lift(block_creg[_linear_index((a + da) % r,
                                                     (b + db) % s, s)])
            acc = bit if acc is None else expr.bit_xor(acc, bit)
        return acc
    return fn


def confirmed_bit_fn(block_cregs, r, s):
    """Temporal-differencing syndrome filter: gate/data errors persist
    across rounds, syndrome (readout / ancilla-prep) errors do not.

    Takes 2 or 3 Steane block registers (RAW block readouts of CONSECUTIVE
    same-basis rounds, oldest first) and returns a callable
    (a, b) -> classical expression for the CONFIRMED defect:

      2 registers:  C(a,b) = S_t(a,b) AND S_{t-1}(a,b)
                    — correct only defects present in both rounds.  A
                    measurement flip in round t-1 fires the defect at t-1
                    and again (as the difference) never persists into t's
                    absolute syndrome caused by real data error, so the
                    AND vetoes it.  One-sided: a flip in the LAST round
                    is also vetoed (conservative — the final destructive
                    data readout catches what this defers).
      3 registers:  C(a,b) = MAJ(S_{t-2}, S_{t-1}, S_t)(a,b)
                    — two-sided majority vote; tolerates one bad round
                    anywhere in the window.

    Cost: each S is an XOR-of-4 of raw block bits, so the confirmed-bit
    expression is ~2.3x (AND-pair) / ~7x (majority-of-3) the node count
    of a single-round bit.  Everything downstream
    (add_classical_correction via _rect_xor_expr) is unchanged: this is
    a drop-in replacement for steane_bit_fn as the `syn_creg` callable.
    """
    if len(block_cregs) not in (2, 3):
        raise ValueError("confirmed_bit_fn: need 2 or 3 block registers, "
                         f"got {len(block_cregs)}")
    fns = [steane_bit_fn(cr, r, s) for cr in block_cregs]
    if len(fns) == 2:
        f0, f1 = fns
        return lambda a, b: expr.bit_and(f0(a, b), f1(a, b))
    f0, f1, f2 = fns
    def maj(a, b):
        e0, e1, e2 = f0(a, b), f1(a, b), f2(a, b)
        return expr.bit_or(expr.bit_or(expr.bit_and(e0, e1),
                                       expr.bit_and(e0, e2)),
                           expr.bit_and(e1, e2))
    return maj


def numpy_confirmed(syn, round_idx):
    """Offline numpy mirror of confirmed_bit_fn for hardware-run audits.

    Args:
        syn: (shots, rounds, r, s) uint8 syndrome array (already the
             XOR-of-4 rolled syndrome, e.g. from steane_syndromes).
        round_idx: list of 2 or 3 round indices (oldest first) — must
             match the registers fed to confirmed_bit_fn in-circuit.

    Returns:
        (confirmed, stats) where confirmed is (shots, r, s) uint8 and
        stats is a dict with:
          'raw_defects':  total defects in the LAST round of the window
          'confirmed':    defects surviving the filter
          'vetoed':       raw_defects - confirmed (presumed syndrome errors)
    """
    syn = syn.astype(np.uint8)
    w = [syn[:, t] for t in round_idx]
    if len(w) == 2:
        conf = w[0] & w[1]
    elif len(w) == 3:
        conf = (w[0] & w[1]) | (w[0] & w[2]) | (w[1] & w[2])
    else:
        raise ValueError("numpy_confirmed: need 2 or 3 round indices")
    raw = int(w[-1].sum())
    c = int(conf.sum())
    return conf, {"raw_defects": raw, "confirmed": c, "vetoed": raw - c}


def _sector_coords(a, b):
    """Return (px, py, ha, hb) for parity-sector coords."""
    return a & 1, b & 1, a >> 1, b >> 1


def _linear_index(a, b, s):
    """Row-major flat index for grid position (a,b)."""
    return a * s + b


def _rect_xor_expr(syn_creg, r, s, a, b):
    """Build a classical expression = XOR of syn[i][j] for i < a, j < b
    within the same parity sector as (a,b).  Returns None for boundary.
    `syn_creg` may be a ClassicalRegister of raw syndrome bits or a
    callable (a, b) -> expression (see steane_bit_fn)."""
    if a == 0 or b == 0:
        return None
    px, py, ha, hb = _sector_coords(a, b)
    clause = None
    for i in range(ha):
        for j in range(hb):
            bit = _bit_expr(syn_creg, (i << 1) + px, (j << 1) + py, s)
            if clause is None:
                clause = bit
            else:
                clause = expr.bit_xor(clause, bit)
    return clause


def _all_and(exprs):
    """Classical expression for AND of all expressions in iterable."""
    it = iter(exprs)
    try:
        acc = next(it)
    except StopIteration:
        return None
    for e in it:
        acc = expr.bit_and(acc, e)
    return acc


def _or_of_combinations(terms, k):
    """OR of all C(n,k) k-combinations from terms list as classical expressions.

    Builds the expression for popcount >= k by enumerating all k-combinations
    (O(C(n,k)) nodes).  Practical for n <= 6.
    """
    n = len(terms)
    if k > n or k <= 0:
        return None
    if k == n:
        return _all_and(terms)
    result = None
    def _recurse(start, chosen):
        nonlocal result
        if len(chosen) == k:
            clause = _all_and(chosen)
            if clause is not None:
                result = clause if result is None else expr.bit_or(result, clause)
            return
        if start >= n:
            return
        chosen.append(terms[start])
        _recurse(start + 1, chosen)
        chosen.pop()
        _recurse(start + 1, chosen)
    _recurse(0, [])
    return result


def _row_flip_expr(col_terms, hs):
    """Build row-flip expression: majority of all hs positions in the row.

    The row consists of `hs` booleans.  Flip the row if popcount > hs/2
    (i.e. more ones than zeros — the strict majority of the canonicalize
    function).

    For hs <= 5 we enumerate all k-combinations explicitly; for larger hs
    this would be O(hs choose ceil(hs/2+1)) which is impractical in
    classical expressions, but the qubit mode handles large grids.
    """
    k = hs // 2 + 1  # strict majority: popcount > hs/2 → popcount >= hs//2+1
    return _or_of_combinations(col_terms, k)


def numpy_correction(syn, r, s):
    """Offline numpy replica of add_classical_correction: given syndrome
    arrays S (shots, r, s) uint8, return the correction E (shots, r, s)
    the dynamic stage applied in-circuit.  Bit-exact mirror of the
    expression logic — equivalence-tested against the circuit.  Use to
    AUDIT hardware runs: the steane block register returns with every
    shot, so what the conditionals did is fully reconstructible."""
    syn = syn.astype(np.uint8)
    shots = syn.shape[0]
    hr, hs = r // 2, s // 2
    E = np.zeros_like(syn)
    for px in (0, 1):
        for py in (0, 1):
            S = syn[:, px::2, py::2]
            ep = np.zeros_like(S)
            c = np.cumsum(S, axis=1) % 2
            c = np.cumsum(c, axis=2) % 2
            ep[:, 1:, 1:] = c[:, :-1, :-1]
            colw = ep[:, 1:, :].sum(axis=1)
            cf = (colw >= (hr // 2 + 1)).astype(np.uint8)
            ecol = ep ^ cf[:, None, :]
            roww = ecol[:, 1:, :].sum(axis=2)
            rf = (roww >= (hs // 2 + 1)).astype(np.uint8)
            etot = ecol.copy()
            etot[:, 1:, :] ^= rf[:, :, None]
            # pin: position (0,0) of the sector is NOT pinned to 0
            # (column flip at hb=0 can set it); only the coefficient bit
            # at pixel (px,py) is overridden by E_p (=0 at ha=0/hb=0)
            E[:, px::2, py::2] = etot

    # Logical fixup: nullspace element 2 (Z_L1Z_L2 = 1)
    k_fix = _get_logical_fixup(r, s)
    if k_fix is not None:
        L_before = (E[:, 0, :].sum(axis=1) + E[:, :, 0].sum(axis=1)) % 2
        rows, cols = np.where(k_fix)
        for a, b in zip(rows, cols):
            E[:, a, b] ^= L_before
        L_after = (E[:, 0, :].sum(axis=1) + E[:, :, 0].sum(axis=1)) % 2
        bad = np.where(L_after != 0)[0]
        if len(bad):
            print(f"\n### FIXUP FAIL on {len(bad)}/{len(L_after)} shots ###")
            for idx in bad[:5]:
                print(f"  shot {idx}: E wt before={E[idx].sum()}, L_before={L_before[idx]}, L_after={L_after[idx]}")
                print(f"  row0 sum={E[idx,0,:].sum()}, col0 sum={E[idx,:,0].sum()}")
                print(f"  E[{idx},K_fix]: {E[idx][k_fix]}")
                print(f"  E[{idx}]:\n{E[idx].astype(int)}")
    return E


def add_coherent_correction(qc_source, r, s, data_indices, block_indices,
                            block_creg_name):
    """Zero-branch in-circuit correction via the deferred-measurement
    principle: measure-then-conditionally-X equals CX-then-measure, so the
    entire feed-forward stage becomes a Clifford CX network with NO
    classical evaluation stall.

    Pipeline (all before the block's destructive measurement):
      1. comp(i,j)  <-  XOR-of-4 of block Z-info (4 CX per site): the
         syndrome S, computed coherently from the block qubits.
      2. 2D prefix-XOR on comp per parity sector (CX chains): E_p.
      3. corr(a,b)  <-  comp(a-2,b-2) for a,b >= 2 (row0/col0 pinned).
      4. CX corr -> data: the particular solution, applied physically.
      5. Measure corr into 'dc_corr_m' (audit record).
      6. Row-majority descent, HYBRID: the descent is nonlinear (no CX
         network computes a majority), but at hr=2 it reduces to
         (hr-1)*4 = 4 sector rows, each a popcount >= hs//2+1 of hs
         MEASURED bits — so the non-pinned comp rows are measured into
         'dc_row_m' and 4 SMALL if_tests apply the row flips to data.
         This is what rescues the col-0 (l2-flipping) error class that
         the linear prefix alone mis-corrects with weight-hs row reps
         (measured: without it, (1,0)/(3,0) fail on 4x8).  4 shallow
         expressions vs 16 deep ones in expr mode.  hr > 4 grids would
         also need the column majority — assert-guarded.

    Returns a NEW circuit with the stage spliced in immediately before
    the first measurement into `block_creg_name`."""
    from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

    n = r * s
    hr, hs = r // 2, s // 2
    assert hr <= 2, "coherent mode: hr > 2 also needs the column majority"

    comp_qr = QuantumRegister(n, "dc_comp")
    corr_qr = QuantumRegister(n, "dc_corr")
    corr_cr = ClassicalRegister(n, "dc_corr_m")
    nrow = 4 * (hr - 1) * (hs - 1)           # non-pinned comp-row bits
    row_cr = ClassicalRegister(max(nrow, 1), "dc_row_m")

    blk_creg = [cr for cr in qc_source.cregs if cr.name == block_creg_name][0]
    blk_bits = set(blk_creg)
    new = QuantumCircuit(*qc_source.qregs, comp_qr, corr_qr,
                         *qc_source.cregs, corr_cr, row_cr)

    def emit_stage():
        q = lambda a, b: block_indices[_linear_index(a % r, b % s, s)]
        comp = {(a, b): comp_qr[_linear_index(a, b, s)]
                for a in range(r) for b in range(s)}
        corr = {(a, b): corr_qr[_linear_index(a, b, s)]
                for a in range(r) for b in range(s)}
        # 1. coherent syndrome: comp <- XOR-of-4 block Z-info
        for a in range(r):
            for b in range(s):
                for da, db in ((0, 0), (2, 0), (0, 2), (2, 2)):
                    new.cx(q(a + da, b + db), comp[(a, b)])
        # 2. 2D prefix per parity sector (their Step-2 scan order)
        for px in range(2):
            for py in range(2):
                for ha in range(hr):
                    for hb in range(1, hs):
                        a, b = (ha << 1) + px, (hb << 1) + py
                        new.cx(comp[(a, ((hb - 1) << 1) + py)], comp[(a, b)])
                for hb in range(hs):
                    for ha in range(1, hr):
                        a, b = (ha << 1) + px, (hb << 1) + py
                        new.cx(comp[(((ha - 1) << 1) + px, b)], comp[(a, b)])
        # 3. exclusive shift onto corr (pin a<2 or b<2)
        for a in range(2, r):
            for b in range(2, s):
                new.cx(comp[(a - 2, b - 2)], corr[(a, b)])
        # 4. apply correction coherently
        for a in range(r):
            for b in range(s):
                new.cx(corr[(a, b)], data_indices[_linear_index(a, b, s)])
        # 5. audit record
        for a in range(r):
            for b in range(s):
                k = _linear_index(a, b, s)
                new.measure(corr[(a, b)], corr_cr[k])
        # 6. hybrid row-majority descent: measure the non-pinned comp rows
        # (comp now holds E_p; at hr=2 ecol == E_p since the column
        # majority never fires), then per sector row a small
        # popcount >= hs//2+1 if_test applies the row flip to data.
        ridx = 0
        row_groups = []                       # (global row a, py, bits)
        for px in (0, 1):
            for py in (0, 1):
                for ha in range(1, hr):
                    a = (ha << 1) + px
                    bits = []
                    # row values: {hb=0: col_flip = 0 at hr=2} plus
                    # ep(ha,hb) = inclusive_comp(ha-1, hb-1), hb=1..hs-1
                    for j in range(hs - 1):
                        b = (j << 1) + py
                        new.measure(comp[(((ha - 1) << 1) + px, b)],
                                    row_cr[ridx])
                        bits.append(row_cr[ridx])
                        ridx += 1
                    row_groups.append((a, py, bits))
        k_maj = hs // 2 + 1
        for a, py, bits in row_groups:
            cond = _or_of_combinations([expr.lift(bit) for bit in bits],
                                       k_maj)
            if cond is None:
                continue
            with new.if_test(cond):
                for hb in range(hs):
                    new.x(data_indices[_linear_index(a, (hb << 1) + py, s)])

    emitted = False
    for ins in qc_source.data:
        if not emitted and ins.operation.name == "measure" \
                and any(c in blk_bits for c in ins.clbits):
            emit_stage()
            emitted = True
        new.append(ins.operation, ins.qubits, ins.clbits)
    if not emitted:
        raise ValueError(f"no measurement into {block_creg_name} found")
    return new


def _sigma_prefix_c_terms(u, v):
    """c-terms (sector-prefix coords) whose XOR equals the INCLUSIVE
    sigma-prefix  I_S(u, v) = XOR_{i<=u, j<=v} sigma(i, j),  where
    sigma(i,j) = beta(i,j)^beta(i+1,j)^beta(i,j+1)^beta(i+1,j+1) is the
    sector-local syndrome and c is the inclusive 2D prefix of the RAW
    block bits beta.  Inclusion-exclusion over the four beta offsets:
      I_S(u,v) = XOR_{di,dj in {0,1}} rect(di..u+di, dj..v+dj)
      rect(i1..i2, j1..j2) = c(i2,j2)^c(i1-1,j2)^c(i2,j1-1)^c(i1-1,j1-1)
    (negative-index terms dropped; no wraparound needed for u <= hr-2,
    which the hr==2 guard of the in-place stage guarantees).  Returns
    the set of (i, j) c-coordinates with odd multiplicity — at hr=2 this
    always collapses to <= 3 terms."""
    par = {}
    def _rect(i1, i2, j1, j2):
        for i, j in ((i2, j2), (i1 - 1, j2), (i2, j1 - 1), (i1 - 1, j1 - 1)):
            if i >= 0 and j >= 0:
                par[(i, j)] = par.get((i, j), 0) ^ 1
    for di in (0, 1):
        for dj in (0, 1):
            _rect(di, u + di, dj, v + dj)
    return {k for k, p in par.items() if p}


def add_inplace_coherent_correction(qc_source, r, s, data_indices,
                                    block_indices, block_creg_name):
    """Coherent correction stage with ZERO fresh qubits: the Steane block
    qubits double as the compute register.

    Rationale (measured): add_coherent_correction's 2n fresh ancillas
    (dc_comp/dc_corr) route terribly on heavy-hex — ~160 logical CX
    ballooned to ~1030 transpiled 2q, and the data decohered through the
    routed depth.  The block qubits, by contrast, are already placed
    adjacent to data (transversal CX layout) and adjacent to each other
    (sector neighbours are grid neighbours), so the same algebra runs
    near-natively on them.

    The XOR-of-4 roll map b -> S is NOT invertible (its kernel is the
    code), so S cannot be materialised in place.  Instead only the
    INVERTIBLE sector prefix  c = P.b  is computed in place (triangular,
    pure CX), and every needed quantity is expanded in c:

      E_p at data site (a,b), sector coord (u,v), u,v >= 1:
          = I_S(u-1, v-1)  ->  XOR of <= 3 c-qubits at hr=2
            (u=1: c(1,v-1) ^ c(1,v) ^ c(1,0))
      row-majority bits (hr=2, sector row ha=1):
          I_S(0, j) = c(1,j) ^ c(1,j+1) ^ c(1,0),  j = 0..hs-2

    Pipeline (spliced before the block's destructive measurement):
      1. In-place sector prefix on block qubits (~2n neighbour CX).
      2. Row bits: fold the non-anchor c-terms onto an anchor block
         qubit via CX, measure into 'dc_row_m', un-fold.  (Mid-circuit
         Z measurement commutes with everything downstream: the block
         is only ever a CX control / Z-readout from here on.)
      3. E_p application: <= 3 CX from c-qubits into each data site with
         a, b >= 2.  E_p never touches grid row 0 / col 0, so the LINEAR
         part carries ZERO Z_L1Z_L2 action by construction.
      4. Row-flip if_tests (popcount >= hs//2+1 of the measured bits),
         X on the full sector row — and, for py == 0 groups, the K_fix
         kernel pattern INSIDE the same if_test: a py=0 row flip is the
         only logical-parity source (it flips site (a,0)), and K_fix has
         logical action 1, so each fired flip self-cancels.  This is the
         kernel fixup the fresh-ancilla stage lacked (measured: 48.9%
         logical action without it).
      5. UNCOMPUTE the prefix (reverse CX): the destructive readout
         returns the RAW block bits b — steane_syndromes, steane_bit_fn,
         confirmed_bit_fn and every audit path work unchanged.

    Audit: the applied correction is fully reconstructible offline from
    the raw block readout + 'dc_row_m' — see numpy_inplace_replica
    (bit-exact equal to numpy_correction at hr=2, fixup included).

    Returns a NEW circuit with the stage spliced in immediately before
    the first measurement into `block_creg_name`."""
    from qiskit import ClassicalRegister, QuantumCircuit

    hr, hs = r // 2, s // 2
    assert hr == 2, "in-place mode derived for hr == 2 (row majority only)"

    nrow = 4 * (hr - 1) * (hs - 1)
    row_cr = ClassicalRegister(max(nrow, 1), "dc_row_m")
    blk_creg = [cr for cr in qc_source.cregs if cr.name == block_creg_name][0]
    blk_bits = set(blk_creg)
    new = QuantumCircuit(*qc_source.qregs, *qc_source.cregs, row_cr)

    k_fix = _get_logical_fixup(r, s)

    def emit_stage():
        # c-qubit lookup: sector (px,py), prefix coord (u,v) -> block qubit
        def cq(px, py, u, v):
            return block_indices[_linear_index((u << 1) + px,
                                               (v << 1) + py, s)]

        # 1. in-place inclusive 2D sector prefix (same scan order as the
        #    fresh-ancilla stage; targets processed in increasing order so
        #    each CX adds an already-completed prefix)
        prefix_cx = []
        for px in range(2):
            for py in range(2):
                for u in range(hr):
                    for v in range(1, hs):
                        prefix_cx.append((cq(px, py, u, v - 1),
                                          cq(px, py, u, v)))
                for v in range(hs):
                    for u in range(1, hr):
                        prefix_cx.append((cq(px, py, u - 1, v),
                                          cq(px, py, u, v)))
        for ctl, tgt in prefix_cx:
            new.cx(ctl, tgt)

        # 2. row-majority bits: fold onto anchor, measure, un-fold
        ridx = 0
        row_groups = []                       # (grid row a, py, creg bits)
        for px in (0, 1):
            for py in (0, 1):
                for ha in range(1, hr):       # ha == 1 at hr == 2
                    a = (ha << 1) + px
                    bits = []
                    for j in range(hs - 1):
                        terms = _sigma_prefix_c_terms(ha - 1, j)
                        anchor = (1, j + 1)   # c(1, j+1) always present
                        rest = [t for t in terms if t != anchor]
                        assert anchor in terms
                        aq = cq(px, py, *anchor)
                        for (ti, tj) in rest:
                            new.cx(cq(px, py, ti, tj), aq)
                        new.measure(aq, row_cr[ridx])
                        for (ti, tj) in rest:
                            new.cx(cq(px, py, ti, tj), aq)
                        bits.append(row_cr[ridx])
                        ridx += 1
                    row_groups.append((a, py, bits))

        # 3. E_p application: <= 3 CX per data site, block -> data
        for a in range(2, r):
            for b in range(2, s):
                px, py, u, v = _sector_coords(a, b)
                for (ti, tj) in _sigma_prefix_c_terms(u - 1, v - 1):
                    new.cx(cq(px, py, ti, tj),
                           data_indices[_linear_index(a, b, s)])

        # 4. row flips + in-branch kernel fixup
        k_maj = hs // 2 + 1
        for a, py, bits in row_groups:
            cond = _or_of_combinations([expr.lift(bit) for bit in bits],
                                       k_maj)
            if cond is None:
                continue
            with new.if_test(cond):
                for hb in range(hs):
                    new.x(data_indices[_linear_index(a, (hb << 1) + py, s)])
                if py == 0 and k_fix is not None:
                    # this flip touches (a, 0): logical parity 1 — cancel
                    # with the logical-1 kernel element, syndrome-neutral
                    for (ka, kb) in zip(*np.where(k_fix)):
                        new.x(data_indices[_linear_index(ka, kb, s)])

        # 5. uncompute the prefix: raw b restored for destructive readout
        for ctl, tgt in reversed(prefix_cx):
            new.cx(ctl, tgt)

    emitted = False
    for ins in qc_source.data:
        if not emitted and ins.operation.name == "measure" \
                and any(c in blk_bits for c in ins.clbits):
            emit_stage()
            emitted = True
        new.append(ins.operation, ins.qubits, ins.clbits)
    if not emitted:
        raise ValueError(f"no measurement into {block_creg_name} found")
    return new


def numpy_inplace_replica(block_bits, row_bits, r, s):
    """Offline reconstruction of the correction applied by
    add_inplace_coherent_correction, from returned registers.

    Args:
        block_bits: (shots, r, s) uint8 RAW block readout (the stage
            uncomputes its prefix, so steane_{t} holds raw b as always).
        row_bits: (shots, nrow) uint8 'dc_row_m' readout.

    Returns:
        (shots, r, s) uint8 applied correction E — bit-exact to what the
        circuit did, and (at hr == 2) to numpy_correction(S) including
        the kernel fixup.
    """
    b = block_bits.astype(np.uint8)
    hr, hs = r // 2, s // 2
    # syndrome via the roll map (same math as steane_bit_fn)
    S = (b ^ np.roll(b, -2, 1) ^ np.roll(b, -2, 2)
           ^ np.roll(np.roll(b, -2, 1), -2, 2)) & 1
    E = np.zeros_like(b)
    for px in (0, 1):
        for py in (0, 1):
            sec = S[:, px::2, py::2]
            c = np.cumsum(np.cumsum(sec, axis=1), axis=2) & 1
            ep = np.zeros_like(sec)
            ep[:, 1:, 1:] = c[:, :-1, :-1]
            E[:, px::2, py::2] = ep
    # measured row flips + in-branch kernel fixup
    k_fix = _get_logical_fixup(r, s)
    ridx = 0
    for px in (0, 1):
        for py in (0, 1):
            for ha in range(1, hr):
                a = (ha << 1) + px
                grp = row_bits[:, ridx:ridx + hs - 1]
                ridx += hs - 1
                rf = (grp.sum(axis=1) >= hs // 2 + 1).astype(np.uint8)
                E[:, a, py::2] ^= rf[:, None]
                if py == 0 and k_fix is not None:
                    E[:, k_fix] ^= rf[:, None]
    return E


def _get_logical_fixup(r, s):
    """Return the K_fix pattern: a nullspace element with Z_L1Z_L2 logical = 1.

    For the (1+x²)(1+y²) BB code, Z_L1Z_L2 is the parity of the correction
    on row 0 XOR column 0.  The 16 nullspace elements are the 2D propagations
    of the 4 corner bits in the first 2×2 block.  Applying K_fix conditionally
    (when the correction's logical parity is 1) zeros the logical action
    without changing the syndrome (it's a pure kernel element).

    Corner (0,1) propagation gives Z_L1Z_L2 = 1 for all even r, s >= 4.

    Returns (r×s) bool array or None for unsupported grid sizes.
    """
    if r % 2 == 0 and s % 2 == 0 and r >= 4 and s >= 4:
        k = np.zeros((r, s), dtype=bool)
        hr, hs = r // 2, s // 2
        # Corner (0,1) → pixel (0, 1) set, then 2D propagation within
        # the py=1, px=0 sector: positions (2*ha, 2*hb+1) for ha>0, hb>0
        k[0, 1] = True  # the original corner bit
        for ha in range(1, hr):
            for hb in range(1, hs):
                k[2 * ha, 2 * hb + 1] = True
        return k
    return None


def add_classical_correction(qc, r, s, data_indices, syn_creg):
    """Append dynamic correction stage with full kernel descent.

    Computes:
      1. Particular solution E_p (2D prefix XOR within each parity sector).
      2. Column flips: for each sector column b, if popcount(E_p[:,b]) > hr/2,
         flip the entire column (to minimise weight within the coset).
      3. Apply column flips → E_col.
      4. Row flips: for each sector row a (a>0), if popcount(E_col[a,:]) >
         hs/2, flip the entire row.
      5. Apply row flips → E_total.
      6. Compute Z_L1Z_L2 logical parity L of E_total from its boundary
         (row-0 and column-0) values.
      7. For positions in a precomputed kernel element K_fix (logical=1),
         apply: E_final = E_total XOR L.  This zeros the logical action
         without affecting the syndrome (K_fix is a pure stabiliser).

    This is equivalent to one full iteration of plane_warp.c's column/row
    descent, with an additional logical-fixup stage that the C decoder
    achieves implicitly by enumerating all 16 nullspace candidates.
    """
    px_set = (0, 0), (0, 1), (1, 0), (1, 1)
    hr = r // 2
    hs = s // 2

    # 1a. Build E_p expressions for every grid position.
    # ep[a][b] = expr or None (boundary)
    ep = {}
    for a in range(r):
        for b in range(s):
            clause = _rect_xor_expr(syn_creg, r, s, a, b)
            if clause is not None:
                ep[(a, b)] = clause

    # 1b. Build column-flip expressions per sector column.
    # Flip the column iff a strict majority of its hr entries are 1
    # (pinned ha=0 entry is identically 0, so it counts as a zero — the
    # majority is over hr with only the non-pinned terms able to be set).
    col_flip = {}
    k_col = hr // 2 + 1  # strict majority of hr entries
    for px, py in px_set:
        for hb in range(hs):
            b = (hb << 1) + py
            terms = []
            for ha in range(1, hr):
                a = (ha << 1) + px
                e = ep.get((a, b))
                if e is not None:
                    terms.append(e)
            cf = _or_of_combinations(terms, k_col)
            if cf is not None:
                col_flip[(px, py, hb)] = cf

    # 2. Build E_col = E_p XOR col_flip for each position
    ecol = {}
    for a in range(r):
        for b in range(s):
            e = ep.get((a, b))
            if e is None:
                continue
            px, py, ha, hb = _sector_coords(a, b)
            cf = col_flip.get((px, py, hb))
            if cf is not None:
                ecol[(a, b)] = expr.bit_xor(e, cf)
            else:
                ecol[(a, b)] = e

    # 3. Build row-flip expressions per sector row.
    # row_flip[(px, py, ha)] = majority of ALL hs positions in the row
    #   hb=0: value = col_flip[(px, py, 0)]  (E_p = 0 at boundary)
    #   hb>=1: value = ecol = E_p XOR col_flip[hb]
    row_flip = {}
    for px, py in px_set:
        for ha in range(1, hr):
            a = (ha << 1) + px
            row_terms = []
            for hb in range(hs):
                b = (hb << 1) + py
                e = ecol.get((a, b))
                if e is not None:
                    row_terms.append(e)
                else:
                    cf = col_flip.get((px, py, hb))
                    if cf is not None:
                        row_terms.append(cf)
            if row_terms:
                rf = _row_flip_expr(row_terms, hs)
                if rf is not None:
                    row_flip[(px, py, ha)] = rf

    # 4. Build E_total expressions for every grid position.
    #
    #   ha>0, hb>0  → E = E_p XOR col_flip[hb] XOR row_flip[ha]
    #   ha=0, hb>0  → E = col_flip[hb]          (column flip only)
    #   ha>0, hb=0  → E = col_flip[hb=0] XOR row_flip[ha]
    #   ha=0, hb=0  → E = col_flip[hb=0]        (column flip only)
    #
    # NOTE: hb=0 sectors have no non-pinned E_p, so col_flip for those
    # columns is None (identically 0).  We still create an entry with a
    # constant-False expression so the logical fixup can XOR L into it.
    _zero = expr.lift(False)
    etotal = {}
    for a in range(r):
        for b in range(s):
            px, py, ha, hb = _sector_coords(a, b)
            terms = []
            e_ep = ep.get((a, b))
            if e_ep is not None:
                terms.append(e_ep)
            cf = col_flip.get((px, py, hb))
            if cf is not None:
                terms.append(cf)
            if ha > 0:
                rf = row_flip.get((px, py, ha))
                if rf is not None:
                    terms.append(rf)
            if terms:
                clause = terms[0]
                for t in terms[1:]:
                    clause = expr.bit_xor(clause, t)
            else:
                clause = _zero  # constant False — fixup may XOR L into it
            etotal[(a, b)] = clause

    # 5. Compute L = Z_L1Z_L2 logical parity of E_total.
    #    = XOR of E_total on row 0 XOR XOR of E_total on column 0.
    L = None
    for b in range(s):
        clause = etotal.get((0, b))
        if clause is not None:
            L = clause if L is None else expr.bit_xor(L, clause)
    for a in range(1, r):
        clause = etotal.get((a, 0))
        if clause is not None:
            L = clause if L is None else expr.bit_xor(L, clause)

    # 6. Apply corrections with logical-fixup.
    #    E_final = E_total XOR (L AND K_fix) for positions in K_fix.
    #    K_fix is a precomputed nullspace element with logical=1.
    k_fix = _get_logical_fixup(r, s)
    for (a, b), clause in etotal.items():
        if k_fix is not None and k_fix[a, b] and L is not None:
            clause = expr.bit_xor(clause, L)
        with qc.if_test(clause):
            qc.x(data_indices[_linear_index(a, b, s)])


def add_qubit_correction(qc, r, s, data_indices, syn_creg):
    """Append dynamic correction stage using ancilla qubits + CX gates.

    Allocates r*s correction-compute qubits and r*s correction-result
    qubits.  Loads syndrome from classical register via conditional X,
    then computes the 2D prefix-XOR using CX gates (Clifford, O(r*s)),
    measures the result, and applies conditional X to data.

    This avoids O(n^2) classical expression growth and is suitable for
    80x80 grids.

    Args:
        qc: QuantumCircuit (modified in-place; new registers added)
        r, s: grid dimensions (even)
        data_indices: list[int] of r*s data qubit indices in row-major order
        syn_creg: ClassicalRegister with r*s bits, holding last-round syndrome

    Returns:
        Number of CX gates added
    """
    from qiskit import QuantumRegister, ClassicalRegister

    n_data = r * s
    n_cx = 0

    # Allocate compute and correction ancillas
    comp_qr = QuantumRegister(n_data, "dc_comp")
    corr_qr = QuantumRegister(n_data, "dc_corr")
    corr_cr = ClassicalRegister(n_data, "dc_corr_m")
    qc.add_register(comp_qr)
    qc.add_register(corr_qr)
    qc.add_register(corr_cr)

    comp_base = len(qc.qubits) - 2 * n_data
    corr_base = len(qc.qubits) - n_data

    comp_ofs = lambda a, b: comp_base + _linear_index(a, b, s)
    corr_ofs = lambda a, b: corr_base + _linear_index(a, b, s)

    # Step 1: Load syndrome from creg into compute qubits via conditional X
    for k in range(n_data):
        with qc.if_test((syn_creg[k], 1)):
            qc.x(comp_base + k)

    # Step 2: 2D prefix XOR on compute qubits (in-place, per parity sector)
    for px in range(2):
        for py in range(2):
            hr, hs = r // 2, s // 2
            for ha in range(hr):
                for hb in range(1, hs):
                    a, b = (ha << 1) + px, (hb << 1) + py
                    ap, bp = (ha << 1) + px, ((hb - 1) << 1) + py
                    qc.cx(comp_ofs(ap, bp), comp_ofs(a, b))
                    n_cx += 1
            for hb in range(hs):
                for ha in range(1, hr):
                    a, b = (ha << 1) + px, (hb << 1) + py
                    ap, bp = ((ha - 1) << 1) + px, (hb << 1) + py
                    qc.cx(comp_ofs(ap, bp), comp_ofs(a, b))
                    n_cx += 1

    # Step 3: Copy correction to result ancillas
    #   2D prefix is per parity sector with stride 2.
    #   E[a][b] = prefix XOR at (a-2, b-2) within the same sector,
    #   which equals comp[a-2][b-2] after the two-pass XOR.
    #   Boundary: a<2 or b<2 → E=0 (row-0 / col-0 pin).
    for a in range(r):
        for b in range(s):
            if a >= 2 and b >= 2:
                qc.cx(comp_ofs(a - 2, b - 2), corr_ofs(a, b))
                n_cx += 1

    # Step 4: Measure correction ancillas
    for a in range(r):
        for b in range(s):
            k = _linear_index(a, b, s)
            qc.measure(corr_ofs(a, b), corr_cr[k])

    # Step 5: Apply conditional X to data
    for a in range(r):
        for b in range(s):
            k = _linear_index(a, b, s)
            with qc.if_test((corr_cr[k], 1)):
                qc.x(data_indices[k])

    return n_cx


def add_dynamic_correction(qc, r, s, data_indices, syn_creg,
                           classical=True):
    """Convenience wrapper — dispatches to classical or qubit mode.

    Args:
        qc: QuantumCircuit (modified in-place)
        r, s: grid dimensions
        data_indices: r*s data qubit indices in row-major order
        syn_creg: ClassicalRegister with r*s bits (last-round syndrome)
        classical: if True, use classical expressions (O(n^2) expr nodes);
                   if False, use qubit-based CX approach (O(n) gates, ~3n qubits)
    Returns:
        Number of added gates (CX for qubit mode, 0 for classical mode)
    """
    if classical:
        add_classical_correction(qc, r, s, data_indices, syn_creg)
        return 0
    else:
        return add_qubit_correction(qc, r, s, data_indices, syn_creg)


def inject_dynamic_stage(circuit, r, s, data_indices, syn_creg,
                         classical=True):
    """Return a NEW circuit with dynamic correction stage injected
    immediately before the final data measurement.

    The original circuit is unmodified.  The new circuit executes all
    rounds, then applies the dynamic correction (computed from the
    last-round syndrome classical register), then measures the data qubits.

    This is the correct placement: data qubits are corrected BEFORE
    readout, so the raw readout is the corrected value.

    Args:
        circuit: source QuantumCircuit (e.g. from build_circuit)
        r, s: grid dimensions
        data_indices: r*s data qubit indices in row-major order
        syn_creg: ClassicalRegister holding the last-round syndrome
        classical: whether to use classical-expression mode

    Returns:
        New QuantumCircuit with the injected stage.
    """
    # Identify the "data" classical register
    data_reg = None
    data_bits = set()
    for cr in circuit.cregs:
        if cr.name == "data":
            data_reg = cr
            data_bits = set(cr)
            break

    # Separate data-measurement instructions from everything else
    other = []
    data_meas = []
    for inst, qargs, cargs in circuit.data:
        if inst.name == "measure" and any(c in data_bits for c in cargs):
            data_meas.append((inst, qargs, cargs))
        else:
            other.append((inst, qargs, cargs))

    new = circuit.copy_empty_like()
    for inst, qargs, cargs in other:
        new._append(inst, qargs, cargs)

    add_dynamic_correction(new, r, s, data_indices, syn_creg,
                           classical=classical)

    for inst, qargs, cargs in data_meas:
        new._append(inst, qargs, cargs)

    return new