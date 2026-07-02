"""
pw_opt.py — Adapted for IONQ Forte (36-qubit trapped-ion, all-to-all).

Key differences from the Heron version:
  - All-to-all connectivity: no routing, no SWAPs, any qubit talks to any other.
  - 36-qubit hard cap. Largest periodic grid: 4×6 (24 data + 12 anc = 36).
  - Native gates: GPi, GPi2, MS (Mølmer-Sørensen).  We emit CX and let the
    transpiler decompose to MS; Forte's compiler handles this efficiently.
  - compact=True always — no heavy-hex flag layout.
  - periodic=False recommended (open boundaries save ancillas and shorten
    syndrome paths on small grids).

Syndrome format unchanged: (shots, rounds, r, s) — compatible with decoder.py.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def _check_anchors(r, s):
    hr, hs = r // 2, s // 2
    return [(2 * p + px, 2 * q + py)
            for px in range(2) for py in range(2)
            for p in range(hr - 1) for q in range(hs)]


def _unpack_indices(r, s):
    a = _check_anchors(r, s)
    return np.array([x[0] for x in a]), np.array([x[1] for x in a])


def _bell_support(r, s, periodic):
    if periodic:
        return [(i, 0) for i in range(1, r)] + [(0, j) for j in range(1, s)]
    return [(i, 0) for i in range(r)] + [(i, 2) for i in range(r)]


def build_circuit(r, s, rounds, logical_state="00", bell=False,
                  bell_measure=False, measure_x=False, partial_x=False,
                  stabilizer_basis="Z", no_reset=True,
                  free_final_round=False, full_stabilizer=False,
                  periodic=False, initial_reset=False,
                  share_extra_ancilla=False, bell_ancilla=True):
    """Build a QEC circuit for the Forte architecture (all-to-all).

    periodic=False is recommended: for 4×6 open boundaries, we have 24 data
    + 12 ancillas = 36 qubits (exactly Forte's capacity).  With Bell ancilla
    the total is 25 + 12 = 37 — we squeeze by share_extra_ancilla=True.
    """
    n_data = r * s
    hr, hs = r // 2, s // 2
    n_anc = 4 * (hr - 1) * hs
    checks = _check_anchors(r, s)

    extra_flags = [name for name, on in
                   (("bell", bell and bell_ancilla),
                    ("bell_m", bell_measure)) if on]

    # compact layout: data[i][j] = i*s + j, anc[k] = n_data + k
    def _dq(i, j):
        return i * s + j

    _anc_index = {c: n_data + k for k, c in enumerate(checks)}

    def _aq(i, j):
        return _anc_index[(i, j)]

    base = n_data + n_anc

    if share_extra_ancilla and extra_flags:
        extra_idx = {name: base - 1 for name in extra_flags}
        n_extra = 0
    else:
        extra_idx = {name: base + k for k, name in enumerate(extra_flags)}
        n_extra = len(extra_flags)

    total = base + n_extra
    assert total <= 36, f"Circuit needs {total} qubits, Forte has 36.  "
    f"Reduce r,s or use share_extra_ancilla=True."

    qec_rounds = rounds - 1 if free_final_round else rounds
    basis_seq = stabilizer_basis.upper()
    first_basis = basis_seq[0]

    qr = QuantumRegister(total, "q")
    cr_syn = [ClassicalRegister(n_anc, f"syn_{c}") for c in range(qec_rounds)]
    cr_data = ClassicalRegister(n_data, "data")
    cregs = [*cr_syn, cr_data]
    extra_cr = {}
    for name in extra_flags:
        cr = ClassicalRegister(1, name)
        extra_cr[name] = cr
        cregs.append(cr)
    qc = QuantumCircuit(qr, *cregs)

    extra_used = [False]

    def _parity_measure(anc, qubits, cbit):
        qc.h(anc)
        for dq_ in qubits:
            qc.cx(anc, dq_)
        qc.h(anc)
        qc.measure(anc, cbit)

    if bell:
        if bell_ancilla:
            sup = [_dq(i, j) for (i, j) in _bell_support(r, s, periodic)]
            _parity_measure(extra_idx["bell"], sup, extra_cr["bell"][0])
        else:
            qc.h(_dq(0, 0))
    else:
        if first_basis == "X":
            for ii in range(r):
                for jj in range(s):
                    qc.h(_dq(ii, jj))
        if "1" in logical_state:
            flip = qc.z if first_basis == "X" else qc.x
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

    # QEC rounds
    def row2(i):
        return (i + 2) % r if periodic else i + 2

    anc_list = [_aq(i, j) for (i, j) in checks]
    offsets = [(0, 0), (2, 0), (0, 2), (2, 2)] if full_stabilizer \
              else [(0, 0), (2, 0)]

    for rnd in range(qec_rounds):
        rb = basis_seq[rnd % len(basis_seq)]
        # No reset on Forte — XOR differencing in all_syndromes_opt
        if rb == "X":
            for a in anc_list:
                qc.h(a)
        for (di, dj) in offsets:
            for (i, j), a in zip(checks, anc_list):
                ti = row2(i) if di else i
                tj = (j + dj) % s
                if periodic or (0 <= ti < r and 0 <= tj < s):
                    if rb == "X":
                        qc.cx(a, _dq(ti, tj))
                    else:
                        qc.cx(_dq(ti, tj), a)
        if rb == "X":
            for a in anc_list:
                qc.h(a)
        for slot, a in enumerate(anc_list):
            qc.measure(a, cr_syn[rnd][slot])

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

    eff_data_map = [[_dq(ii, jj) for jj in range(s)] for ii in range(r)]
    return qc, eff_data_map, [], [], n_anc


# ── syndrome extraction (same as Heron version) ──

def all_syndromes_opt(pub_result, rounds, r, s, n_anc,
                       no_reset=True, free_final_round=False,
                       data_raw=None, full_stabilizer=False, periodic=True):
    anc_rounds = rounds - 1 if free_final_round else rounds
    if anc_rounds == 0:
        return np.zeros((data_raw.shape[0], rounds, r, s), dtype=np.uint8)

    first = getattr(pub_result.data, "syn_0")
    shots = first.num_shots

    m_raw = np.zeros((shots, anc_rounds, n_anc), dtype=np.uint8)
    for c in range(anc_rounds):
        bits = getattr(pub_result.data, f"syn_{c}").to_bool_array(order="little")
        m_raw[:, c] = bits[:, :n_anc].astype(np.uint8)

    if no_reset:
        m_parity = m_raw.copy()
        m_parity[:, 1:] ^= m_raw[:, :-1]
    else:
        m_parity = m_raw

    ui, uj = _unpack_indices(r, s)
    V = np.zeros((shots, anc_rounds, r, s), dtype=np.uint8)
    V[:, :, ui, uj] = m_parity

    if periodic:
        V[:, :, r - 2, :] = V[:, :, 0:r - 2:2, :].sum(axis=2) % 2
        V[:, :, r - 1, :] = V[:, :, 1:r - 1:2, :].sum(axis=2) % 2

    syn = np.zeros((shots, rounds, r, s), dtype=np.uint8)
    if full_stabilizer:
        syn[:, :anc_rounds] = V
    else:
        syn[:, :anc_rounds] = V ^ np.roll(V, shift=-2, axis=3)

    if free_final_round and data_raw is not None:
        V_last = data_raw.astype(np.uint8) ^ np.roll(data_raw.astype(np.uint8), shift=-2, axis=1)
        syn[:, -1] = V_last ^ np.roll(V_last, shift=-2, axis=2)

    return syn


def round_bases(rounds, stabilizer_basis):
    seq = stabilizer_basis.upper()
    return [seq[t % len(seq)] for t in range(rounds)]
