"""Derive |0>_L / |+>_L for THIS code (periodic weight-2-gauge, weight-4 S)
straight from its own stabilizer + logical operators, then verify.

Logicals (periodic), matched to bell_complex's readout:
  Z_L1 = Z(row 0), Z_L2 = Z(col 0)          (ZZ arm = <Z_L1 Z_L2>)
  X_L1 = X(col 0), X_L2 = X(row 0)          (X_L1.X_L2 support = row0 u col0)
Stabilizers (full_stabilizer): S_P(i,j) = P(i,j)P(i+2,j)P(i,j+2)P(i+2,j+2),
  P in {X,Z}, indices mod r/s, anchors = _check_anchors.  V_P(i,j)=P(i,j)P(i+2,j)
  are the weight-2 gauge operators (used only to gauge-fix to a pure state).
"""
import numpy as np
from qiskit.quantum_info import Pauli, StabilizerState
from qiskit.synthesis import synth_circuit_from_stabilizers
from pw_opt import _check_anchors


def _P(n, kind, coords):
    """Pauli string on n qubits (q = i*s + j), leftmost char = qubit n-1."""
    ch = ["I"] * n
    for q in coords:
        ch[q] = kind
    return Pauli("".join(reversed(ch)))


def _sym(p):
    return np.concatenate([p.x.astype(np.uint8), p.z.astype(np.uint8)])


def _commute(a, b):
    return (int(np.dot(a.x.astype(int), b.z.astype(int))) +
            int(np.dot(a.z.astype(int), b.x.astype(int)))) % 2 == 0


def _independent(vec, basis):
    """Gaussian-eliminate vec against reduced basis rows; return residual or None."""
    v = vec.copy()
    for piv, row in basis:
        if v[piv]:
            v ^= row
    nz = np.nonzero(v)[0]
    return (nz[0], v) if len(nz) else None


def code_operators(r, s):
    n = r * s
    q = lambda i, j: (i % r) * s + (j % s)
    anchors = _check_anchors(r, s)

    def S(kind):
        return [_P(n, kind, [q(i, j), q(i + 2, j), q(i, j + 2), q(i + 2, j + 2)])
                for (i, j) in anchors]

    def V(kind):   # weight-2 gauge, all independent rows i=0..r-3
        return [_P(n, kind, [q(i, j), q(i + 2, j)])
                for i in range(r - 2) for j in range(s)]

    L = dict(
        XL1=_P(n, "X", [q(i, 0) for i in range(r)]),          # col 0
        XL2=_P(n, "X", [q(0, j) for j in range(s)]),          # row 0
        ZL1=_P(n, "Z", [q(0, j) for j in range(s)]),          # row 0
        ZL2=_P(n, "Z", [q(i, 0) for i in range(r)]),          # col 0
    )
    return n, {"SX": S("X"), "SZ": S("Z"), "VX": V("X"), "VZ": V("Z"), **L}


def _vec_to_pauli(vec, n):
    x, z = vec[:n], vec[n:]
    ch = ["IXZY"[int(x[q]) + 2 * int(z[q])] for q in range(n)]
    return Pauli("".join(reversed(ch)))


def _gf2_null(rows, ncols):
    """Basis of the GF(2) null space of the given rows (full RREF)."""
    R = [r.copy() % 2 for r in rows]
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


def _complete_isotropic(chosen_syms, n):
    """Extend an isotropic set (symplectic vecs x|z) to a maximal one (size n)
    by drawing independent generators from its symplectic centralizer.  This is
    what fixes the spectator-sector DOF that no weight-2/single-qubit operator
    in a fixed pool could reach."""
    chosen = [v.copy() % 2 for v in chosen_syms]
    while len(chosen) < n:
        # commuting w must satisfy, for each c=(cx|cz): cz.wx + cx.wz = 0
        cons = [np.concatenate([c[n:], c[:n]]) for c in chosen]
        null = _gf2_null(cons, 2 * n)
        span = []
        for c in chosen:
            r = _independent(c, span)
            if r:
                span.append(r)
        for w in null:
            if w.any() and _independent(w, span) is not None:
                chosen.append(w)
                break
        else:
            break
    return chosen


def encoded_stabilizer_list(r, s, state):
    """Maximal abelian independent generating set of n Paulis for |0>_L/|+>_L.

    All S (both types) + the state's two logicals go in first; remaining
    spectator-sector DOF are pinned by symplectic-centralizer completion."""
    n, ops = code_operators(r, s)
    if state == "plus":
        must = ops["SX"] + ops["SZ"] + [ops["XL1"], ops["XL2"]]
    else:  # zero
        must = ops["SX"] + ops["SZ"] + [ops["ZL1"], ops["ZL2"]]
    # Fill spectator-sector DOF. Greedy commute check refuses anything
    # anticommuting with the state's chosen logical, so the complementary
    # witness logical provably stays OUT of the group (uniform): e.g. Z_L1
    # anticommutes with the already-added X_L1.
    fill = ops["VX"] + ops["VZ"]

    chosen, basis = [], []
    def try_add(p):
        if not all(_commute(p, c) for c in chosen):
            return False
        r_ = _independent(_sym(p), basis)
        if r_ is None:
            return False
        chosen.append(p); basis.append(r_)
        return True

    n_must_S = len(ops["SX"]) + len(ops["SZ"])
    got_S = sum(try_add(p) for p in must[:n_must_S])
    for p in must[n_must_S:]:
        try_add(p)
    for p in fill:
        if len(chosen) == n:
            break
        try_add(p)
    # rigorous completion of any remaining spectator DOF
    syms = _complete_isotropic([_sym(p) for p in chosen], n)
    gens = [_vec_to_pauli(v, n) for v in syms]
    return n, gens, got_S


def verify(r, s):
    n, ops = code_operators(r, s)
    for state in ("plus", "zero"):
        n_, gens, got_S = encoded_stabilizer_list(r, s, state)
        assert len(gens) == n, f"{state}: {len(gens)} gens != {n} (not pure)"
        st = StabilizerState(synth_circuit_from_stabilizers(gens))
        # every S must be +1
        for kind in ("SX", "SZ"):
            for P in ops[kind]:
                ev = st.expectation_value(P)
                assert abs(ev - 1) < 1e-9, f"{state}: {kind} exp {ev} != +1"
        # logical structure
        if state == "plus":
            xl = st.expectation_value(ops["XL1"]) * st.expectation_value(ops["XL2"])
            zl = abs(st.expectation_value(ops["ZL1"]))  # uniform -> 0
            assert abs(xl - 1) < 1e-9 and zl < 1e-9, (state, xl, zl)
        else:
            zl = st.expectation_value(ops["ZL1"]) * st.expectation_value(ops["ZL2"])
            xl = abs(st.expectation_value(ops["XL1"]))
            assert abs(zl - 1) < 1e-9 and xl < 1e-9, (state, zl, xl)
        print(f"  {r}x{s} |{'+'if state=='plus' else '0'}>_L: "
              f"{len(gens)} gens, all S=+1, logical OK "
              f"({'X_L=+1,Z_L uniform' if state=='plus' else 'Z_L=+1,X_L uniform'})")


if __name__ == "__main__":
    for (r, s) in ((4, 4), (4, 8)):
        verify(r, s)
    print("encoded-ancilla prep verified from the code's own operators.")