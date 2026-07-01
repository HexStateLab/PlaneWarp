"""
decoder.py — Core decoder for the (1+x²)(1+y²) code.

Provides:
  - tesseract_decode_ffinal(syndromes, r, s)  — ffinal decoder (no AND-vote)
  - prep(syn, r, s)          — C library preprocess_syndrome wrapper
  - solve(syn, r, s)         — C library solve_plane_layered + min-weight kernel
  - S_of(E, r, s)            — compute syndrome from error pattern
  - check_logical(corr, r, s) — logical Z values from correction

All functions take explicit (r, s) grid dimensions — no global side effects.
"""
import numpy as np
import ctypes as _ct
import os as _os

_lib_dir = _os.path.dirname(_os.path.abspath(__file__))
_lib = _ct.CDLL(_os.path.join(_lib_dir, "libplane_warp.so"))
_lib.preprocess_syndrome.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8)]
_lib.preprocess_syndrome.restype = None
_lib.solve_plane_layered.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
_lib.solve_plane_layered.restype = _ct.c_int
_lib.syndrome_of.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
_lib.syndrome_of.restype = None

# Cache for min-weight kernel LUT per grid size
_lut_cache = {}

def _get_lut(r, s):
    key = (r, s)
    if key in _lut_cache:
        return _lut_cache[key]
    hr, hs = r // 2, s // 2
    lut = np.zeros((1 << (hr * hs), hr, hs), dtype=np.uint8)
    for idx in range(1 << (hr * hs)):
        sl = np.zeros((hr, hs), dtype=np.uint8)
        for b in range(hr * hs):
            if idx & (1 << b):
                sl[b // hs, b % hs] = 1
        best = sl.copy()
        best_wt = sl.sum()
        for rmask in range(1 << hr):
            for cmask in range(1 << hs):
                temp = sl.copy()
                for ri in range(hr):
                    if rmask & (1 << ri):
                        temp[ri, :] ^= 1
                for ci in range(hs):
                    if cmask & (1 << ci):
                        temp[:, ci] ^= 1
                wt = temp.sum()
                if wt < best_wt:
                    best_wt = wt
                    best = temp.copy()
        lut[idx] = best
    _lut_cache[key] = lut
    return lut


def min_weight_kernel_fast(corr, r, s):
    hr, hs = r // 2, s // 2
    lut = _get_lut(r, s)
    best = corr.copy()
    best_wt = best.sum()
    for target_z1 in (0, 1):
        for target_z2 in (0, 1):
            cur = corr.copy()
            if cur[0, :].sum() % 2 != target_z1:
                cur[0, :] ^= 1
            if cur[:, 0].sum() % 2 != target_z2:
                cur[:, 0] ^= 1
            for px in range(2):
                for py in range(2):
                    sl = cur[px::2, py::2]
                    idx = 0
                    for ri in range(hr):
                        for ci in range(hs):
                            if sl[ri, ci]:
                                idx |= 1 << (ri * hs + ci)
                    cur[px::2, py::2] = lut[idx]
            wt = cur.sum()
            if wt < best_wt:
                best_wt = wt
                best = cur.copy()
    return best


def prep(syn, r, s):
    _lib.preprocess_syndrome(r, s,
        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))


def solve(syn, r, s):
    out = np.zeros((r, s), dtype=np.uint8)
    _lib.solve_plane_layered(r, s,
        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
        out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
    return min_weight_kernel_fast(out, r, s)


def S_of(E, r, s):
    out = np.zeros((r, s), dtype=np.uint8)
    _lib.syndrome_of(r, s,
        E.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
        out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
    return out


def check_logical(corr, r, s):
    return corr[0, :].sum() % 2, corr[:, 0].sum() % 2


def build_H_shear(r, s, k):
    """Build the V_k check matrix: V_k(i,j) = E(i,j) ⊕ E(i+2, j+2k mod s).
    
    Returns (N, N) array over GF(2), where N = r*s.
    Row = V_k(i,j) at index i*s + j.
    Column = E(i,j) at index i*s + j.
    
    E(i,j) is the FIRST qubit of V_k(i,j) and the SECOND qubit of V_k(i-2, j-2k).
    """
    N = r * s
    H = np.zeros((N, N), dtype=np.uint8)
    for qi in range(r):
        for qj in range(s):
            col = qi * s + qj
            # E(i,j) is first qubit of V_k(i,j): row (qi, qj)
            H[qi * s + qj, col] = 1
            # E(i,j) is second qubit of V_k(i-2, j-2k): row ((qi-2)%r, (qj-2*k)%s)
            H[((qi - 2) % r) * s + ((qj - 2 * k) % s), col] = 1
    return H


def gauss_elim(A_b):
    """Gaussian elimination over GF(2). Returns (E, consistent) where
    E is the solution vector or None if inconsistent.
    A_b: (rows, cols) augmented matrix where last column is RHS.
    """
    aug = A_b.copy()
    n_rows, n_cols = aug.shape
    pivot_row = 0
    pivot_cols = []
    
    for col in range(n_cols - 1):
        row = -1
        for r2 in range(pivot_row, n_rows):
            if aug[r2, col]:
                row = r2
                break
        if row < 0:
            continue
        if row != pivot_row:
            aug[[row, pivot_row]] = aug[[pivot_row, row]]
        pivot_cols.append(col)
        for r2 in range(n_rows):
            if r2 != pivot_row and aug[r2, col]:
                aug[r2] ^= aug[pivot_row]
        pivot_row += 1
    
    for r2 in range(pivot_row, n_rows):
        if aug[r2, -1] and not aug[r2, :-1].any():
            return None, False
    
    E = np.zeros(n_cols - 1, dtype=np.uint8)
    for pi, col in enumerate(pivot_cols):
        E[col] = aug[pi, -1]
    return E, True


def find_chains(r, s, k):
    """Find all chains in the shear-k system.
    
    Each chain traces (i,j) → (i+2, j+2k) → (i+4, j+4k) → ...
    until it cycles. Returns list of chains, each chain is a list
    of (i,j) positions in traversal order.
    """
    visited = np.zeros((r, s), dtype=bool)
    chains = []
    for si in range(r):
        for sj in range(s):
            if visited[si, sj]:
                continue
            chain = []
            i, j = si, sj
            while not visited[i, j]:
                visited[i, j] = True
                chain.append((i, j))
                i = (i + 2) % r
                j = (j + 2 * k) % s
            if len(chain) > 0:
                chains.append(chain)
    return chains


def solve_shear(V, r, s, k):
    """Solve V_k = H_k · E for E via chain-wise O(N) elimination.
    
    V_k(i,j) = E(i,j) + E(i+2, j+2k).
    
    For each chain (size L), tries both E[0]=0 and E[0]=1, propagates
    WITHOUT enforcing cycle closure (noise-robust). Chooses the
    lower-weight correction.
    
    Returns (r, s) correction array.
    """
    chains = find_chains(r, s, k)
    E = np.zeros((r, s), dtype=np.uint8)

    for chain in chains:
        L = len(chain)
        best_vals = None

        for start_bit in (0, 1):
            vals = np.zeros(L, dtype=np.uint8)
            vals[0] = start_bit
            for t in range(L - 1):
                ci, cj = chain[t]
                vals[t + 1] = vals[t] ^ V[ci, cj]

            # Option 1: use vals as-is (may violate last equation)
            # Option 2: flip the whole chain
            wt0 = vals.sum()
            wt1 = (L - wt0)
            if wt0 <= wt1:
                this_vals = vals.copy()
            else:
                this_vals = vals ^ 1

            if best_vals is None or this_vals.sum() < best_vals.sum():
                best_vals = this_vals.copy()

        for t, (ci, cj) in enumerate(chain):
            E[ci, cj] = best_vals[t]

    return E


def decode_shear(syndromes, r, s, k):
    """Decode 4-qubit syndrome in shear-k frame.
    
    Converts S_k to V_k, then solves via Gaussian elimination.
    syndromes: (1, r, s) or (r, s) array.
    """
    if syndromes.ndim == 3:
        S = syndromes[0].copy()
    else:
        S = syndromes.copy()
    # Recover V_k from S_k: need to invert S = V ^ roll(V, -2, cols)
    # This is a 1D problem per row: V'(t) + V'(t+2) = S'(t) for t=0..s-1
    # Solve via: V(0) = 0, V(2) = S(0), V(4) = S(2), ... 
    # V(1) = 0, V(3) = S(1), ...
    # Then for periodic: V(j) = V(j mod s) + partial sum of S
    V = np.zeros((r, s), dtype=np.uint8)
    for i in range(r):
        row = S[i]
        for parity in (0, 1):
            acc = 0
            for t in range(parity, s, 2):
                V[i, t] = acc
                acc ^= row[t]
            # Check consistency
            V[i, parity] ^= acc  # periodic closure
    return solve_shear(V, r, s, k)


def V_shear_of(E, r, s, k):
    """Compute V_k pair measurements from error E.
    V_k(i,j) = E(i,j) ⊕ E(i+2, j+2k).
    E: (..., r, s) array.
    """
    E_rolled = np.roll(E, shift=(-2, -2*k), axis=(-2, -1))
    return E ^ E_rolled


def syndrome_in_frame(E, r, s, k):
    """Compute 4-qubit syndrome in shear-k frame.
    S_k(i,j) = V_k(i,j) + V_k(i, j+2) where V_k is the sheared pair.
    E: (..., r, s) array.
    """
    V = V_shear_of(E, r, s, k)
    return V ^ np.roll(V, shift=-2, axis=-1)


def decode_shear_combined(syndromes_by_k, r, s):
    """Decode using multiple shear parameters via iterative refinement.
    
    syndromes_by_k: dict {k: S_k_array}, each S_k_array is (shots, r, s)
    in the shear-k 4-qubit syndrome frame.
    
    1. Decode first shear → correction
    2. Compute residual for next shear → refine
    3. Repeat
    
    Returns (shots, r, s) correction.
    """
    shears = sorted(syndromes_by_k.keys(), key=lambda k: -abs(k))
    n = next(iter(syndromes_by_k.values())).shape[0]
    C = np.zeros((n, r, s), dtype=np.uint8)
    
    for idx, k in enumerate(shears):
        S_k = syndromes_by_k[k].copy()
        if idx > 0:
            S_resid = S_k ^ syndrome_in_frame(C, r, s, k)
        else:
            S_resid = S_k
        
        for i in range(n):
            C_k = decode_shear(S_resid[i], r, s, k)
            C[i] ^= C_k
    
    return C


def tesseract_decode_ffinal(syndromes, r, s):
    """Decode ffinal: use LAST ROUND syndrome directly (skip AND-vote)."""
    syn = syndromes[-1].copy().astype(np.uint8)
    prep(syn, r, s)
    return solve(syn, r, s)


def solve_combined(V_vert, V_horiz, r, s):
    """Solve for error E given BOTH V_vert and V_horiz 2-qubit syndromes.

    Builds the combined 2N×N linear system over GF(2) and solves via
    Gaussian elimination. Returns minimum-weight correction.
    
    V_vert(i,j) = E(i,j) + E(i+2,j)  (vertical pair measurements)
    V_horiz(i,j) = E(i,j) + E(i,j+2) (horizontal pair measurements)

    Combined rank = 44/48 for 6×8 torus (vs 24/48 for 4-qubit S alone).
    """
    N = r * s
    nn = N  # number of variables

    # Build combined matrix H: 2N × N
    H = np.zeros((2 * N, nn), dtype=np.uint8)
    for qi in range(r):
        for qj in range(s):
            col = qi * s + qj
            # V_vert row
            H[qi * s + qj, col] = 1
            H[((qi + 2) % r) * s + qj, col] = 1
            # V_horiz row (offset by N)
            H[N + qi * s + qj, col] = 1
            H[N + qi * s + (qj + 2) % s, col] = 1

    # Build syndrome vector
    s_vec = np.concatenate([V_vert.reshape(-1), V_horiz.reshape(-1)])

    # Gaussian elimination over GF(2)
    aug = np.hstack([H, s_vec.reshape(-1, 1)])  # 2N × (N+1)
    n_rows, n_cols = aug.shape
    pivot_row = 0
    pivot_cols = []
    for col in range(n_cols - 1):
        row = -1
        for r2 in range(pivot_row, n_rows):
            if aug[r2, col]:
                row = r2
                break
        if row < 0:
            continue
        if row != pivot_row:
            aug[[row, pivot_row]] = aug[[pivot_row, row]]
        pivot_cols.append(col)
        for r2 in range(n_rows):
            if r2 != pivot_row and aug[r2, col]:
                aug[r2] ^= aug[pivot_row]
        pivot_row += 1

    # Check consistency
    consistent = True
    for r2 in range(pivot_row, n_rows):
        if aug[r2, -1] and not aug[r2, :-1].any():
            consistent = False
            break

    if not consistent:
        # Fallback: use V_vert only with standard decoder
        syn = V_vert ^ np.roll(V_vert, shift=-2, axis=1)
        prep(syn, r, s)
        return solve(syn, r, s)

    # Extract solution
    E = np.zeros(nn, dtype=np.uint8)
    for pi, col in enumerate(pivot_cols):
        E[col] = aug[pi, -1]

    return min_weight_kernel_fast(E.reshape(r, s), r, s)


def tesseract_decode(syndromes, r, s):
    """AND-vote + viability + solve decoder."""
    rr, hr, hs = syndromes.shape[0], r // 2, s // 2
    syn_and = np.ones((r, s), dtype=np.uint8)
    for t in range(rr):
        syn_and &= syndromes[t]

    viable = 1
    for px in range(2):
        for py in range(2):
            for si in range(hr):
                rp = 0
                for sj in range(hs):
                    rp ^= syn_and[(px + 2 * si) % r, (py + 2 * sj) % s]
                if rp:
                    viable = 0
                    break
            if not viable:
                break
        if not viable:
            break

    if viable:
        syn = syn_and.copy()
    else:
        syn = syndromes[-1].copy()

    prep(syn, r, s)
    return solve(syn, r, s)
