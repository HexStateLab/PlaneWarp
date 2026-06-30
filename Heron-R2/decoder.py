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

# Expose C global knobs
def set_singleshot(val):
    ptr = _ct.c_int.in_dll(_lib, "g_singleshot")
    ptr.value = 1 if val else 0

def get_singleshot():
    ptr = _ct.c_int.in_dll(_lib, "g_singleshot")
    return ptr.value

def set_weight_cap(cap):
    ptr = _ct.c_int.in_dll(_lib, "g_weight_cap")
    ptr.value = cap

def set_cap_auto_rate(rate):
    ptr = _ct.c_double.in_dll(_lib, "g_cap_auto_rate")
    ptr.value = rate

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
        # Only enumerate STABILIZER kernel elements: row-0 and column-0 never flipped.
        for rmask in range(0, 1 << hr, 2):      # bit 0 (row 0) pinned
            for cmask in range(0, 1 << hs, 2):  # bit 0 (col 0) pinned
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
    """Min-weight stabilizer-equivalent correction, preserving logical state."""
    hr, hs = r // 2, s // 2
    lut = _get_lut(r, s)
    # The LUT now preserves row-0 and col-0 parity, so the logical state is fixed.
    # Just apply it to each sub-block.
    cur = corr.copy()
    for px in range(2):
        for py in range(2):
            sl = cur[px::2, py::2]
            idx = 0
            for ri in range(hr):
                for ci in range(hs):
                    if sl[ri, ci]:
                        idx |= 1 << (ri * hs + ci)
            cur[px::2, py::2] = lut[idx]
    return cur


def prep(syn, r, s):
    """Preprocess syndrome (DEPOLARIZE2 repair). Call explicitly if needed."""
    _lib.preprocess_syndrome(r, s,
        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))


def solve(syn, r, s, preprocess=False):
    """Decode syndrome using solve_plane_layered + min-weight kernel.
    
    Args:
        preprocess: if True, run preprocess_syndrome first (DEPOLARIZE2 repair).
                    Default False — hardware syndromes from CX circuits don't need it.
    """
    if preprocess:
        prep(syn, r, s)
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


def tesseract_decode_ffinal(syndromes, r, s, preprocess=False):
    """Decode ffinal: use LAST ROUND syndrome directly (skip AND-vote)."""
    syn = syndromes[-1].copy().astype(np.uint8)
    return solve(syn, r, s, preprocess=preprocess)


def tesseract_decode(syndromes, r, s, preprocess=False):
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

    return solve(syn, r, s, preprocess=preprocess)


class StaticCircuitsDecoder:
    """Emulate Dynamic Circuits in software from accumulated-ancilla data.
    
    Uses joint decoding across rounds with measurement-noise tolerance.
    Precomputes all weight-1 and weight-2 corrections for fast table lookup.
    
    In accumulated-ancilla mode, ancillas are never reset between rounds,
    so the cumulative measurement at the end gives only the LAST round's
    syndrome: A = S(E_n).  Earlier rounds' contributions cancel.
    
    Joint decoding:
      Find (C1, C2) minimizing:
        Hamming(A_measured, S(C2)) + Hamming(S_data_measured, S(C1⊕C2))
        + λ·(weight(C1) + weight(C2))
    
    The Hamming terms tolerate measurement noise (~3% per bit) on both the
    ancilla readout (A) and the data readout (S_data).  The sequential
    decoder doubles noise by computing S1 = S_data ⊕ A_measured.
    """

    def __init__(self, r, s):
        self.r = r
        self.s = s
        n = r * s
        self.n = n

        # Generate all weight-1 and weight-2 corrections
        corrections = []
        for i in range(n):
            c = np.zeros((r, s), dtype=np.uint8)
            c[i // s, i % s] = 1
            corrections.append((c.copy(), 1))
        for i in range(n):
            for j in range(i + 1, n):
                c = np.zeros((r, s), dtype=np.uint8)
                c[i // s, i % s] = 1
                c[j // s, j % s] = 1
                corrections.append((c.copy(), 2))
        self.corrections = corrections
        self.n_corr = len(corrections)

        # Precompute syndrome for each correction
        self.syndromes = np.zeros((self.n_corr, r, s), dtype=np.uint8)
        for idx, (c, _) in enumerate(corrections):
            self.syndromes[idx] = S_of(c, r, s)

        # Build LUT: syndrome tuple → (correction_index, weight)
        self.lut = {}
        for idx, (c, w) in enumerate(corrections):
            key = tuple(self.syndromes[idx].ravel())
            if key not in self.lut or w < self.lut[key][1]:
                self.lut[key] = (c.copy(), w)

    def decode_shots(self, A_all, data_raw_all, rounds=2, lam=0.5, k_candidates=200):
        """Decode multiple shots.

        Args:
            A_all: (n_shots, r, s) uint8 — accumulated syndromes
            data_raw_all: (n_shots, r, s) uint8 — final data readout
            rounds: number of QEC rounds
            lam: weight penalty (default 0.5)
            k_candidates: top-K C2 candidates to try per shot

        Returns:
            (n_shots, r, s) uint8 — total correction
        """
        n_shots = A_all.shape[0]
        C_total = np.zeros((n_shots, self.r, self.s), dtype=np.uint8)
        for shot in range(n_shots):
            A = A_all[shot]
            D = data_raw_all[shot]
            S_data = S_of(D, self.r, self.s)
            C_total[shot] = self._decode_shot(A, S_data, rounds, lam, k_candidates)
        return C_total

    def _decode_shot(self, A, S_data, rounds, lam, k_candidates):
        """Decode a single shot using joint optimization."""
        r, s, n = self.r, self.s, self.n
        best_score = float('inf')
        best_C1 = None
        best_C2 = None

        # Compute Hamming distance from A to each correction's syndrome
        # A.ravel() is n bits, each syndrome is n bits
        A_flat = A.ravel().astype(np.int8)
        syn_flat = self.syndromes.reshape(self.n_corr, n).astype(np.int8)
        hamming_A = (syn_flat ^ A_flat).sum(axis=1)

        # Top-K C2 candidates
        cand_idx = np.argpartition(hamming_A, k_candidates)[:k_candidates]
        cand_order = np.argsort(hamming_A[cand_idx])
        cand_idx = cand_idx[cand_order]

        S_data_flat = S_data.ravel().astype(np.int8)

        for idx2 in cand_idx:
            C2, w2 = self.corrections[idx2]
            S_C2 = self.syndromes[idx2]
            err_A = hamming_A[idx2]

            # Expected round-1 syndrome
            S1 = S_data ^ S_C2

            # Look up C1 from LUT
            key = tuple(S1.ravel())
            if key not in self.lut:
                C1 = solve(S1, r, s)
                w1 = C1.sum()
            else:
                C1, w1 = self.lut[key]
                C1 = C1.copy()

            # Compute consistency with S_data
            S_check = S_of(C1, r, s) ^ S_C2
            err_S = int((S_check.ravel() ^ S_data_flat).sum())

            score = err_A + err_S + lam * (w1 + w2)

            if score < best_score:
                best_score = score
                best_C1 = C1
                best_C2 = C2

        if best_C1 is None or best_C2 is None:
            return np.zeros((r, s), dtype=np.uint8)
        return best_C1 ^ best_C2


# Cache decoders per grid size
_static_decoders = {}

def static_circuits_decode(A, data_raw, r, s, rounds=2):
    """Emulate Dynamic Circuits in software from accumulated-ancilla data.
    
    Uses joint decoding across rounds with measurement-noise tolerance.
    The accumulated ancilla gives A = S(E_n) (last round only).  Combined
    with free-final-round data S(D) = S(E₁⊕...⊕Eₙ), we find the minimum-
    weight pair (C₁, C₂) that best explains both noisy measurements.

    Args:
        A: (n_shots, r, s) uint8 — accumulated syndrome (endpoint ancilla)
        data_raw: (n_shots, r, s) uint8 — final data readout
        r, s: grid dimensions
        rounds: number of QEC rounds

    Returns:
        (n_shots, r, s) uint8 — total correction
    """
    key = (r, s)
    if key not in _static_decoders:
        _static_decoders[key] = StaticCircuitsDecoder(r, s)
    dec = _static_decoders[key]
    return dec.decode_shots(A, data_raw, rounds=rounds)
