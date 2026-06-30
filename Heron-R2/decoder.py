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


def tesseract_decode_ffinal(syndromes, r, s):
    """Decode ffinal: use LAST ROUND syndrome directly (skip AND-vote)."""
    syn = syndromes[-1].copy().astype(np.uint8)
    prep(syn, r, s)
    return solve(syn, r, s)


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
