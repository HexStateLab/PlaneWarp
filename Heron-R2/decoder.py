"""
decoder.py — Core decoder for the (1+x²)(1+y²) code.

Provides:
  - tesseract_decode_ffinal(syndromes, r, s)  — ffinal decoder (no AND-vote)
  - tesseract_decode_rot(syndromes, r, s)    — ffinal + best single rotation
  - tesseract_decode_rot4(syndromes, r, s)   — all 4 rotations stacked
  - decode_np(syn, r, s)     — subprocess: plane_warp --decode-np
  - prep(syn, r, s)          — C library preprocess_syndrome wrapper
  - solve(syn, r, s)         — C library solve_plane + min-weight kernel
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
_lib.syndrome_of.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
_lib.syndrome_of.restype = None
_lib.is_stabilizer.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8)]
_lib.is_stabilizer.restype = _ct.c_int

# Subprocess decode — flat-view decoder (rebuild Jun 2026, includes
# solve_plane_flat for beyond-distance performance).
import subprocess as _sp
_BIN = _os.path.join(_lib_dir, 'plane_warp')

# Set True to use the flat-view decoder (global K_fix, optimal for odd
# block dimensions and concentrated error weight beyond code distance).
FLAT_DECODER = True

def _sub_decode(syn, r, s, timeout=30):
    """Call ./plane_warp [--flat] --decode-np via subprocess."""
    args = [_BIN, str(r), str(s)]
    if FLAT_DECODER:
        args.append('--flat')
    args.append('--decode-np')
    proc = _sp.run(args, input=syn.tobytes(), capture_output=True, timeout=timeout)
    return np.frombuffer(proc.stdout, np.uint8).reshape(r, s)
_lib.rot_4d_fwd.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8),
    _ct.c_int, _ct.c_int, _ct.c_int]
_lib.rot_4d_fwd.restype = None
_lib.rot_4d_inv.argtypes = [_ct.c_int, _ct.c_int,
    _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8),
    _ct.c_int, _ct.c_int, _ct.c_int]
_lib.rot_4d_inv.restype = None

# ---- Translation helpers (origin-shift on the torus) ----
# Grid-matched rotation params: diverse shifts + SL(2,Z) modes
# Generated dynamically via rotation_grid() for each (r,s).
USE_ROTATION = True

def translation_grid(r, s):
    """Generate diverse translations for given grid size."""
    r4 = max(1, r // 4)
    s4 = max(1, s // 4)
    shifts = [
        (0, 0), (r4, 0), (0, s4), (r4, s4),
        (r4*2, s4*2), (r4*3, s4*3), (r//2, 0), (0, s//2),
    ]
    return [(dx % r, dy % s) for dx, dy in shifts]

def default_rot(r, s):
    """Best single-translation params for given grid size."""
    return (0, max(1, r // 4), 0)

def _decode_with_rot(syn, r, s, rot=None):
    """Decode with optional translation. If rot=None and USE_ROTATION, applies default."""
    if USE_ROTATION and rot is None:
        rot = default_rot(r, s)
    if rot:
        dx, dy, mi = rot
        syn_t = translate_syn(syn, dx, dy)
        corr_t = solve(syn_t, r, s)
        return untranslate_corr(corr_t, dx, dy)
    return solve(syn, r, s)

def flatness_translation(syn, r, s):
    """Pick translation that puts syndrome in geometrically flat zone.
    The decoder's K_fix column (block col hs/2) is a 'curved' region —
    syndrome there is harder to correct.  The seed column (block col 0)
    is 'flat' — the decoder has maximum freedom.
    Score each candidate dy by how much syndrome weight falls in
    flat (b=0) vs curved (b=1) block columns.  Row shifts (dx)
    don't affect column flatness so we fix dx=0.
    """
    hs = s // 2
    khs = hs // 2  # K_fix column index
    best_dy = 0
    best_score = -1e9
    for dy in range(s):
        score = 0
        for j in range(s):
            wt = int(syn[:, j].sum())
            if wt == 0:
                continue
            b = ((j - dy) % s) // 2
            if b < khs:
                score += wt     # flat: seed column
            elif b == khs:
                score -= wt     # curved: K_fix column
        if score > best_score:
            best_score = score
            best_dy = dy
    return (0, best_dy)

def decode_tracking(syn, r, s):
    """Decode with geometric-flatness-guided translation."""
    dx, dy = flatness_translation(syn, r, s)
    syn_t = translate_syn(syn, dx, dy)
    corr_t = solve(syn_t, r, s)
    return untranslate_corr(corr_t, dx, dy)

def translate_syn(syn, dx, dy):
    """C-backed translation: syndrome → shifted syndrome."""
    out = np.zeros_like(syn)
    r, s = syn.shape
    _lib.rot_4d_fwd(r, s,
        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
        out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)), dx, dy, 0)
    return out

def untranslate_corr(corr, dx, dy):
    """C-backed inverse translation: correction → unshifted."""
    out = np.zeros_like(corr)
    r, s = corr.shape
    _lib.rot_4d_inv(r, s,
        corr.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
        out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)), dx, dy, 0)
    return out

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
    """Decode via subprocess --decode-np (uses working Jun 27 binary)."""
    return _sub_decode(syn, r, s)


def decode_np(syn, r, s, timeout=30):
    """Subprocess-based decode — alias for solve()."""
    return _sub_decode(syn, r, s)


def S_of(E, r, s):
    out = np.zeros((r, s), dtype=np.uint8)
    _lib.syndrome_of(r, s,
        E.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
        out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
    return out


def check_logical(corr, r, s):
    """Returns True if corr is a stabilizer (no logical error).
    Checks all 4 parity sectors, not just row0/col0."""
    return bool(_lib.is_stabilizer(r, s,
        corr.ctypes.data_as(_ct.POINTER(_ct.c_uint8))))


def tesseract_decode_ffinal(syndromes, r, s):
    """Decode ffinal: last-round syndrome, auto-rotation applied by default."""
    syn = syndromes[-1].copy().astype(np.uint8)
    return _decode_with_rot(syn, r, s)


def tesseract_decode_sequential(syndromes, r, s):
    """Sequential decode: each round decodes in a frame shifted by the
    previous round's correction centroid.  Errors accumulate; the frame
    tracks them round by round, keeping the decoder's origin near where
    the error cluster is concentrated.
    """
    nr = syndromes.shape[0]
    acc = np.zeros((r, s), dtype=np.uint8)
    for t in range(nr):
        syn = syndromes[t].copy().astype(np.uint8)
        if acc.sum() > 0:
            ys, xs = np.nonzero(acc)
            ci = float(ys.mean())
            cj = float(xs.mean())
            di = int(ci + 0.5) % r
            dj = int(cj + 0.5) % s
            dx, dy = (-di) % r, (-dj) % s
            syn_t = translate_syn(syn, dx, dy)
            corr_t = solve(syn_t, r, s)
            acc ^= untranslate_corr(corr_t, dx, dy)
        else:
            corr = solve(syn, r, s)
            acc ^= corr
    return acc


def tesseract_decode_tracking(syndromes, r, s):
    """Decode with adaptive tracking: centroid-based translation from
    the accumulated syndrome across all rounds.
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    return decode_tracking(syn, r, s)


def tesseract_decode_rot(syndromes, r, s, mi=3, dx=33, dy=33):
    """ffinal decoder with explicit 4D rotation (bypasses auto-rotation default)."""
    syn = syndromes[-1].copy().astype(np.uint8)
    return _decode_with_rot(syn, r, s, rot=(dx, dy, mi))


def tesseract_decode_rotn(syndromes, r, s):
    """Union decode: try all grid translations, pick lowest-weight valid.
    Use this for best error-correction quality at the cost of N× runtime.
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    return decode_union(syn, r, s)


def tesseract_decode_rot4(syndromes, r, s):
    """Run all grid translations, return corrections stacked (N,r,s).
    For external union-rate evaluation; not for production single-shot.
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    out = []
    for dx, dy in translation_grid(r, s):
        syn_t = translate_syn(syn, dx, dy)
        prep(syn_t, r, s)
        corr_t = solve(syn_t, r, s)
        out.append(untranslate_corr(corr_t, dx, dy))
    return np.stack(out)


# Alias for backward compatibility
tesseract_decode_union = tesseract_decode_rotn


def tesseract_decode_np(syndromes, r, s, timeout=30):
    """Subprocess decoder with auto-rotation."""
    syn = syndromes[-1].copy().astype(np.uint8)
    return _decode_with_rot(syn, r, s)


def tesseract_decode_np_rot4(syndromes, r, s, timeout=30):
    """N-translation ANY-of-N pipeline using subprocess --decode-np.
    Returns stacked (N,r,s) corrections.
    """
    syn = syndromes[-1].copy().astype(np.uint8)
    out = []
    for dx, dy in translation_grid(r, s):
        syn_t = translate_syn(syn, dx, dy)
        corr_t = decode_np(syn_t, r, s, timeout)
        out.append(untranslate_corr(corr_t, dx, dy))
    return np.stack(out)


def tesseract_decode(syndromes, r, s):
    """AND-vote + viability + solve + auto-rotation."""
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

    syn = syn_and.copy() if viable else syndromes[-1].copy()
    return _decode_with_rot(syn, r, s)
