"""
waxis_decode.py — Persistent basis decoder.

Basis accumulates across shots. Each shot injects verified (syn, corr) pairs.
Once basis spans all of Im(H), every decode uses basis-decompose directly.
"""

import numpy as np
import os, ctypes as _ct


# Persistent basis cache (shared across all decoder instances)
_basis_cache = {}  # key: (r, s), value: (basis_syn, basis_corr, pivots, rank)


class WaxisDecoder:
    def __init__(self, r, s):
        self.r = r; self.s = s; self.n = r * s
        self._H = None
        self._load_c_lib()
        # Load persistent basis if available
        key = (r, s)
        if key in _basis_cache:
            self._basis_syn, self._basis_corr, self._pivots, self._basis_rank = _basis_cache[key]
        else:
            self._basis_syn = None
            self._basis_corr = None
            self._pivots = None
            self._basis_rank = 0

    def _load_c_lib(self):
        _lib_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(_lib_dir, "libplane_warp.so")
        self._lib = _ct.CDLL(lib_path)
        self._lib.solve_plane.argtypes = [_ct.c_int, _ct.c_int,
            _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
        self._lib.solve_plane.restype = _ct.c_int
        self._lib.solve_plane_layered.argtypes = [_ct.c_int, _ct.c_int,
            _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
        self._lib.solve_plane_layered.restype = _ct.c_int
        self._lib.preprocess_syndrome.argtypes = [_ct.c_int, _ct.c_int,
            _ct.POINTER(_ct.c_uint8)]
        self._lib.preprocess_syndrome.restype = None
        self._lib.syndrome_of.argtypes = [_ct.c_int, _ct.c_int,
            _ct.POINTER(_ct.c_uint8), _ct.POINTER(_ct.c_uint8)]
        self._lib.syndrome_of.restype = None

    def _syn_of(self, corr):
        syn = np.zeros((self.r, self.s), dtype=np.uint8)
        self._lib.syndrome_of(self.r, self.s,
            corr.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
            syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
        return syn.reshape(-1)

    def _preprocess(self, syn):
        self._lib.preprocess_syndrome(self.r, self.s,
            syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))

    def _solve(self, syn):
        out = np.zeros((self.r, self.s), dtype=np.uint8)
        self._lib.solve_plane_layered(self.r, self.s,
            syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
            out.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
        return out

    def _forward_eliminate(self, ws, wc):
        """RREF forward-eliminate. Modifies ws, wc in-place."""
        n = ws.shape[0]; nn = self.n
        pivot = [-1] * n
        for b in range(n):
            for q in range(nn):
                if ws[b, q]:
                    pivot[b] = q; break
            if pivot[b] < 0: continue
            for b2 in range(b + 1, n):
                if ws[b2, pivot[b]]:
                    ws[b2] ^= ws[b]; wc[b2] ^= wc[b]
        return pivot

    def _decompose(self, syn):
        """Decompose syn using RREF basis. Returns correction or None."""
        if self._pivots is None: return None
        nn = self.n; n = len(self._pivots)
        temp = syn.copy()
        coeffs = np.zeros(n, dtype=np.uint8)
        for b in range(n):
            p = self._pivots[b]
            if p >= 0 and temp[p]:
                temp ^= self._basis_syn[b]; coeffs[b] = 1
        if temp.any(): return None
        out = np.zeros(nn, dtype=np.uint8)
        for b in range(n):
            if coeffs[b]: out ^= self._basis_corr[b]
        return out

    def _inject_pair(self, syn, corr):
        """Inject one verified (syn, corr) pair. Updates persistent cache."""
        nn = self.n
        if self._basis_syn is None:
            self._basis_syn = syn.reshape(1, nn).copy()
            self._basis_corr = corr.reshape(1, nn).copy()
        else:
            self._basis_syn = np.vstack([self._basis_syn, syn.reshape(1, nn)])
            self._basis_corr = np.vstack([self._basis_corr, corr.reshape(1, nn)])
        ws = self._basis_syn.copy(); wc = self._basis_corr.copy()
        self._pivots = self._forward_eliminate(ws, wc)
        self._basis_syn = ws; self._basis_corr = wc
        self._basis_rank = sum(1 for p in self._pivots if p >= 0)
        # Update persistent cache
        _basis_cache[(self.r, self.s)] = (self._basis_syn, self._basis_corr, self._pivots, self._basis_rank)

    def decode(self, syndromes):
        """Persistent basis decoder.

        If basis spans Im(H), decompose directly.
        Otherwise, decode each round, verify, inject into basis.
        """
        rr, r, s, nn = syndromes.shape[0], self.r, self.s, self.n
        rounds_data = syndromes.reshape(rr, nn).astype(np.uint8)

        # If basis is full, decompose consensus directly
        if self._basis_rank >= 24:  # dim(Im(H)) for 6×6
            majority = (rounds_data.sum(axis=0) > rr // 2).astype(np.uint8)
            corr = self._decompose(majority)
            if corr is not None:
                return corr.reshape(r, s)

        # Otherwise, decode each round and inject verified pairs
        round_corrs = []
        for c in range(rr):
            syn = rounds_data[c].copy().reshape(r, s)
            self._preprocess(syn)
            corr = self._solve(syn)
            round_corrs.append(corr.reshape(-1))

            # Verify: syndrome_of(corr) == syn
            chk = self._syn_of(corr)
            if (chk == rounds_data[c]).all():
                self._inject_pair(rounds_data[c], corr.reshape(-1))

        # Try basis-decompose after injection
        if self._basis_rank >= 3:
            majority = (rounds_data.sum(axis=0) > rr // 2).astype(np.uint8)
            corr = self._decompose(majority)
            if corr is not None:
                return corr.reshape(r, s)

        # Fallback: consensus vote
        consensus = np.zeros(nn, dtype=np.uint8)
        for q in range(nn):
            cnt = sum(1 for rc in round_corrs if rc[q])
            if cnt * 3 > rr: consensus[q] = 1
        return consensus.reshape(r, s)
