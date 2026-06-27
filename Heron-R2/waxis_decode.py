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
            # Initialize with weight-1 error syndromes (span Im(H))
            self._init_basis_from_weight1()

    def _init_basis_from_weight1(self):
        """Initialize basis with weight-1, weight-2, weight-3, then random errors until full."""
        nn = self.n; r, s = self.r, self.s
        target_rank = 24  # dim(Im(H)) for 6×6

        # Weight-1 syndromes
        syns = np.zeros((nn, nn), dtype=np.uint8)
        corrs = np.eye(nn, dtype=np.uint8)
        err = np.zeros((r, s), dtype=np.uint8)
        for qi in range(r):
            for qj in range(s):
                err.fill(0); err[qi, qj] = 1
                syn = np.zeros((r, s), dtype=np.uint8)
                self._lib.syndrome_of(r, s,
                    err.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                    syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
                syns[qi*s+qj] = syn.reshape(-1)
        self._inject_basis(syns, corrs)

        # Weight-2 syndromes
        if self._basis_rank < target_rank:
            for qi1 in range(r):
                for qj1 in range(s):
                    for qi2 in range(r):
                        for qj2 in range(s):
                            if qi1*s+qj1 >= qi2*s+qj2: continue
                            err.fill(0); err[qi1, qj1] = 1; err[qi2, qj2] = 1
                            syn = np.zeros((r, s), dtype=np.uint8)
                            self._lib.syndrome_of(r, s,
                                err.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                                syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
                            syn_flat = syn.reshape(-1)
                            _, residual = self._decompose_with_residual(syn_flat)
                            if residual.any():
                                self._inject_pair(syn_flat, err.reshape(-1))
                                if self._basis_rank >= target_rank:
                                    return

        # Weight-3 syndromes
        if self._basis_rank < target_rank:
            from itertools import combinations
            positions = [(qi, qj) for qi in range(r) for qj in range(s)]
            for combo in combinations(positions, 3):
                err.fill(0)
                for qi, qj in combo:
                    err[qi, qj] = 1
                syn = np.zeros((r, s), dtype=np.uint8)
                self._lib.syndrome_of(r, s,
                    err.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                    syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
                syn_flat = syn.reshape(-1)
                _, residual = self._decompose_with_residual(syn_flat)
                if residual.any():
                    self._inject_pair(syn_flat, err.reshape(-1))
                    if self._basis_rank >= target_rank:
                        return

        # Random errors of increasing weight until full
        if self._basis_rank < target_rank:
            rng = np.random.default_rng(42)
            for weight in range(4, nn):
                for _ in range(1000):
                    E = (rng.random(nn) < weight/nn).astype(np.uint8)
                    if E.sum() < weight: continue
                    syn = np.zeros((r, s), dtype=np.uint8)
                    self._lib.syndrome_of(r, s,
                        E.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
                    syn_flat = syn.reshape(-1)
                    if not syn_flat.any(): continue
                    _, residual = self._decompose_with_residual(syn_flat)
                    if residual.any():
                        self._inject_pair(syn_flat, E)
                        if self._basis_rank >= target_rank:
                            return

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

    def _decompose_with_residual(self, syn):
        """Decompose syn into basis. Returns (correction, residual).
        Residual is the part of syn not in the basis span."""
        if self._pivots is None: return None, syn.copy()
        nn = self.n; n = len(self._pivots)
        temp = syn.copy()
        coeffs = np.zeros(n, dtype=np.uint8)
        for b in range(n):
            p = self._pivots[b]
            if p >= 0 and temp[p]:
                temp ^= self._basis_syn[b]; coeffs[b] = 1
        residual = temp.copy()
        out = np.zeros(nn, dtype=np.uint8)
        for b in range(n):
            if coeffs[b]: out ^= self._basis_corr[b]
        if not residual.any():
            return out, residual
        return None, residual

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

    def _inject_basis(self, syns, corrs):
        """Inject multiple (syn, corr) pairs at once."""
        self._basis_syn = syns.copy()
        self._basis_corr = corrs.copy()
        ws = self._basis_syn.copy(); wc = self._basis_corr.copy()
        self._pivots = self._forward_eliminate(ws, wc)
        self._basis_syn = ws; self._basis_corr = wc
        self._basis_rank = sum(1 for p in self._pivots if p >= 0)
        _basis_cache[(self.r, self.s)] = (self._basis_syn, self._basis_corr, self._pivots, self._basis_rank)

    def decode(self, syndromes):
        """Persistent basis decoder.

        If basis spans Im(H), decompose directly.
        Otherwise, use basis to decode if possible, verify, inject.
        Fall back to project-decode or consensus.
        """
        rr, r, s, nn = syndromes.shape[0], self.r, self.s, self.n
        rounds_data = syndromes.reshape(rr, nn).astype(np.uint8)

        # If basis is full, decompose consensus directly
        if self._basis_rank >= 24:  # dim(Im(H)) for 6×6
            majority = (rounds_data.sum(axis=0) > rr // 2).astype(np.uint8)
            corr = self._decompose(majority)
            if corr is not None:
                return corr.reshape(r, s)

        # Try basis-decompose each round
        round_corrs = []
        for c in range(rr):
            syn = rounds_data[c]
            if self._basis_rank >= 3:
                corr, residual = self._decompose_with_residual(syn)
                if corr is not None:
                    # Fully decomposed! Verify and inject
                    chk = self._syn_of(corr.reshape(r, s))
                    if (chk == syn).all():
                        self._inject_pair(syn, corr)
                        round_corrs.append(corr)
                        continue

            # Basis-decompose failed or not available, use solver
            syn_2d = syn.copy().reshape(r, s)
            self._preprocess(syn_2d)
            corr = self._solve(syn_2d)
            round_corrs.append(corr.reshape(-1))

            # Inject solver correction: syndrome_of(corr) is in Im(H) by construction
            corr_syn = self._syn_of(corr)
            # Check if linearly independent of current basis
            if self._basis_rank >= 3:
                test_corr, test_res = self._decompose_with_residual(corr_syn)
                if test_res.any():
                    # New info! Inject it
                    self._inject_pair(corr_syn, corr.reshape(-1))
            else:
                # Basis too small, inject anyway
                self._inject_pair(corr_syn, corr.reshape(-1))

        # If basis is stuck, use project-decode to inject clean syndromes
        if self._basis_rank < 24 and self._basis_rank >= 3:
            clean_idx = []
            for q in range(nn):
                val = rounds_data[0, q]
                all_same = 1
                for rd in range(1, rr):
                    if rounds_data[rd, q] != val:
                        all_same = 0; break
                if all_same: clean_idx.append(q)

            if len(clean_idx) >= 12:  # Lower threshold to inject more
                if self._H is None:
                    self._H = np.zeros((nn, nn), dtype=np.uint8)
                    err = np.zeros(nn, dtype=np.uint8)
                    syn = np.zeros(nn, dtype=np.uint8)
                    for col in range(nn):
                        err.fill(0); err[col] = 1
                        self._lib.syndrome_of(r, s,
                            err.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                            syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
                        self._H[:, col] = syn

                A = np.zeros((len(clean_idx), nn + 1), dtype=np.uint8)
                for i, q in enumerate(clean_idx):
                    A[i, :nn] = self._H[q]
                    A[i, nn] = rounds_data[0, q]

                rank = 0; pivot_col = [-1] * len(clean_idx)
                for col in range(nn):
                    if rank >= len(clean_idx): break
                    pv = -1
                    for i in range(rank, len(clean_idx)):
                        if A[i, col]: pv = i; break
                    if pv < 0: continue
                    if pv != rank: A[[rank, pv]] = A[[pv, rank]]
                    pivot_col[rank] = col
                    for i in range(len(clean_idx)):
                        if i != rank and A[i, col]:
                            A[i] ^= A[rank]
                    rank += 1

                consistent = True
                for i in range(rank, len(clean_idx)):
                    if A[i, nn] and A[i, :nn].sum() == 0:
                        consistent = False; break

                if consistent:
                    E = np.zeros(nn, dtype=np.uint8)
                    for i in range(rank):
                        c = pivot_col[i]
                        if c >= 0: E[c] = A[i, nn]

                    proj_syn = np.zeros(nn, dtype=np.uint8)
                    self._lib.syndrome_of(r, s,
                        E.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                        proj_syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))

                    # Inject clean projected syndrome
                    chk = self._syn_of(E.reshape(r, s))
                    if (chk == proj_syn).all():
                        # Check if this is linearly independent of current basis
                        test_corr, test_res = self._decompose_with_residual(proj_syn)
                        if test_res.any():
                            # New info! Inject it
                            self._inject_pair(proj_syn, E)

        # Try basis-decompose consensus after injection
        if self._basis_rank >= 3:
            majority = (rounds_data.sum(axis=0) > rr // 2).astype(np.uint8)
            corr = self._decompose(majority)
            if corr is not None:
                return corr.reshape(r, s)

        # Fallback: project-decode
        clean_idx = []
        for q in range(nn):
            val = rounds_data[0, q]
            all_same = 1
            for rd in range(1, rr):
                if rounds_data[rd, q] != val:
                    all_same = 0; break
            if all_same: clean_idx.append(q)

        if len(clean_idx) >= 24:
            if self._H is None:
                self._H = np.zeros((nn, nn), dtype=np.uint8)
                err = np.zeros(nn, dtype=np.uint8)
                syn = np.zeros(nn, dtype=np.uint8)
                for col in range(nn):
                    err.fill(0); err[col] = 1
                    self._lib.syndrome_of(r, s,
                        err.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                        syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))
                    self._H[:, col] = syn

            A = np.zeros((len(clean_idx), nn + 1), dtype=np.uint8)
            for i, q in enumerate(clean_idx):
                A[i, :nn] = self._H[q]
                A[i, nn] = rounds_data[0, q]

            rank = 0; pivot_col = [-1] * len(clean_idx)
            for col in range(nn):
                if rank >= len(clean_idx): break
                pv = -1
                for i in range(rank, len(clean_idx)):
                    if A[i, col]: pv = i; break
                if pv < 0: continue
                if pv != rank: A[[rank, pv]] = A[[pv, rank]]
                pivot_col[rank] = col
                for i in range(len(clean_idx)):
                    if i != rank and A[i, col]:
                        A[i] ^= A[rank]
                rank += 1

            consistent = True
            for i in range(rank, len(clean_idx)):
                if A[i, nn] and A[i, :nn].sum() == 0:
                    consistent = False; break

            if consistent:
                E = np.zeros(nn, dtype=np.uint8)
                for i in range(rank):
                    c = pivot_col[i]
                    if c >= 0: E[c] = A[i, nn]

                proj_syn = np.zeros(nn, dtype=np.uint8)
                self._lib.syndrome_of(r, s,
                    E.ctypes.data_as(_ct.POINTER(_ct.c_uint8)),
                    proj_syn.ctypes.data_as(_ct.POINTER(_ct.c_uint8)))

                chk = self._syn_of(E.reshape(r, s))
                if (chk == proj_syn).all():
                    self._inject_pair(proj_syn, E)

                if self._basis_rank >= 3:
                    corr = self._decompose(proj_syn)
                    if corr is not None:
                        return corr.reshape(r, s)

        # Ultimate fallback: consensus vote
        consensus = np.zeros(nn, dtype=np.uint8)
        for q in range(nn):
            cnt = sum(1 for rc in round_corrs if rc[q])
            if cnt * 3 > rr: consensus[q] = 1
        return consensus.reshape(r, s)
