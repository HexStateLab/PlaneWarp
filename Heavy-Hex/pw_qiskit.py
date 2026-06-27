"""
pw_qiskit.py -- Plane-Warp decoder integration with Qiskit.

Provides:
  PlaneWarp   -- ctypes wrapper for libplane_warp.so
  QECBuilder  -- builds (1+x^2)(1+y^2) code circuits for Qiskit
  decode_run  -- single-call: build circuit -> run on backend -> decode -> return
"""

import ctypes
import os
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_lib_dir, "libplane_warp.so")
_lib = ctypes.CDLL(_lib_path)

# --- set argument / return types for each exported C function ---

# void preprocess_syndrome(int r, int s, uint8_t *syn)
_lib.preprocess_syndrome.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_uint8)]
_lib.preprocess_syndrome.restype = None

# void syndrome_of(int r, int s, uint8_t *err, uint8_t *syn)
_lib.syndrome_of.argtypes = [ctypes.c_int, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_uint8),
                             ctypes.POINTER(ctypes.c_uint8)]
_lib.syndrome_of.restype = None

# int solve_plane(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane.argtypes = [ctypes.c_int, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_uint8),
                             ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane.restype = ctypes.c_int

# int solve_plane_layered(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane_layered.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_uint8),
                                     ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane_layered.restype = ctypes.c_int

# void canonicalize(int r, int s, uint8_t *corr)
_lib.canonicalize.argtypes = [ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_uint8)]
_lib.canonicalize.restype = None

# int is_stabilizer(int r, int s, uint8_t *diff)
_lib.is_stabilizer.argtypes = [ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_uint8)]
_lib.is_stabilizer.restype = ctypes.c_int

# int decode_Z(int r, int s, uint8_t *err_z, uint8_t *dec_z)
_lib.decode_Z.argtypes = [ctypes.c_int, ctypes.c_int,
                          ctypes.POINTER(ctypes.c_uint8),
                          ctypes.POINTER(ctypes.c_uint8)]
_lib.decode_Z.restype = ctypes.c_int

# int solve_plane_fast(int r, int s, uint8_t *syn, uint8_t *out)
_lib.solve_plane_fast.argtypes = [ctypes.c_int, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_uint8),
                                  ctypes.POINTER(ctypes.c_uint8)]
_lib.solve_plane_fast.restype = ctypes.c_int


class PlaneWarp:
    """ctypes wrapper for the plane_warp decoder."""

    def __init__(self, fast=False, singleshot=True, escape=True,
                 weight_cap=0, cap_auto_rate=0.0):
        self.fast = fast
        try:
            g_fast = ctypes.c_int.in_dll(_lib, "g_fast")
            g_fast.value = 1 if fast else 0
        except ValueError:
            pass
        try:
            g_ss = ctypes.c_int.in_dll(_lib, "g_singleshot")
            g_ss.value = 1 if singleshot else 0
        except ValueError:
            pass
        try:
            g_esc = ctypes.c_int.in_dll(_lib, "g_escape_enabled")
            g_esc.value = 1 if escape else 0
        except ValueError:
            pass
        try:
            g_wc = ctypes.c_int.in_dll(_lib, "g_weight_cap")
            g_wc.value = weight_cap
        except ValueError:
            pass
        try:
            g_car = ctypes.c_double.in_dll(_lib, "g_cap_auto_rate")
            g_car.value = cap_auto_rate
        except ValueError:
            pass

    @staticmethod
    def _to_ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    def syndrome_of(self, error):
        r, s = error.shape
        syn = np.zeros((r, s), dtype=np.uint8)
        _lib.syndrome_of(r, s, self._to_ptr(error), self._to_ptr(syn))
        return syn

    def preprocess(self, syndrome):
        r, s = syndrome.shape
        _lib.preprocess_syndrome(r, s, self._to_ptr(syndrome))

    def decode(self, syndrome):
        r, s = syndrome.shape
        out = np.zeros((r, s), dtype=np.uint8)
        ok = _lib.solve_plane(r, s, self._to_ptr(syndrome), self._to_ptr(out))
        return out, bool(ok)

    def decode_layered(self, syndrome):
        r, s = syndrome.shape
        out = np.zeros((r, s), dtype=np.uint8)
        ok = _lib.solve_plane_layered(r, s, self._to_ptr(syndrome),
                                      self._to_ptr(out))
        return out, bool(ok)

    def decode_fast(self, syndrome):
        r, s = syndrome.shape
        out = np.zeros((r, s), dtype=np.uint8)
        ok = _lib.solve_plane_fast(r, s, self._to_ptr(syndrome),
                                   self._to_ptr(out))
        return out, bool(ok)

    def canonicalize(self, correction):
        r, s = correction.shape
        _lib.canonicalize(r, s, self._to_ptr(correction))

    def is_stabilizer(self, diff):
        r, s = diff.shape
        return bool(_lib.is_stabilizer(r, s, self._to_ptr(diff)))

    def decode_tesseract(self, syndromes):
        rounds, r, s = syndromes.shape
        if rounds == 1:
            corr, _ = self.decode_layered(syndromes[0])
            return corr
        n = r * s
        buf = struct.pack('<I', rounds)
        buf += syndromes.tobytes()
        import subprocess
        exe = os.path.join(_lib_dir, "plane_warp")
        flags = ["--fast"] if self.fast else []
        p = subprocess.run([exe, str(r), str(s)] + flags +
                           ["--decode-tesseract"],
                           input=buf, capture_output=True)
        if p.returncode != 0:
            print(f"WARNING: tesseract decoder failed (rc={p.returncode}, stderr={p.stderr.decode().strip()}), falling back to 2D on round 0", file=sys.stderr)
            corr, _ = self.decode_layered(syndromes[0])
            return corr
        corr = np.frombuffer(p.stdout, dtype=np.uint8).reshape(r, s)
        return corr

    def decode_linear_basis(self, syns, corrs, new_syn):
        """Linear basis decoder via the C binary.

        syns, corrs — lists of (r,s) numpy uint8 arrays (verified pairs).
                      Must be non-empty.
        new_syn     — (r,s) numpy uint8 array, the syndrome to decode.

        Returns (correction, ok) where ok is True if new_syn was in the span.
        """
        if not syns:
            return np.zeros_like(new_syn), False
        r, s = syns[0].shape
        nn = r * s
        n = len(syns)
        buf = struct.pack('<I', n)
        for k in range(n):
            buf += np.ascontiguousarray(syns[k], dtype=np.uint8).tobytes()
            buf += np.ascontiguousarray(corrs[k], dtype=np.uint8).tobytes()
        buf += np.ascontiguousarray(new_syn, dtype=np.uint8).tobytes()
        import subprocess
        exe = os.path.join(_lib_dir, "plane_warp")
        p = subprocess.run([exe, str(r), str(s), "--decode-basis"],
                           input=buf, capture_output=True)
        if len(p.stdout) != nn:
            return np.zeros((r, s), dtype=np.uint8), False
        corr = np.frombuffer(p.stdout, dtype=np.uint8).reshape(r, s)
        return corr, p.returncode == 0

    _STAB_GROUP = None

    def _build_stab_group(self, r=6, s=8):
        """Return (256, r, s) uint8 array of all stabilizer elements.

        Computes ker(H) ∩ stab_space via combined nullspace of the
        syndrome matrix H and the 28 topological stabilizer constraints.
        Enumerates all 256 linear combinations of the 8 generators.
        Cached after first call.
        """
        if PlaneWarp._STAB_GROUP is not None:
            return PlaneWarp._STAB_GROUP.reshape(-1, r, s)
        n = r * s
        # Build H
        H = np.zeros((n, n), dtype=np.uint8)
        for q in range(n):
            e = np.zeros((r, s), dtype=np.uint8)
            e.ravel()[q] = 1
            H[:, q] = self.syndrome_of(e).ravel()
        # Build stabilizer constraint matrix (28 checks)
        Cst = np.zeros((28, n), dtype=np.uint8)
        ci = 0
        for px in (0, 1):
            for py in (0, 1):
                hr2, hs2 = r // 2, s // 2
                for si in range(hr2):
                    row = np.zeros(n, dtype=np.uint8)
                    for sj in range(hs2):
                        row[(px + 2 * si) * s + (py + 2 * sj)] = 1
                    Cst[ci] = row; ci += 1
                for sj in range(hs2):
                    col = np.zeros(n, dtype=np.uint8)
                    for si in range(hr2):
                        col[(px + 2 * si) * s + (py + 2 * sj)] = 1
                    Cst[ci] = col; ci += 1
        # Combined nullspace: ker(H) ∩ stab_space
        A = np.vstack([H, Cst])
        m, nn = A.shape
        pivots = []
        for col in range(nn):
            nz = np.where(A[:, col])[0]
            if len(nz) == 0:
                continue
            pv = nz[0]
            if pv != len(pivots):
                A[[len(pivots), pv]] = A[[pv, len(pivots)]]
            pivots.append(col)
            pr = len(pivots) - 1
            for r2 in range(m):
                if r2 != pr and A[r2, col]:
                    A[r2] ^= A[pr]
        free = [c for c in range(nn) if c not in pivots]
        null = np.zeros((len(free), nn), dtype=np.uint8)
        for ki, fc in enumerate(free):
            v = np.zeros(nn, dtype=np.uint8)
            v[fc] = 1
            for pi, pc in enumerate(pivots):
                val = 0
                for fc2 in free:
                    if A[pi, fc2]:
                        val ^= v[fc2]
                v[pc] = val
            null[ki] = v
        # Enumerate all 256 linear combinations
        all_s = []
        for mask in range(256):
            s = np.zeros(n, dtype=np.uint8)
            for bit in range(null.shape[0]):
                if mask >> bit & 1:
                    s ^= null[bit]
            all_s.append(s)
        PlaneWarp._STAB_GROUP = np.array(all_s, dtype=np.uint8)
        return PlaneWarp._STAB_GROUP.reshape(-1, r, s)

    def refine_min_weight(self, C):
        """Return the minimum-weight correction coset-equivalent to C.

        Searches all 256 stabilizer elements S and returns C_min = C ⊕ S
        with the smallest Hamming weight. This is the ML-optimal correction
        for CSS codes under independent noise.
        """
        r, s = C.shape
        stab = self._build_stab_group(r, s)
        c_flat = C.ravel()
        best = c_flat.copy()
        best_wt = int(best.sum())
        for si in range(256):
            cand = c_flat ^ stab[si].ravel()
            w = int(cand.sum())
            if w < best_wt:
                best = cand
                best_wt = w
        return best.reshape(r, s)

    def project_decode(self, syns, corrs, rounds):
        """Consistency projection + linear basis decoder, all in C.

        Takes 4 rounds of raw syndrome measurements (each (r,s) uint8),
        projects onto Col(H) via consistent-bit Gaussian elimination,
        then decodes with the linear basis decoder.

        syns, corrs — lists of (r,s) numpy uint8 arrays (basis pairs).
        rounds      — (4, r, s) numpy uint8 array, one round per axis-0 entry.

        Returns (correction, ok) where ok=True if projection+decode succeeded.
        """
        if not syns or len(rounds) != 4:
            return np.zeros(rounds[0].shape, dtype=np.uint8), False
        r, s = syns[0].shape
        nn = r * s
        n = len(syns)
        buf = struct.pack('<I', n)
        for k in range(n):
            buf += np.ascontiguousarray(syns[k], dtype=np.uint8).tobytes()
            buf += np.ascontiguousarray(corrs[k], dtype=np.uint8).tobytes()
        for rd in range(4):
            buf += np.ascontiguousarray(rounds[rd], dtype=np.uint8).tobytes()
        import subprocess
        exe = os.path.join(_lib_dir, "plane_warp")
        p = subprocess.run([exe, str(r), str(s), "--project-decode"],
                           input=buf, capture_output=True)
        if len(p.stdout) != nn:
            return np.zeros((r, s), dtype=np.uint8), False
        corr = np.frombuffer(p.stdout, dtype=np.uint8).reshape(r, s)
        return corr, p.returncode == 0


# =========================================================================
# Qiskit circuit builder for the (1+x^2)(1+y^2) code
# =========================================================================

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.transpiler import CouplingMap
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


class QECBuilder:
    """Builds syndrome-extraction circuits for the (1+x^2)(1+y^2) toric code.

    Layout (r x s data qubits + r x s ancilla qubits):
      data[i,j]   = qubit at index  i*s + j          (0 ... N-1)
      anc[i,j]    = qubit at index  N + i*s + j       (N ... 2N-1)
    where N = r*s.
    """

    def __init__(self, r, s, code="z"):
        assert r >= 4 and s >= 4, "minimum 4x4 grid"
        assert r % 2 == 0 and s % 2 == 0, "even dimensions required"
        self.r = r
        self.s = s
        self.n = r * s
        self.code = code

    @property
    def num_qubits(self):
        return 2 * self.n

    def _di(self, i):
        return (i + 2) % self.r if i + 2 < self.r else i + 2 - self.r

    def _dj(self, j):
        return (j + 2) % self.s if j + 2 < self.s else j + 2 - self.s

    def _qdata(self, i, j):
        return i * self.s + j

    def _qanc(self, i, j):
        return self.n + i * self.s + j

    def build_rounds_circuit(self, rounds, barrier=True):
        """Syndrome extraction with CX(data, anc). Measures Z.Z.Z.Z."""
        qr = QuantumRegister(self.num_qubits, "q")
        crs = [ClassicalRegister(self.n, "syn_%d" % c) for c in range(rounds)]
        qc = QuantumCircuit(qr, *crs)
        for rnd in range(rounds):
            for i in range(self.r):
                for j in range(self.s):
                    a = self._qanc(i, j)
                    qc.cx(self._qdata(i, j), a)
                    qc.cx(self._qdata(self._di(i), j), a)
                    qc.cx(self._qdata(i, self._dj(j)), a)
                    qc.cx(self._qdata(self._di(i), self._dj(j)), a)
                    qc.measure(a, crs[rnd][i * self.s + j])
                    qc.reset(a)
            if barrier:
                qc.barrier()
        return qc

    def syndrome_from_counts(self, counts, rounds):
        n = self.n
        syndromes = np.zeros((rounds, self.r, self.s), dtype=np.uint8)
        for bitstring, cnt in counts.items():
            blocks = bitstring.split(' ')
            for c in range(rounds):
                blk = blocks[c]
                for q in range(n):
                    if blk[-(q + 1)] == '1':
                        syndromes[c, q // self.s, q % self.s] = 1
            break
        return syndromes


# =========================================================================
# Heavy-hex native embedding via flag-qubit syndrome extraction
# =========================================================================

def heavy_hex_flag_layout(r, s):
    """Generate a heavy-hex-native qubit layout for the (1+x^2)(1+y^2) code.

    Uses flag-qubit extraction: each stabilizer uses 2 ancillas (each deg 2)
    instead of 1 ancilla (deg 4). Maps to heavy-hex where:
      - Data qubits land on degree-4 nodes
      - Ancilla qubits land on degree-2 nodes between data pairs

    Returns (data_map, anc_maps, edges, total_q).
    """
    edges = set()
    n_data = r * s
    n_anc = 2 * r * s
    total_q = n_data + n_anc

    data_map = [[i * s + j for j in range(s)] for i in range(r)]
    anc_maps = {}
    idx = n_data
    for i in range(r):
        for j in range(s):
            for k in range(2):
                anc_maps[(i, j, k)] = idx
                idx += 1

    # Each ancilla connects to exactly 2 data qubits
    for i in range(r):
        for j in range(s):
            anc0 = anc_maps[(i, j, 0)]
            d1 = data_map[i][j]
            d2 = data_map[(i + 2) % r][j]
            edges.add((d1, anc0))
            edges.add((d2, anc0))

            anc1 = anc_maps[(i, j, 1)]
            d3 = data_map[i][(j + 2) % s]
            d4 = data_map[(i + 2) % r][(j + 2) % s]
            edges.add((d3, anc1))
            edges.add((d4, anc1))

    # No edge between ancillas: the XOR anc0 ^ anc1 is done classically
    # Each ancilla has degree exactly 2 (native on heavy-hex degree-2 nodes)

    return data_map, anc_maps, list(edges), total_q


def build_flag_circuit(r, s, rounds, data_map, anc_maps):
    """Build flag-qubit syndrome extraction circuit.

    Each stabilizer uses 2 ancillas (degree 2 each):
      anc0: CX from data (i,j), CX from data (i+2,j)  -> row pair
      anc1: CX from data (i,j+2), CX from data (i+2,j+2) -> col pair
    Stabilizer value = anc0 XOR anc1 (computed classically).
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    n_data = r * s
    n_anc = 2 * r * s
    total = n_data + n_anc

    qr = QuantumRegister(total, "q")
    crs = [ClassicalRegister(n_anc, "syn_%d" % c) for c in range(rounds)]
    qc = QuantumCircuit(qr, *crs)

    for rnd in range(rounds):
        for i in range(r):
            for j in range(s):
                anc0 = anc_maps[(i, j, 0)]
                anc1 = anc_maps[(i, j, 1)]

                qc.reset(anc0)
                qc.reset(anc1)

                # Row pair: anc0 connects to (i,j) and (i+2,j)
                qc.cx(data_map[i][j], anc0)
                qc.cx(data_map[(i + 2) % r][j], anc0)

                # Col pair: anc1 connects to (i,j+2) and (i+2,j+2)
                qc.cx(data_map[i][(j + 2) % s], anc1)
                qc.cx(data_map[(i + 2) % r][(j + 2) % s], anc1)

                qc.measure(anc0, crs[rnd][i * s * 2 + j * 2])
                qc.measure(anc1, crs[rnd][i * s * 2 + j * 2 + 1])

    return qc


def syndrome_from_flag_counts(counts, rounds, r, s):
    """Convert flag-qubit measurements to (rounds, r, s) syndromes.

    Stabilizer value = anc0 XOR anc1.
    """
    n_anc = 2 * r * s
    syndromes = np.zeros((rounds, r, s), dtype=np.uint8)
    for bitstring, cnt in counts.items():
        blocks = bitstring.split(' ')
        for c in range(rounds):
            blk = blocks[c]
            for q in range(n_anc // 2):
                anc0 = int(blk[-(2 * q + 1)]) if (2 * q + 1) <= len(blk) else 0
                anc1 = int(blk[-(2 * q + 2)]) if (2 * q + 2) <= len(blk) else 0
                i = q // s
                j = q % s
                syndromes[c, i, j] = anc0 ^ anc1
        break
    return syndromes


# =========================================================================
# High-level integration
# =========================================================================

def decode_run(r, s, rounds=5, shots=1, backend=None,
               transpile_to=None, use_flags=False,
               pw_kwargs=None, builder_kwargs=None):
    """One-call: build circuits, run, decode, return correction.

    Parameters
    ----------
    r, s : int -- grid dimensions (even, >=4)
    rounds : int -- number of QEC rounds
    shots : int -- measurement shots
    backend : Qiskit backend or None (use AerSimulator)
    transpile_to : str or list or None
        None -- no transpilation
        'heavy-hex' -- auto-generate heavy-hex coupling map
    use_flags : bool -- flag-qubit extraction (2 ancillas/stabilizer, deg 2)
    pw_kwargs : dict -- passed to PlaneWarp()
    builder_kwargs : dict -- passed to QECBuilder()

    Returns (correction, syndromes, info)
    """
    if pw_kwargs is None:
        pw_kwargs = {}
    if builder_kwargs is None:
        builder_kwargs = {}

    info = {}

    if use_flags:
        data_map, anc_maps, cm_edges, total_q = heavy_hex_flag_layout(r, s)
        qc = build_flag_circuit(r, s, rounds, data_map, anc_maps)
        n_anc = 2 * r * s
    else:
        builder = QECBuilder(r, s, **builder_kwargs)
        qc = builder.build_rounds_circuit(rounds)
        n_anc = r * s
        cm_edges = None

    # Transpile if requested
    if transpile_to == 'heavy-hex':
        if cm_edges is None:
            n_q = qc.num_qubits
            cols = max(1, (n_q + 5) // 6)
            rows = max(1, (n_q + 5) // (6 * cols))
            cm_edges = heavy_hex_coupling(rows, cols)
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
        cm = CouplingMap(couplinglist=cm_edges)
        qc = transpile(
            qc, coupling_map=cm,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
            optimization_level=3, routing_method='sabre',
            seed_transpiler=42,
        )
        info['depth'] = qc.depth()
        info['gates'] = dict(qc.count_ops())
        info['phys_qubits'] = qc.num_qubits

    # Run
    if backend is None:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Convert counts to syndrome array
    if use_flags:
        syn = syndrome_from_flag_counts(counts, rounds, r, s)
    else:
        builder = QECBuilder(r, s, **builder_kwargs)
        syn = builder.syndrome_from_counts(counts, rounds)

    # Decode
    pw = PlaneWarp(**pw_kwargs)
    correction = pw.decode_tesseract(syn)

    return correction, syn, info


# =========================================================================
# Utility: generate heavy-hex coupling map
# =========================================================================

def heavy_hex_coupling(rows, cols):
    """Generate IBM-style heavy-hex coupling map edges.

    Returns list of (q0, q1) directed pairs.
    """
    edges = set()
    for r in range(rows):
        for c in range(cols):
            base = (r * cols + c) * 6
            for i in range(6):
                edges.add((base + i, base + (i + 1) % 6))
            if c > 0:
                left_base = (r * cols + (c - 1)) * 6
                edges.add((base + 1, left_base + 3))
            if r > 0:
                above_base = ((r - 1) * cols + c) * 6
                if c % 2 == 0:
                    edges.add((base + 5, above_base + 1))
                else:
                    edges.add((base + 2, above_base + 4))
    return list(edges)
