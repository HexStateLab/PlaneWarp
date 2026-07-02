#!/usr/bin/env python3
"""parse_ionq.py — all-purpose parser for IonQ result outputs + decoder.

Accepts (auto-detected):
  * IonQ API job JSON:      {"data": {"histogram": {...}}}, {"histogram": ...},
                            {"results": ...}, {"probabilities": ...}
  * histogram keyed by decimal ints ("0", "133") with counts OR probabilities
  * histogram keyed by bitstrings ("0110...") with counts or probabilities
  * qiskit get_counts() dicts (space-separated registers OK)
  * per-shot lists: {"measurements": [[0,1,...], ...]} or [[...], ...]
  * plain text: one "bitstring count" per line (what you pasted last time)

Bit order is auto-detected (or forced with --bit-order): both q0=LSB
(leftmost char = highest qubit) and q0=MSB interpretations are scored by the
GHZ witness / syndrome consistency and the better one is used.

Qubit layout (pw_opt compact, grid r x s):
  q[0 .. r*s-1]                data, q[i*s+j] = site (i, j)
  q[r*s .. r*s+n_anc-1]        syndrome ancillas
  q[r*s+n_anc]                 bell   (m1)
  q[r*s+n_anc+1]               bell_m (m2)

Decoding: with per-qubit final readout only the LAST raw ancilla values
survive; under no-reset those equal the cumulative check parity, which is a
valid final-round syndrome.  If decoder.py + libplane_warp.so are present the
weight-4 syndrome is decoded per shot (deduped by unique syndrome); otherwise
raw results are reported.

Usage:
  python3 parse_ionq.py results.json --grid 4 4
  python3 parse_ionq.py results.txt  --grid 4 4 --shots 100 --bit-order auto
"""
import argparse, json, sys
import numpy as np
from pw_opt import _unpack_indices, _bell_support

try:
    from decoder import tesseract_decode_ffinal
    HAVE_DECODER = True
except Exception as _e:                                   # missing .so etc.
    HAVE_DECODER = False
    _DECODER_ERR = _e


# ---------------- input normalisation ----------------

def _dig_for_histogram(obj):
    """Recursively find the first dict that looks like a histogram, or a
    per-shot list of bit lists."""
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if keys and all(isinstance(k, str) and
                        (k.isdigit() or set(k) <= set("01 ")) for k in keys) \
                and all(isinstance(v, (int, float)) for v in obj.values()):
            return ("hist", obj)
        for name in ("histogram", "probabilities", "counts", "results",
                     "data", "measurements", "shots"):
            if name in obj:
                found = _dig_for_histogram(obj[name])
                if found:
                    return found
        for v in obj.values():
            found = _dig_for_histogram(v)
            if found:
                return found
    elif isinstance(obj, list) and obj and isinstance(obj[0], (list, str)):
        return ("shotlist", obj)
    return None


def load_counts(path, n_qubits, shots_hint=None):
    """Return dict {bitstring(width n_qubits): int count} in file order.
    Bitstrings are returned exactly as given (or int->binary, MSB-left)."""
    text = open(path).read().strip()
    counts = {}
    try:
        obj = json.loads(text)
        found = _dig_for_histogram(obj)
        if not found:
            raise ValueError("no histogram/shot list found in JSON")
        kind, payload = found
        if kind == "shotlist":
            for row in payload:
                key = "".join(str(int(b)) for b in row) if isinstance(row, list) else row
                counts[key] = counts.get(key, 0) + 1
        else:
            vals = list(payload.values())
            probabilistic = all(0 <= v <= 1 for v in vals) and \
                            abs(sum(vals) - 1.0) < 1e-3 and \
                            any(v not in (0, 1) for v in vals)
            for k, v in payload.items():
                key = k.replace(" ", "")
                if key.isdigit() and set(key) - set("01"):   # decimal int key
                    key = format(int(key), f"0{n_qubits}b")
                elif set(key) <= set("01") and len(key) != n_qubits and key.isdigit():
                    # ambiguous all-0/1 decimal like "10": pad as bitstring
                    key = key.zfill(n_qubits)
                cnt = v * (shots_hint or 1) if probabilistic else v
                counts[key] = counts.get(key, 0) + cnt
            if probabilistic and not shots_hint:
                print("  note: probabilities given, no --shots; using weights")
    except json.JSONDecodeError:                             # plain text lines
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            cnt = int(parts[1]) if len(parts) > 1 else 1
            counts[key] = counts.get(key, 0) + cnt
    # sanity: pad / validate width
    fixed = {}
    for k, v in counts.items():
        if len(k) < n_qubits:
            k = k.zfill(n_qubits)
        if len(k) != n_qubits or set(k) - set("01"):
            raise ValueError(f"key {k!r} incompatible with {n_qubits} qubits")
        fixed[k] = fixed.get(k, 0) + v
    return fixed


# ---------------- physics ----------------

def unpack(counts, n_qubits, q0_lsb=True):
    """counts -> (Q, N): Q[shot_class, qubit] uint8, N weights.
    q0_lsb=True: leftmost char is the HIGHEST qubit (standard binary)."""
    keys = list(counts.keys())
    B = np.array([[int(c) for c in k] for k in keys], dtype=np.uint8)
    N = np.array([counts[k] for k in keys], dtype=float)
    Q = B[:, ::-1] if q0_lsb else B
    return Q, N


def analyze(Q, N, r, s, decode=True, full_stabilizer=True, periodic=True):
    n_data = r * s
    n_anc = 4 * (r // 2 - 1) * (s // 2)
    data = Q[:, :n_data].reshape(-1, r, s)
    anc = Q[:, n_data:n_data + n_anc]
    m1 = Q[:, n_data + n_anc]
    m2 = Q[:, n_data + n_anc + 1]
    total = N.sum()

    # last-round cumulative syndrome from raw ancilla bits
    ui, uj = _unpack_indices(r, s)
    V = np.zeros((data.shape[0], r, s), dtype=np.uint8)
    V[:, ui, uj] = anc
    if periodic:
        V[:, r - 2, :] = V[:, 0:r - 2:2, :].sum(axis=1) % 2
        V[:, r - 1, :] = V[:, 1:r - 1:2, :].sum(axis=1) % 2
    syn = V if full_stabilizer else V ^ np.roll(V, -2, axis=2)

    corr = np.zeros_like(data)
    decoded = False
    if decode and HAVE_DECODER:
        cache = {}
        for i in range(data.shape[0]):
            key = syn[i].tobytes()
            if key not in cache:
                cache[key] = tesseract_decode_ffinal(syn[i][None], r, s)
            corr[i] = cache[key]
        decoded = True
    fixed = data ^ corr

    def witness(d):
        vals = np.column_stack([d[:, i, :].sum(1) % 2 for i in range(r - 1)] +
                               [d[:, :, j].sum(1) % 2 for j in range(s - 1)])
        P = ((vals.max(1) == vals.min(1)) * N).sum() / total
        C = 2 * ((m1 == m2) * N).sum() / total - 1
        return dict(P=P, C=C, F=(P + C) / 2,
                    all_zero=((vals.sum(1) == 0) * N).sum() / total,
                    flip=[(vals[:, k] * N).sum() / total
                          for k in range(vals.shape[1])])

    return dict(raw=witness(data),
                dec=witness(fixed) if decoded else None,
                decoded=decoded,
                syn_weight=(syn.sum(axis=(1, 2)) * N).sum() / total,
                m1_frac=(m1 * N).sum() / total,
                shots=total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="IonQ .json, qiskit counts .json, or text")
    ap.add_argument("--grid", type=int, nargs=2, default=(4, 4))
    ap.add_argument("--shots", type=int, default=None,
                    help="total shots (needed if file gives probabilities)")
    ap.add_argument("--bit-order", choices=["auto", "q0-lsb", "q0-msb"],
                    default="auto")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--open-boundaries", action="store_true",
                    help="grid was built with periodic=False")
    opts = ap.parse_args()

    r, s = opts.grid
    n_anc = 4 * (r // 2 - 1) * (s // 2)
    n_q = r * s + n_anc + 2
    periodic = not opts.open_boundaries

    counts = load_counts(opts.file, n_q, opts.shots)
    print(f"parsed {len(counts)} distinct outcomes, "
          f"{sum(counts.values()):g} shots, {n_q} qubits expected")

    if not HAVE_DECODER and not opts.no_decode:
        print(f"  decoder unavailable ({type(_DECODER_ERR).__name__}: "
              f"{_DECODER_ERR}) — reporting raw results")

    results = {}
    orders = {"q0-lsb": True, "q0-msb": False}
    todo = orders if opts.bit_order == "auto" else \
        {opts.bit_order: orders[opts.bit_order]}
    for name, lsb in todo.items():
        Q, N = unpack(counts, n_q, q0_lsb=lsb)
        results[name] = analyze(Q, N, r, s, decode=not opts.no_decode,
                                periodic=periodic)

    # auto-detect: prefer higher raw F, tiebreak on lower syndrome weight
    best = max(results, key=lambda k: (results[k]["raw"]["F"],
                                       -results[k]["syn_weight"]))
    if opts.bit_order == "auto" and len(results) > 1:
        o = {k: results[k]["raw"]["F"] for k in results}
        print(f"bit order auto-detect: {best}  (raw F: " +
              ", ".join(f"{k}={v:+.3f}" for k, v in o.items()) + ")")

    res = results[best]
    names = [f"Z_row_{i}" for i in range(r - 1)] + \
            [f"Z_col_{j}" for j in range(s - 1)]

    def show(tag, w):
        print(f"\n[{tag}]")
        print(f"  P (all logicals agree) = {w['P']:.4f}")
        print(f"  C (m2==m1 coherence)   = {w['C']:+.4f}")
        print(f"  F ~ (P+C)/2            = {w['F']:.4f}  "
              f"-> {'GHZ ENTANGLED' if w['F'] > 0.5 else 'not certified'}")
        print(f"  P(all zero) = {w['all_zero']:.4f}")
        for nm, f in zip(names, w["flip"]):
            print(f"    {nm}: flipped {f:.3f}")

    print(f"\nm1 split: {1 - res['m1_frac']:.2f}/{res['m1_frac']:.2f}   "
          f"mean syndrome weight: {res['syn_weight']:.3f}")
    show("raw (undecoded)", res["raw"])
    if res["decoded"]:
        show("decoded", res["dec"])
    print("\ndone.")


if __name__ == "__main__":
    main()
