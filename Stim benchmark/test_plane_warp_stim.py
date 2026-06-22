#!/usr/bin/env python3
"""
STIM-based test harness for plane_warp.c — 2D BB code plane-warp decoder.

Stabilizer structure (4-body plus-shaped checks on r×s torus):
  HZ (Z-type, detects X errors): check (a,b) =
      Z(a,b) * Z(a+2,b) * Z(a,b+2) * Z(a+2,b+2)
  HX (X-type, detects Z errors): check (a,b) = HZ pattern shifted (-2,-2) =
      X(a-2,b-2) * X(a-2,b) * X(a,b-2) * X(a,b)

Both check types use identical connectivity (just different Pauli basis).
The C decoder's solve_plane works on syndromes from either check type
since the mathematical problem (plus-shaped recurrence) is the same.

Usage:
  python3 test_plane_warp_stim.py [r] [s] [--trials N] [--weights w1,w2,...]
"""

import argparse, os, subprocess, sys, time
import numpy as np
import stim

DECODER_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")


# =========================================================================
#  STIM circuit
# =========================================================================

def _hz_check(r: int, s: int, a: int, b: int) -> stim.PauliString:
    """HZ check at (a,b): Z on (a,b), (a+2,b), (a,b+2), (a+2,b+2)."""
    qs = [
        (a % r) * s + (b % s),
        ((a + 2) % r) * s + (b % s),
        (a % r) * s + ((b + 2) % s),
        ((a + 2) % r) * s + ((b + 2) % s),
    ]
    return stim.PauliString("Z" + "*Z".join(str(q) for q in qs))


def _hx_check(r: int, s: int, a: int, b: int) -> stim.PauliString:
    """HX check at (a,b): X on (a-2,b-2), (a-2,b), (a,b-2), (a,b)."""
    qs = [
        ((a - 2 + r) % r) * s + ((b - 2 + s) % s),
        ((a - 2 + r) % r) * s + (b % s),
        (a % r) * s + ((b - 2 + s) % s),
        (a % r) * s + (b % s),
    ]
    return stim.PauliString("X" + "*X".join(str(q) for q in qs))


def build_circuit(r: int, s: int) -> stim.Circuit:
    """Build noiseless MPP circuit: qubit coords + all HZ + HX checks.

    Result has 2n measurement bits: first n = HZ (X-syndrome), next n = HX (Z-syndrome).
    """
    n = r * s
    c = stim.Circuit()

    for qi in range(r):
        for qj in range(s):
            c.append("QUBIT_COORDS", qi * s + qj, [qi, qj])

    hz = [_hz_check(r, s, a, b) for a in range(r) for b in range(s)]
    hx = [_hx_check(r, s, a, b) for a in range(r) for b in range(s)]
    c.append("MPP", hz + hx)
    return c


def build_circuit_with_errors(r: int, s: int, px: float = 0.0, pz: float = 0.0) -> stim.Circuit:
    """Build circuit with X_ERROR / Z_ERROR channels before measurement."""
    n = r * s
    ec = stim.Circuit()
    if px > 0.0:
        ec.append("X_ERROR", list(range(n)), px)
    if pz > 0.0:
        ec.append("Z_ERROR", list(range(n)), pz)
    return ec + build_circuit(r, s)


# =========================================================================
#  Syndrome helpers — replicate C-code logic for cross-validation
# =========================================================================

def syndrome_x(r: int, s: int, err_x: bytes) -> bytes:
    """HZ syndrome from X errors (mirrors syndrome_of in plane_warp.c).

    X error at (qi,qj) flips syn at (qi,qj), (qi-2,qj), (qi,qj-2), (qi-2,qj-2).
    """
    n = r * s
    syn = bytearray(n)
    for q in range(n):
        if not err_x[q]:
            continue
        qi, qj = q // s, q % s
        for di in (0, 2):
            for dj in (0, 2):
                ci = (qi - di + r) % r
                cj = (qj - dj + s) % s
                syn[ci * s + cj] ^= 1
    return bytes(syn)


def syndrome_z(r: int, s: int, err_z: bytes) -> bytes:
    """HX syndrome from Z errors (mirrors decode_Z in plane_warp.c).

    Z error at (qi,qj) flips syn at (qi+2,qj+2), (qi+2,qj), (qi,qj+2), (qi,qj).
    """
    n = r * s
    syn = bytearray(n)
    for q in range(n):
        if not err_z[q]:
            continue
        qi, qj = q // s, q % s
        for di in (0, 2):
            for dj in (0, 2):
                ci = (qi + 2 - di + r) % r
                cj = (qj + 2 - dj + s) % s
                syn[ci * s + cj] ^= 1
    return bytes(syn)


# =========================================================================
#  Decoder subprocess
# =========================================================================

def decode(r: int, s: int, syn: bytes, z_mode: bool = False) -> bytes:
    """Pipe syndrome to plane_warp --decode (X) or --decode-z (Z), return correction."""
    assert len(syn) == r * s
    flag = "--decode-z" if z_mode else "--decode"
    proc = subprocess.run(
        [DECODER_BIN, str(r), str(s), flag],
        input=syn, capture_output=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Decoder rc={proc.returncode}: {proc.stderr.decode()}")
    return proc.stdout


# =========================================================================
#  Stabilizer check (from plane_warp.c is_stabilizer)
# =========================================================================

def is_stabilizer(r: int, s: int, vec: bytes) -> bool:
    """True iff all sub-lattice row/col parity sums are even.

    The 2D BB code decomposes into 4 independent (r/2 × s/2) sub-lattices
    (one per parity class). A vector is a stabilizer iff every row and column
    within each sub-lattice has even parity.
    """
    for px in range(2):
        for py in range(2):
            hr, hs = r // 2, s // 2
            for si in range(hr):
                rp = 0
                for sj in range(hs):
                    if vec[(px + 2 * si) * s + (py + 2 * sj)]:
                        rp ^= 1
                if rp:
                    return False
            for sj in range(hs):
                cp = 0
                for si in range(hr):
                    if vec[(px + 2 * si) * s + (py + 2 * sj)]:
                        cp ^= 1
                if cp:
                    return False
    return True


# =========================================================================
#  Tier-1: STIM ↔ C cross-validation
# =========================================================================

def test_stim_x_cross_validation(r: int, s: int) -> bool:
    """Exhaustive weight-1: verify STIM HZ measurements == C syndrome_x()."""
    n = r * s
    print("[Tier 1] STIM ↔ C syndrome cross-validation (weight-1, all qubits) … ",
          end="", flush=True)

    c_noisy_template = build_circuit(r, s)
    sampler_base = stim.Circuit()
    # Build single-error circuits on the fly — not worth precomputing for 6400.

    mismatches = 0
    for q in range(n):
        err = bytearray(n)
        err[q] = 1
        syn_c = list(syndrome_x(r, s, bytes(err)))

        c_err = stim.Circuit()
        for qi in range(r):
            for qj in range(s):
                c_err.append("QUBIT_COORDS", qi * s + qj, [qi, qj])
        c_err.append("X_ERROR", [q], 1.0)
        c_err += c_noisy_template
        bits = c_err.compile_sampler().sample(shots=1)[0].astype(np.uint8)
        syn_stim_hz = bits[:n].tolist()

        if syn_stim_hz != syn_c:
            mismatches += 1
            if mismatches <= 3:
                idx = [i for i in range(n) if syn_stim_hz[i] != syn_c[i]]
                print(f"\n  Mismatch q={q}: {len(idx)} differing positions")
                print(f"    stim: {[syn_stim_hz[i] for i in idx[:5]]}")
                print(f"    c:    {[syn_c[i] for i in idx[:5]]}")

    if mismatches == 0:
        print("PASS")
        return True
    else:
        print(f"\n  FAIL: {mismatches} qubits had mismatching syndrome")
        return False


# =========================================================================
#  Tier-2/3: Error-decode tests
# =========================================================================

def test_weight(r: int, s: int, err_type: str, weight: int, trials: int) -> dict:
    """Run decoder on random weight-w errors."""
    n = r * s
    rng = np.random.RandomState((42 + hash(err_type)) & 0xFFFFFFFF)
    sound_ok, logical_ok = 0, 0

    z_mode = err_type == "z"
    for _ in range(trials):
        err = bytearray(n)
        qubits = rng.choice(n, size=min(weight, n), replace=False)
        for q in qubits:
            err[q] = 1
        err_bytes = bytes(err)

        syn = syndrome_x(r, s, err_bytes) if err_type == "x" else syndrome_z(r, s, err_bytes)
        correction = decode(r, s, syn, z_mode=z_mode)

        # Soundness: correction reproduces the same syndrome
        corr_syn = syndrome_x(r, s, correction) if err_type == "x" else syndrome_z(r, s, correction)
        if corr_syn == syn:
            sound_ok += 1

        # Correctness: err XOR correction is a stabilizer (no logical error)
        diff = bytes(a ^ b for a, b in zip(err_bytes, correction))
        if is_stabilizer(r, s, diff):
            logical_ok += 1

    return {"sound_ok": sound_ok, "logical_ok": logical_ok}


def wilson_interval(ok: int, trials: int):
    if trials == 0:
        return 0.0, 1.0
    z = 1.95996398454005
    p = ok / trials
    denom = 1.0 + z * z / trials
    center = p + z * z / (2 * trials)
    margin = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    lo = max(0.0, (center - margin) / denom)
    hi = min(1.0, (center + margin) / denom)
    return lo, hi


# =========================================================================
#  main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="STIM test for plane_warp.c decoder")
    parser.add_argument("r", nargs="?", type=int, default=8,
                        help="Grid dimension (default 8, must be even)")
    parser.add_argument("s", nargs="?", type=int, default=None,
                        help="Grid cols (default = r, must be even)")
    parser.add_argument("--trials", type=int, default=200,
                        help="Trials per weight")
    parser.add_argument("--weights", type=str, default="1,2,3,5,7,10",
                        help="Comma-separated error weights")
    parser.add_argument("--verbose", action="store_true",
                        help="Show STIM circuit, progress, etc.")
    args = parser.parse_args()

    r, s = args.r, args.s if args.s is not None else args.r
    n = r * s
    if r % 2 != 0 or s % 2 != 0:
        print("Error: r and s must be even for sub-lattice decomposition")
        return 1

    if not os.path.isfile(DECODER_BIN) or not os.access(DECODER_BIN, os.X_OK):
        print(f"Error: decoder not found or not executable: {DECODER_BIN}")
        print("Build with: gcc -std=gnu11 -O3 -o plane_warp plane_warp.c -lm")
        return 1

    weights = [int(w) for w in args.weights.split(",")]

    print(f"{'='*60}")
    print(f"STIM Test Harness — plane_warp Decoder")
    print(f"Grid: {r}x{s} torus, n={n}, decoder at {DECODER_BIN}")
    print(f"Trials/weight: {args.trials}")
    print(f"{'='*60}\n")

    # Tier 1
    if not test_stim_x_cross_validation(r, s):
        print("\nAborting: STIM and C disagree on syndrome computation.")
        return 1

    # Tier 2: X errors
    print("\n[Tier 2] X-error decode — soundness + logical error rate")
    print(f"  {'Weight':>6}  {'OK/Trials':>10}  {'Rate':>8}  {'Sound OK':>9}  95% Wilson")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*22}")
    for w in weights:
        if w > n:
            continue
        res = test_weight(r, s, "x", w, args.trials)
        lo, hi = wilson_interval(res["logical_ok"], args.trials)
        print(f"  {w:6d}  {res['logical_ok']:4d}/{args.trials:<5d}"
              f"  {100*res['logical_ok']/args.trials:7.1f}%"
              f"  {res['sound_ok']:4d}/{args.trials:<4d}"
              f"  [{lo:.4f}, {hi:.4f}]")

    # Tier 3: Z errors
    print("\n[Tier 3] Z-error decode — soundness + logical error rate")
    print(f"  {'Weight':>6}  {'OK/Trials':>10}  {'Rate':>8}  {'Sound OK':>9}  95% Wilson")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*22}")
    for w in weights:
        if w > n:
            continue
        res = test_weight(r, s, "z", w, args.trials)
        lo, hi = wilson_interval(res["logical_ok"], args.trials)
        print(f"  {w:6d}  {res['logical_ok']:4d}/{args.trials:<5d}"
              f"  {100*res['logical_ok']/args.trials:7.1f}%"
              f"  {res['sound_ok']:4d}/{args.trials:<4d}"
              f"  [{lo:.4f}, {hi:.4f}]")

    # Tier 4: Exhaustive weight-2 on small grids
    if n <= 256:
        d_code = min(r // 2, s // 2)
        print(f"\n[Tier 4] Exhaustive weight-2 X-error test (code distance ≈ {d_code})")
        print(f"  expected: weight-1 always correctable, some weight-2 might be degenerate")
        print(f"  testing … ", end="", flush=True)
        fails, tested = 0, 0
        for q1 in range(n):
            for q2 in range(q1 + 1, n):
                err = bytearray(n)
                err[q1] = 1
                err[q2] = 1
                syn = syndrome_x(r, s, bytes(err))
                correction = decode(r, s, syn, z_mode=False)
                diff = bytes(a ^ b for a, b in zip(err, correction))
                if not is_stabilizer(r, s, diff):
                    fails += 1
                tested += 1
                if tested % 10000 == 0 and args.verbose:
                    print(f"\r    tested {tested}…", end="")
        pct = 100 * fails / tested if tested else 0
        print(f"{tested} pairs, {fails} logical error(s) ({pct:.1f}%)")

    if args.verbose and n <= 64:
        print(f"\n--- STIM circuit for {r}x{s} torus ---")
        print(build_circuit(r, s))

    print(f"\n{'='*60}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
