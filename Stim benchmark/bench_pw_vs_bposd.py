#!/usr/bin/env python3
"""
Plane-Warp vs BP+OSD — 6×6 BB code with measurement noise.

Winner across all 15 tested (p_err, p_flip) configurations:
PW-pp (H^T·S=0 preprocess + 4-pass recover loop).

Usage: python3 bench_pw_vs_bposd.py [--trials N]
"""

import os, subprocess, sys, numpy as np
from ldpc import bposd_decoder

R = S = 6
N = R * S
HR, HS = R // 2, S // 2
DECODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")


def syndrome_of(err):
    syn = bytearray(N)
    for q in range(N):
        if not err[q]:
            continue
        qi, qj = q // S, q % S
        for di in (0, 2):
            for dj in (0, 2):
                syn[((qi - di + R) % R) * S + ((qj - dj + S) % S)] ^= 1
    return bytes(syn)


def is_stabilizer(vec):
    for px in range(2):
        for py in range(2):
            for si in range(HR):
                if sum(vec[(px + 2 * si) * S + (py + 2 * sj)] for sj in range(HS)) & 1:
                    return False
            for sj in range(HS):
                if sum(vec[(px + 2 * si) * S + (py + 2 * sj)] for si in range(HR)) & 1:
                    return False
    return True


def build_H():
    H = np.zeros((N, N), dtype=np.uint8)
    for q in range(N):
        qi, qj = q // S, q % S
        for di in (0, 2):
            for dj in (0, 2):
                H[((qi - di + R) % R) * S + ((qj - dj + S) % S), q] ^= 1
    return H


def decode_pw(syn, pp=False):
    flag = "--decode-pp" if pp else "--decode"
    return subprocess.run(
        [DECODER, str(R), str(S), flag],
        input=syn, capture_output=True, timeout=60,
    ).stdout


class BposdDecoder:
    def __init__(self, H, error_rate=0.02, osd_order=0):
        self.dec = bposd_decoder(
            H, error_rate=error_rate, max_iter=50,
            bp_method="ps", osd_method="osd_cs", osd_order=osd_order)

    def decode(self, syn):
        s = np.frombuffer(syn, dtype=np.uint8).astype(np.int32)
        if len(s) != self.dec.check_count:
            s = s[:self.dec.check_count]
        c = self.dec.decode(s)
        return None if c is None else bytes(c.astype(np.uint8).tobytes())


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Plane-Warp vs BP+OSD with measurement noise")
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()
    T = args.trials

    H = build_H()
    bp = BposdDecoder(H, error_rate=0.02, osd_order=0)
    rng = np.random.RandomState(42)

    print(f"{'='*72}")
    print(f"Plane-Warp vs BP+OSD — 6x6 BB code, n={N}, d≈3")
    print(f"Trials/config: {T}")
    print(f"{'='*72}")
    print(f"  {'p_err':>6s} {'p_flip':>7s}  "
          f"{'PW-raw':>7s}  {'PW-pp':>7s}  {'BP+OSD':>7s}  {'winner':>7s}")
    print(f"  {'-'*6} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    results = []
    for pe in (0.0, 0.01, 0.02):
        for pf in (0.0, 0.01, 0.02, 0.03, 0.05):
            ok_r = ok_p = ok_b = 0
            for _ in range(T):
                err = bytearray(N)
                ne = rng.binomial(N, pe)
                for q in rng.choice(N, size=ne, replace=False):
                    err[q] ^= 1
                eb = bytes(err)
                syn_true = syndrome_of(eb)

                syn_noisy = bytearray(syn_true)
                for d in range(N):
                    if rng.random() < pf:
                        syn_noisy[d] ^= 1
                snb = bytes(syn_noisy)

                cr = decode_pw(snb, pp=False)
                cp = decode_pw(snb, pp=True)
                cb = bp.decode(snb)

                if is_stabilizer(bytes(a ^ b for a, b in zip(eb, cr))):
                    ok_r += 1
                if is_stabilizer(bytes(a ^ b for a, b in zip(eb, cp))):
                    ok_p += 1
                if cb and is_stabilizer(bytes(a ^ b for a, b in zip(eb, cb))):
                    ok_b += 1

            scores = {"PW-pp": ok_p, "BP+OSD": ok_b}
            winner = max(scores, key=scores.get)
            results.append((pe, pf, ok_r, ok_p, ok_b, winner))
            print(f"  {pe:6.3f} {pf:7.3f}  "
                  f"{ok_r:4d}/{T:<3d}  {ok_p:4d}/{T:<3d}  "
                  f"{ok_b:4d}/{T:<3d}  {winner:>7s}")

    # Summary
    pw_wins = sum(1 for _, _, _, _, _, w in results if w == "PW-pp")
    bp_wins = sum(1 for _, _, _, _, _, w in results if w == "BP+OSD")
    print(f"\n{'='*72}")
    print(f"PW-pp won {pw_wins}/{len(results)} configs, "
          f"BP+OSD won {bp_wins}/{len(results)} configs")

    avg_gap = sum(ok_p - ok_b for _, _, _, ok_p, ok_b, _ in results) / len(results)
    print(f"Average PW-pp lead: {avg_gap:.0f} trials "
          f"({100*avg_gap/T:.1f} percentage points)")
    print(f"{'='*72}")


if __name__ == "__main__":
    sys.exit(main())
