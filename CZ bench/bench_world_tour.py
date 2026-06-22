#!/usr/bin/env python3
"""World-tour STIM bench — multiple circuit types, plane_warp decoder, next-gen hardware."""

import sys, os, subprocess, struct, math, time
import numpy as np, stim

DECODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")

CONFIGS = [
    (6, 6, 0.0002, 0.001, "cz",       "6×6  CZ-based"),
    (6, 6, 0.0002, 0.001, "phenom",   "6×6  phenomenological"),
    (6, 6, 0.0005, 0.001, "cn",       "6×6  CNOT-based"),
    (6, 6, 0.0005, 0.001, "correlated","6×6  correlated-pair"),
    (6, 6, 0.0005, 0.001, "asymmetric","6×6  asymmetric (10× hot sub)"),
    (20,20,0.0001, 0.001, "cz",       "20×20 CZ-based"),
    (20,20,0.0001, 0.001, "phenom",   "20×20 phenomenological"),
    (20,20,0.0002, 0.001, "cn",       "20×20 CNOT-based"),
]


def build_circuit(R, S, p_g, p_meas, ctype, rounds=5):
    N = R * S
    ND = NA = N
    c = stim.Circuit()
    rng = np.random.RandomState(12345)

    if ctype == "phenom":
        # Phenomenological: i.i.d. data X errors per round + measurement flips.
        # Use CZ gates for measurement but with NO gate noise — only data X + meas flips.
        for rnd in range(rounds):
            c.append("X_ERROR", list(range(ND)), p_g)
            c.append("R", range(ND, ND + NA))
            c.append("H", range(ND, ND + NA))
            for a in range(R):
                for b in range(S):
                    anc = ND + a * S + b
                    qs = [(a % R) * S + (b % S), ((a + 2) % R) * S + (b % S),
                          (a % R) * S + ((b + 2) % S), ((a + 2) % R) * S + ((b + 2) % S)]
                    for q in qs:
                        c.append("CZ", [anc, q])  # no gate noise
            c.append("H", range(ND, ND + NA))
            c.append("M", range(ND, ND + NA))
            c.append("X_ERROR", list(range(ND, ND + NA)), p_meas)
        c.append("M", range(ND))
        return c, N

    # Gate-based circuits
    for rnd in range(rounds):
        c.append("R", range(ND, ND + NA))
        c.append("H", range(ND, ND + NA))
        c.append("DEPOLARIZE1", list(range(ND)), p_g / 10)

        for a in range(R):
            for b in range(S):
                anc = ND + a * S + b
                qs = [(a % R) * S + (b % S), ((a + 2) % R) * S + (b % S),
                      (a % R) * S + ((b + 2) % S), ((a + 2) % R) * S + ((b + 2) % S)]

                if ctype == "asymmetric":
                    sl = (a % 2) * 2 + (b % 2)
                    pg_eff = p_g * 10 if sl == 0 else p_g
                else:
                    pg_eff = p_g

                for qi, q in enumerate(qs):
                    pz = rng.lognormal(mean=math.log(pg_eff), sigma=0.2)
                    gate = "CZ" if ctype != "cn" else "CNOT"
                    c.append(gate, [anc, q])
                    c.append("DEPOLARIZE2", [anc, q], float(pz))

                if ctype == "correlated":
                    # Extra correlated errors on adjacent qubit pairs within the check
                    for qi in range(4):
                        for qj in range(qi + 1, 4):
                            if rng.random() < p_g * 2:
                                c.append("DEPOLARIZE2", [qs[qi], qs[qj]], float(p_g * 0.3))

        c.append("DEPOLARIZE1", list(range(ND, ND + NA)), p_g / 10)
        c.append("H", range(ND, ND + NA))
        c.append("M", range(ND, ND + NA))
        c.append("X_ERROR", list(range(ND, ND + NA)), p_meas)

    c.append("M", range(ND))
    return c, N


def decode_raw(syn, R, S):
    return subprocess.run(
        [DECODER, str(R), str(S), "--decode"],
        input=syn, capture_output=True, timeout=60,
    ).stdout


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    print(f"{'='*90}")
    print(f"World-Tour STIM Bench — plane_warp Decoder, Next-Gen Hardware")
    print(f"Trials/config: {T}    p_meas=0.1%    5 rounds")
    print(f"{'='*90}")
    header = (f"  {'circuit':>24s}  {'grid':>5s}  {'p_g':>7s}  "
              f"{'base':>7s}  {'decode':>7s}  {'reduction':>10s}  {'FT':>3s}")
    print(header)
    print(f"  {'─'*24}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*3}")

    results = []
    for R, S, p_g, p_meas, ctype, label in CONFIGS:
        c, N = build_circuit(R, S, p_g, p_meas, ctype)
        sampler = c.compile_sampler()
        shots = sampler.sample(shots=T).astype(np.uint8)
        obs = range(0, S, 2)

        e_base = e_dec = 0
        for t in range(T):
            shot = shots[t]
            syn = bytes(shot[4 * N : 5 * N])
            dm = shot[5 * N : 6 * N]
            ov = int(sum(dm[q] for q in obs) % 2)
            cr = decode_raw(syn, R, S)
            if ov == 1: e_base += 1
            if (ov ^ sum(cr[q] for q in obs) % 2) == 1: e_dec += 1

        base = 100.0 * e_base / T
        dec = 100.0 * e_dec / T
        red = (base - dec) / base * 100 if base > 0 else 0
        best = min(base, dec)
        m = lambda v: f"*{v:.2f}%" if v == best else f"{v:.2f}%"
        ft = "✓" if dec < base else ("≈" if dec == base else "✗")
        print(f"  {label:>24s}  {R}x{S:<2d}  {p_g:7.4f}  {m(base):>7s}  "
              f"{m(dec):>7s}  {red:+9.1f}%  {ft:>3s}")
        results.append((label, base, dec, red, ft))

    print(f"{'='*90}")
    print("phenomenological = data X errors + measurement flips (no gate noise)")
    print("asymmetric = one sub-lattice 10× worse gate fidelity")
    print("correlated-pair = CZ + adjacent-qubit correlated DEPOLARIZE2")
    print("✓ = decoder LER < baseline (fault tolerance achieved)")
    print(f"{'='*90}")

    # Summary
    wins = sum(1 for _, b, d, _, _ in results if d < b)
    print(f"\nDecoder achieves FT in {wins}/{len(results)} configurations.")
    if wins:
        avg_red = sum(r[3] for r in results if r[3] > 0) / max(1, wins)
        print(f"Average reduction when FT achieved: {avg_red:.1f}%")


if __name__ == "__main__":
    main()
