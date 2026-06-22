#!/usr/bin/env python3
"""80×80 NISQ bench — plane_warp decoder on 12,800 qubits, d≈40.

Single observable (Z on first sub-lattice row), 3 measurement rounds,
per-gate DEPOLARIZE2 noise, with optional mid-circuit logical X gate.

Usage: python3 80x80.py [trials]

Build: gcc -std=gnu11 -O3 -o plane_warp plane_warp.c -lm
"""

import sys, os, subprocess, struct, time
import numpy as np, stim

DECODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")
R = S = 80
N = R * S  # 6400
OBS = list(range(0, S, 2))  # first sub-lattice row qubits
LOGICAL_X = [si * 2 * S + 0 for si in range(R // 2)]  # first sub-lattice column


def build_circuit(p_gate, with_gate=False, rounds=3):
    c = stim.Circuit()
    ND = NA = N
    for rnd in range(rounds):
        c.append("R", range(ND, ND + NA))
        c.append("H", range(ND, ND + NA))
        for a in range(R):
            for b in range(S):
                anc = ND + a * S + b
                qs = [
                    (a % R) * S + (b % S),
                    ((a + 2) % R) * S + (b % S),
                    (a % R) * S + ((b + 2) % S),
                    ((a + 2) % R) * S + ((b + 2) % S),
                ]
                for q in qs:
                    c.append("CZ", [anc, q])
                    c.append("DEPOLARIZE2", [anc, q], p_gate)
        c.append("H", range(ND, ND + NA))
        c.append("M", range(ND, ND + NA))
        if with_gate and rnd == 1:
            for q in LOGICAL_X:
                c.append("X", q)
    c.append("M", range(ND))
    return c


def decode(syn):
    return subprocess.run(
        [DECODER, str(R), str(S), "--decode"],
        input=syn, capture_output=True, timeout=60,
    ).stdout


def logical_error_rate(shots, with_gate):
    n_err = 0
    for t in range(len(shots)):
        shot = shots[t]
        syn = bytes(shot[2 * N : 3 * N])  # last round syndrome
        dm = shot[3 * N : 4 * N]          # final data measurements
        corr = decode(syn)
        ov = int(sum(dm[q] for q in OBS) % 2)
        expected = 1 if with_gate else 0
        if (ov ^ sum(corr[q] for q in OBS) % 2) != expected:
            n_err += 1
    return 100.0 * n_err / len(shots)


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    configs = [
        (0.00005, "p_g=0.005%"),
        (0.00010, "p_g=0.010%"),
        (0.00020, "p_g=0.020%"),
        (0.00050, "p_g=0.050%"),
        (0.00100, "p_g=0.100%"),
    ]

    print(f"{'='*72}")
    print(f"80×80 NISQ Bench — Plane-Warp Decoder")
    print(f"{N} data + {N} ancilla = {2*N} qubits, d≈40")
    print(f"3 rounds × {N*4} CZ/round = {N*4*3} CZ gates total")
    print(f"Trials per config: {T}")
    print(f"{'='*72}")
    print(f"  {'p_gate':>10s}  {'no-gate':>9s}  {'with-gate':>10s}  {'penalty':>8s}")
    print(f"  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*8}")

    for pg, label in configs:
        t0 = time.time()

        c_ng = build_circuit(pg, with_gate=False)
        shots_ng = c_ng.compile_sampler().sample(shots=T).astype(np.uint8)
        ler_ng = logical_error_rate(shots_ng, with_gate=False)

        c_g = build_circuit(pg, with_gate=True)
        shots_g = c_g.compile_sampler().sample(shots=T).astype(np.uint8)
        ler_g = logical_error_rate(shots_g, with_gate=True)

        dt = time.time() - t0
        print(f"  {label:>10s}  {ler_ng:8.1f}%  {ler_g:9.1f}%  {ler_g-ler_ng:+8.1f}pp  ({dt:3.0f}s)")

    print(f"{'='*72}")
    print("no-gate: decoder corrects noise, target observable = 0")
    print("with-gate: logical X injected mid-circuit, target observable = 1")
    print("penalty = additional LER from gate presence. 0pp = gate transparent.")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
