#!/usr/bin/env python3
"""Escalating error benchmark for plane_warp --decode-tesseract.

Sweeps physical error rates and reports the threshold where decoding fails.
"""
import stim, subprocess, struct, numpy as np, sys, time, argparse

# ---------------------------------------------------------------------------
# Syndrome computation for H = (1+x^2)(1+y^2)
# ---------------------------------------------------------------------------
def syndrome_of(errors, r, s):
    err = errors.reshape(r, s)
    syn = np.zeros((r, s), dtype=np.uint8)
    for i in range(r):
        for j in range(s):
            syn[i, j] = err[i, j] ^ err[(i+2)%r, j] ^ err[i, (j+2)%s] ^ err[(i+2)%r, (j+2)%s]
    return syn

# ---------------------------------------------------------------------------
# Logical qubit counting
# ---------------------------------------------------------------------------
def count_logicals(r, s):
    n = r * s
    H = np.zeros((n, n), dtype=np.uint8)
    for i in range(r):
        for j in range(s):
            row = i * s + j
            for di in (0, 2):
                for dj in (0, 2):
                    col = ((i + di) % r) * s + ((j + dj) % s)
                    H[row, col] ^= 1
    A = H.copy()
    rank = 0
    col = 0
    for row in range(n):
        if col >= n: break
        pivot = None
        for r2 in range(row, n):
            if A[r2, col]: pivot = r2; break
        if pivot is None: col += 1; continue
        if pivot != row: A[[row, pivot]] = A[[pivot, row]]
        for r2 in range(row + 1, n):
            if A[r2, col]: A[r2] ^= A[row]
        rank += 1
        col += 1
    return n - rank, rank  # kernel_dim, rank

# ---------------------------------------------------------------------------
# Circuit builder
# ---------------------------------------------------------------------------
def make_circuit(R, S, rounds, pm):
    """Stim circuit: depol on data, X_ERROR(pm) on ancilla measurement."""
    N = R * S
    c = stim.Circuit()
    for rnd in range(rounds):
        c.append('X_ERROR', range(N), pm * 0.1)  # data Z errors
        c.append('R', range(N, 2 * N))
        c.append('H', range(N, 2 * N))
        for a in range(R):
            for b in range(S):
                anc = N + a * S + b
                qs = [(a % R) * S + (b % S),
                      ((a + 2) % R) * S + (b % S),
                      (a % R) * S + ((b + 2) % S),
                      ((a + 2) % R) * S + ((b + 2) % S)]
                for q in qs:
                    c.append('CZ', [anc, q])
        c.append('H', range(N, 2 * N))
        c.append('X_ERROR', range(N, 2 * N), pm)
        c.append('M', range(N, 2 * N))
    c.append('M', range(N))
    return c

# ---------------------------------------------------------------------------
# Single-shot decode
# ---------------------------------------------------------------------------
def decode_shot(R, S, rounds, shot, decoder_flag):
    N = R * S
    if rounds == 1 and decoder_flag in ('--decode',):
        # Single-round: just the flat syndrome
        buf = bytes(shot[:N].astype(np.uint8))
    else:
        # Multi-round: 4-byte round count + syndromes
        buf = struct.pack('<I', rounds)
        for rnd in range(rounds):
            start = rnd * N
            buf += bytes(shot[start:start + N].astype(np.uint8))
    p = subprocess.run(['./plane_warp', str(R), str(S), decoder_flag],
                       input=buf, capture_output=True, timeout=120)
    return np.frombuffer(p.stdout, dtype=np.uint8)

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=14)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--shots', type=int, default=500)
    parser.add_argument('--decoder', default='tesseract', choices=['persist', 'tesseract'])
    opts = parser.parse_args()

    R = opts.grid; S = opts.grid; rounds = opts.rounds
    if opts.decoder == 'tesseract' and rounds == 1:
        dec = '--decode'
    elif opts.decoder == 'tesseract':
        dec = '--decode-tesseract'
    else:
        dec = '--decode-persist'
    N = R * S

    k, rk = count_logicals(R, S)
    print(f'Code: {R}x{S} grid, n={N} data qubits')
    print(f'rank(H)={rk}, ker(H)={k} logical degrees')
    print(f'Code rate k/n = {k/N:.3f}')
    print(f'Decoder: {opts.decoder}')
    print(f'Rounds: {rounds}, Shots/rate: {opts.shots}')
    print()

    rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]
    print(f'{"pm":>8}  {"ler":>10}  {"fail":>6}  {"time":>8}')
    print('-' * 38)

    for pm in rates:
        circuit = make_circuit(R, S, rounds, pm)
        sampler = circuit.compile_sampler()
        samples = sampler.sample(opts.shots, bit_packed=False)

        logical_errs = 0
        decoder_fails = 0
        t0 = time.time()

        for shot_idx in range(opts.shots):
            shot = samples[shot_idx]
            correction = decode_shot(R, S, rounds, shot, dec)
            if len(correction) < N:
                decoder_fails += 1
                continue  # decoder returned nothing usable
            true_errors = shot[rounds * N:].astype(np.uint8).reshape(R, S)
            corr_mat = correction.reshape(R, S)
            residual = true_errors ^ corr_mat

            if residual.sum() == 0:
                continue  # perfect correction

            # Check residual syndrome
            syn_sum = syndrome_of(residual, R, S).sum()
            if syn_sum > 0:
                decoder_fails += 1  # decoder returned invalid correction
            else:
                logical_errs += 1  # residual in kernel but not zero -> logical error

        dt = time.time() - t0
        ler = logical_errs / opts.shots
        print(f'{pm:>8.4f}  {ler:>10.4f}  {decoder_fails:>6}  {dt:>8.1f}s')
        sys.stdout.flush()

if __name__ == '__main__':
    main()
