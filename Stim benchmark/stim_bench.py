#!/usr/bin/env python3
"""
Comprehensive STIM bench for the plane_warp decoder.

For each circuit type it runs the decoder under a small panel of settings and
reports the logical-observable error rate with Wilson 95% CIs, so we can see
(a) where plain decoding already wins, and (b) where the confidence cap earns
its keep. Every shot is decoded under every setting on the SAME sample, so the
comparison within a circuit is paired.

Best-settings axis (the one this project established):
  - plain  (--decode)            : full nullspace solver, escape on, no cap.
                                   Best when the syndrome is TRUSTWORTHY.
  - cap-auto R (--cap-auto R)     : abstain when the correction is implausibly
                                   heavy for the noise. Best when the syndrome
                                   is UNTRUSTWORTHY (basis-mismatched / meas-
                                   dominated). cap = round(R*n + 2*sqrt(R*n(1-R))).

Reading the FT column:
  corrects  : decode CI strictly below baseline  -> real error correction.
  =baseline : decode CI overlaps baseline         -> abstains to baseline; no
                                                    uplift is possible because
                                                    the syndrome carries no
                                                    usable info (e.g. CNOT).
  WORSE     : decode CI strictly above baseline    -> decoder is hurting you.
"""
import sys, os, math, time
import subprocess
import numpy as np
import stim

DECODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_warp")


def build_circuit(R, S, p_g, p_meas, ctype, rounds=5):
    """Circuit-level noise models on an R×S torus, plus-shaped weight-4 Z checks
    at stride 2. Bit layout per shot: rounds*N ancilla syndrome bits, then N
    final data bits. ctype: cz | cn (CNOT) | phenom | correlated | asymmetric."""
    N = R * S
    ND = NA = N
    c = stim.Circuit()
    rng = np.random.RandomState(12345)

    if ctype == "phenom":
        # data X errors per round + measurement flips; CZ readout, no gate noise
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
                        c.append("CZ", [anc, q])
            c.append("H", range(ND, ND + NA))
            c.append("M", range(ND, ND + NA))
            c.append("X_ERROR", list(range(ND, ND + NA)), p_meas)
        c.append("M", range(ND))
        return c, N

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

# (R, S, p_g, p_meas, ctype, shots, label)
CONFIGS = [
    (6,  6,  0.0005, 0.001, "cz",         2000, "CZ-based  (basis-matched)"),
    (6,  6,  0.0008, 0.001, "phenom",     2000, "phenomenological (data+meas)"),
    (6,  6,  0.0005, 0.001, "correlated", 2000, "correlated-pair"),
    (6,  6,  0.0005, 0.001, "asymmetric", 1000, "asymmetric (10x hot sub)"),
    (6,  6,  0.0005, 0.001, "cn",         1000, "CNOT-based (basis-mismatched)"),
    (20, 20, 0.0004, 0.001, "cz",          300, "CZ-based  (basis-matched)"),
    (20, 20, 0.0002, 0.001, "cn",          300, "CNOT-based (basis-mismatched)"),
]

# decoder setting panel: (name, config-flags-before-action). action is --decode.
def cap_value(R, n):
    mu = R * n
    return max(1, round(mu + 2.0 * math.sqrt(mu * (1.0 - R))))

CAP_RATES = [0.015, 0.030]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def decode(syn, R, S, flags):
    return subprocess.run([DECODER, str(R), str(S), *flags],
                          input=syn, capture_output=True, timeout=180).stdout


def ft_mark(dec_lo, dec_hi, base_lo, base_hi):
    if dec_hi < base_lo:
        return "corrects"
    if dec_lo > base_hi:
        return "WORSE"
    return "=baseline"


def run_config(R, S, p_g, p_meas, ctype, T, label):
    n = R * S
    c, N = build_circuit(R, S, p_g, p_meas, ctype)
    shots = c.compile_sampler(seed=2024).sample(shots=T).astype(np.uint8)
    obs = list(range(0, S, 2))

    # precompute per-shot syndrome bytes + baseline observable parity
    syns, ovs = [], []
    base_err = 0
    for t in range(T):
        shot = shots[t]
        syns.append(bytes(shot[4 * N:5 * N]))
        ov = int(sum(int(shot[5 * N + q]) for q in obs) % 2)
        ovs.append(ov)
        base_err += ov
    base_p = base_err / T
    base_lo, base_hi = wilson(base_err, T)

    panel = [("plain", ["--decode"])]
    for R_ in CAP_RATES:
        panel.append((f"cap-auto {R_:.3f} (cap={cap_value(R_, n)})",
                      ["--cap-auto", f"{R_}", "--decode"]))

    results = []
    for name, flags in panel:
        err = 0
        for t in range(T):
            cr = decode(syns[t], R, S, flags)
            dp = sum(int(cr[q]) for q in obs) % 2 if len(cr) >= n else 0
            if (ovs[t] ^ dp) == 1:
                err += 1
        lo, hi = wilson(err, T)
        results.append((name, err / T, lo, hi))

    # best = lowest point estimate
    best_i = min(range(len(results)), key=lambda i: results[i][1])

    # ---- print block ----
    print(f"\n── {label} ──  grid {R}×{S} (n={n}), p_g={p_g:.1e}, "
          f"p_meas={p_meas:.0e}, 5 rounds, {T} shots")
    print(f"     {'baseline (no correction)':<30s} {base_p*100:6.2f}%  "
          f"[{base_lo*100:5.2f}, {base_hi*100:5.2f}]")
    for i, (name, p, lo, hi) in enumerate(results):
        star = "*" if i == best_i else " "
        mark = ft_mark(lo, hi, base_lo, base_hi)
        print(f"  {star}  {name:<30s} {p*100:6.2f}%  "
              f"[{lo*100:5.2f}, {hi*100:5.2f}]  {mark}")

    best_name, best_p, best_lo, best_hi = results[best_i]
    best_mark = ft_mark(best_lo, best_hi, base_lo, base_hi)
    return {
        "label": label, "grid": f"{R}×{S}", "ctype": ctype, "p_g": p_g,
        "base": base_p, "best_name": best_name, "best_p": best_p,
        "best_mark": best_mark,
        "plain_p": results[0][1],
    }


def main():
    print("=" * 92)
    print(" Comprehensive plane_warp Bench — STIM circuit-level noise, 5 rounds")
    print(" Panel per circuit:  plain --decode   vs   --cap-auto at two calibrated rates")
    print(" LER = logical-observable error over the shown shots, [Wilson 95% CI].  * = best setting.")
    print("=" * 92)

    t0 = time.time()
    summary = []
    for cfg in CONFIGS:
        summary.append(run_config(*cfg))

    print("\n" + "=" * 92)
    print(" SUMMARY — best setting per circuit")
    print("=" * 92)
    print(f"  {'circuit':<31s} {'grid':>6s} {'p_g':>7s}  {'baseline':>8s}  "
          f"{'best setting':<22s} {'LER':>7s}  {'result':<10s}")
    print(f"  {'─'*31} {'─'*6} {'─'*7}  {'─'*8}  {'─'*22} {'─'*7}  {'─'*10}")
    for s in summary:
        print(f"  {s['label']:<31s} {s['grid']:>6s} {s['p_g']:7.1e}  "
              f"{s['base']*100:7.2f}%  {s['best_name']:<22s} "
              f"{s['best_p']*100:6.2f}%  {s['best_mark']:<10s}")

    print("\n" + "─" * 92)
    print(" How to read this")
    print("─" * 92)
    print(" • Basis-matched / trustworthy syndromes (CZ, phenom, correlated, asymmetric):")
    print("   plain decode corrects below baseline. The cap only ever abstains here, so it")
    print("   ties or slightly trails plain — engaging it on clean noise is a mild tax.")
    print(" • Basis-mismatched syndrome (CNOT): plain decode is catastrophic (~46%) because it")
    print("   trusts a syndrome that reports the wrong basis. cap-auto recognises the implausibly")
    print("   heavy correction and abstains, recovering to ≈baseline. That is damage control, not")
    print("   correction: a useless syndrome cannot be turned into uplift, so 'best' = baseline.")
    print(" • Net: the decoder corrects wherever the syndrome is honest, and the cap is the safety")
    print("   net that stops it being fooled where the syndrome is not. Gate the cap on a trust")
    print("   signal; never leave it on for data-dominated noise.")
    print(" • Scope: this harness is single-frame — it decodes the last round's syndrome against the")
    print("   final data observable. That under-serves measurement-dominated noise (phenom), where the")
    print("   right tool is a spacetime/multi-round decode over the full syndrome history; testing that")
    print("   fairly needs a detector-based harness and is out of scope here.")
    print(f"\n (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
