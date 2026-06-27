#!/usr/bin/env python3
"""
Build three basis files for linear-basis decoding of the (1+x²)(1+y²) code:

  basis_16.npz    16 verifiable (syndrome, correction) pairs — H*C=S AND is_stabilizer(C)
  basis_24.npz    24 synthetic pairs spanning the full column space — H*C=S only
  inject_16.npz   16 injection patterns (X gates) matching basis_16, for hardware verification

Usage:
  python3 build_basis.py                    # writes to ~/.planewarp_clean/
  python3 build_basis.py --outdir /path     # write elsewhere
"""
import sys, os, numpy as np
from pathlib import Path

# Add project root for pw_qiskit import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pw_qiskit import PlaneWarp

R, S = 6, 8
N = R * S


def build_H(pw):
    """Return 48×48 uint8 binary matrix: H[:, q] = syndrome_of(e_q)."""
    H = np.zeros((N, N), dtype=np.uint8)
    for q in range(N):
        e = np.zeros((R, S), dtype=np.uint8)
        e.ravel()[q] = 1
        H[:, q] = pw.syndrome_of(e).ravel()
    return H


def rref_basis(H):
    """Return (pivot_cols, nullspace_matrix) for GF(2) matrix H.

    pivot_cols: list of column indices forming a basis of Col(H).
    nullspace:  (N - rank, N) array whose rows span ker(H).
    """
    A = H.copy().astype(np.uint8)
    m, n = A.shape
    pivots, rank = [], 0
    for col in range(n):
        nz = np.where(A[rank:, col])[0]
        if len(nz) == 0:
            continue
        pv = nz[0] + rank
        if pv != rank:
            A[[rank, pv]] = A[[pv, rank]]
        pivots.append(col)
        for r2 in range(m):
            if r2 != rank and A[r2, col]:
                A[r2] ^= A[rank]
        rank += 1
    free = [c for c in range(n) if c not in pivots]
    null = np.zeros((len(free), n), dtype=np.uint8)
    for ki, fc in enumerate(free):
        v = np.zeros(n, dtype=np.uint8)
        v[fc] = 1
        for pi, pc in enumerate(pivots):
            val = np.uint8(0)
            for fc2 in free:
                if A[pi, fc2]:
                    val ^= v[fc2]
            v[pc] = val
        null[ki] = v
    return pivots, null


def build_all(pw, H, pivots):
    """Build basis_16, basis_24, inject_16.

    Returns dict of npz-ready arrays.
    """
    # --- 24-dim synthetic basis (full column space) ---
    # Each entry: syn = H column at pivot q, corr = e_q (single-qubit error)
    syn_24 = np.zeros((len(pivots), N), dtype=np.uint8)
    corr_24 = np.zeros((len(pivots), N), dtype=np.uint8)
    for i, pc in enumerate(pivots):
        syn_24[i] = H[:, pc]
        e = np.zeros(N, dtype=np.uint8)
        e[pc] = 1
        corr_24[i] = e

    # --- 16-dim verifiable basis (H*C=S AND is_stabilizer) ---
    # Build nullspace of stabilizer generators, then find independent
    # syndrome vectors whose corrections are stabilizers.
    # Stabilizer generators: X-type row/column checks within each 2×2 block.
    stab_gen = np.zeros((28, N), dtype=np.uint8)
    ci = 0
    for px in (0, 1):
        for py in (0, 1):
            hr2, hs2 = R // 2, S // 2
            for si in range(hr2):
                row = np.zeros(N, dtype=np.uint8)
                for sj in range(hs2):
                    row[(px + 2 * si) * S + (py + 2 * sj)] = 1
                stab_gen[ci] = row
                ci += 1
            for sj in range(hs2):
                col = np.zeros(N, dtype=np.uint8)
                for si in range(hr2):
                    col[(px + 2 * si) * S + (py + 2 * sj)] = 1
                stab_gen[ci] = col
                ci += 1

    # Nullspace of stabilizer generators = stabilizer-preserving corrections
    _, stab_space = rref_basis(stab_gen)  # (24, 48)

    # Map through H: all_S[i] = syndrome of stabilizer-preserving correction i
    all_S = (H @ stab_space.T) & 1  # (48, 24)

    # Find 16 linearly independent columns of all_S via RREF
    B = all_S.copy()
    pivot_cols_16 = []
    for r2 in range(48):
        if len(pivot_cols_16) >= 16:
            break
        for col in range(24):
            if col in pivot_cols_16:
                continue
            if B[r2, col]:
                for r3 in range(48):
                    if r3 != r2 and B[r3, col]:
                        B[r3] ^= B[r2]
                pivot_cols_16.append(col)
                break

    syn_16 = np.zeros((16, N), dtype=np.uint8)
    corr_16 = np.zeros((16, N), dtype=np.uint8)
    for i, ci in enumerate(pivot_cols_16):
        C_2d = stab_space[ci].reshape(R, S)
        Sval = all_S[:, ci]
        assert pw.is_stabilizer(C_2d), f"Entry {i} not a stabilizer"
        assert np.array_equal(pw.syndrome_of(C_2d).ravel(), Sval), f"Entry {i} H*C != S"
        syn_16[i] = Sval
        corr_16[i] = C_2d.ravel()

    # --- Injection patterns matching basis_16 ---
    inj_errors = corr_16.copy().reshape(16, R, S)  # identical X patterns

    return {
        'syn_24': syn_24,
        'corr_24': corr_24,
        'syn_16': syn_16,
        'corr_16': corr_16,
        'inj_errors': inj_errors,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Build basis files for (1+x²)(1+y²) linear decoder')
    ap.add_argument('--outdir', type=str, default=None,
                    help='Output directory (default: ~/.planewarp_clean)')
    args = ap.parse_args()

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path.home() / '.planewarp_clean'
    outdir.mkdir(parents=True, exist_ok=True)

    print("Building PlaneWarp code object...")
    pw = PlaneWarp()

    print("Building syndrome matrix H...")
    H = build_H(pw)

    print("Computing RREF to find pivot columns...")
    pivots, _ = rref_basis(H)
    print(f"  rank(H) = {len(pivots)}")

    print("Building basis files...")
    data = build_all(pw, H, pivots)

    # Save basis_24.npz
    np.savez_compressed(outdir / 'basis_24.npz',
                        syn=data['syn_24'], corr=data['corr_24'],
                        r=R, s=S)
    print(f"  basis_24.npz: {len(data['syn_24'])} entries (full column space, rank {len(pivots)})")

    # Save basis_16.npz
    np.savez_compressed(outdir / 'basis_16.npz',
                        syn=data['syn_16'], corr=data['corr_16'],
                        r=R, s=S)
    print(f"  basis_16.npz: {len(data['syn_16'])} entries (verifiable: H*C=S + is_stabilizer)")

    # Save inject_16.npz
    np.savez_compressed(outdir / 'inject_16.npz',
                        errors=data['inj_errors'],
                        rounds=1, shots_inject=200)
    print(f"  inject_16.npz: {len(data['inj_errors'])} injection patterns (matching basis_16)")

    # Quick sanity checks
    print("\nSanity checks:")
    b24 = np.load(outdir / 'basis_24.npz')
    assert np.array_equal(b24['syn'], data['syn_24'])
    print("  ✓ basis_24.npz loads correctly")

    b16 = np.load(outdir / 'basis_16.npz')
    assert np.array_equal(b16['syn'], data['syn_16'])
    print("  ✓ basis_16.npz loads correctly")

    inj = np.load(outdir / 'inject_16.npz')
    assert np.array_equal(inj['errors'], data['inj_errors'])
    print("  ✓ inject_16.npz loads correctly")

    # Verify all basis_16 entries pass corr_matches
    for i in range(16):
        C = data['corr_16'][i].reshape(R, S)
        Sval = data['syn_16'][i]
        assert pw.is_stabilizer(C), f"  basis_16[{i}]: not a stabilizer"
        assert np.array_equal(pw.syndrome_of(C).ravel(), Sval), f"  basis_16[{i}]: H*C != S"
    print(f"  ✓ All {len(data['syn_16'])} basis_16 entries satisfy H*C=S + is_stabilizer")

    # Verify all basis_24 entries satisfy H*C=S
    for i in range(len(pivots)):
        C = data['corr_24'][i].reshape(R, S)
        Sval = data['syn_24'][i]
        assert np.array_equal(pw.syndrome_of(C).ravel(), Sval), f"  basis_24[{i}]: H*C != S"
    print(f"  ✓ All {len(data['syn_24'])} basis_24 entries satisfy H*C=S")

    print(f"\nFiles written to {outdir}/")
    print("Usage in deploy_heron.py:")
    print(f"  --load-basis {outdir}/basis_16.npz")
    print(f"  --inject {outdir}/inject_16.npz")


if __name__ == '__main__':
    main()
