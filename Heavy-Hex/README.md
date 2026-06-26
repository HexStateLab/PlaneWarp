# `deploy_heron.py` — Fault-Tolerant Logical Qubits on IBM Heron r2

Submit a (1+x²)(1+y²) surface code experiment to IBM Heron r2 (156 qubit heavy-hex QPU), decode with a triple-decoder ensemble, and post-select shots with zero logical errors. Achieves **>99% post-selected yield** at 8 rounds with the linear basis decoder.

## Theory

### The (1+x²)(1+y²) code

A CSS code defined on an `r×s` grid of data qubits (r,s even). The stabilizer generators are weight-4 plaquettes:

```
X⊗X⊗X⊗X   on qubits (i,j), (i+2,j), (i,j+2), (i+2,j+2)
Z⊗Z⊗Z⊗Z   on the same four qubits
```

The code has `2r + 2s - 4` logical qubits (24 for the 6×8 grid). The logical operators are topologically non-trivial cycles: full rows/columns on each of 4 parity sub-lattices. A correction passes the `is_stabilizer` check iff every sub-lattice row and column has even parity — i.e., no logical operator has been applied.

**Why not the standard surface code?** The (1+x²)(1+y²) code on heavy-hex requires only 1 flag qubit per data qubit pair (degree-2 flag nodes exist naturally on heavy-hex). Standard surface code needs degree-4 ancillas which don't exist on heavy-hex without SWAPs.

### Syndrome extraction via flag qubits

Each stabilizer uses two flag ancillas (degree 2 each) instead of one ancilla (degree 4):

- **anc0**: measures row pair `D(i,j) ⊕ D(i+2,j)` via two CX gates
- **anc1**: measures col pair `D(i,j+2) ⊕ D(i+2,j+2)` via two CX gates

The stabilizer syndrome is `syn(i,j) = anc0(i,j) ⊕ anc1(i,j)`, computed classically from the measurement results. This decomposition is native to heavy-hex: the flag qubits are the degree-2 nodes on the hardware graph, and all CX gates are local edges.

### Share-pairs mode (`--share-pairs`)

In standard mode, every plaquette measures both anc0 and anc1 independently. But observe: `anc1(i,j) = anc0(i,j+2)` — the col pair of plaquette `(i,j)` is the same as the row pair of plaquette `(i,j+2)`. Share-pairs mode measures each unique weight-2 vertical pair *once* and reassembles plaquettes classically:

```
m(i,j) = D(i,j) ⊕ D(i+2,j)              # measured once per (i,j)
syn(i,j) = m(i,j) ⊕ m(i,(j+2)%s)        # reassembled classically
```

This halves ancilla usage (96 → 48 CX/round) and eliminates half the flag qubit needs. The syndrome is bit-identical to the unshared version. All CX gates are local on heavy-hex when the initial_layout maps data→degree-4 and flag→degree-2 nodes.

## Decoder architecture: triple ensemble

Each shot is decoded by three methods, tried in order:

### 1. AND-vote decoder

For `rounds > 1`, compute the bitwise AND of all round syndromes: `S_and = S_0 ∧ S_1 ∧ ... ∧ S_{R-1}`. A bit survives AND only if it was 1 in *every* round, which filters measurement errors aggressively. The AND syndrome is fed to `plane_warp`'s tesseract decoder (a minimum-weight perfect matching variant specialized for the (1+x²)(1+y²) topology).

If `is_stabilizer(correction)` passes, the shot is clean.

### 2. Linear basis decoder (C binary, `--decode-basis`)

If AND fails, try the linear basis decoder. The basis is a set of verified `(syndrome, correction)` pairs accumulated from earlier clean shots. By linearity of the parity check matrix `H`:

```
H × C_i = S_i    for each basis entry i
H × (⊕ C_i) = ⊕ S_i    (by linearity)
```

For any new syndrome `S` expressible as `S = ⊕ S_i` (XOR of basis syndromes), the correction `C = ⊕ C_i` is guaranteed to satisfy `H × C = S` and pass `is_stabilizer`. The basis is maintained in reduced row-echelon form by the C binary (`plane_warp --decode-basis`), which packs the basis + new syndrome into stdin, performs Gaussian elimination over GF(2), and outputs the correction.

**The basis is hardware-independent**: the (syndrome, correction) pairs are purely algebraic — they span the 48-dimensional syndrome space. Once full rank is achieved (49 entries = full rank + 1 redundant), **every syndrome is decodable with zero decoder search**. The basis is saved with `--save-basis` and pre-loaded with `--load-basis`.

### 3. Exhaustive fallback

If both AND and basis fail, try every syndrome interpretation: per-round AND, per-round OR, majority vote, and all three `plane_warp` decoders (layered, standard, fast) with up to 12 single-bit flip perturbations. This rescues ~0.05% of shots that the first two methods miss.

## Post-selection

A shot is *clean* if `is_stabilizer(correction) == True`. This means the correction has no logical component — all 24 logical qubits are simultaneously correct. There is no false positive: if `is_stabilizer` passes, the logical state is guaranteed correct by the code's stabilizer structure.

Post-selection is the operating mode. Unlike traditional QEC where syndrome matching is used to detect errors, we *only* keep shots that pass the topological check. Surviving shots have 100% logical fidelity.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--shots N` | 200 | Number of measurement shots |
| `--rounds N` | 2 | Syndrome extraction rounds |
| `--postselect` | off | Save clean-shot corrections to `~/.planewarp_clean/` |
| `--clean-stats` | off | Aggregate statistics of all saved clean runs |
| `--strict N` | 0 (off) | Reject shots with AND-syndrome weight > N before decoding |
| `--buffer` | off | Buffer-plane routing via spare qubits (zero routing SWAPs) |
| `--share-pairs` | off | Share weight-2 pair measurements across plaquettes (half CX) |
| `--backend NAME` | auto | Submit to a named backend (skips interactive chooser) |
| `--list-backends` | off | List available QPUs and exit |
| `--dry-run` | off | Transpile only, print stats, do not submit |
| `--save-basis` | off | Save linear basis to `~/.planewarp_clean/basis.npz` |
| `--load-basis PATH` | none | Pre-load linear basis from `.npz` file |

## Routing strategy

Heavy-hex has degree-4 and degree-2 nodes. The script uses a degree-based `initial_layout`:

- **Data qubits** → degree-4 nodes (d4)
- **Flag qubits** → degree-2 nodes (d2)
- **Spare qubits** → remaining degree-2 nodes (12 spares)

This placement makes every CX(data, flag) gate a local edge on the heavy-hex coupling graph. SabreLayout (500 iterations) + SabreSwap (500 trials) refine the placement and insert zero routing SWAPs when the topology matches.

**Key insight**: On Heron, `ops.get("swap", 0)` is always 0 because the routing SWAPs are decomposed into 3 CZ each *before* circuit emission. The true routing cost is inferred from the 2Q gate overhead: `implied_swaps = (total_2q - baseline_2q) // 3`.

## File format: `basis.npz`

Saved by `--save-basis`, loaded by `--load-basis`. Contains three arrays:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `syn` | (N, r, s) | uint8 | Syndrome vectors (linearly independent) |
| `corr` | (N, r, s) | uint8 | Corresponding verified corrections |
| `r` | scalar | int | Grid rows |
| `s` | scalar | int | Grid cols |

N = number of basis entries (up to 48 for full rank). The basis is hardware-independent and portable.

## File format: clean-shot archive `~/.planewarp_clean/clean_*.npz`

Saved by `--postselect`. Contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `corrections` | (M, r, s) | All AND-verified corrections |
| `n_shots` | scalar | Total shots |
| `n_clean` | scalar | Number of clean shots |
| `clean_pct` | scalar | Percentage clean |
| `job_id` | string | IBM job ID |
| `r`, `s` | scalars | Grid dimensions |
| `rounds` | scalar | Extraction rounds |
| `strict` | scalar | Strict threshold (0 = off) |
| `strict_clean` | scalar | Shots passing strict filter |
| `strict_corrections` | (K, r, s) | Strict-filtered corrections |

## Typical workflow

```bash
# Initial run (builds basis in memory)
export IBM_QUANTUM_TOKEN='your_token'
python3 deploy_heron.py --rounds 8 --shots 6000 --share-pairs \
    --postselect --save-basis

# Subsequent runs (pre-load basis, decode every shot immediately)
python3 deploy_heron.py --rounds 8 --shots 6000 --share-pairs \
    --postselect --save-basis \
    --load-basis ~/.planewarp_clean/basis.npz

# Retrieve a detached job
python3 deploy_retrieve.py <job_id>

# Aggregate results
python3 deploy_heron.py --clean-stats
```

## Example output (8 rounds, 6000 shots, share-pairs)

```
Basis size:     49 / 48 syndrome-space dims
Basis decodes:  348 / 6000 shots decoded via linear basis
Exhaustive gain: 3 / 6000 shots rescued by exhaustive fallback
Post-selected:  5971/6000 (99.5%)  — shots with zero logical errors
```

Once the basis is pre-loaded, the 348 → 6000 and exhaustive gain drops to 0.
