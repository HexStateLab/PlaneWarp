# All-Logicals GHZ on IonQ Forte

Entangling **every logical qubit** of a 4×4 `(1+x²)(1+y²)` code block into a single
logical GHZ state on a 36-qubit trapped-ion machine — and certifying it with a
two-term fidelity witness that needs only one circuit.

**Result (100 shots, raw / undecoded): F ≈ 0.845 ± 0.04 — well above the 0.5
GHZ-entanglement threshold.**

## Idea

The code block is an r×s grid of data qubits with weight-4 stabilizer checks
`Z(i,j) Z(i+2,j) Z(i,j+2) Z(i+2,j+2)`. Its tracked Z-logicals are the parities of
rows 0..r−2 and columns 0..s−2 — six logical qubits on a 4×4 periodic grid.

The experiment:

1. **Prepare** all logicals in `|0…0⟩_L`.
2. **Project into GHZ**: a single ancilla measures the joint operator
   `X_all = ∏ᵢ X_Lᵢ` (H–CX…CX–H parity gadget). The outcome `m₁ ∈ {0,1}` projects onto

   ```
   (|00…0⟩_L + (−1)^m₁ |11…1⟩_L) / √2
   ```

   The support of `X_all` — row 0 plus column 0, corner excluded — was chosen so
   it **commutes with every weight-4 check** (verified algebraically and
   empirically). This is essential: with the weight-2 gauge checks
   (`full_stabilizer=False`) the operator anticommutes with four checks and the
   syndrome rounds dephase the GHZ to nothing, even noiselessly.
3. **Survive QEC**: run `n` rounds of weight-4 Z-stabilizer measurements
   (no-reset ancillas, XOR differencing in software).
4. **Probe coherence**: a second, fresh ancilla measures `X_all` again → `m₂`.
   On a coherent GHZ state `m₂ = m₁` deterministically; on a classical mixture
   `P(m₂ = m₁) = ½`. Crucially `X_all` commutes with every pairwise `Z_i Z_j`,
   so this probe does not disturb the population statistic measured next.
5. **Read out** all data in Z and compute the six logicals.

### Witness

One circuit yields both terms:

| Term | Definition | GHZ | Product / mixture |
|---|---|---|---|
| Population `P` | P(all Z-logicals equal) | 1 | 1 (product `|0…0⟩`) |
| Coherence `C` | `2·P(m₂ = m₁) − 1` | +1 | 0 |
| **Fidelity** | `F ≈ (P + C) / 2` | 1 | ≤ 0.5 |

`F > 0.5` certifies logical GHZ entanglement. Secondary signatures: `m₁` splits
~50/50, each individual logical is maximally random (~50 % flipped), and
`P(all zero) ≈ 0.5` (a product state would give ≈ 1.0).

A common pitfall this design avoids: comparing Z-logical readouts to `m₁`
directly. `m₁` is an **X-type** outcome — it is uncorrelated with Z readout even
for a perfect GHZ state, so that "witness" reads ~0 unconditionally.

## Hardware results

4×4 periodic grid, 2 syndrome rounds, 26 qubits, 76 CX, 100 shots:

```
P (all 6 logicals agree)  = 0.850
C (m2 == m1 coherence)    = +0.840
F ≈ (P + C)/2             = 0.845   → GHZ ENTANGLED
P(all zero)               = 0.430   (≈ 0.5 expected)
per-logical flip rate     = 0.46 – 0.52
```

81/100 shots land exactly on the four ideal GHZ outcomes
(m, data) ∈ {(0, 0¹⁶), (0, S), (1, 0¹⁶), (1, S)} where S is the physical
representative of `|1…1⟩_L` (the `X_all` support pattern). Noiseless simulation
of the same circuit gives F = 1.000 through 2 rounds.

## Repository layout

| File | Purpose |
|---|---|
| `pw_opt.py` | Circuit builder (compact all-to-all layout, ≤ 36 qubits). Bell-ancilla `X_all` prep, weight-2/weight-4 checks, no-reset syndrome rounds, second coherence probe (`bell_measure=True`). |
| `ghz_logicals_v2.py` | Builds and runs the GHZ experiment (local Aer or `qiskit-ionq`), computes the witness. Also exports QASM. |
| `parse_ionq.py` | All-purpose result parser + decoder driver. Feeds raw hardware outputs to the witness and (optionally) the decoder. |
| `decoder.py` + `libplane_warp.so` | `(1+x²)(1+y²)` decoder (ffinal / rotation / multi-pass variants). |
| `ghz_4x4_r2.qasm3`, `ghz_4x4_r2.qasm` | Ready-to-submit OpenQASM 3 / 2 exports of the 4×4, 2-round circuit. |

## Usage

```bash
# inspect / simulate locally (noiseless check should give F = 1.000)
python3 ghz_logicals_v2.py --grid 4 4 --rounds 2 --dry-run
python3 ghz_logicals_v2.py --grid 4 4 --rounds 2 --backend aer --shots 2000

# export QASM for direct API submission
python3 ghz_logicals_v2.py --grid 4 4 --rounds 2 --output-qasm ghz.qasm3

# run on IonQ via qiskit-ionq (needs IONQ_API_KEY)
python3 ghz_logicals_v2.py --backend qpu.forte-1 --shots 1000

# analyze hardware results (JSON histogram, probabilities, qiskit counts,
# per-shot lists, or plain "bitstring count" text — all auto-detected)
python3 parse_ionq.py results.json --grid 4 4
python3 parse_ionq.py results.txt  --grid 4 4 --shots 100
```

`parse_ionq.py` auto-detects bit endianness by scoring both interpretations with
the witness (wrong order scrambles the structure: 0.845 vs 0.170 on our data);
force with `--bit-order q0-lsb|q0-msb`.

## Bit layout

Qubits (grid r×s, `n_anc = 4·(r/2−1)·(s/2)`):

```
q[0 .. rs−1]          data,  q[i·s + j] = site (i, j)
q[rs .. rs+n_anc−1]   syndrome ancillas
q[rs+n_anc]           bell   (m1, GHZ projection)
q[rs+n_anc+1]         bell_m (m2, coherence probe)
```

Classical registers in the QASM (Qiskit convention, last-declared leftmost in
count keys): `bell_m[1] | bell[1] | data[16] | syn_1[8] | syn_0[8]`.

## Caveats & notes

- **Mid-circuit measurement**: the scheme measures ancillas before the data.
  There is no feed-forward, so by the deferred-measurement principle the
  statistics are unchanged if the backend defers measurements — but a backend
  that returns only one bit per qubit discards the round-0 syndrome record.
  `parse_ionq.py` handles this: under no-reset the last raw ancilla value is
  the cumulative check parity, a valid final-round syndrome.
- **`full_stabilizer=True` is required** for the all-logicals GHZ. With the
  weight-2 checks, no X operator that flips a column logical commutes with the
  check group (any vertical-pair-commuting X string intersects every column an
  even number of times), so at most the row logicals can be linked.
- Reported hardware F is **raw**; with per-round syndrome records the decoder
  corrects part of the error tail and F improves.
- Decoder rotation heuristics (`default_rot`, `WEIGHT_FLOOR`) are tuned for
  large grids; on 4×4 the decode goes through the identity path.
- 100 shots ⇒ ~±0.04 statistical error on F. Use ≥ 1000 shots for publication
  numbers.

## Requirements

```
qiskit >= 1.0
qiskit-aer          # local simulation
qiskit-ionq         # hardware submission (optional)
numpy
```
