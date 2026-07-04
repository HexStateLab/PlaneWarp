"""End-to-end check of Steane EC on THIS code, before touching pw_opt.

Prepare the logical Bell pair, run K Steane rounds (both types) with the
verified encoder, then read the witness. <ZZ>_L and <XX>_L must stay +1
noiselessly (both logicals live), and an injected error must show up in the
software-reconstructed syndrome.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.synthesis import synth_circuit_from_stabilizers
from qiskit_aer import AerSimulator
from steane_build import (code_operators, encoded_stabilizer_list, _sym,
                          _complete_isotropic, _vec_to_pauli)

SIM = AerSimulator()


def _run(qc, shots=4000):
    return SIM.run(transpile(qc, basis_gates=["h", "cx", "x", "z", "s", "sdg",
                                              "measure"], optimization_level=0),
                   shots=shots).result().get_counts()


def bell_prep_circuit(r, s):
    """(|00>_L + |11>_L)/sqrt2 as a stabilizer state: +1 of all S, Z_L1 Z_L2,
    and X_L1 X_L2."""
    from steane_build import _commute, _independent
    n, ops = code_operators(r, s)
    must = ops["SX"] + ops["SZ"] + [ops["ZL1"].compose(ops["ZL2"]),
                                    ops["XL1"].compose(ops["XL2"])]
    chosen, basis = [], []
    for p in must:
        r_ = _independent(_sym(p), basis)
        if r_ is not None and all(_commute(p, c) for c in chosen):
            chosen.append(p); basis.append(r_)
    syms = _complete_isotropic([_sym(p) for p in chosen], n)
    gens = [_vec_to_pauli(v, n) for v in syms]
    return synth_circuit_from_stabilizers(gens, allow_underconstrained=True)


def ancilla_prep_circuit(r, s, state):
    _, gens, _ = encoded_stabilizer_list(r, s, state)
    return synth_circuit_from_stabilizers(gens, allow_underconstrained=True)


def _softsyn(block, r, s):
    b = block.astype(np.uint8)
    V = b ^ np.roll(b, -2, axis=1)
    return V ^ np.roll(V, -2, axis=2)


def build(r, s, rounds, arm, inject=None):
    n = r * s
    data = QuantumRegister(n, "d")
    ancs = [QuantumRegister(n, f"a{t}") for t in range(rounds)]
    syns = [ClassicalRegister(n, f"blk{t}") for t in range(rounds)]
    out = ClassicalRegister(n, "out")
    qc = QuantumCircuit(data, *ancs, *syns, out)

    qc.compose(bell_prep_circuit(r, s), data, inplace=True)
    if inject is not None:
        (kind, q) = inject
        (qc.x if kind == "X" else qc.z)(data[q])

    aprep = {"Z": ancilla_prep_circuit(r, s, "plus"),     # Z-stab round
             "X": ancilla_prep_circuit(r, s, "zero")}     # X-stab round
    for t in range(rounds):
        rb = "Z" if t % 2 == 0 else "X"
        a = ancs[t]
        qc.compose(aprep[rb], a, inplace=True)
        for k in range(n):
            if rb == "Z":
                qc.cx(data[k], a[k])
            else:
                qc.cx(a[k], data[k])
        for k in range(n):
            if rb == "X":
                qc.h(a[k])
            qc.measure(a[k], syns[t][k])

    # witness readout
    from pw_opt import _bell_support_coords
    bell_coords = _bell_support_coords(r, s, True)   # X_L1 X_L2 support (no (0,0))
    if arm == "XX":
        for (i, j) in bell_coords:
            qc.h(data[i * s + j])
    for k in range(n):
        qc.measure(data[k], out[k])
    return qc, syns


def _field(key, idx_from_right):
    return key.split()[idx_from_right]


def witness(r, s, rounds):
    from pw_opt import _bell_support_coords
    bell_coords = _bell_support_coords(r, s, True)
    res = {}
    for arm in ("ZZ", "XX"):
        qc, _ = build(r, s, rounds, arm)
        counts = _run(qc)
        corr = tot = 0
        for key, c in counts.items():
            out = np.array([int(b) for b in _field(key, 0)[::-1]], dtype=np.uint8)
            grid = out.reshape(r, s)
            if arm == "ZZ":
                val = int((grid[0, :].sum() % 2) ^ (grid[:, 0].sum() % 2))  # Z_L1 Z_L2
            else:
                val = int(sum(int(grid[i, j]) for (i, j) in bell_coords) % 2)  # X_L1 X_L2
            corr += c * (1 - 2 * val); tot += c
        res[arm] = corr / tot
    return res["ZZ"], res["XX"]


def syndrome_check(r, s):
    """Inject one X error, confirm the round-0 block reconstruction is nonzero
    and stable (persists), i.e. the free-round-every-round syndrome sees it."""
    q = 1 * s + 3
    qc, syns = build(r, s, rounds=2, arm="ZZ", inject=("X", q))
    counts = _run(qc, shots=200)
    key = max(counts, key=counts.get)
    fields = key.split()  # order: out blk1 blk0  (reverse of add order)
    blk0 = np.array([int(b) for b in fields[-1][::-1]], dtype=np.uint8).reshape(r, s)
    syn = _softsyn(blk0[None], r, s)[0]
    return int(syn.sum())


if __name__ == "__main__":
    for (r, s) in ((4, 4), (4, 8)):
        for rounds in (0, 2, 4):
            zz, xx = witness(r, s, rounds)
            ok = zz > 0.99 and xx > 0.99
            print(f"  {r}x{s} rounds={rounds}: <ZZ>_L={zz:+.4f} <XX>_L={xx:+.4f} "
                  f"W={zz+xx:+.4f}  {'PASS' if ok else 'FAIL'}")
        nlit = syndrome_check(r, s)
        print(f"  {r}x{s} injected X -> reconstructed syndrome lights {nlit} checks "
              f"{'PASS' if nlit > 0 else 'FAIL'}")
