#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigsh

PAULI = {
    "I": csr_matrix(np.array([[1, 0], [0, 1]], dtype=complex)),
    "X": csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    "Y": csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    "Z": csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
}


def popcount(x: int) -> int:
    return int(x.bit_count())


def build_term_matrix(n_qubits: int, paulis):
    ops = ["I"] * n_qubits
    for q, p in paulis:
        ops[int(q)] = p
    mat = PAULI[ops[0]]
    for op in ops[1:]:
        mat = kron(mat, PAULI[op], format="csr")
    return mat


def main():
    ap = argparse.ArgumentParser(description="Exact diagonalization of exported qubit Hamiltonian JSON.")
    ap.add_argument("json_path", help="Path to qubit_hamiltonian_*.json")
    ap.add_argument("--n-electrons", type=int, default=None, help="Restrict to fixed-electron subspace")
    ap.add_argument("--fermionic-json", default="", help="Optional fermionic_active_space.json to read n_active_electrons")
    ap.add_argument("--k", type=int, default=4, help="Number of lowest eigenvalues to compute")
    ap.add_argument("--dense-threshold", type=int, default=4096, help="Use dense diagonalization if subspace dimension <= threshold")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    path = Path(args.json_path)
    data = json.loads(path.read_text())
    n_qubits = int(data["n_qubits"])
    terms = data["terms"]

    n_electrons = args.n_electrons
    if n_electrons is None and args.fermionic_json:
        fdata = json.loads(Path(args.fermionic_json).read_text())
        n_electrons = int(fdata["n_active_electrons"])

    dim = 2 ** n_qubits
    H = csr_matrix((dim, dim), dtype=complex)
    for term in terms:
        coeff = complex(float(term["coeff_real"]), float(term["coeff_imag"]))
        if abs(coeff) == 0:
            continue
        paulis = term["paulis"]
        if len(paulis) == 0:
            H = H + coeff * identity(dim, format="csr", dtype=complex)
        else:
            H = H + coeff * build_term_matrix(n_qubits, paulis)

    full_dim = dim
    subspace_dim = dim

    if n_electrons is not None:
        basis = np.array([i for i in range(dim) if popcount(i) == n_electrons], dtype=np.int64)
        subspace_dim = len(basis)
        if subspace_dim == 0:
            raise ValueError(f"No basis states found for n_qubits={n_qubits}, n_electrons={n_electrons}")
        print(f"Restricting to fixed-electron sector: N={n_electrons}, dim={subspace_dim} / {full_dim}")
        H = H[basis][:, basis]

    if subspace_dim <= args.dense_threshold:
        evals = np.linalg.eigvalsh(H.toarray())
        evals = np.real_if_close(evals)
        evals = np.sort(evals)[:args.k]
        solver = "dense"
    else:
        k = min(args.k, subspace_dim - 2)
        evals = eigsh(H, k=k, which="SA", return_eigenvectors=False)
        evals = np.real_if_close(evals)
        evals = np.sort(evals)
        solver = "eigsh"

    out = {
        "source_json": str(path),
        "site": data.get("site"),
        "mapping": data.get("mapping"),
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "full_hilbert_dim": int(full_dim),
        "diagonalized_dim": int(subspace_dim),
        "solver": solver,
        "ground_state_energy_hartree": float(np.real(evals[0])),
        "lowest_eigenvalues_hartree": [float(np.real(x)) for x in evals[:args.k]],
    }

    outpath = Path(args.out) if args.out else path.with_name(path.stem + "_exact_diag.json")
    outpath.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
