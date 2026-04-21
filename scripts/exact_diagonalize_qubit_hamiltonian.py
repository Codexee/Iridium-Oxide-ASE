#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh

PAULI = {
    "I": csr_matrix(np.array([[1,0],[0,1]], dtype=complex)),
    "X": csr_matrix(np.array([[0,1],[1,0]], dtype=complex)),
    "Y": csr_matrix(np.array([[0,-1j],[1j,0]], dtype=complex)),
    "Z": csr_matrix(np.array([[1,0],[0,-1]], dtype=complex)),
}

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
    ap.add_argument("--k", type=int, default=4, help="Number of lowest eigenvalues to compute")
    ap.add_argument("--dense-threshold", type=int, default=12, help="Use dense diag if n_qubits <= threshold")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    path = Path(args.json_path)
    data = json.loads(path.read_text())
    n_qubits = int(data["n_qubits"])
    terms = data["terms"]

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

    if n_qubits <= args.dense_threshold:
        evals, evecs = np.linalg.eigh(H.toarray())
        evals = np.real_if_close(evals)
        idx = np.argsort(evals)
        evals = evals[idx][:args.k]
    else:
        evals, evecs = eigsh(H, k=min(args.k, dim - 2), which="SA")
        evals = np.real_if_close(evals)
        evals = np.sort(evals)

    out = {
        "source_json": str(path),
        "site": data.get("site"),
        "mapping": data.get("mapping"),
        "n_qubits": n_qubits,
        "ground_state_energy_hartree": float(np.real(evals[0])),
        "lowest_eigenvalues_hartree": [float(np.real(x)) for x in evals[:args.k]],
    }
    outpath = Path(args.out) if args.out else path.with_name(path.stem + "_exact_diag.json")
    outpath.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
