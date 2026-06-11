#!/usr/bin/env python3
"""
Export a qubit Hamiltonian JSON from an active-space integral NPZ.

Input: fermionic_active_space.npz containing:
  - h1: (n,n) spatial one-electron integrals
  - h2: (n,n,n,n) spatial two-electron integrals (chemist/physicist consistent with ao2mo; we treat as (pq|rs))
  - ecore: float

Output:
  qubit_hamiltonian_<mapping>.json with Pauli strings + coeffs

Requires:
  pip install openfermion
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def json_sanitize(obj):
    """Convert numpy scalars/arrays into plain Python types for JSON."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    return obj

def infer_n_qubits_from_qubitop(qubit_op) -> int:
    """Infer number of qubits from the largest qubit index appearing in terms."""
    max_q = -1
    for term in qubit_op.terms.keys():
        # term is a tuple like ((q,'X'), (q2,'Z'), ...)
        for q, _ in term:
            if q > max_q:
                max_q = q
    return max_q + 1  # if no terms, returns 0

def spatial_to_spin_orbital(h1_spatial: np.ndarray, h2_spatial: np.ndarray):
    """
    Expand spatial integrals to spin-orbital integrals.
    h2_spatial is in chemist notation (pq|rs).
    OpenFermion InteractionOperator expects physicist notation <pq|rs>.
    Conversion: <pq|rs> = (pr|qs) = h2_spatial[p,r,q,s]
    """
    n = h1_spatial.shape[0]
    nso = 2 * n
    h1 = np.zeros((nso, nso), dtype=float)
    h2 = np.zeros((nso, nso, nso, nso), dtype=float)

    def so(p, spin):
        return 2 * p + spin

    # One-electron: block-diagonal in spin
    for p in range(n):
        for q in range(n):
            v = h1_spatial[p, q]
            h1[so(p,0), so(q,0)] = v
            h1[so(p,1), so(q,1)] = v

    # Two-electron: chemist (pq|rs) -> physicist <pq|rs> = (pr|qs)
    # h2_so[so(p,sp), so(q,sq), so(r,sp), so(s,sq)] = h2_spatial[p,r,q,s]
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    # OpenFermion InteractionOperator term a_p^dag a_q^dag a_r a_s,
                    # coefficient 0.5 * <pq|sr>_phys = 0.5 * (ps|qr)_chem.
                    v = 0.5 * h2_spatial[p, s, q, r]
                    for sp in (0, 1):
                        for sq in (0, 1):
                            h2[so(p,sp), so(q,sq), so(r,sq), so(s,sp)] = v
    return h1, h2
  
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to fermionic_active_space.npz")
    ap.add_argument("--mapping", choices=["jw", "bk"], default="jw")
    ap.add_argument("--out", default="", help="Output JSON path (optional)")
    args = ap.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)
    h1_sp = data["h1"]
    h2_sp = data["h2"]
    ecore = float(data["ecore"])

    site = npz_path.parent.name

    # Build spin-orbital integrals
    h1_so, h2_so = spatial_to_spin_orbital(h1_sp, h2_sp)

    try:
        from openfermion import InteractionOperator
        from openfermion.transforms import jordan_wigner, bravyi_kitaev
    except Exception as e:
        raise RuntimeError("OpenFermion not installed. Try: pip install openfermion") from e

    interaction = InteractionOperator(ecore, h1_so, h2_so)
    qubit_op = jordan_wigner(interaction) if args.mapping == "jw" else bravyi_kitaev(interaction)

    # Serialize Pauli terms
    terms = []
    for term, coeff in qubit_op.terms.items():
        terms.append({
            "paulis": [(int(q), str(p)) for (q, p) in term],
            "coeff_real": float(np.real(coeff)),
            "coeff_imag": float(np.imag(coeff)),
        })

    outpath = Path(args.out) if args.out else (npz_path.parent / f"qubit_hamiltonian_{args.mapping}.json")

    payload = {
        "site": site,
        "mapping": args.mapping,
        "n_spatial_orbitals": int(h1_sp.shape[0]),
        "n_spin_orbitals": int(h1_so.shape[0]),
        "n_qubits": int(infer_n_qubits_from_qubitop(qubit_op)),
        "ecore": ecore,
        "terms": terms,
        "provenance": {
            "npz": str(npz_path),
            "charge": int(data["charge"]) if "charge" in data else None,
            "spin": int(data["spin"]) if "spin" in data else None,
            "basis": str(data["basis"]) if "basis" in data else None,
            "method": str(data["method"]) if "method" in data else None,
        }
    }

    outpath.write_text(json.dumps(json_sanitize(payload), indent=2))
    print(f"Wrote: {outpath}")

if __name__ == "__main__":
    main()
