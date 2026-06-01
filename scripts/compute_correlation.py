#!/usr/bin/env python3
"""
Compute the static correlation energy of the active space.

E_corr = E_ground - E_HF_active

where E_ground is the CASCI ground-state eigenvalue from exact diagonalization
and E_HF_active is the expectation value of the active-space Hamiltonian on the
closed-shell HF determinant restricted to active orbitals.

Reads:
  - fermionic_active_space.npz   (h1, h2, ecore, active_mos, n_active_electrons)
  - the corresponding exact-diag JSON (ground_state_energy_hartree)
  - optionally, the original PySCF SCF log to recover mo_occ on active MOs
    (we infer it from n_active_electrons and assume the lowest-index orbitals
    in the active set are doubly occupied; for a single closed-shell HF
    reference this is correct.)

Usage:
  python compute_correlation.py path/to/site/fermionic_active_space.npz \\
      path/to/site/qubit_hamiltonian_jw_exact_diag.json \\
      --out path/to/site/correlation.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def hf_energy_active_space(h1: np.ndarray,
                           h2: np.ndarray,
                           ecore: float,
                           n_active_electrons: int) -> dict:
    """
    Closed-shell HF energy of the active-space Hamiltonian.

    Assumes the lowest n_active_electrons/2 spatial active orbitals are doubly
    occupied. This is the standard convention for a closed-shell active space
    built around HOMO-LUMO.

    h2 is in chemist notation (pq|rs).

    Returns a dict with components for traceability.
    """
    if n_active_electrons % 2 != 0:
        raise ValueError(
            f"This routine assumes closed-shell active spaces (even electron "
            f"count). Got n_active_electrons = {n_active_electrons}."
        )

    n_occ = n_active_electrons // 2
    n_orb = h1.shape[0]
    if n_occ > n_orb:
        raise ValueError(
            f"More occupied orbitals ({n_occ}) than active orbitals ({n_orb})."
        )

    occ = list(range(n_occ))

    one_body = 2.0 * sum(h1[i, i] for i in occ)

    coulomb = 0.0
    exchange = 0.0
    for i in occ:
        for j in occ:
            coulomb += 2.0 * h2[i, i, j, j]   # 2 * (ii|jj)
            exchange += h2[i, j, j, i]        # (ij|ji)
    two_body = coulomb - exchange

    e_hf = float(ecore + one_body + two_body)

    return {
        "ecore": float(ecore),
        "one_body": float(one_body),
        "two_body_coulomb": float(coulomb),
        "two_body_exchange": float(exchange),
        "two_body_total": float(two_body),
        "E_HF_active": e_hf,
        "n_active_electrons": int(n_active_electrons),
        "n_occupied_spatial": int(n_occ),
        "n_active_orbitals": int(n_orb),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="Path to fermionic_active_space.npz")
    ap.add_argument("diag_json", help="Path to *_exact_diag.json")
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    h1 = npz["h1"]
    h2 = npz["h2"]
    ecore = float(npz["ecore"])
    n_active_electrons = int(npz["n_active_electrons"])

    diag = json.loads(Path(args.diag_json).read_text())
    e_ground = float(diag["ground_state_energy_hartree"])

    hf_components = hf_energy_active_space(h1, h2, ecore, n_active_electrons)
    e_hf = hf_components["E_HF_active"]
    e_corr = e_ground - e_hf

    out = {
        "site": diag.get("site"),
        "source_npz": str(args.npz),
        "source_diag": str(args.diag_json),
        "E_ground_hartree": e_ground,
        "E_HF_active_hartree": e_hf,
        "E_correlation_hartree": float(e_corr),
        "E_correlation_eV": float(e_corr * 27.211386245988),
        "hf_breakdown": hf_components,
        "definition": (
            "E_corr = E_ground - E_HF_active, where E_HF_active is the "
            "expectation value of the active-space second-quantized "
            "Hamiltonian on the closed-shell HF determinant of the active "
            "orbitals (lowest n_active_electrons/2 spatial orbitals doubly "
            "occupied). Chemist notation (pq|rs) assumed for h2."
        ),
    }

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"Wrote: {args.out}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
