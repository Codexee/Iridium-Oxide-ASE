#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def collect_final_energies(outputs_dir: Path) -> Dict[str, float]:
    """
    Collect e_final for each base_name from *_results.json under outputs/states/**/results.
    """
    energies: Dict[str, float] = {}
    for rf in outputs_dir.glob("states/**/results/*_results.json"):
        d = load_json(rf)
        base = d.get("base_name")
        e = d.get("e_final")
        if isinstance(base, str) and isinstance(e, (int, float)):
            energies[base] = float(e)
    return energies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_set", default="inputs/state_set.json")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default="outputs/potential_window.json")
    ap.add_argument("--u_min", type=float, default=-0.2)
    ap.add_argument("--u_max", type=float, default=1.2)
    ap.add_argument("--u_steps", type=int, default=71)
    args = ap.parse_args()

    cfg = load_json(Path(args.state_set))
    thermo = cfg.get("thermo", {})
    E_H2 = thermo.get("E_H2_eV", None)
    d_corr = float(thermo.get("delta_ZPE_minus_TdS_eV", 0.0))

    energies = collect_final_energies(Path(args.outputs))
    if "slab_clean_ready" not in energies:
        raise RuntimeError("Missing clean slab result: expected base_name 'slab_clean_ready' in results.")

    E_clean = energies["slab_clean_ready"]

    # Construct U grid
    U = np.linspace(args.u_min, args.u_max, args.u_steps)

    # Define a simple CHE-like adsorption free energy proxy:
    # ΔG_H*(U) ≈ (E_slab+H - E_clean) - 0.5*E_H2 + d_corr - eU
    # Here eU is in eV if U in V and charge is 1.
    # If E_H2 is unknown, we omit the -0.5*E_H2 term (relative curves still useful).
    def deltaG(E_slabH: float) -> np.ndarray:
        base = (E_slabH - E_clean) + d_corr
        if isinstance(E_H2, (int, float)):
            base = base - 0.5 * float(E_H2)
        return base - U  # -eU in eV

    results = {
        "assumptions": {
            "E_H2_eV": E_H2,
            "delta_ZPE_minus_TdS_eV": d_corr,
            "note": "If E_H2_eV is null, curves are relative and shifted by an unknown constant."
        },
        "U_grid_V": U.tolist(),
        "states": {}
    }

    # Track H* states and hydrated test (if present)
    for base_name, E in sorted(energies.items()):
        if base_name.startswith("slab_H_o") or base_name.startswith("slab_H_hydrated_o"):
            results["states"][base_name] = {
                "E_final_eV": E,
                "deltaG_vs_U_eV": deltaG(E).tolist()
            }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2))
    print(f"Wrote {outp}")
    print(f"Found {len(results['states'])} H-containing states.")


if __name__ == "__main__":
    main()

