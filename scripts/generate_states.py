#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from ase import Atom
from ase.constraints import FixAtoms
from ase.io import read, write


def load_state_set(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text())
    return data


def surface_oxygen_indices(atoms, threshold_A: float) -> Tuple[List[int], float]:
    pos = atoms.get_positions()
    z_top = float(pos[:, 2].max())
    symbols = atoms.get_chemical_symbols()
    surface_O = [
        i for i, s in enumerate(symbols)
        if s == "O" and (z_top - pos[i, 2]) < threshold_A
    ]
    return surface_O, z_top


def freeze_bottom(atoms, z_freeze_A: float) -> None:
    z = atoms.get_positions()[:, 2]
    mask = z < z_freeze_A
    atoms.set_constraint(FixAtoms(mask=mask))


def add_H_on_O(atoms, o_index: int, oh_distance_A: float) -> int:
    o_pos = atoms.positions[o_index]
    h_pos = o_pos + np.array([0.0, 0.0, oh_distance_A])
    atoms.append(Atom("H", position=h_pos))
    return len(atoms) - 1


def add_water_near_site(atoms, o_index: int) -> List[int]:
    """
    Deterministic, simple water placement:
    - Put water O a fixed lateral offset from the site and slightly above the surface.
    - Standard H2O geometry, one H roughly pointing toward the surface normal.
    This is a *test* hydration state (not a full solvation model).
    """
    # reference positions
    site = atoms.positions[o_index]
    z_top = atoms.positions[:, 2].max()

    # Place water oxygen ~2.8 Å laterally and ~1.8 Å above z_top (tunable)
    O_w = np.array([site[0] + 2.8, site[1], max(site[2] + 1.8, z_top + 1.0)])

    # Water geometry (approx): OH=0.9572 Å, angle 104.5°
    oh = 0.9572
    angle = np.deg2rad(104.5)

    # Local frame: z up, x along +x
    # Put one H "down-ish" relative to water O (toward surface), one sideways.
    H1 = O_w + np.array([0.0, 0.0, -oh])
    H2 = O_w + np.array([oh * np.sin(angle), 0.0, -oh * np.cos(angle)])

    atoms.append(Atom("O", position=O_w))
    iO = len(atoms) - 1
    atoms.append(Atom("H", position=H1))
    iH1 = len(atoms) - 1
    atoms.append(Atom("H", position=H2))
    iH2 = len(atoms) - 1
    return [iO, iH1, iH2]


def write_state(out_root: Path, state_name: str, base_name: str, atoms, meta: Dict[str, Any]) -> None:
    struct_dir = out_root / "states" / state_name / "structures"
    res_dir = out_root / "states" / state_name / "results"
    struct_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    traj_path = struct_dir / f"{base_name}.traj"
    meta_path = res_dir / f"metadata_{base_name}.json"

    meta = dict(meta)
    meta.update({
        "base_name": base_name,
        "structure_file": str(traj_path),
        "metadata_file": str(meta_path),
    })

    write(str(traj_path), atoms)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[ok] {traj_path}")
    print(f"[ok] {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_set", default="inputs/state_set.json", help="State set JSON")
    ap.add_argument("--outputs", default="outputs", help="Outputs directory")
    args = ap.parse_args()

    cfg = load_state_set(args.state_set)
    out_root = Path(args.outputs)
    out_root.mkdir(parents=True, exist_ok=True)

    slab_in = cfg["reference"]["slab_qe_input"]
    oh_dist = float(cfg["settings"].get("oh_distance_A", 1.0))
    z_freeze = float(cfg["settings"].get("z_freeze_A", 20.0))
    surf_thr = float(cfg["settings"].get("surface_threshold_A", 1.5))
    candidate_O = list(map(int, cfg["candidate_sites"]["o_indices"]))

    # Read QE slab. We keep PBC off (matches your xTB/ASE pipeline style).
    slab0 = read(slab_in, format="espresso-in")
    slab0.set_pbc((False, False, False))

    surface_O, z_top = surface_oxygen_indices(slab0, threshold_A=surf_thr)

    # Generate each state
    for st in cfg["states"]:
        st_name = st["name"]
        st_type = st["type"]

        if st_type == "clean_slab":
            atoms = slab0.copy()
            freeze_bottom(atoms, z_freeze_A=z_freeze)
            base = "slab_clean_ready"
            meta = {
                "state": st_name,
                "type": st_type,
                "surface_threshold_A": surf_thr,
                "surface_O_count": len(surface_O),
                "z_top_A": z_top,
                "z_freeze_A": z_freeze,
            }
            write_state(out_root, st_name, base, atoms, meta)

        elif st_type == "single_adsorbate":
            ads = st.get("adsorbate", "H")
            if ads != "H":
                raise ValueError("This template currently supports adsorbate=H only.")
            sites = list(map(int, st["sites"]))
            for o_idx in sites:
                atoms = slab0.copy()
                freeze_bottom(atoms, z_freeze_A=z_freeze)
                h_idx = add_H_on_O(atoms, o_idx, oh_distance_A=oh_dist)
                base = f"slab_H_o{o_idx}_ready"
                meta = {
                    "state": st_name,
                    "type": st_type,
                    "target_o_index": o_idx,
                    "occupied_o_indices": [o_idx],
                    "h_indices": [h_idx],
                    "oh_distance_A": oh_dist,
                    "z_freeze_A": z_freeze,
                    "surface_threshold_A": surf_thr,
                    "warnings": [] if o_idx in surface_O else [f"O[{o_idx}] not detected as surface O by threshold {surf_thr} Å"]
                }
                write_state(out_root, st_name, base, atoms, meta)

        elif st_type == "hydrated_single_adsorbate":
            ads = st.get("adsorbate", "H")
            if ads != "H":
                raise ValueError("This template currently supports adsorbate=H only.")
            o_idx = int(st["site"])
            atoms = slab0.copy()
            freeze_bottom(atoms, z_freeze_A=z_freeze)
            h_idx = add_H_on_O(atoms, o_idx, oh_distance_A=oh_dist)
            w_ids = add_water_near_site(atoms, o_idx)
            base = f"slab_H_hydrated_o{o_idx}_ready"
            meta = {
                "state": st_name,
                "type": st_type,
                "target_o_index": o_idx,
                "occupied_o_indices": [o_idx],
                "h_indices": [h_idx],
                "water_atom_indices": w_ids,
                "water_mode": st.get("water_mode", "near_site"),
                "oh_distance_A": oh_dist,
                "z_freeze_A": z_freeze,
                "surface_threshold_A": surf_thr,
                "warnings": [] if o_idx in surface_O else [f"O[{o_idx}] not detected as surface O by threshold {surf_thr} Å"]
            }
            write_state(out_root, st_name, base, atoms, meta)

        else:
            raise ValueError(f"Unknown state type: {st_type}")


if __name__ == "__main__":
    main()

