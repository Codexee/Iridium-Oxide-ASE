#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ase import Atom
from ase.io import read, write


# Cap parameters. Centralised here so they're easy to find and change.
OH_BOND_LENGTH = 0.96   # Angstrom, standard O-H bond length
IR_O_CUTOFF = 2.6       # Angstrom, max distance to consider Ir-O bonded
                        # (rutile IrO2 Ir-O bonds are ~2.0 A)
CAP_TAG = 1             # ASE tag for cap H atoms
ADSORBATE_TAG = 0       # ASE tag for adsorbate H (default)


def find_ir_neighbours_in_slab(slab_atoms, o_idx, cutoff):
    """For oxygen at o_idx in the slab, return indices of all Ir atoms within
    cutoff. Uses the slab geometry, so this gives the 'real' coordination
    before any cutting."""
    o_pos = slab_atoms.positions[o_idx]
    distances = np.linalg.norm(slab_atoms.positions - o_pos, axis=1)
    ir_indices = [i for i, a in enumerate(slab_atoms)
                  if a.symbol == "Ir" and distances[i] <= cutoff]
    return ir_indices


def cap_dangling_oxygens(slab_atoms, cluster_atoms, slab_indices_kept,
                        ir_o_cutoff=IR_O_CUTOFF, oh_length=OH_BOND_LENGTH):
    """For each O in the cluster, check if it lost any Ir neighbours during
    the radial cut. If so, add a cap H along the vector to each missing Ir.

    Parameters
    ----------
    slab_atoms : ase.Atoms
        Original slab geometry, used to look up real Ir neighbours.
    cluster_atoms : ase.Atoms
        Cluster after radial cut (will be modified in place).
    slab_indices_kept : list[int]
        Slab indices that ended up in the cluster, in cluster order.

    Returns
    -------
    n_caps : int
        Number of cap H atoms added.
    """
    n_caps = 0
    kept_set = set(slab_indices_kept)

    # Iterate over oxygens in the cluster
    for cluster_i, slab_i in enumerate(slab_indices_kept):
        if slab_atoms[slab_i].symbol != "O":
            continue
        # Skip oxygens that already have an H bound (these are adsorbate-bonded Os)
        # This prevents capping the adsorbate site itself.
        o_pos = slab_atoms.positions[slab_i]
        distances_to_H = [
            np.linalg.norm(slab_atoms.positions[k] - o_pos)
            for k, a in enumerate(slab_atoms)
            if a.symbol == "H"
        ]
        if any(d < 1.2 for d in distances_to_H):
            continue
        # Find this O's Ir neighbours in the original slab
        ir_neighbours_in_slab = find_ir_neighbours_in_slab(
            slab_atoms, slab_i, ir_o_cutoff
        )

        # Identify the ones that did NOT survive the cut
        missing_ir = [j for j in ir_neighbours_in_slab if j not in kept_set]

        # Place a cap H along the vector to each missing Ir
        o_pos = slab_atoms.positions[slab_i]
        for ir_idx in missing_ir:
            ir_pos = slab_atoms.positions[ir_idx]
            vec = ir_pos - o_pos
            vec_len = np.linalg.norm(vec)
            if vec_len < 1e-6:
                continue   # safety, should never happen
            unit = vec / vec_len
            cap_pos = o_pos + unit * oh_length
            cluster_atoms.append(Atom("H", position=cap_pos, tag=CAP_TAG))
            n_caps += 1

    return n_caps


def main():
    ap = argparse.ArgumentParser(
        description="Extract a local cluster around an adsorption site from "
                    "an optimized slab, with optional H-capping of "
                    "undercoordinated boundary oxygens."
    )
    ap.add_argument("--traj", required=True,
                    help="Input optimized .traj (e.g. slab_H_o69_ready_final.traj)")
    ap.add_argument("--center_atom_index", type=int, default=None,
                    help="Atom index in the slab to center on. If omitted, uses H atom.")
    ap.add_argument("--cutoff", type=float, default=6.0,
                    help="Radius cutoff in Angstrom")
    ap.add_argument("--no_cap", action="store_true",
                    help="Disable H-capping of dangling O bonds (capping is on by default).")
    ap.add_argument("--outdir", default="outputs/clusters", help="Output directory")
    ap.add_argument("--tag", default="o69", help="Tag name for output files")
    args = ap.parse_args()

    slab = read(args.traj)

    # Pick the centre atom. The adsorbate H is tagged with ADSORBATE_TAG=0
    # by default in the input traj, so finding the H here also sets up the
    # tag convention that downstream scripts rely on.
    if args.center_atom_index is None:
        H_indices = [i for i, a in enumerate(slab) if a.symbol == "H"]
        if not H_indices:
            raise SystemExit("No H atom found. Provide --center_atom_index.")
        center_i = H_indices[0]
    else:
        center_i = args.center_atom_index

    # Ensure the adsorbate H has the right tag. We assume the centre H is the
    # adsorbate. If center_atom_index points at an O, no tag-setting needed.
    if slab[center_i].symbol == "H":
        slab[center_i].tag = ADSORBATE_TAG

    # Radial cut
    center = slab.positions[center_i]
    distances = np.linalg.norm(slab.positions - center, axis=1)
    slab_indices_kept = np.where(distances <= args.cutoff)[0].tolist()
    cluster = slab[slab_indices_kept]

    print(f"[ok] kept {len(cluster)} atoms within {args.cutoff:.2f} Å of slab index {center_i}")

    # Capping
    if not args.no_cap:
        n_caps = cap_dangling_oxygens(slab, cluster, slab_indices_kept)
        print(f"[ok] added {n_caps} cap H atoms on undercoordinated boundary oxygens")
    else:
        print("[ok] capping disabled (--no_cap), cluster left with dangling bonds")

    # Write
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_traj = outdir / f"{args.tag}_cluster.traj"
    out_xyz = outdir / f"{args.tag}_cluster.xyz"
    write(out_traj, cluster)
    write(out_xyz, cluster)
    print(f"[ok] wrote {out_traj}")
    print(f"[ok] wrote {out_xyz}")


if __name__ == "__main__":
    main()
