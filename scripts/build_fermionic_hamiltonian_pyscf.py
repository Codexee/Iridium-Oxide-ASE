#!/usr/bin/env python3
"""
Build an active-space fermionic Hamiltonian for a cluster using PySCF.

Why PySCF?
- We need 1e/2e integrals (h1, h2) in a chosen MO basis.
- xTB molden provides orbitals but not the ab initio integrals needed for a second-quantized Hamiltonian.

What this script does:
1) Reads cluster geometry (ASE-supported, e.g. .traj/.xyz)
2) Runs RHF or UHF (or RKS/UKS if you switch method) in PySCF
3) Selects an active space:
   - either by explicit MO indices (PySCF MO indices), OR
   - by localization/population on a user-defined "region" (recommended)
4) Builds active-space integrals (h1, h2) and exports:
   - outputs/hamiltonians/<site>/fermionic_active_space.npz
   - outputs/hamiltonians/<site>/fermionic_active_space.json (OpenFermion-style payload)

Notes:
- Heavy elements (Ir) may require ECP/basis availability. Start with a small basis and iterate.
- For robust "match xTB orbitals", use region-based scoring rather than reusing xTB MO indices.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from ase.io import read

# PySCF imports
from pyscf import gto, scf, ao2mo
from pyscf.tools import molden
from pyscf import dft

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
   
def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    # resolve relative to repo root (parent of scripts/)
    repo_root = Path(__file__).resolve().parents[1]
    cand = (repo_root / path).resolve()
    return cand if cand.exists() else path.resolve()


def ase_to_pyscf_geom(atoms) -> List[Tuple[str, Tuple[float, float, float]]]:
    """Return PySCF geometry list in Angstrom."""
    geom = []
    for sym, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        geom.append((sym, (float(x), float(y), float(z))))
    return geom


def parse_int_list(s: str) -> List[int]:
    """
    Parse comma-separated or space-separated ints. Example: "1,2,5" or "1 2 5"
    """
    if not s.strip():
        return []
    parts = [p.strip() for p in s.replace(",", " ").split()]
    return [int(p) for p in parts if p]


def build_region_atom_indices(atoms, h_index: int, cutoff_ang: float) -> List[int]:
    """H + neighbors within cutoff. Returns 0-based indices."""
    pos = atoms.get_positions()
    c = pos[h_index]
    d = np.linalg.norm(pos - c, axis=1)
    shell = [i for i, dist in enumerate(d) if (dist <= cutoff_ang and i != h_index)]
    return [h_index] + shell


def mo_region_score(mo_coeff: np.ndarray, mol: gto.Mole, region_aos: np.ndarray) -> np.ndarray:
    """
    Score each MO by its AO weight on the region AOs.
    region_aos is a boolean mask over AO indices.
    Returns array of scores length nmo.
    """
    # MO coefficients: (nao, nmo)
    c2 = mo_coeff ** 2
    total = np.sum(c2, axis=0) + 1e-18
    region = np.sum(c2[region_aos, :], axis=0)
    return region / total


def aos_for_atom_indices(mol: gto.Mole, atom_indices_0based: List[int]) -> np.ndarray:
    """
    Return boolean mask (nao,) for AOs belonging to the chosen atoms.
    """
    # mol.aoslice_by_atom() gives AO ranges per atom
    slices = mol.aoslice_by_atom()  # (natm, 4): (sh0, sh1, p0, p1)
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    for a in atom_indices_0based:
        p0, p1 = int(slices[a, 2]), int(slices[a, 3])
        mask[p0:p1] = True
    return mask


def choose_active_by_region(
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    region_aos: np.ndarray,
    n_occ: int,
    n_virt: int,
) -> List[int]:
    """
    Choose active orbitals by region score, separately for occupied and virtual.

    Returns PySCF MO indices (0-based).
    """
    scores = mo_region_score(mo_coeff, None, region_aos)  # mol not needed for scoring
    occ_idx = np.where(mo_occ > 1e-3)[0]
    virt_idx = np.where(mo_occ <= 1e-3)[0]

    occ_rank = occ_idx[np.argsort(scores[occ_idx])[::-1]]
    virt_rank = virt_idx[np.argsort(scores[virt_idx])[::-1]]

    chosen = list(occ_rank[:n_occ]) + list(virt_rank[:n_virt])
    return chosen


def active_space_integrals(
    mol: gto.Mole,
    mf,
    mo_coeff: np.ndarray,
    active_mos: List[int]
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Build (h1, h2, ecore) using ao2mo with correct frozen-core energy.
    h1 and h2 are in chemist notation (pq|rs).
    ecore includes nuclear repulsion + frozen-core electron energy.
    """
    from pyscf import mcscf

    nmo = mo_coeff.shape[1]
    mo_occ = mf.mo_occ
    n_active_electrons = int(sum(round(mo_occ[i]) for i in active_mos))
    nact = len(active_mos)

    # Identify frozen occupied orbitals (occupied but not in active space)
    frozen_occ = [i for i in range(nmo) if mo_occ[i] > 1e-3 and i not in active_mos]

    C_act = mo_coeff[:, active_mos]
    C_fro = mo_coeff[:, frozen_occ]

    # Core Hamiltonian
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

    # Frozen-core density matrix
    dm_fro = 2.0 * C_fro @ C_fro.T if len(frozen_occ) > 0 else np.zeros_like(hcore_ao)

    # Frozen-core Fock matrix contribution
    # veff_fro = J_fro - 0.5*K_fro
    from pyscf import scf as pyscf_scf
    veff_fro = mf.get_veff(mol, dm_fro) if len(frozen_occ) > 0 else np.zeros_like(hcore_ao)

    # Effective one-electron integrals for active space
    heff_ao = hcore_ao + veff_fro
    h1 = C_act.T @ heff_ao @ C_act

    # Two-electron integrals for active space (chemist notation)
    h2 = ao2mo.kernel(mol, C_act, aosym='s1', compact=False).reshape(nact, nact, nact, nact)

    # Core energy: nuclear repulsion + frozen electron energy
    ecore = mol.energy_nuc()
    if len(frozen_occ) > 0:
        # E_core = E_nuc + sum_i (hcore[i,i] + 0.5*veff_fro[i,i]) * 2 (factor 2 for closed shell)
        ecore += float(np.einsum('ij,ji->', hcore_ao + 0.5 * veff_fro, dm_fro))

    return h1, h2, ecore, n_active_electrons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster", help="Cluster geometry file (.traj/.xyz/... ASE-readable)")
    ap.add_argument("--site", default="o69", help="Site label for outputs")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=0, help="2S, i.e. Nalpha - Nbeta. spin=0 for closed-shell.")
    ap.add_argument("--basis", default="def2-svp", help="Basis set name (PySCF)")
    ap.add_argument("--ecp", default="", help="ECP name for heavy elements (e.g. def2-svp). " "Empty string = all-electron (default, non-PP branch).")
    ap.add_argument("--method", choices=["RHF", "UHF"], default="RHF")
    ap.add_argument("--h-index", type=int, default=-1, help="0-based H atom index; -1 to infer single/last H")
    ap.add_argument("--region-cutoff", type=float, default=3.0, help="Å cutoff for region atoms around H")
    ap.add_argument("--active-mos", default="", help="Explicit PySCF MO indices (0-based). Example: '10,11,12,13'")
    ap.add_argument("--n-occ", type=int, default=4, help="If using region-based selection: number of occupied actives")
    ap.add_argument("--n-virt", type=int, default=3, help="If using region-based selection: number of virtual actives")
    ap.add_argument("--write-molden", action="store_true", help="Write PySCF molden for visualization")
    ap.add_argument("--outdir", default="outputs/hamiltonians")
    args = ap.parse_args()

    cluster_path = resolve_path(args.cluster)
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster not found: {cluster_path}")
    print(f"Reading cluster: {cluster_path}")

    atoms = read(str(cluster_path))

    # infer H index
    syms = atoms.get_chemical_symbols()
    h_candidates = [i for i, s in enumerate(syms) if s.upper() == "H"]
    if args.h_index >= 0:
        h_idx = args.h_index
    else:
        if not h_candidates:
            raise ValueError("No H found; provide --h-index explicitly.")
        h_idx = h_candidates[0] if len(h_candidates) == 1 else h_candidates[-1]

    region_atoms = build_region_atom_indices(atoms, h_idx, args.region_cutoff)

    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = ase_to_pyscf_geom(atoms)
    mol.unit = "Angstrom"
    mol.charge = args.charge
    mol.spin = args.spin
    mol.basis = args.basis
    if args.ecp:
        mol.ecp = args.ecp
    mol.build()

    # Mean-field
    from pyscf import dft
    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.verbose = 4
    mf.max_cycle = 200
    mf.conv_tol = 1e-6

    e = mf.kernel()

    if not mf.converged:
        mf = mf.newton()
        e = mf.kernel()

    if not mf.converged:
        import warnings
        warnings.warn(
            f"SCF not fully converged after newton(). "
            f"Final E={e:.6f} Ha. Proceeding with last density matrix.",
            UserWarning
        )
    print(f"SCF energy: {e}")

    # Extract MO coefficients and occupations
    if args.method == "RHF":
        mo_coeff = mf.mo_coeff  # (nao, nmo)
        mo_occ = mf.mo_occ      # (nmo,)
    else:
        # For UHF, treat alpha and beta separately; simplest first step is alpha channel selection
        # (You can extend later to spin-orbital Hamiltonians explicitly.)
        mo_coeff = mf.mo_coeff[0]
        mo_occ = mf.mo_occ[0]
        print("NOTE: UHF detected; selecting active orbitals from ALPHA MOs for now.")

    # region AO mask
    region_aos = aos_for_atom_indices(mol, region_atoms)

    # Choose active orbitals
    explicit = parse_int_list(args.active_mos)
    if explicit:
        active_mos = explicit
        print(f"Using explicit PySCF MO indices (0-based): {active_mos}")
    else:
        # Region-based selection
        scores = (mo_coeff ** 2)
        total = np.sum(scores, axis=0) + 1e-18
        region = np.sum(scores[region_aos, :], axis=0)
        frac = region / total

        occ_idx = np.where(mo_occ > 1e-3)[0]
        virt_idx = np.where(mo_occ <= 1e-3)[0]
        occ_rank = occ_idx[np.argsort(frac[occ_idx])[::-1]]
        virt_rank = virt_idx[np.argsort(frac[virt_idx])[::-1]]

        active_mos = list(occ_rank[:args.n_occ]) + list(virt_rank[:args.n_virt])
        print(f"Selected active MOs by region fraction: {active_mos}")

    # Build active-space integrals
    h1, h2, ecore, n_active_electrons = active_space_integrals(mol, mf, mo_coeff, active_mos)

    outdir = Path(args.outdir) / args.site
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / "fermionic_active_space.npz"
    np.savez_compressed(
        npz_path,
        h1=h1,
        h2=h2,
        ecore=ecore,
        active_mos=np.array(active_mos, dtype=int),
        charge=args.charge,
        spin=args.spin,
        basis=args.basis,
        method=args.method,
        h_index=h_idx,
        region_atoms=np.array(region_atoms, dtype=int),
        cluster=str(cluster_path),
        n_active_electrons=n_active_electrons,
    )
    print(f"Wrote: {npz_path}")

    # Lightweight JSON payload (so Superstaq side can read it without NumPy if needed)
    payload = {
        "site": args.site,
        "cluster": str(cluster_path),
        "charge": args.charge,
        "spin": args.spin,
        "basis": args.basis,
        "method": args.method,
        "ecore": ecore,
        "active_mos_0based": active_mos,
        "n_orb": int(h1.shape[0]),
        "h1": h1.tolist(),
        "h2": h2.tolist(),
        "n_active_electrons": n_active_electrons,
    }
    json_path = outdir / "fermionic_active_space.json"
    json_path.write_text(json.dumps(json_sanitize(payload), indent=2))
    print(f"Wrote: {json_path}")

    if args.write_molden:
        molden_path = outdir / "pyscf.molden"
        with open(molden_path, "w") as f:
            molden.header(mol, f)
            molden.orbital_coeff(mol, f, mo_coeff)
        print(f"Wrote: {molden_path}")


if __name__ == "__main__":
    main()
