#!/usr/bin/env python3
"""
Orbital analysis for an extracted cluster using xTB (GFN2-xTB) + Molden output.

What it does:
- Reads a cluster geometry (.traj/.xyz/etc. supported by ASE)
- Runs single-point xTB with --molden (requires xtb binary on PATH)
- Parses molden.input to extract MO energies, occupations, and AO coefficients
- Computes per-orbital contributions on:
    (a) H* atom
    (b) neighbor shell around H* within a cutoff (default 3.0 Å)
- Suggests an "active space" as a set of orbitals near the frontier with high region contribution
- Writes:
    orbitals.csv
    active_orbitals.json
    orbital_report.md

Notes:
- This is intentionally "first-pass": good enough to identify frontier/region-dominant orbitals.
- Later we can add: spin resolution, higher-level QC interfaces, localization, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from ase.io import read
except Exception as e:
    print("ERROR: ASE is required. Try: pip install ase", file=sys.stderr)
    raise


# ----------------------------
# Helpers: xTB run
# ----------------------------

def run_xtb_single_point(
    xyz_path: Path,
    workdir: Path,
    gfn: int = 2,
    chrg: int = 0,
    uhf: int = 0,
    extra_args: Optional[List[str]] = None,
) -> Path:
    """
    Run xTB single-point with Molden output.
    Returns path to xtb.out.
    """
    if shutil.which("xtb") is None:
        raise RuntimeError(
            "xtb binary not found on PATH. Install xTB and ensure `xtb` is available."
        )

    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "xtb",
        str(xyz_path.resolve()),
        "--gfn", str(gfn),
        "--chrg", str(chrg),
        "--uhf", str(uhf),
        "--molden",
    ]
    if extra_args:
        cmd.extend(extra_args)

    out_path = workdir / "xtb.out"
    err_path = workdir / "xtb.err"

    with out_path.open("w") as fout, err_path.open("w") as ferr:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            stdout=fout,
            stderr=ferr,
            text=True,
        )

    if proc.returncode != 0:
        msg = (
            f"xTB failed with return code {proc.returncode}.\n"
            f"See: {out_path} and {err_path}\n"
        )
        raise RuntimeError(msg)

    molden_path = workdir / "molden.input"
    if not molden_path.exists():
        raise RuntimeError(
            "xTB completed but molden.input not found. "
            "Check xTB output for why Molden file was not written."
        )

    return out_path


# ----------------------------
# Molden parser
# ----------------------------

@dataclass
class MoldenMO:
    index_1based: int
    energy_hartree: float
    occ: float
    spin: str  # "Alpha", "Beta", or "Unknown"
    coeffs: np.ndarray  # shape (nAO,)


@dataclass
class MoldenData:
    atom_symbols: List[str]
    atom_coords_ang: np.ndarray  # (nAtoms, 3)
    ao_atom_index_1based: List[int]  # length nAO, maps AO -> atom index (1-based)
    mos: List[MoldenMO]


def parse_molden(molden_path: Path) -> MoldenData:
    """
    Parse a Molden file written by xTB.
    We use:
      - [Atoms] block (symbols + coordinates)
      - [GTO] block (AO-to-atom mapping; count AOs per atom from shells)
      - [MO] block (energies, occupations, coefficients)
    """
    text = molden_path.read_text(errors="replace").splitlines()

    # Locate sections
    def find_section(name: str) -> int:
        target = f"[{name.lower()}]"
        for i, line in enumerate(text):
            s = line.strip().lower()
            # Accept "[atoms]" and also "[atoms] (angs)" etc.
            if s.startswith(target):
                return i
        return -1


    i_atoms = find_section("Atoms")
    i_gto = find_section("GTO")
    i_mo = find_section("MO")
    if i_atoms < 0 or i_gto < 0 or i_mo < 0:
        raise ValueError("Molden file missing required sections: [Atoms], [GTO], [MO].")
    
    print("Molden headers found:")
    for line in text:
        if line.strip().startswith("["):
            print(line.strip())

    # -------- Parse [Atoms]
    atom_symbols: List[str] = []
    atom_coords: List[List[float]] = []
    # [Atoms] may have a header line like: "[Atoms] (Angs)" or separate line; accept both
    j = i_atoms + 1
    while j < len(text):
        line = text[j].strip()
        if not line or line.startswith("["):
            break
        # Expected: Symbol  index  atomic_number  x  y  z
        parts = line.split()
        if len(parts) >= 6:
            atom_symbols.append(parts[0])
            atom_coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
        j += 1
    atom_coords_ang = np.array(atom_coords, dtype=float)

    n_atoms = len(atom_symbols)
    if n_atoms == 0:
        raise ValueError("Failed to parse atoms from [Atoms] section.")

    # -------- Parse [GTO] to count number of AOs per atom (and map AO -> atom)
    # Molden [GTO] format: blocks per atom start with "<atom_index> 0"
    # then shell lines: "<shell> <nprim> 1.00" followed by exponent/coeff lines
    # AO count per shell: s=1, p=3, d=5, f=7 ...
    shell_ao_count = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9}

    ao_atom_index_1based: List[int] = []
    j = i_gto + 1
    current_atom = None
    while j < len(text):
        line = text[j].strip()
        if not line:
            j += 1
            continue
        if line.startswith("["):
            break

        # Atom block header
        m = re.match(r"^(\d+)\s+0\s*$", line)
        if m:
            current_atom = int(m.group(1))
            j += 1
            continue

        # Shell line
        m2 = re.match(r"^([spdfgSPDFG])\s+(\d+)\s+([\d\.Ee\+\-]+)\s*$", line)
        if m2 and current_atom is not None:
            shell = m2.group(1).lower()
            nprim = int(m2.group(2))
            nao = shell_ao_count.get(shell)
            if nao is None:
                raise ValueError(f"Unsupported shell type '{shell}' in Molden [GTO].")
            # Append AO->atom mapping for this shell
            ao_atom_index_1based.extend([current_atom] * nao)
            # Skip primitive lines
            j += 1 + nprim
            continue

        # Otherwise just advance
        j += 1

    n_ao = len(ao_atom_index_1based)
    if n_ao == 0:
        raise ValueError("Failed to determine AO mapping from [GTO] section.")

    # -------- Parse [MO]
    mos: List[MoldenMO] = []
    j = i_mo + 1

    def read_float_after(prefix: str, line: str) -> Optional[float]:
        if line.lower().startswith(prefix.lower()):
            parts = line.split("=")
            if len(parts) == 2:
                try:
                    return float(parts[1].strip())
                except ValueError:
                    return None
        return None

    # Molden MO blocks repeat:
    #   Sym=
    #   Ene=
    #   Spin=
    #   Occup=
    #   <ao_index> <coeff>
    # until next Sym= or EOF/section
    mo_idx = 0
    while j < len(text):
        line = text[j].strip()
        if not line:
            j += 1
            continue
        if line.startswith("["):
            break

        if line.lower().startswith("sym="):
            mo_idx += 1
            energy = None
            occ = None
            spin = "Unknown"
            coeffs = np.zeros(n_ao, dtype=float)

            # consume header lines
            j += 1
            while j < len(text):
                line2 = text[j].strip()
                if not line2:
                    j += 1
                    continue
                if line2.lower().startswith("sym=") or line2.startswith("["):
                    # next MO begins (or new section)
                    break
                if line2.lower().startswith("ene="):
                    # energy in Hartree (Molden convention)
                    parts = line2.split("=")
                    energy = float(parts[1].strip())
                elif line2.lower().startswith("spin="):
                    spin = line2.split("=")[1].strip()
                elif line2.lower().startswith("occup="):
                    occ = float(line2.split("=")[1].strip())
                else:
                    # Coeff line: "<ao_index> <coeff>"
                    p = line2.split()
                    if len(p) >= 2 and p[0].isdigit():
                        ao_i = int(p[0]) - 1
                        if 0 <= ao_i < n_ao:
                            coeffs[ao_i] = float(p[1])
                j += 1

            if energy is None or occ is None:
                raise ValueError("Failed to parse MO header (missing Ene= or Occup=).")

            mos.append(
                MoldenMO(
                    index_1based=mo_idx,
                    energy_hartree=float(energy),
                    occ=float(occ),
                    spin=spin,
                    coeffs=coeffs,
                )
            )
            continue

        j += 1

    if len(mos) == 0:
        raise ValueError("No MOs parsed from [MO] section.")

    return MoldenData(
        atom_symbols=atom_symbols,
        atom_coords_ang=atom_coords_ang,
        ao_atom_index_1based=ao_atom_index_1based,
        mos=mos,
    )


# ----------------------------
# Region definitions and scoring
# ----------------------------

def guess_hstar_index(atoms) -> int:
    """Return 0-based index of H* atom. If exactly one H, use it; else pick last H."""
    symbols = atoms.get_chemical_symbols()
    h_indices = [i for i, s in enumerate(symbols) if s.upper() == "H"]
    if not h_indices:
        raise ValueError("No hydrogen atoms found in cluster; cannot infer H* index.")
    if len(h_indices) == 1:
        return h_indices[0]
    return h_indices[-1]


def neighbor_shell_indices(atoms, center_idx: int, cutoff_ang: float) -> List[int]:
    """Return indices within cutoff of center_idx, excluding center itself."""
    pos = atoms.get_positions()
    c = pos[center_idx]
    d = np.linalg.norm(pos - c, axis=1)
    idxs = [i for i, dist in enumerate(d) if (dist <= cutoff_ang and i != center_idx)]
    return idxs


def orbital_contribution_on_atoms(
    mo: MoldenMO, ao_atom_index_1based: List[int], atom_indices_0based: List[int]
) -> float:
    """
    Contribution = sum over AOs on selected atoms of |c|^2, normalized by total |c|^2.
    """
    if len(atom_indices_0based) == 0:
        return 0.0
    target_atoms_1based = set([i + 1 for i in atom_indices_0based])
    c2 = mo.coeffs ** 2
    total = float(np.sum(c2))
    if total <= 0.0:
        return 0.0
    mask = np.array([a in target_atoms_1based for a in ao_atom_index_1based], dtype=bool)
    return float(np.sum(c2[mask]) / total)


def hartree_to_ev(x: float) -> float:
    return x * 27.211386245988  # CODATA 2018-ish, fine for ranking


def find_homo_index(mos: List[MoldenMO]) -> int:
    """Return index in list (0-based) of HOMO-like orbital (occ > ~0)."""
    occs = np.array([mo.occ for mo in mos])
    # xTB may use 2.0 for closed-shell alpha or separate spins; use a small threshold
    occ_thresh = 1e-3
    occ_idx = np.where(occs > occ_thresh)[0]
    if len(occ_idx) == 0:
        return 0
    return int(occ_idx[-1])


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cluster", type=str, help="Path to cluster geometry (.traj/.xyz/...)")
    ap.add_argument("--site", type=str, default="o69", help="Site label for output naming")
    ap.add_argument("--charge", type=int, default=0, help="Total cluster charge")
    ap.add_argument("--uhf", type=int, default=0, help="Unpaired electrons (0 for closed-shell)")
    ap.add_argument("--gfn", type=int, default=2, choices=[0, 1, 2], help="xTB GFN level")
    ap.add_argument("--hstar-index", type=int, default=-1,
                    help="0-based index of H* in ASE atoms. If omitted, inferred.")
    ap.add_argument("--shell-cutoff", type=float, default=3.0,
                    help="Neighbor shell cutoff (Å) around H* for region scoring")
    ap.add_argument("--frontier-window-ev", type=float, default=2.0,
                    help="Energy window (eV) around HOMO energy for suggested active orbitals")
    ap.add_argument("--topn", type=int, default=12,
                    help="Max number of orbitals to suggest for active set")
    ap.add_argument("--outdir", type=str, default="outputs/orbital_analysis",
                    help="Output directory")
    ap.add_argument(
    "--pseudo-metadata",
    type=str,
    default=None,
    help="Optional QE pseudopotential provenance metadata JSON",
    )
    args = ap.parse_args()

    cluster_path = Path(args.cluster)

    # If a relative path was provided, resolve it relative to repo root (parent of scripts/)
    if not cluster_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        candidate = (repo_root / cluster_path).resolve()
    if candidate.exists():
        cluster_path = candidate

    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_path}")

    print(f"Reading cluster geometry from: {cluster_path}")

        
    outdir = Path(args.outdir) / args.site
    outdir.mkdir(parents=True, exist_ok=True)

    # Read geometry
    atoms = read(str(cluster_path))
    symbols = atoms.get_chemical_symbols()

    # Determine H*
    if args.hstar_index >= 0:
        h_idx = args.hstar_index
    else:
        h_idx = guess_hstar_index(atoms)

    # Build neighbor shell
    shell_idxs = neighbor_shell_indices(atoms, h_idx, args.shell_cutoff)
    region_idxs = [h_idx] + shell_idxs

    # Write xyz for xTB
    xyz_path = outdir / f"cluster_{args.site}.xyz"
    atoms.write(str(xyz_path))

    # Run xTB in a clean workdir (keeps outputs organized)
    xtb_workdir = outdir / "xtb_sp"
    xtb_out = run_xtb_single_point(
        xyz_path=xyz_path,
        workdir=xtb_workdir,
        gfn=args.gfn,
        chrg=args.charge,
        uhf=args.uhf,
        extra_args=["--verbose"],
    )

    # Parse Molden
    molden_path = xtb_workdir / "molden.input"
    md = parse_molden(molden_path)

    # Compute contributions per MO
    rows = []
    for mo in md.mos:
        e_ev = hartree_to_ev(mo.energy_hartree)
        contrib_h = orbital_contribution_on_atoms(mo, md.ao_atom_index_1based, [h_idx])
        contrib_shell = orbital_contribution_on_atoms(mo, md.ao_atom_index_1based, shell_idxs)
        contrib_region = orbital_contribution_on_atoms(mo, md.ao_atom_index_1based, region_idxs)
        rows.append({
            "mo": mo.index_1based,
            "spin": mo.spin,
            "energy_hartree": mo.energy_hartree,
            "energy_ev": e_ev,
            "occ": mo.occ,
            "contrib_H": contrib_h,
            "contrib_shell": contrib_shell,
            "contrib_region": contrib_region,
        })

    # Sort by energy
    rows_sorted = sorted(rows, key=lambda r: r["energy_ev"])
    homo_i = find_homo_index(md.mos)
    homo_mo = md.mos[homo_i]
    homo_energy_ev = hartree_to_ev(homo_mo.energy_hartree)

    # Suggested active orbitals: within energy window of HOMO, ranked by region contribution
    window = args.frontier_window_ev
    frontier = [r for r in rows_sorted if abs(r["energy_ev"] - homo_energy_ev) <= window]
    frontier_ranked = sorted(frontier, key=lambda r: r["contrib_region"], reverse=True)
    suggested = frontier_ranked[: max(1, args.topn)]

    # Write CSV
    csv_path = outdir / "orbitals.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mo", "spin", "energy_hartree", "energy_ev", "occ",
                "contrib_H", "contrib_shell", "contrib_region",
            ],
        )
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    pseudo_metadata = None

    if args.pseudo_metadata:
        pseudo_path = Path(args.pseudo_metadata)
    
        if not pseudo_path.exists():
            raise FileNotFoundError(
                f"Pseudopotential metadata file not found: {pseudo_path}"
            )
    
        pseudo_metadata = json.loads(pseudo_path.read_text())
    
    # Write JSON for active orbitals
    active = {
        "site": args.site,
        "input_cluster": str(cluster_path),
        "xtb": {
            "gfn": args.gfn,
            "charge": args.charge,
            "uhf": args.uhf,
        },
        "pseudopotential_provenance": pseudo_metadata,
        "hstar_index_0based": h_idx,
        "hstar_symbol": symbols[h_idx],
        "shell_cutoff_ang": args.shell_cutoff,
        "shell_indices_0based": shell_idxs,
        "frontier_window_ev": window,
        "homo_energy_ev": homo_energy_ev,
        "suggested_active_orbitals_1based": [r["mo"] for r in suggested],
        "suggested_orbitals_detail": suggested,
    }
    json_path = outdir / "active_orbitals.json"
    json_path.write_text(json.dumps(active, indent=2))

    # Write quick report
    report_path = outdir / "orbital_report.md"
    lines = []
    lines.append(f"# Orbital analysis (xTB) for site {args.site}\n")
    lines.append(f"- Input cluster: `{cluster_path}`")
    lines.append(f"- xTB level: GFN{args.gfn}-xTB, charge={args.charge}, uhf={args.uhf}")
    lines.append(f"- H* index (0-based): {h_idx}  (element={symbols[h_idx]})")
    lines.append(f"- Neighbor shell cutoff: {args.shell_cutoff:.2f} Å")
    lines.append(f"- Shell size (atoms): {len(shell_idxs)}")
    lines.append(f"- HOMO energy (approx): {homo_energy_ev:.3f} eV\n")
    if pseudo_metadata:
        lines.append("## Parent QE pseudopotential provenance\n")
        lines.append(
            "This xTB orbital analysis is linked to a QE pseudopotential-aware "
            "parent electronic-structure configuration for provenance tracking.\n"
        )
        lines.append(
            f"- QE input: `{pseudo_metadata['qe_input_file']}`"
        )
        lines.append(
            f"- XC functional: `{pseudo_metadata['input_dft']}`"
        )
        lines.append(
            f"- ecutwfc: `{pseudo_metadata['ecutwfc']}` Ry"
        )
        lines.append(
            f"- ecutrho: `{pseudo_metadata['ecutrho']}` Ry\n"
        )
        lines.append("## Suggested active orbitals (ranked by region contribution)\n")
        lines.append("| MO | Energy (eV) | Occ | contrib(H) | contrib(shell) | contrib(region) |")
        lines.append("|---:|------------:|----:|----------:|---------------:|---------------:|")
    for r in suggested:
        lines.append(
            f"| {r['mo']} | {r['energy_ev']:.3f} | {r['occ']:.3f} | "
            f"{r['contrib_H']:.3f} | {r['contrib_shell']:.3f} | {r['contrib_region']:.3f} |"
        )
    lines.append("\n## Files\n")
    lines.append(f"- `orbitals.csv` — all MOs sorted by energy")
    lines.append(f"- `active_orbitals.json` — suggested active set + metadata")
    lines.append(f"- `xtb_sp/molden.input` — molden file used for analysis")
    report_path.write_text("\n".join(lines))

    print("Done.")
    print(f"Outputs written to: {outdir}")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print(f"- {report_path}")
    print(f"HOMO (approx): {homo_energy_ev:.3f} eV")
    print("Suggested active orbitals:", [r["mo"] for r in suggested])


if __name__ == "__main__":
    main()
