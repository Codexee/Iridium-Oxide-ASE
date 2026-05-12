#!/usr/bin/env python3
"""
Write a Quantum ESPRESSO input file with explicit pseudopotential provenance.

Purpose
-------
This script adds a pseudopotential-aware parent-DFT step to the IrO2 workflow.

It does NOT modify the reduced Hamiltonian directly.
Instead, it prepares a QE calculation whose outputs can later feed orbital /
projection analysis and active-space Hamiltonian construction.

Typical workflow insertion
--------------------------
Current:
    slab_clean_2x2.in
        -> iro2_slab_setup.py
        -> optimization_iro2_test.py using xTB
        -> orbital_analysis_xtb_cluster.py
        -> manual active orbital selection
        -> build_fermionic_hamiltonian_pyscf.py

Development:
    slab_clean_2x2.in
        -> iro2_slab_setup.py
        -> optional xTB optimisation
        -> this script: QE input with explicit pseudopotentials
        -> QE single-point or relax calculation
        -> orbital/projection analysis
        -> manual active orbital selection
        -> build_fermionic_hamiltonian_pyscf.py

Expected manifest structure
---------------------------
inputs/pseudopotentials/pseudopotential_manifest.yaml

Example:

system: IrO2_H
backend: quantum_espresso
exchange_correlation: PBE
pseudo_dir: inputs/pseudopotentials/upf
calculation_defaults:
  ecutwfc: 60
  ecutrho: 480
  occupations: smearing
  smearing: mv
  degauss: 0.02
  nspin: 1
elements:
  Ir:
    file: Ir.pbe-spn-kjpaw_psl.1.0.0.UPF
    family: PSLibrary
    relativistic: scalar-relativistic
    valence_configuration: TBD
    source: TBD
  O:
    file: O.pbe-n-kjpaw_psl.1.0.0.UPF
    family: PSLibrary
    relativistic: scalar-relativistic
    valence_configuration: TBD
    source: TBD
  H:
    file: H.pbe-kjpaw_psl.1.0.0.UPF
    family: PSLibrary
    relativistic: scalar-relativistic
    valence_configuration: 1s1
    source: TBD
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.io import read


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load YAML or JSON manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    text = path.read_text()

    if path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "This manifest appears to be YAML, but PyYAML is not installed. "
            "Install it with `pip install pyyaml`, or save the manifest as JSON."
        ) from exc

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest did not parse to a dictionary: {path}")
    return data


def repo_relative(path: Path) -> str:
    """Return a readable path string without requiring the repo root."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def get_nested(dct: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Return the first present key from a dict."""
    for key in keys:
        if key in dct and dct[key] is not None:
            return dct[key]
    return default


def element_pseudo_file(element_data: Dict[str, Any], symbol: str) -> str:
    """Extract pseudopotential filename using several accepted key names."""
    fname = get_nested(
        element_data,
        keys=("file", "filename", "pseudo_file", "pseudopotential_file", "upf"),
        default=None,
    )
    if not fname:
        raise KeyError(
            f"No pseudopotential file specified for element {symbol}. "
            "Use one of: file, filename, pseudo_file, pseudopotential_file, upf."
        )
    return str(fname)


def unique_symbols_in_order(symbols: List[str]) -> List[str]:
    """Unique chemical symbols preserving first occurrence order."""
    seen = set()
    out = []
    for sym in symbols:
        if sym not in seen:
            out.append(sym)
            seen.add(sym)
    return out


def qe_bool(value: bool) -> str:
    return ".true." if value else ".false."


def qe_value(value: Any) -> str:
    """Format Python values for QE input."""
    if isinstance(value, bool):
        return qe_bool(value)
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{value}'"


def constraint_flags(atoms, atom_index: int) -> Tuple[int, int, int]:
    """
    Return QE position flags for atom_index.

    QE flags after ATOMIC_POSITIONS:
      1 1 1 means movable
      0 0 0 means fixed

    The existing slab setup applies ASE FixAtoms to bottom layers.
    This function preserves those constraints when possible.
    """
    for constraint in atoms.constraints:
        # ASE FixAtoms exposes get_indices().
        if hasattr(constraint, "get_indices"):
            fixed = set(int(i) for i in constraint.get_indices())
            if atom_index in fixed:
                return (0, 0, 0)

    return (1, 1, 1)


def write_qe_input(
    atoms,
    manifest: Dict[str, Any],
    output_path: Path,
    calculation: str,
    prefix: str,
    pseudo_dir: str,
    kpoints: str,
    ecutwfc: float,
    ecutrho: float,
    input_dft: str,
    occupations: str,
    smearing: str,
    degauss: float,
    nspin: int,
) -> None:
    """Write a QE pw.x input file."""
    symbols = atoms.get_chemical_symbols()
    unique_symbols = unique_symbols_in_order(symbols)
    elements = manifest.get("elements", {})

    missing = [sym for sym in unique_symbols if sym not in elements]
    if missing:
        raise KeyError(
            "Manifest is missing pseudopotential entries for: "
            + ", ".join(missing)
            + ". Add them under `elements:`."
        )

    cell = np.asarray(atoms.get_cell())
    if abs(np.linalg.det(cell)) < 1e-8:
        raise ValueError(
            "Input atoms object has a near-zero cell volume. "
            "QE requires CELL_PARAMETERS for this workflow."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.extend(
        [
            "&CONTROL",
            f"  calculation = {qe_value(calculation)},",
            f"  prefix = {qe_value(prefix)},",
            f"  pseudo_dir = {qe_value(pseudo_dir)},",
            "  outdir = './qe_tmp',",
            "  verbosity = 'high',",
            "/",
            "",
            "&SYSTEM",
            "  ibrav = 0,",
            f"  nat = {len(atoms)},",
            f"  ntyp = {len(unique_symbols)},",
            f"  ecutwfc = {ecutwfc},",
            f"  ecutrho = {ecutrho},",
            f"  input_dft = {qe_value(input_dft)},",
            f"  occupations = {qe_value(occupations)},",
            f"  smearing = {qe_value(smearing)},",
            f"  degauss = {degauss},",
            f"  nspin = {int(nspin)},",
            "/",
            "",
            "&ELECTRONS",
            "  conv_thr = 1.0d-8,",
            "  mixing_beta = 0.3,",
            "/",
            "",
        ]
    )

    if calculation in {"relax", "vc-relax"}:
        lines.extend(
            [
                "&IONS",
                "  ion_dynamics = 'bfgs',",
                "/",
                "",
            ]
        )

    lines.append("ATOMIC_SPECIES")
    for sym in unique_symbols:
        z = atomic_numbers[sym]
        mass = float(atomic_masses[z])
        pseudo_file = element_pseudo_file(elements[sym], sym)
        lines.append(f"{sym:2s}  {mass:.6f}  {pseudo_file}")
    lines.append("")

    lines.append("CELL_PARAMETERS angstrom")
    for row in cell:
        lines.append(f"  {row[0]: .10f}  {row[1]: .10f}  {row[2]: .10f}")
    lines.append("")

    lines.append("ATOMIC_POSITIONS angstrom")
    for i, (sym, pos) in enumerate(zip(symbols, atoms.get_positions())):
        fx, fy, fz = constraint_flags(atoms, i)
        lines.append(
            f"{sym:2s}  {pos[0]: .10f}  {pos[1]: .10f}  {pos[2]: .10f}  {fx} {fy} {fz}"
        )
    lines.append("")

    lines.append("K_POINTS automatic")
    lines.append(f"  {kpoints}")
    lines.append("")

    output_path.write_text("\n".join(lines))


def write_metadata(
    metadata_path: Path,
    structure_path: Path,
    qe_input_path: Path,
    manifest_path: Path,
    manifest: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write provenance metadata next to the generated QE input."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage": "qe_pseudopotential_input_generation",
        "structure_file": repo_relative(structure_path),
        "qe_input_file": repo_relative(qe_input_path),
        "manifest_file": repo_relative(manifest_path),
        "calculation": args.calculation,
        "prefix": args.prefix,
        "pseudo_dir": args.pseudo_dir,
        "kpoints": args.kpoints,
        "ecutwfc": args.ecutwfc,
        "ecutrho": args.ecutrho,
        "input_dft": args.input_dft,
        "occupations": args.occupations,
        "smearing": args.smearing,
        "degauss": args.degauss,
        "nspin": args.nspin,
        "pseudopotential_manifest_snapshot": deepcopy(manifest),
        "notes": [
            "Pseudopotentials are applied at the parent QE electronic-structure stage.",
            "They are not added as a post-hoc correction to the reduced Hamiltonian.",
            "Downstream active-space and Hamiltonian outputs should reference this metadata file.",
        ],
    }

    metadata_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a QE input file with explicit pseudopotential provenance."
    )

    parser.add_argument(
        "structure",
        help="ASE-readable structure file, e.g. H-added or optimized .traj",
    )
    parser.add_argument(
        "--manifest",
        default="inputs/pseudopotentials/pseudopotential_manifest.yaml",
        help="YAML/JSON pseudopotential manifest",
    )
    parser.add_argument(
        "--output",
        default="outputs/qe/iro2_h_pseudopotential.in",
        help="Output QE input file",
    )
    parser.add_argument(
        "--metadata-output",
        default="outputs/qe/iro2_h_pseudopotential_metadata.json",
        help="Output provenance JSON",
    )
    parser.add_argument(
        "--calculation",
        choices=["scf", "relax", "vc-relax"],
        default="scf",
        help="QE calculation type. Start with scf for a low-disruption integration.",
    )
    parser.add_argument(
        "--prefix",
        default="iro2_h",
        help="QE prefix",
    )
    parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="Path to UPF pseudopotential directory. Defaults to manifest pseudo_dir.",
    )
    parser.add_argument(
        "--kpoints",
        default="2 2 1 0 0 0",
        help="QE automatic K_POINTS line: nk1 nk2 nk3 sk1 sk2 sk3",
    )
    parser.add_argument("--ecutwfc", type=float, default=None)
    parser.add_argument("--ecutrho", type=float, default=None)
    parser.add_argument("--input-dft", default=None)
    parser.add_argument("--occupations", default=None)
    parser.add_argument("--smearing", default=None)
    parser.add_argument("--degauss", type=float, default=None)
    parser.add_argument("--nspin", type=int, default=None)

    args = parser.parse_args()

    structure_path = Path(args.structure)
    manifest_path = Path(args.manifest)
    qe_input_path = Path(args.output)
    metadata_path = Path(args.metadata_output)

    manifest = load_manifest(manifest_path)
    defaults = manifest.get("calculation_defaults", {})

    args.pseudo_dir = args.pseudo_dir or manifest.get(
        "pseudo_dir", "inputs/pseudopotentials/upf"
    )
    args.ecutwfc = float(args.ecutwfc if args.ecutwfc is not None else defaults.get("ecutwfc", 60))
    args.ecutrho = float(args.ecutrho if args.ecutrho is not None else defaults.get("ecutrho", 480))
    args.input_dft = str(
        args.input_dft
        or defaults.get("input_dft")
        or manifest.get("exchange_correlation", "PBE")
    )
    args.occupations = str(args.occupations or defaults.get("occupations", "smearing"))
    args.smearing = str(args.smearing or defaults.get("smearing", "mv"))
    args.degauss = float(args.degauss if args.degauss is not None else defaults.get("degauss", 0.02))
    args.nspin = int(args.nspin if args.nspin is not None else defaults.get("nspin", 1))

    atoms = read(str(structure_path))

    write_qe_input(
        atoms=atoms,
        manifest=manifest,
        output_path=qe_input_path,
        calculation=args.calculation,
        prefix=args.prefix,
        pseudo_dir=args.pseudo_dir,
        kpoints=args.kpoints,
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho,
        input_dft=args.input_dft,
        occupations=args.occupations,
        smearing=args.smearing,
        degauss=args.degauss,
        nspin=args.nspin,
    )

    write_metadata(
        metadata_path=metadata_path,
        structure_path=structure_path,
        qe_input_path=qe_input_path,
        manifest_path=manifest_path,
        manifest=manifest,
        args=args,
    )

    print(f"[ok] wrote QE input: {qe_input_path}")
    print(f"[ok] wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
