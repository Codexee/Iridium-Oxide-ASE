# Goal:
Introduce pseudopotential-aware parent electronic-structure workflows before active-space extraction.

Principle:
Pseudopotentials enter at the DFT/QE/GPAW parent-calculation stage, not as an ad hoc correction after reduced Hamiltonian construction.

Initial scope:
Track pseudopotential provenance, input files, chosen functional, valence configuration, relativistic treatment, and generated parent Hamiltonian/source data.

# Pseudopotential workflow

The pseudopotential-aware workflow introduces explicit pseudopotential choices at the parent electronic-structure calculation stage.

Current submitted workflow:
QE slab input → H addition → XTB optimisation → XTB orbital analysis → manual active orbital selection → fermionic Hamiltonian construction.

Development workflow:
QE slab input → H addition → QE pseudopotential calculation → orbital/projection analysis → manual active orbital selection → fermionic Hamiltonian construction.

Pseudopotentials are not added as a correction to the reduced Hamiltonian. They define the parent electronic structure from which the reduced Hamiltonian is derived.

Initial implementation will preserve the existing XTB route and add a parallel QE/pseudopotential route.
