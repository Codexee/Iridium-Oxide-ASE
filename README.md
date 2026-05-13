[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19491027.svg)](https://zenodo.org/doi/10.5281/zenodo.19491027)

# Iridium Oxide ASE

This repository contains an open-source, reproducible workflow for constructing, analysing,
and reducing atomistic models of IrO₂ electrocatalyst surfaces using the Atomic Simulation
Environment (ASE).

The project is designed to support comparative surface science studies, hypothesis
generation for operando experiments (e.g. AP-XPS / AP-XAS), and model reduction for
hybrid classical–quantum simulation workflows.

While the current focus is IrO₂ and hydrogen adsorption intermediates, the workflow is
deliberately modular and extensible to other materials systems and physics-driven
comparative studies.

---

## Preprint

**A Reproducible Workflow for Extracting Quantum Hamiltonians from Surface-Adsorbate Models**  
Under review at *Nature* (2026)

DOI: https://doi.org/10.5281/zenodo.19491027

---

## Scientific Scope

The workflow supports:

- Automated generation of IrO₂ slab models and adsorption configurations
- Geometry optimisation using semi-empirical and first-principles backends
- Systematic comparison of surface-bound intermediates across sites and coverages
- Extraction of chemically meaningful active subspaces
- Reduction of *ab initio* Hamiltonians for downstream classical and quantum simulations
- Benchmarking and validation against experimental observables

The pipeline is intended to bridge atomistic simulation, reduced-order modelling,
and emerging quantum algorithms in a single reproducible framework.

---

## Repository Layout

```text
.
├── scripts/      # Slab construction, adsorption-site generation,
│                 # optimisation, Hamiltonian construction, analysis
│
├── inputs/       # Reference structures, slab definitions,
│                 # workflow configuration files
│
├── examples/     # End-to-end example workflows
│
├── outputs/      # Generated structures, optimisation results,
│                 # reduced Hamiltonians, analysis artefacts
│
├── tests/        # Automated validation tests
│
└── README.md
```

### Directory Details

#### `scripts/`

Python scripts for:

- Slab construction
- Adsorption-site generation
- Geometry optimisation
- Active-space extraction
- Hamiltonian construction
- Analysis and post-processing

#### `inputs/`

Reference structures, slab definitions, and configuration files used by the workflow.

#### `examples/`

Self-contained example workflows demonstrating full end-to-end execution of the
pipeline for specific surface states and adsorbates.

#### `outputs/`

Generated structures, optimisation results, reduced Hamiltonians, and analysis artefacts.

Large generated datasets are excluded from version control where appropriate.

#### `tests/`

Automated tests executed locally and via GitHub Actions to validate core workflow
components and maintain reproducibility.

---

## Installation

### Requirements

Recommended environment:

- Python 3.10+
- Linux or macOS
- Git
- Bash shell

Core Python packages:

- ase
- numpy
- scipy
- matplotlib
- pandas

Optional external software depending on workflow stage:

- Quantum ESPRESSO
- xTB
- PySCF
- Qiskit

---

### Clone the Repository

```bash
git clone https://github.com/<your-org>/<repo-name>.git
cd <repo-name>
```

---

### Create a Local Environment

Using Python virtual environments:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If a requirements file is not yet provided:

```bash
pip install ase numpy scipy matplotlib pandas
```

---

## Running Locally

An end-to-end example of the IrO₂ H* adsorption workflow is provided in:

```text
examples/iro2_h_star_o69/
```

This example executes the full workflow via a shell script that orchestrates
the individual Python stages in the correct order.

### Run the Workflow

```bash
cd examples/iro2_h_star_o69
bash run_all.sh
```

---

## Workflow Orchestration Philosophy

The workflow is intentionally **not** driven by a single monolithic Python script.

Instead, `run_all.sh` acts as a lightweight orchestration layer that:

- Executes each workflow stage explicitly
- Preserves intermediate outputs for inspection and debugging
- Makes execution order transparent and reproducible
- Allows individual stages to be rerun independently
- Enables substitution of methods or backends without modifying core scripts

This design reflects the exploratory and comparative nature of atomistic
surface-science workflows.

Users extending the workflow to new surface states, adsorbates, or materials systems
are encouraged to copy and adapt the example `run_all.sh` scripts rather than invoking
internal scripts individually.

---

## Manual Workflow Design

The repository intentionally preserves a semi-manual workflow structure.

Rather than hiding all execution logic behind opaque automation layers, the project
prioritises:

- Transparency of simulation stages
- Inspectability of intermediate artefacts
- Explicit execution order
- Ease of debugging
- Scientific reproducibility
- Flexible substitution of computational methods

This approach is particularly valuable for:

- Comparative surface studies
- Active-space experimentation
- Benchmark generation
- Exploratory adsorption analysis
- Hybrid classical–quantum workflows

Intermediate files are considered scientifically meaningful outputs rather than
temporary artefacts.

---

## Continuous Integration

Core workflow components are validated through automated testing using GitHub Actions.

Run tests locally:

```bash
pytest tests/
```

Example GitHub Actions workflow:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/
```

CI validation is intended to ensure workflow stability while preserving the
modular and exploratory nature of the repository.

---

## Reproducibility and Development Status

Initial IrO₂ slab construction and baseline relaxation were performed using
Quantum ESPRESSO prior to ASE-based automation and analysis.

The workflow is under active development.

### `main` branch

Contains:

- Stable examples
- Documentation
- Tested reference scripts
- Reproducible baseline workflows

### `development` branch

Contains:

- Full end-to-end benchmark generation pipeline
- Active-space selection workflows
- Hamiltonian reduction workflows
- Experimental comparative pipelines

Versioned releases will be used to tag fully reproducible benchmark states.

---

## Citation

If you use this workflow, please cite:

```bibtex
@misc{malhotra2026iro2,
  author       = {Malhotra, M. and Ramarapu, N. and Rinaldi, F.},
  title        = {A Reproducible Workflow for Extracting Quantum Hamiltonians from Surface-Adsorbate Models},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19491027},
  url          = {https://doi.org/10.5281/zenodo.19491027}
}
```

Under review at *Nature*.

---

## License

Add your preferred open-source license here (e.g. MIT, BSD-3-Clause, Apache-2.0).

---

## Acknowledgements

This work integrates concepts from:

- Surface science and heterogeneous catalysis
- Electronic-structure theory
- Reduced-order modelling
- Quantum simulation workflows
- Reproducible computational science
