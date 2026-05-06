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

**Preprint:** [A Reproducible Workflow for Extracting Quantum Hamiltonians from 
Surface-Adsorbate Models](https://zenodo.org/doi/10.5281/zenodo.19491027) — under review at *Nature* (2026)

## Scientific scope

The workflow supports:

- Automated generation of IrO₂ slab models and adsorption configurations

- Geometry optimisation using semi-empirical and first-principles backends

- Systematic comparison of surface-bound intermediates across sites and coverages

- Extraction of chemically meaningful active subspaces

- Reduction of ab initio Hamiltonians for downstream classical and quantum simulations

- Benchmarking and validation against experimental observables

The pipeline is intended to bridge atomistic simulation, reduced-order modelling,
and emerging quantum algorithms in a single reproducible framework.

## Repository layout

**scripts/**  
Python scripts for slab construction, adsorption-site generation, geometry optimisation,
active-space extraction, Hamiltonian construction, and analysis.

**inputs/**  
Reference structures, slab definitions, and configuration files used by the workflow.

**examples/**  
Self-contained example workflows demonstrating full end-to-end execution of the
pipeline for specific surface states and adsorbates.

**outputs/**  
Generated structures, optimisation results, reduced Hamiltonians, and analysis artefacts
(excluded from version control where appropriate).

**tests/**  
Automated tests executed via GitHub Actions to validate core workflow components.

## Reproducibility and development status

Initial IrO₂ slab construction and baseline relaxation were performed using Quantum
ESPRESSO, prior to ASE-based automation and analysis.

The workflow is under active development:

The main branch contains stable examples, documentation, and tested reference scripts.

A dedicated development branch contains the full end-to-end pipeline used to generate
current benchmark datasets, including active-space selection and Hamiltonian reduction.

Versioned releases will be used to tag fully reproducible benchmark states.

### Running locally

An end-to-end example of the IrO₂ H* adsorption workflow is provided in:

```examples/iro2_h_star_o69/```


This example executes the full pipeline via a shell script that orchestrates the individual
Python stages in the correct order.

To run locally:

```cd examples/iro2_h_star_o69```  
```bash run_all.sh``` 

### Note on workflow orchestration

The workflow is intentionally not driven by a single monolithic Python script.
Instead, run_all.sh acts as a lightweight orchestration layer that:

Executes each stage (setup, optimisation, analysis, reduction) explicitly

Preserves intermediate outputs for inspection and debugging

Makes the execution order transparent and reproducible

This design choice reflects the exploratory and comparative nature of the project, and
allows individual stages to be rerun or substituted without modifying core scripts.

Users extending the workflow to new surface states, adsorbates, or materials systems are
encouraged to copy and adapt the example run_all.sh rather than invoking scripts
individually.

## Citation

If you use this workflow, please cite:

Malhotra, M., Ramarapu, N., & Rinaldi, F. (2026). A Reproducible Workflow for 
Extracting Quantum Hamiltonians from Surface-Adsorbate Models. Zenodo. 
https://doi.org/10.5281/zenodo.19491027

Under review at *Nature*.
