# Changelog

All notable changes to the Soft Potato project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [PEP 440](https://peps.python.org/pep-0440/).

## [3.0.0.dev1] - 2026-09-20

### Initial Development Pre-release
Initial development build (`dev1`) marking the ground-up architectural rewrite for **Soft Potato 3.0**. This release sets up the core data models, analytical solutions, module scaffolding, and development infrastructure.

### Added
- **Core Module (`softpotato.core`)**:
  - Implemented the `Species` class with strict CGS unit enforcement:
    - Diffusion coefficient ($D$) in $\text{cm}^2/\text{s}$ ($D \ge 0$).
    - Bulk concentration ($c_{\text{bulk}}$) in $\text{mol}/\text{cm}^3$ ($c_{\text{bulk}} \ge 0$, $1\text{ mM} = 10^{-6}\text{ mol}/\text{cm}^3$).
    - Spatial concentration profile ($c_{\text{profile}}$) in $\text{mol}/\text{cm}^3$.
    - Ionic charge / valence number ($z$).
  - Added strict parameter validation for non-negative values, non-empty names, and integer charge.
  - Implemented profile memory management (`initialize_profile` to allocate 1D NumPy arrays and `reset` to restore initial bulk conditions).
  - Added `__hash__` method to `Species` allowing instances to be used in sets and as dictionary keys (e.g., in stoichiometry mappings).
  - Implemented `ElectrochemicalReaction` representing interfacial electron transfer ($O + n e^- \rightleftharpoons R$) with strict validation for `n_electrons`, `E0`, `kinetics`, and stoichiometric coefficients.
  - Implemented `ChemicalReaction` representing homogeneous bulk reactions ($\sum \nu_r R \rightleftharpoons \sum \nu_p P$) with support for kinetics and stoichiometry.
  - Implemented `Mechanism` container orchestrating reactions and species, featuring:
    - Automatic species discovery and order preservation from reactions.
    - Categorization into `e_reactions` (`electrochemical_reactions`) and `c_reactions` (`chemical_reactions`).
    - Indexing by species name (`mech["O"]` / `mech.get_species`) and reaction index (`mech[0]`).
    - Delegated profile initialization and reset across all species.
  - Implemented centralized physical constants module (`softpotato.core.constants`):
    - `FARADAY`: Faraday constant ($\approx 96485.33212\text{ C/mol}$).
    - `GAS_CONSTANT`: Molar gas constant $R$ ($\approx 8.3144626\text{ J/(mol}\cdot\text{K)}$).
    - `STANDARD_TEMPERATURE`: Standard temperature ($298.15\text{ K}$).
- **Analytical Solutions (`softpotato.analytical`)**:
  - Implemented closed-form, vectorized analytical solutions across modular subpackages:
    - **Potential Step & Chronocoulometry (`softpotato.analytical.techniques.step`)**:
      - `cottrell`: Transient current decay for planar diffusion.
      - `anson`: Cumulative charge transient for chronocoulometry including double-layer ($Q_{dl}$) and adsorption ($Q_{ads}$) charges in Coulombs.
      - `spherical_cottrell`: Transient current at a spherical electrode with area derived from radius $r_0$.
    - **Voltammetry (`softpotato.analytical.techniques.voltammetry`)**:
      - `randles_sevcik`: Peak current for reversible electron transfer.
      - `randles_sevcik_irreversible`: Peak current for totally irreversible electron transfer.
      - `peak_potential_irreversible`: Peak potential shift for totally irreversible electron transfer.
    - **Microelectrode Geometries (`softpotato.analytical.geometry.microelectrodes`)**:
      - `steady_state_microdisc`: Saito equation for steady-state limiting current at an inlaid microdisc with radius $a$.
      - `steady_state_microhemisphere`: Steady-state limiting current at a microhemisphere electrode.
    - **Hydrodynamics (`softpotato.analytical.geometry.hydrodynamics`)**:
      - `levich`: Mass-transport limiting current at a rotating disk electrode (RDE).
      - `koutecky_levich`: Net steady-state current combining kinetic and mass-transport limitations.
    - **Kinetics & Reversibility (`softpotato.analytical.kinetics.reversibility`)**:
      - `nicholson_psi`: Nicholson kinetic reversibility parameter $\Psi$.
    - **Reaction Mechanisms (`softpotato.analytical.kinetics.mechanisms`)**:
      - `catalytic_current`: Steady-state catalytic current for an $EC'$ mechanism (with alias `catalytic_current_ec_prime`).
  - Added top-level facade imports in `softpotato.analytical` for direct access to all equations.

- **Module Scaffolding**:
  - Established subpackage hierarchy:
    - `softpotato.geometry`: Grids (`grids.py`) and electrode geometries (`electrodes.py`).
    - `softpotato.simulate`: PDE solvers (`solver.py`), explicit finite difference (`efd.py`), and implicit finite difference (`ifd.py`).
    - `softpotato.techniques`: Electrochemical waveforms including voltammetry (`voltammetry.py`) and potential step (`step.py`).
    - `softpotato.kinetics`: Pluggable kinetic models (in development).
- **Testing & Quality Infrastructure**:
  - Added comprehensive test suite in `tests/test_core.py` covering `Species`, `ElectrochemicalReaction`, `ChemicalReaction`, and `Mechanism` (initialization, validations, memory allocations, property setters, representations, indexing, and stoichiometry matrix).
  - Configured `pyproject.toml` with package metadata, runtime dependencies (`numpy`, `scipy`, `matplotlib`), test dependencies (`pytest`), and tool configurations for `black`, `ruff`, and `mypy`.
  - Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) matrix testing across Python 3.10, 3.11, and 3.12.
  - Added root `.gitignore` covering Python cache directories, virtual environments, build artifacts, and editor settings.
- **Licensing & Project Metadata**:
  - Adopted the open-source **BSD 3-Clause License** (`LICENSE`).
  - Added license metadata and PyPI classifier (`License :: OSI Approved :: BSD License`) in `pyproject.toml`.
- **Documentation & Sphinx Framework (`docs/`)**:
  - Configured Sphinx documentation with Read the Docs integration (`.readthedocs.yaml`) using the `sphinx_rtd_theme`.
  - Added Sphinx configuration (`docs/conf.py`) integrating `autodoc`, `napoleon`, `mathjax`, `viewcode`, `todo`, `sphinx-math-dollar`, and `nbsphinx`.
  - Added main landing page (`docs/index.rst`), roadmap document (`docs/roadmap.rst`), and API reference structure (`docs/api/index.rst`).
  - Added Sphinx `.. warning::` and `.. todo::` directives across all placeholder/scaffolding modules (`geometry`, `simulate`, `techniques`, `kinetics`) to clearly signal their development status in generated documentation.
  - Added `docs/Makefile` and `docs/make.bat` build scripts for local documentation generation.
  - Added dedicated license page (`docs/license.rst`) embedding the root `LICENSE` file into the Sphinx documentation table of contents.
  - Added automated `docs` build validation job in GitHub Actions workflow (`.github/workflows/ci.yml`).
- **Interactive Tutorials & Examples (`examples/`)**:
  - Added comprehensive tutorial notebook `examples/mechanisms.ipynb` covering mechanism definitions in Soft Potato 3.0:
    - Single-electron transfers ($E$).
    - Chemical equilibria ($C$).
    - Coupled EC and multi-step ECE mechanisms.
    - Disproportionation pathways (ECE vs. DISP competition with multi-reactant/product stoichiometry, and bimolecular radical disproportionation $2R \rightleftharpoons O + Z$).
    - Solver integration features (profile initialization `initialize_profiles`, reset `reset`, dictionary indexing, and stoichiometry matrix inspection).
  - Linked interactive tutorial notebook into Sphinx documentation via `docs/examples/mechanisms.nblink`.
- **Dependencies (`pyproject.toml`)**:
  - Added optional `docs` dependency group (`sphinx`, `sphinx-rtd-theme`, `sphinx-autodoc-typehints`, `nbsphinx`, `nbsphinx-link`, `ipython`, `ipykernel`).

### Changed
- Configured Black `target-version = ["py310"]` in `pyproject.toml` to ensure consistent AST checks across the Python 3.10–3.12 CI matrix.
- Structured package exports in `softpotato/__init__.py` using explicit `__all__` declarations.
- Cleaned up docstrings in `src/softpotato/core/reactions.py` by removing redundant `Attributes:` sections to optimize Sphinx autodoc rendering.
- Formatted `src/softpotato/core/reactions.py` and `tests/test_core.py` to adhere strictly to Ruff formatting and lint rules.

### Removed
- Removed legacy `src/softpotato/core/README.md` in favor of Sphinx documentation.