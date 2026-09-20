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
  - Added placeholder classes in `reactions.py` (`ElectrochemicalReaction`, `ChemicalReaction`, `Mechanism`) with backwards compatibility aliases (`HeterogeneousReaction`, `HomogeneousReaction`).
- **Analytical Solutions (`softpotato.analytical`)**:
  - Added vectorized analytical equations for baseline comparisons:
    - `randles_sevcik`: Peak current for reversible cyclic voltammetry.
    - `cottrell`: Transient current response for planar potential steps.
    - `steady_state_microdisc`: Steady-state limiting current (Saito equation).
- **Module Scaffolding**:
  - Established subpackage hierarchy:
    - `softpotato.geometry`: Grids (`grids.py`) and electrode geometries (`electrodes.py`).
    - `softpotato.simulate`: PDE solvers (`solver.py`), explicit finite difference (`efd.py`), and implicit finite difference (`ifd.py`).
    - `softpotato.techniques`: Electrochemical waveforms including voltammetry (`voltammetry.py`) and potential step (`step.py`).
    - `softpotato.kinetics`: Pluggable kinetic models (in development).
- **Testing & Quality Infrastructure**:
  - Added comprehensive test suite in `tests/test_core.py` covering `Species` initialization, validations, memory allocations, property setters, and representations.
  - Configured `pyproject.toml` with package metadata, runtime dependencies (`numpy`, `scipy`, `matplotlib`), test dependencies (`pytest`), and tool configurations for `black`, `ruff`, and `mypy`.
  - Added GitHub Actions CI workflow (`.github/workflows/ci.yml`) matrix testing across Python 3.10, 3.11, and 3.12.
  - Added root `.gitignore` covering Python cache directories, virtual environments, build artifacts, and editor settings.

### Changed
- Configured Black `target-version = ["py310"]` in `pyproject.toml` to ensure consistent AST checks across the Python 3.10–3.12 CI matrix.
- Structured package exports in `softpotato/__init__.py` using explicit `__all__` declarations.