# Contributing to Soft Potato

Thank you for your interest in contributing to **Soft Potato**! Soft Potato is an open-source electrochemical simulator and toolkit designed for electrochemists, materials scientists, and engineers.

We welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, new electrochemical models, and simulation algorithms.

---

## Code of Conduct

Please help us keep the Soft Potato community welcoming and respectful. We expect all contributors and participants to adhere to respectful, constructive, and inclusive collaboration.

---

## Development Setup

### 1. Fork & Clone
Clone the repository to your local machine:
```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
```

### 2. Set Up a Virtual Environment
Create and activate a virtual environment using `venv` or `uv`:
```bash
# Using standard venv
python3 -m venv .venv
source .venv/bin/activate

# Or using uv
uv venv
source .venv/bin/activate
```

### 3. Install in Editable Mode
Install the package along with development and documentation dependencies:
```bash
pip install -e ".[test,docs]"
```

---

## Running Tests

Soft Potato uses [pytest](https://docs.pytest.org/) for its test suite. Ensure all tests pass before submitting a pull request:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test module
pytest tests/test_core.py
```

---

## Building Documentation

Soft Potato uses [Sphinx](https://www.sphinx-doc.org/) for documentation generation.

To build the HTML documentation locally:
```bash
cd docs
make html
```
The generated documentation will be available in `docs/_build/html/index.html`.

---

## Codebase Structure

Below is an overview of the repository layout and module responsibilities:

```text
softpotato/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI workflow (multi-version test matrix & doc builds)
├── .gitignore                              # Git ignore rules for Python, virtual environments, and build artifacts
├── .readthedocs.yaml                       # Read the Docs configuration file
├── CHANGELOG.md                            # Project changelog following Keep a Changelog format
├── CONTRIBUTING.md                         # Contributor guidelines and repository structure overview
├── docs/                                   # Sphinx documentation source files
│   ├── _static/
│   │   └── .gitkeep                        # Preserves static assets directory in git
│   ├── _templates/
│   │   └── .gitkeep                        # Preserves custom HTML templates directory in git
│   ├── api/
│   │   └── index.rst                       # API reference documentation index
│   ├── examples/
│   │   ├── index.rst                       # User guide and tutorial notebooks index
│   │   └── mechanisms.nblink               # Sphinx link to mechanisms.ipynb tutorial notebook
│   ├── conf.py                             # Sphinx build configuration and extension settings
│   ├── design_spec.rst                     # API and UI design specifications
│   ├── index.rst                           # Documentation homepage and table of contents
│   ├── make.bat                            # Windows batch build script for Sphinx documentation
│   ├── Makefile                            # Unix makefile for Sphinx documentation
│   └── roadmap.rst                         # Development roadmap and pending implementation tracking
├── examples/                               # Interactive Jupyter notebook tutorials
│   ├── analytical_benchmarks.ipynb         # Interactive benchmark calculations (Randles-Sevcik, Cottrell, Saito)
│   └── mechanisms.ipynb                    # Tutorial notebook on defining electrochemical and chemical mechanisms
├── LICENSE                                 # BSD 3-Clause License file
├── pyproject.toml                          # Build system, package metadata, dependencies, and tool settings
├── README.md                               # Project overview, specifications, examples, and roadmap
├── scaffold.sh                             # Shell script for repository and module scaffolding
├── src/
│   └── softpotato/                         # Root namespace for the Soft Potato package
│       ├── __init__.py                     # Package root exposing top-level API and submodules
│       ├── analytical/                     # Closed-form analytical electrochemical benchmark equations
│       │   ├── __init__.py                 # Facade exposing primary analytical functions (Randles-Sevcik, Cottrell, Saito)
│       │   ├── geometry/                   # Geometry-dependent steady-state analytical solutions
│       │   │   ├── __init__.py             # Exposes geometry-dependent analytical functions
│       │   │   ├── hydrodynamics.py        # Analytical solutions for hydrodynamic electrodes (e.g., Levich)
│       │   │   └── microelectrodes.py      # Analytical steady-state solutions for microelectrodes (e.g., Saito)
│       │   ├── kinetics/                   # Kinetic reversibility and mechanism diagnostic equations
│       │   │   ├── __init__.py             # Exposes analytical kinetic diagnostics
│       │   │   ├── mechanisms.py           # Diagnostic equations for coupled chemical mechanisms (EC, ECE)
│       │   │   └── reversibility.py        # Diagnostic criteria for electrochemical reversibility (Matsuda-Ayabe, Nicholson)
│       │   └── techniques/                 # Transient technique-specific analytical solutions
│       │       ├── __init__.py             # Exposes technique-specific analytical functions
│       │       ├── step.py                 # Potential step solutions (e.g., Cottrell equation)
│       │       └── voltammetry.py          # Voltammetry peak current solutions (e.g., Randles-Sevcik)
│       ├── core/                           # Core data structures and reaction mechanism representations
│       │   ├── __init__.py                 # Exposes core classes (Species, Reactions, Mechanism)
│       │   ├── reactions.py                # ElectrochemicalReaction, ChemicalReaction, and Mechanism classes
│       │   └── species.py                  # Species data class with strict CGS unit enforcement
│       ├── geometry/                       # Spatial discretizations and electrode geometry models
│       │   ├── __init__.py                 # Exposes grid and electrode classes
│       │   ├── electrodes.py               # Electrode geometry implementations (Planar, Spherical, ThinLayer)
│       │   └── grids.py                    # Spatial grid generators (UniformGrid, ExpandingGrid)
│       ├── kinetics/                       # Electrochemical and chemical kinetic models
│       │   ├── __init__.py                 # Exposes kinetic rate and boundary condition models
│       │   └── models.py                   # Butler-Volmer, Nernst, and chemical kinetic rate models
│       ├── simulate/                       # Numerical PDE solvers and simulation orchestrators
│       │   ├── __init__.py                 # Exposes simulation solvers and engines
│       │   ├── efd.py                      # Explicit Finite Difference (EFD) solver implementation
│       │   ├── ifd.py                      # Implicit Finite Difference (IFD) / Crank-Nicolson solver
│       │   └── solver.py                   # Main Solver class orchestrating simulation execution
│       └── techniques/                     # Electrochemical excitation signals and waveforms
│           ├── __init__.py                 # Exposes technique waveform generators
│           ├── step.py                     # Potential step waveforms (e.g., Chronoamperometry)
│           └── voltammetry.py              # Voltammetry waveforms (e.g., CyclicVoltammetry)
└── tests/                                  # Test suite for unit, integration, and validation tests
    ├── conftest.py                         # Shared pytest fixtures and test configurations
    ├── test_core.py                        # Unit tests for Species, Reactions, and Mechanism classes
    ├── test_geometry_laplacians.py         # Tests for spatial discretization grids and Laplacian operators
    ├── test_kinetics.py                    # Tests for Butler-Volmer and chemical reaction kinetics
    └── test_solvers.py                     # Validation tests for numerical solvers against analytical baselines
```

---

## Contribution Guidelines

1. **Create a branch**: Branch off `main` using a descriptive name (e.g., `feature/marcus-hush` or `fix/grid-discretization`).
2. **Follow code style**: Soft Potato adheres to standard Python formatting (PEP 8) and type annotations.
3. **Unit system**: Remember that Soft Potato strictly uses the **CGS unit system** ($D$ in $\text{cm}^2/\text{s}$, $c_{\text{bulk}}$ in $\text{mol}/\text{cm}^3$, area in $\text{cm}^2$, radius in $\text{cm}$, potential in $\text{V}$).
4. **Write tests**: Add unit and validation tests under `tests/` for any new functionality or bug fixes.
5. **Update documentation**: If changing or adding APIs, update docstrings, Sphinx documentation under `docs/`, and `CHANGELOG.md`.
6. **Submit a Pull Request**: Open a PR against `main` with a clear description of the changes.
