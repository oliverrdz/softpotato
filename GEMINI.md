# Soft Potato Project Context & Architecture Guidelines

## Project Overview
Soft Potato is an open-source, highly modular Python library for electrochemical simulations (e.g., Cyclic Voltammetry, Chronoamperometry, EIS).

## Core Architecture & Principles
1. **Strict Decoupling:** Mesh, Physics, Techniques, Discretizers, Solvers, and Data Exporters are completely independent components.
2. **Abstract Base Classes:** All core abstractions inherit from BaseMesh, BaseModel, BaseTechnique, BaseDiscretizer, BaseSolver, etc.
3. **Pydantic Validation:** Models, techniques, and meshes use Pydantic v2 for attribute validation.
4. **Clean Top-Level Exports:** Every submodule re-exports its public classes in its __init__.py file for intuitive imports.

## Directory Structure
```
softpotato/
├── GEMINI.md                      # AI Memory & context rules
├── LICENSE                        # MIT License
├── pyproject.toml                 # Build configuration
├── README.md                      # Documentation
├── examples/                      # Runnable simulation examples
│   └── 01_reversible_cv.py        # 1D Reversible Cyclic Voltammetry example
├── src/
│   └── softpotato/
│       ├── __init__.py            # Top-level exports & version
│       ├── core/                  # Core abstract base classes (abcs.py)
│       ├── mesh/                  # Spatial grid generators (uniform_1d.py)
│       ├── physics/               # Kinetic models & boundary conditions
│       ├── discretizers/          # Spatial derivative stencils (fdm_1d.py)
│       ├── techniques/            # Potential excitation waveforms (cyclic_voltammetry.py)
│       ├── io/                    # Data export/import module (planned)
│       └── solvers/               # Time integration engines (ode_solver.py)
└── tests/                         # Pytest suite
```

## Roadmap & Tasks
- [x] Phase 1 MVP: 1D Reversible CV with FDM and Randles-Ševčík validation suite
- [x] Promote excitation waveforms to `techniques` submodule (`CyclicVoltammetry`)
- [x] Re-export top-level classes across all submodule `__init__.py` files
- [x] Create `examples/` directory with `01_reversible_cv.py`
- [ ] Develop data export module (`softpotato.io` or `softpotato.exporters`) to export `SimulationResult` outputs to CSV, JSON, Pandas DataFrames, and HDF5 files
- [ ] Implement Butler-Volmer kinetics for quasi-reversible and irreversible electron transfer processes
- [ ] Add Chronoamperometry (`Chronoamperometry`) technique and Cottrell equation validation test suite

## Prompting Rules for AI Collaborators

When generating code for Soft Potato:
1. Always check if the module respects `src/softpotato/core/abcs.py`.
2. Do not hardcode mesh indices into physical equations.
3. Provide unit tests using `pytest` alongside any new module implementation.
