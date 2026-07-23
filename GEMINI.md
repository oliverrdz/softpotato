# Soft Potato (v0.1.0) Architecture & Memory

## Project overview
Soft Potato is an open-source, extensible Python library for electrochemical
simulations. Its core design goal is to decouple physical modeling (kinetics,
thermodynamics, species transport) from numerical mechanics (meshing, spatial
discretization, matrix assembly, and ODE/DAE solving).

## The 3 golden architecture rules

1. Strict Physics-Numerics Decoupling
   - Non-developer electrochemists must be able to define chemical species,
     diffusion coefficients, and boundary kinetics symbolically (e.g., using
     SymPy or high-level declarative classes) without touching mesh indexing
     or sparse matrix operators.

2. Interface Enforcement via Core ABCs
   - Every modular component MUST inherit from its corresponding Abstract Base
     Class in src/softpotato/core/abcs.py:
     - BaseMesh: Spatial grid definitions.
     - BaseModel: Symbolic PDE equations and species properties.
     - BaseBoundaryCondition: Interface flux and concentration boundaries.
     - BaseDiscretizer: Operator matrix assembly (e.g., Finite Volume Method).
     - BaseSolver: Time integration step engines (e.g., SciPy solve_ivp).

3. Standard Package Layout (src/)
   - All package code resides in src/softpotato/.
   - Never import internal files across submodules directly if it bypasses
     abstract interfaces.
   - Development is managed via pyproject.toml (PEP 517/621).

## Directory and submodule layout

```
softpotato/
├── GEMINI.text                    <-- AI Project Memory
├── pyproject.toml                 <-- Package build configuration
├── src/
│   └── softpotato/
│       ├── __init__.py            <-- Version (0.1.0) and top-level ABC exports
│       ├── core/
│       │   ├── __init__.py
│       │   └── abcs.py            <-- Core Abstract Base Classes
│       ├── mesh/                  <-- 1D/2D/3D non-uniform spatial grid generators
│       ├── physics/               <-- Species, Butler-Volmer/Marcus kinetics, transport
│       ├── discretizers/          <-- Finite Volume / Finite Difference operators
│       └── solvers/               <-- SciPy / SUNDIALS integration wrappers
└── tests/                         <-- Pytest unit tests (mirrors src structure)
```

## Technical stack and quality standards

- Python Version: >= 3.11
- Core Dependencies: numpy, scipy (sparse matrices), sympy (symbolic parsing),
  pydantic (data validation)
- Dev Dependencies: pytest (testing), ruff (formatting & linting),
  matplotlib (visualization)
- Testing: All new numerical components must include corresponding pytest suites
  in tests/.
- Typing: Use strict type hints (typing) across all signatures and Pydantic
  models where applicable.

## Current development phase: Phase 1 MVP

* [X] Repository initialization (src/ layout + pyproject.toml)
* [X] Core contract enforcement (src/softpotato/core/abcs.py)
* [ ] Active Task: Implement 1D Non-Uniform Mesh Generator (src/softpotato/mesh/mesh_1d.py)
* [ ] Next Task: Implement 1D FVM Discretizer for Fickian Diffusion
* [ ] Next Task: Implement Butler-Volmer Boundary Condition
* [ ] Next Task: Complete first end-to-end Cyclic Voltammetry simulation test

## Prompting rules for AI collaborators

When generating code for Soft Potato:
1. Always check if the module respects src/softpotato/core/abcs.py.
2. Do not mix spatial grid calculations into kinetics or species classes.
3. Prefer SciPy sparse matrices (scipy.sparse.csr_matrix) over dense NumPy
   arrays for spatial discretization matrices.
4. Provide unit tests using pytest alongside any new module implementation.
