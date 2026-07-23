# GEMINI.md — Soft Potato (v0.1.0) Architecture & Memory

## Project Overview
**Soft Potato** is an open-source, extensible Python library for electrochemical simulations. 
Its core design goal is to decouple physical modeling (kinetics, thermodynamics, species transport) from numerical mechanics (meshing, spatial discretization, matrix assembly, and ODE/DAE solving).

---

## The 3 Golden Architectural Rules

1. **Strict Physics-Numerics Decoupling**
   - Non-developer electrochemists must be able to define chemical species, diffusion coefficients, and boundary kinetics symbolically (e.g., using SymPy or high-level declarative classes) without touching mesh indexing or sparse matrix operators.

2. **Interface Enforcement via Core ABCs**
   - Every modular component **MUST** inherit from its corresponding Abstract Base Class in `src/softpotato/core/abcs.py`:
     - `BaseMesh`: Spatial grid definitions.
     - `BaseModel`: Symbolic PDE equations and species properties.
     - `BaseBoundaryCondition`: Interface flux and concentration boundaries.
     - `BaseDiscretizer`: Operator matrix assembly (e.g., Finite Volume Method).
     - `BaseSolver`: Time integration step engines (e.g., SciPy `solve_ivp`).

3. **Standard Package Layout (`src/`)**
   - All package code resides in `src/softpotato/`.
   - Never import internal files across submodules directly if it bypasses abstract interfaces.
   - Development is managed via `pyproject.toml` (PEP 517/621).

---

## Directory & Submodule Layout

```text
softpotato/
├── GEMINI.md                      <-- AI Project Memory
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
