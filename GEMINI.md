# GEMINI.md — Soft Potato (v0.1.0) Architecture & Memory

## Project Overview
**Soft Potato** is an open-source, extensible Python library for electrochemical simulations. 
Its core design goal is to decouple physical modeling (kinetics, thermodynamics, species transport) from numerical mechanics (meshing, spatial discretization, matrix assembly, and ODE/DAE solving).

---

## The 3 Golden Architectural Rules

1. **Strict Physics-Numerics Decoupling**
   - Non-developer electrochemists must be able to define chemical species ($R$, $O$), diffusion coefficients ($D_R$, $D_O$), and Nernstian boundary conditions symbolically without touching spatial indices or stencil operators.

2. **Interface Enforcement via Core ABCs**
   - Every modular component **MUST** inherit from its corresponding Abstract Base Class in `src/softpotato/core/abcs.py`:
     - `BaseMesh`: Uniform 1D spatial grid generator.
     - `BaseModel`: Species transport equations ($C_R$, $C_O$) and physical constants.
     - `BaseBoundaryCondition`: Nernstian potential-dependent concentration ratio at $x=0$ & Dirichlet bulk at $x=L$.
     - `BaseDiscretizer`: Finite Difference Method (FDM) spatial Laplacian matrix assembly.
     - `BaseSolver`: Time integration engine (e.g., SciPy `solve_ivp` with BDF method).

3. **Standard Package Layout (`src/`)**
   - All package code resides in `src/softpotato/`.
   - Never import internal files across submodules directly if it bypasses abstract interfaces.
   - Development is managed via `pyproject.toml` (PEP 517/621).

---

## Directory & Submodule Layout

softpotato/
├── GEMINI.md                      <-- AI Project Memory
├── pyproject.toml                 <-- Package build configuration
├── src/
│   └── softpotato/
│       ├── __init__.py            <-- Version (0.1.0) and top-level ABC exports
│       ├── core/
│       │   ├── __init__.py
│       │   └── abcs.py            <-- Core Abstract Base Classes
│       ├── mesh/                  <-- Uniform 1D grid generator (Uniform1DMesh)
│       ├── physics/               <-- Species (R, O) & Nernstian BCs
│       ├── discretizers/          <-- 1D Finite Difference operators (FDM1D)
│       └── solvers/               <-- SciPy BDF solver wrapper & CV voltage waveform
└── tests/                         <-- Pytest unit tests & Randles-Sevcik validation

---

## Technical Stack & Quality Standards

- **Python Version:** `>= 3.11`
- **Core Dependencies:** `numpy`, `scipy` (sparse matrices & BDF ODE solver), `sympy` (symbolic parsing), `pydantic` (parameter validation)
- **Dev Dependencies:** `pytest` (testing), `ruff` (formatting & linting), `matplotlib` (plotting cyclic voltammograms)
- **Testing:** All numerical components must include unit tests in `tests/`.
- **Typing:** Use strict type hints (`typing`) across all signatures.

---

## Current Development Phase: Phase 1 MVP (Reversible CV with FDM)

Target Physics:
- Reaction: $R \rightarrow O + e^-$ (Fully reversible oxidation)
- Initial conditions at $t=0$: $C_R(x, 0) = C_R^*$, $C_O(x, 0) = 0$
- Boundary at $x=0$: Nernst equation $C_O/C_R = \exp\left[\frac{F}{RT}(E(t) - E^0)\right]$ with equal/opposite flux constraint ($D_R \frac{\partial C_R}{\partial x} = -D_O \frac{\partial C_O}{\partial x}$)
- Boundary at $x=L$: Bulk concentration $C_R(L, t) = C_R^*$, $C_O(L, t) = 0$

Task Checklist:
- [x] Repository initialization (`src/` layout + `pyproject.toml`)
- [x] Core contract enforcement (`src/softpotato/core/abcs.py`)
- [ ] **Active Task:** Implement Uniform 1D Mesh (`src/softpotato/mesh/uniform_1d.py`)
- [ ] Next Task: Implement Two-Species Transport Model for $R$ and $O$ (`src/softpotato/physics/species.py`)
- [ ] Next Task: Implement 1D Finite Difference Discretizer (`src/softpotato/discretizers/fdm_1d.py`)
- [ ] Next Task: Implement Nernstian Equilibrium Boundary Condition (`src/softpotato/physics/nernst_bc.py`)
- [ ] Next Task: Implement ODE Solver Wrapper & Triangular Voltage Waveform (`src/softpotato/solvers/cv_solver.py`)
- [ ] Next Task: Validation test: Verify peak current $i_p$ matches Randles-Sevcik equation ($i_p = 0.4463 n F A C^* \sqrt{\frac{n F v D}{R T}}$) in `tests/test_reversible_cv.py`

---

## Prompting Rules for AI Collaborators

When generating code for Soft Potato:
1. Always check if the module respects `src/softpotato/core/abcs.py`.
2. Do not hardcode mesh indices into physical equations.
3. Use second-order central finite differences ($\frac{\partial^2 C}{\partial x^2} \approx \frac{C_{i+1} - 2C_i + C_{i-1}}{\Delta x^2}$) for spatial diffusion operators.
4. Provide unit tests using `pytest` alongside any new module implementation.
