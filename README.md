# Soft Potato

An open-source, extensible Python library for electrochemical simulations. Soft Potato decouples physical modeling (kinetics, thermodynamics, species transport) from numerical mechanics (meshing, spatial discretization, matrix assembly, and time integration solvers).

---

## Key Features

* **Physics-Numerics Decoupling:** Define chemical species, diffusion coefficients, and boundary reactions symbolically without touching mesh indices or matrix operators.
* **Strict Component Contracts:** Built on Abstract Base Classes (`BaseMesh`, `BaseModel`, `BaseBoundaryCondition`, `BaseDiscretizer`, `BaseSolver`).
* **Modern Python Packaging:** Structured using the standard `src/` layout and PEP 517/621 `pyproject.toml` standards targeting Python >= 3.11.

---

## Installation

Clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/your-org/softpotato.git
cd softpotato
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To verify the installation and run the architectural unit tests:

```bash
pytest
```

---

## Quickstart & Usage Example

Here is how to set up and run a 1-electron reversible oxidation (R -> O + e-) Cyclic Voltammetry simulation using Soft Potato once Phase 1 MVP components are assembled:

```bash
import softpotato as sp
from softpotato.mesh import Uniform1DMesh
from softpotato.physics import ReversibleOxidationModel, NernstianBoundary
from softpotato.discretizers import FDM1DDiscretizer
from softpotato.solvers import CVSolver
```

# 1. Generate a 1D spatial mesh (0 to 0.1 cm with 200 grid points)
```python
mesh = Uniform1DMesh(x_min=0.0, x_max=0.1, n_points=200)
```

# 2. Define species transport model (D_R = 1e-5 cm²/s, D_O = 1e-5 cm²/s)
```python
model = ReversibleOxidationModel(D_R=1e-5, D_O=1e-5, C_R_bulk=1.0, C_O_bulk=0.0)
```

# 3. Set up Nernstian surface boundary condition at x = 0
```python
boundary = NernstianBoundary(E_0=0.0, n_electrons=1, T=298.15)
```

# 4. Discretize spatial derivatives using 2nd-order Finite Difference Method
```python
discretizer = FDM1DDiscretizer()
matrices = discretizer.assemble(model, mesh)
```
# 5. Run Cyclic Voltammetry solver (scan rate v = 0.1 V/s)
```python
solver = CVSolver(E_start=-0.2, E_vertex=0.4, v=0.1)
results = solver.solve(model, mesh, matrices, boundary)
```
# 6. Access current response and potential array
```python
currents = results.current
potentials = results.potential
```
---

## Project Directory Structure

```
softpotato/
├── GEMINI.txt                      <-- AI Project Memory
├── CHANGELOG.md                    <-- Project changelog (Keep a Changelog format)
├── PROJECT_MANAGEMENT.md           <-- Workflow, commits, and release guidelines
├── pyproject.toml                  <-- PEP 517/621 build configuration
├── src/
│   └── softpotato/
│       ├── __init__.py             <-- Top-level package exports
│       ├── core/
│       │   └── abcs.py             <-- Foundational ABC contracts
│       ├── mesh/                   <-- 1D/2D spatial mesh generators
│       ├── physics/                <-- Species definitions & boundary kinetics
│       ├── discretizers/           <-- Spatial operators (FDM, FVM)
│       └── solvers/                <-- Time integrators & CV voltage sweep engines
└── tests/                          <-- Pytest unit tests & Randles-Sevcik validation
```

---

## Current Development Phase: Phase 1 MVP

* **Target Reaction:** R -> O + e- (Fully reversible oxidation)
* **Spatial Grid:** Uniform 1D Mesh (`Uniform1DMesh`)
* **Discretization:** Second-order Central Finite Difference Method (`FDM1DDiscretizer`)
* **Surface Kinetics:** Nernstian equilibrium potential-dependent concentration ratio at x = 0
* **Solver Engine:** SciPy BDF time integration paired with triangular voltage sweep waveform
* **Validation:** Benchmark simulation peak current (i_p) against the analytical Randles-Sevcik equation

---

## Documentation & Guidelines

* For developer guidelines, commit standards, and release workflows, see **PROJECT_MANAGEMENT.md**.
* For historical version updates, consult **CHANGELOG.md**.

---

## License

Soft Potato is released under the **MIT License**.
