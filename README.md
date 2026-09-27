# Soft Potato

[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/softpotato/badge/?version=latest)](https://softpotato.readthedocs.io)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

> [!WARNING]
> **Active Rewrite in Progress:** Soft Potato is currently undergoing a complete rewrite (`v3.0.0a1`) and is **incomplete**. The API is experimental, under rapid development, and subject to breaking changes. It is not yet ready for production use.

**Full documentation**: [https://softpotato.readthedocs.io](https://softpotato.readthedocs.io)

## Tutorials & Examples

Interactive Jupyter notebooks are available in the [`examples/`](examples/) directory:

| Tutorial | Description |
| :--- | :--- |
| [Cottrell Chronoamperometry with `solve_ivp`](examples/cottrell_solve_ivp_tutorial.ipynb) | Simulates the Cottrell potential-step experiment using Soft Potato's adaptive `solve_ivp` solver and benchmarks against the Cottrell equation. |
| [Cyclic Voltammetry with Nernst Kinetics](examples/cyclic_voltammetry_solve_ivp_tutorial.ipynb) | Simulates reversible cyclic voltammetry using `solve_ivp`, visualizes spatio-temporal concentration profiles, validates peak currents against the Randles–Ševčík equation across scan rates, and evaluates diagnostic reversibility criteria. |
| [Cyclic Voltammetry with Butler–Volmer Kinetics](examples/cyclic_voltammetry_butler_volmer_tutorial.ipynb) | Simulates cyclic voltammetry under Butler–Volmer quasi-reversible kinetics with unequal diffusion coefficients ($D_{\text{Red}} \neq D_{\text{Ox}}$) using second-order ghost-node flux discretization and benchmarks against Nicholson theory. |
| [Rotating Disk Electrode (RDE) Voltammetry with Levich Analysis](examples/rde_cyclic_voltammetry_solve_ivp_tutorial.ipynb) | Simulates RDE cyclic voltammetry with Nernst kinetics using the Nernst stagnant diffusion layer approximation ($\delta = 1.61 D^{1/3} \nu^{1/6} \omega^{-1/2}$). Demonstrates the transition from transient peaks to steady-state waves across scan rates and validates limiting currents against the Levich equation. |
| [RDE Voltammetry with Butler–Volmer Kinetics & Koutecký–Levich Analysis](examples/rde_butler_volmer_koutecky_levich_tutorial.ipynb) | Simulates RDE cyclic voltammetry for an oxidation under IUPAC convention with Butler–Volmer kinetics using the Nernst diffusion layer approximation ($\delta = 1.61 D^{1/3} \nu^{1/6} \omega^{-1/2}$). Demonstrates kinetic shifts via normalized voltammograms ($I/I_L$), validates limiting currents against the Levich equation, and extracts $k_0$ and $\alpha$ using Koutecký–Levich and Tafel analysis. |

## Installation

Clone the repository and install in development / editable mode:

```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
pip install -e ".[dev,docs]"
```

## Quick Start: Solving 1D Diffusion with `scipy_ivp`

Soft Potato makes it straightforward to simulate chemical diffusion:

$$\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2}$$

Here is a minimal example setting boundary conditions to $0$ at the left boundary ($x=0$) and $1.0$ at the right boundary ($x=1.0$):

```python
import numpy as np
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver

# 1. Define grid and diffusion problem (c = 0 at left, c = 1.0 at right)
grid = np.linspace(0.0, 1.0, 50)
problem = DiffusionProblem(
    grid=grid,
    diffusivity={"c": 1e-4},
    boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(1.0))},
    initial_conditions={"c": 1.0},
)

# 2. Solve using SciPy's Method of Lines solver
solver = get_solver("scipy_ivp")
result = solver.solve(problem, t_span=(0.0, 1.0))

# 3. Save the concentration profile to a variable
c_profile = result["c"]  # 2D array of shape (N_times, N_points)
print(f"Simulation success: {result.success}")
print(f"Concentration profile shape: {c_profile.shape}")
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
