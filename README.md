# Soft Potato

[![PyPI](https://img.shields.io/pypi/v/softpotato.svg)](https://pypi.org/project/softpotato/)
[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/softpotato/badge/?version=latest)](https://softpotato.readthedocs.io)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

> [!NOTE]
> **Active Development (Milestone 1 in progress):**
> The `main` branch contains work-in-progress features for **Milestone 1** (v3.1.0).
> The current stable release available on [PyPI](https://pypi.org/project/softpotato/) is **v3.0.0**.
> To test or use in-development features, install directly from GitHub:
> ```bash
> pip install git+https://github.com/oliverrdz/softpotato.git
> ```

**Full documentation**: [https://softpotato.readthedocs.io](https://softpotato.readthedocs.io) | **Roadmap**: [ROADMAP.md](ROADMAP.md) | **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Tutorials & Examples

Interactive Jupyter notebooks are available in the [`examples/`](examples/) directory:

| Tutorial | Description |
| :--- | :--- |
| [Cottrell Chronoamperometry with `solve_ivp`](examples/cottrell_solve_ivp_tutorial.ipynb) | Simulates the Cottrell potential-step experiment using Soft Potato's adaptive `solve_ivp` solver and benchmarks against the Cottrell equation. |
| [Cyclic Voltammetry with Nernst Kinetics](examples/cyclic_voltammetry_solve_ivp_tutorial.ipynb) | Simulates reversible cyclic voltammetry using `solve_ivp`, visualizes spatio-temporal concentration profiles, validates peak currents against the Randles–Ševčík equation across scan rates, and evaluates diagnostic reversibility criteria. |
| [Cyclic Voltammetry with Butler–Volmer Kinetics](examples/cyclic_voltammetry_butler_volmer_tutorial.ipynb) | Simulates cyclic voltammetry under Butler–Volmer quasi-reversible kinetics with unequal diffusion coefficients ($D_{\text{Red}} \neq D_{\text{Ox}}$) using second-order ghost-node flux discretization and benchmarks against Nicholson theory. |
| [Rotating Disk Electrode (RDE) Voltammetry with Levich Analysis](examples/rde_cyclic_voltammetry_solve_ivp_tutorial.ipynb) | Simulates RDE cyclic voltammetry with Nernst kinetics using the stagnant diffusion layer approximation ($\delta = 1.61 D^{1/3} \nu^{1/6} \omega^{-1/2}$). Demonstrates the transition from transient peaks to steady-state waves across scan rates and validates limiting currents against the Levich equation. |
| [RDE Voltammetry with Butler–Volmer Kinetics & Koutecký–Levich Analysis](examples/rde_butler_volmer_koutecky_levich_tutorial.ipynb) | Simulates RDE cyclic voltammetry for an oxidation with Butler–Volmer kinetics using the stagnant diffusion layer approximation ($\delta = 1.61 D^{1/3} \nu^{1/6} \omega^{-1/2}$). Extracts $k_0$ and $\alpha$ using Koutecký–Levich and Tafel analysis. |

## Installation

### Stable Release (v3.0.0)

Soft Potato v3.0.0 is available on [PyPI](https://pypi.org/project/softpotato/) and can be installed via pip:

```bash
pip install softpotato
```

### Development Version (Milestone 1 in progress)

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/oliverrdz/softpotato.git
```

Or clone the repository and install in development mode:

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
c_flux = result.fluxes["c"]  # The current can be calculated with i = n*F*A*c_flux
```

## Analytical Equations & Benchmarking (`softpotato.analytical`)

Soft Potato provides exact closed-form analytical solutions and empirical models for benchmarking numerical solvers and rapid parameter estimation:

```python
import numpy as np
import softpotato as sp

# 1. Planar Cottrell chronoamperometry current transient
t = np.linspace(0.1, 5.0, 50)
i_transient = sp.cottrell(t, n=1, D=1e-5, c_bulk=1e-3, area=1e-4)

# 2. Spherical electrode transient and steady-state limiting current
i_sphere = sp.cottrell_spherical(t, r=1e-5, n=1, D=1e-5, c_bulk=1e-3)
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

