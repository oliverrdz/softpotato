# Soft Potato

[![PyPI](https://img.shields.io/pypi/v/softpotato.svg)](https://pypi.org/project/softpotato/)
[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/softpotato/badge/?version=latest)](https://softpotato.readthedocs.io)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

> [!NOTE]
> **Active Development (Milestone 2 in progress):**
> The `main` branch contains work-in-progress features for **Milestone 2** (v3.2.0: Grids).
> The current stable release available on [PyPI](https://pypi.org/project/softpotato/) is **v3.1.0**.
> To test or use in-development features, install directly from GitHub:
> ```bash
> pip install git+https://github.com/oliverrdz/softpotato.git
> ```

**Full documentation**: [https://softpotato.readthedocs.io](https://softpotato.readthedocs.io) | **Roadmap**: [ROADMAP.md](ROADMAP.md) | **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Installation

### Stable Release (v3.1.0)

Soft Potato v3.1.0 is available on [PyPI](https://pypi.org/project/softpotato/) and can be installed via pip:

```bash
pip install softpotato
```

### Development Version (Milestone 2 in progress)

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

## Development Roadmap & Milestones

Soft Potato follows a staged milestone development roadmap for the 3.x series, maintaining strict backwards compatibility across releases:

| Milestone | Target | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Current Baseline** | `v3.0.0` | 1D multi-species diffusion solvers (`scipy_ivp`, Crank–Nicolson, BTCS, FTCS) | Released |
| **Milestone 1: Analytical Equations** | `v3.1.0` | Vectorized analytical models (`step`, `voltammetry`, `hydrodynamics`, `microelectrodes`, `secm`, `kinetics`), automated solver benchmarking (`benchmark`), and parameter fitting (`fitting`) | Released |
| **Milestone 2: Grids** | `v3.2.0` | Non-uniform, geometric, and exponentially expanding spatial discretization meshes (`grid`) | In Development |
| **Milestone 3: Reactions** | `v3.3.0` | Homogeneous chemical mechanisms (EC, EC', catalytic) and heterogeneous charge-transfer kinetics (`reactions`) | Planned |
| **Milestone 4: Techniques** | `v3.4.0` | Standard electrochemical experimental waveform generators (CV, CA, DPV, SWV, RDE) and high-level simulation pipelines (`techniques`) | Planned |

For detailed architectural plans and deliverable specifications, see the full [Roadmap](ROADMAP.md).

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

