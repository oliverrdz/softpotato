# Soft Potato

[![PyPI](https://img.shields.io/pypi/v/softpotato.svg)](https://pypi.org/project/softpotato/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w9NfLV2j8ZZjjcVhAiPcogVnKb65wqE2)
[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/DOI-10.26434%2Fchemrxiv--2022--4bs3w-orange.svg)](https://doi.org/10.26434/chemrxiv-2022-4bs3w)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

> [!NOTE]
> **Active Development (Milestone 3 in progress):**
> The `main` branch contains work-in-progress features for **Milestone 3** (v3.3.0: Reactions).
> The current stable release available on [PyPI](https://pypi.org/project/softpotato/) is **v3.2.0**.
> To test or use in-development features, install directly from GitHub:
> ```bash
> pip install git+https://github.com/oliverrdz/softpotato.git
> ```

**Full documentation**: [https://softpotato.xyz/](https://softpotato.xyz/) | **Try Live**: [Google Colab Starter](https://colab.research.google.com/drive/1w9NfLV2j8ZZjjcVhAiPcogVnKb65wqE2) | **Roadmap**: [ROADMAP.md](ROADMAP.md) | **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Installation

### Zero-Install Browser Trial (Google Colab)

Run Soft Potato directly in your browser with no local Python setup required:
- [**Launch Soft Potato v3 Starter in Google Colab**](https://colab.research.google.com/drive/1w9NfLV2j8ZZjjcVhAiPcogVnKb65wqE2)

### Stable Release (v3.2.0)

Soft Potato v3.2.0 is available on [PyPI](https://pypi.org/project/softpotato/) and can be installed via pip:

```bash
pip install softpotato
```

### Development Version (Milestone 3 in progress)

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

## Non-Uniform Spatial Discretization (`softpotato.grid`)

Concentration gradients in electrochemical experiments are confined to a thin diffusion boundary layer ($\delta \approx 6\sqrt{D t_{\max}}$) adjacent to the working electrode. Soft Potato's `grid` module provides uniform, geometrically expanding, and exponential/hyperbolic coordinate meshes that cluster nodes near active interfaces while coarsening into the bulk solution:

```python
from softpotato.grid import (
    create_diffusion_mesh,
    GeometricExpandingMesh,
    ExponentialMesh,
)
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver

# Automatically size a geometrically expanding mesh to delta = 6 * sqrt(D * t_max)
mesh = create_diffusion_mesh(
    D=1e-9,
    t_max=1.0,
    kind="geometric",
    n_points=60,
    gamma=1.08,
)

problem = DiffusionProblem(
    grid=mesh,
    diffusivity={"Ox": 1e-9},
    boundary_conditions={"Ox": (DirichletBC(0.0), DirichletBC(1.0))},
    initial_conditions={"Ox": 1.0},
)
result = get_solver("scipy_ivp").solve(problem, t_span=(0.0, 1.0))
```

## Electrode Coordinate Geometries (`softpotato.geometry`)

Soft Potato natively supports 1D Cartesian planar (`PlanarGeometry`), radial spherical/hemispherical (`SphericalGeometry`), and radial cylindrical (`CylindricalGeometry`) coordinate systems, with automatic Faradaic current integration ($I = n F A J$):

```python
from softpotato.geometry import SphericalGeometry
from softpotato.grid import GeometricExpandingMesh
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver

r0 = 10e-6  # 10 um spherical microelectrode radius
mesh = GeometricExpandingMesh(x_min=r0, x_max=r0 + 500e-6, n_points=80, gamma=1.06)

problem = DiffusionProblem(
    grid=mesh,
    diffusivity={"Ox": 1e-9},
    boundary_conditions={"Ox": (DirichletBC(0.0), DirichletBC(1.0))},
    initial_conditions={"Ox": 1.0},
    geometry=SphericalGeometry(r0=r0),
    n_electrons={"Ox": -1},
)
result = get_solver("scipy_ivp").solve(problem, t_span=(0.0, 2.0))
i_cathodic = result.currents["Ox"]  # Faradaic current in Amperes (IUPAC convention)
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

## Interactive Tutorials & Examples

Step-by-step Jupyter notebook tutorials are located in [`examples/`](examples/) and rendered in the [online documentation](https://softpotato.xyz/tutorials/):

| Tutorial | Description | Notebook |
| :--- | :--- | :--- |
| **Cottrell Chronoamperometry** | 1D potential-step diffusion, error metrics, and parameter fitting | [`examples/cottrell.ipynb`](examples/cottrell.ipynb) |
| **Cyclic Voltammetry** | Reversible Nernstian CV, concentration profiles, and Randles–Ševčík diagnostics | [`examples/cyclic_voltammetry.ipynb`](examples/cyclic_voltammetry.ipynb) |
| **Cottrell Across Geometries** | Planar, spherical, and cylindrical electrode transients and steady-state limits | [`examples/cottrell_geometries.ipynb`](examples/cottrell_geometries.ipynb) |
| **Non-Uniform Expanding Grids** | Geometric and exponential mesh clustering, convergence, and 5–20× solver speedups | [`examples/expanding_grids.ipynb`](examples/expanding_grids.ipynb) |

## Development Roadmap & Milestones

Soft Potato follows a staged milestone development roadmap for the 3.x series, maintaining strict backwards compatibility across releases:

| Milestone | Target | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Current Baseline** | `v3.0.0` | 1D multi-species diffusion solvers (`scipy_ivp`, Crank–Nicolson, BTCS, FTCS) | Released |
| **Milestone 1: Analytical Equations** | `v3.1.0` | Vectorized analytical models (`step`, `voltammetry`, `hydrodynamics`, `microelectrodes`, `secm`, `kinetics`), automated solver benchmarking (`benchmark`), and parameter fitting (`fitting`) | Released |
| **Milestone 2: Grids** | `v3.2.0` | Non-uniform, geometric, and exponentially expanding spatial discretization meshes (`grid`) and planar/spherical/cylindrical coordinate geometries (`geometry`) | Released |
| **Milestone 3: Reactions** | `v3.3.0` | Homogeneous chemical mechanisms (EC, EC', catalytic) and heterogeneous charge-transfer kinetics (`reactions`) | In Development |
| **Milestone 4: Techniques** | `v3.4.0` | Standard electrochemical experimental waveform generators (CV, CA, DPV, SWV, RDE) and high-level simulation pipelines (`techniques`) | Planned |

For detailed architectural plans and deliverable specifications, see the full [Roadmap](ROADMAP.md).

## Citation

If you use Soft Potato in your research or project, please cite the ChemRxiv preprint:

> Rodríguez, O. (2022). Soft Potato: an open source electrochemical simulator and toolkit. *ChemRxiv*. [doi:10.26434/chemrxiv-2022-4bs3w](https://doi.org/10.26434/chemrxiv-2022-4bs3w)

```bibtex
@article{rodriguez2022softpotato,
  author    = {Rodr{\'i}guez, Oliver},
  title     = {Soft Potato: an open source electrochemical simulator and toolkit},
  journal   = {ChemRxiv},
  year      = {2022},
  doi       = {10.26434/chemrxiv-2022-4bs3w},
  url       = {https://chemrxiv.org/doi/full/10.26434/chemrxiv-2022-4bs3w}
}
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

