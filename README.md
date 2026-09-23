# Soft Potato

[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/softpotato/badge/?version=latest)](https://softpotato.readthedocs.io)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Soft Potato** is an open-source electrochemical simulation and analysis toolkit in Python.

- **Documentation**: [https://softpotato.readthedocs.io](https://softpotato.readthedocs.io)
- **Interactive Tutorials & Examples**: [`examples/`](examples/) (including the [1D Diffusion Solvers Comparison](examples/1d_diffusion_comparison.ipynb))

> [!WARNING]
> **Active Rewrite in Progress:** Soft Potato is currently undergoing a complete rewrite (`v3.0.0a1`) and is **incomplete**. The API is experimental, under rapid development, and subject to breaking changes. It is not yet ready for production use.

## Installation

Clone the repository and install in development / editable mode:

```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
pip install -e ".[dev,docs]"
```

## Quick Start: Solving Chemical Diffusion with `scipy_ivp`

Soft Potato provides a unified `solver` module for 1D multi-species chemical diffusion and reaction-diffusion problems:

$$\frac{\partial c_i}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} + R_i(\mathbf{c}, x, t)$$

Here is a complete, working example simulating diffusion of two species using the Method of Lines and SciPy's stiff ODE solver (`scipy_ivp`):

```python
import numpy as np
import softpotato as sp
from softpotato.solver import (
    DiffusionProblem,
    DirichletBC,
    NeumannBC,
    get_solver,
)

# 1. Discretize spatial domain (e.g. electrode surface at x=0, bulk at x=1.0 cm)
x = np.linspace(0.0, 1.0, 100)

# 2. Formulate the diffusion problem for species 'A' and 'B'
problem = DiffusionProblem(
    grid=x,
    diffusivity={"A": 1e-5, "B": 2e-5},               # Diffusion coefficients (cm²/s)
    initial_conditions={"A": 1.0, "B": 0.0},          # Initial concentrations (mol/cm³)
    boundary_conditions={
        # Species A: depleted at electrode (x=0), bulk concentration maintained at x=1
        "A": (DirichletBC(0.0), DirichletBC(1.0)),
        # Species B: zero flux at electrode (x=0), zero in bulk (x=1)
        "B": (NeumannBC(0.0), DirichletBC(0.0)),
    },
)

# 3. Instantiate the SciPy IVP solver (Method of Lines, defaults to stiff 'Radau' method)
solver = get_solver("scipy_ivp", method="Radau")

# 4. Integrate across time span (t_start=0 to t_end=100 s)
t_eval = np.linspace(0.0, 100.0, 11)
result = solver.solve(problem, t_span=(0.0, 100.0), t_eval=t_eval)

# 5. Inspect results
print(f"Success: {result.success}")
print(f"Simulated species: {result.species}")

# Access concentration profile array of shape (N_times, N_points):
c_A = result["A"]            # Or result.concentrations["A"]
c_B = result["B"]
print(f"Species A profile shape: {c_A.shape}")

# Access surface flux at x=0 (e.g. for Faraday's law / voltammetry current calculations):
flux_A = result.fluxes["A"]
print(f"Surface flux of A at final time: {flux_A[-1]:.4e} mol/(cm²·s)")
```

### Switching Numerical Solvers

All solvers share the identical `BaseSolver` interface. You can switch algorithms by passing a different name to `get_solver(...)`:

```python
# 2nd-order Crank-Nicolson finite difference (unconditionally stable)
solver = get_solver("crank_nicolson", dt=0.5)

# Fully implicit Backward-Time Central-Space (unconditionally stable)
solver = get_solver("implicit", dt=0.5)

# Explicit Forward-Time Central-Space (includes CFL stability check)
solver = get_solver("explicit", dt=1e-3)
```

### Creating User-Made Solvers

Custom solvers can be implemented and automatically registered by subclassing `BaseSolver` and specifying `name`:

```python
from softpotato.solver import BaseSolver, SolverResult, get_solver

class SpectralDiffusionSolver(BaseSolver, name="spectral"):
    def _run_solver(self, problem, t_span, t_eval, **kwargs):
        # Implement custom algorithm here...
        return SolverResult(t=..., x=problem.grid, concentrations=...)

# Once defined, it is automatically available via the solver factory:
custom_solver = get_solver("spectral")
```

Alternatively, use the `@register_solver("name")` decorator.

## Tutorials & Examples

Interactive Jupyter notebooks are available in the [`examples/`](examples/) directory:
- [1D Chemical Diffusion & Solvers Comparison](examples/1d_diffusion_comparison.ipynb): Step-by-step tutorial comparing `scipy_ivp`, `crank_nicolson`, `implicit`, and `explicit` against exact analytical solutions, including runtime benchmarks and convergence order analysis.

## Testing

Run the test suite using `pytest`:

```bash
pytest
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
