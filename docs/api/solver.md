# Solver Module (`softpotato.solver`)

The `softpotato.solver` subpackage provides numerical solvers for 1D multi-species chemical diffusion and reaction-diffusion problems across planar Cartesian and curvilinear (spherical, cylindrical) geometries.

## Architecture and Spatial Operator Delegation

Solvers follow the Single Responsibility Principle:
- **Time Stepping & Linear Algebra**: Managed exclusively by solver classes (`ScipyIVPSolver`, `CrankNicolson`, `ImplicitFiniteDifference`, `ExplicitFiniteDifference`).
- **Spatial Operators & Coordinate Metrics**: Discretization stencils ($\nabla^2 c$), boundary conditions, interfacial fluxes, and Faradaic current calculations are delegated directly to [`Geometry`][softpotato.geometry.Geometry] abstractions (`PlanarGeometry`, `SphericalGeometry`, `CylindricalGeometry`).

## Solvers and Supported Options

| Solver Class & Aliases | Supported Configuration Options | Description & Defaults |
| :--- | :--- | :--- |
| [`ExplicitFiniteDifference`][softpotato.solver.ExplicitFiniteDifference] (`'explicit'`, `'ftcs'`) | `dt` (float, optional) | Explicit Forward-Time Central-Space. If `dt=None`, automatically defaults to $0.45 \times \text{CFL limit}$ ($\Delta t \le \Delta x^2 / 2D_{\max}$). |
| [`ImplicitFiniteDifference`][softpotato.solver.ImplicitFiniteDifference] (`'implicit'`, `'btcs'`) | `dt` (float, optional) | Backward-Time Central-Space implicit scheme using banded tridiagonal matrix inversion. If `dt=None`, defaults to 100 uniform steps: $(t_{\text{end}} - t_{\text{start}}) / 100$. |
| [`CrankNicolson`][softpotato.solver.CrankNicolson] (`'crank_nicolson'`, `'crank-nicolson'`) | `dt` (float, optional) | Second-order in time and space Crank–Nicolson implicit scheme using banded tridiagonal solve. If `dt=None`, defaults to 100 uniform steps: $(t_{\text{end}} - t_{\text{start}}) / 100$. |
| [`ScipyIVPSolver`][softpotato.solver.ScipyIVPSolver] (`'scipy_ivp'`, `'solve_ivp'`) | `method` (str, default `'Radau'`), `rtol` (float, default `1e-6`), `atol` (float, default `1e-8`) | Method of Lines ODE integration using `scipy.integrate.solve_ivp`. Supports stiff (`'Radau'`, `'BDF'`) and non-stiff (`'RK45'`) integrators. |

## Examples

### 1. Planar 1D Diffusion
```python
import numpy as np
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver

# Linear domain x in [0, 1]
x = np.linspace(0.0, 1.0, 50)
problem = DiffusionProblem(
    grid=x,
    diffusivity={"c": 1e-5},
    boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(1.0))},
    initial_conditions={"c": 1.0},
    geometry="planar",  # default
)

solver = get_solver("crank_nicolson", dt=0.01)
result = solver.solve(problem, t_span=(0.0, 1.0))
print(result["c"].shape)       # (N_t, N_x) concentration profile
print(result.fluxes["c"].shape) # (N_t,) interfacial flux
print(result.currents["c"].shape) # (N_t,) Faradaic current in Amperes
```

### 2. Spherical Microelectrode Step Response
```python
import numpy as np
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver
from softpotato.geometry import SphericalGeometry

r0 = 25e-6  # 25 um radius
r_grid = np.geomspace(r0, r0 + 300e-6, 150)
problem = DiffusionProblem(
    grid=r_grid,
    diffusivity={"c": 1e-9},
    boundary_conditions={"c": (DirichletBC(0.0), DirichletBC(1.0))},
    initial_conditions={"c": 1.0},
    geometry=SphericalGeometry(r0=r0),
    n_electrons=1,
)

solver = get_solver("scipy_ivp")
result = solver.solve(problem, t_span=(0.0, 2.0))
# result.currents['c'] contains exact spherical Cottrell chronoamperometric transient
```

## Module Contents

::: softpotato.solver

