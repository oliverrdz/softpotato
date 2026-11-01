# Grid Module (`softpotato.grid`)

The `softpotato.grid` subpackage provides 1D spatial discretization meshes and grid generators for numerical PDE simulations, including uniform, geometrically expanding, and exponential/conformal-mapping meshes, as well as automatic domain sizing from the semi-infinite diffusion layer thickness $\delta = \text{factor} \cdot \sqrt{D \, t_{\max}}$.

## Why Non-Uniform Spatial Discretization?

In electrochemical simulations, concentration perturbations and diffusive fluxes are concentrated within a narrow **diffusion boundary layer** adjacent to the working electrode surface ($x = x_{\min}$), while the bulk solution ($\nabla^2 c \approx 0$) experiences negligible gradients.

- **Uniform grids** ([`Uniform1DMesh`][softpotato.grid.Uniform1DMesh]) distribute spatial nodes equally across the entire domain, requiring hundreds or thousands of nodes to resolve steep interfacial gradients $\left.\partial c / \partial x\right|_{x_{\min}}$ while oversampling the quiescent bulk.
- **Expanding and stretched grids** ([`GeometricExpandingMesh`][softpotato.grid.GeometricExpandingMesh] and [`ExponentialMesh`][softpotato.grid.ExponentialMesh]) cluster fine intervals ($\Delta x_0 \ll \delta$) near active boundaries and smoothly coarsen into the bulk solution, achieving **5–20× faster simulation runtimes** and **80–90% fewer spatial nodes** at equal or superior flux accuracy while preserving second-order spatial convergence ($\mathcal{O}(\Delta x^2)$).

---

## Mesh Classes and Generators

| Class / Function | Spacing Profile | Key Parameters | Description |
| :--- | :--- | :--- | :--- |
| [`Mesh1D`][softpotato.grid.Mesh1D] | Abstract | `nodes`, `dx` | Abstract base class implementing the 1D mesh interface and NumPy array protocol (`__array__`, `__array_ufunc__`, slicing, arithmetic). |
| [`Uniform1DMesh`][softpotato.grid.Uniform1DMesh] | $\Delta x_j = \Delta x$ | `x_min`, `x_max`, `n_points` or `dx` | Equidistant 1D spatial mesh with constant node spacing $\Delta x = (x_{\max} - x_{\min}) / (N - 1)$. |
| [`GeometricExpandingMesh`][softpotato.grid.GeometricExpandingMesh] | $\Delta x_j = \Delta x_0 \gamma^j$ | `x_min`, `x_max`, any two of (`n_points`, `gamma`, `dx_min`), `two_sided` | Non-uniform mesh with constant interval expansion ratio $\gamma > 1$. |
| [`ExponentialMesh`][softpotato.grid.ExponentialMesh] | $x(\xi)$ via `exponential`, `tanh`, or `sinh` | `x_min`, `x_max`, `n_points`, `beta`, `two_sided`, `mapping` | Smooth non-uniform mesh using exponential or hyperbolic coordinate stretching of uniform $\xi \in [0, 1]$. |
| [`create_diffusion_mesh`][softpotato.grid.create_diffusion_mesh] | `'geometric'`, `'exponential'`, or `'uniform'` | `D`, `t_max`, `kind`, `factor` | High-level factory constructing a 1D mesh automatically sized to $\delta = \text{factor} \cdot \sqrt{D \, t_{\max}}$. |
| [`diffusion_layer_thickness`][softpotato.grid.diffusion_layer_thickness] | Scalar $\delta$ | `D`, `t_max`, `factor` | Estimates the characteristic semi-infinite diffusion layer thickness $\delta = \text{factor} \cdot \sqrt{D \, t_{\max}}$. |

---

## Mathematical Formulations

### 1. Physics-Informed Domain Sizing
For semi-infinite diffusion over duration $t_{\max}$ with diffusivity $D$, the characteristic penetration thickness is:

$$\delta = \text{factor} \cdot \sqrt{D \, t_{\max}}, \quad x_{\max} = x_{\min} + \delta$$

At the default `factor=6.0`, the complementary error function decays to $\operatorname{erfc}(6 / 2) \approx 2.2 \times 10^{-5}$, ensuring outer boundary perturbations remain negligible.

### 2. Geometrically Expanding Meshes (`GeometricExpandingMesh`)
Consecutive cell intervals expand by a constant geometric factor $\gamma > 1$:

$$\Delta x_j = \Delta x_0 \cdot \gamma^j \quad (j = 0, 1, \dots, N - 2), \quad x_{j+1} = x_j + \Delta x_j$$

where the total domain length satisfies the geometric series:

$$L = x_{\max} - x_{\min} = \Delta x_0 \sum_{j=0}^{N-2} \gamma^j = \Delta x_0 \frac{\gamma^{N-1} - 1}{\gamma - 1}$$

`GeometricExpandingMesh` supports **three flexible parameterizations** by specifying any two of `n_points` ($N$), `gamma` ($\gamma$), and `dx_min` ($\Delta x_0$):
1. **`(n_points, gamma)`**: Solves analytically for the initial boundary step $\Delta x_0$.
2. **`(dx_min, gamma)`**: Computes the required node count $N$ so the boundary step is at most `dx_min`.
3. **`(n_points, dx_min)`**: Solves numerically via Brent's root-finding method (`scipy.optimize.brentq`) for the exact expansion ratio $\gamma > 1$.

### 3. Exponential & Conformal-Mapping Meshes (`ExponentialMesh`)
Maps uniform computational coordinates $\xi_j = j / (N - 1) \in [0, 1]$ to physical coordinates $x(\xi) \in [x_{\min}, x_{\max}]$ parameterized by a dimensionless stretching factor $\beta > 0$:

- **Exponential mapping** (`mapping="exponential"`, default):
  $$x(\xi) = x_{\min} + (x_{\max} - x_{\min}) \frac{\exp(\beta \xi) - 1}{\exp(\beta) - 1}$$
- **Hyperbolic tangent mapping** (`mapping="tanh"` or `ExponentialMesh.from_tanh`):
  $$x(\xi) = x_{\min} + (x_{\max} - x_{\min}) \left(1 - \frac{\tanh(\beta (1 - \xi))}{\tanh(\beta)}\right)$$
- **Hyperbolic sine mapping** (`mapping="sinh"` or `ExponentialMesh.from_sinh`):
  $$x(\xi) = x_{\min} + (x_{\max} - x_{\min}) \frac{\sinh(\beta \xi)}{\sinh(\beta)}$$

### 4. Two-Sided Symmetric Clustering (`two_sided=True`)
Both [`GeometricExpandingMesh`][softpotato.grid.GeometricExpandingMesh] and [`ExponentialMesh`][softpotato.grid.ExponentialMesh] support `two_sided=True`, clustering fine spatial intervals symmetrically at both $x_{\min}$ and $x_{\max}$ while coarsening toward the domain midpoint $x_{\text{mid}} = \frac{1}{2}(x_{\min} + x_{\max})$. This is ideal for thin-layer electrochemical cells, SECM tip–substrate gaps (generation–collection / feedback), and dual-working-electrode (`electrodes="both"`) simulations.

---

## User Guide & Examples

### 1. Automatic Diffusion-Layer Mesh Generation
Use [`create_diffusion_mesh`][softpotato.grid.create_diffusion_mesh] or the `.from_diffusion_layer()` classmethods to automatically size the domain to $\delta = 6\sqrt{D t_{\max}}$:

```python
from softpotato.grid import (
    ExponentialMesh,
    GeometricExpandingMesh,
    Uniform1DMesh,
    create_diffusion_mesh,
)

D = 1e-9  # m^2/s
t_max = 1.0  # s

# High-level factory (defaults to kind="geometric", factor=6.0)
mesh_geom = create_diffusion_mesh(
    D=D, t_max=t_max, kind="geometric", n_points=60, gamma=1.08
)

# Or construct directly via classmethods
mesh_exp = ExponentialMesh.from_diffusion_layer(
    D=D, t_max=t_max, n_points=60, beta=3.5, mapping="sinh"
)
mesh_uni = Uniform1DMesh.from_diffusion_layer(D=D, t_max=t_max, n_points=300)

print(f"Domain bounds: {mesh_geom.bounds}")
print(f"Geometric dx_min = {mesh_geom.dx_min:.3e} m, dx_max = {mesh_geom.dx_max:.3e} m")
```

### 2. Three Parameterizations of `GeometricExpandingMesh`
Specify any two of `n_points`, `gamma`, and `dx_min`:

```python
from softpotato.grid import GeometricExpandingMesh

# 1. Specify (n_points, gamma) -> computes dx_min
m1 = GeometricExpandingMesh(0.0, 1e-3, n_points=51, gamma=1.08)

# 2. Specify (dx_min, gamma) -> determines required n_points
m2 = GeometricExpandingMesh(0.0, 1e-3, dx_min=1e-6, gamma=1.08)

# 3. Specify (n_points, dx_min) -> solves for gamma > 1
m3 = GeometricExpandingMesh(0.0, 1e-3, n_points=51, dx_min=1e-6)

# Two-sided symmetric mesh for thin-layer / SECM cells
m_thin = GeometricExpandingMesh(0.0, 50e-6, n_points=61, gamma=1.06, two_sided=True)
```

### 3. Coupling Expanding Grids with `DiffusionProblem` and Solvers
All [`Mesh1D`][softpotato.grid.Mesh1D] subclasses implement the NumPy array protocol and integrate seamlessly with [`DiffusionProblem`][softpotato.solver.DiffusionProblem] and all Soft Potato solvers (`scipy_ivp`, `crank_nicolson`, `implicit`, `explicit`) across planar, spherical, and cylindrical geometries:

```python
from softpotato.grid import create_diffusion_mesh
from softpotato.solver import DiffusionProblem, DirichletBC, get_solver

D = 1e-9
t_max = 1.0
mesh = create_diffusion_mesh(
    D=D, t_max=t_max, kind="geometric", n_points=60, gamma=1.08
)

problem = DiffusionProblem(
    grid=mesh,
    diffusivity={"Ox": D},
    boundary_conditions={"Ox": (DirichletBC(0.0), DirichletBC(1.0))},
    initial_conditions={"Ox": 1.0},
    geometry="planar",
    n_electrons=1,
)

solver = get_solver("scipy_ivp")
result = solver.solve(problem, t_span=(0.0, t_max))
print(result.grid is mesh)  # True: preserves Mesh1D reference
print(result.currents["Ox"].shape)  # (100,) Faradaic current transient in Amperes
```

---

## Module Contents

::: softpotato.grid


