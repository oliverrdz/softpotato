# Grid Module (`softpotato.grid`)

The `softpotato.grid` subpackage provides 1D spatial discretization meshes and grid generators for numerical PDE simulations, including uniform, geometrically expanding, and exponential/conformal-mapping meshes, as well as automatic domain sizing from the semi-infinite diffusion layer thickness $\delta = \text{factor} \cdot \sqrt{D \, t_{\max}}$.

## Mesh Classes and Generators

| Class / Function | Spacing Profile | Key Parameters | Description |
| :--- | :--- | :--- | :--- |
| [`Mesh1D`][softpotato.grid.Mesh1D] | Abstract | `nodes`, `dx` | Abstract base class implementing the 1D mesh interface and NumPy array protocol. |
| [`Uniform1DMesh`][softpotato.grid.Uniform1DMesh] | $\Delta x_j = \Delta x$ | `x_min`, `x_max`, `n_points` or `dx` | Equidistant 1D spatial mesh with constant node spacing. |
| [`GeometricExpandingMesh`][softpotato.grid.GeometricExpandingMesh] | $\Delta x_j = \Delta x_0 \gamma^j$ | `x_min`, `x_max`, any two of (`n_points`, `gamma`, `dx_min`), `two_sided` | Non-uniform mesh with constant expansion ratio $\gamma > 1$. |
| [`ExponentialMesh`][softpotato.grid.ExponentialMesh] | $x(\xi)$ via `exponential`, `tanh`, or `sinh` | `x_min`, `x_max`, `n_points`, `beta`, `two_sided`, `mapping` | Smooth non-uniform mesh using exponential or hyperbolic coordinate stretching. |
| [`create_diffusion_mesh`][softpotato.grid.create_diffusion_mesh] | `'geometric'`, `'exponential'`, or `'uniform'` | `D`, `t_max`, `kind`, `factor` | High-level factory constructing a mesh sized to $\delta = \text{factor} \cdot \sqrt{D \, t_{\max}}$. |
| [`diffusion_layer_thickness`][softpotato.grid.diffusion_layer_thickness] | Scalar $\delta$ | `D`, `t_max`, `factor` | Estimates the characteristic semi-infinite diffusion layer thickness $\delta$. |

## Module Contents

::: softpotato.grid

