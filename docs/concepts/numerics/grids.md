# Spatial Grids & Meshes

To solve Fick's second law numerically, Soft Potato replaces the continuous spatial coordinate $x \in [x_{\min}, x_{\max}]$ with a discrete 1D mesh of $N$ nodes $x_0 < x_1 < \dots < x_{N-1}$. How those nodes are spaced determines both **numerical accuracy** and **computational speed**.

---

## Sizing the Domain: The Diffusion Layer

In semi-infinite planar diffusion, concentration perturbations spread outward from the electrode ($x = 0$) over a characteristic distance $\sim \sqrt{D t}$. From the exact error-function solution for a potential step, at $x = 6\sqrt{D t_{\max}}$:

$$\operatorname{erfc}\!\left(\frac{6\sqrt{D t_{\max}}}{2\sqrt{D t_{\max}}}\right) = \operatorname{erfc}(3) \approx 2.2 \times 10^{-5}$$

Setting the outer domain boundary to $\delta = 6\sqrt{D t_{\max}}$ guarantees that the bulk concentration remains undisturbed ($c(x_{\max}, t) \approx c^*$) to within $0.002\%$, without wasting grid points far out in the bulk solution.

Soft Potato automates this calculation via `create_diffusion_mesh(D, t_max, factor=6.0, kind=...)` (and `diffusion_layer_thickness`).

!!! note "When the Domain Is Fixed Physically"
    For a **Rotating Disk Electrode** (`model="nernst_layer"`), the outer boundary is instead set by the hydrodynamic Nernst diffusion layer $\delta_N = 1.610\,D^{1/3}\nu^{1/6}\omega^{-1/2}$ (or $k \delta_N$ for `"convective_diffusion"`). See [Mass Transport](../electrochemistry/transport.md).

---

## Boundary-Layer Resolution & Non-Uniform Stencils

Near $x = 0$, concentration gradients $\left.\frac{\partial c}{\partial x}\right|_{x=0}$ are steepest—especially right after a potential step ($t \to 0^+$) or inside a fast homogeneous reaction layer $\mu = \sqrt{D / k_c}$. Far from the electrode, concentration varies smoothly.

- A **uniform grid** wastes most of its nodes in the smooth bulk region in order to make $\Delta x$ small enough at the surface.
- A **stretched (expanding) grid** clusters nodes tightly near $x = 0$ (small $\Delta x_0$) and expands smoothly toward $x_{\max}$, achieving the same surface flux accuracy with **10×–50× fewer nodes**.

On any non-uniform grid with local spacings $h_{j-1} = x_j - x_{j-1}$ and $h_j = x_{j+1} - x_j$, Soft Potato discretizes the second derivative using the standard 3-point finite-difference stencil:

$$\left.\frac{\partial^2 c}{\partial x^2}\right|_{x_j} \approx \frac{2}{h_{j-1} + h_j} \left( \frac{c_{j+1} - c_j}{h_j} - \frac{c_j - c_{j-1}}{h_{j-1}} \right)$$

---

## Mesh Comparison: `Uniform1DMesh` vs. `ExponentialMesh` vs. `GeometricExpandingMesh`

| Mesh Class | `create_diffusion_mesh` `kind` | Node Spacing Law | Key Parameters | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **`Uniform1DMesh`** | `"uniform"` | Constant $\Delta x = \frac{x_{\max} - x_{\min}}{N - 1}$ | `n_points` or `dx` | Smooth interior profiles, Fourier decay benchmarks, and teaching/prototyping. |
| **`GeometricExpandingMesh`** | `"geometric"` *(default)* | $\Delta x_j = \Delta x_0 \, \gamma^j$ ($\gamma > 1$) | Any two of `n_points`, `gamma`, `dx_min`; `two_sided` | General electrochemistry (CV, CA, RDE, `EC`/`EC'` mechanisms). Predictable surface spacing $\Delta x_0$ and smooth geometric growth. |
| **`ExponentialMesh`** | `"exponential"` | Smooth coordinate mapping (`"exponential"`, `"tanh"`, `"sinh"`) | `n_points`, stretching `beta` ($\beta > 0$), `mapping`, `two_sided` | High-order smooth stretching across $[x_{\min}, x_{\max}]$ or symmetric two-electrode cells (`two_sided=True`). |

!!! warning "CFL Penalty on Fine Surface Grids"
    Clustered meshes have a very small $\Delta x_0$ at the electrode. Because explicit time-stepping requires $\Delta t \le \frac{\Delta x_{\min}^2}{2 D}$, **never pair a tightly clustered expanding mesh with an explicit solver**—use an implicit or adaptive solver (`"scipy_ivp"`, `"crank_nicolson"`, or `"implicit"`).

---

## Where to Go Next

- **User Guide**: [Voltammetry (LSV & CV)](../../user-guide/techniques/voltammetry.md) · [Step Techniques (CA & CP)](../../user-guide/techniques/step.md)
- **Numerics**: [Time-Stepping & Solvers](solvers.md) · [Benchmarking & Validation](benchmarking.md) (`run_grid_efficiency_benchmark`)
