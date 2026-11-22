# Benchmarking & Validation

A numerical simulation is only trustworthy once it has been verified against known exact mathematics. Soft Potato includes an automated validation suite in `softpotato.analytical.benchmark` that compares numerical PDE solvers (`softpotato.solver`) against closed-form analytical solutions (`softpotato.analytical`).

---

## Error Norms (`BenchmarkMetrics`)

Given a numerical array $u^{\text{num}}$ and an analytical reference $u^{\text{exact}}$ over $N$ points, `compute_error_metrics` reports:

| Metric | Formula | What It Captures |
| :--- | :--- | :--- |
| **RMSE** | $\sqrt{\frac{1}{N}\sum_{i=1}^N (u_i^{\text{num}} - u_i^{\text{exact}})^2}$ | Overall root-mean-square deviation in physical units. |
| **Max Absolute Error ($L_\infty$)** | $\max_i \lvert u_i^{\text{num}} - u_i^{\text{exact}} \rvert$ | Worst-case pointwise error (often at $x = 0$ or sharp fronts). |
| **Relative $L_2$ Error** | $\frac{\lVert u^{\text{num}} - u^{\text{exact}} \rVert_2}{\lVert u^{\text{exact}} \rVert_2}$ | Scale-independent global accuracy across the entire profile or transient. |
| **MAE / MRE** | Mean absolute & mean relative error | Average pointwise accuracy with zero-division protection. |

---

## Order-of-Convergence Verification

For a well-posed discretization of order $p$, halving the spatial step $\Delta x$ or time step $\Delta t$ should reduce the error by a factor of $2^p$:

$$\text{Error}(h) \approx C \, h^p \implies \log(\text{Error}) \approx \log C + p \log h$$

Soft Potato automates this check via `estimate_convergence_order`, `verify_spatial_convergence`, and `verify_temporal_convergence`:

- **Spatial convergence**: All built-in finite-difference stencils should observe $p \approx 2$ ($\mathcal{O}(\Delta x^2)$) on uniform grids.
- **Temporal convergence**: `ImplicitFiniteDifference` (BTCS) should observe $p \approx 1$ ($\mathcal{O}(\Delta t)$), while `CrankNicolson` should observe $p \approx 2$ ($\mathcal{O}(\Delta t^2)$).

---

## Canonical Electrochemical Benchmarks

`softpotato.analytical.benchmark` ships with five canonical benchmark problems, executable individually via `run_benchmark()` or all together via `run_benchmark_suite()`:

| Benchmark Function / Case | Analytical Reference | What It Validates |
| :--- | :--- | :--- |
| `run_fourier_decay_benchmark` | 1D sine-mode decay $c(x,t) = \sin(\pi x / L)\exp(-D\pi^2 t / L^2)$ | Interior diffusion operator and pure spatial/temporal accuracy without boundary singularities. |
| `run_cottrell_benchmark` | Planar Cottrell profile $c(x,t) = c^*\operatorname{erf}\!\left(\frac{x}{2\sqrt{D t}}\right)$ & Cottrell current | Surface flux extraction, boundary-layer resolution, and step-edge handling. |
| `run_spherical_cottrell_benchmark` | Spherical Cottrell transient & radial profile | Radial $p = 2$ spherical Laplacian $\frac{\partial^2 c}{\partial r^2} + \frac{2}{r}\frac{\partial c}{\partial r}$ and steady-state convergence. |
| `run_mass_conservation_benchmark` | Constant total integral $\int c(x,t)\,dx = \text{const}$ | Zero-flux Neumann boundary conditions and discrete mass conservation. |
| `run_grid_efficiency_benchmark` | Planar Cottrell on `Uniform1DMesh` vs. `GeometricExpandingMesh` vs. `ExponentialMesh` | Accuracy vs. node count, runtime speedup, and memory scaling of stretched grids. |

---

## Where to Go Next

- **User Guide**: [Analytical Models](../../user-guide/analytical/models.md) · [Curve Fitting & Regression](../../user-guide/analytical/fitting.md) · [Diagnostics & Post-Processing](../../user-guide/postprocessing/diagnostics.md)
- **Tutorials**: [Chronoamperometry Quickstart](../../tutorials/start/ca_quickstart.ipynb)
- **Numerics**: [Spatial Grids & Meshes](grids.md) · [Time-Stepping & Solvers](solvers.md)
- **Benchmark Script**: [`modeller_benchmark.py`](../../scripts/modeller_benchmark.py)
