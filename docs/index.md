---
hide:
  - navigation
  - toc
---

# Soft Potato

**An open-source electrochemical simulation and analysis toolkit in Python — plus tutorials on scientific computing, Python, and Linux.**

[Get Started](installation.md){ .md-button .md-button--primary }
[Explore API Reference](api/index.md){ .md-button }
[Read the Blog](blog/index.md){ .md-button }

---

<div class="grid cards" markdown>

-   :material-flash-outline: **Analytical Electrochemistry**

    ---

    Closed-form and asymptotic equations for transient potential/current steps (Cottrell, Anson, Sand), hydrodynamics (Levich, Koutecký–Levich, RRDE), interfacial kinetics (Nernst, Butler–Volmer, Tafel), voltammetry, ultramicroelectrodes, and SECM.

    [:octicons-arrow-right-24: Analytical Module](api/analytical/index.md)

-   :material-matrix: **Numerical PDE Solvers**

    ---

    1D multi-species reaction-diffusion solvers including stiff Method of Lines ODE integration (`scipy_ivp` Radau/BDF), Crank–Nicolson, and implicit/explicit finite-difference engines.

    [:octicons-arrow-right-24: Solver Module](api/solver.md)

-   :material-chart-scatter-plot: **Parameter Fitting & Benchmarking**

    ---

    Extract diffusion coefficients ($D$), rate constants ($k^0$), transfer coefficients ($\alpha$), and SECM parameters ($R_g$, $d_{\text{offset}}$) with full confidence intervals, or validate numerical solvers against exact analytical solutions.

    [:octicons-arrow-right-24: Fitting & Benchmarks](api/analytical/fitting.md)

-   :material-school-outline: **Tutorials: Electrochemistry, Python & Linux**

    ---

    Interactive Jupyter notebooks for electrochemical simulations alongside practical tutorials on basic programming in Python and working effectively in Linux.

    [:octicons-arrow-right-24: Browse Tutorials](tutorials/index.md)

</div>

---

## Quick Example

Simulate or evaluate analytical electrochemistry in just a few lines of Python:

```python
import numpy as np
import softpotato as sp

# 1. Analytical Cottrell current transient at a planar electrode
t = np.linspace(0.01, 10.0, 200)
current = sp.analytical.cottrell(
    t=t,
    n=1,
    D=1e-9,  # m^2 / s
    c_bulk=1.0,  # mol / m^3 (1 mM)
    A=1e-4,  # m^2 (1 cm^2)
)

# 2. Fit experimental chronoamperometric data to recover diffusivity D
fit = sp.analytical.fit_cottrell(t=t, current=current, n=1, A=1e-4, c_bulk=1.0)
print(f"Extracted D = {fit.D:.3e} m^2/s (R^2 = {fit.r_squared:.5f})")
```

!!! note "Support Soft Potato"
    If you find Soft Potato useful in your research or teaching, consider supporting the project on [Buy Me a Coffee](https://www.buymeacoffee.com/oliverrdz).
