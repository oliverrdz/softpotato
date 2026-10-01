# Soft Potato Development Roadmap

This document outlines the strategic development plan and versioning milestones for **Soft Potato** (3.x series).

Soft Potato adheres to [Semantic Versioning (SemVer 2.0.0)](https://semver.org/) and [PEP 440](https://peps.python.org/pep-0440/):
- **Major release (`3.0.0`)**: Establishes the new stable architecture and API baseline following the complete version 3 rewrite.
- **Minor releases (`3.x.0`)**: Introduce new, backwards-compatible domain modules and functional enhancements without breaking existing APIs.
- **Patch releases (`3.x.y`)**: Bug fixes, performance optimizations, and documentation improvements.

---

## Milestone Overview

| Milestone | Target Version | Focus Area | Status | Release Type |
| :--- | :--- | :--- | :--- | :--- |
| **Current State Stabilization** | `v3.0.0` | Finalize core solver module & `solve_ivp` integration | Released | **Stable Baseline** |
| **Milestone 1: Analytical Equations** | `v3.1.0` | Analytical & empirical equations, simulation benchmarks & parameter fitting | Released | **Minor Feature Addition** |
| **Milestone 2: Grids** | `v3.2.0` | Spatial discretization & non-uniform/expanding meshes | In Development | **Minor Feature Addition** |
| **Milestone 3: Reactions** | `v3.3.0` | Homogeneous & heterogeneous chemical reaction kinetics | Planned | **Minor Feature Addition** |
| **Milestone 4: Techniques** | `v3.4.0` | Electrochemical waveform generators & simulation pipelines | Planned | **Minor Feature Addition** |

---

## Current State Stabilization: v3.0.0

**Goal**: Finalize the current solver module (including the `solve_ivp` integration) and release it as the new stable baseline for the Soft Potato 3.x series.

### Objectives & Deliverables
- **Core Solver Module (`softpotato.solver`)**:
  - Consolidate and finalize the unified PDE problem formulation:
    - `DiffusionProblem`: 1D multi-species diffusion and reaction-diffusion specification with strict validation for positive diffusion coefficients ($D_i > 0$), initial conditions, and domain boundaries.
    - `BoundaryCondition`: Robust abstraction supporting Dirichlet (`DirichletBC`) and Neumann (`NeumannBC`) boundary conditions with constant values or time-dependent callables $f(t)$.
    - `SolverResult`: Structured output container providing spatial and temporal concentration distributions, dict-like species access (`result["A"]`), and automated interfacial flux calculation ($J_i = -D_i \left.\frac{\partial c_i}{\partial x}\right|_{x=0}$).
- **`scipy_ivp` Solver Integration**:
  - Finalize the Method of Lines (MOL) semi-discretization engine (`ScipyIVPSolver`) leveraging `scipy.integrate.solve_ivp`.
  - Default to high-order stiff ODE/DAE integrators (implicit 5th-order Runge-Kutta `Radau` and `BDF`), crucial for steep electrochemical concentration gradients and stiff boundary kinetics.
  - Support user-configurable integration tolerances (`rtol`, `atol`) and adaptive time-stepping.
- **Classical Finite Difference Solvers**:
  - `CrankNicolson`: Second-order implicit scheme ($\mathcal{O}(\Delta t^2, \Delta x^2)$) using efficient tridiagonal banded solvers (`scipy.linalg.solve_banded`).
  - `ImplicitFiniteDifference` (BTCS): Unconditionally stable first-order implicit scheme ($\mathcal{O}(\Delta t, \Delta x^2)$) resistant to high-frequency initial-condition shocks.
  - `ExplicitFiniteDifference` (FTCS): Conditionally stable explicit scheme ($\mathcal{O}(\Delta t, \Delta x^2)$) with Courant-Friedrichs-Lewy (CFL) stability verification ($\Delta t \le \frac{\Delta x^2}{2 D_{\max}}$) and automated substepping.
- **Verification & Quality Assurance**:
  - Validate all solvers against exact analytical solutions (Fourier sine decay, Cottrell chronoamperometry, and total mass conservation under zero-flux boundaries).
  - Reach 100% passing unit and benchmark tests across Python 3.10–3.13.
  - Complete Sphinx documentation with Read the Docs integration and interactive Jupyter tutorials.
- **Release Status**:
  - Successfully stabilized and released `v3.0.0` as the official stable baseline.

---

## Milestone 1 (Analytical Equations): v3.1.0

**Goal**: Introduce a dedicated, vectorized analytical and empirical electrochemistry equations module (`softpotato.analytical`), providing exact benchmark solutions for numerical solvers, parameter estimation and curve fitting utilities for experimental data, and rapid response comparison tools.

### Rationale
Analytical and empirical equations form the classical bedrock of electrochemistry. While numerical PDE solvers are indispensable for complex geometries, coupled homogeneous reaction mechanisms, or arbitrary potential waveforms, closed-form analytical solutions provide:
1. **Exact Ground-Truth Benchmarking**: Rigorous validation standards to verify numerical solvers (`softpotato.solver`), ensuring spatial and temporal discretization convergence, flux conservation, and boundary stencil fidelity without truncation errors.
2. **Instant Parameter Fitting & Estimation**: Direct extraction of key physical and kinetic parameters ($D$, $k^0$, $\alpha$, $c^*$, electrode area $A$, microdisc radius $a$, spherical radius $r$) from experimental or simulated data via non-linear regression, eliminating the computational cost of iterative PDE solving in optimization loops.
3. **Rapid Response Comparison & Screening**: Instantaneous theoretical baselines for voltammetric, chronoamperometric, hydrodynamic, microelectrode, SECM, and steady-state responses to rapidly interpret experimental observations and inform numerical model configuration.

### Objectives & Deliverables
- **Transient Diffusion & Potential Step Equations (`softpotato.analytical.diffusion` / `softpotato.analytical.step`)**:
  - `cottrell`: Planar Cottrell equation for semi-infinite linear diffusion chronoamperometry:
    $$I(t) = \frac{n F A \sqrt{D} c^*}{\sqrt{\pi t}}$$
  - `cottrell_spherical`: Cottrell equation for spherical electrodes, hanging mercury drops, and micro-spheres of radius $r$:
    $$I(t) = n F A D c^* \left(\frac{1}{\sqrt{\pi D t}} + \frac{1}{r}\right)$$
  - `cottrell_step`: Double potential step chronoamperometry reversal transient ($I(t)$ for $t > \tau$).
  - `anson`: Anson equation for chronocoulometry cumulative charge transients:
    $$Q(t) = \frac{2 n F A \sqrt{D} c^* \sqrt{t}}{\sqrt{\pi}} + Q_{\text{dl}} + n F A \Gamma$$
  - `sand`: Sand equation for chronopotentiometry transition time $\tau$ under constant current:
    $$I \tau^{1/2} = \frac{n F A \sqrt{\pi D} c^*}{2}$$
  - `cottrell_cylinder`: Short- and long-time asymptotic expansions for cylindrical wire/fiber electrodes.
- **Hydrodynamics & Convection Equations (`softpotato.analytical.hydrodynamics`)**:
  - `levich`: Levich equation for convective mass-transport limiting current at a Rotating Disk Electrode (RDE):
    $$I_L = 0.620\, n F A D^{2/3} \omega^{1/2} \nu^{-1/6} c^*$$
  - `koutecky_levich`: Koutecký–Levich equation for mixed kinetic and mass-transfer limitation:
    $$\frac{1}{I} = \frac{1}{I_K} + \frac{1}{I_L} = \frac{1}{n F A k(E) c^*} + \frac{1}{0.620\, n F A D^{2/3} \omega^{1/2} \nu^{-1/6} c^*}$$
  - `rotating_ring_disk`: Theoretical collection efficiency ($N$) and ring limiting currents for RRDE systems.
- **Microelectrode & Ultramicroelectrode (UME) Geometries (`softpotato.analytical.microelectrodes`)**:
  - `microdisc_limiting_current`: Saito steady-state limiting current for an inlaid microdisc of radius $a$:
    $$I_{\text{lim}} = 4 n F D c^* a$$
  - `microdisc_transient`: Shoup and Szabo empirical approximation for inlaid microdisc chronoamperometry with radius $a$, spanning the entire transition from short-time planar Cottrell decay to long-time steady state ($<0.6\%$ maximum relative error):
    $$I(t) = \frac{4 n F D c^* a}{f(\tau)}, \quad \tau = \frac{4 D t}{a^2}$$
  - `mahon_oldham_transient`: Mahon and Oldham analytical expression for transient diffusion current at an inlaid microdisc electrode of radius $a$ across all time regimes:
    $$I(t) = 4 n F D c^* a \cdot f(\theta), \quad \theta = \frac{4 D t}{a^2}$$
  - `microhemisphere_limiting_current` & `microsphere_limiting_current`: Steady-state limiting currents for hemispherical ($I_{\text{lim}} = 2 \pi n F D c^* r$) and spherical ($I_{\text{lim}} = 4 \pi n F D c^* r$) microelectrodes of radius $r$.
  - `microband_limiting_current`: Steady-state and quasi-steady-state current approximations for microband geometries.
- **Scanning Electrochemical Microscopy (SECM) Approach Curves (`softpotato.analytical.secm`)**:
  - `secm_approach_positive_feedback`: Lefrou and Cornut analytical approximation for steady-state normalized tip current ($I_T = i_T / i_{T,\infty}$ with $i_{T,\infty} = 4 n F D c^* a$) over a conductive substrate (positive feedback / diffusion-controlled mediator regeneration) as a function of normalized distance ($L = d / a$) and insulator radius ratio ($RG = r_g / a$):
    $$I_T^{\text{cond}}(L, RG)$$
  - `secm_approach_negative_feedback`: Lefrou and Cornut analytical approximation for steady-state normalized tip current over an insulating substrate (negative feedback / hindered diffusion) as a function of normalized distance ($L = d / a$) and insulator radius ratio ($RG = r_g / a$):
    $$I_T^{\text{ins}}(L, RG)$$
- **Voltammetry & Kinetic Diagnostics (`softpotato.analytical.voltammetry`)**:
  - `randles_sevcik`: Randles–Ševčík peak current for reversible electron transfer at planar electrodes:
    $$I_p = 0.4463\, n F A c^* \sqrt{\frac{n F D v}{R T}}$$
  - `randles_sevcik_irreversible`: Peak current for totally irreversible electron transfer:
    $$I_p = 0.4958\, n F A c^* \sqrt{\frac{\alpha n_\alpha F D v}{R T}}$$
  - `randles_sevcik_quasi`: Peak current and shape factor approximations across intermediate quasi-reversible regimes.
  - `peak_potential_irreversible`: Peak potential shift relationship with scan rate $v$ for irreversible systems.
  - `nicholson_psi`: Nicholson method relating peak potential separation ($\Delta E_p$) to the dimensionless kinetic parameter $\Psi$ and standard rate constant $k^0$, providing both forward rational approximations and inverse solvers.
  - `matsuda_ayabe`: Dimensionless parameter $\Lambda$ boundary criteria classifying reversible ($\Lambda \ge 15$), quasi-reversible ($10^{-3} < \Lambda < 15$), and irreversible ($\Lambda \le 10^{-3}$) voltammetric systems.
- **Thermodynamics & Interfacial Kinetics (`softpotato.analytical.kinetics`)**:
  - `nernst`: Nernst equation for equilibrium electrode potential and surface concentration ratios:
    $$E = E^{0\prime} + \frac{R T}{n F} \ln\left(\frac{c_{\text{Ox}}}{c_{\text{Red}}}\right)$$
  - `butler_volmer`: Butler–Volmer equation for current density and net current as a function of overpotential $\eta = E - E^{0\prime}$:
    $$I = I_0 \left[\exp\left(\frac{(1 - \alpha) n F \eta}{R T}\right) - \exp\left(-\frac{\alpha n F \eta}{R T}\right)\right], \quad I_0 = n F A k^0 (c_{\text{Ox}}^*)^{1-\alpha} (c_{\text{Red}}^*)^\alpha$$
  - `tafel`: Anodic and cathodic high-overpotential Tafel approximations ($\eta = a + b \log_{10}|I|$) with automated extraction of Tafel slopes ($b$) and exchange currents ($I_0$).
- **Simulation Benchmarking Suite (`softpotato.analytical.benchmark`)**:
  - Automated benchmark harness comparing `softpotato.solver` numerical results against analytical solutions.
  - Standardized error metrics: Root-Mean-Square Error (RMSE), maximum pointwise deviation ($L_\infty$), relative error, and order-of-convergence verification across $\Delta x$ and $\Delta t$.
- **Parameter Fitting & Experimental Data Analysis (`softpotato.analytical.fitting`)**:
  - Optimization wrappers around `scipy.optimize.curve_fit` / `least_squares` with physically bounded parameter constraints ($D > 0$, $0 < \alpha < 1$, $k^0 \ge 0$, $c^* \ge 0$, $A > 0$, $a > 0$, $r > 0$).
  - Direct linear and non-linear regression utilities: Cottrell $I$ vs. $t^{-1/2}$, Randles–Ševčík $I_p$ vs. $v^{1/2}$, Levich $I_L$ vs. $\omega^{1/2}$, Koutecký–Levich $1/I$ vs. $\omega^{-1/2}$, Tafel plots ($\eta$ vs. $\log_{10}|I|$), and SECM approach curve fitting ($RG$, distance $d$, and tip offset).
  - Statistical confidence intervals, standard errors, parameter correlation matrices, and goodness-of-fit indicators ($R^2$, reduced $\chi^2$).
- **Architecture & Compatibility**:
  - Fully vectorized NumPy implementations supporting scalar and array inputs with dimensional broadcasting.
  - Centralized physical constants (`FARADAY`, `GAS_CONSTANT`, `STANDARD_TEMPERATURE`) with explicit unit conventions.
- **Release Status**:
  - Successfully implemented, benchmarked, tested, and released as `v3.1.0`.

---

## Milestone 2 (Grids): v3.2.0

**Status**: In Development

**Goal**: Introduce a dedicated spatial discretization and meshing module (`softpotato.grid`), adding non-uniform and expanding grids as a new, backwards-compatible feature.

### Rationale
In electrochemical simulations, diffusion layers are typically confined to a narrow region adjacent to the electrode surface ($x=0$). Uniform grids require an excessively large number of spatial points ($N_x$) to resolve steep surface gradients, inflating computational cost. Expanding grids concentrate spatial resolution near the electrode while coarsening into the bulk solution, delivering orders-of-magnitude faster simulations without sacrificing precision.

### Objectives & Deliverables
- **Meshing Module (`softpotato.grid`)**:
  - `Uniform1DMesh`: Standard equidistant grid generator.
  - `GeometricExpandingMesh`: Exponential / geometrically expanding spatial discretization:
    $$\Delta x_j = \Delta x_0 \cdot \gamma^j \quad (\gamma > 1)$$
  - `ExponentialMesh` and conformal transformation-based coordinates for semi-infinite diffusion domains.
- **Non-Uniform Discretization Operators**:
  - Implement second-order finite difference stencils adapted for non-uniform spacing for both first ($\partial c / \partial x$) and second ($\partial^2 c / \partial x^2$) spatial derivatives.
  - Ghost-node and boundary stencil formulations on irregular node intervals for Dirichlet and Neumann boundary conditions.
- **Backwards Compatibility**:
  - Ensure `DiffusionProblem` seamlessly accepts either raw 1D NumPy arrays (`np.ndarray`) or structured mesh objects (`Mesh1D`).
  - Existing `v3.0.0` and `v3.1.0` solver scripts and tutorials continue executing with zero code changes.
- **Solver Adaptations**:
  - Enable `ScipyIVPSolver` and banded implicit solvers to consume non-uniform spatial discretizations natively.

---

## Milestone 3 (Reactions): v3.3.0

**Goal**: Introduce a comprehensive chemical and electrochemical kinetics module (`softpotato.reactions`), providing modular multi-species reaction mechanisms.

### Rationale
Electrochemical systems frequently involve coupled chemical transformations (e.g., EC, EC', ECE, catalytic, and disproportionation pathways) as well as non-linear interfacial charge-transfer kinetics. A dedicated reactions module decouples kinetic mechanism definitions from the spatial diffusion solvers.

### Objectives & Deliverables
- **Chemical Species Abstraction (`softpotato.reactions.species`)**:
  - `Species`: Class encapsulating species identity, standard diffusion coefficient ($D_i$), bulk concentration ($c_i^*$), formal charge ($z_i$), and standard redox potential ($E^0$).
- **Homogeneous Reaction Kinetics (`softpotato.reactions.homogeneous`)**:
  - Flexible multi-reaction formulation computing reaction rate vectors $\mathbf{R}(\mathbf{c}, x, t)$:
    $$\frac{\partial c_i}{\partial t} = D_i \nabla^2 c_i + R_i(\mathbf{c})$$
  - Built-in reaction primitives: first-order, second-order, reversible equilibrium ($A \rightleftharpoons B$), and catalytic cycles.
  - Automated kinetic Jacobian matrix formulation for implicit stiff ODE integration (`Radau`, `BDF`).
- **Heterogeneous Charge-Transfer Kinetics (`softpotato.reactions.heterogeneous`)**:
  - Butler–Volmer kinetics:
    $$k_f = k^0 \exp\left(-\frac{\alpha n F}{R T}(E(t) - E^0)\right), \quad k_b = k^0 \exp\left(\frac{(1 - \alpha) n F}{R T}(E(t) - E^0)\right)$$
  - Marcus–Hush–Chidsey (MHC) kinetics for non-adiabatic electron transfer.
  - Reversible Nernstian boundary condition helpers.
- **Integration**:
  - Plug-and-play linkage with `DiffusionProblem(..., reactions=...)`.
  - Backwards-compatible: `DiffusionProblem` remains functional without explicit reaction definitions.

---

## Milestone 4 (Techniques): v3.4.0

**Goal**: Introduce high-level electrochemical experimental techniques and waveform generators (`softpotato.techniques`), enabling turnkey simulation and diagnostic workflows.

### Rationale
With solvers, analytical benchmark models, grids, and reaction kinetics in place, users need standard experimental protocols (voltammetry, chronoamperometry) and automated signal pipelines without manually coding potential excitation waveforms and boundary conditions.

### Objectives & Deliverables
- **Waveform & Signal Generators (`softpotato.techniques`)**:
  - `Chronoamperometry`: Single- and double-potential step sequences with configurable pulse durations and resting times.
  - `LinearSweepVoltammetry` (LSV): Unidirectional potential sweeps with configurable start/end potentials and scan rates ($v$).
  - `CyclicVoltammetry` (CV): Multi-cycle triangular potential waveforms with customizable vertices and switching potentials.
  - `RotatingDiskElectrode` (RDE): Hydrodynamic convective-diffusion modeling under the Levich and Nernst stagnant diffusion layer approximations.
  - `PulseTechniques`: Differential Pulse Voltammetry (DPV), Square Wave Voltammetry (SWV), and Chronopotentiometry (CP).
  - `CustomWaveform`: Composite excitation generator allowing arbitrary chaining of potential/current segments.
- **Simulation Pipeline Orchestrator**:
  - High-level execution API (e.g., `simulate(technique=..., mechanism=..., solver=...)`) that coordinates grid creation, boundary condition assignment, numerical integration, and result extraction in a single call.
- **Post-Processing & Electrochemical Diagnostics**:
  - Automated peak detection ($E_p$, $I_p$, $\Delta E_p$).
  - Diagnostics: Randles–Ševčík peak current regression, Nicholson kinetic parameter extraction, Levich and Koutecký–Levich slopes, and Tafel analysis leveraging the `softpotato.analytical` module.
- **Backwards Compatibility**:
  - Modular addition that imports and utilizes `softpotato.solver`, `softpotato.analytical`, `softpotato.grid`, and `softpotato.reactions` without breaking lower-level solver APIs.

---

## Summary of Semantic Versioning Strategy

```text
v3.0.0 (Stable Baseline)
 └── Core Solvers Finalized (scipy_ivp, Crank-Nicolson, Implicit, Explicit)
      │
      ├── v3.1.0 (Minor Bump: Non-breaking feature)
      │    └── Analytical & empirical electrochemical equations (softpotato.analytical)
      │
      ├── v3.2.0 (Minor Bump: Non-breaking feature)
      │    └── Non-uniform and expanding spatial grids (softpotato.grid)
      │
      ├── v3.3.0 (Minor Bump: Non-breaking feature)
      │    └── Homogeneous & heterogeneous chemical reactions (softpotato.reactions)
      │
      └── v3.4.0 (Minor Bump: Non-breaking feature)
           └── Electroanalytical techniques & waveforms (softpotato.techniques)
```

Each minor release maintains strict backwards compatibility with previous releases, ensuring scripts written against `v3.0.0` continue to run unmodified throughout the entire `3.x` lifecycle.
