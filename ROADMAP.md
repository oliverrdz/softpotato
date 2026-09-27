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
| **Milestone 1: Grids** | `v3.1.0` | Spatial discretization & non-uniform/expanding meshes | Planned | **Minor Feature Addition** |
| **Milestone 2: Reactions** | `v3.2.0` | Homogeneous & heterogeneous chemical reaction kinetics | Planned | **Minor Feature Addition** |
| **Milestone 3: Techniques** | `v3.3.0` | Electrochemical waveform generators & simulation pipelines | Planned | **Minor Feature Addition** |

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

## Milestone 1 (Grids): v3.1.0

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
  - Existing `v3.0.0` solver scripts and tutorials continue executing with zero code changes.
- **Solver Adaptations**:
  - Enable `ScipyIVPSolver` and banded implicit solvers to consume non-uniform spatial discretizations natively.

---

## Milestone 2 (Reactions): v3.2.0

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

## Milestone 3 (Techniques): v3.3.0

**Goal**: Introduce high-level electrochemical experimental techniques and waveform generators (`softpotato.techniques`), enabling turnkey simulation and diagnostic workflows.

### Rationale
With solvers, grids, and reaction kinetics in place, users need standard experimental protocols (voltammetry, chronoamperometry) and automated signal pipelines without manually coding potential excitation waveforms and boundary conditions.

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
  - Diagnostics: Randles–Ševčík peak current regression, Nicholson kinetic parameter extraction, Levich and Koutecký–Levich slopes, and Tafel analysis.
- **Backwards Compatibility**:
  - Modular addition that imports and utilizes `softpotato.solver`, `softpotato.grid`, and `softpotato.reactions` without breaking lower-level solver APIs.

---

## Summary of Semantic Versioning Strategy

```text
v3.0.0 (Stable Baseline)
 └── Core Solvers Finalized (scipy_ivp, Crank-Nicolson, Implicit, Explicit)
      │
      ├── v3.1.0 (Minor Bump: Non-breaking feature)
      │    └── Non-uniform and expanding spatial grids (softpotato.grid)
      │
      ├── v3.2.0 (Minor Bump: Non-breaking feature)
      │    └── Homogeneous & heterogeneous chemical reactions (softpotato.reactions)
      │
      └── v3.3.0 (Minor Bump: Non-breaking feature)
           └── Electroanalytical techniques & waveforms (softpotato.techniques)
```

Each minor release maintains strict backwards compatibility with previous releases, ensuring scripts written against `v3.0.0` continue to run unmodified throughout the entire `3.x` lifecycle.
