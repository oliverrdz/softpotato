Ideally:
```python
import softpotato as sp
import matplotlib.pyplot as plt

ox = sp.physics.species(D_ox, c0_ox)
red = sp.physics.species(D_ref, c0_red)
p = sp.physics.species(D_p, c0_p)

bv = sp.physics.kinetics("BV", params)
fo = sp.physics.kinetics("first_order", params)

E_mech = sp.mechanism([ox, Red], kinetics=bv)
C_mech = sp.mechanism([p], kinetics=fo)
mech = sp.mechanism.build([E_mech, C_mech])

ca = sp.technique.ca(E_step, t_tot, dt)
lsv = sp.technique.lsv(E_ini, Ev1, Ev2, sr, dE, n_cycles)
waveform = sp.technique.build([ca, lsv])

grid = sp.grid.planar(x_max, nx, type=”expanding”)

solver = sp.solver.solve_ivp(params)

sim = sp.simulate(waveform, mech, grid, solver)

plt.figure()
plt.plot(sim.E, sim.i)
plt.figure()
plt.plot(sim.x, sim.c_ox)
plt.plot(sim.x, sim.c_red)
```

# softpotato Master System Specification & Instruction Manual

This document serves as the master specification for softpotato, a modular, object-oriented electrochemical simulation engine written in Python. It details the architectural design, component responsibility matrix, governing mathematical equations, data export contracts, and guidelines for AI-assisted module development. Enforce the use of SI units.

---

## 1. Project Overview & Architectural Principles

### Core Objectives
* **Decoupled Architecture**: Strictly separates physical definitions, reaction mechanisms, potential signal waveforms, spatial grid generation, and numerical solvers.
* **Modular Pipeline**: Provides clean top-level abstractions so user code can assemble complex electrochemical simulations with minimal syntax.
* **Performance**: Employs NumPy vectorization for high-speed matrix computations and SciPy for stiff initial-value problem integration.

### Core Design Patterns
1. **Composite Pattern**: Used in reaction mechanisms (`sp.mechanism.build`) and dynamic potential waveforms (`sp.technique.build`) to combine simple primitive steps into multi-step operational schemes.
2. **Factory Pattern**: Utilized in kinetic rate selection (`sp.physics.kinetics`), spatial grid generation (`sp.grid.planar`), and solver instantiation (`sp.solver.solve_ivp`) to select operational algorithms via high-level string identifiers.
3. **Data-Centric Pipeline Pattern**: `sp.simulate()` acts as the single execution gateway, compiling input specifications, building Partial Differential Equations (PDEs), enforcing boundary conditions, running time integration, and returning a unified results container.

---

## 2. Directory Architecture & Component Responsibility Matrix

The repository follows a standard Python `src/` layout compatible with `setuptools` and `pyproject.toml` packaging.

### Package Hierarchy
* **softpotato/**
  * `pyproject.toml` (Build configuration and dependency specifications)
  * `README.md` (Project overview and installation documentation)
  * **src/softpotato/**
    * `__init__.py` (Main package initialization and top-level exports)
    * **physics/** (`species.py`, `kinetics.py`)
    * **mechanism/** (`steps.py`, `builder.py`)
    * **technique/** (`base.py`, `chrono.py`, `sweep.py`)
    * **grid/** (`spatial.py`, `mesh_generators.py`)
    * **solver/** (`ivp.py`)
    * **core/** (`pde_builder.py`, `boundary_conditions.py`)
    * `results.py` (Output container and export utilities)
    * `simulator.py` (Pipeline orchestrator)
  * **tests/** (`unit/`, `integration/`)

### Component Responsibility Matrix

| Submodule | Target File | Key Classes / Functions | Responsibility |
| :--- | :--- | :--- | :--- |
| **sp.physics** | `species.py` | `Species` | Dataclass representing mass-transport properties ($D$, $c_0$). |
| | `kinetics.py` | `Kinetics`, `kinetics()` | Factory for kinetic rate models (Butler-Volmer, Marcus-Hush, first-order chemical). |
| **sp.mechanism** | `steps.py` | `ElementaryStep` | Data representations for elementary electron-transfer ($E$) and chemical ($C$) steps. |
| | `builder.py` | `Mechanism`, `build()` | Merges $E$ and $C$ steps into composite reaction schemes ($EC$, $ECE$, catalytic $EC'$). |
| **sp.technique** | `chrono.py` | `ca()` | Chronoamperometry wave generator ($E_{\text{step}}$, $t_{\text{tot}}$, $dt$). |
| | `sweep.py` | `lsv()`, `cv()` | Voltammetry sweep generators ($E_{\text{ini}}$, $E_{v1}$, $E_{v2}$, $\nu$, $dE$, $n_{\text{cycles}}$). |
| | `base.py` | `Waveform`, `build()` | Composite builder for chaining sequential potential waveforms over time. |
| **sp.grid** | `spatial.py` | `Grid` | Dataclass representing spatial node coordinates ($x$) and intervals ($dx$). |
| | `mesh_generators.py` | `planar()`, `spherical()` | Generates uniform and expanding non-uniform 1D spatial meshes. |
| **sp.solver** | `ivp.py` | `solve_ivp()` | Factory wrapping stiff initial-value problem integrators (BDF, Radau, Thomas algorithm). |
| **sp.core** | `pde_builder.py` | `PDEAssembler` | Formulates second-order spatial derivative matrices and reaction terms $R_i(c, t)$. |
| | `boundary_conditions.py` | `BoundaryHandler` | Enforces surface flux boundary conditions ($x = 0$) and far-field bulk conditions ($x = x_{\text{max}}$). |
| **Top-Level** | `simulator.py` | `simulate()` | Orchestrator running time integration and calculating electrode current response $i(t)$. |
| | `results.py` | `SimulationResult` | Output container storing results, native export methods, and optional plotting helpers. |

---

## 3. Mathematical Model & Boundary Conditions

The core simulation engine solves coupled Fickian diffusion and chemical reaction rate terms for species $i$:

$$\frac{\partial c_i}{\partial t} = D_i \frac{\partial^2 c_i}{\partial x^2} + R_i(c, t)$$

### Boundary Conditions
1. **Electrode Surface Interface ($x = 0$)**: Flux is governed by heterogeneous kinetic rate constants and applied potential $E(t)$:
   $$-D_i \left( \frac{\partial c_i}{\partial x} \right)_{x=0} = f(E(t), c|_{x=0})$$
2. **Bulk Boundary ($x = x_{\text{max}}$)**: Concentrations remain fixed at initial bulk values $c_0$:
   $$c_i(x_{\text{max}}, t) = c_{0, i}$$

### Analytical Benchmarks for Verification
Generated modules must pass automated mathematical verification tests:
* **Cottrell Equation Benchmark (Chronoamperometry)**:
  $$i(t) = \frac{n F A D^{1/2} c_0}{\pi^{1/2} t^{1/2}}$$
* **Randles-Sevcik Scaling (Cyclic Voltammetry)**: Peak current density $i_p$ must scale linearly with the square root of scan rate ($i_p \propto \sqrt{\nu}$).

---

## 4. Submodule Specifications & API Contracts

### A. Physics (`sp.physics`)
* `sp.physics.species(D, c0)`: Instantiates a `Species` dataclass containing $D > 0$ (diffusion coefficient) and $c_0 \ge 0$ (initial bulk concentration).
* `sp.physics.kinetics(model, params)`: Instantiates rate evaluation models including heterogeneous kinetics (`"BV"`, `"Marcus"`) and homogeneous kinetics (`"first_order"`, `"second_order"`, `"catalytic"`).

### B. Reaction Mechanisms (`sp.mechanism`)
* `sp.mechanism.build([...])`: Merges elementary electron transfer steps ($E$) and chemical transformation steps ($C$) into composite mechanisms like $EC$, $ECE$, or $EC'$.

### C. Experimental Techniques (`sp.technique`)
* `sp.technique.ca(E_step, t_tot, dt)`: Chronoamperometry signal generator.
* `sp.technique.lsv(E_ini, Ev1, Ev2, sr, dE, n_cycles)`: Voltammetric sweep signal generator.
* `sp.technique.build([...])`: Composite builder chaining sequential waveforms into a unified potential signal $E(t)$.

### D. Spatial Grid (`sp.grid`)
* `sp.grid.planar(x_max, nx, type="expanding")`: Generates 1D spatial node meshes. Supports geometry selection (`"planar"`, `"spherical"`, `"cylindrical"`, `"microdisc"`) and mesh density distribution (`"expanding"` / `"exponential"` for fine spatial nodes near $x=0$, or `"uniform"` for even spacing $dx$).

### E. Numerical Solver (`sp.solver`)
* `sp.solver.solve_ivp(params)`: Numerical solver interface wrapping stiff ODE integrators (SciPy BDF/Radau or implicit Thomas algorithm / Crank-Nicolson).

### F. Simulation Execution (`sp.simulate`)
* **Signature**: `sim = sp.simulate(waveform, mech, grid, solver)`.
* **Returns**: A `SimulationResult` object containing time series (`sim.t`, `sim.E`, `sim.i`) and spatial concentration arrays (`sim.x`, `sim.c_ox`, `sim.c_red`).

---

## 5. Output Handling, Data Export & Optional Plotting (`sp.results`)

The `SimulationResult` object (`src/softpotato/results.py`) stores execution outputs and provides native methods for data export and optional visualization.

### Output Data Attributes
* `sim.t`: 1D array of time points $t$.
* `sim.E`: 1D array of applied potential values $E(t)$.
* `sim.i`: 1D array of calculated current responses $i(t)$.
* `sim.x`: 1D array of spatial node coordinates $x$.
* `sim.c_ox`, `sim.c_red`, etc.: 2D concentration matrices across time and space $c(x, t)$.

### Data Export Methods
1. `to_dataframe()`: Converts time-series parameters (`t`, `E`, `i`) into a Pandas DataFrame for in-memory processing.
2. `to_csv(filepath)`: Exports time-series output directly to a CSV file.
3. `to_json(filepath)`: Exports simulation parameters, metadata, and options to a JSON file.
4. `to_npz(filepath)`: Saves raw time, potential, current, spatial grid coordinates, and full 2D concentration profiles into a compressed NumPy binary file (`.npz`).

### Built-In Plotting Helpers
* `sim.plot_cv(ax=None, **kwargs)`: Plots current $i$ versus potential $E$ (Cyclic Voltammogram).
* `sim.plot_profiles(time_index=-1, ax=None, **kwargs)`: Plots species concentration profiles $c(x)$ across spatial coordinates at a specified time step.

---

## 6. AI Development Guidelines & Prompting Protocol

When instructing Gemini to build submodules for `softpotato`, enforce these standard instructions:

1. **Scope Isolation**: Implement one target file per prompt according to the component matrix. Do not attempt to generate multiple submodules simultaneously.
2. **Coding Standards**: Maintain PEP 8 compliance, explicit Python type hints (`typing`), and standard NumPy docstrings.
3. **Vectorization**: Utilize pure NumPy array operations. Explicit Python `for` loops inside numerical evaluation functions or spatial matrix calculations are prohibited.
4. **Test-First Delivery**: Every generated module must include a corresponding `pytest` suite covering normal execution, edge cases, boundary parameters, and error handling.