# Soft Potato

[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/soft-potato/badge/?version=latest)](https://soft-potato.readthedocs.io/en/latest)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**Soft Potato** is an open-source electrochemical simulator and toolkit designed for electrochemists, materials scientists, and engineers. It enables modeling of complex multi-step reaction mechanisms, spatial mass transport, and transient electrochemical techniques with rigorous physical unit enforcement.

> [!NOTE]
> Soft Potato is currently in active development (`v3.0.0rc1`). Release **v3.0.0** includes an end-to-end simulation MVP for Cyclic Voltammetry with Butler-Volmer kinetics at a planar electrode via Explicit Finite Differences, alongside core mechanisms and analytical solutions. See the [Development Roadmap](docs/roadmap.rst) for full details.

---

## Key Features

* **Numerical Simulation MVP** `[Available · v3.0]`: End-to-end simulation of Cyclic Voltammetry (CV) with Butler-Volmer kinetics, 1D semi-infinite planar diffusion (`PlanarElectrode`), and an Explicit Finite Difference (`EFD`) PDE solver validated against the Randles-Sevcik analytical benchmark.
* **Electrochemical Mechanisms** `[Available · v3.0]`: Formulate complex reaction networks (E, EC, CE, EE, ECE, and DISP) combining heterogeneous electron transfers and homogeneous chemical reactions.
* **Analytical Benchmarks** `[Available · v3.0]`: Vectorized closed-form solutions (Randles–Sevcik, Cottrell, Anson, Saito microdisc, Levich, Koutecký-Levich) for instant benchmarking and parameter sweeping.
* **Extended Techniques** `[Planned · v3.1]`: Chronoamperometry (CA) and Linear Sweep Voltammetry (LSV).
* **Extended Kinetics** `[Planned · v3.1]`: Nernstian equilibrium and Tafel kinetics models.
* **Advanced Numerical Solvers** `[Planned · v3.2]`: SciPy `solve_ivp` wrapper and Crank-Nicolson Implicit Finite Difference (`IFD`).
* **Non-Planar Geometries & Grids** `[Planned · v3.3]`: Spherical electrodes (`SphericalElectrode`), thin-layer cells, and exponentially expanding spatial grids (`ExpandingGrid`).
* **Physical Units**: Strict enforcement of the CGS unit system ($D$ in $\text{cm}^2/\text{s}$, $c$ in $\text{mol}/\text{cm}^3$, area in $\text{cm}^2$, radius in $\text{cm}$, potential in $\text{V}$).

---

## Implementation Status

Soft Potato is undergoing phased development. Below is the implementation readiness by module:

| Module | Status | Target | Description |
| :--- | :--- | :--- | :--- |
| `softpotato.core` | **Ready** | v3.0 | Species, reactions, and mechanisms with CGS unit enforcement |
| `softpotato.analytical` | **Ready** | v3.0 | Cottrell, Randles–Sevcik, Saito, Levich, Koutecký-Levich closed-form solutions |
| `softpotato.geometry` | **MVP Ready** | v3.0 | `PlanarElectrode` and `UniformGrid` (Spherical & expanding grids planned for v3.3) |
| `softpotato.techniques` | **MVP Ready** | v3.0 | `CyclicVoltammetry` waveform generator (CA & LSV planned for v3.1) |
| `softpotato.kinetics` | **MVP Ready** | v3.0 | `ButlerVolmer` electron transfer model (Nernst & Tafel planned for v3.1) |
| `softpotato.simulate` | **MVP Ready** | v3.0 | `EFD` finite difference solver & `Solver` dispatcher (`solve_ivp` & `IFD` planned for v3.2) |

For detailed item-by-item progress and active development tasks, see the [Development Roadmap](docs/roadmap.rst).

---

## Installation

Install Soft Potato from GitHub:

```bash
# Direct install from GitHub
pip install "git+https://github.com/oliverrdz/softpotato.git"
```

For local development and running the tutorial notebooks:

```bash
git clone https://github.com/oliverrdz/softpotato.git
cd softpotato
pip install -e ".[test,docs]"
```

---

## Quickstart

### 1. Analytical Benchmark (Available · v3.0)

Compute peak currents from the Randles–Sevcik equation across a range of scan rates:

```python
import numpy as np
from softpotato.analytical import randles_sevcik

# Parameters (strict CGS units)
n = 1
area = 0.0707          # cm^2 (macroelectrode)
D_O = 1e-5             # cm^2/s
c_bulk = 1e-6          # mol/cm^3 (1 mM)
scan_rates = np.linspace(0.01, 1.0, 50)  # V/s

# Vectorized calculation of peak currents (A)
i_peak = randles_sevcik(n=n, area=area, D=D_O, c_bulk=c_bulk, scan_rate=scan_rates)
print(f"Peak current at 0.1 V/s: {i_peak[4] * 1e6:.2f} µA")
```

### 2. Defining Reaction Mechanisms (Available in v3.0)

Define chemical species and interfacial electron transfer reactions:

```python
import softpotato as sp

# 1. Define species with diffusion coefficients (cm^2/s) and bulk concentrations (mol/cm^3)
O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)

# 2. Define Butler-Volmer electron transfer
bv = sp.kinetics.ButlerVolmer(k0=1e-3, alpha=0.5)
rxn_E = sp.core.ElectrochemicalReaction(
    reactants=[O], products=[R], n_electrons=1, E0=0.0, kinetics=bv
)

# 3. Create mechanism
mechanism = sp.core.Mechanism([rxn_E])
print(mechanism)
```

### 3. Simulating Cyclic Voltammetry (Available in v3.0 MVP)

Simulate a complete Cyclic Voltammogram (CV) with Butler-Volmer kinetics at a planar macroelectrode:

```python
import softpotato as sp

# 1. Species & Butler-Volmer reaction
O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)
bv = sp.kinetics.ButlerVolmer(k0=1e-2, alpha=0.5)
rxn = sp.core.ElectrochemicalReaction(
    reactants=[O], products=[R], n_electrons=1, E0=0.0, kinetics=bv
)
mechanism = sp.core.Mechanism([rxn])

# 2. Geometry & Grid (Planar semi-infinite diffusion)
grid = sp.geometry.UniformGrid(x_max=0.08, nodes=500)
electrode = sp.geometry.PlanarElectrode(area=0.0707, grid=grid)

# 3. Excitation waveform (Cyclic Voltammetry)
cv = sp.techniques.CyclicVoltammetry(
    E_initial=0.4, E_vertex1=-0.4, scan_rate=0.1, n_sweeps=2, dE=0.002
)

# 4. Run explicit finite difference simulation
solver = sp.simulate.Solver(mechanism, electrode, cv, method="EFD")
result = solver.run()

# 5. Plot the voltammogram
result.plot()
```

---

## Documentation & Tutorials

* **Full Documentation**: [Read the Docs](https://soft-potato.readthedocs.io/)
* **Tutorial Notebooks**:
  * [Cyclic Voltammetry Simulation Tutorial](examples/cv_simulation.ipynb): Step-by-step guide to simulating a CV with Butler-Volmer kinetics and planar diffusion.
  * [Reaction Mechanisms Tutorial](examples/mechanisms.ipynb): Step-by-step guide to E, EC, CE, EE, ECE, and DISP schemes.
  * [Analytical Benchmarks Tutorial](examples/analytical_benchmarks.ipynb): Randles–Sevcik, Cottrell, and Saito microdisc solutions.
* **Design & Architecture**:
  * [API & UI Design Specifications](docs/design_spec.rst): Design targets, modular architecture, and simulation examples.
  * [Development Roadmap](docs/roadmap.rst): Release milestones and implementation status.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
* Development environment setup and instructions.
* Testing guidelines.
* **Codebase Structure**: An annotated map of the repository files and modules.

---

## License

Soft Potato is licensed under the [BSD 3-Clause License](LICENSE).