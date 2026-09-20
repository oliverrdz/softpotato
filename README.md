# Soft Potato

[![CI](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverrdz/softpotato/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/soft-potato/badge/?version=latest)](https://soft-potato.readthedocs.io/en/latest)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

**Soft Potato** is an open-source electrochemical simulator and toolkit designed for electrochemists, materials scientists, and engineers. It enables modeling of complex multi-step reaction mechanisms, spatial mass transport, and transient electrochemical techniques with rigorous physical unit enforcement.

> [!NOTE]
> Soft Potato is currently in active development (`v3.0.0.dev1`). Core mechanisms and analytical solutions are ready for use, while numerical solvers and technique waveforms are in development or targeting upcoming milestones. See the [Development Roadmap](docs/roadmap.rst) for full details.

---

## Key Features

* **Electrochemical Mechanisms** `[Available · v3.0]`: Formulate complex reaction networks (E, EC, CE, EE, ECE, and DISP) combining heterogeneous electron transfers and homogeneous chemical reactions.
* **Analytical Benchmarks** `[Available · v3.0]`: Vectorized closed-form solutions (Randles–Sevcik, Cottrell, Saito microdisc) for instant benchmarking and parameter sweeping.
* **Mass Transport & Simulation** `[In Development · v3.1]`: Numerical PDE solvers (Explicit & Implicit Finite Difference) for diffusion and coupled chemical kinetics.
* **Electrode Geometries & Grids** `[In Development · v3.1]`: Support for planar macroelectrodes, spherical electrodes, and microdisc geometries with uniform and expanding spatial grids.
* **Electrochemical Techniques** `[Planned · v3.2]`: Cyclic voltammetry (CV), chronoamperometry, and potential step simulations.
* **Kinetics Models** `[Planned · v3.2]`: Butler–Volmer and Marcus–Hush kinetics rate models.
* **Physical Units**: Strict enforcement of the CGS unit system ($D$ in $\text{cm}^2/\text{s}$, $c$ in $\text{mol}/\text{cm}^3$, area in $\text{cm}^2$, radius in $\text{cm}$, potential in $\text{V}$).

---

## Implementation Status

Soft Potato is undergoing phased development. Below is the implementation readiness by module:

| Module | Status | Target | Description |
| :--- | :--- | :--- | :--- |
| `softpotato.core` | **Ready** | v3.0 | Species, reactions, and mechanisms with CGS unit enforcement |
| `softpotato.analytical` | **Ready** | v3.0 | Cottrell, Randles–Sevcik, Saito microdisc closed-form solutions |
| `softpotato.geometry` | *In Development* | v3.1 | Planar, spherical electrode geometries & spatial grids |
| `softpotato.simulate` | *In Development* | v3.1 | EFD and IFD finite difference PDE solvers |
| `softpotato.techniques` | *Planned* | v3.2 | Waveforms for cyclic voltammetry and potential step methods |
| `softpotato.kinetics` | *Planned* | v3.2 | Butler–Volmer & Marcus–Hush electron transfer kinetics |

For detailed item-by-item progress and active development tasks, see the [Development Roadmap](docs/roadmap.rst).


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

---

## Documentation & Tutorials

* **Full Documentation**: [Read the Docs](https://soft-potato.readthedocs.io/)
* **Tutorial Notebooks**:
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