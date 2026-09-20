# Soft Potato 3.0 planning

Planning.

# Examples of how the UI would work

## Analytical module
```python
import numpy as np
import matplotlib.pyplot as plt

# Import analytical equations directly from the Soft Potato 3.0 facade
from softpotato.analytical import randles_sevcik, cottrell, steady_state_microdisc

# --- Parameters (Strict CGS Units Enforced by Soft Potato) ---
# Assuming standard T = 298.15 K handled internally by the analytical module
n_electrons = 1
D_O = 1e-5            # Diffusion coefficient (cm^2/s)
C_bulk = 1e-6         # Bulk concentration (mol/cm^3) -> 1 mM
area = 0.0707         # Planar macroelectrode area (cm^2)

# --- Independent Variable Arrays (Vectorized) ---
# NumPy arrays are passed directly to avoid slow Python for-loops
v_array = np.linspace(0.01, 1.0, 200)      # Scan rate (V/s)
t_array = np.linspace(0.001, 5.0, 500)     # Time (s)
r_array = np.linspace(1e-4, 25e-4, 200)    # Microdisc radius (cm)

# --- Compute Analytical Solutions ---
i_p = randles_sevcik(n=n_electrons, area=area, D=D_O, c_bulk=C_bulk, scan_rate=v_array)
i_t = cottrell(t=t_array, n=n_electrons, area=area, D=D_O, c_bulk=C_bulk)
i_ss = steady_state_microdisc(n=n_electrons, radius=r_array, D=D_O, c_bulk=C_bulk)

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 1. Randles-Sevcik: i_p vs. sqrt(v)
axes[0].plot(np.sqrt(v_array), i_p * 1e6, color='#1f77b4', lw=2)
axes[0].set_xlabel(r'$\nu^{1/2}$ / (V/s)$^{1/2}$')
axes[0].set_ylabel(r'$i_p$ / $\mu$A')
axes[0].set_title('Randles-Sevcik (Reversible CV)')
axes[0].grid(True, linestyle='--', alpha=0.7)

# 2. Cottrell: i vs. t
axes[1].plot(t_array, i_t * 1e6, color='#d62728', lw=2)
axes[1].set_xlabel('Time / s')
axes[1].set_ylabel(r'$i$ / $\mu$A')
axes[1].set_title('Cottrell (Planar Step)')
axes[1].grid(True, linestyle='--', alpha=0.7)

# 3. Microdisc: i_ss vs. r
# Converting radius to um and current to nA for standard plotting scaling
axes[2].plot(r_array * 1e4, i_ss * 1e9, color='#2ca02c', lw=2)
axes[2].set_xlabel(r'Radius / $\mu$m')
axes[2].set_ylabel(r'$i_{ss}$ / nA')
axes[2].set_title('Saito (Steady-State Microdisc)')
axes[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

## E Mechanism, Macroelectrode, Cyclic Voltammetry, Butler-Volmer
```python
import softpotato as sp

# 1. Species (D in cm^2/s, c_bulk in mol/cm^3)
spec_O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
spec_R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)

# 2. Kinetics & Mechanism
# Butler-Volmer kinetics for a quasi-reversible process
bv = sp.kinetics.ButlerVolmer(k0=1e-3, alpha=0.5)
rxn_E = sp.core.ElectrochemicalReaction(
    reactants=[spec_O], products=[spec_R], n_electrons=1, E0=0.0, kinetics=bv
)
mechanism = sp.core.Mechanism([rxn_E])

# 3. Grid & Geometry
# Planar semi-infinite diffusion. Area in cm^2.
grid = sp.geometry.UniformGrid(x_max=0.05, nodes=500)
electrode = sp.geometry.Planar(area=0.0707, grid=grid)

# 4. Technique
cv = sp.techniques.CyclicVoltammetry(
    E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=2, dE=0.001
)

# 5. Simulate
sim = sp.simulate.Solver(mechanism, electrode, cv, method="EFD")
results = sim.run()
```

## ErCi Mechanism, Macroelectrode, Cyclic Voltammetry, Expanding Grid
```python
import softpotato as sp

# 1. Species 
spec_O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
spec_R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)
spec_Z = sp.core.Species(name="Z", D=1e-5, c_bulk=0.0) # Electroinactive product

# 2. Kinetics & Mechanism
# Er: Nernstian boundary condition (fast kinetics, governed by thermodynamics)
nernst = sp.kinetics.Nernst()
rxn_E = sp.core.ElectrochemicalReaction(
    reactants=[spec_O], products=[spec_R], n_electrons=1, E0=0.0, kinetics=nernst
)

# Ci: Irreversible first-order chemical reaction (kf in s^-1, kb = 0)
chem_irrev = sp.kinetics.FirstOrder(kf=10.0, kb=0.0)
rxn_C = sp.core.ChemicalReaction(
    reactants=[spec_R], products=[spec_Z], kinetics=chem_irrev
)
mechanism = sp.core.Mechanism([rxn_E, rxn_C])

# 3. Grid & Geometry
# Expanding grid (gamma=1.05) to capture sharp concentration gradients near the electrode
# critical for fast following chemical reactions.
exp_grid = sp.geometry.ExpandingGrid(x_max=0.05, nodes=300, gamma=1.05)
electrode = sp.geometry.Planar(area=0.0707, grid=exp_grid)

# 4. Technique & Simulate
cv = sp.techniques.CyclicVoltammetry(
    E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=2, dE=0.001
)
sim = sp.simulate.Solver(mechanism, electrode, cv, method="EFD")
results = sim.run()
```

## E Mechanism, Spherical Electrode, Cyclic Voltammetry
```python
import softpotato as sp

# 1. Species
spec_O = sp.core.Species(name="O", D=1e-5, c_bulk=1e-6)
spec_R = sp.core.Species(name="R", D=1e-5, c_bulk=0.0)

# 2. Kinetics & Mechanism
bv = sp.kinetics.ButlerVolmer(k0=1e-3, alpha=0.5)
rxn_E = sp.core.ElectrochemicalReaction(
    reactants=[spec_O], products=[spec_R], n_electrons=1, E0=0.0, kinetics=bv
)
mechanism = sp.core.Mechanism([rxn_E])

# 3. Grid & Geometry
# Spherical electrode (e.g., a hanging mercury drop or ultramicroelectrode). 
# Radius is strictly in cm.
grid = sp.geometry.UniformGrid(x_max=0.05, nodes=500)
electrode = sp.geometry.Spherical(radius=0.01, grid=grid)

# 4. Technique & Simulate
cv = sp.techniques.CyclicVoltammetry(
    E_initial=0.5, E_vertex1=-0.5, scan_rate=0.1, n_sweeps=2, dE=0.001
)
sim = sp.simulate.Solver(mechanism, electrode, cv, method="EFD")
results = sim.run()
```

# Repository structure

```text
softpotato/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI workflow (multi-version test matrix & doc builds)
├── .gitignore                              # Git ignore rules for Python, virtual environments, and build artifacts
├── .readthedocs.yaml                       # Read the Docs configuration file
├── CHANGELOG.md                            # Project changelog following Keep a Changelog format
├── docs/                                   # Sphinx documentation source files
│   ├── _static/
│   │   └── .gitkeep                        # Preserves static assets directory in git
│   ├── _templates/
│   │   └── .gitkeep                        # Preserves custom HTML templates directory in git
│   ├── api/
│   │   └── index.rst                       # API reference documentation index
│   ├── examples/
│   │   ├── index.rst                       # User guide and tutorial notebooks index
│   │   └── mechanisms.nblink               # Sphinx link to mechanisms.ipynb tutorial notebook
│   ├── conf.py                             # Sphinx build configuration and extension settings
│   ├── index.rst                           # Documentation homepage and table of contents
│   ├── make.bat                            # Windows batch build script for Sphinx documentation
│   ├── Makefile                            # Unix makefile for Sphinx documentation
│   └── roadmap.rst                         # Development roadmap and pending implementation tracking
├── examples/                               # Interactive Jupyter notebook tutorials
│   └── mechanisms.ipynb                    # Tutorial notebook on defining electrochemical and chemical mechanisms
├── LICENSE                                 # BSD 3-Clause License file
├── pyproject.toml                          # Build system, package metadata, dependencies, and tool settings
├── README.md                               # Project overview, specifications, examples, and roadmap
├── scaffold.sh                             # Shell script for repository and module scaffolding
├── src/
│   └── softpotato/                         # Root namespace for the Soft Potato package
│       ├── __init__.py                     # Package root exposing top-level API and submodules
│       ├── analytical/                     # Closed-form analytical electrochemical benchmark equations
│       │   ├── __init__.py                 # Facade exposing primary analytical functions (Randles-Sevcik, Cottrell, Saito)
│       │   ├── geometry/                   # Geometry-dependent steady-state analytical solutions
│       │   │   ├── __init__.py             # Exposes geometry-dependent analytical functions
│       │   │   ├── hydrodynamics.py        # Analytical solutions for hydrodynamic electrodes (e.g., Levich)
│       │   │   └── microelectrodes.py      # Analytical steady-state solutions for microelectrodes (e.g., Saito)
│       │   ├── kinetics/                   # Kinetic reversibility and mechanism diagnostic equations
│       │   │   ├── __init__.py             # Exposes analytical kinetic diagnostics
│       │   │   ├── mechanisms.py           # Diagnostic equations for coupled chemical mechanisms (EC, ECE)
│       │   │   └── reversibility.py        # Diagnostic criteria for electrochemical reversibility (Matsuda-Ayabe, Nicholson)
│       │   └── techniques/                 # Transient technique-specific analytical solutions
│       │       ├── __init__.py             # Exposes technique-specific analytical functions
│       │       ├── step.py                 # Potential step solutions (e.g., Cottrell equation)
│       │       └── voltammetry.py          # Voltammetry peak current solutions (e.g., Randles-Sevcik)
│       ├── core/                           # Core data structures and reaction mechanism representations
│       │   ├── __init__.py                 # Exposes core classes (Species, Reactions, Mechanism)
│       │   ├── reactions.py                # ElectrochemicalReaction, ChemicalReaction, and Mechanism classes
│       │   └── species.py                  # Species data class with strict CGS unit enforcement
│       ├── geometry/                       # Spatial discretizations and electrode geometry models
│       │   ├── __init__.py                 # Exposes grid and electrode classes
│       │   ├── electrodes.py               # Electrode geometry implementations (Planar, Spherical, ThinLayer)
│       │   └── grids.py                    # Spatial grid generators (UniformGrid, ExpandingGrid)
│       ├── kinetics/                       # Electrochemical and chemical kinetic models
│       │   ├── __init__.py                 # Exposes kinetic rate and boundary condition models
│       │   └── models.py                   # Butler-Volmer, Nernst, and chemical kinetic rate models
│       ├── simulate/                       # Numerical PDE solvers and simulation orchestrators
│       │   ├── __init__.py                 # Exposes simulation solvers and engines
│       │   ├── efd.py                      # Explicit Finite Difference (EFD) solver implementation
│       │   ├── ifd.py                      # Implicit Finite Difference (IFD) / Crank-Nicolson solver
│       │   └── solver.py                   # Main Solver class orchestrating simulation execution
│       └── techniques/                     # Electrochemical excitation signals and waveforms
│           ├── __init__.py                 # Exposes technique waveform generators
│           ├── step.py                     # Potential step waveforms (e.g., Chronoamperometry)
│           └── voltammetry.py              # Voltammetry waveforms (e.g., CyclicVoltammetry)
└── tests/                                  # Test suite for unit, integration, and validation tests
    ├── conftest.py                         # Shared pytest fixtures and test configurations
    ├── test_core.py                        # Unit tests for Species, Reactions, and Mechanism classes
    ├── test_geometry_laplacians.py         # Tests for spatial discretization grids and Laplacian operators
    ├── test_kinetics.py                    # Tests for Butler-Volmer and chemical reaction kinetics
    └── test_solvers.py                     # Validation tests for numerical solvers against analytical baselines
```