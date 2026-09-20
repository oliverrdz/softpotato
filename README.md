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
│       └── ci.yml                # Automated pytest runs for GitHub Actions
├── pyproject.toml                # Build system (Hatchling or Flit), dependencies, and package metadata
├── README.md
├── LICENSE
├── src/
│   └── softpotato/               # Root namespace
│       ├── __init__.py           # Exposes main API: from .core import Species, Mechanism
│       ├── analytical/
│       │   ├── __init__.py
│       │   ├── geometry/
│       │   │   ├── __init__.py
│       │   │   ├── hydrodynamics.py
│       │   │   └── microelectrodes.py
│       │   ├── kinetics/
│       │   │   ├── __init__.py
│       │   │   ├── mechanisms.py
│       │   │   └── reversibility.py
│       │   └── techniques/
│       │       ├── __init__.py
│       │       ├── step.py
│       │       └── voltammetry.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── species.py        # Species class (enforces CGS unit storage)
│       │   └── reactions.py      # ElectrochemicalReaction, ChemicalReaction, Mechanism
│       ├── kinetics/
│       │   ├── __init__.py
│       │   └── models.py         # ButlerVolmer, Nernst, FirstOrder classes (flux & boundary conditions)
│       ├── geometry/
│       │   ├── __init__.py
│       │   ├── grids.py          # UniformGrid, ExpandingGrid (returns mesh and Laplacian weights)
│       │   └── electrodes.py     # Planar, Spherical, ThinLayer
│       ├── techniques/
│       │   ├── __init__.py
│       │   ├── voltammetry.py    # CyclicVoltammetry (generates discretized t and E arrays)
│       │   └── step.py           # Chronoamperometry
│       └── simulate/
│           ├── __init__.py
│           ├── solver.py         # Main Solver class orchestrating the simulation loop
│           ├── efd.py            # Explicit Finite Difference implementation (vectorized numpy)
│           └── ifd.py            # Implicit Finite Difference / Crank-Nicolson (scipy.sparse)
└── tests/                        # Comprehensive test suite
    ├── conftest.py               # Shared pytest fixtures (e.g., standard [O], [R] species)
    ├── test_core.py
    ├── test_kinetics.py
    ├── test_geometry_laplacians.py
    └── test_solvers.py           # Validation against analytical solutions (e.g., Randles-Sevcik)
```